'''
    Authored by: Victor Agostinelli, from Oregon State University 
        (also affiliated with Pacific Northwest National Laboratory)

    Intended to interface with the MHA implementation in Fairseq, so this isn't
    constructed as a class or PyTorch module and is instead meant to be
    a file filled with functions. 

    This also means that the MHA implementation in Fairseq needs to be augmented a 
    bit. For example, it needs to construct the LeaP modules (provided as q_denom 
    and k_denom).

    SimulMT and SimulST cross-attention is a little more involved, implementation for
    that is grafted on the relevant example in Fairseq.
'''

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from fairseq import utils
from fairseq.modules.fairseq_dropout import FairseqDropout

'''
    Generally expect Q, K, and V tensors to be in format B x N x d where...
        B: batch size
        N: source or target sequence length
        d: dimensionality of a given attention head, or alternatively dimensionality of model

    Provided functions are intended for specific use.
        - leapformer_attn_train: meant to be a flexible version of the training flow for leapformers, not
            necessarily a performant version. Linear training should be disabled by default for causal
            attention, as replication of masking behavior almost always causes a massive memory bottleneck.
        - leapformer_attn_bidir_infer: meant for bidirectional attention with NO padding or masking.
        - leapformer_attn_causal_infer: meant for causal attention with a Fairseq incremental_state supporting it.
'''

def leapformer_attn_train(
    q,
    k: Optional[Tensor],
    v: Optional[Tensor],
    num_heads: int, 
    q_denom,
    k_denom,
    out_proj,
    key_padding_mask: Optional[Tensor] = None,
    attn_mask: Optional[Tensor] = None,
    need_weights = False,
    need_head_weights = False,
):

    src_len = k.size(1)
    tgt_len = q.size(1)
    bsz = int(k.size(0) / num_heads)

    # implementation differs from typical key_padding_mask application, but this is useful later and should be fine
    if key_padding_mask is not None:
        key_pad_mask_unsqueeze = key_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool)
        k = k.view(bsz, num_heads, src_len, list(k.shape)[2])          
        k = k.masked_fill(key_pad_mask_unsqueeze, 0)
        k = k.view(bsz*num_heads, src_len, list(k.shape)[3])          

    # ReLU similarity function
    q = F.relu(q)
    k = F.relu(k)

    # application of LeaP module and re-weighting function construction
    tgt_denoms = q_denom(q)
    sin_tr_q = torch.sin((math.pi / 2) * tgt_denoms)
    cos_tr_q = torch.cos((math.pi / 2) * tgt_denoms)

    src_denoms = k_denom(k)
    sin_tr_k = torch.sin((math.pi / 2) * src_denoms)
    cos_tr_k = torch.cos((math.pi / 2) * src_denoms)
    
    # query transforms, elementwise
    q_sin = torch.mul(q, sin_tr_q)
    q_cos = torch.mul(q, cos_tr_q)
    
    # key transforms, elementwise
    k_sin = torch.mul(k, sin_tr_k)
    k_cos = torch.mul(k, cos_tr_k)

    # linearized training turned off by default, can take much longer when not using
    # a specialized implementation for causal or chunked causal attention
    # TODO: need to add back some parallel scan behavior via einsum/cumsum for linearized
    #       training should a user want to enable it despite memory constraint issues
    if linearized_train:
        kTv_sin = torch.bmm(k_sin.transpose(1, 2), v)
        kTv_cos = torch.bmm(k_cos.transpose(1, 2), v)

        norm_sin = torch.sum(k_sin.unsqueeze(-1), dim=1)
        norm_cos = torch.sum(k_cos.unsqueeze(-1), dim=1)

        # final attn calculations
        attn_weights_sin = torch.bmm(q_sin, kTv_sin)
        attn_weights_cos = torch.bmm(q_cos, kTv_cos)
        attn_weights = attn_weights_sin + attn_weights_cos

        prob_norm_sin = torch.bmm(q_sin, norm_sin)
        prob_norm_cos = torch.bmm(q_cos, norm_cos)
        prob_norm = prob_norm_sin + prob_norm_cos

        prob_norm = torch.clamp_min(prob_norm, 0.1)
        attn_probs = attn_weights / prob_norm

        attn = attn_probs

    # quadratic doesn't experience a memory bottleneck for naive linear attention
    else:
        attn_weights_sin = torch.bmm(q_sin, k_sin.transpose(1, 2))
        attn_weights_cos = torch.bmm(q_cos, k_cos.transpose(1, 2))
        attn_weights = attn_weights_sin + attn_weights_cos

        if attn_mask is not None:
            attn_mask_bool = attn_mask.to(torch.bool)
            attn_weights = attn_weights.masked_fill(attn_mask_bool, 0)
        
        attn_weights = dropout_module(attn_weights)
        attn_probs = torch.bmm(attn_weights, v)

        # section to try and replicate casual relationship in normalization
        if attn_mask is not None:
            norm_sin = torch.cumsum(k_sin, dim=1).transpose(1, 2)
            norm_cos = torch.cumsum(k_cos, dim=1).transpose(1, 2)
        else:
            norm_sin = torch.sum(k_sin, dim=1).unsqueeze(-1)
            norm_cos = torch.sum(k_cos, dim=1).unsqueeze(-1)
        
        prob_norm_sin = torch.bmm(q_sin, norm_sin)
        prob_norm_cos = torch.bmm(q_cos, norm_cos)
        prob_norm = prob_norm_sin + prob_norm_cos

        if attn_mask is not None:
            prob_norm = torch.diagonal(prob_norm, dim1=1, dim2=2).unsqueeze(-1)

        prob_norm = torch.clamp_min(prob_norm, 0.1)

        attn = attn_probs / prob_norm
        
    attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn = out_proj(attn)

    if need_weights:
        attn_weights = attn_weights.view(
            bsz, num_heads, tgt_len, src_len
        ).transpose(1, 0)
        if not need_head_weights:
            # average attention weights over heads
            attn_weights = attn_weights.mean(dim=0)
    
    return attn, attn_weights
    
# really only intended to be called during bidirectional attention, separated for readability
# the intended application for this is usually SimulMT or SimulST, in which case we usually don't batch
# together examples for more accurate latency measurements, so no padding and masking behavior is supported
# by default
def leapformer_attn_bidir_infer(
    q,
    k: Optional[Tensor],
    v: Optional[Tensor],
    num_heads: int,
    q_denom,
    k_denom,
    out_proj,
    need_weights = False,
    need_head_weights = False,
):
    
    bsz = int(k.size(0) / num_heads)
    src_len = k.size(1)
    tgt_len = q.size(1)
    
    # ReLU similarity function
    q = F.relu(q)
    k = F.relu(k)

    # application of LeaP module and construction of transforms
    tgt_denoms = q_denom(q)
    sin_tr_q = torch.sin((math.pi / 2) * tgt_denoms)
    cos_tr_q = torch.cos((math.pi / 2) * tgt_denoms)
    
    src_denoms = k_denom(k)
    sin_tr_k = torch.sin((math.pi / 2) * src_denoms)
    cos_tr_k = torch.cos((math.pi / 2) * src_denoms)

    # application of re-weighting transforms
    q_sin = torch.mul(q, sin_tr_q)
    q_cos = torch.mul(q, cos_tr_q)
    
    k_sin = torch.mul(k, sin_tr_k)
    k_cos = torch.mul(k, cos_tr_k)
    
    # construct d x d intermediate matrices and normalization tensors
    kTv_sin = torch.bmm(k_sin.transpose(1, 2), v)
    kTv_cos = torch.bmm(k_cos.transpose(1, 2), v)

    norm_sin = torch.sum(k_sin.unsqueeze(-1), dim=1)
    norm_cos = torch.sum(k_cos.unsqueeze(-1), dim=1)

    # final attn calculations
    attn_weights_sin = torch.bmm(q_sin, kTv_sin)
    attn_weights_cos = torch.bmm(q_cos, kTv_cos)
    attn_weights = attn_weights_sin + attn_weights_cos

    prob_norm_sin = torch.bmm(q_sin, norm_sin)
    prob_norm_cos = torch.bmm(q_cos, norm_cos)
    prob_norm = prob_norm_sin + prob_norm_cos

    prob_norm = torch.clamp_min(prob_norm, 0.1)
    attn_probs = attn_weights / prob_norm

    attn = attn_probs
    attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn = out_proj(attn)
    
    if need_weights:
        attn_weights = torch.bmm(q_sin, k_sin.transpose(1, 2)) + torch.bmm(q_cos, k_cos.transpose(1, 2))
        attn_weights = attn_weights / prob_norm
        attn_weights = attn_weights.view(
            bsz, num_heads, tgt_len, src_len
        ).transpose(1, 0)
        if not need_head_weights:
            # average attention weights over heads
            attn_weights = attn_weights.mean(dim=0)

    return attn, attn_weights


# usually should be applied to causal attention (i.e. decoder attention), but is meant to also support
# cross-attention behavior if necessary, not built to be extremely performant, triton implementation is TODO
# simul_attn_chkpts is an added dictionary used primarily for more performant inference with causal attention,
# applied in this work towards SimulMT/SimulST kinds of tasks (thus the naming convention)
def leapformer_attn_causal_infer( 
    q,
    k: Optional[Tensor],
    v: Optional[Tensor],
    num_heads: int,
    q_denom,
    k_denom,
    out_proj,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    simul_attn_chkpts: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    need_weights = False,
    need_head_weights = False,
    layer_idx: int = None,
    attn_type = True,
):
    
    bsz = int(q.size(0) / num_heads)
    tgt_len = q.size(1)
    src_len = None
    if k is not None:
        src_len = k.size(1)
    
    # ReLU similarity function
    q = F.relu(q)
    if k is not None:
        k = F.relu(k)
    
    # acquire old data from incremental state
    attn_block = "attn" if attn_type else "cross_attn"
    old_attn_weights_v_sin = simul_attn_chkpts["layers"][layer_idx][attn_block]["kTv_sin"]
    old_attn_weights_v_cos = simul_attn_chkpts["layers"][layer_idx][attn_block]["kTv_cos"]
    norm_sin_old = simul_attn_chkpts["layers"][layer_idx][attn_block]["norm_sin"]
    norm_cos_old = simul_attn_chkpts["layers"][layer_idx][attn_block]["norm_cos"]

    # apply LeaP module and build transforms 
    tgt_denoms = q_denom(q)
    sin_tr_q = torch.sin((math.pi / 2) * tgt_denoms)
    cos_tr_q = torch.cos((math.pi / 2) * tgt_denoms)
    
    src_denoms = k_denom(k)
    sin_tr_k = torch.sin((math.pi / 2) * src_denoms)
    cos_tr_k = torch.cos((math.pi / 2) * src_denoms)
    
    q_sin = torch.mul(q, torch.sin((math.pi / 2) * sin_tr_q))
    q_cos = torch.mul(q, torch.cos((math.pi / 2) * cos_tr_q))
    k_sin = torch.mul(k, torch.sin((math.pi / 2) * sin_tr_k))
    k_cos = torch.mul(k, torch.cos((math.pi / 2) * cos_tr_k))

    # build normalization vectors
    if norm_sin_old is not None and norm_cos_old is not None:
        if attn_block == "attn":
            norm_sin = norm_sin_old + k_sin.transpose(1, 2)
            norm_cos = norm_cos_old + k_cos.transpose(1, 2)
        else:
            norm_sin = norm_sin_old
            norm_cos = norm_cos_old
    else:
        if attn_block == "attn":
            norm_sin = k_sin.transpose(1, 2)
            norm_cos = k_cos.transpose(1, 2)
        else:
            norm_sin = torch.sum(k_sin, dim=1).unsqueeze(-1)
            norm_cos = torch.sum(k_cos, dim=1).unsqueeze(-1)

    # build out d x d intermediate matrix
    if old_attn_weights_v_sin is not None and old_attn_weights_v_cos is not None:
        if attention:
            attn_weights_v_sin = old_attn_weights_v_sin + torch.bmm(k_sin.transpose(1, 2), v)
            attn_weights_v_cos = old_attn_weights_v_cos + torch.bmm(k_cos.transpose(1, 2), v)
        else:
            attn_weights_v_sin = old_attn_weights_v_sin
            attn_weights_v_cos = old_attn_weights_v_cos
    else:
        attn_weights_v_sin = torch.bmm(k_sin.transpose(1, 2), v)
        attn_weights_v_cos = torch.bmm(k_cos.transpose(1, 2), v)

    attn_weights_sin = torch.bmm(q_sin, attn_weights_v_sin)
    attn_weights_cos = torch.bmm(q_cos, attn_weights_v_cos)
    attn_weights = attn_weights_sin + attn_weights_cos

    prob_norm_sin = torch.bmm(q_sin, norm_sin)
    prob_norm_cos = torch.bmm(q_cos, norm_cos)
    prob_norm = prob_norm_sin + prob_norm_cos

    prob_norm = torch.clamp_min(prob_norm, 0.1)

    attn = attn_weights / prob_norm
    attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn = out_proj(attn)

    simul_attn_chkpts["layers"][layer_idx][attn_block]["kTv_sin"] = attn_weights_v_sin
    simul_attn_chkpts["layers"][layer_idx][attn_block]["kTv_cos"] = attn_weights_v_cos
    simul_attn_chkpts["layers"][layer_idx][attn_block]["norm_sin"] = norm_sin
    simul_attn_chkpts["layers"][layer_idx][attn_block]["norm_cos"] = norm_cos
    
    if need_weights:
        attn_weights = torch.bmm(q_sin, k_sin.transpose(1, 2)) + torch.bmm(q_cos, k_cos.transpose(1, 2))
        attn_weights = attn_weights / prob_norm
        attn_weights = attn_weights.view(
            bsz, num_heads, tgt_len, src_len
        ).transpose(1, 0)
        if not need_head_weights:
            # average attention weights over heads
            attn_weights = attn_weights.mean(dim=0)

    return attn, attn_weights