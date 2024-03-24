import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from rotary_embedding_torch import RotaryEmbedding

# rotary positional embedding w/ xpos
# https://arxiv.org/abs/2104.09864
# https://arxiv.org/abs/2212.10554v1

#class RotaryEmbedding(nn.Module):
#    def __init__(self, dim, scale_base = 512, use_xpos = True):
#        super().__init__()
#        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
#
#        self.use_xpos = use_xpos
#        self.scale_base = scale_base
#        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
#
#    def forward(self, seq_len, device):
#        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
#        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
#        freqs = torch.cat((freqs, freqs), dim = -1)
#
#        if not self.use_xpos:
#            return freqs, torch.ones(1, device = device)
#
#        power = (t - (seq_len // 2)) / self.scale_base
#        scale = self.scale ** rearrange(power, 'n -> n 1')
#        scale = torch.cat((scale, scale), dim = -1)
#
#        return freqs, scale
#
#def rotate_half(x):
#    x1, x2 = x.chunk(2, dim=-1)
#    return torch.cat((-x2, x1), dim=-1)
#
#
#def apply_rotary_pos_emb(pos, t, scale = 1.):
#    return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)


class RoPEELULinearAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.num_head = config["num_head"]
        self.head_dim = config["head_dim"]
        self.max_seq_len = config["max_seq_len"]
        self.device = config['device'] if 'device' in config else 'cuda'

        self.drop_attn = nn.Dropout(p = config["attention_dropout"])
        
        self.rotary_emb = RotaryEmbedding(dim=int(self.head_dim / 2))

#    def get_rotary_embedding(self, n, device):
#        if exists(self.pos_emb) and self.pos_emb.shape[-2] >= n:
#            return self.pos_emb[:n], self.pos_emb_scale[:n]
#
#        pos_emb, scale = self.rotary_emb(n, device=device)
#        self.register_buffer("pos_emb", pos_emb, persistent=False)
#        self.register_buffer("pos_emb_scale", scale, persistent=False)
#        return pos_emb, scale

    def forward(self, Q, K, V, mask):
        scaling = self.head_dim ** -(1/2)
        Q = Q * scaling

        Q = F.elu(Q) + 1
        K = F.elu(K) + 1
        
        # apply mask
        K = K.masked_fill(~(mask.to(bool)[:, None, :, None]), 0)
        
        # build out normalization before rotating
        norm = torch.matmul(Q, torch.abs(K.transpose(-2, -1)).sum(-1).unsqueeze(-1))

        Q = self.rotary_emb.rotate_queries_or_keys(Q)
        K = self.rotary_emb.rotate_queries_or_keys(K)

        # build out d x d intermediate matrices, then attn weights
        attn_inter = torch.matmul(K.transpose(-2, -1), V)
        attn_weights = torch.matmul(Q, attn_inter)
        
        # final product for attn scores
        attn = attn_weights / norm

        return attn
