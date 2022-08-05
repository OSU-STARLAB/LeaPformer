# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, List
from itertools import groupby

import torch
import torch.nn.functional as F
from fairseq.models import FairseqEncoder
from fairseq.models.speech_to_text import (
    ConvTransformerEncoder, CTCCompressStrategy,
)
from fairseq.models.speech_to_text.utils import attention_suppression
from fairseq.models.speech_to_text.utils import (
    lengths_to_encoder_padding_mask,
    segments_to_sequence,
    sequence_to_segments,
)
from fairseq.modules import MultiheadAttention, TransformerEncoderLayer
from torch import nn, Tensor
import json

# ------------------------------------------------------------------------------
#   AugmentedMemoryConvTransformerEncoder
# ------------------------------------------------------------------------------


class AugmentedMemoryConvTransformerEncoder(ConvTransformerEncoder):
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

        args.encoder_stride = self.stride()

        self.left_context = args.left_context // args.encoder_stride

        self.right_context = args.right_context // args.encoder_stride

        self.left_context_after_stride = args.left_context // args.encoder_stride
        self.right_context_after_stride = args.right_context // args.encoder_stride

        self.transformer_layers = nn.ModuleList([])

        # Layer sharing code
        encoder_weight_share_list = getattr(args, "share_encoder_ffn_attn_layer", None)
        if encoder_weight_share_list is None:
            encoder_weight_share_list = []
        else:
            shared_weights_layer = AugmentedMemoryTransformerEncoderLayer(args)
        print(f"Encoder: Sharing layers: {encoder_weight_share_list}")
        for layer_idx in range(args.encoder_layers):
            if layer_idx+1 in encoder_weight_share_list:
                self.transformer_layers.append(shared_weights_layer)
            else:
                self.transformer_layers.append(AugmentedMemoryTransformerEncoderLayer(args))

        self.share_mem_bank_layers = args.share_mem_bank_layers

    def stride(self):
        # Hard coded here. Should infer from convs in future
        stride = 4
        return stride

    def initialize_states(self, states):
        if states is None:
            # Creates states;
            states = [{"memory_banks": [], "encoder_states": None} for i in range(len(self.transformer_layers))]
            if self.share_mem_bank_layers is not None:
                # Initializes Memory banks
                rows = len(self.share_mem_bank_layers)
                for i in range(rows):
                    cols = len(self.share_mem_bank_layers[i]);
                    for j in range(cols):
                        states[self.share_mem_bank_layers[i][j]]["memory_banks"] = states[self.share_mem_bank_layers[i][0]]["memory_banks"]
        return states

    def forward(self, src_tokens, src_lengths, states=None):
        """Encode input sequence.
        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        bsz, max_seq_len, _ = src_tokens.size()
        x = (
            src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
            .transpose(1, 2)
            .contiguous()
        )
        x = self.conv(x)
        bsz, _, output_seq_len, _ = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous().view(output_seq_len, bsz, -1)
        x = self.out(x)
        x = self.embed_scale * x

        subsampling_factor = max_seq_len * 1.0 / output_seq_len

        input_lengths = torch.min(
            (src_lengths.float() / subsampling_factor).ceil().long(),
            x.size(0) * src_lengths.new_ones([src_lengths.size(0)]).long(),
        )

        encoder_padding_mask, _ = lengths_to_encoder_padding_mask(
            input_lengths, batch_first=True
        )

        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)

        x += positions

        x = F.dropout(x, p=self.dropout, training=self.training)

        # State to store memory banks etc.
        states = self.initialize_states(states)
        #print("states after initialize_states: ", states)

        for i, layer in enumerate(self.transformer_layers):
            # x size:
            # (self.left_size + self.segment_size + self.right_size)
            # / self.stride, num_heads, dim
            # TODO: Consider mask here 
            x = layer(x, states[i], i)
            if self.right_context_after_stride != 0:
                states[i]["encoder_states"] = x[self.left_context_after_stride : -self.right_context_after_stride]
            else:
                states[i]["encoder_states"] = x[self.left_context_after_stride : ]

        if self.right_context_after_stride != 0:
            lengths = (
                (
                    ~encoder_padding_mask[:, self.left_context_after_stride : -self.right_context_after_stride]
                )
                .sum(dim=1, keepdim=True)
                .long()
            )
        else:
            lengths = (
                (
                    ~encoder_padding_mask[:, self.left_context_after_stride :]
                )
                .sum(dim=1, keepdim=True)
                .long()
            )
        return states[-1]["encoder_states"], lengths, states


# ------------------------------------------------------------------------------
#   AugmentedMemoryTransformerEncoderLayer
# ------------------------------------------------------------------------------
class AugmentedMemoryTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)

        self.left_context = args.left_context // args.encoder_stride
        self.right_context = args.right_context // args.encoder_stride
        self.increase_context = args.increase_context

        self.share_mem_bank_layers = args.share_mem_bank_layers

    def forward(self, x, state, layer_num):

        length, batch_size, x_dim = x.size()

        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # TODO reseach new sum_query method
        if self.increase_context:
            seg_start = 0
        else:
            seg_start = self.left_context
        seg_end = length - self.right_context

        if seg_start < seg_end:
            summarization_query = torch.mean(x[seg_start:seg_end], keepdim=True, dim=0)
        else:
            summarization_query = x.new_zeros(1, batch_size, x_dim)

        x = torch.cat([x, summarization_query], dim=0)

        x = self.self_attn(input_and_summary=x, state=state, layer_num=layer_num)

        x = self.dropout_module(x)

        x = residual + x

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x

    def build_self_attention(self, embed_dim, args):
        return AugmentedMemoryMultiheadAttention(
            embed_dim=embed_dim,
            num_heads=args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            tanh_on_mem=True,
            max_memory_size=args.max_memory_size,
            share_mem_bank_layers=args.share_mem_bank_layers,
        )


# ------------------------------------------------------------------------------
#   AugmentedMemoryMultiheadAttention
# ------------------------------------------------------------------------------
class AugmentedMemoryMultiheadAttention(MultiheadAttention):
    """
    Augmented Memory Attention from
    Streaming Transformer-based Acoustic Models
    Using Self-attention with Augmented Memory
    https://arxiv.org/abs/2005.08042
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
        tanh_on_mem=False,
        memory_dim=None,
        std_scale=0.5,  # 0.5 based on https://arxiv.org/abs/2005.09137
        max_memory_size=-1,
        disable_mem_on_mem_attn=True,
        share_mem_bank_layers=None,
    ):
        super().__init__(
            embed_dim,
            num_heads,
            kdim,
            vdim,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            self_attention,
            encoder_decoder_attention,
            q_noise,
            qn_block_size,
        )

        self.memory_dim = memory_dim if memory_dim is not None else embed_dim
        self.std_scale = std_scale
        self.disable_mem_on_mem_attn = disable_mem_on_mem_attn

        # This Operator was used for factorization in PySpeech
        self.v2e = lambda x: x

        if tanh_on_mem:
            self.squash_mem = torch.tanh
            self.nonlinear_squash_mem = True
        else:
            self.squash_mem = lambda x: x
            self.nonlinear_squash_mem = False

        self.max_memory_size = max_memory_size

        self.share_mem_bank_layers = share_mem_bank_layers

    def update_mem_banks(self, state, output_and_memory, layer_num):
        if self.share_mem_bank_layers is not None:
          if not any(layer_num in layer for layer in self.share_mem_bank_layers):
            next_m = output_and_memory[-1:]
            next_m = self.squash_mem(next_m)
            state["memory_banks"].append(next_m)
          else:
            for pairs in self.share_mem_bank_layers:
                if layer_num == pairs[0]:
                    next_m = output_and_memory[-1:]
                    next_m = self.squash_mem(next_m)
                    state["memory_banks"].append(next_m)        
        else:
            next_m = output_and_memory[-1:]
            next_m = self.squash_mem(next_m)
            state["memory_banks"].append(next_m) 
        return state

    def forward(self, input_and_summary, state, layer_num):
        """
        input: Encoder states of current segment with left or right context,
            plus one summarization query

        """

        length, batch_size, _ = input_and_summary.shape
        length = length - 1  # not include sum_query, last index

        memory = state["memory_banks"]
        # TODO: positional embedding on memory

        if self.max_memory_size > -1 and len(memory) > self.max_memory_size:
            memory = memory[-self.max_memory_size :]
            state["memory_banks"] = memory

        memory_and_input = torch.cat(memory + [input_and_summary[:-1]], dim=0)
        input_and_sum_query = input_and_summary

        q = self.q_proj(self.v2e(input_and_sum_query))
        k = self.k_proj(self.v2e(memory_and_input))
        v = self.v_proj(self.v2e(memory_and_input))

        q = (
            q.contiguous()
            .view(-1, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
            * self.scaling
        )
        k = (
            k.contiguous()
            .view(-1, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        v = (
            v.contiguous()
            .view(-1, batch_size * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        attention_weights = torch.bmm(q, k.transpose(1, 2))
        
        if self.disable_mem_on_mem_attn:
            attention_weights = self.suppress_mem_on_mem_attention(
                batch_size, self.num_heads, len(memory), attention_weights
            )
       
        if self.std_scale is not None:
            attention_weights = attention_suppression(attention_weights, self.std_scale)
       
        assert list(attention_weights.shape) == [
            batch_size * self.num_heads,
            length + 1,
            length + len(memory),
        ]

        attention_weights = torch.nn.functional.softmax(
            attention_weights.float(), dim=-1
        ).type_as(attention_weights)
      
        attention_probs = self.dropout_module(attention_weights)

        # [T, T, B, n_head] + [T, B, n_head, d_head] -> [T, B, n_head, d_head]
        attention = torch.bmm(attention_probs, v)

        assert list(attention.shape) == [
            batch_size * self.num_heads,
            length + 1,
            self.head_dim,
        ]

        attention = (
            attention.transpose(0, 1)
            .contiguous()
            .view(length + 1, batch_size, self.embed_dim)
        )
        
        output_and_memory = self.out_proj(attention)
        
        output = output_and_memory[:-1]
        
        if self.max_memory_size != 0:
            state = self.update_mem_banks(state, output_and_memory, layer_num)

        return output

    def suppress_mem_on_mem_attention(
        self, B: int, num_heads: int, mem_size: int, attention_weight: Tensor
    ):
        """
        Arguments:
            - B: batch size
            - num_heads: number of attention heads
            - mem_size: size of memory bank
            - attention_weight: a [B*num_heads, T + 1, T + mem_size] vector

        Return:
            modified attention_weight with [B*num_heads, -1, :mem_size] = -inf
        """
        attention_weight[:, -1, :mem_size] = float("-inf")
        return attention_weight


# ------------------------------------------------------------------------------
#   SequenceEncoder
# ------------------------------------------------------------------------------
class SequenceEncoder(FairseqEncoder):
    """
    SequenceEncoder encodes sequences.

    More specifically, `src_tokens` and `src_lengths` in `forward()` should
    describe a batch of "complete" sequences rather than segments.

    Segment-by-segment inference can be triggered by `segment_size`:
    1) `segment_size` is None:
        SequenceEncoder treats the input sequence as one single segment.
    2) `segment_size` is not None (some int instead):
        SequenceEncoder does the following:
            1. breaks the input sequence into several segments
            2. inference on each segment and collect the outputs
            3. concatanete segment outputs into the output sequence.
    Note that `segment_size` here shouldn't include additional left/right
    contexts needed, for example if we wish to infer with LC-BLSTM where the
    middle chunk size is 100 and right context is 20, `segment_size` should be
    100.
    """

    def __init__(self, args, dictionary, module):
        super().__init__(dictionary)

        self.module = module
        self.input_time_axis = 1
        self.output_time_axis = 0
        self.segment_size = args.segment_size
        self.left_context = args.left_context
        self.right_context = args.right_context

        self.ctc_compress_out = getattr(args, "ctc_compress_out", False)
        if self.ctc_compress_out:
            self.ctc_fc = nn.Linear(args.encoder_embed_dim, len(dictionary))
            assert args.criterion == "ctc_multi_loss"
            self.ctc_compress_method = getattr(CTCCompressStrategy, args.ctc_compress_strategy)

    def forward(
        self,
        src_tokens: Tensor,
        src_lengths: Tensor,
        return_all_hiddens: bool,
        states=None,
    ):

        seg_src_tokens_lengths = sequence_to_segments(
            sequence=src_tokens,
            time_axis=self.input_time_axis,
            lengths=src_lengths,
            segment_size=self.segment_size,
            extra_left_context=self.left_context,
            extra_right_context=self.right_context,
        )

        seg_encoder_states_lengths: List[Tuple[Tensor, Tensor]] = []

        for seg_src_tokens, seg_src_lengths in seg_src_tokens_lengths:
            (seg_encoder_states, seg_enc_lengths, states) = self.module(
                seg_src_tokens,
                seg_src_lengths,
                states=states,
            )

            seg_encoder_states_lengths.append((seg_encoder_states, seg_enc_lengths))

        encoder_out, enc_lengths = segments_to_sequence(
            segments=seg_encoder_states_lengths, time_axis=self.output_time_axis
        )

        if self.ctc_compress_out:
                encoder_out, enc_lengths = self.average_same_ctc_features(encoder_out, enc_lengths)

        encoder_padding_mask, _ = lengths_to_encoder_padding_mask(
            enc_lengths, batch_first=True
        )

        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        return {
            "encoder_out": [encoder_out],
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask is not None
            else [],
            "encoder_embedding": [],
            "encoder_states": [[encoder_out]],
            "src_tokens": [],
            "src_lengths": [],
        }

    def incremental_encode(
        self,
        seg_src_tokens: Tensor,
        seg_src_lengths: Tensor,
        states=None,
    ):
        """
        Different from forward function, this function takes segmented speech
        as input, and append encoder states to previous states
        """
        (seg_encoder_states, seg_enc_lengths, states) = self.module(
            seg_src_tokens,
            seg_src_lengths,
            states=states,
        )
        return seg_encoder_states, seg_enc_lengths, states
    
    def average_same_ctc_features(self, x, src_lengths):
        x_ctc = self.ctc_fc(x)
        with torch.no_grad():
            batch_predicted = []
            prob_ctc = F.softmax(x_ctc, dim=-1).transpose(0, 1)  # from T x B x D to B x T x D
            for b in range(prob_ctc.shape[0]):
                predicted = prob_ctc[b][: src_lengths[b]].argmax(-1).tolist()
                batch_predicted.append([(p[0], len(list(p[1]))) for p in groupby(predicted)])

            new_lengths = [len(p) for p in batch_predicted]
            weights_matrix = self.ctc_compress_method(prob_ctc, batch_predicted, new_lengths, x.dtype, x.device)
        # x is T x B x C -> B x C x T; weights_matrix is B x T x T'
        compressed_output = x.permute(1, 2, 0).bmm(weights_matrix)  # B x C x T'
        return compressed_output.permute(2, 0, 1), src_lengths.new(new_lengths)

# ------------------------------------------------------------------------------
#   Augmented memory model decorator
# ------------------------------------------------------------------------------
def augmented_memory(klass):
    class StreamSeq2SeqModel(klass):
        @staticmethod
        def add_args(parser):
            super(StreamSeq2SeqModel, StreamSeq2SeqModel).add_args(parser)
            parser.add_argument(
                "--segment-size", type=int, required=True, help="Length of the segment."
            )
            parser.add_argument(
                "--left-context",
                type=int,
                default=0,
                help="Left context for the segment.",
            )
            parser.add_argument(
                "--right-context",
                type=int,
                default=0,
                help="Right context for the segment.",
            )
            parser.add_argument(
                "--max-memory-size",
                type=int,
                default=-1,
                help="Right context for the segment.",
            )
            parser.add_argument(
                "--increase-context",
                action="store_true",
                default=False,
                help="if True, memory bank calculation uses left and center context",
            )
            parser.add_argument(
                "--share-mem-bank-layers",
                type=json.loads,
                default=None,
                help=":The list of memory bank sharing layers",
            )

    StreamSeq2SeqModel.__name__ = klass.__name__
    return StreamSeq2SeqModel