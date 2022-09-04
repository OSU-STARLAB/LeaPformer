#!/usr/bin/env python3

import logging
import math
from typing import Dict, List, Optional, Tuple
from itertools import groupby

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerEncoderLayer
from torch import Tensor

logger = logging.getLogger(__name__)


@register_model("convtransformer")
class ConvTransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer-based Speech translation model from ESPNet-ST
    https://arxiv.org/abs/2004.10234
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--input-feat-per-channel",
            type=int,
            metavar="N",
            help="encoder input dimension per input channel",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--decoder-output-dim",
            type=int,
            metavar="N",
            help="decoder output dimension (extra linear layer if different from decoder embed dim)",
        )
        parser.add_argument(
            "--decoder-input-dim",
            type=int,
            metavar="N",
            help="decoder input dimension (extra linear layer if different from decoder embed dim)",
        )
        parser.add_argument( 
            "--token-embed-dim",
            type=int,
            metavar="N",
            help="token embedding dimension (used with decoder-input-dim and decoder-output-dim for embedding matrix factorization)",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )
        parser.add_argument(
            "--conv-out-channels",
            type=int,
            metavar="INT",
            help="the number of output channels of conv layer",
        )
        parser.add_argument(
            "--share-encoder-ffn-attn-layer",
            nargs="+",
            type=int,
            metavar="INT INT",
            help=":The list of sharing layers for both feed-forward network and attention. The range of layer number starts from 1, and the layer number which is over the range would not work.",
        )
        parser.add_argument(
            "--share-decoder-ffn-attn-layer",
            nargs="+",
            type=int,
            metavar="INT INT",
            help=":The list of sharing layers for both feed-forward network and attention. The range of layer number starts from 1, and the layer number which is over the range would not work.",
        )
        parser.add_argument(
            '--ctc-compress-out',  
            action='store_true', 
            default=False,
            help="If set, compress the CTC output based on predictions"
        )
        parser.add_argument(
            '--ctc-compress-strategy', 
            type=str, 
            default="avg",
            choices=['avg', 'weighted', 'softmax'],
            help="Strategy to use when compressing CTC output"
        )


    @classmethod
    def build_encoder(cls, args, task):
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        encoder = ConvTransformerEncoder(args, src_dict if src_dict is not None else tgt_dict)
        if getattr(args, "load_pretrained_encoder_from", None):
            encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=encoder, checkpoint=args.load_pretrained_encoder_from
            )
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        decoder = TransformerDecoderNoExtra(args, task.target_dictionary, embed_tokens)
        if getattr(args, "load_pretrained_decoder_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_decoder_from
            )
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.token_embed_dim 
        )
        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    @staticmethod
    @torch.jit.unused
    def set_batch_first(lprobs):
        lprobs.batch_first = True

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        if self.training:
            self.set_batch_first(lprobs)
        return lprobs

    def output_layout(self):
        return "BTD"

    """
    The forward method inherited from the base class has a **kwargs argument in
    its input, which is not supported in torchscript. This method overrites the forward
    method definition without **kwargs.
    """

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out


class ConvTransformerEncoder(FairseqEncoder):
    """Conv + Transformer encoder"""

    def __init__(self, args, dictionary):
        """Construct an Encoder object."""
        super().__init__(dictionary)

        self.encoder_mask_future_delay = getattr(args, "encoder_mask_future_delay", float('inf'))
        self.encoder_mask_block_size = getattr(args, "encoder_mask_block_size", 1)

        self.dropout = args.dropout
        self.embed_scale = (
            1.0 if args.no_scale_embedding else math.sqrt(args.encoder_embed_dim)
        )
        self.padding_idx = 1
        self.in_channels = 1
        self.input_dim = args.input_feat_per_channel
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, args.conv_out_channels, 3, stride=2, padding=3 // 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                args.conv_out_channels,
                args.conv_out_channels,
                3,
                stride=2,
                padding=3 // 2,
            ),
            torch.nn.ReLU(),
        )
        transformer_input_dim = self.infer_conv_output_dim(
            self.in_channels, self.input_dim, args.conv_out_channels
        )
        self.out = torch.nn.Linear(transformer_input_dim, args.encoder_embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions,
            args.encoder_embed_dim,
            self.padding_idx,
            learned=False,
        )

        self.transformer_layers = nn.ModuleList([])

        # Layer sharing code
        encoder_weight_share_list = getattr(args, "share_encoder_ffn_attn_layer", None)
        if encoder_weight_share_list is None:
            encoder_weight_share_list = []
        else:
            shared_weights_layer = TransformerEncoderLayer(args)
        print(f"Encoder: Sharing layers: {encoder_weight_share_list}")
        for layer_idx in range(args.encoder_layers):
            if layer_idx+1 in encoder_weight_share_list:
                self.transformer_layers.append(shared_weights_layer)
            else:
                self.transformer_layers.append(TransformerEncoderLayer(args))
                    

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        self.ctc_compress_out = getattr(args, "ctc_compress_out", False)
        if self.ctc_compress_out:
            self.ctc_fc = nn.Linear(args.encoder_embed_dim, len(dictionary))
            assert args.criterion == "ctc_multi_loss"
            self.ctc_layer = args.ctc_encoder_layer
            self.ctc_compress_method = getattr(CTCCompressStrategy, args.ctc_compress_strategy)

    def pooling_ratio(self):
        return 4

    def infer_conv_output_dim(self, in_channels, input_dim, out_channels):
        sample_seq_len = 200
        sample_bsz = 10
        x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
        x = torch.nn.Conv2d(1, out_channels, 3, stride=2, padding=3 // 2)(x)
        x = torch.nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=3 // 2)(x)
        x = x.transpose(1, 2)
        mb, seq = x.size()[:2]
        return x.contiguous().view(mb, seq, -1).size(-1)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        delay = self.encoder_mask_future_delay
        block_size = self.encoder_mask_block_size

        if (delay >= dim-1) or (block_size >= dim): # Full attention allowed, no need to check other conditions
            self._future_mask = torch.zeros([dim, dim])
        else: # Start with mask that disallows looking into future
            tri_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )

            # Create additional masks that consider self.encoder_mask_future_delay and self.encoder_mask_block_size
            block_count = math.ceil(dim / block_size)
            blocks = torch.full((block_count, block_size, block_size), 1, dtype=torch.bool)
            block_mask = torch.nn.functional.pad(input=torch.block_diag(*blocks), pad=(0, 0, 0, 0))[:dim,:dim]            
            delay_mask = torch.cat(
                (
                    torch.full((dim,delay+1), 1, dtype=torch.bool),
                    torch.zeros( (dim,dim-(delay+1)), dtype=torch.bool)
                ), 1
            )
            corr_mask = torch.logical_or(block_mask, delay_mask)

            self._future_mask = tri_mask.masked_fill_(corr_mask, 0) # Apply correction

        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def forward(self, src_tokens, src_lengths, return_all_hiddens=False,):
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

        subsampling_factor = int(max_seq_len * 1.0 / output_seq_len + 0.5)
        input_len_0 = (src_lengths.float() / subsampling_factor).ceil().long()
        input_len_1 = x.size(0) * torch.ones([src_lengths.size(0)]).long().to(
            input_len_0.device
        )
        input_lengths = torch.min(input_len_0, input_len_1)

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)

        encoder_states = [] if return_all_hiddens else None

        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        for l_idx, layer in enumerate(self.transformer_layers):
            x = layer(x=x, encoder_padding_mask=encoder_padding_mask, attn_mask=self.buffered_future_mask(x))
            if self.ctc_compress_out and self.ctc_layer == l_idx + 1:
                ctc_padding_mask = encoder_padding_mask
                x_ctc, x, input_lengths = self.average_same_ctc_features(x, input_lengths)
                encoder_padding_mask = lengths_to_padding_mask(input_lengths)
            
            if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)

        if not encoder_padding_mask.any():
            maybe_encoder_padding_mask = None
        else:
            maybe_encoder_padding_mask = encoder_padding_mask

        if self.ctc_compress_out:
            if not ctc_padding_mask.any():
                maybe_ctc_padding_mask = None
            else:
                maybe_ctc_padding_mask = ctc_padding_mask
                
            return {
                "encoder_out": [x],
                "encoder_padding_mask": [maybe_encoder_padding_mask]
                if maybe_encoder_padding_mask is not None
                else [],
                "encoder_embedding": [],
                "encoder_states": [encoder_states],
                "src_tokens": [],
                "src_lengths": [],
                "ctc_out": [x_ctc],
                "ctc_padding_mask": [maybe_ctc_padding_mask]
                if maybe_ctc_padding_mask is not None
                else [],
            }
        else:  
            return {
                "encoder_out": [x],
                "encoder_padding_mask": [maybe_encoder_padding_mask]
                if maybe_encoder_padding_mask is not None
                else [],
                "encoder_embedding": [],
                "encoder_states": [encoder_states],
                "src_tokens": [],
                "src_lengths": [],
            }

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
        return x_ctc, compressed_output.permute(2, 0, 1), src_lengths.new(new_lengths)

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                (encoder_out["encoder_padding_mask"][0]).index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                (encoder_out["encoder_embedding"][0]).index_select(0, new_order)
            ]
        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,
            "encoder_padding_mask": new_encoder_padding_mask,
            "encoder_embedding": new_encoder_embedding,
            "encoder_states": encoder_states,
            "src_tokens": [],
            "src_lengths": [],
        }


class TransformerDecoderNoExtra(TransformerDecoder):
    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, _ = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        return x, None

class CTCCompressStrategy:
    @staticmethod
    def avg(prob_ctc, predicted, new_lengths, dtype, device):
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = 1.0 / same[1]
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix.to(device)

    @staticmethod
    def weighted(prob_ctc, predicted, new_lengths, dtype, device):
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype, device=device)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                # Get the probabilities of the prediction for the different time steps as weight
                weights = prob_ctc[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, same[0]]
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = \
                    weights / weights.sum()
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix

    @staticmethod
    def softmax(prob_ctc, predicted, new_lengths, dtype, device):
        new_maxlen = max(new_lengths)
        weights_matrix = torch.zeros((prob_ctc.shape[0], prob_ctc.shape[1], new_maxlen), dtype=dtype, device=device)
        for b_idx, pred in enumerate(predicted):
            processed_inputs_cnt = 0
            for t_idx, same in enumerate(pred):
                new_processed_inputs_cnt = processed_inputs_cnt + same[1]
                # Get the probabilities of the prediction for the different time steps as weight
                weights = F.softmax(prob_ctc[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, same[0]])
                weights_matrix[b_idx, processed_inputs_cnt:new_processed_inputs_cnt, t_idx] = \
                    weights / weights.sum()
                processed_inputs_cnt = new_processed_inputs_cnt
        return weights_matrix

@register_model_architecture(model_name="convtransformer", arch_name="convtransformer")
def base_architecture(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)

    args.token_embed_dim = getattr(args, "token_embed_dim", args.decoder_embed_dim) 
    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.token_embed_dim) 
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.token_embed_dim) 
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.max_source_positions = getattr(args, "max_source_positions", 3000)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.conv_out_channels = getattr(args, "conv_out_channels", args.encoder_embed_dim)


@register_model_architecture("convtransformer", "convtransformer_espnet")
def convtransformer_espnet(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
