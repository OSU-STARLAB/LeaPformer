# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import torch.nn as nn

from . import FairseqDecoder, FairseqEncoder


class FairseqModel(nn.Module):
    """Base class for encoder-decoder models."""

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, FairseqDecoder)

        self.src_dict = encoder.dictionary
        self.dst_dict = decoder.dictionary
        assert self.src_dict.pad() == self.dst_dict.pad()
        assert self.src_dict.eos() == self.dst_dict.eos()
        assert self.src_dict.unk() == self.dst_dict.unk()

        self._is_generation_fast = False

    def forward(self, src_tokens, input_tokens):
        encoder_out = self.encoder(src_tokens)
        decoder_out, _ = self.decoder(input_tokens, encoder_out)
        return decoder_out.view(-1, decoder_out.size(-1))

    def max_encoder_positions(self):
        """Maximum input length supported by the encoder."""
        return self.encoder.max_positions()

    def max_decoder_positions(self):
        """Maximum output length supported by the decoder."""
        return self.decoder.max_positions()

    def make_generation_fast_(self, **kwargs):
        """Optimize model for faster generation."""
        if self._is_generation_fast:
            return  # only apply once
        self._is_generation_fast = True

        # remove weight norm from all modules in the network
        def apply_remove_weight_norm(module):
            try:
                nn.utils.remove_weight_norm(module)
            except ValueError:  # this module didn't have weight norm
                return
        self.apply(apply_remove_weight_norm)

        def train(mode):
            if mode:
                raise RuntimeError('cannot train after make_generation_fast')

        # this model should no longer be used for training
        self.eval()
        self.train = train

        def apply_make_generation_fast_(module):
            if module != self and hasattr(module, 'make_generation_fast_'):
                module.make_generation_fast_(**kwargs)
        self.apply(apply_make_generation_fast_)
