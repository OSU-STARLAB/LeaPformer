#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import os
from typing import List

import torch
from torch import nn

from fairseq import utils
from fairseq.data import encoders


def from_pretrained(
    model_name_or_path,
    checkpoint_file='model.pt',
    data_name_or_path='.',
    archive_map=None,
    **kwargs
):
    from fairseq import checkpoint_utils, file_utils

    if archive_map is not None:
        if model_name_or_path in archive_map:
            model_name_or_path = archive_map[model_name_or_path]
        if data_name_or_path is not None and data_name_or_path in archive_map:
            data_name_or_path = archive_map[data_name_or_path]

    model_path = file_utils.load_archive_file(model_name_or_path)

    # convenience hack for loading data and BPE codes from model archive
    if data_name_or_path.startswith('.'):
        kwargs['data'] = os.path.abspath(os.path.join(model_path, data_name_or_path))
    else:
        kwargs['data'] = file_utils.load_archive_file(data_name_or_path)
    for file, arg in {
        'code': 'bpe_codes',
        'bpecodes': 'bpe_codes',
        'sentencepiece.bpe.model': 'sentencepiece_vocab',
    }.items():
        path = os.path.join(model_path, file)
        if os.path.exists(path):
            kwargs[arg] = path

    models, args, task = checkpoint_utils.load_model_ensemble_and_task(
        [os.path.join(model_path, cpt) for cpt in checkpoint_file.split(':')],
        arg_overrides=kwargs,
    )

    return {
        'args': args,
        'task': task,
        'models': models,
    }


class GeneratorHubInterface(nn.Module):
    """
    PyTorch Hub interface for generating sequences from a pre-trained
    translation or language model.
    """

    def __init__(self, args, task, models):
        super().__init__()
        self.args = args
        self.task = task
        self.models = nn.ModuleList(models)
        self.src_dict = task.source_dictionary
        self.tgt_dict = task.target_dictionary

        # optimize model for generation
        for model in self.models:
            model.make_generation_fast_(
                beamable_mm_beam_size=(
                    None if getattr(args, 'no_beamable_mm', False)
                    else getattr(args, 'beam', 5)
                ),
                need_attn=getattr(args, 'print_alignment', False),
            )

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(getattr(args, 'replace_unk', None))

        self.tokenizer = encoders.build_tokenizer(args)
        self.bpe = encoders.build_bpe(args)

        # this is useful for determining the device
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def translate(self, sentence: str, verbose: bool = False, **kwargs) -> str:
        input = self.encode(sentence)
        hypo = self.generate(input, verbose, **kwargs)
        return self.decode(hypo)

    def generate(self, tokens: torch.LongTensor, verbose: bool = False, **kwargs) -> torch.LongTensor:
        sample = self._build_sample(tokens)

        # build generator using current args as well as any kwargs
        gen_args = copy.copy(self.args)
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator(gen_args)

        translations = self.task.inference_step(generator, self.models, sample)

        if verbose:
            src_str_with_unk = self.string(tokens)
            print('S\t{}'.format(src_str_with_unk))

        # Process top predictions
        for hypo in translations[0][:min(len(translations), getattr(self.args, 'nbest', 1))]:
            hypo_str = self.decode(hypo['tokens'])
            if verbose:
                print('H\t{}\t{}'.format(hypo['score'], hypo_str))
                print('P\t{}'.format(
                    ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                ))
                if hypo['alignment'] is not None and getattr(self.args, 'print_alignment', False):
                    print('A\t{}'.format(
                        ' '.join(map(lambda x: str(utils.item(x)), hypo['alignment'].int().cpu()))
                    ))

        return hypo['tokens']

    def encode(self, sentence: str) -> torch.LongTensor:
        sentence = self.tokenize(sentence)
        sentence = self.apply_bpe(sentence)
        return self.binarize(sentence)

    def decode(self, tokens: torch.LongTensor) -> str:
        sentence = self.string(tokens)
        sentence = self.remove_bpe(sentence)
        return self.detokenize(sentence)

    def tokenize(self, sentence: str) -> str:
        if self.tokenizer is not None:
            sentence = self.tokenizer.encode(sentence)
        return sentence

    def detokenize(self, sentence: str) -> str:
        if self.tokenizer is not None:
            sentence = self.tokenizer.decode(sentence)
        return sentence

    def apply_bpe(self, sentence: str) -> str:
        if self.bpe is not None:
            sentence = self.bpe.encode(sentence)
        return sentence

    def remove_bpe(self, sentence: str) -> str:
        if self.bpe is not None:
            sentence = self.bpe.decode(sentence)
        return sentence

    def binarize(self, sentence: str) -> torch.LongTensor:
        return self.src_dict.encode_line(sentence, add_if_not_exist=False).long()

    def string(self, tokens: torch.LongTensor) -> str:
        return self.tgt_dict.string(tokens)

    def _build_sample(self, src_tokens: torch.LongTensor):
        assert torch.is_tensor(src_tokens)
        dataset = self.task.build_dataset_for_inference([src_tokens], [src_tokens.numel()])
        sample = dataset.collater([dataset[0]])
        sample = utils.apply_to_sample(
            lambda tensor: tensor.to(self.device),
            sample
        )
        return sample


class BPEHubInterface(object):
    """PyTorch Hub interface for Byte-Pair Encoding (BPE)."""

    def __init__(self, bpe, **kwargs):
        super().__init__()
        args = argparse.Namespace(bpe=bpe, **kwargs)
        self.bpe = encoders.build_bpe(args)
        assert self.bpe is not None

    def encode(self, sentence: str) -> str:
        return self.bpe.encode(sentence)

    def decode(self, sentence: str) -> str:
        return self.bpe.decode(sentence)


class TokenizerHubInterface(object):
    """PyTorch Hub interface for tokenization."""

    def __init__(self, tokenizer, **kwargs):
        super().__init__()
        args = argparse.Namespace(tokenizer=tokenizer, **kwargs)
        self.tokenizer = encoders.build_tokenizer(args)
        assert self.tokenizer is not None

    def encode(self, sentence: str) -> str:
        return self.tokenizer.encode(sentence)

    def decode(self, sentence: str) -> str:
        return self.tokenizer.decode(sentence)
