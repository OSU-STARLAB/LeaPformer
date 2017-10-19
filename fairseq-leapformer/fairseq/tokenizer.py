# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

from collections import Counter
import re

import torch

from fairseq import dictionary


def tokenize_line(line):
    line = re.sub(r"\t", "", line)
    line = re.sub(r"^\s+", "", line)
    line = re.sub(r"\s+$", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.split()


class Tokenizer:

    @staticmethod
    def build_dictionary(filename, tokenize=tokenize_line):
        dict = dictionary.Dictionary()
        Tokenizer.add_file_to_dictionary(filename, dict, tokenize)
        dict.finalize()
        return dict

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize):
        with open(filename, 'r') as f:
            for line in f:
                for word in tokenize(line):
                    dict.add_symbol(word)
                dict.add_symbol(dict.eos_word)

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenize_line):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, 'r') as f:
            for line in f:
                ids = Tokenizer.tokenize(line, dict, tokenize, add_if_not_exist=False, consumer=replaced_consumer)
                nseq += 1

                consumer(ids)
                ntok += len(ids)
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': len(replaced)}

    @staticmethod
    def tokenize(line, dict, tokenize=tokenize_line, add_if_not_exist=True, consumer=None):
        words = tokenize(line)
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1)
        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = dict.add_symbol(word)
            else:
                idx = dict.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        ids[nwords] = dict.eos_index
        return ids
