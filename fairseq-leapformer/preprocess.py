#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
#

import argparse
from itertools import zip_longest
import os
import shutil

from fairseq import dictionary, indexed_dataset
from fairseq.tokenizer import Tokenizer, tokenize_line


def get_parser():
    parser = argparse.ArgumentParser(
        description='Data pre-processing: Create dictionary and store data in binary format')
    parser.add_argument('-s', '--source-lang', default=None, metavar='SRC', help='source language')
    parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET', help='target language')
    parser.add_argument('--trainpref', metavar='FP', default=None, help='target language')
    parser.add_argument('--validpref', metavar='FP', default=None, help='comma separated, valid language prefixes')
    parser.add_argument('--testpref', metavar='FP', default=None, help='comma separated, test language prefixes')
    parser.add_argument('--destdir', metavar='DIR', default='data-bin', help='destination dir')
    parser.add_argument('--thresholdtgt', metavar='N', default=0, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--thresholdsrc', metavar='N', default=0, type=int,
                        help='map words appearing less than threshold times to unknown')
    parser.add_argument('--tgtdict', metavar='FP', help='reuse given target dictionary')
    parser.add_argument('--srcdict', metavar='FP', help='reuse given source dictionary')
    parser.add_argument('--nwordstgt', metavar='N', default=-1, type=int, help='number of target words to retain')
    parser.add_argument('--nwordssrc', metavar='N', default=-1, type=int, help='number of source words to retain')
    parser.add_argument('--alignfile', metavar='ALIGN', default=None, help='an alignment file (optional)')
    parser.add_argument('--output-format', metavar='FORMAT', default='binary', choices=['binary', 'raw'],
                        help='output format (optional)')
    parser.add_argument('--joined-dictionary', action='store_true', help='Generate joined dictionary')
    parser.add_argument('--only-source', action='store_true', help='Only process the source language')
    return parser


def main(args):
    print(args)
    os.makedirs(args.destdir, exist_ok=True)
    target = not args.only_source

    if args.joined_dictionary:
        assert not args.srcdict, 'cannot combine --srcdict and --joined-dictionary'
        assert not args.tgtdict, 'cannot combine --tgtdict and --joined-dictionary'
        src_dict = dictionary.Dictionary()
        for lang in [args.source_lang, args.target_lang]:
            Tokenizer.add_file_to_dictionary(
                filename='{}.{}'.format(args.trainpref, lang),
                dict=src_dict,
                tokenize=tokenize_line,
            )
        src_dict.finalize()
        tgt_dict = src_dict
    else:
        if args.srcdict:
            src_dict = dictionary.Dictionary.load(args.srcdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = Tokenizer.build_dictionary(filename='{}.{}'.format(args.trainpref, args.source_lang))
        if target:
            if args.tgtdict:
                tgt_dict = dictionary.Dictionary.load(args.tgtdict)
            else:
                assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = Tokenizer.build_dictionary(filename='{}.{}'.format(args.trainpref, args.target_lang))

    src_dict.save(os.path.join(args.destdir, 'dict.{}.txt'.format(args.source_lang)),
                  threshold=args.thresholdsrc, nwords=args.nwordssrc)
    if target:
        tgt_dict.save(os.path.join(args.destdir, 'dict.{}.txt'.format(args.target_lang)),
                      threshold=args.thresholdtgt, nwords=args.nwordstgt)

    def make_binary_dataset(input_prefix, output_prefix, lang):
        dict = dictionary.Dictionary.load(os.path.join(args.destdir, 'dict.{}.txt'.format(lang)))
        print('| [{}] Dictionary: {} types'.format(lang, len(dict) - 1))

        ds = indexed_dataset.IndexedDatasetBuilder(
            '{}/{}.{}-{}.{}.bin'.format(args.destdir, output_prefix, args.source_lang,
                                        args.target_lang, lang)
        )

        def consumer(tensor):
            ds.add_item(tensor)

        input_file = '{}.{}'.format(input_prefix, lang)
        res = Tokenizer.binarize(input_file, dict, consumer)
        print('| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}'.format(
            lang, input_file, res['nseq'], res['ntok'],
            100 * res['nunk'] / res['ntok'], dict.unk_word))
        ds.finalize('{}/{}.{}-{}.{}.idx'.format(
            args.destdir, output_prefix,
            args.source_lang, args.target_lang, lang))

    def make_dataset(input_prefix, output_prefix, lang, output_format='binary'):
        if output_format == 'binary':
            make_binary_dataset(input_prefix, output_prefix, lang)
        elif output_format == 'raw':
            # Copy original text file to destination folder
            output_text_file = os.path.join(args.destdir, '{}.{}'.format(output_prefix, lang))
            shutil.copyfile('{}.{}'.format(input_prefix, lang), output_text_file)

    def make_all(args, make_dataset, lang):
        if args.trainpref:
            make_dataset(args.trainpref, 'train', lang, args.output_format)
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(',')):
                outprefix = 'valid{}'.format(k) if k > 0 else 'valid'
                make_dataset(validpref, outprefix, lang, args.output_format)
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(',')):
                outprefix = 'test{}'.format(k) if k > 0 else 'test'
                make_dataset(testpref, outprefix, lang, args.output_format)

    make_all(args, make_dataset, args.source_lang)
    if target:
        make_all(args, make_dataset, args.target_lang)

    print('| Wrote preprocessed data to {}'.format(args.destdir))

    if args.alignfile:
        assert args.trainpref, "--trainpref must be set if --alignfile is specified"
        src_file_name = '{}.{}'.format(args.trainpref, args.source_lang)
        tgt_file_name = '{}.{}'.format(args.trainpref, args.target_lang)
        src_dict = dictionary.Dictionary.load(os.path.join(args.destdir, 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = dictionary.Dictionary.load(os.path.join(args.destdir, 'dict.{}.txt'.format(args.target_lang)))
        freq_map = {}
        with open(args.alignfile, 'r') as align_file:
            with open(src_file_name, 'r') as src_file:
                with open(tgt_file_name, 'r') as tgt_file:
                    for a, s, t in zip_longest(align_file, src_file, tgt_file):
                        si = Tokenizer.tokenize(s, src_dict, add_if_not_exist=False)
                        ti = Tokenizer.tokenize(t, tgt_dict, add_if_not_exist=False)
                        ai = list(map(lambda x: tuple(x.split('-')), a.split()))
                        for sai, tai in ai:
                            srcidx = si[int(sai)]
                            tgtidx = ti[int(tai)]
                            if srcidx != src_dict.unk() and tgtidx != tgt_dict.unk():
                                assert srcidx != src_dict.pad()
                                assert srcidx != src_dict.eos()
                                assert tgtidx != tgt_dict.pad()
                                assert tgtidx != tgt_dict.eos()

                                if srcidx not in freq_map:
                                    freq_map[srcidx] = {}
                                if tgtidx not in freq_map[srcidx]:
                                    freq_map[srcidx][tgtidx] = 1
                                else:
                                    freq_map[srcidx][tgtidx] += 1

        align_dict = {}
        for srcidx in freq_map.keys():
            align_dict[srcidx] = max(freq_map[srcidx], key=freq_map[srcidx].get)

        with open(os.path.join(args.destdir, 'alignment.{}-{}.txt'.format(
                args.source_lang, args.target_lang)), 'w') as f:
            for k, v in align_dict.items():
                print('{} {}'.format(src_dict[k], tgt_dict[v]), file=f)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
