#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
from pathlib import Path
import soundfile as sf
from examples.speech_to_text.prep_covost_data import (
    CoVoST
)

from tqdm import tqdm

log = logging.getLogger(__name__)


def main(args):
    src_lang = args.src_lang
    tgt_lang = args.tgt_lang
    split = args.split
    pair_type = args.pair_type

    root = Path(args.data_root).absolute() / f"{src_lang}"
    assert root.is_dir(), (
        f"{root.as_posix()} does not exist. Skipped."
    )

    dataset = CoVoST(root.as_posix(), split, src_lang, tgt_lang, pair_type)
    output = Path(args.output).absolute()
    output.mkdir(exist_ok=True)
    f_text = open(output / f"{split}.{tgt_lang}", "w")
    f_wav_list = open(output / f"{split}.wav_list", "w")
    for waveform, sample_rate, _, text, _, utt_id in tqdm(dataset):
        sf.write(
            output / f"{utt_id}.wav",
            waveform.squeeze(0).numpy(),
            samplerate=int(sample_rate)
        )
        f_text.write(text + "\n")
        f_wav_list.write(str(output / f"{utt_id}.wav") + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--task", required=True, type=str, choices=["asr", "st"])
    parser.add_argument("--src-lang", required=True, type=str)
    parser.add_argument("--tgt-lang", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--split", required=True, choices=CoVoST.SPLITS)
    parser.add_argument("--pair-type", default=None, type=str, help="Method to create paired sentence dataset, if desired")
    args = parser.parse_args()

    main(args)
