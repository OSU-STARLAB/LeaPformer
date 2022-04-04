#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
import shutil
from itertools import groupby
from tempfile import NamedTemporaryFile
from typing import Tuple
import re

import numpy as np
import pandas as pd
import soundfile as sf
from examples.speech_to_text.data_utils_prep import (
    create_zip,
    extract_fbank_features,
    filter_manifest_df,
    gen_config_yaml,
    gen_vocab,
    get_zip_manifest,
    load_df_from_tsv,
    save_df_to_tsv,
    cal_gcmvn_stats,
)
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from fairseq.data.audio.audio_utils_prep import get_waveform


log = logging.getLogger(__name__)


MANIFEST_COLUMNS = ["id", "audio", "n_frames", "tgt_text", "speaker"]


class MUSTC(Dataset):
    """
    Create a Dataset for MuST-C. Each item is a tuple of the form:
    waveform, sample_rate, source utterance, target utterance, speaker_id,
    utterance_id
    """

    SPLITS = ["train", "dev", "tst-COMMON", "tst-HE"]
    LANGUAGES = ["de", "es", "fr", "it", "nl", "pt", "ro", "ru"]

    def __init__(self, root: str, lang: str, split: str) -> None:
        assert split in self.SPLITS and lang in self.LANGUAGES
        _root = Path(root) / f"en-{lang}" / "data" / split
        wav_root, txt_root = _root / "wav", _root / "txt"
        assert _root.is_dir() and wav_root.is_dir() and txt_root.is_dir()
        # Load audio segments
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load the MuST-C YAML files")
        with open(txt_root / f"{split}.yaml") as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
        # Load source and target utterances
        for _lang in ["en", lang]:
            with open(txt_root / f"{split}.{_lang}") as f:
                utterances = [r.strip() for r in f]
            assert len(segments) == len(utterances)
            for i, u in enumerate(utterances):
                u = self.edit_utterance(u)
                segments[i][_lang] = u
        # Gather info
        self.data = []
        for wav_filename, _seg_group in groupby(segments, lambda x: x["wav"]):
            wav_path = wav_root / wav_filename
            sample_rate = sf.info(wav_path.as_posix()).samplerate
            seg_group = sorted(_seg_group, key=lambda x: x["offset"])
            for i, segment in enumerate(seg_group):
                offset = int(float(segment["offset"]) * sample_rate)
                n_frames = int(float(segment["duration"]) * sample_rate)
                _id = f"{wav_path.stem}_{i}"
                self.data.append(
                    (
                        wav_path.as_posix(),
                        offset,
                        n_frames,
                        sample_rate,
                        segment["en"],
                        segment[lang],
                        segment["speaker_id"],
                        _id,
                    )
                )

    def __getitem__(self, n: int) -> Tuple[torch.Tensor, int, str, str, str, str]:
        wav_path, offset, n_frames, sr, src_utt, tgt_utt, spk_id, utt_id = self.data[n]
        waveform, _ = get_waveform(wav_path, frames=n_frames, start=offset)
        waveform = torch.from_numpy(waveform)
        return waveform, sr, src_utt, tgt_utt, spk_id, utt_id

    def __len__(self) -> int:
        return len(self.data)

    #Edits the utterance by removing unecessary punctuations and adding special characters      
    def edit_utterance(self, utterance):
        #Removes punctuation
        utterance = utterance.replace(" -- ", " ")
        utterance = utterance.replace("--", "")
        utterance = utterance.replace("[", "")
        utterance = utterance.replace("]", "")
        utterance = utterance.replace("\"", "")
        utterance = utterance.replace(",", "")
        #utterance = utterance.replace("'", "")
        utterance = utterance.replace(";", "")
        utterance = utterance.replace("-", " ")
        utterance = utterance.replace("(Video) ", "")
        utterance = utterance.replace("(Audio) ", "")
        utterance = utterance.replace(" / ", " ")  
        utterance = utterance.replace("’", "'")
        utterance = re.sub("\u266A", " ", utterance)	# Remove music note
        utterance = re.sub("\u266B", " ", utterance)	# Remove music note
        #Complicated edits to utterance
        utterance = self.fix_and_remove_apostrophes(utterance)
        utterance = self.remove_repeating_end(utterance)
        #utterance = self.get_terminator_sentence(utterance)
        utterance = self.remove_speaker_initials(utterance)
        utterance = self.remove_speaker_two_name(utterance)
        utterance = self.replace_pauses(utterance)

        utterance = utterance.replace(":", "")
        utterance = utterance.strip()                    # Clean up white space at start/end of sentence
        utterance = ' '.join(utterance.split())          # Clean up white space within sentence
        return utterance

    #Replaces all values enclosed in () and starting with a capital letter to <0>
    def replace_pauses(self, utterance):
        #Provides number of ( in utterance
        pause_count = utterance.count("(")
        #Conditional satisfied if number of ( is greater than 0
        if pause_count > 0:
            #Loops for number of pause_count
            for i in range(pause_count):
                #Finds starting and ending index of pause
                index_start = utterance.find("(")
                index_end = utterance.find(")")
                #Replaces pause with <0> if first letter is capitalized
                if utterance[index_start+1].isupper():
                    utterance = utterance[:index_start] + "<0>" + utterance[index_end+1:]
        return utterance
        
    def fix_and_remove_apostrophes(self, utterance):
        # First loop: fix apostrophes
        #Finds index of first apostrophe
        index = utterance.find("'")
        while index != -1:
            # One condition to fix:
            # apostrophe has char + space before and space + word after, with word after being <= 2 chars (most likely contraction or possession)
            # e.g., They ' re , We ' ve , He ' s   
            if (index-2 >=0 and index+3 < len(utterance)) and \
              (utterance[index-2].isalpha() and utterance[index-1]==' ' and utterance[index+1]==' ' and utterance[index+2].isalpha() and utterance[index+3]==' '):
                utterance = utterance[:index-1] + "'" + utterance[index+2:] # Chop out spaces
            elif (index-2 >=0 and index+4 < len(utterance)) and \
              (utterance[index-2].isalpha() and utterance[index-1]==' ' and utterance[index+1]==' ' and utterance[index+2].isalpha() and utterance[index+3].isalpha() and utterance[index+4]==' '):
                utterance = utterance[:index-1] + "'" + utterance[index+2:] # Chop out spaces
            index+=1
            index = utterance.find("'", index)   
                
        # Second loop: remove undesired apostrophes
        #Finds index of first '
        index = utterance.find("'")
        while index != -1:
            # Only two allowed conditions:
            # 1) after any alpha + "s" and before space (e.g., childrens' )
            # 2) between two alpha chars (e.g., He's )
            if (index-2 >= 0 and index+1 < len(utterance)) and \
              (utterance[index-2].isalpha() and utterance[index-1] == 's' and utterance[index+1]==' '):
                index += 1
                pass
            elif (index-1 >= 0 and index+1 < len(utterance)) and \
              (utterance[index-1].isalpha() and utterance[index+1].isalpha()):
                index += 1
                pass
            else:
                utterance = utterance[:index] + utterance[index+1:]
            index = utterance.find("'", index)
        return utterance

    #Adds </s> after end_character designated
    def add_terminator(self, utterance, end_character):
        #Finds index of first end_character
        index = utterance.find(end_character)
        #Repeats while end_character remaining
        while index != -1:
            index += 1
            #Conditional satisfied if end_character is last in utterance
            if index == len(utterance):
                utterance = utterance + "</s>"
                return utterance
            #Conditional satisfied if first letter after end_character is capitalized, or is a digit, or is a special character not including a period.
            elif index + 1 < len(utterance) and (utterance[index + 1].isupper() or utterance[index + 1].isdigit() or (not utterance[index + 1].isalnum() and utterance[index + 1] != ".")):
                utterance = utterance[:index] + "</s>" + utterance[index:]
            index = utterance.find(end_character, index)
        return utterance

    #Places a terminating character after sentences ending with .,!, and ?
    def get_terminator_sentence(self, utterance):
        utterance = self.add_terminator(utterance, ".")
        utterance = self.add_terminator(utterance, "?")
        utterance = self.add_terminator(utterance, "!")
        return utterance

    #Removes the a speakers initials if it comes before a sentence
    def remove_speaker_initials(self, utterance):
        #Finds index of first :
        index = utterance.find(":")
        #Conditional satisfied if : located in utterance
        if index != -1:
            #Loops until index
            for i in range(index):
                #Continuesif character in utterance is capitalized
                if utterance[i].isupper():
                    continue
                #Returns original utterance if not a capital letter(Not initials)
                else:
                    return utterance
            return utterance[index+2:]
        return utterance

    #Removes the speakers first and last name from beginning of sentence
    def remove_speaker_two_name(self, utterance):
        #Finds index of : and space
        index_colon = utterance.find(":")
        index_space = utterance.find(" ")
        #Condition satisfied if both : and space present
        if index_space != -1 and index_colon != -1:
            #Condition satisfied if index_space is less than index_colon and word after space is capital and is the only other word before the colon
            if index_space < index_colon and utterance[index_space+1].isupper() and utterance[:index_colon].count(" ") <= 1:
                return utterance[index_colon+2:]
        return utterance

    #Removes repeating characters at the end of a sentence
    def remove_repeating_end(self, utterance):
        utterance = self.convert_multi_characters(utterance, ".")
        utterance = self.convert_multi_characters(utterance, "!")
        utterance = self.convert_multi_characters(utterance, "?")
        return utterance

    #Removes a repeating character from the end of sentences
    def convert_multi_characters(self, utterance, end_character):
        start_idx = end_idx = 0
        #Loop executes while there are still repeating end characters
        while utterance.count(end_character + end_character) != 0:
            #Finds the index of next end_character
            start_idx = end_idx = utterance.find(end_character, end_idx)
            #Finds index of the character after end_character
            while end_idx < len(utterance) and utterance[end_idx] == end_character:
                end_idx += 1
            # If not repeating character, no need to check anything so go to next loop
            if end_idx - start_idx == 1:
                continue

            #Conditional satisfied if not at end of sentence
            if end_idx < len(utterance):
                # Conditional satisfied if the following character is a number (e.g., 1.7 billion) or
                # a space at the end (blank caused by prior processing) or
                # the following character + 1 is a capital (e.g., "Good ... Let's do that.") or non-period special character
                if utterance[end_idx].isdigit() or \
                  (utterance[end_idx] == ' ' and end_idx + 1 == len(utterance)) or \
                  (end_idx + 1 < len(utterance) and (utterance[end_idx + 1].isupper() or (not utterance[end_idx + 1].isalnum() and utterance[end_idx + 1] != "."))):
                    utterance = utterance[0:start_idx] + end_character + utterance[end_idx:]
                else:
                    utterance = utterance[0:start_idx] + utterance[end_idx:]
            #Conditional satisfied if at end of sentence
            elif end_idx == len(utterance):
                utterance = utterance[0:start_idx] + end_character
        return utterance

    #Returns the count of the number of sentences in an utterance
    def get_sentence_count(self, utterance):
        sentence_count = 0
        sentence_count = self.increase_sentence_count(utterance, ".", sentence_count)
        sentence_count = self.increase_sentence_count(utterance, "?", sentence_count)
        sentence_count = self.increase_sentence_count(utterance, "!", sentence_count)
        return sentence_count

    #Returns the new sentence_count
    def increase_sentence_count(self, utterance, end_character, sentence_count):
        #Finds the index of the end_character
        index = utterance.find(end_character)
        #Repeats while more end_character remain
        while index != -1:
            index += 1
            #Conditional satisfied if index equal to length of utterance
            if index == len(utterance):
                sentence_count += 1
                return sentence_count
            #Conditional satisfied index less than length of utterance and is capital, or a digit, or a special character that is not a period. 
            elif index + 1 < len(utterance) and (utterance[index + 1].isupper() or utterance[index + 1].isdigit() or (not utterance[index + 1].isalnum() and utterance[index + 1] != ".")):
                sentence_count += 1
            #Finds the index of the next end_character
            index = utterance.find(end_character, index)
        return sentence_count

def process(args):
    root = Path(args.data_root).absolute()
    for lang in MUSTC.LANGUAGES:
        cur_root = root / f"en-{lang}"
        if not cur_root.is_dir():
            print(f"{cur_root.as_posix()} does not exist. Skipped.")
            continue
        # Extract features
        feature_root = cur_root / "fbank80"
        feature_root.mkdir(exist_ok=True)
        for split in MUSTC.SPLITS:
            print(f"Fetching split {split}...", flush=True)
            dataset = MUSTC(root.as_posix(), lang, split)
            print("Extracting log mel filter bank features...", flush=True)
            if split == 'train' and args.cmvn_type == "global":
                print("And estimating cepstral mean and variance stats...", flush=True)
                gcmvn_feature_list = []

            for waveform, sample_rate, _, _, _, utt_id in tqdm(dataset):
                features = extract_fbank_features(waveform, sample_rate)

                np.save(
                    (feature_root / f"{utt_id}.npy").as_posix(),
                    features
                )

                if split == 'train' and args.cmvn_type == "global":
                    if len(gcmvn_feature_list) < args.gcmvn_max_num:
                        gcmvn_feature_list.append(features)

            if split == 'train' and args.cmvn_type == "global":
                # Estimate and save cmv
                stats = cal_gcmvn_stats(gcmvn_feature_list)
                with open(cur_root / "gcmvn.npz", "wb") as f:
                    np.savez(f, mean=stats["mean"], std=stats["std"])

        # Pack features into ZIP
        zip_path = cur_root / "fbank80.zip"
        print("ZIPing features...", flush=True)
        create_zip(feature_root, zip_path)
        print("Fetching ZIP manifest...")
        zip_manifest = get_zip_manifest(zip_path)
        # Generate TSV manifest
        print("Generating manifest...")
        train_text = []
        for split in MUSTC.SPLITS:
            is_train_split = split.startswith("train")
            manifest = {c: [] for c in MANIFEST_COLUMNS}
            dataset = MUSTC(args.data_root, lang, split)
            for wav, sr, src_utt, tgt_utt, speaker_id, utt_id in tqdm(dataset):
                manifest["id"].append(utt_id)
                manifest["audio"].append(zip_manifest[utt_id])
                duration_ms = int(wav.size(1) / sr * 1000)
                manifest["n_frames"].append(int(1 + (duration_ms - 25) / 10))
                manifest["tgt_text"].append(src_utt if args.task == "asr" else tgt_utt)
                manifest["speaker"].append(speaker_id)
            if is_train_split:
                train_text.extend(manifest["tgt_text"])
            df = pd.DataFrame.from_dict(manifest)
            df = filter_manifest_df(df, is_train_split=is_train_split)
            save_df_to_tsv(df, cur_root / f"{split}_{args.task}.tsv")
        # Generate vocab
        v_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
        spm_filename_prefix = f"spm_{args.vocab_type}{v_size_str}_{args.task}"
        with NamedTemporaryFile(mode="w") as f:
            for t in train_text:
                f.write(t + "\n")
            gen_vocab(
                Path(f.name),
                cur_root / spm_filename_prefix,
                args.vocab_type,
                args.vocab_size,
            )
        # Generate config YAML
        gen_config_yaml(
            cur_root,
            spm_filename_prefix + ".model",
            yaml_filename=f"config_{args.task}.yaml",
            specaugment_policy="lb",
            cmvn_type=args.cmvn_type,
            gcmvn_path=(
                cur_root / "gcmvn.npz" if args.cmvn_type == "global"
                else None
            ),
        )
        # Clean up
        shutil.rmtree(feature_root)


def process_joint(args):
    cur_root = Path(args.data_root)
    assert all((cur_root / f"en-{lang}").is_dir() for lang in MUSTC.LANGUAGES), \
        "do not have downloaded data available for all 8 languages"
    # Generate vocab
    vocab_size_str = "" if args.vocab_type == "char" else str(args.vocab_size)
    spm_filename_prefix = f"spm_{args.vocab_type}{vocab_size_str}_{args.task}"
    with NamedTemporaryFile(mode="w") as f:
        for lang in MUSTC.LANGUAGES:
            tsv_path = cur_root / f"en-{lang}" / f"train_{args.task}.tsv"
            df = load_df_from_tsv(tsv_path)
            for t in df["tgt_text"]:
                f.write(t + "\n")
        special_symbols = None
        if args.task == 'st':
            special_symbols = [f'<lang:{lang}>' for lang in MUSTC.LANGUAGES]
        gen_vocab(
            Path(f.name),
            cur_root / spm_filename_prefix,
            args.vocab_type,
            args.vocab_size,
            special_symbols=special_symbols
        )
    # Generate config YAML
    gen_config_yaml(
        cur_root,
        spm_filename_prefix + ".model",
        yaml_filename=f"config_{args.task}.yaml",
        specaugment_policy="ld",
        prepend_tgt_lang_tag=(args.task == "st"),
    )
    # Make symbolic links to manifests
    for lang in MUSTC.LANGUAGES:
        for split in MUSTC.SPLITS:
            src_path = cur_root / f"en-{lang}" / f"{split}_{args.task}.tsv"
            desc_path = cur_root / f"{split}_{lang}_{args.task}.tsv"
            if not desc_path.is_symlink():
                os.symlink(src_path, desc_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument(
        "--vocab-type",
        default="unigram",
        required=True,
        type=str,
        choices=["bpe", "unigram", "char"],
    ),
    parser.add_argument("--vocab-size", default=8000, type=int)
    parser.add_argument("--task", type=str, choices=["asr", "st"])
    parser.add_argument("--joint", action="store_true", help="")
    parser.add_argument("--cmvn-type", default="utterance",
                        choices=["global", "utterance"],
                        help="The type of cepstral mean and variance normalization")
    parser.add_argument("--gcmvn-max-num", default=150000, type=int,
                        help=(
                            "Maximum number of sentences to use to estimate"
                            "global mean and variance"
                            ))
    args = parser.parse_args()

    if args.joint:
        process_joint(args)
    else:
        process(args)


if __name__ == "__main__":
    main()
