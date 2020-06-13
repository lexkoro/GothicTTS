import os
import numpy as np
from glob import glob
import re
import sys
from collections import Counter


def split_dataset(items):
    is_multi_speaker = False
    speakers = [item[-1] for item in items]
    # max 100 eval samples
    eval_split_size = 100 if len(items) * 0.01 > 100 else int(
        len(items) * 0.01)
    np.random.seed(0)
    np.random.shuffle(items)
    return items[:eval_split_size], items[eval_split_size:]


def load_meta_data(datasets):
    meta_data_train_all = []
    meta_data_eval_all = []
    for dataset in datasets:
        name = dataset['name']
        root_path = dataset['path']
        meta_file_train = dataset['metafile_train']
        meta_file_val = dataset['metafile_val']
        preprocessor = get_preprocessor_by_name(name)

        meta_data_train = preprocessor(root_path, meta_file_train)
        if meta_file_val is None:
            meta_data_eval, meta_data_train = split_dataset(meta_data_train)
        else:
            meta_data_eval = preprocessor(root_path, meta_file_val)
        meta_data_train_all += meta_data_train
        meta_data_eval_all += meta_data_eval
    return meta_data_train_all, meta_data_eval_all


def get_preprocessor_by_name(name):
    """Returns the respective preprocessing function."""
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name.lower())


def tts_gen(root_path, meta_file):
    """
    Args:
        root_path: path to root data folder.
        meta_file: path to metadata file which has
            audio and feature file paths at each line
            "/full/path/voice.wav|/full/path/mel.npy"
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('|')
            wav_file = cols[0]
            mel_file = cols[1].strip()
            # voice-file, mel-file, speaker-name
            items.append([wav_file, mel_file, "tts_gen"])
    return items
