import librosa
import yaml
import shutil
import argparse
import matplotlib.pyplot as plt
import math, pickle, os, glob
import numpy as np
from tqdm import tqdm
from parallel_wavegan.utils.audio import AudioProcessor
from multiprocessing import Pool

os.environ["OMP_NUM_THREADS"] = "1"

def get_files(path, extension=".wav"):
    filenames = []
    for filename in glob.iglob(f"{path}/**/*{extension}", recursive=True):
        filenames += [filename]
    return filenames


def process_file(path):
    wav = ap.load_wav(path)
    mel = ap.melspectrogram(wav)
    wav = wav.astype(np.float32)
    # check
    assert len(wav.shape) == 1, \
        f"{path} seems to be multi-channel signal."
    assert np.abs(wav).max() <= 1.0, \
        f"{path} seems to be different from 16 bit PCM."

    # gap when wav is not multiple of hop_length
    gap = wav.shape[0] % ap.hop_length
    assert mel.shape[1] * ap.hop_length == wav.shape[0] + ap.hop_length - gap, f'{mel.shape[1] * ap.hop_length} vs {wav.shape[0] + ap.hop_length + gap}'
    return mel.astype(np.float32), wav


def extract_feats(wav_path):
    idx = wav_path.split("/")[-1][:-4]
    m, wav = process_file(wav_path)   
    mel_path = f"{MEL_PATH}{idx}.npy"
    np.save(mel_path, m.astype(np.float32), allow_pickle=False)
    return wav_path, mel_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="path to config file for feature extraction."
    )
    parser.add_argument(
        "--num_procs", type=int, default=4, help="number of parallel processes."
    )
    parser.add_argument(
        "--data_path", type=str, default='', help="data path to overwrite config.json."
    )
    parser.add_argument(
        "--out_path", type=str, default='', help="destination to write files."
    )
    parser.add_argument(
        "--ignore_errors", type=bool, default=False, help="ignore bad files."
    )
    
    args = parser.parse_args()

    # load config
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    ap = AudioProcessor(**config['audio'])

    SEG_PATH = config['data_path']
    # OUT_PATH = os.path.join(args.out_path, CONFIG.run_name, "data/")
    OUT_PATH = args.out_path
    QUANT_PATH = os.path.join(OUT_PATH, "wavs/")
    MEL_PATH = os.path.join(OUT_PATH, "mel/")
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(QUANT_PATH, exist_ok=True)
    os.makedirs(MEL_PATH, exist_ok=True)

    wav_files = get_files(SEG_PATH)
    print(" > Number of audio files : {}".format(len(wav_files)))

    wav_file = wav_files[1]
    m, wav = process_file(wav_file)

    # This will take a while depending on size of dataset
    with Pool(args.num_procs) as p:
        dataset_ids = list(tqdm(p.imap(extract_feats, wav_files), total=len(wav_files)))

    # save metadata
    with open(os.path.join(OUT_PATH, "metadata.txt"), "w") as f:
        for data in dataset_ids:
            f.write(f"{data[0]}|{data[1]}\n")
