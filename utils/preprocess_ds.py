"""Metadata & audio preprocessor for downstream tasks.

This will create `work/{metadata, 16k}` folders for storing preprocessed files.
The `work/metadata` folder will have .csv files for the downstream tasks.
The `work/16k` folder will have folders that contain audio files.

Usage:
    python -m utils.preprocess_ds spcv2 /path/to/speech_commands_v0.02
    python -m utils.preprocess_ds spcv1 /path/to/speech_commands_v0.01
    python -m utils.preprocess_ds us8k /path/to/UrbanSound8K
"""

from pathlib import Path
import sys
import os
import numpy as np
import scipy as scipy
import librosa
import pickle
import pandas as pd
import json
import glob
import random
from tqdm import tqdm
import fire
from . import flatten_list
from .convert_wav import convert_wav


# UrbanSound8K https://urbansounddataset.weebly.com/urbansound8k.html

def convert_us8k_metadata(root):
    US8K = Path(root)
    df = pd.read_csv(US8K/f'metadata/UrbanSound8K.csv')
    df['file_name'] = df.fold.map(lambda x: f'audio/fold{x}/') + df.slice_file_name

    re_df = pd.DataFrame(df['class'].values, index=df.file_name, columns=['label'])
    re_df.to_csv(f'work/metadata/us8k.csv')

    # test
    df = pd.read_csv(f'work/metadata/us8k.csv').set_index('file_name')
    labels = df.label.values
    classes = sorted(set(list(labels)))
    assert len(classes) == 10
    assert len(df) == 8732
    print(f'Created work/metadata/us8k.csv - test passed')


def us8k(root):
    convert_us8k_metadata(root)
    convert_wav(root, 'work/16k/us8k', verbose=False)


# NSynth https://magenta.tensorflow.org/datasets/nsynth

def convert_nsynth_metadata(root, nsynth='nsynth', n_samples=305979, files=None):

    def read_meta(root, mode):
        j = json.load(open(f'{root}/nsynth-{mode}/examples.json'))
        loop_indexes = files if files and mode == 'train' else j
        file_names = [f'nsynth-{mode}/audio/{file_id}.wav' for file_id in loop_indexes]
        labels = [j[x]['instrument_family_str'] for x in loop_indexes]
        return pd.DataFrame({'file_name': file_names, 'label': labels, 'split': mode})

    df = pd.concat([read_meta(root, mode) for mode in ['train', 'valid', 'test']], ignore_index=True)
    df.to_csv(f'work/metadata/{nsynth}.csv')

    df = pd.read_csv(f'work/metadata/{nsynth}.csv')
    assert len(df) == n_samples, f'{len(df)}'
    print(f'Created work/metadata/{nsynth}.csv - test passed')


def nsynth(root):
    convert_nsynth_metadata(root)
    convert_wav(root, 'work/16k/nsynth', verbose=False)


# FSDnoisy18k http://www.eduardofonseca.net/FSDnoisy18k/

def convert_fsdnoisy18k_metadata(root):
    FSD = Path(root)
    train_df = pd.read_csv(FSD/f'FSDnoisy18k.meta/train.csv')
    # train_df = train_df[train_df.manually_verified != 0]
    # train_df = train_df[train_df.noisy_small == 0]
    test_df = pd.read_csv(FSD/f'FSDnoisy18k.meta/test.csv')
    # fname := split/fname
    train_df['fname'] = 'FSDnoisy18k.audio_train/' + train_df.fname
    test_df['fname'] = 'FSDnoisy18k.audio_test/' + test_df.fname
    # split. train -> train + val
    train_df['split'] = 'train'
    valid_index = np.random.choice(train_df.index.values, int(len(train_df) * 0.1), replace=False)
    train_df.loc[valid_index, 'split'] = 'valid'
    test_df['split'] = 'test'
    df = pd.concat([train_df, test_df], ignore_index=True)
    # filename -> file_name
    df.columns = [c if c != 'fname' else 'file_name' for c in df.columns]
    df.to_csv(f'work/metadata/fsdnoisy18k.csv', index=False)
    n_samples = len(df)

    df = pd.read_csv(f'work/metadata/fsdnoisy18k.csv')
    assert len(df) == n_samples, f'{len(df)}'
    print(f'Created work/metadata/fsdnoisy18k.csv - test passed')


def fsdnoisy18k(root):
    convert_fsdnoisy18k_metadata(root)
    convert_wav(root, 'work/16k/fsdnoisy18k', verbose=False)


# Speech Command https://arxiv.org/abs/1804.03209

def convert_spc_metadata(root, version=2):
    ROOT = Path(root)
    files = sorted(ROOT.glob('[a-z]*/*.wav'))
    
    labels = [f.parent.name for f in files]
    file_names = [f'{f.parent.name}/{f.name}' for f in files]
    df = pd.DataFrame({'file_name': file_names, 'label': labels})
    assert len(df) == [64721, 105829][version - 1] # v1, v2
    assert len(set(labels)) == [30, 35][version - 1] # v1, v2
    
    with open(ROOT/'validation_list.txt') as f:
        vals = [l.strip() for l in f.readlines()]
    with open(ROOT/'testing_list.txt') as f:
        tests = [l.strip() for l in f.readlines()]
    assert len(vals) == [6798, 9981][version - 1] # v1, v2
    assert len(tests) == [6835, 11005][version - 1] # v1, v2
    
    df['split'] = 'train'
    df.loc[df.file_name.isin(vals), 'split'] = 'val'
    df.loc[df.file_name.isin(tests), 'split'] = 'test'
    assert len(df[df.split == 'val']) == [6798, 9981][version - 1] # v1, v2
    assert len(df[df.split == 'test']) == [6835, 11005][version - 1] # v1, v2
    df.to_csv(f'work/metadata/spcv{version}.csv', index=False)

    # test
    df = pd.read_csv(f'work/metadata/spcv{version}.csv').set_index('file_name')
    assert len(df) == [64721, 105829][version - 1] # v1, v2
    print(f'Created work/metadata/spcv{version}.csv - test passed')


def spcv1(root):
    convert_spc_metadata(root, version=1)
    convert_wav(root, 'work/16k/spcv1', verbose=False)


def spcv2(root):
    convert_spc_metadata(root, version=2)
    convert_wav(root, 'work/16k/spcv2', verbose=False)


if __name__ == "__main__":
    Path('work/metadata').mkdir(parents=True, exist_ok=True)
    Path('work/16k').mkdir(parents=True, exist_ok=True)
    fire.Fire()