"""BYOL for Audio: Downstream task dataset utility.

The `create_data_source()` will generate the data source class object for downstream tasks.
The data source will provide task training details such as labels, fold splits,
classes, as well as audio file paths.
"""

import os
import sys
import shutil
import random
import numpy as np
from pathlib import Path
import pandas as pd
import copy
import re
import logging
from collections import defaultdict
from itertools import chain
import subprocess
import torch

from .torch_mlp_clf import train_test_sure_split


def read_task_df(task, base_folder):
    df = pd.read_csv(Path(base_folder)/f'work/metadata/{task}.csv')
    # replace all str label with int label
    df.label = df.label.map({l: i for i, l in enumerate(df.label.unique())})
    return df


def get_us8k(base_folder):
    df = read_task_df('us8k', base_folder)
    # add fold column
    df['fold'] = df.file_name.map(lambda s: int(s.split('/')[1][4:])) # 'audio/foldXX/*.wav'
    idxs = [None for _ in range(10)]
    for i, sdf in df.groupby('fold'):
        idxs[i - 1] = sdf.index.values
        print(f' {i}:{len(sdf)}', end='')
    print(' samples.')
    return df, idxs, True


def get_spc(base_folder, ver):
    df = read_task_df(f'spcv{ver}', base_folder)
    # make split index
    idxs = [None, None, None]
    idxs[0] = df[df.split == 'train'].index.values
    idxs[1] = df[df.split == 'val'].index.values
    idxs[2] = df[df.split == 'test'].index.values
    assert np.all(np.array([len(idx) for idx in idxs]) > 0)
    return df, idxs, False


def get_spcv1(base_folder):
    return get_spc(base_folder, 1)


def get_spcv2(base_folder):
    return get_spc(base_folder, 2)


def get_nsynth(base_folder):
    df = read_task_df('nsynth', base_folder)
    # make split index
    idxs = [None, None, None]
    idxs[0] = df[df.split == 'train'].index.values
    idxs[1] = df[df.split == 'valid'].index.values
    idxs[2] = df[df.split == 'test'].index.values
    assert np.all(np.array([len(idx) for idx in idxs]) > 0)
    return df, idxs, False


def get_fsdnoisy18k(base_folder):
    df = read_task_df('fsdnoisy18k', base_folder)
    # make split index
    idxs = [None, None, None]
    idxs[0] = df[df.split == 'train'].index.values
    idxs[1] = df[df.split == 'valid'].index.values
    idxs[2] = df[df.split == 'test'].index.values
    assert np.all(np.array([len(idx) for idx in idxs]) > 0)
    return df, idxs, False


def load_metadata(mode, base):
    getter = eval('get_' + mode)
    df, fold_idxs, loocv = getter(base)
    return df, fold_idxs, loocv


class BaseDataSource:
    """Data source base class, see TaskDataSource for the detail."""

    def __init__(self, df, fold_idxes, loocv):
        self.df, self.fold_idxes, self.loocv = df, fold_idxes, loocv
        self.get_idxes = None

    @property
    def labels(self):
        if self.get_idxes is not None:
            return self.df.label.values[self.get_idxes]
        return self.df.label.values

    def __len__(self):
        if self.get_idxes is None:
            return len(self.df)
        return len(self.get_idxes)

    def index_of_folds(self, folds):
        idxes = []
        for fold in folds:
            idxes.extend(self.fold_idxes[fold])
        return idxes

    def subset_by_idxes(self, idxes):
        dup = copy.copy(self)
        dup.get_idxes = idxes
        return dup

    def subset(self, folds):
        """Returns a subset data source for the fold indexes.
        folds: List of fold indexes.
        """
        return self.subset_by_idxes(self.index_of_folds(folds))

    @property
    def n_folds(self):
        return len(self.fold_idxes)

    @property
    def n_classes(self):
        return len(set(self.df.label.values))

    def real_index(self, index):
        if self.get_idxes is not None:
            index = self.get_idxes[index]
        return index


class TaskDataSource(BaseDataSource):
    """Downstream task data source class.

    This class provides files and metadata in the dataset,
    as well as methods to manage data splits/folds.

    Properties:
        files: Audio sample pathnames.
        labels: List of int, class indexes of samples.
        classes: List of int, possible class indexes. [0, 1, 2, ] for example of 3 classes.
        n_classes: Number of classes
        n_folds: Number of folds.

    Methods:
        subset: Makes subset data source access.
    """

    def __init__(self, mode):
        super().__init__(*load_metadata(mode, '.'))
        self.audio_folder = Path('work/16k/' + mode)

    def file_name(self, index):
        index = self.real_index(index)
        return self.audio_folder/self.df.file_name.values[index]

    @property
    def files(self):
        return [self.file_name(i) for i in range(len(self))]


def create_data_source(mode):
    """Creates data source object for downstream task you want."""

    assert mode in ['us8k', 'spcv1', 'spcv2', 'nsynth', 'fsdnoisy18k']
    return TaskDataSource(mode)
