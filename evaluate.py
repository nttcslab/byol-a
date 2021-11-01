"""BYOL for Audio: Linear evaluation using MLP classifier.

This program performs downstream task evaluation by following steps.

1. First, all the data audio files in the downstream task dataset are converted to the representation embeddings.
2. With the audio embeddings and corresponding labels, the linear layer is trained by using an MLP classifier,
   which is basically compatible with sklearn implementation, and then test accuracy is calculated.
   For the leave-one-out CV task, this step repeats for all folds and averages the accuracy.
3. Repeat the previous step, and average the accuracy.

Notes:
- TorchMLPClassifier is used instead of sklearn's MLPClassifier for faster evaluation.

"""

import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import chain
import multiprocessing
import torch
import logging
import re

from sklearn.preprocessing import StandardScaler
#from sklearn.neural_network import MLPClassifier
try:
    from utils.torch_mlp_clf import TorchMLPClassifier
except:
    raise Exception('Please follow Getting Started on the README.md to download and patch external modules.')
from utils.downstream_tasks import create_data_source
from utils import append_to_csv
from byol_a.common import seed_everything, load_yaml_config
from byol_a.augmentations import PrecomputedNorm
from byol_a.dataset import WaveInLMSOutDataset
from byol_a.models import AudioNTT2020


logging.basicConfig(level=logging.DEBUG)
device = torch.device('cuda')


def calc_norm_stats(cfg, data_src, n_stats=10000):
    """Calculates statistics of log-mel spectrogram features in a data source for normalization.

    Args:
        cfg: Configuration settings.
        data_src: Data source class object.
        n_stats: Maximum number of files to calculate statistics.
    """

    def data_for_stats(data_src):
        # use all files for LOO-CV (Leave One Out CV)
        if data_src.loocv:
            return data_src
        # use training samples only for non-LOOCV (train/eval/test) split.
        return data_src.subset([0])

    stats_data = data_for_stats(data_src)
    n_stats = min(n_stats, len(stats_data))
    logging.info(f'Calculating mean/std using random {n_stats} samples from training population {len(stats_data)} samples...')
    sample_idxes = np.random.choice(range(len(stats_data)), size=n_stats, replace=False)
    ds = WaveInLMSOutDataset(cfg, stats_data.files, labels=None, tfms=None)
    X = [ds[i] for i in tqdm(sample_idxes)]
    X = np.hstack(X)
    norm_stats = np.array([X.mean(), X.std()])
    logging.info(f'  ==> mean/std: {norm_stats}, {norm_stats.shape} <- {X.shape}')
    return norm_stats


def get_model_feature_d(model_filename):
    """Read number of fature_d in the filename."""

    r = re.search('d\d+', Path(model_filename).stem)
    if r is None:
        print(f'WARNING: feature dimension not found, falling back to 512-d: {model_filename}')
        d = 512
    else:
        d = int(r.group(0)[1:])
    return d


def get_embeddings(cfg, files, model, norm_stats):
    """Get representation embeddings of audio files, converted by the model.

    Args:
        cfg: Configuration settings.
        files: Audio files (.wav) to convert.
        model: Trained model that converts audio to embeddings.
        norm_stats: Mean & standard deviation calcurlated by calc_norm_stats().
    """

    ds = WaveInLMSOutDataset(cfg, files, labels=None, tfms=PrecomputedNorm(norm_stats))
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.bs, num_workers=cfg.num_workers,
                                     pin_memory=False, shuffle=False, drop_last=False)
    embs = []
    with torch.no_grad():
        for X in tqdm(dl):
            Y = model(X.to(device)).cpu().detach()
            embs.extend(Y.numpy())
    return np.array(embs)


def _one_linear_eval(X, y, X_val, y_val, X_test, y_test, hidden_sizes, epochs, early_stopping, debug):
    """Perform a single run of linear evaluation."""

    if len(X_test.shape) > 2:
        X = X.mean(axis=1)
        X_test = X_test.mean(axis=1)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    if X_val is not None:
        X_val = scaler.transform(X_val)

    clf_cls = TorchMLPClassifier
    clf = clf_cls(hidden_layer_sizes=hidden_sizes, max_iter=epochs,
                  early_stopping=early_stopping, debug=debug)
    clf.fit(X, y, X_val=X_val, y_val=y_val)

    X_test = scaler.transform(X_test)
    score = clf.score(X_test, y_test)
    return score


def linear_eval_multi(folds, hidden_sizes=(), epochs=200, early_stopping=True, debug=False):
    """Evaluate multiple folds with MLPClassifier for leave-one-out CV.

    Args:
        folds: Holds dataset X, y.
        hidden_sizes: MLP's hidden layer sizes.
        epochs: Training epochs.
        early_stopping: Enables early stopping or not.
    """

    def return_other_fold_indexes(test_fold_idx):
        return [i for i in range(len(folds)) if i != test_fold_idx]

    scores = []

    for fold_idx, test_fold in enumerate(folds):
        other_fold_indexes = return_other_fold_indexes(fold_idx)
        X = np.array(list(chain(*[folds[idx]['X'] for idx in other_fold_indexes]))).squeeze()
        y = np.array(list(chain(*[folds[idx]['y'] for idx in other_fold_indexes])))
        X_test = np.array(test_fold['X']).squeeze()
        y_test = np.array(test_fold['y'])

        score = _one_linear_eval(X, y, None, None, X_test, y_test, hidden_sizes, epochs, early_stopping, debug)
        scores.append(score)
        print(f' {score:.6f}', end='')
        debug = False # disable debug option for further iterations

    print(f' -> mean: {np.mean(scores)}')

    return np.mean(scores)


def linear_eval_single(folds, hidden_sizes=(), epochs=200, early_stopping=True, debug=False):
    """Evaluate a single train/test split with MLPClassifier.

    Args:
        folds: Holds dataset X, y as follows:
            0 = training set
            1 = validation set
            2 = test set
        hidden_sizes: MLP's hidden layer sizes
        epochs: Training epochs.
        early_stopping: Enables early stopping or not.
    """

    X, y = folds[0]['X'], folds[0]['y']
    X_val, y_val = folds[1]['X'], folds[1]['y']
    X_test, y_test = folds[2]['X'], folds[2]['y']
    print(f'Training:{len(X)}, validation:{len(X_val)}, test:{len(X_test)} samples.')

    score = _one_linear_eval(X, y, X_val, y_val, X_test, y_test, hidden_sizes, epochs, early_stopping, debug)
    print(f' {score:.6f}')

    return score


def prepare_linear_evaluation(weight_file, ds_task, unit_sec, n_stats=10000):
    """Prepare for linear evaluation.
    - Loads configuration settings, model, and downstream task data source.
    - Converts audio to representation embeddings.
    - Build folds for MLP classification.

    Returns:
        cfg: Configuration settings
        folds: Folds that hold X, y for all folds.
        loocv: True if the task is 10-folds LOO-CV, or False if it is a single fold (train/valid/test).
    """

    cfg = load_yaml_config('config.yaml')
    cfg.unit_sec = unit_sec
    cfg.feature_d = get_model_feature_d(weight_file)
    print(cfg)

    model = AudioNTT2020(n_mels=cfg.n_mels, d=cfg.feature_d)
    model.load_weight(weight_file, device)
    data = create_data_source(ds_task)

    # norm_stats
    norm_stats = calc_norm_stats(cfg, data, n_stats=n_stats)

    # embeddings
    model = model.to(device)
    model.eval()
    folds = [defaultdict(list) for _ in range(data.n_folds)]
    for i in range(data.n_folds):
        fold_data = data.subset([i])
        print(f'getting embeddings for fold #{i} ({len(fold_data)} samples)...')
        folds[i]['X'] = get_embeddings(cfg, fold_data.files, model, norm_stats)
        folds[i]['y'] = fold_data.labels

    return cfg, folds, data.loocv


def do_eval(weight, task='spcv2', unit_sec=1.0, repeat=1, epochs=200, early_stopping=True, seed=42):
    """Main program of linear evaluation."""

    # run deterministically
    seed_everything(seed)

    # load labels and corresponding pre-computed embeddings, Leave-One-Out CV flag, hidden layer sizes
    cfg, folds, loocv = prepare_linear_evaluation(weight, task, unit_sec)

    # run evaluation cycle
    results = {}
    for run_idx in range(repeat):
        if loocv:
            score = linear_eval_multi(folds, hidden_sizes=(), epochs=epochs,
                early_stopping=early_stopping, debug=(run_idx == 0))
        else:
            score = linear_eval_single(folds, hidden_sizes=(), epochs=epochs,
                early_stopping=early_stopping, debug=(run_idx == 0))
        results[f'run{run_idx}'] = score

    # calculate stats of scores
    scores = np.array(list(results.values()))
    m, s = scores.mean(), scores.std()
    model_name = Path(weight).stem
    results.update({'1_model': model_name, '2_mean': m, '3_std': s})
    logging.info(f' mean={m}, std={s}\n\n')

    # record score
    append_to_csv(f'results/{task}-scores.csv', results)
    print(m)


if __name__ == '__main__':
    import fire
    fire.Fire(do_eval)
