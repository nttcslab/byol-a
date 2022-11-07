"""Model definitions.

Reference:
    Y. Koizumi, D. Takeuchi, Y. Ohishi, N. Harada, and K. Kashino, “The NTT DCASE2020 challenge task 6 system:
    Automated audio captioning with keywords and sentence length estimation,” DCASE2020 Challenge, Tech. Rep., 2020.
    https://arxiv.org/abs/2007.00225
"""

import re
import logging
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


class NetworkCommonMixIn():
    """Common mixin for network definition."""

    def load_weight(self, weight_file, device, state_dict=None, key_check=True):
        """Utility to load a weight file to a device."""

        state_dict = state_dict or torch.load(weight_file, map_location=device)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # Remove unneeded prefixes from the keys of parameters.
        if key_check:
            weights = {}
            for k in state_dict:
                m = re.search(r'(^fc\.|\.fc\.|^features\.|\.features\.)', k)
                if m is None: continue
                new_k = k[m.start():]
                new_k = new_k[1:] if new_k[0] == '.' else new_k
                weights[new_k] = state_dict[k]
        else:
            weights = state_dict
        # Load weights and set model to eval().
        self.load_state_dict(weights)
        self.eval()
        logging.info(f'Using audio embbeding network pretrained weight: {Path(weight_file).name}')
        return self

    def set_trainable(self, trainable=False):
        for p in self.parameters():
            p.requires_grad = trainable


class AudioNTT2020Task6(nn.Module, NetworkCommonMixIn):
    """DCASE2020 Task6 NTT Solution Audio Embedding Network."""

    def __init__(self, n_mels, d):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (n_mels // (2**3)), d),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        self.d = d

    def forward(self, x):
        x = self.features(x)       # (batch, ch, mel, time)       
        x = x.permute(0, 3, 2, 1) # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C*D)) # (batch, time, mel*ch)
        x = self.fc(x)
        return x


class AudioNTT2020(AudioNTT2020Task6):
    """BYOL-A General Purpose Representation Network.
    This is an extension of the DCASE 2020 Task 6 NTT Solution Audio Embedding Network.
    """

    def __init__(self, n_mels=64, d=512):
        super().__init__(n_mels=n_mels, d=d)

    def forward(self, x):
        x = super().forward(x)
        (x1, _) = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)
        x = x1 + x2
        assert x.shape[1] == self.d and x.ndim == 2
        return x


class AudioNTT2020Task6X(nn.Module, NetworkCommonMixIn):
    """A variant of DCASE2020 Task6 NTT Solution Audio Embedding Network.
    Enabeld to return features by layers.
    Examples:
        model(x) -> returns sample-level features of [B, T, D].
        model(x, layered=True) -> returns sample-level layered features of [B, T, 5*D]
        model.by_layers(model.(x, layered=True)) -> returns sample-level features by layers as a list of [B, T, D] * 5
    """

    def __init__(self, n_mels, d):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64 * (n_mels // (2**3)), d),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(d, d),
            nn.ReLU(),
        )
        self.d = d
        self.n_feature_layer = 5

    def forward(self, x, layered=False):
        def reshape_conv_feature(v):
            B, CH, F, T = v.shape
            v = v.permute(0, 3, 1, 2).reshape(B, T, CH*F)
            # pad 0 at the end to make the feature dimension -> self.d
            if v.shape[-1] < self.d:
                v = torch.nn.functional.pad(v, (0, self.d - v.shape[-1]), 'constant', 0.0)
            # average to the target length
            while v.shape[1] > target_t:
                ## when odd time frames -> average last two frames into one frame
                if v.shape[1] % 2 == 1:
                    v = torch.cat([v[:, :-2], v[:, -2:].mean(1, keepdim=True)], axis=1)
                # [B, T, D] -> [B, T/2, D]
                T = v.shape[1]
                v = v.reshape(B, T//2, 2, v.shape[-1])
                v = v.mean(2) # average adjoining two time frame features.
            return v

        target_t = x.shape[-1] // 8
        features = []
        x = self.conv1(x)         # (batch, ch, mel, time)
        features.append(reshape_conv_feature(x))
        x = self.conv2(x)
        features.append(reshape_conv_feature(x))
        x = self.conv3(x)
        features.append(reshape_conv_feature(x))
        x = x.permute(0, 3, 2, 1) # (batch, time, mel, ch)
        B, T, D, C = x.shape
        x = x.reshape((B, T, C*D)) # (batch, time, mel*ch)
        x = self.fc1(x)
        features.append(x)
        x = self.fc2(x)
        features.append(x)

        if layered:
            return torch.cat(features, dim=-1) # [B, T, 5*D]
        return x # [B, T, D]

    def by_layers(self, layered_features):
        """Decompose layered features into the list of features for each layer."""
        *B, LD = layered_features.shape
        assert LD == self.n_feature_layer * self.d
        layered_features = layered_features.reshape(*B, self.n_feature_layer, self.d)
        layered_features = layered_features.permute(2, 0, 1, 3) if len(layered_features.shape) > 3 else layered_features.permute(1, 0, 2)
        return [layered_features[l] for l in range(self.n_feature_layer)]

    def load_weight(self, weight_file, device):
        """Whapper function for loading BYOL-A pre-trained weights."""
        namemap = {
            'features.0': 'conv1.0', 'features.1': 'conv1.1',
            'features.4': 'conv2.0', 'features.5': 'conv2.1',
            'features.8': 'conv3.0', 'features.9': 'conv3.1',
            'fc.0': 'fc1.0',
            'fc.3': 'fc2.1',
        }
        state_dict = torch.load(weight_file, map_location=device)
        new_dict = {}
        # replace keys and remove 'num_batches_tracked'
        for key in state_dict:
            if 'num_batches_tracked' in key:
                continue
            new_key = key
            for map_key in namemap:
                if map_key in key:
                    new_key = key.replace(map_key, namemap[map_key])
                    break
            new_dict[new_key] = state_dict[key]
        return super().load_weight(weight_file, device, state_dict=new_dict, key_check=False)


class AudioNTT2020X(AudioNTT2020Task6X):
    """BYOL-A General Purpose Representation Network.
    This is an extension of the DCASE 2020 Task 6 NTT Solution Audio Embedding Network.
    Enabeld to return features by layers.

    Examples:
        model(x) -> returns sample-level features of [B, D].
        model(x, layered=True) -> returns sample-level layered features of [B, 5*D]
        model(x, layered=True, by_layers=True) -> returns sample-level features by layers as a list of [B, D] * 5
    """

    def __init__(self, n_mels=64, d=2048):
        super().__init__(n_mels=n_mels, d=d)

    def forward(self, x, layered=False, by_layers=False):
        x = super().forward(x, layered=layered)
        (x1, _) = torch.max(x, dim=1)
        x2 = torch.mean(x, dim=1)
        x = x1 + x2
        if by_layers:
            return self.by_layers(x)
        return x
