"""Wrapper code for:

BYOL for Audio: Exploring Pre-trained General-purpose Audio Representations

## Reference
- [1] https://arxiv.org/abs/2204.07402
"""

from evar.ar_base import (BaseAudioRepr, ToLogMelSpec, calculate_norm_stats, normalize_spectrogram, temporal_pooling)
from evar.model_utils import load_pretrained_weights
import logging
from external.byol_a2.models import AudioNTT2022Encoder


class AR_BYOLA2(BaseAudioRepr):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.to_feature = ToLogMelSpec(cfg)

        self.body = AudioNTT2022Encoder(n_mels=cfg.n_mels, d=cfg.feature_d)
        if cfg.weight_file is not None and cfg.weight_file != '':
            load_pretrained_weights(self.body, cfg.weight_file, model_key='body')

    def precompute(self, device, data_loader):
        self.norm_stats = calculate_norm_stats(device, data_loader, self.to_feature)

    def encode_frames(self, batch_audio):
        x = self.to_feature(batch_audio)
        x = normalize_spectrogram(self.norm_stats, x) # B,F,T
        x = x.unsqueeze(1)    # -> B,1,F,T
        x = self.body(x)      # -> B,T,D=C*F
        x = x.transpose(1, 2) # -> B,D,T
        return x

    def forward(self, batch_audio):
        x = self.encode_frames(batch_audio)
        x = temporal_pooling(self, x)
        return x

