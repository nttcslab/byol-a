"""BYOL for Audio: Training.

SYNOPSIS:
    train.py AUDIO_DIR <flags>

FLAGS:
    --config_path=CONFIG_PATH
        Default: 'config.yaml'
    --d=D
        Default: feature_d in the config.yaml
    --epochs=EPOCHS
        Default: epochs in the config.yaml
    --resume=RESUME
        Pathname to the weight file to continue training
        Default: Not specified

Example of training on FSD50K dataset:
    # Preprocess audio files to convert to 16kHz in advance.
    python -m utils.convert_wav /path/to/fsd50k work/16k/fsd50k
    # Run training on dev set for 300 epochs
    python train.py work/16k/fsd50k/FSD50K.dev_audio --epochs=300
"""

from byol_a2.common import (np, Path, torch,
     get_logger, load_yaml_config, seed_everything, get_timestamp, hash_text)
from byol_a2.byol_pytorch import BYOL
from byol_a2.models import AudioNTT2022, load_pretrained_weights
from byol_a2.augmentations import (RandomResizeCrop, MixupBYOLA, RandomLinearFader, NormalizeBatch, PrecomputedNorm)
from byol_a2.dataset import WavDataset
import multiprocessing
import pytorch_lightning as pl
import fire
import logging
import nnAudio.features


class AugmentationModule:
    """BYOL-A augmentation module example, the same parameter with the paper."""

    def __init__(self, epoch_samples, log_mixup_exp=True, mixup_ratio=0.2):
        self.train_transform = torch.nn.Sequential(
            MixupBYOLA(ratio=mixup_ratio, log_mixup_exp=log_mixup_exp),
            RandomResizeCrop(virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5)),
            RandomLinearFader(),
        )
        logging.info(f'Augmentatoions: {self.train_transform}')

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)


class BYOLALearner(pl.LightningModule):
    """BYOL-A learner. Shows batch statistics for each epochs."""

    def __init__(self, cfg, model, tfms, **kwargs):
        super().__init__()
        self.learner = BYOL(model, image_size=cfg.shape, **kwargs)
        self.lr = cfg.lr
        self.tfms = tfms
        self.post_norm = NormalizeBatch()
        self.to_spec = nnAudio.features.MelSpectrogram(
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            fmin=cfg.f_min,
            fmax=cfg.f_max,
            center=True,
            power=2,
            verbose=False,
        )

    def forward(self, images1, images2):
        return self.learner(images1, images2)

    def training_step(self, wavs, batch_idx):
        def to_np(A): return [a.cpu().numpy() for a in A]
        # Convert raw audio into a log-mel spectrogram and pre-normalize it.
        self.to_spec.to(self.device, non_blocking=True)
        self.learner.to(self.device, non_blocking=True)
        lms_batch = (self.to_spec(wavs) + torch.finfo().eps).log().unsqueeze(1)
        lms_batch = self.pre_norm(lms_batch)
        # Create two augmented views.
        images1, images2 = [], []
        for lms in lms_batch:
            img1, img2 = self.tfms(lms)
            images1.append(img1), images2.append(img2)
        images1 = torch.stack(images1)
        images2 = torch.stack(images2)
        paired_inputs = (images1, images2)
        # Form a batch and post-normalize it.
        bs = paired_inputs[0].shape[0]
        paired_inputs = torch.cat(paired_inputs) # [(B,1,T,F), (B,1,T,F)] -> (2*B,1,T,F)
        mb, sb = to_np((paired_inputs.mean(), paired_inputs.std()))
        paired_inputs = self.post_norm(paired_inputs)
        ma, sa = to_np((paired_inputs.mean(), paired_inputs.std()))
        # Forward to get a loss.
        loss = self.forward(paired_inputs[:bs], paired_inputs[bs:])
        for k, v in {'mb': mb, 'sb': sb, 'ma': ma, 'sa': sa}.items():
            self.log(k, float(v), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_before_zero_grad(self, _):
        self.learner.update_moving_average()

    def calc_norm_stats(self, data_loader, n_stats=10000, device='cuda'):
        # Calculate normalization statistics from the training dataset.
        n_stats = min(n_stats, len(data_loader.dataset))
        logging.info(f'Calculating mean/std using random {n_stats} samples from population {len(data_loader.dataset)} samples...')
        self.to_spec.to(device)
        X = []
        for wavs in data_loader:
            lms_batch = (self.to_spec(wavs.to(device)) + torch.finfo().eps).log().unsqueeze(1)
            X.extend([x for x in lms_batch.detach().cpu().numpy()])
            if len(X) >= n_stats: break
        X = np.stack(X)
        norm_stats = np.array([X.mean(), X.std()])
        logging.info(f'  ==> mean/std: {norm_stats}, {norm_stats.shape} <- {X.shape}')
        self.pre_norm = PrecomputedNorm(norm_stats)
        return norm_stats


def complete_cfg(cfg):
    # Set ID.
    cfg.id = (f'AudioNTT2022-BYOLA-{cfg.shape[0]}x{cfg.shape[1]}d{cfg.feature_d}-{get_timestamp()}'
              f'-e{cfg.epochs}b{cfg.bs}l{str(cfg.lr)[2:]}r{cfg.seed}-{hash_text(str(cfg), L=8)}')
    return cfg


def main(audio_dir, config_path='config_v2.yaml', d=None, epochs=None, resume=None) -> None:
    cfg = load_yaml_config(config_path)
    # Override configs
    cfg.feature_d = d or cfg.feature_d
    cfg.epochs = epochs or cfg.epochs
    cfg.resume = resume or cfg.resume
    cfg.unit_samples = int(cfg.sample_rate * cfg.unit_sec)
    complete_cfg(cfg)
    # Essentials
    get_logger(__name__)
    logging.info(cfg)
    seed_everything(cfg.seed)
    # Data preparation
    files = sorted(Path(audio_dir).glob('*.wav'))
    tfms = AugmentationModule(epoch_samples=2 * len(files))
    ds = WavDataset(cfg, files, labels=None, tfms=None, random_crop=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.bs,
                num_workers=multiprocessing.cpu_count(),
                pin_memory=True, shuffle=True,)
    logging.info(f'Dataset: {len(files)} .wav files from {audio_dir}')
    # Training preparation
    logging.info(f'Training {cfg.id}...')
    # Model
    model = AudioNTT2022(n_mels=cfg.n_mels, d=cfg.feature_d)
    if cfg.resume is not None:
        load_pretrained_weights(model, cfg.resume)
    # Training
    learner = BYOLALearner(cfg, model, tfms=tfms,
        hidden_layer=-1,
        projection_size=cfg.proj_size,
        projection_hidden_size=cfg.proj_dim,
        moving_average_decay=cfg.ema_decay,
    )
    learner.calc_norm_stats(dl)
    trainer = pl.Trainer(gpus=cfg.gpus, max_epochs=cfg.epochs, weights_summary=None, accelerator="ddp")
    trainer.fit(learner, dl)
    if trainer.interrupted:
        logging.info('Terminated.')
        exit(0)
    # Saving trained weight.
    to_file = Path(cfg.checkpoint_folder)/(cfg.id+'.pth')
    to_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), to_file)
    logging.info(f'Saved weight as {to_file}')


if __name__ == '__main__':
    fire.Fire(main)

