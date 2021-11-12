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

from byol_a.common import (os, sys, np, Path, random, torch, nn, DataLoader,
     get_logger, load_yaml_config, seed_everything, get_timestamp)
from byol_a.byol_pytorch import BYOL
from byol_a.models import AudioNTT2020
from byol_a.augmentations import (RandomResizeCrop, MixupBYOLA, RunningNorm, NormalizeBatch)
from byol_a.dataset import WaveInLMSOutDataset
import multiprocessing
import pytorch_lightning as pl
import fire


class AugmentationModule:
    """BYOL-A augmentation module example, the same parameter with the paper."""

    def __init__(self, size, epoch_samples, log_mixup_exp=True, mixup_ratio=0.4):
        self.train_transform = nn.Sequential(
            MixupBYOLA(ratio=mixup_ratio, log_mixup_exp=log_mixup_exp),
            RandomResizeCrop(virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.5), time_scale=(0.6, 1.5)),
        )
        self.pre_norm = RunningNorm(epoch_samples=epoch_samples)
        print('Augmentatoions:', self.train_transform)

    def __call__(self, x):
        x = self.pre_norm(x)
        return self.train_transform(x), self.train_transform(x)


class BYOLALearner(pl.LightningModule):
    """BYOL-A learner. Shows batch statistics for each epochs."""

    def __init__(self, model, lr, shape, **kwargs):
        super().__init__()
        self.learner = BYOL(model, image_size=shape, **kwargs)
        self.lr = lr
        self.post_norm = NormalizeBatch()

    def forward(self, images1, images2):
        return self.learner(images1, images2)

    def training_step(self, paired_inputs, batch_idx):
        def to_np(A): return [a.cpu().numpy() for a in A]

        bs = paired_inputs[0].shape[0]
        paired_inputs = torch.cat(paired_inputs) # [(B,1,F,T), (B,1,F,T)] -> (2*B,1,F,T)
        mb, sb = to_np((paired_inputs.mean(), paired_inputs.std()))
        paired_inputs = self.post_norm(paired_inputs)
        ma, sa = to_np((paired_inputs.mean(), paired_inputs.std()))

        loss = self.forward(paired_inputs[:bs], paired_inputs[bs:])
        for k, v in {'mb': mb, 'sb': sb, 'ma': ma, 'sa': sa}.items():
            self.log(k, float(v), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def on_before_zero_grad(self, _):
        self.learner.update_moving_average()


def main(audio_dir, config_path='config.yaml', d=None, epochs=None, resume=None) -> None:
    cfg = load_yaml_config(config_path)
    # Override configs
    cfg.feature_d = d or cfg.feature_d
    cfg.epochs = epochs or cfg.epochs
    cfg.resume = resume or cfg.resume
    # Essentials
    logger = get_logger(__name__)
    logger.info(cfg)
    seed_everything(cfg.seed)
    # Data preparation
    files = sorted(Path(audio_dir).glob('*.wav'))
    tfms = AugmentationModule((64, 96), 2 * len(files))
    ds = WaveInLMSOutDataset(cfg, files, labels=None, tfms=tfms)
    dl = DataLoader(ds, batch_size=cfg.bs,
                num_workers=multiprocessing.cpu_count(),
                pin_memory=True, shuffle=True,)
    logger.info(f'Dataset: {len(files)} .wav files from {audio_dir}')
    # Training preparation
    name = (f'BYOLA-NTT2020d{cfg.feature_d}s{cfg.shape[0]}x{cfg.shape[1]}-{get_timestamp()}'
            f'-e{cfg.epochs}-bs{cfg.bs}-lr{str(cfg.lr)[2:]}'
            f'-rs{cfg.seed}')
    logger.info(f'Training {name}...')
    # Model
    model = AudioNTT2020(n_mels=cfg.n_mels, d=cfg.feature_d)
    if cfg.resume is not None:
        model.load_weight(cfg.resume)
    # Training
    learner = BYOLALearner(model, cfg.lr, cfg.shape,
        hidden_layer=-1,
        projection_size=cfg.proj_size,
        projection_hidden_size=cfg.proj_dim,
        moving_average_decay=cfg.ema_decay,
    )
    trainer = pl.Trainer(gpus=1, max_epochs=cfg.epochs, weights_summary=None)
    trainer.fit(learner, dl)
    if trainer.interrupted:
        logger.info('Terminated.')
        exit(0)
    # Saving trained weight.
    to_file = Path(cfg.checkpoint_folder)/(name+'.pth')
    to_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), to_file)
    logger.info(f'Saved weight as {to_file}')


if __name__ == '__main__':
    fire.Fire(main)

