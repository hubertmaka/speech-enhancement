import os
from dataclasses import dataclass

import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichModelSummary, RichProgressBar, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from src.metadata import (
    get_ears_personal_metadata,
    preprocess_ears_metadata,
    preprocess_wham_metadata,
    merge_ears_filepaths_with_metadata,
    prepare_for_training
)
from src.configs import (
    MixingAudioDatasetConfig,
    AudioPreprocessorConfig,
    NormalizerConfig,
    AudioAugumentorConfig,
    MelMelReGANTrainConfig
)
from src.callbacks import SpectrogramLogger
from src.datasets import *
from src.mel_regan.models import MelReGANGenerator, MelReGANDiscriminator
from src.mel_regan.strategies import MelReGAN


def create_configs() -> dict[str, dataclass]:
    """Create and return configuration dataclasses for the training pipeline."""
    return {
        "mixing_audio_cfg": MixingAudioDatasetConfig(
            sample_rate=16000,
            # segment_sec=1.328,
            segment_sec=2.096,
            overlap=0.0,
            min_snr=-2.5,
            max_snr=17.5,
            skip_ratio=1
        ),
        "audio_preprocessor_cfg": AudioPreprocessorConfig(
            sample_rate=16000,
            n_fft=1024,
            window_length=1024,
            hop_length=256,
            n_mels=80,
            top_db=80,
            mask_loss_threshold=-1.0,
            mask_loss_weight=1.0,
            spec_type="amplitude",
            mel_scale="slaney",
            max_spec_shapes=(80, 128)
        ),
        "normalizer_cfg": NormalizerConfig(
            min_db=-80.0,
            max_db=0.0,
            scale_type="-1_1",
            std=1.0,
            mean=0.0
        ),
        "audio_augumentor_cfg": AudioAugumentorConfig(
            time_mask_secs=0.1,
            freq_mask_bins=None
        ),
        "train_cfg": MelMelReGANTrainConfig(
            batch_size=64,
            num_workers=4,
            max_epochs=200,
            learning_rate=0.0002,
            lambda_mag=100.0,
            lambda_sc=100.0,
            discriminator_train_freq=2,
            label_smoothing=0.9,
            warmup_epochs=5,
            g_filters=128,
            d_filters=32,
            g_input_channels=1,
            d_input_channels=2
        )
    }



def create_dataset(
        noisy_filepaths: list[str], 
        clean_filepaths: list[str], 
        mixing_audio_cfg: MixingAudioDatasetConfig,
        *,
        skip_ratio: int = 1
    ) -> AudioLoader:
    """Create an AudioLoader for training or validation."""
    ds = AudioLoader(
        clean_filepaths=clean_filepaths,
        noisy_filepaths=noisy_filepaths,
        config=mixing_audio_cfg,
        skip_ratio=skip_ratio
    )
    return ds


def create_scaler(cfg: NormalizerConfig) -> MinMaxFixedNormalizer:
    """Create scaler"""
    return MinMaxFixedNormalizer(cfg)


def create_pipeline(
        audio_preprocessor_cfg: AudioPreprocessorConfig, 
        audio_augmentor_cfg: AudioAugumentorConfig, 
        normalizer_cfg: NormalizerConfig
    ) -> DataPipeline:
    """Create the training pipeline with preprocessor, augmentor, scaler, and adjuster."""
    preprocessor = MelSpectrogramProcessor(config=audio_preprocessor_cfg)
    scale_converter = AudioToDBScaler(audio_preprocessor_cfg)
    augmentor = AudioAugmentor(
        audio_preprocessor_config=audio_preprocessor_cfg, 
        augumentor_config=audio_augmentor_cfg
    )
    adjuster = TrimAdjuster(audio_preprocessor_cfg)
    scaler = create_scaler(normalizer_cfg)

    pipeline = DataPipeline(
        preprocessor=preprocessor,
        augmentor=augmentor,
        scaler=scaler,
        adjuster=adjuster,
        scale_converter=scale_converter
    )
    return pipeline


def create_data_module(
        train_dataset: AudioLoader, 
        val_dataset: AudioLoader, 
        batch_size: int, 
        num_workers: int
    ) -> AudioDataModule:
    """Create a PyTorch Lightning DataModule for training."""
    data_module = AudioDataModule(
        train_ds=train_dataset, 
        val_ds=val_dataset, 
        batch_size=batch_size,
        num_workers=num_workers 
    )
    return data_module


def create_strategy(
        generator: MelReGANGenerator, 
        discriminator: MelReGANDiscriminator, 
        pipeline: DataPipeline, 
        scaler: Normalizer,
        audio_cfg: AudioPreprocessorConfig,
        cfg: MelMelReGANTrainConfig
    ) -> MelReGAN:
    """Create the MelReGAN model for training."""
    model = MelReGAN(
        generator=generator,
        discriminator=discriminator,
        pipeline=pipeline,
        scaler=scaler,
        audio_cfg=audio_cfg,
        lr=cfg.learning_rate,
        lambda_mag=cfg.lambda_mag,
        lambda_sc=cfg.lambda_sc,
        discriminator_train_freq=cfg.discriminator_train_freq
    )
    return model


def create_callbacks() -> list:
    """Create a list of callbacks for training."""
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="MelReGAN-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        verbose=True,
        save_last=True,
        mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=50,
        verbose=True,
        mode="min"
    )

    image_callback = SpectrogramLogger()
    progress_bar_callback = progress_bar_callback = TQDMProgressBar()
    model_summary_callback = ModelSummary(max_depth=2)

    # dev_stats_callback = DeviceStatsMonitor()

    return [checkpoint_callback, early_stopping_callback, image_callback, progress_bar_callback, model_summary_callback]


def create_loggers() -> tuple[TensorBoardLogger, CSVLogger]:
    """Create TensorBoard and CSV loggers."""
    logger = TensorBoardLogger(save_dir="tb_logs", name="mel_MelReGAN")
    csv_logger = CSVLogger(save_dir="csv_logs", name="mel_MelReGAN")
    return logger, csv_logger

def create_trainer(
        train_size: int,
        # val_size: int,
        train_percentage: float,
        max_epochs: int,
        cfg: MelMelReGANTrainConfig,
        loggers: list,
        callbacks: list,
):
    limit_train_batches = train_size // cfg.batch_size
    limit_val_batches = train_size // cfg.batch_size

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=loggers,
        # limit_train_batches=limit_train_batches,
        # limit_val_batches=limit_val_batches,
        accelerator="auto",
        devices="auto",
        benchmark=True,
        precision="16-mixed",
        deterministic=False,
        log_every_n_steps=50,
    )
    return trainer