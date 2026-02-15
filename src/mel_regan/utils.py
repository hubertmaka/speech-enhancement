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
            segment_sec=2.04,
            overlap=0.0,
            min_snr=-2.5,
            max_snr=17.5,
            skip_ratio=2
        ),
        "audio_preprocessor_cfg": AudioPreprocessorConfig(
            sample_rate=16000,
            n_fft=1024,
            window_length=1024,
            hop_length=256,
            n_mels=80,
            top_db=80,
            mask_loss_threshold=0.1,
            mask_loss_weight=0.5,
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
            time_mask_secs=0.12,
            freq_mask_bins=None
        ),
        "train_cfg": MelMelReGANTrainConfig(
            batch_size=32,
            num_workers=4,
            max_epochs=200,
            learning_rate=0.0002,
            lambda_mag=45.0,
            lambda_sc=25.0,
            discriminator_train_freq=2,
            label_smoothing=0.9,
            warmup_epochs=5,
            g_filters=32,
            d_filters=32,
            g_input_channels=1,
            d_input_channels=2
        )
    }


def split_dataframes(
        ears_df: pd.DataFrame, 
        wham_df: pd.DataFrame, 
        train_percentage: float = 0.8, 
        reduced_size: int | float | None = None
    ) -> dict[str, pd.DataFrame]:
    """Split EARS and WHAM dataframes into training, validation, and test sets."""
    train_ears_df, val_ears_df, test_ears_df = prepare_for_training(
        ears_df, train_percentage=train_percentage, reduce_to=reduced_size, verbose=True
    )
    train_wham_df, val_wham_df, test_wham_df = prepare_for_training(
        wham_df, train_percentage=train_percentage, reduce_to=reduced_size, verbose=True
    )
    return {
        "train_ears_df": train_ears_df,
        "val_ears_df": val_ears_df,
        "test_ears_df": test_ears_df,
        "train_wham_df": train_wham_df,
        "val_wham_df": val_wham_df,
        "test_wham_df": test_wham_df
    }


def create_dataframes(shared_path: str, kaggle: bool = False) -> dict[str, pd.DataFrame]:
    """Load and preprocess metadata for EARS and WHAM datasets."""
    if kaggle:
        COMMON_PATH = os.path.join(shared_path, "speech-enhancement")
        EARS_DATASET = os.path.join(COMMON_PATH, "ears_dataset", "ears_dataset", "speaker_statistics.json")
        EARS_FILES = os.path.join(COMMON_PATH, "ears_dataset", "ears_dataset", "ears_dataset_resampled")
        WHAM_DATA_TT = os.path.join(COMMON_PATH, "wham_noise", "wham_noise", "metadata", "mix_param_meta_tt.csv")
        WHAM_DATA_TR = os.path.join(COMMON_PATH, "wham_noise", "wham_noise", "metadata", "mix_param_meta_tr.csv")
        WHAM_DATA_CV = os.path.join(COMMON_PATH, "wham_noise", "wham_noise", "metadata", "mix_param_meta_cv.csv")
        WHAM_DATA_NOISE_TT = os.path.join(COMMON_PATH, "wham_noise", "wham_noise", "metadata", "noise_meta_tt.csv")
        WHAM_DATA_NOISE_TR = os.path.join(COMMON_PATH, "wham_noise", "wham_noise", "metadata", "noise_meta_tr.csv")
        WHAM_DATA_NOISE_CV = os.path.join(COMMON_PATH, "wham_noise", "wham_noise", "metadata", "noise_meta_cv.csv")
        WHAM_FILES_TT = os.path.join(COMMON_PATH, "wham_noise", "wham_noise", "resampled_tt")
        WHAM_FILES_CV = os.path.join(COMMON_PATH, "wham_noise", "wham_noise", "resampled_cv")
        WHAM_FILES_TR = os.path.join(COMMON_PATH, "wham_noise", "wham_noise", "resampled_tr")
    else:
        COMMON_PATH = shared_path
        EARS_DATASET = os.path.join(COMMON_PATH, "ears_dataset", "speaker_statistics.json")
        EARS_FILES = os.path.join(COMMON_PATH, "ears_dataset", "ears_dataset_resampled")
        WHAM_DATA_TT = os.path.join(COMMON_PATH, "wham_noise", "metadata", "mix_param_meta_tt.csv")
        WHAM_DATA_TR = os.path.join(COMMON_PATH, "wham_noise", "metadata", "mix_param_meta_tr.csv")
        WHAM_DATA_CV = os.path.join(COMMON_PATH, "wham_noise", "metadata", "mix_param_meta_cv.csv")
        WHAM_DATA_NOISE_TT = os.path.join(COMMON_PATH, "wham_noise", "metadata", "noise_meta_tt.csv")
        WHAM_DATA_NOISE_TR = os.path.join(COMMON_PATH, "wham_noise", "metadata", "noise_meta_tr.csv")
        WHAM_DATA_NOISE_CV = os.path.join(COMMON_PATH, "wham_noise", "metadata", "noise_meta_cv.csv")
        WHAM_FILES_TT = os.path.join(COMMON_PATH, "wham_noise", "resampled_tt")
        WHAM_FILES_CV = os.path.join(COMMON_PATH, "wham_noise", "resampled_cv")
        WHAM_FILES_TR = os.path.join(COMMON_PATH, "wham_noise", "resampled_tr")

    personal_metadata_df = get_ears_personal_metadata(EARS_DATASET)
    ears_metadata_df = preprocess_ears_metadata(EARS_FILES, verbose=False)
    wham_df = preprocess_wham_metadata(
        wham_data_cv=WHAM_DATA_CV,
        wham_data_tr=WHAM_DATA_TR,
        wham_data_tt=WHAM_DATA_TT,
        wham_files_cv=WHAM_FILES_CV,
        wham_files_tr=WHAM_FILES_TR,
        wham_files_tt=WHAM_FILES_TT,
        wham_noise_cv=WHAM_DATA_NOISE_CV,
        wham_noise_tr=WHAM_DATA_NOISE_TR,
        wham_noise_tt=WHAM_DATA_NOISE_TT,
        verbose=False
    )
    ears_df = merge_ears_filepaths_with_metadata(ears_metadata_df, personal_metadata_df)
    return {"wham_df": wham_df, "ears_df": ears_df}


def create_dataset(
        ears_df: pd.DataFrame, 
        wham_df: pd.DataFrame, 
        mixing_audio_cfg: MixingAudioDatasetConfig,
        *,
        train: bool = True, 
        skip_ratio: int = 2
    ) -> AudioMixingDataset:
    """Create an AudioMixingDataset for training or validation."""
    ds = AudioMixingDataset(
        ears_df=ears_df,
        wham_df=wham_df,
        config=mixing_audio_cfg,
        mode="train" if train else "val",
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
        train_dataset: AudioMixingDataset, 
        val_dataset: AudioMixingDataset, 
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
        filename="mel_MelReGAN-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        verbose=True,
        save_weights_only=True,
        mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=40,
        verbose=True,
        mode="min"
    )

    image_callback = SpectrogramLogger()
    progress_bar_callback = RichProgressBar(leave=True)
    model_summary_callback = RichModelSummary(max_depth=2)
    dev_stats_callback = DeviceStatsMonitor()

    return [checkpoint_callback, early_stopping_callback, image_callback, progress_bar_callback, model_summary_callback, dev_stats_callback]


def create_loggers() -> tuple[TensorBoardLogger, CSVLogger]:
    """Create TensorBoard and CSV loggers."""
    logger = TensorBoardLogger(save_dir="tb_logs", name="mel_MelReGAN")
    csv_logger = CSVLogger(save_dir="csv_logs", name="mel_MelReGAN")
    return logger, csv_logger

def create_trainer(
        train_size: int,
        train_percentage: float,
        max_epochs: int,
        cfg: MelMelReGANTrainConfig,
        loggers: list,
        callbacks: list,
):
    limit_train_batches = train_size // cfg.batch_size
    limit_val_batches = (1 - train_percentage) * train_size // cfg.batch_size

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=loggers,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        accelerator="auto",
        devices="auto",
        benchmark=True,
        deterministic=False,
        log_every_n_steps=50,
    )
    return trainer