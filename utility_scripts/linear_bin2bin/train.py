import os
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichModelSummary, RichProgressBar, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.tuner import Tuner

from utility_scripts.metadata import (
    get_ears_personal_metadata,
    preprocess_ears_metadata,
    preprocess_wham_metadata,
    merge_ears_filepaths_with_metadata,
    prepare_for_training
)
from utility_scripts.configs import (
    MixingAudioDatasetConfig,
    AudioPreprocessorConfig,
    NormalizerConfig,
    AudioAugumentorConfig,
    LinearBin2BinTrainConfig
)
from utility_scripts.callbacks import SpectrogramLogger
from utility_scripts.datasets import *
from utility_scripts.linear_bin2bin.models import Bin2BinGenerator, Bin2BinDiscriminator
from utility_scripts.linear_bin2bin.strategies import Bin2Bin
from utility_scripts.utils import init_weights


def create_configs() -> dict[str, dataclass]:
    """Create and return configuration dataclasses for the training pipeline."""
    return {
        "mixing_audio_cfg": MixingAudioDatasetConfig(
            sample_rate=16000,
            segment_sec=1.02,
            overlap=0.0,
            min_snr=-2.5,
            max_snr=17.5,
            skip_ratio=2
        ),
        "audio_preprocessor_cfg": AudioPreprocessorConfig(
            sample_rate=16000,
            n_fft=512,
            window_length=512,
            hop_length=64,
            n_mels=None,
            top_db=80,
            spec_type="amplitude",
            mel_scale="htk",
            max_spec_shapes=(256, 256)
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
        "train_cfg": LinearBin2BinTrainConfig(
            batch_size=16,
            num_workers=4,
            max_epochs=200,
            learning_rate=0.0002,
            lambda_mag=45.0,
            lambda_sc=25.0,
            discriminator_train_freq=1,
            g_filters=32,
            d_filters=32,
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
        ears_df, train_percentage=train_percentage, reduce_to=reduced_size, verbose=True, filter_to={"style": ["regular"]}
    )
    train_wham_df, val_wham_df, test_wham_df = prepare_for_training(
        wham_df, train_percentage=train_percentage, reduce_to=reduced_size, verbose=True
    )
    # Merge val and test for validation
    val_ears_df = pd.concat([val_ears_df, test_ears_df])
    val_wham_df = pd.concat([val_wham_df, test_wham_df])
    
    return {
        "train_ears_df": train_ears_df,
        "val_ears_df": val_ears_df,
        "train_wham_df": train_wham_df,
        "val_wham_df": val_wham_df
    }


def create_dataframes(shared_path: str) -> dict[str, pd.DataFrame]:
    """Load and preprocess metadata for EARS and WHAM datasets."""
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


def create_scaler(cfg: NormalizerConfig) -> MinMaxFixedNormalizer:
    """Create scaler"""
    return MinMaxFixedNormalizer(cfg)


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


def create_pipeline(
        audio_preprocessor_cfg: AudioPreprocessorConfig, 
        audio_augmentor_cfg: AudioAugumentorConfig, 
        normalizer_cfg: NormalizerConfig
    ) -> DataPipeline:
    """Create the training pipeline with preprocessor, augmentor, scaler, and adjuster."""
    preprocessor = AmplitudeSpectrogramProcessor(config=audio_preprocessor_cfg)
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
        generator: Bin2BinGenerator, 
        discriminator: Bin2BinDiscriminator, 
        pipeline: DataPipeline, 
        scaler: Normalizer,
        cfg: LinearBin2BinTrainConfig
    ) -> Bin2Bin:
    """Create the Bin2Bin model for training."""
    model = Bin2Bin(
        generator=generator,
        discriminator=discriminator,
        pipeline=pipeline,
        scaler=scaler,
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
        filename="linear_bin2bin-{epoch:02d}-{val_loss:.4f}",
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
    logger = TensorBoardLogger(save_dir="tb_logs", name="linear_bin2bin")
    csv_logger = CSVLogger(save_dir="csv_logs", name="linear_bin2bin")
    return logger, csv_logger


def create_trainer(
        train_size: int,
        train_percentage: float,
        max_epochs: int,
        cfg: LinearBin2BinTrainConfig,
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


def train(train_size: int = 10_000, train_percentage: float = 0.8) -> tuple[pl.Trainer, Bin2Bin, AudioDataModule]:
    """Main function to set up and start training the Bin2Bin model."""
    configs = create_configs()
    dataframes = create_dataframes(shared_path="/kaggle/input")
    split_dfs = split_dataframes(dataframes["ears_df"], dataframes["wham_df"], train_percentage=train_percentage, reduced_size=None)
    
    train_dataset = create_dataset(
        ears_df=split_dfs["train_ears_df"],
        wham_df=split_dfs["train_wham_df"],
        mixing_audio_cfg=configs["mixing_audio_cfg"],
        train=True,
        skip_ratio=configs["mixing_audio_cfg"].skip_ratio
    )

    val_dataset = create_dataset(
        ears_df=split_dfs["val_ears_df"],
        wham_df=split_dfs["val_wham_df"],
        mixing_audio_cfg=configs["mixing_audio_cfg"],
        train=False,
        skip_ratio=configs["mixing_audio_cfg"].skip_ratio
    )

    pipeline = create_pipeline(
        audio_preprocessor_cfg=configs["audio_preprocessor_cfg"], 
        audio_augmentor_cfg=configs["audio_augumentor_cfg"], 
        normalizer_cfg=configs["normalizer_cfg"]
    )

    data_module = create_data_module(
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        batch_size=configs["train_cfg"].batch_size, 
        num_workers=configs["train_cfg"].num_workers
    )

    generator = Bin2BinGenerator(start_filters=configs["train_cfg"].g_filters)
    discriminator = Bin2BinDiscriminator(
        in_channels=configs["train_cfg"].d_input_channels, 
        filters=configs["train_cfg"].d_filters
    )

    generator.apply(init_weights)
    discriminator.apply(init_weights)

    model = create_strategy(
        generator=generator,
        discriminator=discriminator,
        pipeline=pipeline,
        scaler=create_scaler(configs["normalizer_cfg"]),
        cfg=configs["train_cfg"]
    )

    callbacks = create_callbacks()
    loggers = create_loggers()

    trainer = create_trainer(
        train_size=train_size,
        train_percentage=train_percentage,
        max_epochs=configs["train_cfg"].max_epochs,
        cfg=configs["train_cfg"],
        loggers=loggers,
        callbacks=callbacks
    )

    return trainer, model, data_module


# trainer, model, data_module = train(train_size=10_000, train_percentage=0.8)
# trainer.fit(model, datamodule=data_module)
