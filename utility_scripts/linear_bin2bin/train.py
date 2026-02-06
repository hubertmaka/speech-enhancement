import os

import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
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
    AudioAugumentorConfig
)
from utility_scripts.callbacks import SpectrogramLogger
from utility_scripts.datasets import *
from utility_scripts.linear_bin2bin.models import Bin2BinGenerator, Bin2BinDiscriminator
from utility_scripts.linear_bin2bin.strategies import Bin2Bin
from utility_scripts.utils import init_weights


def train(
        use_compile: bool = True, 
        use_channels_last: bool = True, 
        find_batch_size: bool = False
    ) -> None:
    """Function to train the Bin2Bin model."""
    print("=" * 80)
    print("🚀 STARTING TRAINING")
    print("=" * 80)
    
    SAMPLE_RATE = 16000
    SEGMENT_SEC = 1.02
    OVERLAP = 0.0

    MIN_SNR = -2.5
    MAX_SNR = 17.5

    FRAME_LEN = 512
    FRAME_STEP = 64
    FFT_LEN = 512
    TIME_MASK_SECS = 0.1
    TOP_DB = 80
    MELS = None

    BATCH_SIZE = 16 
    NUM_WORKERS = 4 
    
    MAX_EPOCHS = 200
    LEARNING_RATE = 0.0002

    LAMBDA_MAG = 45.0  
    LAMBDA_SC = 25.0   
    
    print(f"📊 Batch size: {BATCH_SIZE}")
    print(f"👷 Num workers: {NUM_WORKERS}")
    print(f"🎯 Lambda weights: mag={LAMBDA_MAG}, sc={LAMBDA_SC}")
    print(f"🎭 Time Masking: {TIME_MASK_SECS}s (Aggressive for inpainting)")
    print()

    # Configurations
    mixing_audio_cfg = MixingAudioDatasetConfig(
        sample_rate=SAMPLE_RATE,
        segment_sec=SEGMENT_SEC,
        overlap=OVERLAP,
        min_snr=MIN_SNR,
        max_snr=MAX_SNR
    )

    audio_preprocessor_cfg = AudioPreprocessorConfig(
        sample_rate=SAMPLE_RATE,
        n_fft=FRAME_LEN,
        hop_length=FRAME_STEP,
        window_length=FFT_LEN,
        n_mels=MELS,
        top_db=TOP_DB,
        max_spec_shapes=256
    )

    normalizer_cfg = NormalizerConfig()

    audio_augumentor_cfg = AudioAugumentorConfig(
        time_mask_secs=TIME_MASK_SECS,
        freq_mask_bins=None
    )
    
    print("📂 Loading datasets...")
    
    TRAIN_SIZE = 0.8
    REDUCE_SIZE = None 

    COMMON_PATH = os.path.join("/kaggle", "input", "speech-enhancement")
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

    train_ears_df, val_ears_df, test_ears_df = prepare_for_training(
        ears_df, train_percentage=TRAIN_SIZE, reduce_to=REDUCE_SIZE, verbose=True, filter_to={"style": ["regular"]}
    )
    train_wham_df, val_wham_df, test_wham_df = prepare_for_training(
        wham_df, train_percentage=TRAIN_SIZE, reduce_to=REDUCE_SIZE, verbose=True
    )
    val_ears_df = pd.concat([val_ears_df, test_ears_df])
    val_wham_df = pd.concat([val_wham_df, test_wham_df])

    train_dataset = AudioMixingDataset(
        ears_df=train_ears_df,
        wham_df=train_wham_df,
        config=mixing_audio_cfg,
        mode="train",
        skip_ratio=2
    )

    val_dataset = AudioMixingDataset(
        ears_df=val_ears_df,
        wham_df=val_wham_df,
        config=mixing_audio_cfg,
        mode="val",
        skip_ratio=2
    )
    
    print(f"Train samples: {train_dataset}")
    print(f"Val samples: {val_dataset}")
    print()

    
    preprocessor = AmplitudeSpectrogramProcessor(config=audio_preprocessor_cfg)
    scale_converter = AmplitudeToDBScaler(audio_preprocessor_cfg)
    augmentor = AudioAugmentor(
        audio_preprocessor_config=audio_preprocessor_cfg, 
        augumentor_config=audio_augumentor_cfg
    )
    adjuster = TrimAdjuster(audio_preprocessor_cfg)
    scaler = MinMaxFixedNormalizer(normalizer_cfg)

    pipeline = TrainPipeline(
        preprocessor=preprocessor,
        augmentor=augmentor,
        scaler=scaler,
        adjuster=adjuster,
        scale_converter=scale_converter
    )
    
    data_module = AudioDataModule(
        train_ds=train_dataset, 
        val_ds=val_dataset, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS 
    )
    
    print("Pipeline and DataModule ready.\n")
    
    generator = Bin2BinGenerator(start_filters=32)
    discriminator = Bin2BinDiscriminator(in_channels=2, filters=32)

    generator.apply(init_weights)
    discriminator.apply(init_weights)
    
    print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    if use_channels_last and torch.cuda.is_available():
        generator = generator.to(memory_format=torch.channels_last)
        discriminator = discriminator.to(memory_format=torch.channels_last)
        print("* Channels-last enabled")
    
    if use_compile and hasattr(torch, 'compile') and torch.cuda.is_available():
         try:
             generator = torch.compile(generator)
             discriminator = torch.compile(discriminator)
             print("* torch.compile successful")
         except Exception:
             print("* torch.compile skipped")

    model = Bin2Bin(
        generator=generator,
        discriminator=discriminator,
        pipeline=pipeline,
        scaler=scaler,
        lr=LEARNING_RATE,
        lambda_mag=LAMBDA_MAG,
        lambda_sc=LAMBDA_SC,
        discriminator_train_freq=1
    )
    
    print("Model ready for training.\n")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_gen",
        filename="gen-unet-{epoch:02d}-{val_loss:.4f}",
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
    progress_bar_callback = TQDMProgressBar(leave=True)

    logger = TensorBoardLogger(save_dir="tb_logs_gen", name="bin2bin_gen")
    csv_logger = CSVLogger(save_dir="csv_logs_gen", name="bin2bin_gen")
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stopping_callback, image_callback, progress_bar_callback],
        logger=[logger, csv_logger],
        accelerator="gpu",
        devices="auto",
        precision="16-mixed",
        benchmark=True,
        deterministic=False,
        log_every_n_steps=50,
    )
    
    if find_batch_size:
        print("Finding optimal batch size...")
        tuner = Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=data_module, mode='power', init_val=8)

    print("Starting training...")
    trainer.fit(model, datamodule=data_module)
