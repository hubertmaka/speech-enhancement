from src.datasets import *
from src.mel_regan.models import MelReGANGenerator, MelReGANDiscriminator
from src.mel_regan.strategies import MelReGAN
from src.utils import init_weights

from src.mel_regan.utils import *


def train(train_size: int = 10_000, train_percentage: float = 0.8) -> tuple[pl.Trainer, MelReGAN, AudioDataModule]:
    """Main function to set up and start training the MelReGAN model."""
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

    generator = MelReGANGenerator(start_filters=configs["train_cfg"].g_filters)
    discriminator = MelReGANDiscriminator(
        in_channels=configs["train_cfg"].d_input_channels, 
        start_filters=configs["train_cfg"].d_filters
    )

    generator.apply(init_weights)
    discriminator.apply(init_weights)

    model = create_strategy(
        generator=generator,
        discriminator=discriminator,
        pipeline=pipeline,
        scaler=create_scaler(configs["normalizer_cfg"]),
        audio_cfg=configs["audio_preprocessor_cfg"],
        cfg=configs["train_cfg"]
    )

    callbacks = create_callbacks()
    loggers= create_loggers()

    trainer = create_trainer(
        train_size=train_size,
        train_percentage=train_percentage,
        max_epochs=configs["train_cfg"].max_epochs,
        cfg=configs["train_cfg"],
        loggers=loggers,
        callbacks=callbacks
    )

    return trainer, model, data_module, split_dfs


# trainer, model, data_module = train(train_size=10_000, train_percentage=0.8)
# trainer.fit(model, datamodule=data_module)