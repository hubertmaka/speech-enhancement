from dataclasses import dataclass
from typing import Literal


@dataclass
class MixingAudioDatasetConfig:
    sample_rate: int
    segment_sec: float
    overlap: float
    min_snr: float
    max_snr: float
    skip_ratio: int


@dataclass
class AudioPreprocessorConfig:
    sample_rate: int
    n_fft: int
    window_length: int
    hop_length: int
    n_mels: int
    top_db: int
    mask_loss_threshold: float
    mask_loss_weight: float
    max_spec_shapes: tuple[int, int]
    spec_type: Literal["amplitude", "power"] = "amplitude"
    mel_scale: Literal["htk", "slaney"] = "htk"


@dataclass
class NormalizerConfig:
    min_db: float = -80.0
    max_db: float = 0.0
    scale_type: Literal["-1_1", "0_1"] = "-1_1"
    std: float = 1.0
    mean: float = 0.0


@dataclass
class AudioAugumentorConfig:
    time_mask_secs: float
    freq_mask_bins: int


@dataclass
class MelMelReGANTrainConfig:
    batch_size: int
    num_workers: int
    max_epochs: int
    learning_rate: float
    lambda_mag: float
    lambda_sc: float
    discriminator_train_freq: int
    label_smoothing: float
    warmup_epochs: int
    g_filters: int
    d_filters: int
    g_input_channels: int
    d_input_channels: int


@dataclass
class Pix2PixTrainConfig:
    batch_size: int
    num_workers: int
    max_epochs: int
    learning_rate: float
    lambda_recon: float


@dataclass
class LinearMelReGANTrainConfig:
    batch_size: int
    num_workers: int
    max_epochs: int
    learning_rate: float
    lambda_mag: float
    lambda_sc: float
    discriminator_train_freq: int
    g_filters: int
    d_filters: int
    d_input_channels: int
