from dataclasses import dataclass
from typing import Literal


@dataclass
class MixingAudioDatasetConfig:
    sample_rate: int
    segment_sec: float
    overlap: float
    min_snr: float
    max_snr: float


@dataclass
class AudioPreprocessorConfig:
    sample_rate: int
    n_fft: int
    window_length: int
    hop_length: int
    n_mels: int
    top_db: int
    max_spec_shapes: tuple[int, int]


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


