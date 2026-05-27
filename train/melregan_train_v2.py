# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-28T09:22:37.191754Z","iopub.execute_input":"2026-03-28T09:22:37.192403Z","iopub.status.idle":"2026-03-28T09:22:37.197595Z","shell.execute_reply.started":"2026-03-28T09:22:37.192355Z","shell.execute_reply":"2026-03-28T09:22:37.196520Z"}}
#### %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Import libraries

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-28T09:22:37.199027Z","iopub.execute_input":"2026-03-28T09:22:37.199301Z","iopub.status.idle":"2026-03-28T09:24:06.119370Z","shell.execute_reply.started":"2026-03-28T09:22:37.199270Z","shell.execute_reply":"2026-03-28T09:24:06.118428Z"}}
# !pip install "protobuf==3.20.3" speechbrain pesq pystoi torchmetrics['audio']

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2026-03-28T09:24:06.120677Z","iopub.execute_input":"2026-03-28T09:24:06.121028Z","iopub.status.idle":"2026-03-28T09:24:27.483204Z","shell.execute_reply.started":"2026-03-28T09:24:06.120988Z","shell.execute_reply":"2026-03-28T09:24:27.482168Z"}}
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Generator, Literal
import itertools
import random
import math
from dataclasses import dataclass

import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar, RichModelSummary, ModelSummary, TQDMProgressBar, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.tuner import Tuner
import IPython.display as ipd
from speechbrain.inference.vocoders import HIFIGAN

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Pretrain helpers

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Config

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2026-03-28T09:24:27.485293Z","iopub.execute_input":"2026-03-28T09:24:27.485845Z","iopub.status.idle":"2026-03-28T09:24:27.500738Z","shell.execute_reply.started":"2026-03-28T09:24:27.485813Z","shell.execute_reply":"2026-03-28T09:24:27.499729Z"}}
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

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## DataFrames utils

# %% [code] {"execution":{"iopub.status.busy":"2026-03-28T09:24:27.501831Z","iopub.execute_input":"2026-03-28T09:24:27.502292Z","iopub.status.idle":"2026-03-28T09:24:27.528529Z","shell.execute_reply.started":"2026-03-28T09:24:27.502259Z","shell.execute_reply":"2026-03-28T09:24:27.527579Z"},"jupyter":{"outputs_hidden":false}}

def create_filepaths(root_dir: str, subset: str, extension: str = ".wav") -> dict[str, list[str]]:
    """Create lists of file paths for noisy and clean audio files in the specified subset."""
    subset_path = os.path.join(root_dir, subset)
    noisy_dir = os.path.join(subset_path, "noisy")
    clean_dir = os.path.join(subset_path, "clean")
    noisy_persons = [os.path.join(noisy_dir, d) for d in os.listdir(noisy_dir) if os.path.isdir(os.path.join(noisy_dir, d))]
    clean_persons = [os.path.join(clean_dir, d) for d in os.listdir(clean_dir) if os.path.isdir(os.path.join(clean_dir, d))]
    noisy_files = []
    clean_files = []
    for person_dir in noisy_persons:
        person_noisy_files = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.endswith(extension)]
        noisy_files.extend(person_noisy_files)
    
    for person_dir in clean_persons:
        person_clean_files = [os.path.join(person_dir, f) for f in os.listdir(person_dir) if f.endswith(extension)]
        clean_files.extend(person_clean_files)
    
    return {
        "noisy": sorted(noisy_files),
        "clean": sorted(clean_files)
    }

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Pipeline utils

# %% [code] {"execution":{"iopub.status.busy":"2026-03-28T09:24:27.529759Z","iopub.execute_input":"2026-03-28T09:24:27.530112Z","iopub.status.idle":"2026-03-28T09:24:27.571222Z","shell.execute_reply.started":"2026-03-28T09:24:27.530079Z","shell.execute_reply":"2026-03-28T09:24:27.570384Z"},"jupyter":{"outputs_hidden":false}}
# ====================================================
# Abstract Classes  
# ====================================================

class Normalizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class Scaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    

class SpectrogramProcessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    

class Adjuster(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ====================================================
# Implementations  
# ====================================================

class AudioLoader(IterableDataset):
    """Dataset for loading clean and noisy audio at random SNR levels."""
    def __init__(
            self,
            clean_filepaths: list[str],
            noisy_filepaths: list[str],
            config: MixingAudioDatasetConfig,
            skip_ratio: int = 1
    ) -> None:
        self.clean_paths = clean_filepaths
        self.noisy_paths = noisy_filepaths
        self.c = config
        self.sr = skip_ratio
        
        self.segment_samples = int(self.c.segment_sec * self.c.sample_rate)
        self.overlap_samples = int(self.c.overlap * self.c.sample_rate)
        
        self.base_step = self.segment_samples - self.overlap_samples
        
    def load_audio(self, path: str) -> torch.Tensor:
        """Load audio from a given file path."""
        waveform, sr = torchaudio.load(path)
        
        if sr != self.c.sample_rate:
            raise ValueError(f"Sample rate mismatch: expected {self.c.sample_rate}, got {sr}")
        
        return waveform.squeeze(0)

    def align_probe(self, audio: torch.Tensor) -> torch.Tensor:
        """Align audio into overlapping windows of fixed size. 
        Pads if audio is too short, truncates remainder if longer."""
        L = audio.shape[0]
        W = self.segment_samples
        S = self.base_step

        if L < W:
            # Jeśli audio jest krótsze niż wymagana długość okna - dopełniamy (padding)
            pad_amount = W - L
            audio = torch.nn.functional.pad(audio, (0, pad_amount))
            return audio.unsqueeze(0)
        else:
            windows = audio.unfold(0, W, S)
            return windows

    def __iter__(self) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Iterator to yield noisy and clean audio pairs."""
        worker_info = get_worker_info()
        clean_paths = self.clean_paths[:]
        noisy_paths = self.noisy_paths[:]
        
        if worker_info is not None:
            per_worker = int(math.ceil(len(clean_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(clean_paths))
            clean_paths = clean_paths[iter_start:iter_end]
            noisy_paths = noisy_paths[iter_start:iter_end]

        for noisy_path, clean_path in zip(noisy_paths, clean_paths):
            noisy_audio = self.load_audio(noisy_path)
            clean_audio = self.load_audio(clean_path)

            noisy_windows = self.align_probe(noisy_audio)
            clean_windows = self.align_probe(clean_audio)

            num_windows = min(noisy_windows.shape[0], clean_windows.shape[0])

            for i in range(0, num_windows, self.sr):
                yield noisy_windows[i], clean_windows[i]


# ======= Spectrogram Processors =======

class AmplitudeSpectrogramProcessor(SpectrogramProcessor):
    """Compute amplitude spectrograms from audio signals."""
    def __init__(
            self,
            config: AudioPreprocessorConfig
    ) -> None:
        super().__init__()
        self.c = config
        self.spectrogram = T.Spectrogram(
            n_fft=self.c.n_fft,
            win_length=self.c.window_length,
            hop_length=self.c.hop_length
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute amplitude spectrogram and add channel dimension."""
        x = self.spectrogram(x)
        return x.unsqueeze(1)


class MelSpectrogramProcessor(SpectrogramProcessor):
    """Compute mel spectrograms from audio signals."""
    def __init__(
            self,
            config: AudioPreprocessorConfig
    ) -> None:
        super().__init__()
        self.c = config
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=self.c.sample_rate,
            n_fft=self.c.n_fft,
            win_length=self.c.window_length,
            hop_length=self.c.hop_length,
            n_mels=self.c.n_mels,
            f_min=0,
            f_max=self.c.sample_rate // 2,
            power=1.0 if self.c.spec_type == "amplitude" else 2.0,
            mel_scale=self.c.mel_scale,
            norm=self.c.mel_scale,
            center=False,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram and add channel dimension."""
        x = self.mel_spectrogram(x)
        return x.unsqueeze(1)


# ======= Scalers =======

class AudioToDBScaler(Scaler):
    """Convert amplitude or power spectrograms to decibel scale."""
    def __init__(self, audio_preprocessor_config: AudioPreprocessorConfig) -> None:
        super().__init__()
        self.top_db = audio_preprocessor_config.top_db
        self.stype = audio_preprocessor_config.spec_type
        self.amplitude_to_db = T.AmplitudeToDB(top_db=self.top_db, stype=self.stype)
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Convert amplitude or power spectrogram to decibel scale."""
        return self.amplitude_to_db(spec)
    

class DBToLogScaler(Scaler):
    """
    Convert decibel spectrograms to the scale used by HiFi-GAN.
    HiFi-GAN uses a log10 scale with a specific top_db, so we need to convert from the standard amplitude-to-dB scale to the HiFi-GAN scale.

    The conversion is as follows:
    1. Convert from dB to linear scale using the formula: linear = 10^(dB / divider), where divider is 10 for amplitude spectrograms and 20 for power spectrograms.
    2. Apply natural logarithm and clamping as per HiFi-GAN implementation.
    """
    def __init__(self, audio_preprocessor_config: AudioPreprocessorConfig) -> None:
        super().__init__()
        self.cfg = audio_preprocessor_config
        self.factor = self._determine_factor(audio_preprocessor_config)

    def _determine_factor(self, cfg: AudioPreprocessorConfig) -> float:
        """Determine the divider factor based on the spectrogram type."""
        return 20.0 if cfg.spec_type == "amplitude" else 10.0

    def forward(self, spec_db: torch.Tensor) -> torch.Tensor:
        """Convert decibel spectrogram to log scale used by HiFi-GAN."""
        spec_linear = torch.pow(10.0, spec_db / self.factor)
        spec_log = torch.log(torch.clamp(spec_linear, min=1e-5))
        return spec_log
    

class AmplitudeToLog1pScaler(Scaler):
    """Convert amplitude spectrograms to log1p scale."""
    def __init__(self) -> None:
        super().__init__()

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Convert amplitude spectrogram to log1p scale."""
        spec = torch.log1p(spec)
        return spec


# ======= Normalizers =======

class StandardNormalizer(Normalizer):
    """Normalize spectrograms using standard normalization."""
    def __init__(self, config: NormalizerConfig) -> None:
        super().__init__()
        self.mean = config.mean
        self.std = config.std

    def normalize_standard(self, spec: torch.Tensor) -> torch.Tensor:
        """Normalize spectrogram using standard normalization."""
        return (spec - self.mean) / (self.std + 1e-12)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Normalize spectrogram using standard normalization."""
        return self.normalize_standard(spec)


class MinMaxFixedNormalizer(Normalizer):
    """Normalize spectrograms using fixed min-max scaling."""
    def __init__(self, config: NormalizerConfig) -> None:
        super().__init__()
        self.min_db = config.min_db
        self.max_db = config.max_db
        self.db_range = config.max_db - config.min_db
        self.scale_type = config.scale_type

    def forward(self, spec_db: torch.Tensor) -> torch.Tensor:
        """Normalize spectrogram from decibel scale to normalized scale."""
        spec_db = torch.clamp(spec_db, min=self.min_db, max=self.max_db)
        spec_norm = (spec_db - self.min_db) / (self.db_range + 1e-12)
        if self.scale_type == "0_1":
            return spec_norm
        else:
            return spec_norm * 2.0 - 1.0

    def denormalize(self, spec_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize spectrogram from normalized scale back to decibel scale."""
        if self.scale_type == "0_1":
            spec_0_1 = spec_norm
        else:
            spec_0_1 = (spec_norm + 1.0) / 2.0
        spec_db = spec_0_1 * self.db_range + self.min_db
        return spec_db


# ======= Augumentors =======

class AudioAugmentor(nn.Module):
    """Applies time and frequency masking to spectrograms."""
    def __init__(
            self,
            augumentor_config: AudioAugumentorConfig,
            audio_preprocessor_config: AudioPreprocessorConfig
    ) -> None:
        super().__init__()
        self.a_c = augumentor_config
        self.ap_c = audio_preprocessor_config
        self.time_mask_param = int(self.a_c.time_mask_secs * self.ap_c.sample_rate / self.ap_c.hop_length)
        self.freq_mask_param = self.a_c.freq_mask_bins
        self.time_mask = T.TimeMasking(time_mask_param=self.time_mask_param) if self.time_mask_param != 0 else None
        self.freq_mask = T.FrequencyMasking(freq_mask_param=self.freq_mask_param) if self.freq_mask_param is not None else None

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply time and frequency masking to the spectrogram."""
        if self.time_mask:
            spec = self.time_mask(spec)  
        if self.freq_mask:
            spec = self.freq_mask(spec) 
        return spec
    

# ====== Adjusters =======

class TrimAdjuster(Adjuster):
    """Adjuster that trims spectrograms to a target shape."""
    def __init__(self, audio_preprocessor_config: AudioPreprocessorConfig) -> None:
        super().__init__()
        self.target_shape = audio_preprocessor_config.max_spec_shapes

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Trim spectrogram to target shape."""
        spec = spec[:, :, :self.target_shape[0], :self.target_shape[1]]
        return spec


# ======= Pipelines =======

class DataPipeline(nn.Module):
    """Pipeline to process audio data through steps:
    - Preprocessing (e.g., spectrogram computation)
    - Augmentation (e.g., time/frequency masking)
    - Scaling (e.g., converting to dB or log scale)
    - Normalization (e.g., standard or min-max normalization)
    - Adjustment (e.g., trimming to target shape)
    """
    def __init__(
            self,
            preprocessor: SpectrogramProcessor = None,
            augmentor: AudioAugmentor = None,
            scale_converter: Scaler | None = None,
            scaler: Normalizer | None = None,
            adjuster: Adjuster | None = None
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.augmentor = augmentor
        self.scale_converter = scale_converter
        self.scaler = scaler
        self.adjuster = adjuster

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the training pipeline. Returns pair of input and target data"""
        if self.preprocessor:
            x = self.preprocessor(x)
            y = self.preprocessor(y)
        
        if self.augmentor:
            x = self.augmentor(x)

        if self.scale_converter:
            x = self.scale_converter(x)
            y = self.scale_converter(y)
        
        if self.scaler:
            x = self.scaler(x)        
            y = self.scaler(y)

        if self.adjuster:
            x = self.adjuster(x)
            y = self.adjuster(y)

        return x, y


# ======= DataModule =======

class AudioDataModule(pl.LightningDataModule):
    """DataModule for audio mixing dataset."""
    def __init__(
            self, 
            train_ds: AudioLoader,
            val_ds: AudioLoader,
            batch_size: int = 32, 
            num_workers: int = 4
        ) -> None:
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        """Training dataloader."""
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=16 if self.num_workers > 0 else None,
            drop_last=True,
            shuffle=False
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=max(4, self.num_workers),
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=16 if self.num_workers > 0 else None,
            drop_last=False,
            shuffle=False
        )

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Callbacks

# %% [code] {"jupyter":{"source_hidden":true},"execution":{"iopub.status.busy":"2026-03-28T09:24:27.572391Z","iopub.execute_input":"2026-03-28T09:24:27.572759Z","iopub.status.idle":"2026-03-28T09:24:27.597736Z","shell.execute_reply.started":"2026-03-28T09:24:27.572730Z","shell.execute_reply":"2026-03-28T09:24:27.596692Z"}}
class SpectrogramLogger(pl.Callback):
    """Callback to log spectrograms during validation."""
    def __init__(self, save_dir: str = "saved_spectrograms", num_samples: int = 30):
        super().__init__()
        self.save_dir = self._versionize_dir(save_dir)
        self.num_samples = num_samples
        os.makedirs(self.save_dir, exist_ok=True)

    def _versionize_dir(self, base_dir: str) -> str:
        """Create a versioned directory to avoid overwriting previous logs."""
        version = 0
        versioned_dir = os.path.join(base_dir, f"version_{version}")
        while os.path.exists(versioned_dir):
            version += 1
            versioned_dir = os.path.join(base_dir, f"version_{version}")
        return versioned_dir

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Log spectrograms at the end of the validation batch."""
        if batch_idx != 0 or outputs is None:
            return

        try:
            noisy_tensor = outputs["lossy_spec"]
            clean_tensor = outputs["clean_spec"]
            fake_tensor = outputs["fake_spec"]
        except KeyError:
            noisy_tensor = outputs.get("noisy")
            clean_tensor = outputs.get("clean")
            fake_tensor = outputs.get("fake")

        if noisy_tensor is None:
            print("SpectrogramLogger: Missing spectrograms in outputs, skipping logging.")
            return

        batch_size = noisy_tensor.shape[0]
        n = min(batch_size, self.num_samples)
        
        noisy_batch = noisy_tensor[:n].detach().cpu().numpy().squeeze(1)
        fake_batch = fake_tensor[:n].detach().cpu().numpy().squeeze(1)
        clean_batch = clean_tensor[:n].detach().cpu().numpy().squeeze(1)

        fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n), squeeze=False)
        
        plot_kwargs = {'origin': 'lower', 'aspect': 'auto', 'cmap': 'magma', 'vmin': -1, 'vmax': 1, 'interpolation': 'nearest'}

        for i in range(n):
            im1 = axes[i, 0].imshow(noisy_batch[i], **plot_kwargs)
            axes[i, 0].set_ylabel(f"Sample {i}\nFreq (bins)", fontsize=10, fontweight='bold')
            
            im2 = axes[i, 1].imshow(fake_batch[i], **plot_kwargs)
            
            im3 = axes[i, 2].imshow(clean_batch[i], **plot_kwargs)

            cbar = fig.colorbar(im3, ax=axes[i, 2], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

            for col_idx, ax in enumerate(axes[i]):
                ax.tick_params(axis='both', which='major', labelsize=8)
                
                if i == n - 1:
                    ax.set_xlabel("Time (frames)", fontsize=10)
                else:
                    ax.set_xticklabels([])
                if col_idx > 0:
                    ax.set_yticklabels([])
            if i == 0:
                axes[i, 0].set_title("Input (Lossy/Noisy)", fontsize=14)
                axes[i, 1].set_title("Generated (Restored)", fontsize=14)
                axes[i, 2].set_title("Target (Clean)", fontsize=14)

        fig.suptitle(f'Validation Epoch {trainer.current_epoch}', fontsize=16, y=1.005)
        plt.tight_layout()

        if hasattr(trainer.logger, 'experiment'):
            try:
                trainer.logger.experiment.add_figure(f"Validation/Spectrograms_Batch", fig, global_step=trainer.global_step)
            except AttributeError:
                pass

        filename = f"epoch_{trainer.current_epoch:03d}_grid.png"
        save_path = os.path.join(self.save_dir, filename)
        fig.savefig(save_path, bbox_inches='tight')
        
        plt.close(fig)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Utils

# %% [code] {"execution":{"iopub.status.busy":"2026-03-28T09:24:27.598914Z","iopub.execute_input":"2026-03-28T09:24:27.599279Z","iopub.status.idle":"2026-03-28T09:24:27.620529Z","shell.execute_reply.started":"2026-03-28T09:24:27.599248Z","shell.execute_reply":"2026-03-28T09:24:27.619572Z"}}
def init_weights(m: nn.Module) -> None:
    """Initialize weights of convolutional and normalization layers."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Mel Regenerative Generative Adversarial Network (MelReGAN)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Layers

# %% [code] {"execution":{"iopub.status.busy":"2026-03-28T09:24:27.621726Z","iopub.execute_input":"2026-03-28T09:24:27.622071Z","iopub.status.idle":"2026-03-28T09:24:27.658062Z","shell.execute_reply.started":"2026-03-28T09:24:27.622045Z","shell.execute_reply":"2026-03-28T09:24:27.657192Z"}}
class ConvEncoderBlock(nn.Module):
    """ Convolutional Encoder block: convolutional -> instance norm -> leakyrelu""" 
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int = 4, 
            stride: int = 2, 
            use_norm: bool = True,
            use_activation: bool = True
        ) -> None:
        super().__init__()
        padding = (kernel_size - stride) // 2

        self.padding = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=False if use_norm else True)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder block."""
        x = self.padding(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.lrelu(x)
        return x
    

class ConvDecoderBlock(nn.Module):
    """Convolutional Decoder block: upsample -> conv -> instance norm -> leakyrelu -> (optional dropout)"""
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int = 3, 
            dropout: float = 0.0,
            use_norm: bool = True,
            use_activation: bool = True
        ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.padding = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False if use_norm else True)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder block."""
        x = self.upsample(x)
        x = self.padding(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.lrelu(x)
        x = self.dropout(x)
        return x


class SpectralAttention(nn.Module):
    """Uwaga widmowa (częstotliwościowo-czasowa) dla wąskiego gardła."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, max(1, channels // reduction), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.time_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(channels, max(1, channels // reduction), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq_att = self.freq_attention(x)
        time_att = self.time_attention(x)
        return x * (self.alpha * freq_att) * (self.beta * time_att)

class AttentionGate(nn.Module):
    """
    Klasyczna bramka uwagi (Attention Gate) z Twojego poprzedniego rozwiązania.
    Filtruje cechy ze złącza omijającego (x) bazując na sygnale z dekodera (g).
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int, affine=True)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int, affine=True)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1, affine=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi



class DilatedResBlock(nn.Module):
    """Dilated Residual Block with dilation for larger receptive field.
    Schema: ReflectionPad -> Conv(dilated) -> InstanceNorm -> LeakyReLU -> Conv(1x1) -> InstanceNorm -> Add & Activation
    """
    def __init__(
            self, 
            channels: int,
            kernel_size: int = 3,
            dilation: int = 2,
            ) -> None:
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(channels, channels, kernel_size, dilation=dilation, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels, affine=True),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dilated residual block."""
        return self.activation(x + self.residual_block(x))


class BasicResBlock(nn.Module):
    """Czysty blok rezydualny, który nie zmienia liczby kanałów ani rozdzielczości."""
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1, use_norm: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, bias=False),
            nn.InstanceNorm2d(channels, affine=True) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ReflectionPad2d(padding),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, bias=False),
            nn.InstanceNorm2d(channels, affine=True) if use_norm else nn.Identity(),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ResidualEncoderBlock(nn.Module):
    """Enkoder: Transformacja kanałów -> PODWÓJNY blok rezydualny -> downsampling"""
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int,
            kernel_size: int = 3,
            padding: int = 1,
            stride: int = 2,
            use_norm: bool = True
        ) -> None:
        super().__init__()
        
        # 1. Zmiana liczby kanałów (jeśli inna)
        self.channel_adj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        ) if in_channels != out_channels else nn.Identity()

        # 2. PODWOJONE bloki rezydualne na tej samej rozdzielczości (Twój pomysł!)
        self.res_blocks = nn.Sequential(
            BasicResBlock(out_channels, kernel_size, padding, use_norm),
            BasicResBlock(out_channels, kernel_size, padding, use_norm) # <-- Drugi blok!
        )
        
        # 3. Downsampling przekazujący sygnał głębiej
        self.downsample = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.channel_adj(x)
        skip_out = self.res_blocks(x)        # To poleci do AttentionGate (Skip Connection)
        next_layer_in = self.downsample(skip_out) # To leci w dół U-Netu
        return skip_out, next_layer_in


class ResidualDecoderBlock(nn.Module):
    """Dekoder: Upsampling -> Redukcja kanałów -> PODWÓJNY blok rezydualny"""
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int = 3,
            padding: int = 1,
            dropout: float = 0.0,
            use_norm: bool = True
        ) -> None:
        super().__init__()
        
        # 1. Upsampling i redukcja kanałów
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 2. PODWOJONE bloki rezydualne odzyskujące jakość po upsamplingu
        self.res_blocks = nn.Sequential(
            BasicResBlock(out_channels, kernel_size, padding, use_norm),
            BasicResBlock(out_channels, kernel_size, padding, use_norm) # <-- Drugi blok!
        )
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.reduce(x)
        x = self.res_blocks(x)
        x = self.dropout(x)
        return x

class PatchBlock(nn.Module):
    """PatchGAN Block bez interpolacji (stride=2 dla downsamplingu)."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size=4, stride=2, padding=1, use_norm=True):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_norm))
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Models

# %% [code] {"execution":{"iopub.status.busy":"2026-03-28T09:24:27.660672Z","iopub.execute_input":"2026-03-28T09:24:27.660953Z","iopub.status.idle":"2026-03-28T09:24:27.693668Z","shell.execute_reply.started":"2026-03-28T09:24:27.660930Z","shell.execute_reply":"2026-03-28T09:24:27.692371Z"}}
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

# ==========================================
# 1. BLOKI BAZOWE I UWAGA (ATTENTION)
# ==========================================

class BigGANResBlock(nn.Module):
    """
    Blok Resztkowy BigGAN z InstanceNorm.
    Wykorzystuje klasyczną interpolację ('nearest') + Resize-Conv dla antyaliasingu.
    Obsługuje konwolucje dylatacyjne (dilation) wymagane w wąskim gardle.
    """
    def __init__(self, in_channels: int, out_channels: int, resample: str = None, dilation: int = 1):
        super().__init__()
        self.resample = resample 
        
        # --- Gałąź główna ---
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.act1 = nn.SiLU(inplace=True) 
        
        # Dylatacja wymaga odpowiedniego paddingu, aby zachować rozmiar. ReflectionPad chroni krawędzie.
        self.pad1 = nn.ReflectionPad2d(dilation)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, dilation=dilation, bias=False)
        
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act2 = nn.SiLU(inplace=True)
        
        self.pad2 = nn.ReflectionPad2d(dilation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, dilation=dilation, bias=False)

        # --- Gałąź omijająca (Skip Connection wewnątrz bloku) ---
        self.skip_conv = nn.Identity()
        if in_channels != out_channels or resample is not None:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def _apply_resample(self, x: torch.Tensor) -> torch.Tensor:
        if self.resample == 'up':
            # Klasyczne skalowanie najbliższego sąsiada (wygładzane potem przez konwolucję)
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        elif self.resample == 'down':
            # Uśrednianie to najbezpieczniejsza klasyczna metoda downsamplingu (antyaliasing)
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Główna gałąź ---
        h = self.norm1(x)
        h = self.act1(h)
        h = self._apply_resample(h)
            
        h = self.pad1(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.act2(h)
        
        h = self.pad2(h)
        h = self.conv2(h)

        # --- Gałąź omijająca ---
        s = self._apply_resample(x)
        s = self.skip_conv(s)

        out = (h + s) / math.sqrt(2.0)
        return out





# ==========================================
# 2. GENERATOR (SGMSE+ / NCSN++ U-Net)
# ==========================================


# Zostawiam Twoją klasę BigGANResBlock bez zmian, 
# ponieważ jest napisana poprawnie i wspiera wszystko, czego potrzebujemy.

# ==========================================
# 2. GENERATOR (SGMSE+ / NCSN++ U-Net)
# ==========================================

class MelReGANGenerator(nn.Module):
    def __init__(self, start_filters: int = 128) -> None:
        super().__init__()
        nf = start_filters
        
        # Wejście: 1 kanał -> 128 kanałów (80x80)
        self.conv_in = nn.Conv2d(1, nf, kernel_size=3, padding=1)

        # --- ENKODER (Dokładnie 1 blok na poziom z wbudowanym downsamplingiem) ---
        # Zgodnie ze schematem NCSN++ kanały idą: 128 -> 128 -> 256 -> 256 -> 256 (Bottleneck)
        
        # Poziom 1: 80x80 -> 40x40 (zostajemy przy nf kanałów)
        self.down1 = BigGANResBlock(nf, nf, 'down')
        
        # Poziom 2: 40x40 -> 20x20 (przeskok kanałów z nf -> nf*2, zgodnie z obrazkiem 128->256)
        self.down2 = BigGANResBlock(nf, nf*2, 'down')
        
        # Poziom 3: 20x20 -> 10x10 (utrzymujemy nf*2)
        self.down3 = BigGANResBlock(nf*2, nf*2, 'down')

        # Poziom 4: 10x10 -> 5x5 (utrzymujemy nf*2)
        self.down4 = BigGANResBlock(nf*2, nf*2, 'down')

        # --- WĄSKIE GARDŁO (5x5) ---
        self.bottleneck = nn.Sequential(
            BigGANResBlock(nf*2, nf*2, dilation=2), 
            BigGANResBlock(nf*2, nf*2, dilation=4),      
            # SpectralAttention(nf*2),                     # Zakładam, że masz tę klasę zdefiniowaną
            # BigGANResBlock(nf*2, nf*2, dilation=1)     # Opcjonalny powrót z dylatacji
        )

        # --- DEKODER Z BRAMKAMI UWAGI (Dokładnie 1 blok na poziom) ---
        # Aby nie powielać bloków, robimy najpierw klasyczny upsampling, potem concat, potem 1 blok przetwarzający.
        
        # Poziom 4: 5x5 -> 10x10
        self.att4 = AttentionGate(F_g=nf*2, F_l=nf*2, F_int=nf)
        # Wejście: g (nf*2) + skip4 (nf*2) = nf*4
        self.dec4 = BigGANResBlock(nf*4, nf*2) 

        # Poziom 3: 10x10 -> 20x20
        self.att3 = AttentionGate(F_g=nf*2, F_l=nf*2, F_int=nf)
        # Wejście: g (nf*2) + skip3 (nf*2) = nf*4
        self.dec3 = BigGANResBlock(nf*4, nf*2)

        # Poziom 2: 20x20 -> 40x40
        self.att2 = AttentionGate(F_g=nf*2, F_l=nf, F_int=nf)
        # Wejście: g (nf*2) + skip2 (nf) = nf*3
        self.dec2 = BigGANResBlock(nf*3, nf) 

        # Poziom 1: 40x40 -> 80x80
        self.att1 = AttentionGate(F_g=nf, F_l=nf, F_int=nf//2)
        # Wejście: g (nf) + skip1 (nf) = nf*2
        self.dec1 = BigGANResBlock(nf*2, nf) 

        # --- WYJŚCIE ---
        self.final = nn.Sequential(
            nn.InstanceNorm2d(nf, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(nf, 1, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        
        # --- ENKODER ---
        skip1 = x                           # Rozmiar: 80x80, Kanały: nf
        x = self.down1(x)                   
        
        skip2 = x                           # Rozmiar: 40x40, Kanały: nf
        x = self.down2(x)                   
        
        skip3 = x                           # Rozmiar: 20x20, Kanały: nf*2
        x = self.down3(x)                   

        skip4 = x                           # Rozmiar: 10x10, Kanały: nf*2
        x = self.down4(x)                   
        
        # --- WĄSKIE GARDŁO ---
        x = self.bottleneck(x)              # Rozmiar: 5x5, Kanały: nf*2
        
        # --- DEKODER ---
        # Poziom 4
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        s4 = self.att4(g=x, x=skip4)
        x = torch.cat([x, s4], dim=1)
        x = self.dec4(x)
        
        # Poziom 3
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        s3 = self.att3(g=x, x=skip3)
        x = torch.cat([x, s3], dim=1)
        x = self.dec3(x)
        
        # Poziom 2
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        s2 = self.att2(g=x, x=skip2)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x)
        
        # Poziom 1
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        s1 = self.att1(g=x, x=skip1)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1(x)
        
        return self.final(x)

# ==========================================
# 3. DYSKRYMINATOR (SGMSE+ Scale PatchGAN)
# ==========================================


class DiscResBlock(nn.Module):
    """
    Blok Resztkowy dla Dyskryminatora.
    Wykorzystuje SpectralNorm, InstanceNorm oraz LeakyReLU.
    Downsampling realizowany przez uśrednianie (AvgPool) zapobiega aliasingowi.
    """
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super().__init__()
        self.downsample = downsample

        # --- Główna gałąź ---
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))

        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))

        # --- Gałąź omijająca ---
        self.skip_conv = nn.Identity()
        if in_channels != out_channels or downsample:
            # 1x1 conv do zrównania liczby kanałów ze SpectralNorm
            self.skip_conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Główna gałąź ---
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        
        if self.downsample:
            h = F.avg_pool2d(h, kernel_size=2, stride=2)
            
        # --- Gałąź omijająca ---
        s = self.skip_conv(x)
        if self.downsample:
            s = F.avg_pool2d(s, kernel_size=2, stride=2)
            
        # Zsumowanie i skalowanie 1/sqrt(2) dla stabilizacji
        return (h + s) / math.sqrt(2.0)


class MelReGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 2, start_filters: int = 64) -> None:
        super().__init__()
        nf = start_filters

        self.conv_in = spectral_norm(nn.Conv2d(in_channels, nf, kernel_size=3, padding=1))
        
        # Redukcja 80x80 -> 40x40
        self.block1 = DiscResBlock(nf, nf * 2, downsample=True)
        
        # Redukcja 40x40 -> 20x20
        self.block2 = DiscResBlock(nf * 2, nf * 4, downsample=True)
        self.drop_2 = nn.Dropout2d(0.3) # Możesz podbić np. do 0.2
        
        # Redukcja 20x20 -> 10x10
        self.block3 = DiscResBlock(nf * 4, nf * 8, downsample=True)
        self.drop_3 = nn.Dropout2d(0.3) # Możesz podbić np. do 0.2

        # USUNIĘTO block4 i drop_4. Dyskryminator zatrzymuje się na 10x10.

        # Finałowa warstwa klasyfikująca łatki - teraz wejście to nf * 8 (z block3)
        # Wyjście: [B, 1, 10, 10]
        self.final = spectral_norm(nn.Conv2d(nf * 8, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        d_in = torch.cat([x, condition], dim=1)

        d0 = self.conv_in(d_in)
        d1 = self.block1(d0)
        d2 = self.block2(d1)
        d2_drop = self.drop_2(d2)
        d3 = self.block3(d2_drop)
        d3_drop = self.drop_3(d3)
        
        output = self.final(d3_drop)

        # Zwracamy cechy dla ewentualnego FM Loss (tylko 3 bloki)
        return output, [d1, d2, d3]

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Losses

# %% [code] {"execution":{"iopub.status.busy":"2026-03-28T09:24:27.694737Z","iopub.execute_input":"2026-03-28T09:24:27.695048Z","iopub.status.idle":"2026-03-28T09:24:27.723735Z","shell.execute_reply.started":"2026-03-28T09:24:27.695023Z","shell.execute_reply":"2026-03-28T09:24:27.722829Z"},"jupyter":{"outputs_hidden":false}}
class MaskedSpectralLoss(nn.Module):
    """Computes Masked Magnitude Loss and Spectral Convergence Loss."""
    def __init__(self, scaler: nn.Module, cfg: AudioPreprocessorConfig) -> None:
        super().__init__()
        self.scaler = scaler
        self.cfg = cfg
        self.factor = 20.0 if self.cfg.spec_type == "amplitude" else 10.0
        self.threshold = self.cfg.mask_loss_threshold
        self.mask_weight = self.cfg.mask_loss_weight
        self.log_conversion = torch.log(torch.tensor(10.0)) / self.factor


    def forward(self, pred_norm: torch.Tensor, target_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the Masked Magnitude Loss and Spectral Convergence Loss."""
        threshold = torch.max(target_norm.reshape(target_norm.size(0), -1), dim=1)[0] * self.threshold
        threshold = threshold.reshape(-1, 1, 1, 1)
        
        mask = torch.ones_like(target_norm)
        mask[target_norm < threshold] = self.mask_weight
        
        l1_diff = torch.abs(pred_norm - target_norm)
        l_mag = torch.mean(l1_diff * mask) 

        pred_db = self.scaler.denormalize(pred_norm)
        target_db = self.scaler.denormalize(target_norm)
        
        pred_lin = torch.exp(pred_db * self.log_conversion)
        target_lin = torch.exp(target_db * self.log_conversion)
        
        diff = target_lin - pred_lin
        numerator = torch.sqrt(torch.sum(diff * diff) + 1e-12)
        denominator = torch.sqrt(torch.sum(target_lin * target_lin) + 1e-12)
        l_sc = numerator / denominator
        
        return l_mag, l_sc

import torch
import torch.nn as nn
import torch.nn.functional as F

class RobustSpectrogramLoss(nn.Module):
    def __init__(self, scaler: nn.Module, cfg) -> None:
        super().__init__()
        self.scaler = scaler
        self.factor = 20.0 if cfg.spec_type == "amplitude" else 10.0
        self.log_conversion = torch.log(torch.tensor(10.0)) / self.factor

    def forward(self, pred_norm: torch.Tensor, target_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        l_mag = F.l1_loss(pred_norm, target_norm)

        # Zabezpieczenie przed ekstremalnymi wartościami wygenerowanymi przez sieć
        pred_norm_safe = torch.clamp(pred_norm, -1.0, 1.0)

        pred_db = self.scaler.denormalize(pred_norm_safe)
        target_db = self.scaler.denormalize(target_norm)

        pred_lin = torch.exp(pred_db * self.log_conversion)
        target_lin = torch.exp(target_db * self.log_conversion)

        diff = target_lin - pred_lin
        numerator = torch.norm(diff, p="fro", dim=(2, 3))
        denominator = torch.norm(target_lin, p="fro", dim=(2, 3))

        l_sc = torch.mean(numerator / (denominator + 1e-7))

        return l_mag, l_sc

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Strategies

# %% [code] {"execution":{"iopub.status.busy":"2026-03-28T09:24:27.724659Z","iopub.execute_input":"2026-03-28T09:24:27.724994Z","iopub.status.idle":"2026-03-28T09:24:27.749790Z","shell.execute_reply.started":"2026-03-28T09:24:27.724967Z","shell.execute_reply":"2026-03-28T09:24:27.748856Z"}}
class MelReGAN(pl.LightningModule):
    def __init__(
            self,
            generator: MelReGANGenerator,
            discriminator: MelReGANDiscriminator,
            pipeline: nn.Module,
            scaler: nn.Module,
            audio_cfg: AudioPreprocessorConfig,
            lr: float = 0.0002, 
            lambda_mag: float = 30.0,
            lambda_sc: float = 15.0,  
            lambda_fm: float = 5.0,  
            lambda_gan: float = 1.0,   # Zmniejszona waga GAN Loss
            discriminator_train_freq: int = 4, # Zwiększony odstęp (rzadszy Dyskryminator)
            label_smoothing: float = 0.9,
            warmup_epochs: int = 5
        ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator", "pipeline", "scaler"])

        self.generator = generator
        self.discriminator = discriminator
        self.pipeline = pipeline
        self.scaler = scaler
        self.audio_cfg = audio_cfg
        self.ls = label_smoothing

        self.spectral_losses = RobustSpectrogramLoss(scaler=scaler, cfg=audio_cfg)
        self.adversarial_loss = nn.MSELoss()

        self.automatic_optimization = False
        self.gradient_clip_val = 5.0
        self.example_input_array = torch.randn(32, 1, 80, 128)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fake_spec = self.generator(x)
        disc_out, _ = self.discriminator(fake_spec, x)
        return fake_spec, disc_out

    def configure_optimizers(self):
        lr_g = 0.0002
        lr_d = 0.0002 * 0.5

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=self.trainer.max_epochs, eta_min=1e-6)
        scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=self.trainer.max_epochs, eta_min=1e-6)

        return [opt_g, opt_d], [scheduler_g, scheduler_d]

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        g_opt, d_opt = self.optimizers()
        mixed_audio, clean_audio = batch

        real_lossy_spec, real_clean_spec = self.pipeline(mixed_audio, clean_audio)

        is_warmup = self.current_epoch < self.hparams.warmup_epochs
        train_d = (batch_idx % self.hparams.discriminator_train_freq == 0) and (not is_warmup)

        loss_d_total = None

        # DODANO: Lekki szum na wejście Dyskryminatora utrudnia mu zapamiętywanie
        # noise_std = max(0.01, 0.05 - (self.current_epoch / self.trainer.max_epochs) * 0.05)
        # noisy_real_clean_spec = real_clean_spec + torch.randn_like(real_clean_spec) * noise_std
        # noisy_real_lossy_spec = real_lossy_spec + torch.randn_like(real_lossy_spec) * noise_std

        # ----------------------------------------------------------------------
        # TRAIN DISCRIMINATOR
        # ----------------------------------------------------------------------
        if train_d:
            with torch.no_grad():
                fake_clean_spec_detached = self.generator(real_lossy_spec)
                # noisy_fake_clean_spec = fake_clean_spec_detached + torch.randn_like(fake_clean_spec_detached) * noise_std

            pred_real, _ = self.discriminator(real_clean_spec, real_lossy_spec) # lub noisy
            loss_d_real = self.adversarial_loss(pred_real, torch.ones_like(pred_real) * self.ls)

            pred_fake_d, _ = self.discriminator(fake_clean_spec_detached, real_lossy_spec) # lub noisy
            loss_d_fake = self.adversarial_loss(pred_fake_d, torch.zeros_like(pred_fake_d))

            loss_d_total = (loss_d_real + loss_d_fake) * 0.5

            d_opt.zero_grad()
            self.manual_backward(loss_d_total)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.gradient_clip_val)
            d_opt.step()

        # ----------------------------------------------------------------------
        # TRAIN GENERATOR
        # ----------------------------------------------------------------------
        fake_clean_spec = self.generator(real_lossy_spec)

        if is_warmup:
            loss_gan = torch.tensor(0.0, device=self.device)
            loss_fm = torch.tensor(0.0, device=self.device)
        else:
            pred_fake_g, fake_features = self.discriminator(fake_clean_spec, real_lossy_spec)
            loss_gan = self.adversarial_loss(pred_fake_g, torch.ones_like(pred_fake_g))

            with torch.no_grad():
                _, real_features = self.discriminator(real_clean_spec, real_lossy_spec)

            # ZMODYFIKOWANO: Znormalizowany Feature Matching
            loss_fm = 0.0
            for feat_fake, feat_real in zip(fake_features, real_features):
                real_detach = feat_real.detach()
                norm_factor = torch.mean(torch.abs(real_detach)) + 1e-6
                loss_fm += F.l1_loss(feat_fake, real_detach) / norm_factor

        loss_mag, loss_sc = self.spectral_losses(fake_clean_spec, real_clean_spec)

        # ZMODYFIKOWANO: Użyto nowej (znacznie mniejszej) wagi dla GAN Loss
        total_gen_loss = (self.hparams.lambda_gan * loss_gan) + \
                         (self.hparams.lambda_fm * loss_fm) + \
                         (self.hparams.lambda_mag * loss_mag) + \
                         (self.hparams.lambda_sc * loss_sc)

        g_opt.zero_grad()
        self.manual_backward(total_gen_loss)
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.gradient_clip_val)
        g_opt.step()

        # LOG METRICS
        log_metrics = {
            "g_total": total_gen_loss.detach(),
            "g_gan": loss_gan.detach(),
            "g_fm": loss_fm.detach(),
            "g_mag": loss_mag.detach(),
            "g_sc": loss_sc.detach(),
        }

        if loss_d_total is not None:
            log_metrics["d_loss"] = loss_d_total.detach()

        self.log_dict(log_metrics, prog_bar=True, on_step=False, on_epoch=True)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        mixed_audio, clean_audio = batch
        real_lossy_spec, real_clean_spec = self.pipeline(mixed_audio, clean_audio)
        fake_clean_spec = self.generator(real_lossy_spec)

        loss_mag, loss_sc = self.spectral_losses(fake_clean_spec, real_clean_spec)
        # print(f"loss_mag: {loss_mag} : loss_sc: {loss_sc}")

        val_loss = (self.hparams.lambda_mag * loss_mag) + (self.hparams.lambda_sc * loss_sc)

        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return {
            "lossy_spec": real_lossy_spec,
            "clean_spec": real_clean_spec,
            "fake_spec": fake_clean_spec,
            "loss": val_loss
        }

    def on_validation_epoch_end(self) -> None:
       schedulers = self.lr_schedulers()
       if schedulers:
           if isinstance(schedulers, list):
               for scheduler in schedulers:
                   scheduler.step() # USUNIĘTO val_loss
           else:
               schedulers.step()

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Train utils

# %% [code] {"execution":{"iopub.status.busy":"2026-03-28T09:24:27.750625Z","iopub.execute_input":"2026-03-28T09:24:27.750911Z","iopub.status.idle":"2026-03-28T09:24:27.776198Z","shell.execute_reply.started":"2026-03-28T09:24:27.750891Z","shell.execute_reply":"2026-03-28T09:24:27.775329Z"},"jupyter":{"outputs_hidden":false}}
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



def train(train_size: int = 10_000, train_percentage: float = 0.8) -> tuple[pl.Trainer, MelReGAN, AudioDataModule]:
    """Main function to set up and start training the MelReGAN model."""
    configs = create_configs()

    train_filepaths = create_filepaths("/kaggle/input/datasets/hubertmka/ears-wham/train", subset="train")
    val_filepaths = create_filepaths("/kaggle/input/datasets/hubertmka/ears-wham/valid", subset="valid")
    
    train_dataset = create_dataset(
        clean_filepaths=train_filepaths["clean"],
        noisy_filepaths=train_filepaths["noisy"],
        mixing_audio_cfg=configs["mixing_audio_cfg"],
        skip_ratio=configs["mixing_audio_cfg"].skip_ratio
    )

    val_dataset = create_dataset(
        clean_filepaths=val_filepaths["clean"],
        noisy_filepaths=val_filepaths["noisy"],
        mixing_audio_cfg=configs["mixing_audio_cfg"],
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

    return trainer, model, data_module

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ### Evaluation utils

# %% [code] {"execution":{"iopub.status.busy":"2026-03-28T09:24:27.777173Z","iopub.execute_input":"2026-03-28T09:24:27.777443Z","iopub.status.idle":"2026-03-28T09:24:27.811745Z","shell.execute_reply.started":"2026-03-28T09:24:27.777422Z","shell.execute_reply":"2026-03-28T09:24:27.810922Z"},"jupyter":{"source_hidden":true}}
# def evaluate_from_checkpoint(
#     checkpoint_path: str,
#     test_ears_df: pd.DataFrame,
#     test_wham_df: pd.DataFrame,
#     num_samples: int = 3,
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"
# ) -> None:
#     """Evaluate the MelReGAN model from a checkpoint and visualize results."""
    
#     configs = create_configs()
    
#     pipeline = create_pipeline(
#         audio_preprocessor_cfg=configs["audio_preprocessor_cfg"], 
#         audio_augmentor_cfg=configs["audio_augumentor_cfg"], 
#         normalizer_cfg=configs["normalizer_cfg"]
#     )
    
#     generator = MelReGANGenerator(start_filters=configs["train_cfg"].g_filters)
#     discriminator = MelReGANDiscriminator(
#         in_channels=configs["train_cfg"].d_input_channels, 
#         start_filters=configs["train_cfg"].d_filters
#     )
    
#     scaler = create_scaler(configs["normalizer_cfg"])

#     model = MelReGAN.load_from_checkpoint(
#         checkpoint_path,
#         generator=generator,
#         discriminator=discriminator,
#         pipeline=pipeline,
#         scaler=scaler,
#         map_location=device
#     )
    
#     model.to(device)
#     model.eval()

#     test_dataset = create_dataset(
#         ears_df=test_ears_df,
#         wham_df=test_wham_df,
#         mixing_audio_cfg=configs["mixing_audio_cfg"],
#         train=False,
#         skip_ratio=10
#     )
    
#     hifi_gan = HIFIGAN.from_hparams(
#         source="speechbrain/tts-hifigan-libritts-16kHz",
#         savedir="tmp_hifigan",
#         run_opts={"device": device}
#     )
    
#     hifi_converter = DBToLogScaler(configs["audio_preprocessor_cfg"]).to(device)

#     data_iter = iter(test_dataset)

#     for i in range(num_samples):
#         try:
#             mixed_window, clean_window = next(data_iter)
#         except StopIteration:
#             print("Reached end of test dataset.")
#             break

#         mixed_windows = mixed_window.unsqueeze(0).to(device) 
#         clean_windows = clean_window.unsqueeze(0).to(device)

#         with torch.no_grad():
#             noisy_spec_norm, clean_spec_norm = model.pipeline(mixed_windows, clean_windows)

#             enhanced_spec_norm = model.generator(noisy_spec_norm)

#             noisy_spec_db = model.scaler.denormalize(noisy_spec_norm)
#             clean_spec_db = model.scaler.denormalize(clean_spec_norm)
#             enhanced_spec_db = model.scaler.denormalize(enhanced_spec_norm)

#             noisy_hifi = hifi_converter(noisy_spec_db).squeeze(1)
#             clean_hifi = hifi_converter(clean_spec_db).squeeze(1)
#             enhanced_hifi = hifi_converter(enhanced_spec_db).squeeze(1)

#             audio_noisy = hifi_gan.decode_batch(noisy_hifi).cpu().squeeze().numpy()
#             audio_clean = hifi_gan.decode_batch(clean_hifi).cpu().squeeze().numpy()
#             audio_enhanced = hifi_gan.decode_batch(enhanced_hifi).cpu().squeeze().numpy()
            
#             viz_noisy = noisy_spec_db[0].squeeze().cpu().numpy()
#             viz_clean = clean_spec_db[0].squeeze().cpu().numpy()
#             viz_enhanced = enhanced_spec_db[0].squeeze().cpu().numpy()

        
#         print(f"Sample {i+1}/{num_samples}")
        
#         fig, axes = plt.subplots(1, 3, figsize=(18, 5))
#         kwargs = {'origin': 'lower', 'aspect': 'auto', 'cmap': 'magma', 'vmin': -80, 'vmax': 0}
        
#         axes[0].imshow(viz_noisy, **kwargs)
#         axes[0].set_title("Input (Noisy)")
        
#         axes[1].imshow(viz_enhanced, **kwargs)
#         axes[1].set_title("Output (Enhanced)")
        
#         axes[2].imshow(viz_clean, **kwargs)
#         axes[2].set_title("Target (Clean)")
#         plt.show()
        
#         print("Input (Noisy):")
#         ipd.display(ipd.Audio(audio_noisy, rate=16000))
        
#         print("Output (Enhanced):")
#         ipd.display(ipd.Audio(audio_enhanced, rate=16000))
        
#         print("Target (Clean):")
#         ipd.display(ipd.Audio(audio_clean, rate=16000))
#         print("\n")

# import os
# import soundfile as sf # Pamiętaj o dodaniu tego importu na górze pliku!
# import matplotlib.pyplot as plt
# import IPython.display as ipd
# import torch
# import pandas as pd

def evaluate_from_checkpoint(
    checkpoint_path: str,
    test_ears_df: pd.DataFrame,
    test_wham_df: pd.DataFrame,
    num_samples: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "eval_outputs" # <-- NOWY PARAMETR
) -> None:
    """Evaluate the MelReGAN model from a checkpoint and visualize and save results."""
    
    # Tworzenie katalogu wyjściowego, jeśli nie istnieje
    os.makedirs(output_dir, exist_ok=True)
    
    configs = create_configs()
    
    pipeline = create_pipeline(
        audio_preprocessor_cfg=configs["audio_preprocessor_cfg"], 
        audio_augmentor_cfg=configs["audio_augumentor_cfg"], 
        normalizer_cfg=configs["normalizer_cfg"]
    )
    
    generator = MelReGANGenerator(start_filters=configs["train_cfg"].g_filters)
    discriminator = MelReGANDiscriminator(
        in_channels=configs["train_cfg"].d_input_channels, 
        start_filters=configs["train_cfg"].d_filters
    )
    
    scaler = create_scaler(configs["normalizer_cfg"])

    model = MelReGAN.load_from_checkpoint(
        checkpoint_path,
        generator=generator,
        discriminator=discriminator,
        pipeline=pipeline,
        scaler=scaler,
        map_location=device
    )
    
    model.to(device)
    model.eval()

    test_dataset = create_dataset(
        ears_df=test_ears_df,
        wham_df=test_wham_df,
        mixing_audio_cfg=configs["mixing_audio_cfg"],
        train=False,
        skip_ratio=10
    )
    
    hifi_gan = HIFIGAN.from_hparams(
        source="speechbrain/tts-hifigan-libritts-16kHz",
        savedir="tmp_hifigan",
        run_opts={"device": device}
    )
    
    hifi_converter = DBToLogScaler(configs["audio_preprocessor_cfg"]).to(device)

    data_iter = iter(test_dataset)

    for i in range(num_samples):
        try:
            mixed_window, clean_window = next(data_iter)
        except StopIteration:
            print("Reached end of test dataset.")
            break

        mixed_windows = mixed_window.unsqueeze(0).to(device) 
        clean_windows = clean_window.unsqueeze(0).to(device)

        with torch.no_grad():
            noisy_spec_norm, clean_spec_norm = model.pipeline(mixed_windows, clean_windows)

            enhanced_spec_norm = model.generator(noisy_spec_norm)

            noisy_spec_db = model.scaler.denormalize(noisy_spec_norm)
            clean_spec_db = model.scaler.denormalize(clean_spec_norm)
            enhanced_spec_db = model.scaler.denormalize(enhanced_spec_norm)

            noisy_hifi = hifi_converter(noisy_spec_db).squeeze(1)
            clean_hifi = hifi_converter(clean_spec_db).squeeze(1)
            enhanced_hifi = hifi_converter(enhanced_spec_db).squeeze(1)

            audio_noisy = hifi_gan.decode_batch(noisy_hifi).cpu().squeeze().numpy()
            audio_clean = hifi_gan.decode_batch(clean_hifi).cpu().squeeze().numpy()
            audio_enhanced = hifi_gan.decode_batch(enhanced_hifi).cpu().squeeze().numpy()
            
            viz_noisy = noisy_spec_db[0].squeeze().cpu().numpy()
            viz_clean = clean_spec_db[0].squeeze().cpu().numpy()
            viz_enhanced = enhanced_spec_db[0].squeeze().cpu().numpy()

        
        print(f"Sample {i+1}/{num_samples}")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        kwargs = {'origin': 'lower', 'aspect': 'auto', 'cmap': 'magma', 'vmin': -80, 'vmax': 0}
        
        axes[0].imshow(viz_noisy, **kwargs)
        axes[0].set_title("Input (Noisy)")
        
        axes[1].imshow(viz_enhanced, **kwargs)
        axes[1].set_title("Output (Enhanced)")
        
        axes[2].imshow(viz_clean, **kwargs)
        axes[2].set_title("Target (Clean)")
        
        # Zapis figury ze spektrogramami do pliku
        spec_path = os.path.join(output_dir, f"sample_{i+1}_spectrograms.png")
        plt.savefig(spec_path, bbox_inches='tight')
        plt.show()
        
        # Zapis plików audio do formatu .wav
        sf.write(os.path.join(output_dir, f"sample_{i+1}_noisy.wav"), audio_noisy, 16000)
        sf.write(os.path.join(output_dir, f"sample_{i+1}_enhanced.wav"), audio_enhanced, 16000)
        sf.write(os.path.join(output_dir, f"sample_{i+1}_clean.wav"), audio_clean, 16000)
        
        print(f"Dane zapisane w katalogu: '{output_dir}'")
        
        print("Input (Noisy):")
        ipd.display(ipd.Audio(audio_noisy, rate=16000))
        
        print("Output (Enhanced):")
        ipd.display(ipd.Audio(audio_enhanced, rate=16000))
        
        print("Target (Clean):")
        ipd.display(ipd.Audio(audio_clean, rate=16000))
        print("\n")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Train

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-28T09:24:27.812758Z","iopub.execute_input":"2026-03-28T09:24:27.813118Z","iopub.status.idle":"2026-03-28T10:35:45.706032Z","shell.execute_reply.started":"2026-03-28T09:24:27.813085Z","shell.execute_reply":"2026-03-28T10:35:45.704512Z"}}
trainer, model, data_module = train(train_size=90_000, train_percentage=0.8)

CHECKPOINT_TO_RESUME = "checkpoints/last.ckpt"  # Path to the checkpoint you want to resume from
if os.path.exists(CHECKPOINT_TO_RESUME):
    print(f"Resuming training from checkpoint: {CHECKPOINT_TO_RESUME}")
    trainer.fit(model, datamodule=data_module, ckpt_path=CHECKPOINT_TO_RESUME)
else:
    print("No checkpoint found, starting training from scratch.")
    trainer.fit(model, datamodule=data_module)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## Evaluate

# %% [code] {"execution":{"iopub.status.busy":"2026-03-28T10:35:45.706728Z","iopub.status.idle":"2026-03-28T10:35:45.707004Z","shell.execute_reply.started":"2026-03-28T10:35:45.706869Z","shell.execute_reply":"2026-03-28T10:35:45.706880Z"},"jupyter":{"outputs_hidden":false}}
!ls -la /kaggle/input/datasets/hubertmka/ears-wham

# %% [code] {"execution":{"iopub.status.busy":"2026-03-28T10:35:45.708366Z","iopub.status.idle":"2026-03-28T10:35:45.708779Z","shell.execute_reply.started":"2026-03-28T10:35:45.708554Z","shell.execute_reply":"2026-03-28T10:35:45.708572Z"},"jupyter":{"outputs_hidden":false}}
# !ls -la /kaggle/working/checkpoints

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-03-28T10:35:45.710492Z","iopub.status.idle":"2026-03-28T10:35:45.710770Z","shell.execute_reply.started":"2026-03-28T10:35:45.710644Z","shell.execute_reply":"2026-03-28T10:35:45.710655Z"}}

# CHECKPOINT_PATH = "checkpoints/mel_bin2bin-epoch=152-val_loss=0.2860.ckpt" # - 15_000 probes
# CHECKPOINT_PATH = "checkpoints/mel_bin2bin-epoch=153-val_loss=0.2355.ckpt" # - 50_000 probes
# CHECKPOINT_PATH = "checkpoints/mel_MelReGAN-epoch=101-val_loss=0.2311.ckpt"


# evaluate_from_checkpoint(
#     checkpoint_path=CHECKPOINT_PATH,
#     test_ears_df=split_dfs["test_ears_df"],
#     test_wham_df=split_dfs["test_wham_df"],
#     num_samples=50
# )