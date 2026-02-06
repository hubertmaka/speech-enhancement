import math
import random
import itertools
from typing import Literal, Generator

import torch
import pandas as pd
import pytorch_lightning as pl
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from torch.utils.data import IterableDataset, get_worker_info, DataLoader
import torch.nn as nn

from utility_scripts.configs import (
    MixingAudioDatasetConfig, 
    AudioPreprocessorConfig, 
    AudioAugumentorConfig, 
    NormalizerConfig
)


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

class AudioMixingDataset(IterableDataset):
    """Dataset for mixing clean audio with noise at random SNR levels."""
    def __init__(
            self,
            ears_df: pd.DataFrame,
            wham_df: pd.DataFrame,
            config: MixingAudioDatasetConfig,
            mode: Literal["train", "val", "test"],
            skip_ratio: int = 1
    ) -> None:
        self.ears_paths = ears_df["path"].tolist()
        self.wham_paths = wham_df["path"].tolist()
        self.c = config
        self.mode = mode
        self.sr = skip_ratio
        
        self.segment_samples = int(self.c.segment_sec * self.c.sample_rate)
        self.overlap_samples = int(self.c.overlap * self.c.sample_rate)
        self.step_samples = (self.segment_samples - self.overlap_samples) * self.sr
        
        if self.mode in ["val", "test"]:
            torch.manual_seed(42)
            self.fixed_snr_values = torch.empty(4000).uniform_(self.c.min_snr, self.c.max_snr)
            self.snr_index = 0
        else:
            self.fixed_snr_values = None
        
    def load_audio(self, path: str) -> torch.Tensor:
        """Load audio from a given file path."""
        waveform, sr = torchaudio.load(path)
        
        if sr != self.c.sample_rate:
            raise ValueError(f"Sample rate mismatch: expected {self.c.sample_rate}, got {sr}")
        
        return waveform.squeeze(0)
    
    def align_noise(self, noise: torch.Tensor, target_len: int) -> torch.Tensor:
        """Align noise to the target length by trimming or repeating."""
        noise_len = noise.shape[0]
        
        if noise_len == target_len:
            return noise
        
        if noise_len < target_len:
            repeats = (target_len + noise_len - 1) // noise_len
            return noise.repeat(repeats)[:target_len]
        else:
            start = torch.randint(0, noise_len - target_len + 1, (1,), dtype=torch.long).item()
            return noise[start:start + target_len]
    
    def cut_audio(self, audio: torch.Tensor, target_len: int) -> torch.Tensor:
        """Cut or pad audio to the target length."""
        audio_len = audio.shape[0]
        if audio_len > target_len:
            start = torch.randint(0, audio_len - target_len, (1,)).item()
            return audio[start:start + target_len]
        elif audio_len < target_len:
            padding = target_len - audio_len
            return torch.nn.functional.pad(audio, (0, padding))
        else:
            return audio
    
    def batch_add_noise(self, clean_windows: torch.Tensor, noise_windows: torch.Tensor) -> torch.Tensor:
        """Add noise to clean audio windows at specified SNR levels."""
        num_windows = clean_windows.shape[0]
        
        if self.mode == "train":
            snr_db = torch.empty(num_windows).uniform_(self.c.min_snr, self.c.max_snr)
        else:
            snr_db = torch.empty(num_windows)
            for i in range(num_windows):
                snr_db[i] = self.fixed_snr_values[self.snr_index % len(self.fixed_snr_values)]
                self.snr_index += 1
        
        return torch.vmap(F.add_noise)(clean_windows, noise_windows, snr_db)

    def __iter__(self) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Iterator to yield mixed audio and clean audio pairs."""
        worker_info = get_worker_info()
        clean_paths = self.ears_paths[:]
        wham_paths = self.wham_paths[:]
        
        if worker_info is not None:
            per_worker = int(math.ceil(len(clean_paths) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(clean_paths))
            clean_paths = clean_paths[iter_start:iter_end]
        
        random.shuffle(wham_paths)
        noise_iter = itertools.cycle(wham_paths)
        
        if self.mode == "train":
            random.shuffle(clean_paths)
            
        for clean_path in clean_paths:
            clean_audio = self.load_audio(clean_path)

            if clean_audio.shape[0] < self.segment_samples:
                padding = self.segment_samples - clean_audio.shape[0]
                clean_audio = torch.nn.functional.pad(clean_audio, (0, padding))

            noise_path = next(noise_iter)
            noise_audio = self.load_audio(noise_path)
            
            noise_audio = self.align_noise(noise_audio, clean_audio.shape[0])
            
            clean_windows = clean_audio.unfold(0, self.segment_samples, self.step_samples)
            noise_windows = noise_audio.unfold(0, self.segment_samples, self.step_samples)

            mixed_windows = self.batch_add_noise(clean_windows, noise_windows)
            
            yield from zip(mixed_windows, clean_windows)


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
            f_max=self.c.sample_rate // 2
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram and add channel dimension."""
        x = self.mel_spectrogram(x)
        return x.unsqueeze(1)


# ======= Scalers =======

class AmplitudeToDBScaler(Scaler):
    """Convert amplitude spectrograms to decibel scale."""
    def __init__(self, audio_preprocessor_config: AudioPreprocessorConfig) -> None:
        super().__init__()
        self.top_db = audio_preprocessor_config.top_db
        self.amplitude_to_db = T.AmplitudeToDB(top_db=self.top_db)
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Convert amplitude spectrogram to decibel scale."""
        return self.amplitude_to_db(spec)
    

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
        spec = spec[:, :self.target_shape[0], :self.target_shape[1]]
        return spec


# ======= Pipelines =======

class TrainPipeline(nn.Module):
    """Pipeline for processing audio during training."""
    def __init__(
            self,
            preprocessor: SpectrogramProcessor,
            augmentor: AudioAugmentor,
            scale_converter: Scaler,
            scaler: Normalizer,
            adjuster: Adjuster | None = None
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.augmentor = augmentor
        self.scale_converter = scale_converter
        self.scaler = scaler
        self.adjuster = adjuster

    def forward(self, noisy: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the training pipeline."""
        noisy_spec = self.preprocessor(noisy)
        clean_spec = self.preprocessor(clean)
        
        noisy_spec = self.augmentor(noisy_spec)

        noisy_spec = self.scale_converter(noisy_spec)
        clean_spec = self.scale_converter(clean_spec)
        
        noisy_spec = self.scaler(noisy_spec)        
        clean_spec = self.scaler(clean_spec)

        if self.adjuster:
            noisy_spec = self.adjuster(noisy_spec)
            clean_spec = self.adjuster(clean_spec)

        return noisy_spec, clean_spec


# ======= DataModule =======

class AudioDataModule(pl.LightningDataModule):
    """DataModule for audio mixing dataset."""
    def __init__(
            self, 
            train_ds: AudioMixingDataset,
            val_ds: AudioMixingDataset,
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
            prefetch_factor=4 if self.num_workers > 0 else None,
            drop_last=True,
            shuffle=False
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=max(2, self.num_workers // 2),
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=False,
            shuffle=False
        )