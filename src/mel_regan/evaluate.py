import os
import random
import time
from typing import Dict, Union, Optional

import torch
import matplotlib.pyplot as plt
import pandas as pd
import IPython.display as ipd
from speechbrain.inference.vocoders import HIFIGAN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import torchaudio
import torchaudio.functional as F
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics import MetricCollection

from src.configs import create_configs
from src.mel_regan.utils import create_pipeline, create_dataset, create_scaler
from src.mel_regan.strategies import MelReGAN  
from src.mel_regan.utils import DBToLogScaler
from src.mel_regan.models import MelReGANGenerator, MelReGANDiscriminator


def evaluate_from_checkpoint(
    checkpoint_path: str,
    test_ears_df: pd.DataFrame,
    test_wham_df: pd.DataFrame,
    num_samples: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> None:
    """Evaluate the MelReGAN model from a checkpoint and visualize results."""
    
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
        plt.show()
        
        print("Input (Noisy):")
        ipd.display(ipd.Audio(audio_noisy, rate=16000))
        
        print("Output (Enhanced):")
        ipd.display(ipd.Audio(audio_enhanced, rate=16000))
        
        print("Target (Clean):")
        ipd.display(ipd.Audio(audio_clean, rate=16000))
        print("\n")

# ====================================================================

class TorchMetricsAudioEvaluator(torch.nn.Module):
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate
        pesq_mode = 'wb' if self.sample_rate == 16000 else 'nb'
        self.metrics = MetricCollection({
            'PESQ': PerceptualEvaluationSpeechQuality(fs=self.sample_rate, mode=pesq_mode),
            'STOI': ShortTimeObjectiveIntelligibility(fs=self.sample_rate, extended=False),
            'ESTOI': ShortTimeObjectiveIntelligibility(fs=self.sample_rate, extended=True)
        })

    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        return self.metrics(preds, target)

    def reset_epoch_metrics(self) -> None:
        self.metrics.reset()


class ModelEvaluator:
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.objective_model = SQUIM_OBJECTIVE.get_model().to(self.device).eval()
        self.subjective_model = SQUIM_SUBJECTIVE.get_model().to(self.device).eval()
        self.squim_required_sr = 16000

    def _prepare_tensor(self, waveform: torch.Tensor, original_sr: int) -> torch.Tensor:
        if waveform.dim() == 1: waveform = waveform.unsqueeze(0)
        if original_sr != self.squim_required_sr:
            waveform = F.resample(waveform, original_sr, self.squim_required_sr)
        return waveform.to(self.device)

    @torch.no_grad()
    def evaluate(self, degraded_waveform: torch.Tensor, sample_rate: int,
                 non_matching_ref: Optional[torch.Tensor] = None, ref_sample_rate: Optional[int] = None) -> Dict:
        eval_wav = self._prepare_tensor(degraded_waveform, sample_rate)
        stoi_pred, pesq_pred, si_sdr_pred = self.objective_model(eval_wav)
        
        results = {
            'STOI': stoi_pred.squeeze().item(),
            'PESQ': pesq_pred.squeeze().item(),
            'SI-SDR': si_sdr_pred.squeeze().item()
        }
        
        if non_matching_ref is not None and ref_sample_rate is not None:
            ref_wav = self._prepare_tensor(non_matching_ref, ref_sample_rate)
            mos_pred = self.subjective_model(eval_wav, ref_wav)
            results['MOS'] = mos_pred.squeeze().item()
            
        return results


def mix_signals(clean_wav, noise_wav, snr_db):
    clean_power = torch.mean(clean_wav ** 2) + 1e-8
    noise_power = torch.mean(noise_wav ** 2) + 1e-8
    snr_linear = 10 ** (snr_db / 10)
    
    scaling_factor = torch.sqrt((clean_power / snr_linear) / noise_power)
    mixed_wav = clean_wav + (noise_wav * scaling_factor)
    
    max_amp = torch.max(torch.abs(mixed_wav))
    if max_amp > 0.99:
        mixed_wav = mixed_wav / max_amp * 0.99
        clean_wav = clean_wav / max_amp * 0.99
        
    return mixed_wav, clean_wav


def plot_and_save_results(results_df: pd.DataFrame, save_path: str = "evaluation_results.png"):
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    axes = axes.flatten()

    all_metrics = ['PESQ', 'STOI', 'ESTOI', 'MOS']
    custom_palette = {
        'Masked Noisy (Input)': '#808080',       
        'Noisy -> HiFi-GAN (Baseline)': '#E69F00', 
        'MelReGAN -> HiFi-GAN (Model)': '#009E73', 
        'Clean -> HiFi-GAN (Upper Bound)': '#0072B2'
    }
    
    snr_levels = sorted(results_df['SNR'].unique())

    for i, metric in enumerate(all_metrics):
        ax = axes[i]
        subset = results_df[results_df['Metric'] == metric]
        if subset.empty: continue
            
        sns.lineplot(data=subset, x='SNR', y='Value', hue='Signal', style='Evaluator', 
                     palette=custom_palette, markers=True, dashes=True, linewidth=3, 
                     markersize=10, errorbar=('ci', 95), ax=ax)
        
        ax.set_title(f'Metric: {metric}', fontweight='bold', pad=15)
        ax.set_xlabel('SNR Level [dB]')
        ax.set_ylabel('Score')
        ax.set_xticks(snr_levels)
        
        if i == 0: 
            ax.legend(title='Pipeline & Evaluator', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='12')
        else:
            if ax.get_legend(): ax.legend_.remove()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved evaluation results plot to: {save_path}")


def evaluate_audio_enhancement(
    model, ears_df: pd.DataFrame, wham_df: pd.DataFrame, hifi_gan, hifi_converter, 
    sample_rate: int = 16000, num_samples_per_snr: int = 10, 
    chunk_duration_sec: float = 2.04, max_mask_length_sec: float = 0.0,  
    device: str = "cuda"
) -> pd.DataFrame:
    
    model.eval().to(device)
    tm_evaluator = TorchMetricsAudioEvaluator(sample_rate=sample_rate).to(device)
    squim_evaluator = ModelEvaluator(device=device)
    
    ears_df = ears_df[~ears_df.get('file', pd.Series(dtype=str)).astype(str).str.contains('freeform', case=False, na=False)]
    shuffled_ears = ears_df.sample(frac=1, random_state=42).reset_index(drop=True)
    shuffled_wham = wham_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    sample_ears, sample_wham = [], []
    for idx, row in shuffled_ears.iterrows():
        path = row.get('path', row.get('file_path', row.iloc[0]))
        info = torchaudio.info(path)
        if 8.0 <= (info.num_frames / info.sample_rate) <= 12.0:
            sample_ears.append(row)
            sample_wham.append(shuffled_wham.iloc[idx % len(shuffled_wham)])
        if len(sample_ears) == num_samples_per_snr: break

    snr_levels = np.arange(-2.5, 17.6, 2.5)
    results_list = []
    chunk_samples = int(chunk_duration_sec * sample_rate)
    
    for snr in tqdm(snr_levels, desc="Evaluating SNRs"):
        for idx in range(len(sample_ears)):
            clean_path = sample_ears[idx].get('path', sample_ears[idx].get('file_path', sample_ears[idx].iloc[0]))
            noise_path = sample_wham[idx].get('path', sample_wham[idx].get('file_path', sample_wham[idx].iloc[0]))
            
            tqdm.write(f"SNR: {snr:5.1f} dB: {os.path.basename(clean_path)}")
            
            clean_wav, sr_c = torchaudio.load(clean_path)
            noise_wav, sr_n = torchaudio.load(noise_path)
            if sr_c != sample_rate: clean_wav = F.resample(clean_wav, sr_c, sample_rate)
            if sr_n != sample_rate: noise_wav = F.resample(noise_wav, sr_n, sample_rate)
            clean_wav, noise_wav = clean_wav[0:1, :], noise_wav[0:1, :]
            
            target_len = clean_wav.shape[1]
            if noise_wav.shape[1] > target_len:
                start_pt = random.randint(0, noise_wav.shape[1] - target_len)
                noise_wav = noise_wav[:, start_pt : start_pt + target_len]
            else:
                noise_wav = noise_wav.repeat(1, (target_len // noise_wav.shape[1]) + 1)[:, :target_len]
                
            mixed_wav, clean_wav_scaled = mix_signals(clean_wav, noise_wav, snr)
            original_len = mixed_wav.shape[-1]
            
            mixed_wav_masked = mixed_wav.clone()
            if max_mask_length_sec > 0:
                num_chunks = int(np.ceil(original_len / chunk_samples))
                for c in range(num_chunks):
                    len_t = random.uniform(0, max_mask_length_sec)
                    start_t = random.uniform(0, chunk_duration_sec - len_t)
                    s_idx = int(((c * chunk_duration_sec) + start_t) * sample_rate)
                    e_idx = min(int(((c * chunk_duration_sec) + start_t + len_t) * sample_rate), original_len)
                    mixed_wav_masked[..., s_idx:e_idx] = 0.0

            mixed_wav_masked, clean_wav_scaled = mixed_wav_masked.to(device), clean_wav_scaled.to(device)
            
            pad_needed = (chunk_samples - (original_len % chunk_samples)) % chunk_samples
            c_mixed = torch.nn.functional.pad(mixed_wav_masked, (0, pad_needed))
            c_clean = torch.nn.functional.pad(clean_wav_scaled, (0, pad_needed))
            
            enhanced_specs, noisy_specs, clean_specs = [], [], []
            with torch.no_grad():
                for c in range(c_mixed.shape[-1] // chunk_samples):
                    s_idx, e_idx = c * chunk_samples, (c + 1) * chunk_samples
                    c_noisy_spec, c_clean_spec = model.pipeline(c_mixed[:, s_idx:e_idx], c_clean[:, s_idx:e_idx])
                    
                    pad_size = (32 - (c_noisy_spec.shape[-1] % 32)) % 32
                    c_noisy_p = torch.nn.functional.pad(c_noisy_spec, (0, pad_size)) if pad_size > 0 else c_noisy_spec
                    c_enh_p = model.generator(c_noisy_p)
                    c_enh = c_enh_p[..., :-pad_size] if pad_size > 0 else c_enh_p
                        
                    enhanced_specs.append(c_enh)
                    noisy_specs.append(c_noisy_spec)
                    clean_specs.append(c_clean_spec)

                def decode_spec(spec):
                    db = model.scaler.denormalize(torch.cat(spec, dim=-1))
                    return hifi_gan.decode_batch(hifi_converter(db).squeeze(1)).squeeze(1)
                
                enhanced_wav_full = decode_spec(enhanced_specs)[:, :original_len]
                noisy_wav_full = decode_spec(noisy_specs)[:, :original_len]
                clean_wav_full = decode_spec(clean_specs)[:, :original_len]

            eval_variants = {
                'Masked Noisy (Input)': (mixed_wav_masked[:, :original_len], clean_wav_scaled[:, :original_len]),
                'Noisy -> HiFi-GAN (Baseline)': (noisy_wav_full, clean_wav_full), 
                'MelReGAN -> HiFi-GAN (Model)': (enhanced_wav_full, clean_wav_full),   
                'Clean -> HiFi-GAN (Upper Bound)': (clean_wav_full, clean_wav_full)
            }
            
            for name, (wav, ref) in eval_variants.items():
                tm_res = tm_evaluator(wav, ref)
                tm_evaluator.reset_epoch_metrics()
                sq_res = squim_evaluator.evaluate(wav, sample_rate, non_matching_ref=clean_wav_scaled[:, :original_len], ref_sample_rate=sample_rate)
                
                results_list.append({'SNR': snr, 'Signal': name, 'Metric': 'MOS', 'Value': sq_res['MOS'], 'Evaluator': 'SQUIM'})
                if name != 'Clean -> HiFi-GAN (Upper Bound)':
                    for m in ['PESQ', 'STOI', 'ESTOI']:
                        results_list.append({'SNR': snr, 'Signal': name, 'Metric': m, 'Value': tm_res[m].item(), 'Evaluator': 'TorchMetrics'})
                        if m in ['PESQ', 'STOI']:
                            results_list.append({'SNR': snr, 'Signal': name, 'Metric': m, 'Value': sq_res[m], 'Evaluator': 'SQUIM'})

    return pd.DataFrame(results_list)



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_PATH = "checkpoints/mel_MelReGAN-epoch=101-val_loss=0.2311.ckpt"

    # Zakładamy, że configs, split_dfs, oraz odpowiednie klasy pochodzą z Twojego środowiska
    configs = create_configs()
    pipeline = create_pipeline(configs["audio_preprocessor_cfg"], configs["audio_augumentor_cfg"], configs["normalizer_cfg"])
    generator = MelReGANGenerator(start_filters=configs["train_cfg"].g_filters)
    discriminator = MelReGANDiscriminator(in_channels=configs["train_cfg"].d_input_channels, start_filters=configs["train_cfg"].d_filters)
    scaler = create_scaler(configs["normalizer_cfg"])

    model = MelReGAN.load_from_checkpoint(
        CHECKPOINT_PATH, generator=generator, discriminator=discriminator,
        pipeline=pipeline, scaler=scaler, map_location=device
    ).to(device).eval()

    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="tmp_hifigan", run_opts={"device": device})
    hifi_converter = DBToLogScaler(configs["audio_preprocessor_cfg"]).to(device)

    # Uruchomienie ewaluacji
    df_results = evaluate_audio_enhancement(
        model=model, 
        ears_df=split_dfs["test_ears_df"], 
        wham_df=split_dfs["test_wham_df"], 
        hifi_gan=hifi_gan, 
        hifi_converter=hifi_converter,
        max_mask_length_sec=0,  # 0 oznacza wyłączenie nakładania masek w tej sesji
        num_samples_per_snr=50, 
        device=device
    )

    # Rysowanie i zapis wykresów na dysk
    plot_and_save_results(df_results, save_path="wyniki_ewaluacji.png")