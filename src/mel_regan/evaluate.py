import torch
import matplotlib.pyplot as plt
import pandas as pd
import IPython.display as ipd
from speechbrain.inference.vocoders import HIFIGAN

from src.configs import create_configs
from src.mel_generative_speech_enhancer.utils import create_pipeline, create_dataset, create_scaler
from src.mel_generative_speech_enhancer.strategies import MelReGAN  
from src.mel_generative_speech_enhancer.utils import DBToLogScaler
from src.mel_generative_speech_enhancer.models import MelReGANGenerator, MelReGANDiscriminator


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
        
        print("🔴 Input (Noisy):")
        ipd.display(ipd.Audio(audio_noisy, rate=16000))
        
        print("🟢 Output (Enhanced):")
        ipd.display(ipd.Audio(audio_enhanced, rate=16000))
        
        print("🔵 Target (Clean):")
        ipd.display(ipd.Audio(audio_clean, rate=16000))
        print("\n")


