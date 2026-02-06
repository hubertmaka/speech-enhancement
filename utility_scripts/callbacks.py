import os

import pytorch_lightning as pl
import matplotlib.pyplot as plt


class SpectrogramLogger(pl.Callback):
    """Callback to log spectrograms during validation."""
    def __init__(self, save_dir: str = "saved_spectrograms", num_samples: int = 10):
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