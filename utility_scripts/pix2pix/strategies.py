import torch
import torch.nn as nn
import pytorch_lightning as pl

from utility_scripts.pix2pix.models import Pix2PixDiscriminator, Pix2PixGenerator
from utility_scripts.utils import init_weights


class Pix2Pix(pl.LightningModule):
    """
    Pix2Pix GAN model for image-to-image translation.
    """
    def __init__(
            self, 
            generator: Pix2PixGenerator, 
            discriminator: Pix2PixDiscriminator,
            pipeline: nn.Module,
            lr: float = 2e-4,
            lambda_recon: float = 100.0
        ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator", "pipeline"])
        self.generator = generator
        self.discriminator = discriminator
        self.pipeline = pipeline
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.automatic_optimization = False

        self.generator.apply(init_weights)
        self.discriminator.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the generator."""
        return self.generator(x)
    
    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list]:
        """Configure optimizers for generator and discriminator."""
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a training step for both generator and discriminator."""
        g_opt, d_opt = self.optimizers()
        mixed_audio, clean_audio = batch
        real_noisy, real_clean = self.pipeline(mixed_audio, clean_audio)

        fake_clean = self.generator(real_noisy)
        pred_real = self.discriminator(real_clean, real_noisy)
        loss_d_real = self.adversarial_loss(pred_real, torch.ones_like(pred_real))

        pred_fake = self.discriminator(fake_clean.detach(), real_noisy)
        loss_d_fake = self.adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
        total_disc_loss = (loss_d_real + loss_d_fake) * 0.5

        d_opt.zero_grad()
        self.manual_backward(total_disc_loss)
        d_opt.step()
        
        self.log("disc_loss", total_disc_loss, prog_bar=True)

        pred_fake_g = self.discriminator(fake_clean, real_noisy)
        gan_loss = self.adversarial_loss(pred_fake_g, torch.ones_like(pred_fake_g))
        l1_loss = self.l1_loss(fake_clean, real_clean) * self.hparams.lambda_recon
        total_gen_loss = gan_loss + l1_loss

        g_opt.zero_grad()
        self.manual_backward(total_gen_loss)
        g_opt.step()
        
        self.log_dict({
            "gen_total": total_gen_loss,
            "gen_gan": gan_loss,
            "gen_l1": l1_loss
        }, prog_bar=True)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor] | None:
        """Perform a validation step."""
        mixed_audio, clean_audio = batch
        real_noisy, real_clean = self.pipeline(mixed_audio, clean_audio)
        fake_clean = self.generator(real_noisy)
        val_loss = self.l1_loss(fake_clean, real_clean)
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        
        if batch_idx == 0:
            return {"noisy": real_noisy, "clean": real_clean, "fake": fake_clean}
        return None
