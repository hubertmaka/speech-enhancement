import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.mel_regan.losses import RobustSpectrogramLoss
from src.mel_regan.models import MelReGANGenerator, MelReGANDiscriminator
from src.configs import AudioPreprocessorConfig

from src.mel_regan.losses import MaskedSpectralLoss
from src.mel_regan.models import MelReGANGenerator, MelReGANDiscriminator
from src.configs import AudioPreprocessorConfig


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
            lambda_gan: float = 1.0,
            discriminator_train_freq: int = 4,
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

        # ----------------------------------------------------------------------
        # TRAIN DISCRIMINATOR
        # ----------------------------------------------------------------------
        if train_d:
            with torch.no_grad():
                fake_clean_spec_detached = self.generator(real_lossy_spec)

            pred_real, _ = self.discriminator(real_clean_spec, real_lossy_spec)
            loss_d_real = self.adversarial_loss(pred_real, torch.ones_like(pred_real) * self.ls)

            pred_fake_d, _ = self.discriminator(fake_clean_spec_detached, real_lossy_spec)
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

            loss_fm = 0.0
            for feat_fake, feat_real in zip(fake_features, real_features):
                real_detach = feat_real.detach()
                norm_factor = torch.mean(torch.abs(real_detach)) + 1e-6
                loss_fm += F.l1_loss(feat_fake, real_detach) / norm_factor

        loss_mag, loss_sc = self.spectral_losses(fake_clean_spec, real_clean_spec)

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
                   scheduler.step()
           else:
               schedulers.step()