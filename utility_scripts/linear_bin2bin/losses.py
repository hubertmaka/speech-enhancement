import torch
import torch.nn as nn

from utility_scripts.configs import AudioPreprocessorConfig


class MaskedSpectralLoss(nn.Module):
    """Computes Masked Magnitude Loss and Spectral Convergence Loss."""
    def __init__(self, scaler: nn.Module, cfg: AudioPreprocessorConfig) -> None:
        super().__init__()
        self.scaler = scaler
        self.cfg = cfg
        self.factor = 10.0 if self.cfg.spec_type == "amplitude" else 20.0
        self.register_buffer('log_conversion', torch.log(torch.tensor(10.0)) / self.factor)

    def forward(self, pred_norm: torch.Tensor, target_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the Masked Magnitude Loss and Spectral Convergence Loss."""
        threshold = torch.max(target_norm.reshape(target_norm.size(0), -1), dim=1)[0] * 0.15
        threshold = threshold.reshape(-1, 1, 1, 1)
        
        mask = torch.ones_like(target_norm)
        mask[target_norm < threshold] = 0.1 
        
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
