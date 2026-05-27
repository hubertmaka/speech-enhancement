import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


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
    """Simple attention mechanism that applies separate attention along frequency and time dimensions."""
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
    """Attention Gate for U-Net skip connections. Computes attention weights based on the decoder's current state and the encoder's skip connection."""
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
    """Basic Residual Block without dilation."""
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
    """Encoder block that outputs both the skip connection and the downsampled output for the next layer."""
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
        
        self.channel_adj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        ) if in_channels != out_channels else nn.Identity()

        self.res_blocks = nn.Sequential(
            BasicResBlock(out_channels, kernel_size, padding, use_norm),
            BasicResBlock(out_channels, kernel_size, padding, use_norm)
        )
        
        self.downsample = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.channel_adj(x)
        skip_out = self.res_blocks(x)
        next_layer_in = self.downsample(skip_out)
        return skip_out, next_layer_in


class ResidualDecoderBlock(nn.Module):
    """Decoder block that performs upsampling and outputs the result for the next layer."""
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
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.reduce = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.res_blocks = nn.Sequential(
            BasicResBlock(out_channels, kernel_size, padding, use_norm),
            BasicResBlock(out_channels, kernel_size, padding, use_norm)
        )
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.reduce(x)
        x = self.res_blocks(x)
        x = self.dropout(x)
        return x

class PatchBlock(nn.Module):
    """Basic block for PatchGAN discriminator: conv -> instance norm -> leakyrelu"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size=4, stride=2, padding=1, use_norm=True):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_norm))
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))
    

class BigGANResBlock(nn.Module):
    """Residual block inspired by BigGAN, with optional upsampling or downsampling and dilation for larger receptive field."""
    def __init__(self, in_channels: int, out_channels: int, resample: str = None, dilation: int = 1):
        super().__init__()
        self.resample = resample 
        
        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.act1 = nn.SiLU(inplace=True) 
        
        self.pad1 = nn.ReflectionPad2d(dilation)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, dilation=dilation, bias=False)
        
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act2 = nn.SiLU(inplace=True)
        
        self.pad2 = nn.ReflectionPad2d(dilation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0, dilation=dilation, bias=False)

        self.skip_conv = nn.Identity()
        if in_channels != out_channels or resample is not None:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def _apply_resample(self, x: torch.Tensor) -> torch.Tensor:
        if self.resample == 'up':
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        elif self.resample == 'down':
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act1(h)
        h = self._apply_resample(h)
            
        h = self.pad1(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.act2(h)
        
        h = self.pad2(h)
        h = self.conv2(h)

        s = self._apply_resample(x)
        s = self.skip_conv(s)

        out = (h + s) / math.sqrt(2.0)
        return out
