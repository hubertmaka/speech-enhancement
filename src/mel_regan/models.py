import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from src.mel_regan.layers import (
    ResidualEncoderBlock,
    ResidualDecoderBlock,
    DilatedResBlock,
    AttentionGate,
    PatchBlock,
    BigGANResBlock
)


# ==========================================
# 2. GENERATOR (SGMSE+ / NCSN++ U-Net)
# ==========================================

class MelReGANGenerator(nn.Module):
    def __init__(self, start_filters: int = 128) -> None:
        super().__init__()
        nf = start_filters
        
        self.conv_in = nn.Conv2d(1, nf, kernel_size=3, padding=1)
        self.down1 = BigGANResBlock(nf, nf, 'down')
        self.down2 = BigGANResBlock(nf, nf*2, 'down')
        self.down3 = BigGANResBlock(nf*2, nf*2, 'down')
        self.down4 = BigGANResBlock(nf*2, nf*2, 'down')

        self.bottleneck = nn.Sequential(
            BigGANResBlock(nf*2, nf*2, dilation=2), 
            BigGANResBlock(nf*2, nf*2, dilation=4),
        )

        self.att4 = AttentionGate(F_g=nf*2, F_l=nf*2, F_int=nf)
        self.dec4 = BigGANResBlock(nf*4, nf*2) 
        self.att3 = AttentionGate(F_g=nf*2, F_l=nf*2, F_int=nf)
        self.dec3 = BigGANResBlock(nf*4, nf*2)
        self.att2 = AttentionGate(F_g=nf*2, F_l=nf, F_int=nf)
        self.dec2 = BigGANResBlock(nf*3, nf) 
        self.att1 = AttentionGate(F_g=nf, F_l=nf, F_int=nf//2)
        self.dec1 = BigGANResBlock(nf*2, nf) 

        self.final = nn.Sequential(
            nn.InstanceNorm2d(nf, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(nf, 1, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        skip1 = x
        x = self.down1(x)                   
        skip2 = x
        x = self.down2(x)                   
        skip3 = x
        x = self.down3(x)                   
        skip4 = x
        x = self.down4(x)                   
        x = self.bottleneck(x)
        
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        s4 = self.att4(g=x, x=skip4)
        x = torch.cat([x, s4], dim=1)
        x = self.dec4(x)
        
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        s3 = self.att3(g=x, x=skip3)
        x = torch.cat([x, s3], dim=1)
        x = self.dec3(x)
        
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        s2 = self.att2(g=x, x=skip2)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x)
        
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        s1 = self.att1(g=x, x=skip1)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1(x)
        
        return self.final(x)

# ==========================================
# 3. DISCRIMINATOR (SGMSE+ Scale PatchGAN)
# ==========================================


class DiscResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False):
        super().__init__()
        self.downsample = downsample

        self.norm1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))

        self.norm2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))

        self.skip_conv = nn.Identity()
        if in_channels != out_channels or downsample:
            self.skip_conv = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        
        if self.downsample:
            h = F.avg_pool2d(h, kernel_size=2, stride=2)
            
        s = self.skip_conv(x)
        if self.downsample:
            s = F.avg_pool2d(s, kernel_size=2, stride=2)
            
        return (h + s) / math.sqrt(2.0)


class MelReGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 2, start_filters: int = 64) -> None:
        super().__init__()
        nf = start_filters

        self.conv_in = spectral_norm(nn.Conv2d(in_channels, nf, kernel_size=3, padding=1))
        
        self.block1 = DiscResBlock(nf, nf * 2, downsample=True)
        
        self.block2 = DiscResBlock(nf * 2, nf * 4, downsample=True)
        self.drop_2 = nn.Dropout2d(0.3)
        
        self.block3 = DiscResBlock(nf * 4, nf * 8, downsample=True)
        self.drop_3 = nn.Dropout2d(0.3)

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

        return output, [d1, d2, d3]