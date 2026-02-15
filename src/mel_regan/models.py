import torch
import torch.nn as nn

from src.mel_generative_speech_enhancer.layers import (
    ResidualEncoderBlock,
    ResidualDecoderBlock,
    DilatedResBlock,
    SpectralAttention,
    AttentionGate,
    PatchBlock
)


class MelReGANGenerator(nn.Module):
    """U-Net based Generator with Residual Blocks and Attention Gates."""
    def __init__(self, start_filters: int = 64) -> None:
        super().__init__()
        nf = start_filters
        
        self.encoder1 = ResidualEncoderBlock(1, nf)      
        self.encoder2 = ResidualEncoderBlock(nf, nf*2)                   
        self.encoder3 = ResidualEncoderBlock(nf*2, nf*4)                 
        self.encoder4 = ResidualEncoderBlock(nf*4, nf*8) 
        
        self.bottleneck = nn.Sequential(
            DilatedResBlock(nf*8, dilation=2),
            # SpectralAttention(nf*8),
            DilatedResBlock(nf*8, dilation=4),
            # SpectralAttention(nf*8),
            DilatedResBlock(nf*8, dilation=4),
            SpectralAttention(nf*8)
        )
        
        self.gate1 = AttentionGate(F_g=nf*4, F_l=nf*4, F_int=nf*2) 
        self.gate2 = AttentionGate(F_g=nf*2, F_l=nf*2, F_int=nf)   
        self.gate3 = AttentionGate(F_g=nf,   F_l=nf,   F_int=nf//2)
        
        self.decoder1 = ResidualDecoderBlock(nf*8, nf*4, dropout=0.5)    
        self.decoder2 = ResidualDecoderBlock(nf*4*2, nf*2, dropout=0.5)  
        self.decoder3 = ResidualDecoderBlock(nf*2*2, nf, dropout=0.0)
        self.final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(nf*2, 1, kernel_size=3, stride=1, padding=0),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MelReGAN Generator."""
        e1_skip, e1_next = self.encoder1(x)      
        e2_skip, e2_next = self.encoder2(e1_next)     
        e3_skip, e3_next = self.encoder3(e2_next)     
        e4_skip, _       = self.encoder4(e3_next)     
        
        x_bottleneck = self.bottleneck(e4_skip)
        
        d1 = self.decoder1(x_bottleneck)        
        e3_gated = self.gate1(d1, e3_skip)      
        d1 = torch.cat([d1, e3_gated], dim=1)   
        
        d2 = self.decoder2(d1)                  
        e2_gated = self.gate2(d2, e2_skip)      
        d2 = torch.cat([d2, e2_gated], dim=1)   

        d3 = self.decoder3(d2)                  
        e1_gated = self.gate3(d3, e1_skip)      
        d3 = torch.cat([d3, e1_gated], dim=1)   
        
        output = self.final(d3)
        return output


class MelReGANDiscriminator(nn.Module):
    """PatchGAN Discriminator."""
    def __init__(
            self, 
            in_channels: int = 2, 
            start_filters: int = 64
        ) -> None:
        super().__init__()
        nf = start_filters

        # Layer 1: Stride 2 (Downsample) -> Output: 40x64
        self.layer_1 = PatchBlock(in_channels, nf, kernel_size=(4,4), stride=2)
        
        # Layer 2: Stride 1 (Zachowaj rozmiar) -> Output: 40x64
        self.layer_2 = PatchBlock(nf, nf * 2, kernel_size=(4,4), stride=2)
        
        # # Layer 3: Stride 1 -> Output: 40x64
        # self.layer_3 = PatchBlock(nf * 2, nf * 4, kernel_size=(4,4), stride=1)
        
        self.final = nn.Conv2d(nf * 2, 1, kernel_size=(4,4), stride=1, padding=1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        d_in = torch.cat([x, condition], dim=1)
        d1 = self.layer_1(d_in)
        d2 = self.layer_2(d1)
        # d3 = self.layer_3(d2)
        output = self.final(d2)
        return output