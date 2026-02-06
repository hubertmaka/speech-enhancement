import torch
import torch.nn as nn


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
    """Spectral Attention Mechanism focusing on frequency and time axes separately."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.channels = channels
        
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.time_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the spectral attention mechanism."""
        freq_att = self.freq_attention(x)
        time_att = self.time_attention(x)
        out = x * (self.alpha * freq_att) * (self.beta * time_att)
        return out


class AttentionGate(nn.Module):
    """Attention Gate for skip connections in U-Net architecture."""
    def __init__(self, F_g: int, F_l: int, F_int: int) -> None:
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the attention gate."""
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


class ResidualEncoderBlock(nn.Module):
    """Residual Encoder block.
    Schema: residual block -> downsampling
    """
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int,
            kernel_size: int = 3,
            padding: int = 1,
            stride: int = 2,
            use_activation: bool = True,
            use_norm: bool = True
        ) -> None:
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ReflectionPad2d(padding),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity(),
        )
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity()
            )
        
        self.downsample = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.final_activation = nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the residual encoder block."""
        residual = self.res_block(x)
        shortcut = self.shortcut(x)
        skip_out = self.final_activation(residual + shortcut)
        next_layer_in = self.downsample(skip_out)
        return skip_out, next_layer_in



class ResidualDecoderBlock(nn.Module):
    """Decoder block with Residual Connection:
    Schema: Upsample -> Reduce Channels -> Residual Block -> Add & Activation -> (Optional Dropout)
    """
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: int = 3,
            padding: int = 1,
            stride: int = 1,
            dropout: float = 0.0,
            use_norm: bool = True,
            use_activation: bool = True,
            upsample: bool = True
        ) -> None:
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') if upsample else nn.Identity()
        
        self.reduce = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity()
        )
        
        self.res_block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity(),
            
            nn.ReflectionPad2d(padding),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity(),
        )
        
        self.final_activation = nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity()
        
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual decoder block."""
        x = self.upsample(x)
        x = self.reduce(x)
        residual = self.res_block(x)
        x = self.final_activation(x + residual)
        x = self.dropout(x)
        return x


class PatchBlock(nn.Module):
    """PatchGAN Discriminator Block: Conv -> InstanceNorm -> LeakyReLU"""
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: tuple[int, int] = (8, 2),
            stride: int = 2,
            padding: tuple[int, int] = (3, 1),
            use_norm: bool = True,
            use_activation: bool = True
        ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_norm)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if use_norm else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the PatchGAN discriminator block."""
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x