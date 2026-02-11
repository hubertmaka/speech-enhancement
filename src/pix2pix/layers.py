import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """Encoder block: convolutional -> batchnorm -> leakyrelu"""
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: tuple[int, int] = (4, 4),
            stride: tuple[int, int] = (2, 2), 
            padding: tuple[int, int] = (1, 1), 
            norm: bool = True,
            activation: bool = True
        ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder block."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        return x
    

class DecoderBlock(nn.Module):
    """Decoder block: transposed convolutional -> batchnorm -> relu -> (optional dropout)"""
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            kernel_size: tuple[int, int] = (4, 4),
            stride: tuple[int, int] = (2, 2), 
            padding: tuple[int, int] = (1, 1), 
            norm: bool = True,
            dropout: float = 0.0
        ) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder block."""
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x