import torch
import torch.nn as nn

from src.pix2pix.layers import EncoderBlock, DecoderBlock


class Pix2PixGenerator(nn.Module):
    """U-Net based Generator for Pix2Pix architecture. Based on Pix2Pix paper."""
    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = EncoderBlock(1, 64, norm=False)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)
        self.encoder5 = EncoderBlock(512, 512)
        self.encoder6 = EncoderBlock(512, 512)
        self.encoder7 = EncoderBlock(512, 512)
        self.encoder8 = EncoderBlock(512, 512, norm=False)

        self.decoder1 = DecoderBlock(512, 512, dropout=0.5)
        self.decoder2 = DecoderBlock(1024, 512, dropout=0.5)
        self.decoder3 = DecoderBlock(1024, 512, dropout=0.5)
        self.decoder4 = DecoderBlock(1024, 512)
        self.decoder5 = DecoderBlock(1024, 256)
        self.decoder6 = DecoderBlock(512, 128)
        self.decoder7 = DecoderBlock(256, 64)
        self.decoder8 = nn.ConvTranspose2d(128, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.tanh = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the generator."""
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)

        d1 = self.decoder1(e8)
        d1 = torch.cat([d1, e7], dim=1)
        d2 = self.decoder2(d1)
        d2 = torch.cat([d2, e6], dim=1)
        d3 = self.decoder3(d2)
        d3 = torch.cat([d3, e5], dim=1)
        d4 = self.decoder4(d3)
        d4 = torch.cat([d4, e4], dim=1)
        d5 = self.decoder5(d4)
        d5 = torch.cat([d5, e3], dim=1)
        d6 = self.decoder6(d5)
        d6 = torch.cat([d6, e2], dim=1)
        d7 = self.decoder7(d6)
        d7 = torch.cat([d7, e1], dim=1)
        d8 = self.decoder8(d7)
        output = self.tanh(d8)

        return output


class Pix2PixDiscriminator(nn.Module):
    """Pix2Pix Discriminator model for conditional GANs. Based on Pix2Pix paper."""
    def __init__(self) -> None:
        super().__init__()
        self.block1 = EncoderBlock(2, 64, norm=False)
        self.block2 = EncoderBlock(64, 128)
        self.block3 = EncoderBlock(128, 256)
        self.block4 = EncoderBlock(256, 512)
        self.conv = nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Forward pass through the discriminator."""
        x = torch.cat([x, condition], dim=1)    
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv(x)
        return x