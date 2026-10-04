import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm3d(out_channels),

            nn.ReLU(inplace=True)
        )


    def forward(self, x):

        return self.block(x)


class UNet3D(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder

        self.enc1 = ConvBlock(1, 16)

        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = ConvBlock(16, 32)

        self.pool2 = nn.MaxPool3d(2)

        # Bottleneck

        self.bottleneck = ConvBlock(32, 64)

        # Decoder

        self.up2 = nn.ConvTranspose3d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec2 = ConvBlock(64, 32)

        self.up1 = nn.ConvTranspose3d(
            32,
            16,
            kernel_size=2,
            stride=2
        )

        self.dec1 = ConvBlock(32, 16)

        # Output

        self.final = nn.Conv3d(
            16,
            1,
            kernel_size=1
        )


    def forward(self, x):

        # Encoder

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool1(e1)
        )

        # Bottleneck

        b = self.bottleneck(
            self.pool2(e2)
        )

        # Decoder

        d2 = self.up2(b)

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(d2)

        d1 = self.up1(d2)

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(d1)

        # Output

        return self.final(d1)