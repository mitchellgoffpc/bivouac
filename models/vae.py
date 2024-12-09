import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from models.unet import Encoder, Decoder, GroupNorm, UNetConfig

@dataclass
class VAEConfig:
    z_channels: int = 16
    unet: UNetConfig = field(default_factory=UNetConfig)


class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        mid_channels = config.unet.block_channels[-1]

        self.encoder = Encoder(config.unet)
        self.decoder = Decoder(config.unet)
        self.norm_z = GroupNorm(mid_channels, config.unet.num_groups(mid_channels))
        self.conv_z = nn.Conv2d(mid_channels, config.z_channels * 2, kernel_size=3, stride=1, padding=1)
        self.conv_post_z = nn.Conv2d(config.z_channels, mid_channels, kernel_size=3, stride=1, padding=1)

    def encode(self, x):
        x, _ = self.encoder(x)
        x = self.conv_z(F.silu(self.norm_z(x)))
        mean, logvar = x.chunk(2, dim=1)
        return mean, logvar

    def decode(self, z):
        x = self.conv_post_z(z)
        return self.decoder(x)

    def forward(self, x, sample=True):
        mean, logvar = self.encode(x)
        if sample:
            mean = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        x = self.decode(mean)
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return x, kl_div

    @property
    def last_layer(self):
        return self.decoder.conv_out
