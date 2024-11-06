import torch
import torch.nn as nn
from dataclasses import dataclass, field
from models.unet import Encoder, Decoder, UNetConfig

@dataclass
class VAEConfig:
    unet: UNetConfig = field(default_factory=UNetConfig)

class VAE(nn.Module):
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.encoder = Encoder(config.unet)
        self.decoder = Decoder(config.unet)

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = x.chunk(2, dim=1)
        return mean, logvar

    def decode(self, quant):
        return self.decoder(quant)

    def forward(self, x, sample=True):
        mean, logvar = self.encode(x)
        if sample:
            mean = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        dec = self.decode(mean)
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return dec, kl_div

    @property
    def last_layer(self):
        return self.decoder.conv_out
