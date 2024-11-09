import torch
import torch.nn as nn
from dataclasses import dataclass, field
from models.unet import Encoder, Decoder, UNetConfig

@dataclass
class DiffusionConfig:
    unet: UNetConfig = field(default_factory=UNetConfig)


class FourierFeatures(nn.Module):
    def __init__(self, cond_channels: int) -> None:
        super().__init__()
        assert cond_channels % 2 == 0
        self.register_buffer("weight", torch.randn(1, cond_channels // 2))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 1
        f = 2 * math.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)

class Denoiser(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.noise_emb = FourierFeatures(2048)
        self.encoder = Encoder(config.unet)
        self.decoder = Decoder(config.unet)

    def encode(self, x):
        x = self.encoder(x)
        return x

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x
