import torch
import torch.nn as nn
from einops import rearrange
from dataclasses import dataclass, field
from models.unet import Encoder, Decoder, UNetConfig

@dataclass
class VQVAEConfig:
    n_embed: int = 16384
    embed_dim: int = 256
    unet: UNetConfig = field(default_factory=UNetConfig)


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)

    def forward(self, z, beta=1.0):
        B, C, H, W = z.shape
        z = rearrange(z, 'bchw -> bhwc').contiguous()
        z_flattened = z.view(B*H*W, C)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd, dn -> bn', z_flattened, rearrange(self.embedding.weight, 'nd -> dn'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        loss = beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'bhwc -> bchw').contiguous()
        return z_q, loss


class VQVAE(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.encoder = Encoder(config.unet)
        self.decoder = Decoder(config.unet)
        self.quantize = VectorQuantizer(config.n_embed, config.embed_dim)
        self.quant_conv = nn.Conv2d(config.unet.z_channels, config.embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(config.embed_dim, config.unet.z_channels, kernel_size=1)

    def encode(self, x):
        x = self.encoder(x)
        x = self.quant_conv(x)
        return self.quantize(x, beta=1.0)

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)

    def forward(self, x):
        quant, diff = self.encode(x)
        dec = self.decode(quant)
        return dec, diff

    @property
    def last_layer(self):
        return self.decoder.conv_out
