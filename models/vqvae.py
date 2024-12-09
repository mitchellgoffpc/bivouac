import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from dataclasses import dataclass, field
from models.unet import Encoder, Decoder, GroupNorm, UNetConfig

@dataclass
class VQVAEConfig:
    n_embed: int = 16384
    embed_dim: int = 256
    z_channels: int = 256
    unet: UNetConfig = field(default_factory=UNetConfig)


class VectorQuantizer(nn.Module):
    def __init__(self, n_embed, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)

    def forward(self, z, beta=1.0):
        B, C, H, W = z.shape
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(B*H*W, C)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('b d, d n -> b n', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        loss = torch.mean((z_q.detach()-z)**2) + beta * torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        return z_q, loss



class VQVAE(nn.Module):
    def __init__(self, config: VQVAEConfig):
        super().__init__()
        mid_channels = config.unet.block_channels[-1]
        z_channels = config.z_channels

        self.encoder = Encoder(config.unet)
        self.norm_z = GroupNorm(mid_channels, config.unet.num_groups(mid_channels))
        self.conv1_z = nn.Conv2d(mid_channels, z_channels, kernel_size=3, stride=1, padding=1)
        self.conv2_post_z = nn.Conv2d(z_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.decoder = Decoder(config.unet)
        self.quantize = VectorQuantizer(config.n_embed, config.embed_dim)

        self.conv2_z = nn.Conv2d(z_channels, config.embed_dim, kernel_size=1)
        self.conv1_post_z = nn.Conv2d(config.embed_dim, z_channels, kernel_size=1)

    def encode(self, x):
        x, _ = self.encoder(x)
        x = self.conv1_z(F.silu(self.norm_z(x)))
        x = self.conv2_z(x)
        return self.quantize(x, beta=0.25)

    def decode(self, z):
        x = self.conv1_post_z(z)
        x = self.conv2_post_z(x)
        return self.decoder(x)

    def forward(self, x):
        z, vq_loss = self.encode(x)
        x = self.decode(z)
        return x, vq_loss

    @property
    def last_layer(self):
        return self.decoder.conv_out
