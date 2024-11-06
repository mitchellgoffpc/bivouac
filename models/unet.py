import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class UNetConfig:
    in_channels: int = 3
    z_channels: int = 256
    block_channels: list[int] = field(default_factory=lambda: [128, 128, 256, 256, 512])
    block_attentions: list[bool] = field(default_factory=lambda: [False, False, False, False, True])
    layers_per_block: int = 2
    dropout: float = 0.0
    double_z: bool = False


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0,1,0,0))
    return emb

def Normalize(in_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# Upsample / Downsample

class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1) if with_conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0) if with_conv else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv:
            return self.conv(nn.functional.pad(x, (0,1,0,1), mode="constant", value=0))  # no asymmetric padding in torch conv, must do it ourselves
        else:
            return nn.functional.avg_pool2d(x, kernel_size=2, stride=2)


# Resnet / Attention Blocks

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temb_channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else nn.Identity()
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        if temb is not None:
            h = h + self.temb_proj(F.silu(temb))[:,:,None,None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.nin_shortcut(x)

class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, C, H*W).permute(0, 2, 1)
        k = self.k(h).reshape(B, C, H*W).permute(0, 2, 1)
        v = self.v(h).reshape(B, C, H*W).permute(0, 2, 1)

        h = F.scaled_dot_product_attention(q, k, v)
        h = h.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        h = self.proj_out(h)
        return x + h


# Encoder / Decoder / Mid Blocks

class EncoderBlock(nn.Module):
    def __init__(self, num_blocks: int, in_channels: int, out_channels: int, with_attn: bool, with_downsample: bool, dropout: float):
        super().__init__()
        self.block = nn.ModuleList()
        self.attn = nn.ModuleList()
        for _ in range(num_blocks):
            self.block.append(ResnetBlock(in_channels, out_channels, temb_channels=0, dropout=dropout))
            self.attn.append(AttnBlock(out_channels) if with_attn else nn.Identity())
            in_channels = out_channels
        self.downsample = Downsample(out_channels, with_conv=True) if with_downsample else None

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor]) -> list[torch.Tensor]:
        outputs = []
        for block, attn in zip(self.block, self.attn):
            x = block(x, temb)
            x = attn(x)
            outputs.append(x)
        if self.downsample:
            outputs.append(self.downsample(x))
        return outputs

class DecoderBlock(nn.Module):
    def __init__(self, num_blocks: int, in_channels: int, out_channels: int, with_attn: bool, with_upsample: bool, dropout: float):
        super().__init__()
        self.block = nn.ModuleList()
        self.attn = nn.ModuleList()
        for _ in range(num_blocks):
            self.block.append(ResnetBlock(in_channels, out_channels, temb_channels=0, dropout=dropout))
            self.attn.append(AttnBlock(out_channels) if with_attn else nn.Identity())
            in_channels = out_channels
        self.upsample = Upsample(out_channels, with_conv=True) if with_upsample else None

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        for block, attn in zip(self.block, self.attn):
            x = block(x, temb)
            x = attn(x)
        if self.upsample:
            x = self.upsample(x)
        return x

class MidBlock(nn.Module):
    def __init__(self, num_channels: int, temb_channels: int, dropout: float):
        super().__init__()
        self.block_1 = ResnetBlock(num_channels, num_channels, temb_channels=temb_channels, dropout=dropout)
        self.attn_1 = AttnBlock(num_channels)
        self.block_2 = ResnetBlock(num_channels, num_channels, temb_channels=temb_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.block_1(x, temb)
        x = self.attn_1(x)
        x = self.block_2(x, temb)
        return x


# Encoder / Decoder

class Encoder(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        channels = [config.block_channels[0], *config.block_channels]
        downsamples = [True] * (len(config.block_channels)-1) + [False]
        attentions = config.block_attentions

        self.conv_in = nn.Conv2d(config.in_channels, channels[0], kernel_size=3, stride=1, padding=1)
        self.down = nn.ModuleList(EncoderBlock(config.layers_per_block, *args, config.dropout) for args in zip(channels[:-1], channels[1:], attentions, downsamples))
        self.mid = MidBlock(channels[-1], temb_channels=0, dropout=config.dropout)
        self.norm_out = Normalize(channels[-1])
        self.conv_out = nn.Conv2d(channels[-1], config.z_channels * (2 if config.double_z else 1), kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hs = [self.conv_in(x)]
        for layer in self.down:
            hs.extend(layer(hs[-1], temb))
        x = self.mid(hs[-1], temb)
        return self.conv_out(F.silu(self.norm_out(x)))

class Decoder(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        channels = [config.block_channels[-1], *config.block_channels[::-1]]
        upsamples = [True] * (len(config.block_channels) - 1) + [False]
        attentions = config.block_attentions[::-1]

        self.conv_in = nn.Conv2d(config.z_channels, channels[0], kernel_size=3, stride=1, padding=1)
        self.mid = MidBlock(channels[0], temb_channels=0, dropout=config.dropout)
        self.up = nn.ModuleList(DecoderBlock(config.layers_per_block+1, *args, config.dropout) for args in zip(channels[:-1], channels[1:], attentions, upsamples))
        self.norm_out = Normalize(channels[-1])
        self.conv_out = nn.Conv2d(channels[-1], config.in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.conv_in(z)
        x = self.mid(x, temb)
        for layer in self.up:
            x = layer(x, temb)
        return self.conv_out(F.silu(self.norm_out(x)))
