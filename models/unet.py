import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
from itertools import repeat
from functools import partial
from dataclasses import dataclass, field

def silu(x):
    return x*torch.sigmoid(x)
F.silu = silu

Count = Callable[[int], int]

@dataclass
class BlockConfig:
    num_layers: int
    in_channels: int
    out_channels: int
    cond_channels: int
    with_resample: bool
    with_attn: bool
    with_skip: bool
    dropout: float
    num_heads: Count
    num_groups: Count

@dataclass
class UNetConfig:
    in_channels: int = 3
    out_channels: int = 3
    cond_channels: int = 0
    block_channels: list[int] = field(default_factory=lambda: [128, 128, 256, 256, 512])
    block_attentions: list[bool] = field(default_factory=lambda: [False, False, False, False, True])
    layers_per_block: int = 2
    dropout: float = 0.0
    skip: bool = False

    @staticmethod
    def num_groups(channels: int) -> int:
        return 32

    @staticmethod
    def num_heads(channels: int) -> int:
        return 1

    def block(self, num_layers: int, in_channels: int, out_channels: int, with_resample: bool, with_attn: bool) -> BlockConfig:
        return BlockConfig(
            num_layers=num_layers, in_channels=in_channels, out_channels=out_channels, cond_channels=self.cond_channels,
            with_resample=with_resample, with_attn=with_attn, with_skip=self.skip,
            num_groups=self.num_groups, num_heads=self.num_heads, dropout=self.dropout)


# Normalization

class GroupNorm(nn.GroupNorm):
    def __init__(self, in_channels: int, num_groups: int):
        super().__init__(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        assert cond is None, "GroupNorm doesn't support conditioning input, use AdaGroupNorm instead"
        return super().forward(x)

class AdaGroupNorm(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int, num_groups: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = num_groups
        self.linear = nn.Linear(cond_channels, in_channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        assert x.size(1) == self.in_channels
        x = F.group_norm(x, self.num_groups, eps=1e-6)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


# Upsample / Downsample

class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))

class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.padding = (0,1,0,1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, self.padding, mode="constant", value=0))  # no asymmetric padding in torch conv, must do it ourselves


# Resnet / Attention Blocks

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_channels: int, num_groups: Count, dropout: float = 0.0):
        super().__init__()
        Normalize = partial(AdaGroupNorm, cond_channels=cond_channels) if cond_channels > 0 else GroupNorm
        self.norm1 = Normalize(in_channels, num_groups=num_groups(in_channels))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels, num_groups=num_groups(out_channels))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x, cond)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h, cond))))
        return h + self.shortcut(x)

class AttnBlock(nn.Module):
    def __init__(self, in_channels: int, num_heads: Count, num_groups: Count):
        super().__init__()
        self.num_heads = num_heads(in_channels)
        self.norm = GroupNorm(in_channels, num_groups(in_channels))
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).view(B, self.num_heads, C // self.num_heads, H*W).permute(0, 1, 3, 2)
        k = self.k(h).view(B, self.num_heads, C // self.num_heads, H*W).permute(0, 1, 3, 2)
        v = self.v(h).view(B, self.num_heads, C // self.num_heads, H*W).permute(0, 1, 3, 2)

        # hx = F.scaled_dot_product_attention(q, k, v)
        # hx = hx.permute(0, 1, 3, 2).reshape(B, C, H, W).contiguous()

        w = q @ k.permute(0,1,3,2)
        w = w * (int(C)**-0.5)
        w = F.softmax(w, dim=-1)
        h = w @ v
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W).contiguous()

        return x + self.out(h)


# Encoder / Decoder / Mid Blocks

class EncoderBlock(nn.Module):
    def __init__(self, config: BlockConfig):
        super().__init__()
        in_channels_list = [config.in_channels] + [config.out_channels] * (config.num_layers - 1)
        out_channels_list = [config.out_channels] * config.num_layers

        self.downsample = Downsample(config.in_channels) if config.with_resample else nn.Identity()
        self.block = nn.ModuleList()
        self.attn = nn.ModuleList()
        for in_channels, out_channels in zip(in_channels_list, out_channels_list):
            self.block.append(ResnetBlock(in_channels, out_channels, config.cond_channels, config.num_groups, dropout=config.dropout))
            self.attn.append(AttnBlock(out_channels, config.num_heads, config.num_groups) if config.with_attn else nn.Identity())

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> list[torch.Tensor]:
        x = self.downsample(x)
        outputs = [x]
        for block, attn in zip(self.block, self.attn):
            x = block(x, cond)
            x = attn(x)
            outputs.append(x)
        return x, outputs

class DecoderBlock(nn.Module):
    def __init__(self, config: BlockConfig):
        super().__init__()
        if config.with_skip:
            in_channels_list = [2 * config.in_channels] * (config.num_layers - 1) + [config.in_channels + config.out_channels]
            out_channels_list = [config.in_channels] * (config.num_layers - 1) + [config.out_channels]
        else:
            in_channels_list = [config.in_channels] + [config.out_channels] * (config.num_layers - 1)
            out_channels_list = [config.out_channels] * config.num_layers

        self.upsample = Upsample(config.in_channels) if config.with_resample else nn.Identity()
        self.block = nn.ModuleList()
        self.attn = nn.ModuleList()
        for in_channels, out_channels in zip(in_channels_list, out_channels_list):
            self.block.append(ResnetBlock(in_channels, out_channels, config.cond_channels, config.num_groups, dropout=config.dropout))
            self.attn.append(AttnBlock(out_channels, config.num_heads, config.num_groups) if config.with_attn else nn.Identity())

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor], skip: list[torch.Tensor] = []) -> torch.Tensor:
        x = self.upsample(x)
        for block, attn, skip in zip(self.block, self.attn, skip[::-1] or repeat(None)):
            x = torch.cat((x, skip), dim=1) if skip is not None else x
            x = block(x, cond)
            x = attn(x)
        return x

class MidBlock(nn.Module):
    def __init__(self, num_channels: int, cond_channels: int, num_heads: Count, num_groups: Count, dropout: float):
        super().__init__()
        self.block1 = ResnetBlock(num_channels, num_channels, cond_channels, num_groups, dropout=dropout)
        self.attn1 = AttnBlock(num_channels, num_heads, num_groups)
        self.block2 = ResnetBlock(num_channels, num_channels, cond_channels, num_groups, dropout=dropout)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.block1(x, cond)
        x = self.attn1(x)
        x = self.block2(x, cond)
        return x


# Encoder / Decoder

class Encoder(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        num_layers = config.layers_per_block
        channels = [config.block_channels[0], *config.block_channels]
        downsamples = [False] + [True] * (len(config.block_channels) - 1)
        block_configs = [config.block(num_layers, *args) for args in zip(channels[:-1], channels[1:], downsamples, config.block_attentions)]

        self.conv_in = nn.Conv2d(config.in_channels, channels[0], kernel_size=3, stride=1, padding=1)
        self.down = nn.ModuleList(EncoderBlock(block_config) for block_config in block_configs)
        self.mid = MidBlock(channels[-1], config.cond_channels, config.num_heads, config.num_groups, dropout=config.dropout)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        x = self.conv_in(x)
        outputs = []
        for layer in self.down:
            x, outs = layer(x, cond)
            outputs.append(outs)
        x = self.mid(x, cond)
        return x, outputs

class Decoder(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        num_layers = config.layers_per_block+1
        if config.skip:
            channels = [*config.block_channels[::-1], config.block_channels[0]]  # diamond uses symmetrical encoder/decoder channels
        else:
            channels = [config.block_channels[-1], *config.block_channels[::-1]]  # taming uses asymmetrical encoder/decoder channels
        upsamples = [False] + [True] * (len(config.block_channels) - 1)
        block_configs = [config.block(num_layers, *args) for args in zip(channels[:-1], channels[1:], upsamples, config.block_attentions[::-1])]

        self.mid = MidBlock(channels[0], config.cond_channels, config.num_heads, config.num_groups, dropout=config.dropout)
        self.up = nn.ModuleList(DecoderBlock(block_config) for block_config in block_configs)
        self.norm_out = GroupNorm(channels[-1], config.num_groups(channels[-1]))
        self.conv_out = nn.Conv2d(channels[-1], config.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None, skip: list[list[torch.Tensor]] = []) -> torch.Tensor:
        x = self.mid(x, cond)
        for layer, skip in zip(self.up, skip[::-1] or repeat([])):
            x = layer(x, cond, skip)
        return self.conv_out(F.silu(self.norm_out(x)))
