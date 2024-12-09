import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import MethodType
from dataclasses import dataclass, field
from models.unet import Encoder, Decoder, AttnBlock, UNetConfig

@dataclass
class DiffusionConfig:
    num_actions: int = 51
    num_cond_steps: int = 4
    unet: UNetConfig = field(default_factory=UNetConfig)


def replace_attn(module):
    for name, submodule in module.named_children():
        if isinstance(submodule, AttnBlock):
            submodule.forward = MethodType(weird_attn_forward, submodule)
        else:
            replace_attn(submodule)

def weird_attn_forward(self, x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    x = h = self.norm(x)  # This is weird, normally you don't normalize x. Might be a bug?
    q = self.q(h).view(B, self.num_heads, C // self.num_heads, H*W).permute(0, 1, 3, 2)
    k = self.k(h).view(B, self.num_heads, C // self.num_heads, H*W).permute(0, 1, 3, 2)
    v = self.v(h).view(B, self.num_heads, C // self.num_heads, H*W).permute(0, 1, 3, 2)
    h = F.scaled_dot_product_attention(q, k, v)
    h = h.permute(0, 1, 3, 2).reshape(B, C, H, W).contiguous()
    return x + self.out(h)


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
        self.config = config
        self.encoder = Encoder(config.unet)
        self.decoder = Decoder(config.unet)

        self.noise_emb = FourierFeatures(2048)
        self.noise_emb = FourierFeatures(config.unet.cond_channels)
        self.noise_cond_emb = FourierFeatures(config.unet.cond_channels)
        self.act_emb = nn.Sequential(
            nn.Embedding(config.num_actions, config.unet.cond_channels // config.num_cond_steps),
            nn.Flatten(),  # b t e -> b (t e)
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(config.unet.cond_channels, config.unet.cond_channels),
            nn.SiLU(),
            nn.Linear(config.unet.cond_channels, config.unet.cond_channels),
        )

        # Patches for DIAMOND
        replace_attn(self.encoder)
        replace_attn(self.decoder)
        for layer in self.encoder.down:
            layer.downsample.padding = (1,1,1,1)
        for layer in self.decoder.up:
            layer.upsample.padding = (1,1,1,1)

    def forward(self, noisy_next_obs: torch.Tensor, c_noise: torch.Tensor, c_noise_cond: torch.Tensor, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        assert act.ndim == 2 or (act.ndim == 3 and act.size(2) == self.act_emb[0].num_embeddings and set(act.unique().tolist()).issubset(set([0, 1])))
        act_emb = self.act_emb(act) if act.ndim == 2 else self.act_emb[1]((act.float() @ self.act_emb[0].weight))
        cond = self.cond_proj(self.noise_emb(c_noise) + self.noise_cond_emb(c_noise_cond) + act_emb)
        obs = torch.cat((obs, noisy_next_obs), dim=1)

        # NOTE: We have to do all the forward logic of the encoder/decoder ourselves because of this weird padding after the first conv
        x = self.encoder.conv_in(obs)
        *_, h, w = x.size()
        n = len(self.config.unet.block_channels) - 1
        padding_h = math.ceil(h / 2 ** n) * 2 ** n - h
        padding_w = math.ceil(w / 2 ** n) * 2 ** n - w
        x = F.pad(x, (0, padding_w, 0, padding_h))

        # encoder forward
        skip = []
        for layer in self.encoder.down:
            x, outs = layer(x, cond)
            skip.append(outs)
        x = self.encoder.mid(x, cond)

        # decoder forward
        x = self.decoder.mid(x, cond)
        for layer, skip in zip(self.decoder.up, skip[::-1]):
            x = layer(x, cond, skip)
        x = x[..., :h, :w]
        x = self.decoder.conv_out(F.silu(self.decoder.norm_out(x)))
        return x
