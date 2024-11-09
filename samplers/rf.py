import torch
from dataclasses import dataclass
from models.diffusion import Denoiser

@dataclass
class RFSamplerConfig:
    num_steps: int

def apply_noise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, *_ = x.size()
    t = torch.sigmoid(torch.randn((B, 1), device=x.device))
    x_0 = torch.randn_like(x)
    x_t = (1 - t) * x + t * x_0
    return x_t, x_0, t

@torch.no_grad()
def sample(denoiser: Denoiser, config: RFSamplerConfig, prev_obs: torch.Tensor, prev_act: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    device = prev_obs.device
    B, T, C, H, W = prev_obs.size()
    prev_obs = prev_obs.reshape(B, T * C, H, W)
    x = torch.randn(B, C, H, W, device=device)
    trajectory = [x]

    dt = torch.full((B, 1), 1.0 / config.num_steps, device=device)
    for i in range(num_steps):
        t = torch.full((B, 1), i / num_steps, device=device)
        v = denoiser(x, t, prev_obs, prev_act)
        x = x + dt * v
        trajectory.append(x)
    return x, trajectory
