import torch
from dataclasses import dataclass
from models.diffusion import Denoiser

@dataclass
class KerrasSamplerConfig:
    num_steps_denoising: int
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float("inf")
    s_noise: float = 1

@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float
    sigma_offset_noise: float


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: int, device: torch.device) -> torch.Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))

def apply_noise(x: torch.Tensor, config: SigmaDistributionConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, C, _, _ = x.shape
    device = x.device
    s = torch.randn(B, device=device) * config.scale + config.loc
    sigma = s.exp().clip(config.sigma_min, config.sigma_max)
    offset_noise = config.sigma_offset_noise * torch.randn(B, C, 1, 1, device=device)
    noisy_next_obs = x + offset_noise + torch.randn_like(x) * sigma.view(sigma.shape + [1] * (x.ndim - sigma.ndim))
    return noisy_next_obs, offset_noise, sigma

@torch.no_grad()
def sample(denoiser: Denoiser, config: KerrasSamplerConfig, prev_obs: torch.Tensor, prev_act: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    sigmas = build_sigmas(config.num_steps_denoising, config.sigma_min, config.sigma_max, config.rho, denoiser.device)
    device = prev_obs.device
    B, T, C, H, W = prev_obs.size()
    prev_obs = prev_obs.reshape(B, T * C, H, W)
    s_in = torch.ones(B, device=device)
    gamma_ = min(config.s_churn / (len(sigmas) - 1), 2**0.5 - 1)
    x = torch.randn(B, C, H, W, device=device)
    trajectory = [x]
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        gamma = gamma_ if config.s_tmin <= sigma <= config.s_tmax else 0
        sigma_hat = sigma * (gamma + 1)
        if gamma > 0:
            eps = torch.randn_like(x) * config.s_noise
            x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
        denoised = denoiser.denoise(x, sigma, prev_obs, prev_act)
        d = (x - denoised) / sigma_hat
        dt = next_sigma - sigma_hat
        if config.order == 1 or next_sigma == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser.denoise(x_2, next_sigma * s_in, prev_obs, prev_act)
            d_2 = (x_2 - denoised_2) / next_sigma
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
        trajectory.append(x)
    return x, trajectory
