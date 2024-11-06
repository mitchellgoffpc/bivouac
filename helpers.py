import os
import re
import json
import torch
import requests
import safetensors
from tqdm import tqdm
from typing import Union
from pathlib import Path
from omegaconf import OmegaConf
from tenacity import retry, stop_after_attempt, retry_if_exception_type

StateDict = dict[str, torch.Tensor]

HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
CHECKPOINT_DIR = Path(__file__).parent / 'pretrained'

TAMING_MODEL_URL = 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1'
TAMING_CONFIG_URL = 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'
SD3_MODEL_URL = 'https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/resolve/main/vae/diffusion_pytorch_model.safetensors'
SD3_CONFIG_URL = 'https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/resolve/main/vae/config.json'


@retry(retry=retry_if_exception_type(requests.exceptions.ChunkedEncodingError), stop=stop_after_attempt(3), reraise=True)
def download_file(url: str, path: Path, chunk_size: int = 128000, headers: dict[str, str] = {}) -> None:
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()
    tmp_path = Path(f'{path}.tmp')
    file_size = int(r.headers.get('content-length', 0))
    with open(tmp_path, 'wb') as f, tqdm(desc=f"Fetching {url}", total=file_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            pbar.update(len(chunk))
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    tmp_path.rename(path)

def fix_state_dict(state_dict: StateDict, replacements: dict[str, str]) -> StateDict:
    for pattern, replacement in replacements.items():
        state_dict = {re.sub(pattern, replacement, k): v for k, v in state_dict.items()}
    return state_dict

def load_model(config: Union['VAEConfig', 'VQVAEConfig', 'DiffusionConfig'], state_dict: StateDict, device: torch.device):
    from models.vae import VAE, VAEConfig
    from models.vqvae import VQVAE, VQVAEConfig
    from models.diffusion import Denoiser, DiffusionConfig
    torch.set_default_device(device)
    models = {VQVAEConfig: VQVAE, VAEConfig: VAE, DiffusionConfig: Denoiser}
    model = models[type(config)](config).eval()
    model.load_state_dict(state_dict)
    return model


# Load model data

def load_taming_data() -> tuple['VQVAEConfig', StateDict]:
    from models.vqvae import VQVAEConfig, UNetConfig
    taming_model_path = CHECKPOINT_DIR / 'taming.ckpt'
    taming_config_path = CHECKPOINT_DIR / 'taming.yaml'
    download_file(TAMING_MODEL_URL, taming_model_path)
    download_file(TAMING_CONFIG_URL, taming_config_path)

    config = OmegaConf.load(taming_config_path)
    ddconfig = config.model.params.ddconfig
    unet_config = UNetConfig(
        in_channels=ddconfig.in_channels,
        z_channels=ddconfig.z_channels,
        block_channels=[ddconfig.ch * mult for mult in ddconfig.ch_mult],
        block_attentions=[ddconfig.resolution / (2**i) in ddconfig.attn_resolutions for i in range(len(ddconfig.ch_mult))],
        layers_per_block=ddconfig.num_res_blocks,
        dropout=ddconfig.dropout,
        double_z=ddconfig.double_z)
    vqvae_config = VQVAEConfig(n_embed=config.model.params.n_embed, embed_dim=config.model.params.embed_dim, unet=unet_config)

    state_dict = torch.load(taming_model_path, map_location=torch.device('cpu'), weights_only=False)
    state_dict = {k: v for k, v in state_dict['state_dict'].items() if not k.startswith('loss.')}
    state_dict = fix_state_dict(state_dict, {r'up\.(\d)': lambda m: f'up.{len(ddconfig.ch_mult) - 1 - int(m.group(1))}'})
    return vqvae_config, state_dict


def load_sd3_data() -> tuple['VAEConfig', StateDict]:
    from models.vae import VAEConfig, UNetConfig
    sd3_model_path = CHECKPOINT_DIR / 'sd3.safetensors'
    sd3_config_path = CHECKPOINT_DIR / 'sd3.json'
    download_file(SD3_MODEL_URL, sd3_model_path, headers={'Authorization': f'Bearer {HUGGINGFACE_API_KEY}'})
    download_file(SD3_CONFIG_URL, sd3_config_path, headers={'Authorization': f'Bearer {HUGGINGFACE_API_KEY}'})

    config = json.loads(Path(sd3_config_path).read_text())
    unet_config = UNetConfig(
        in_channels=config['in_channels'],
        z_channels=config['latent_channels'],
        block_channels=config['block_out_channels'],
        block_attentions=[False] * len(config['block_out_channels']),
        layers_per_block=config['layers_per_block'],
        double_z=True)
    vae_config = VAEConfig(unet=unet_config)

    replacements = {
        r'up_blocks': 'up', r'down_blocks': 'down', r'mid_block': 'mid',
        r'resnets': 'block', r'attentions': 'attn',
        r'downsamplers\.0': 'downsample', r'upsamplers\.0': 'upsample',
        r'to_(q|k|v)': r'\1', r'to_out\.0': 'proj_out', r'group_norm': 'norm',
        r'mid\.attn\.0': 'mid.attn_1', r'mid\.block\.0': 'mid.block_1', r'mid\.block\.1': 'mid.block_2',
        r'conv_shortcut': 'nin_shortcut', r'conv_norm_out': 'norm_out'}

    with safetensors.safe_open(sd3_model_path, framework="pt") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    state_dict = fix_state_dict(state_dict, replacements)
    state_dict = {k: v[..., None, None] if re.search(r'attn_1\.(q|k|v|proj_out)\.weight', k) else v for k, v in state_dict.items()}
    return vae_config, state_dict
