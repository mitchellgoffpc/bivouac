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
CSGO_MODEL_URL = 'https://huggingface.co/eloialonso/diamond/resolve/main/csgo/model/csgo.pt'
CSGO_CONFIG_URL = 'https://huggingface.co/eloialonso/diamond/resolve/main/csgo/config/agent/csgo.yaml'


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


# Load pretrained models

def load_taming_data() -> tuple['VQVAEConfig', StateDict]:
    import sys, types  # spoof pytorch lightning for unpickler
    sys.modules["pytorch_lightning"] = True  # Can be anything
    sys.modules["pytorch_lightning.callbacks.model_checkpoint"] = types.SimpleNamespace(ModelCheckpoint=None)

    from models.vqvae import VQVAEConfig, UNetConfig
    taming_model_path = CHECKPOINT_DIR / 'taming.ckpt'
    taming_config_path = CHECKPOINT_DIR / 'taming.yaml'
    download_file(TAMING_MODEL_URL, taming_model_path)
    download_file(TAMING_CONFIG_URL, taming_config_path)

    config = OmegaConf.load(taming_config_path)
    ddconfig = config.model.params.ddconfig
    unet_config = UNetConfig(
        block_channels=[ddconfig.ch * mult for mult in ddconfig.ch_mult],
        block_attentions=[ddconfig.resolution / (2**i) in ddconfig.attn_resolutions for i in range(len(ddconfig.ch_mult))],
        layers_per_block=ddconfig.num_res_blocks,
        dropout=ddconfig.dropout)
    vqvae_config = VQVAEConfig(n_embed=config.model.params.n_embed, embed_dim=config.model.params.embed_dim, z_channels=ddconfig.z_channels, unet=unet_config)

    replacements = {
        r'up\.(\d)': lambda m: f'up.{len(ddconfig.ch_mult) - 1 - int(m.group(1))}',
        r'down\.(\d)\.downsample': lambda m: f'down.{int(m.group(1)) + 1}.downsample',
        r'up\.(\d)\.upsample': lambda m: f'up.{int(m.group(1)) + 1}.upsample',
        r'nin_shortcut': 'shortcut', r'proj_out': 'out',
        r'mid\.(block|attn)_(\d)': r'mid.\1\2',
        r'^quant_conv': 'conv2_z', r'post_quant_conv': 'conv1_post_z',
        r'encoder\.norm_out': 'norm_z', r'encoder\.conv_out': 'conv1_z', r'decoder\.conv_in': 'conv2_post_z'}

    state_dict = torch.load(taming_model_path, map_location=torch.device('cpu'), weights_only=False)
    state_dict = {k: v for k, v in state_dict['state_dict'].items() if not k.startswith('loss.')}
    state_dict = fix_state_dict(state_dict, replacements)
    return vqvae_config, state_dict


def load_sd3_data() -> tuple['VAEConfig', StateDict]:
    from models.vae import VAEConfig, UNetConfig
    sd3_model_path = CHECKPOINT_DIR / 'sd3.safetensors'
    sd3_config_path = CHECKPOINT_DIR / 'sd3.json'
    download_file(SD3_MODEL_URL, sd3_model_path, headers={'Authorization': f'Bearer {HUGGINGFACE_API_KEY}'})
    download_file(SD3_CONFIG_URL, sd3_config_path, headers={'Authorization': f'Bearer {HUGGINGFACE_API_KEY}'})

    config = json.loads(Path(sd3_config_path).read_text())
    unet_config = UNetConfig(
        block_channels=config['block_out_channels'],
        block_attentions=[False] * len(config['block_out_channels']),
        layers_per_block=config['layers_per_block'])
    vae_config = VAEConfig(z_channels=config['latent_channels'], unet=unet_config)

    replacements = {
        r'up_blocks': 'up', r'down_blocks': 'down', r'mid_block': 'mid',
        r'resnets': 'block', r'attentions': 'attn',
        r'downsamplers\.0': 'downsample', r'upsamplers\.0': 'upsample',
        r'down\.(\d)\.downsample': lambda m: f'down.{int(m.group(1)) + 1}.downsample',
        r'up\.(\d)\.upsample': lambda m: f'up.{int(m.group(1)) + 1}.upsample',
        r'to_(q|k|v)': r'\1', r'to_out\.0': 'out', r'group_norm': 'norm',
        r'mid\.attn\.0': 'mid.attn1', r'mid\.block\.0': 'mid.block1', r'mid\.block\.1': 'mid.block2',
        r'conv_shortcut': 'shortcut', r'conv_norm_out': 'norm_out',
        r'encoder\.norm_out': 'norm_z', r'encoder\.conv_out': 'conv_z', r'decoder\.conv_in': 'conv_post_z'}

    with safetensors.safe_open(sd3_model_path, framework="pt") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    state_dict = fix_state_dict(state_dict, replacements)
    state_dict = {k: v[..., None, None] if re.search(r'attn1\.(q|k|v|out)\.weight', k) else v for k, v in state_dict.items()}
    return vae_config, state_dict


def load_csgo_data() -> tuple['DiffusionConfig', StateDict]:
    from models.diffusion import DiffusionConfig, UNetConfig
    csgo_model_path = CHECKPOINT_DIR / 'csgo.ckpt'
    csgo_config_path = CHECKPOINT_DIR / 'csgo.yaml'
    download_file(CSGO_MODEL_URL, csgo_model_path)
    download_file(CSGO_CONFIG_URL, csgo_config_path)

    replacements = {
        r'denoiser\.': '',
        r'inner_model\.': '',
        r'unet\.d_blocks\.': 'encoder.down.',
        r'unet\.u_blocks\.': 'decoder.up.',
        r'unet\.mid_blocks\.resblocks\.0\.attn\.': 'encoder.mid.attn1.',
        r'unet\.mid_blocks\.resblocks\.1\.attn\.': 'decoder.mid.attn1.',
        r'unet\.mid_blocks\.resblocks\.0\.': 'encoder.mid.block1.',
        r'unet\.mid_blocks\.resblocks\.1\.': 'decoder.mid.block1.',
        r'unet\.downsamples\.(\d)\.': r'encoder.down.\1.downsample.',
        r'unet\.upsamples\.(\d)\.': r'decoder.up.\1.upsample.',
        r'\.resblocks\.': '.block.',
        r'\.out_proj\.': '.out.',
        r'\.block\.(\d)\.proj\.': r'.block.\1.shortcut.',
        r'\.block\.(\d)\.attn\.': r'.attn.\1.',
        r'\.norm\.norm\.': '.norm.',
        r'^conv_in': 'encoder.conv_in',
        r'^conv_out': 'decoder.conv_out',
        r'^norm_out\.norm': 'decoder.norm_out'}

    config = OmegaConf.load(csgo_config_path).denoiser.inner_model
    unet_config = UNetConfig(
        in_channels=config.img_channels * (config.num_steps_conditioning + 1),
        out_channels=config.img_channels,
        cond_channels=config.cond_channels,
        block_channels = config.channels,
        block_attentions=config.attn_depths,
        layers_per_block=config.depths[0],
        num_groups=lambda ch: max(1, ch // 32),
        num_heads=lambda ch: max(1, ch // 8),
        skip=True)
    diffusion_config = DiffusionConfig(unet=unet_config)

    state_dict = torch.load(csgo_model_path, map_location=torch.device('cpu'), weights_only=True)
    state_dict = {k:v for k,v in state_dict.items() if k.startswith('denoiser.')}
    for pattern, replacement in replacements.items():
        state_dict = {re.sub(pattern, replacement, k): v for k, v in state_dict.items()}
    for key in list(state_dict.keys()):
        if '.qkv_proj.' in key:
            q, k, v = state_dict.pop(key).chunk(3, dim=0)
            state_dict[key.replace('.qkv_proj.', '.q.')] = q
            state_dict[key.replace('.qkv_proj.', '.k.')] = k
            state_dict[key.replace('.qkv_proj.', '.v.')] = v
        if 'mid.block1' in key:
            state_dict[key.replace('mid.block1', 'mid.block2')] = torch.zeros_like(state_dict[key])

    return diffusion_config, state_dict


# Load model + config

def load_model_data(checkpoint: str) -> tuple[Union['VAEConfig', 'VQVAEConfig', 'DiffusionConfig'], StateDict]:
    if checkpoint == 'taming':
        return load_taming_data()
    elif checkpoint == 'sd3':
        return load_sd3_data()
    elif checkpoint == 'csgo':
        return load_csgo_data()

    from models.vqvae import VQVAEConfig  # TODO: We should allow other types of configs here
    checkpoint_path = Path(checkpoint)
    config_path = checkpoint_path.parent.parent / 'config.yaml'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    schema = OmegaConf.structured(VQVAEConfig)
    config = OmegaConf.merge(schema, OmegaConf.load(config_path).model)
    config = OmegaConf.to_object(config)
    with safetensors.safe_open(checkpoint_path, framework="pt") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    return config, state_dict

def load_model(config: Union['VAEConfig', 'VQVAEConfig', 'DiffusionConfig'], state_dict: StateDict, device: torch.device):
    from models.vae import VAE, VAEConfig
    from models.vqvae import VQVAE, VQVAEConfig
    from models.diffusion import Denoiser, DiffusionConfig
    models = {VQVAEConfig: VQVAE, VAEConfig: VAE, DiffusionConfig: Denoiser}
    model = models[type(config)](config).eval().to(device)
    model.load_state_dict(state_dict)
    return model
