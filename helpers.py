import torch
import requests
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

StateDict = dict[str, torch.Tensor]

CHECKPOINT_DIR = Path(__file__).parent / 'pretrained'
TAMING_MODEL_URL = 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1'
TAMING_CONFIG_URL = 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'


def download_file(url: str, path: Path, chunk_size: int = 128000, headers: dict[str, str] = {}) -> None:
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()
    tmp_path = Path(f'{path}.tmp')
    file_size = int(r.headers.get('content-length', 0))
    with open(tmp_path, 'wb') as f, tqdm(desc="Fetching " + url, total=file_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            f.write(chunk)
            pbar.update(chunk_size)
    tmp_path.rename(path)


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
        base_channels=ddconfig.ch,
        channel_mults=ddconfig.ch_mult,
        attentions=[ddconfig.resolution / (2**i) in ddconfig.attn_resolutions for i in range(len(ddconfig.ch_mult))],
        num_res_blocks=ddconfig.num_res_blocks,
        dropout=ddconfig.dropout,
        double_z=ddconfig.double_z)

    vqvae_config = VQVAEConfig(n_embed=config.model.params.n_embed, embed_dim=config.model.params.embed_dim, unet=unet_config)
    state_dict = torch.load(taming_model_path, map_location=torch.device('cpu'), weights_only=False)
    state_dict = {k: v for k, v in state_dict['state_dict'].items() if not k.startswith('loss.')}

    return vqvae_config, state_dict
