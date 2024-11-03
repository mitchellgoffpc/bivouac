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
