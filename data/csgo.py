#!/usr/bin/env python
import argparse
from pathlib import Path
from helpers import HUGGINGFACE_API_KEY, download_file

DEFAULT_PATH = Path(__file__).parent / 'csgo'
DATA_URLS = [f"https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch/resolve/main/hdf5_dm_july2021_{i + 1}_to_{min(i + 200, 5500)}.tar" for i in range(0, 5600, 200)]

def download_csgo(download_path):
    download_path.mkdir(parents=True, exist_ok=True)
    for url in DATA_URLS:
        local_filename = url.split('/')[-1]
        file_path = download_path / local_filename
        download_file(url, file_path, chunk_size=1024*1024, headers={'Authorization': f'Bearer {HUGGINGFACE_API_KEY}'})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and extract ImageNet dataset.')
    parser.add_argument('download_path', nargs='?', type=Path, default=DEFAULT_PATH, help='Path to download and extract the dataset')
    args = parser.parse_args()

    download_csgo(args.download_path)
