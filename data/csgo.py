#!/usr/bin/env python
import argparse
import tarfile
from tqdm import tqdm
from pathlib import Path
from helpers import HUGGINGFACE_API_KEY, download_file

DEFAULT_PATH = Path(__file__).parent / 'csgo'
DATA_URLS = [f"https://huggingface.co/datasets/TeaPearce/CounterStrike_Deathmatch/resolve/main/hdf5_dm_july2021_{i + 1}_to_{min(i + 200, 5500)}.tar" for i in range(0, 5600, 200)]

def download_and_extract(url, destination):
    local_filename = url.split('/')[-1]
    file_path = destination / 'tar' / local_filename
    download_file(url, file_path, chunk_size=1024*1024, headers={'Authorization': f'Bearer {HUGGINGFACE_API_KEY}'})

    with tarfile.open(file_path) as f:
        members = f.getmembers()
        for member in tqdm(members, desc=f"Extracting {file_path}", unit='file'):
            if not (destination / 'hdf' / member.name).exists():
                f.extract(member, path=destination / 'hdf')

def download_csgo(download_path):
    (download_path / 'tar').mkdir(parents=True, exist_ok=True)
    (download_path / 'hdf').mkdir(parents=True, exist_ok=True)
    for url in DATA_URLS:
        download_and_extract(url, download_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and extract ImageNet dataset.')
    parser.add_argument('download_path', nargs='?', type=Path, default=DEFAULT_PATH, help='Path to download and extract the dataset')
    args = parser.parse_args()

    download_csgo(args.download_path)
