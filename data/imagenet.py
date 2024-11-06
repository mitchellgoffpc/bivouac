#!/usr/bin/env python
import argparse
import tarfile
from tqdm import tqdm
from pathlib import Path
from helpers import HUGGINGFACE_API_KEY, download_file

DEFAULT_PATH = Path(__file__).parent / 'imagenet'
DATA_URLS = {
    "train": [f"https://huggingface.co/datasets/imagenet-1k/resolve/1500f8c59b214ce459c0a593fa1c87993aeb7700/data/train_images_{i}.tar.gz" for i in range(5)],
    "val": ["https://huggingface.co/datasets/imagenet-1k/resolve/1500f8c59b214ce459c0a593fa1c87993aeb7700/data/val_images.tar.gz"],
}

def download_and_extract(url, destination):
    local_filename = url.split('/')[-1]
    file_path = destination.parent / local_filename
    download_file(url, file_path, chunk_size=1024*1024, headers={'Authorization': f'Bearer {HUGGINGFACE_API_KEY}'})

    with tarfile.open(file_path) as f:
        members = f.getmembers()
        for member in tqdm(members, desc=f"Extracting {file_path}", unit='file'):
            synset_id = member.name.split('_')[-1].split('.')[0]
            assert synset_id.startswith('n')
            if not (destination / synset_id / member.name).exists():
                f.extract(member, path=destination / synset_id)

def download_imagenet(download_path):
    for split, urls in DATA_URLS.items():
        destination = download_path / split
        destination.mkdir(parents=True, exist_ok=True)
        for url in urls:
            download_and_extract(url, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and extract ImageNet dataset.')
    parser.add_argument('download_path', nargs='?', type=Path, default=DEFAULT_PATH, help='Path to download and extract the dataset')
    args = parser.parse_args()

    download_imagenet(args.download_path)
