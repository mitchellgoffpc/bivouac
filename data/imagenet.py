#!/usr/bin/env python
import argparse
import requests
import tarfile
from tqdm import tqdm
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / 'imagenet'
DATA_URLS = {
    "train": [f"https://huggingface.co/datasets/imagenet-1k/resolve/1500f8c59b214ce459c0a593fa1c87993aeb7700/data/train_images_{i}.tar.gz" for i in range(5)],
    "val": ["https://huggingface.co/datasets/imagenet-1k/resolve/1500f8c59b214ce459c0a593fa1c87993aeb7700/data/val_images.tar.gz"],
}

def download_and_extract(url, destination, token, chunk_size=1024*1024):
    local_filename = url.split('/')[-1]
    file_path = destination.parent / local_filename

    # Check if the file exists before downloading it
    if not file_path.exists():
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with open(file_path, 'wb') as f, tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                pbar.update(len(chunk))
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    # Extract the file
    with tarfile.open(file_path) as file:
        members = file.getmembers()
        for member in tqdm(members, unit='file'):
            synset_id = member.name.split('_')[-1].split('.')[0]
            assert synset_id.startswith('n')
            if not (destination / synset_id / member.name).exists():
                file.extract(member, path=destination / synset_id)

def download_imagenet(download_path):
    token = input('To download the imagenet dataset from Huggingface, enter your Huggingface access token: ')
    for split, urls in DATA_URLS.items():
        destination = download_path / split
        destination.mkdir(parents=True, exist_ok=True)
        for url in urls:
            print(f"Downloading and extracting {url}")
            download_and_extract(url, destination, token)
            print(f"Finished downloading and extracting {url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and extract ImageNet dataset.')
    parser.add_argument('download_path', nargs='?', type=Path, default=DEFAULT_PATH, help='Path to download and extract the dataset')
    args = parser.parse_args()

    download_imagenet(args.download_path)
