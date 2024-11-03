from pathlib import Path
from typing import Callable
from torchvision.datasets import ImageFolder

class ImageNetDataset(ImageFolder):
    def __init__(self, data_dir: Path, transform: Callable, split: str = 'train'):
        assert split in ('train', 'val', 'test'), f"Invalid split `{split}`"
        assert (data_dir / split).is_dir(), f"Data for {split} set does not exist yet, run `data/imagenet.py` to fetch it."
        super().__init__(root=data_dir / split, transform=transform)
