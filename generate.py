import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from pathlib import Path

from helpers import load_taming_data, load_sd3_data, load_model
from datasets.imagenet import ImageNetDataset


if __name__ == '__main__':
    device = torch.device('cuda')
    # config, state_dict = load_taming_data()
    config, state_dict = load_sd3_data()
    model = load_model(config, state_dict, device)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # scale to [-1, 1]
    ])
    dataset = ImageNetDataset(Path('data/imagenet'), split='val', transform=transform)
    sample, _ = dataset[8]
    original = sample.permute(1, 2, 0).numpy()
    original = (original * 0.5 + 0.5).clip(0, 1)

    with torch.no_grad():
        result, _ = model(sample[None].to(device))
        result = result[0].cpu().permute(1, 2, 0).numpy()
        result = (result * 0.5 + 0.5).clip(0, 1)

    _, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(original)
    ax[0].set_title('Original Sample')
    ax[0].axis('off')

    ax[1].imshow(result)
    ax[1].set_title('VQGAN Result')
    ax[1].axis('off')

    plt.tight_layout()
    plt.savefig('samples.png')
    plt.show()
