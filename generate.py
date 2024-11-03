import re
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
from pathlib import Path
from omegaconf import OmegaConf

from helpers import CHECKPOINT_DIR, download_file
from models.vqvae import VQVAEConfig, VQVAE
from models.unet import UNetConfig
from datasets.imagenet import ImageNetDataset

TAMING_MODEL_URL = 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1'
TAMING_CONFIG_URL = 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1'
TAMING_MODEL_PATH = CHECKPOINT_DIR / 'taming.ckpt'
TAMING_CONFIG_PATH = CHECKPOINT_DIR / 'taming.yaml'


if __name__ == '__main__':
    download_file(TAMING_MODEL_URL, TAMING_MODEL_PATH)
    download_file(TAMING_CONFIG_URL, TAMING_CONFIG_PATH)

    replacements = {
        r'loss.discriminator.main': r'loss.discriminator.layers',
        r'loss.perceptual_loss.scaling_layer': r'loss.perceptual_loss',
        r'loss.perceptual_loss.net.slice2.(\d+)': lambda m: f'loss.perceptual_loss.net.slice2.{int(m.group(1)) - 4}',
        r'loss.perceptual_loss.net.slice3.(\d+)': lambda m: f'loss.perceptual_loss.net.slice3.{int(m.group(1)) - 9}',
        r'loss.perceptual_loss.net.slice4.(\d+)': lambda m: f'loss.perceptual_loss.net.slice4.{int(m.group(1)) - 16}',
        r'loss.perceptual_loss.net.slice5.(\d+)': lambda m: f'loss.perceptual_loss.net.slice5.{int(m.group(1)) - 23}'}

    device = torch.device('cpu')
    checkpoint = torch.load(TAMING_MODEL_PATH, map_location=device, weights_only=False)
    state_dict = {k:v for k,v in checkpoint['state_dict'].items() if not k.startswith('loss.')}
    for old_key, new_key in replacements.items():
        for key in list(state_dict.keys()):
            if re.match(old_key, key):
                state_dict[re.sub(old_key, new_key, key)] = state_dict.pop(key)

    config = OmegaConf.load(TAMING_CONFIG_PATH)
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
    vqgan = VQVAE(vqvae_config).eval().to(device)
    vqgan.load_state_dict(state_dict)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # scale to [-1, 1]
    ])
    dataset = ImageNetDataset(Path('data/imagenet'), split='val', transform=transform)
    sample, _ = dataset[8]
    original = sample.permute(1, 2, 0).numpy()
    original = (original * 0.5 + 0.5).clip(0, 1)

    with torch.no_grad():
        result, _ = vqgan(sample[None].to(device))
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
