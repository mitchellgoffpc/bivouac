# Adapted from https://github.com/mseitzer/pytorch-fid
import os
os.environ['MKL_NUM_THREADS'] = '1'  # sqrtm can be really slow without this, see https://github.com/scipy/scipy/issues/14594
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from helpers import CHECKPOINT_DIR, download_file

FID_INCEPTION_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"
FID_INCEPTION_PATH = CHECKPOINT_DIR / "inception.ckpt"

# Unfortunately the FID Inception model has a few differences from the torchvision model that require patching.
# NOTE: Doing it this way is 5-10% slower compared to overriding the entire forward method like pytorch-fid does,
# because we end up running branch_pool twice, but it's a lot less code and generating the samples is the real bottleneck.

class FIDInceptionA(models.inception.InceptionA):
    def _forward(self, x):
        b1x1, b5x5, b3x3dbl, _ = super()._forward(x)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)  # TF's avg pool doesn't include padding in the average
        branch_pool = self.branch_pool(branch_pool)
        return b1x1, b5x5, b3x3dbl, branch_pool

class FIDInceptionC(models.inception.InceptionC):
    def _forward(self, x):
        b1x1, b7x7, b7x7dbl, _ = super()._forward(x)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        return b1x1, b7x7, b7x7dbl, branch_pool

class FIDInceptionE_1(models.inception.InceptionE):
    def _forward(self, x):
        b1x1, b3x3, b3x3dbl, _ = super()._forward(x)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)
        return b1x1, b3x3, b3x3dbl, branch_pool

class FIDInceptionE_2(models.inception.InceptionE):
    def _forward(self, x):
        b1x1, b3x3, b3x3dbl, _ = super()._forward(x)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)  # FID Inception model uses max pooling instead of average pooling in this layer, this is probably a bug
        branch_pool = self.branch_pool(branch_pool)
        return b1x1, b3x3, b3x3dbl, branch_pool

class InceptionV3Features(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(num_classes=1008, aux_logits=False, init_weights=False)
        inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
        inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
        inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
        inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
        inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
        inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
        inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
        inception.Mixed_7b = FIDInceptionE_1(1280)
        inception.Mixed_7c = FIDInceptionE_2(2048)

        download_file(FID_INCEPTION_URL, FID_INCEPTION_PATH)
        inception.load_state_dict(torch.load(FID_INCEPTION_PATH, weights_only=True))

        self.layers = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3, nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3, nn.MaxPool2d(kernel_size=3, stride=2),
            inception.Mixed_5b, inception.Mixed_5c, inception.Mixed_5d,
            inception.Mixed_6a, inception.Mixed_6b, inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c, nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    @torch.no_grad
    def forward(self, x):
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = self.layers(2 * x - 1)  # scale inputs to [-1, 1])
        return x.squeeze((-1, -2))


# FID calculation

def calculate_activation_statistics(activations):
    mu = torch.mean(activations, dim=0)
    sigma = torch.cov(activations.T).double()
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = (mu1 - mu2).cpu()

    # product might be almost singular
    covmean = torch.as_tensor(scipy.linalg.sqrtm(sigma1.mm(sigma2).cpu().numpy()))
    if not torch.isfinite(covmean).all():
        print("fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps)
        offset = torch.eye(sigma1.shape[0]) * eps
        covmean = torch.as_tensor(scipy.linalg.sqrtm((sigma1 + offset).mm(sigma2 + offset).cpu().numpy()))

    # numerical error might give slight imaginary component
    if torch.is_complex(covmean):
        max_imag = torch.diagonal(covmean).imag.abs().max()
        if max_imag > 1e-3:
            raise ValueError(f"covmean matrix has imaginary component {max_imag}")
        covmean = covmean.real

    return diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(covmean)

def calculate_fid(activations1, activations2):
    mu1, sigma1 = calculate_activation_statistics(activations1)
    mu2, sigma2 = calculate_activation_statistics(activations2)
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2).item()


# CLI tool to calculate a model's FID score

def calculate_model_fid(rank, world_size):
    import time
    import torch
    import torch.distributed as dist
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from pathlib import Path
    from tqdm import tqdm

    from helpers import load_taming_data
    from models.vqvae import VQVAE
    from datasets.imagenet import ImageNetDataset

    # Initialize the distributed environment
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    # Load the model
    config, state_dict = load_taming_data()
    model = VQVAE(config).to(device).eval()
    model.load_state_dict(state_dict)

    # Load the ImageNet validation set
    transform = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor()])
    data_path = Path(__file__).parent.parent / "data/imagenet"
    dataset = ImageNetDataset(data_path, transform=transform, split="val")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=4, sampler=sampler)

    # Load the InceptionV3 model
    inception = InceptionV3Features().to(device).eval()

    # Calculate activations
    start_time = time.time()
    real_activations, recon_activations = [], []
    for images, _ in tqdm(dataloader, disable=rank != 0):
        images = images.to(device)
        with torch.no_grad():
            reconstructions, _ = model(images * 2 - 1)  # scale to [-1, 1]
        reconstructions = ((reconstructions + 1) / 2).clamp(0, 1)  # scale to [0, 1]
        real_activations.append(inception(images).cpu())
        recon_activations.append(inception(reconstructions).cpu())

    # Gather activations from all processes
    real_activations = torch.cat(real_activations).to(device)
    recon_activations = torch.cat(recon_activations).to(device)
    gathered_real = [torch.zeros_like(real_activations) for _ in range(world_size)]
    gathered_recon = [torch.zeros_like(recon_activations) for _ in range(world_size)]
    dist.all_gather(gathered_real, real_activations)
    dist.all_gather(gathered_recon, recon_activations)

    if rank == 0:
        fid_score = calculate_fid(torch.cat(gathered_real), torch.cat(gathered_recon))
        print(f"FID Score: {fid_score:.6f} | Eval Time: {time.time() - start_time:.1f}s")

    dist.destroy_process_group()


if __name__ == "__main__":
    import os
    import torch.multiprocessing as mp

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(calculate_model_fid, args=(world_size,), nprocs=world_size, join=True)
    else:
        calculate_model_fid(0, 1)