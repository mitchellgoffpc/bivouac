import os
import csv
import time
import datetime
from pathlib import Path
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from safetensors.torch import save_file

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets.imagenet import ImageNetDataset
from data.fid import InceptionV3Features, calculate_fid
from models.lpips import LPIPS
from models.vqvae import VQVAEConfig, VQVAE
from models.discriminator import PatchDiscriminator

train_transform = T.Compose([
    T.Resize(256),
    T.RandomCrop(256),
    T.ToTensor(),
])
val_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor(),
])

@dataclass
class VQGANLossConfig:
    vq_weight: float = 1.0
    lpips_weight: float = 1.0
    disc_weight: float = 0.75
    disc_num_layers: int = 2
    disc_num_channels: int = 64
    disc_loss: str = "hinge"

@dataclass
class Config:
    num_steps: int = 1000000
    batch_size: int = 64
    micro_batch_size: int = 8
    learning_rate: float = 3.6e-5
    weight_decay: float = 0.0
    num_workers: int = 4
    model: VQVAEConfig = field(default_factory=VQVAEConfig)
    loss: VQGANLossConfig = field(default_factory=VQGANLossConfig)
    data_path: str = str(Path(__file__).parent / "data/imagenet")
    save: bool = True
    save_every: int = 2000
    eval_every: int = 2000
    sample_every: int = 500
    compiled: bool = False


def repeat(iterable):
    while True:
        yield from iterable

def all_reduce(data, device):
    data = torch.tensor(data, device=device)
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    return data.item()

def all_gather(data, device):
    output = [torch.zeros_like(data, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(output, data.to(device))
    return torch.cat(output, dim=0)

def get_dataloader(config, rank, transform, batch_size, shuffle, split):
    dataset = ImageNetDataset(Path(config.data_path), transform=transform, split=split)
    sampler = DistributedSampler(dataset, rank=rank, shuffle=shuffle)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=config.num_workers)


def train(rank, world_size, config, result_path):
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Load the dataset
    train_loader = repeat(get_dataloader(config, rank, transform=train_transform, batch_size=config.batch_size, shuffle=True, split="train"))
    val_loader = get_dataloader(config, rank, transform=val_transform, batch_size=config.micro_batch_size, shuffle=False, split="val")

    # Instantiate the model and optimizer
    torch.set_float32_matmul_precision('high')
    raw_model = VQVAE(config.model).to(device)
    model = torch.compile(raw_model) if config.compiled else raw_model
    model = DDP(model, device_ids=[rank])
    g_optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.5, 0.9))

    # Instantiate the losses
    lpips = LPIPS().to(device).eval()
    discriminator = PatchDiscriminator(num_layers=config.loss.disc_num_layers, num_channels=config.loss.disc_num_channels).to(device)
    discriminator = DDP(discriminator, device_ids=[rank])
    d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, betas=(0.5, 0.9))

    # Instantiate the metrics
    inception = InceptionV3Features().to(device).eval()

    assert config.batch_size % (config.micro_batch_size * world_size) == 0, "batch_size must be a multiple of micro_batch_size * world_size"
    grad_accum_steps = config.batch_size // (config.micro_batch_size * world_size)
    if rank == 0:
        for name, submodel in {'Encoder': raw_model.encoder, 'Decoder': raw_model.decoder, 'Total': model}.items():
            param_count = sum(p.numel() for p in submodel.parameters())
            print(f"{name} parameters: {param_count:,}")

    # Create results directory and csv file
    save_experiment = config.save and rank == 0
    if save_experiment:
        code_path = result_path / 'code'
        code_path.mkdir(parents=True, exist_ok=True)
        with open(result_path / 'config.yaml', 'w') as f:
            f.write(OmegaConf.to_yaml(config))
        with open(result_path / 'results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'key', 'value'])


    # Helper functions

    def log_values(step, values):
        with open(result_path / 'results.csv', 'a') as f:
            writer = csv.writer(f)
            for key, value in values.items():
                writer.writerow([step, key, value])

    def compute_g_loss(inputs):
        reconstructions, vq_loss = model(inputs)
        vq_loss = vq_loss / grad_accum_steps
        l1_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()).mean() / grad_accum_steps
        lpips_loss = lpips(inputs.contiguous(), reconstructions.contiguous()).mean() / grad_accum_steps
        rec_loss = l1_loss + config.loss.lpips_weight * lpips_loss
        g_loss = discriminator.module.g_loss(reconstructions.contiguous(), rec_loss, model.module.last_layer.weight) / grad_accum_steps
        loss = rec_loss.mean() + config.loss.disc_weight * g_loss + config.loss.vq_weight * vq_loss.mean()
        return loss, reconstructions, {'loss': loss, 'vq': vq_loss, 'l1': l1_loss, 'lpips': lpips_loss, 'g': g_loss}

    def compute_d_loss(inputs, reconstructions):
        return discriminator.module.d_loss(inputs.contiguous(), reconstructions.contiguous(), 'hinge') / grad_accum_steps


    # Training / validation steps

    def train_step(images):
        model.train()
        discriminator.train()
        loss_totals = {}

        # generator step
        g_optimizer.zero_grad()
        for i in range(grad_accum_steps):
            inputs = images[i*config.micro_batch_size : (i+1)*config.micro_batch_size]
            model.require_backward_grad_sync = (i == grad_accum_steps - 1)
            loss, _, sublosses = compute_g_loss(inputs)
            loss.backward()
            for key, value in sublosses.items():
                loss_totals[key] = loss_totals.get(key, 0) + value.mean().item()
        g_optimizer.step()

        # discriminator step
        d_optimizer.zero_grad()
        for i in range(grad_accum_steps):
            inputs = images[i*config.micro_batch_size : (i+1)*config.micro_batch_size]
            discriminator.require_backward_grad_sync = (i == grad_accum_steps - 1)
            with torch.no_grad():
                reconstructions, _ = model(inputs)
            loss = compute_d_loss(inputs, reconstructions)
            loss.backward()
            loss_totals['d'] = loss_totals.get('d', 0) + loss.item()
        d_optimizer.step()

        return loss_totals, reconstructions

    def val_step(images):
        model.eval()
        discriminator.eval()
        model.requires_grad = False
        model.module.last_layer.requires_grad = True  # for gradient computation in g_loss
        _, reconstructions, loss_totals = compute_g_loss(images)
        loss_totals['d'] = compute_d_loss(images, reconstructions)
        model.requires_grad = True
        return {k:v.mean().item() for k,v in loss_totals.items()}, reconstructions

    def val_epoch():
        loss_totals = {}
        real_activations, fake_activations = [], []

        for step, (images, _) in enumerate(val_loader):
            # Compute val losses and activations
            start_time = time.perf_counter()
            images = images.to(device)
            losses, reconstructions = val_step(images)
            reconstructions = ((reconstructions + 1) / 2).clamp(0, 1)  # scale to [0, 1]
            with torch.no_grad():
                real_activations.append(inception(images).cpu())
                fake_activations.append(inception(reconstructions).cpu())
            for key, value in losses.items():
                loss_totals[key] = loss_totals.get(key, 0) + value
            step_time = time.perf_counter() - start_time

            # Display losses and save samples
            if rank == 0:
                losses_str = ' | '.join(f"{k}: {v:.4f}" for k,v in losses.items())
                eta = datetime.timedelta(seconds=int(step_time * (len(val_loader) - step)))
                print(f"val step: {step:6d} | val {losses_str} | dt: {step_time*1000:.1f}ms | eta: {eta}  ", end='\r')
            if rank == 0 and step % config.sample_every == 0:
                val_samples_path = result_path / 'samples' / 'val'
                val_samples_path.mkdir(parents=True, exist_ok=True)
                torchvision.utils.save_image(torch.cat([images[:8].cpu(), reconstructions[:8].cpu()]), val_samples_path / f'{step:06d}.png', nrow=8)
            del images, reconstructions  # prevent OOMs on 3090/4090

        # Compute FID score
        real_activations = all_gather(torch.cat(real_activations), device)
        fake_activations = all_gather(torch.cat(fake_activations), device)
        fid_score = calculate_fid(real_activations, fake_activations) if rank == 0 else None

        return loss_totals, fid_score


    # Training loop

    for step in range(config.num_steps):
        # Run a single training step
        start_time = time.perf_counter()
        images, _ = next(train_loader)
        losses, reconstructions = train_step(images.to(device))
        step_time = time.perf_counter() - start_time

        if rank == 0:
            losses_str = ' | '.join(f"{k}: {v:.4f}" for k,v in losses.items())
            print(f"step: {step:6d} | {losses_str} | dt: {step_time*1000:.1f}ms")
        if rank == 0 and step % config.sample_every == 0:
            train_samples_path = result_path / 'samples' / 'train'
            train_samples_path.mkdir(parents=True, exist_ok=True)
            torchvision.utils.save_image(torch.cat([images[:8].cpu(), reconstructions[:8].cpu()]), train_samples_path / f'{step:06d}.png', nrow=8)
        if save_experiment:
            log_values(step, {**losses, 'step_time': step_time})
        del images, reconstructions  # prevent OOMs on 3090/4090

        # Periodically compute evals
        if step and (step % config.eval_every == 0 or step == config.num_steps - 1):
            if rank == 0:
                print('---')

            start_time = time.perf_counter()
            val_losses, fid_score = val_epoch()
            eval_time = time.perf_counter() - start_time

            if rank == 0:
                losses_str = ' | '.join(f"{k}: {v:.4f}" for k,v in val_losses.items())
                print(f"step: {step:6d} | val {losses_str} | fid: {fid_score:.6f} | eval time: {eval_time:.1f}s\n---")
            if save_experiment:
                log_values(step, {**val_losses, 'eval_time': eval_time})

        # Periodically save the model
        if save_experiment and step and (step % config.save_every == 0 or step == config.num_steps - 1):
            state_dict = {k: v.cpu() for k, v in raw_model.state_dict().items()}
            save_file(state_dict, result_path / f'checkpoint_{step:06d}.safetensors')


# Entry point

if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    schema = OmegaConf.structured(Config)
    config = OmegaConf.merge(schema, OmegaConf.from_cli())
    config = OmegaConf.to_object(config)
    ngpus = torch.cuda.device_count()

    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_path = Path(__file__).parent / 'experiments' / current_time

    if ngpus > 1:
        mp.spawn(train, args=(ngpus, config, result_path), nprocs=ngpus, join=True)
    else:
        train(0, 1, config, result_path)
