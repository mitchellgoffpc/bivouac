import h5py
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

FRAMES_PER_FILE = 1000

class CSGODataset(Dataset):
    def __init__(self, seq_length):
        self.seq_length = seq_length
        self.sequences_per_file = FRAMES_PER_FILE - self.seq_length + 1
        self.data_dir = Path(__file__).parent.parent / 'data' / 'csgo' / 'hdf'
        self.file_paths = sorted(list(self.data_dir.glob('*.hdf5')))
        print(f"Initialized dataset with {len(self.file_paths)} files / {len(self)} samples")

    def __len__(self):
        return len(self.file_paths) * self.sequences_per_file

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        file_idx = idx // self.sequences_per_file
        local_idx = idx % self.sequences_per_file
        with h5py.File(self.file_paths[file_idx], 'r') as f:
            frames = np.stack([f[f'frame_{local_idx + i}_x'] for i in range(self.seq_length)])
            actions = np.stack([f[f'frame_{local_idx + i}_y'] for i in range(self.seq_length)])

        return torch.from_numpy(frames).flip(-1).permute(0,3,1,2).div(255 / 2).sub(1), torch.from_numpy(actions)  # frames are BGR for some reason


if __name__ == "__main__":
    import time
    import argparse
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Benchmark or visualize the CSGODataset")
    parser.add_argument("-m", "--mode", choices=["benchmark", "visualize"], default='benchmark', help="Testing mode")
    parser.add_argument("-t", "--timesteps", type=int, default=5, help="Number of timesteps in each sequence")
    args = parser.parse_args()

    dataset = CSGODataset(seq_length=args.timesteps)

    if args.mode == "benchmark":
        num_samples = 1000
        indices = np.random.randint(0, len(dataset), num_samples)

        st = time.perf_counter()
        for idx in tqdm(indices, desc="Benchmarking", unit="sample"):
            _ = dataset[idx]
        et = time.perf_counter()

        avg_time = (et - st) / num_samples
        print(f"Average time per sample: {avg_time*1000:.3f}ms")

    elif args.mode == "visualize":
        num_samples = 4
        random_indices = np.random.randint(0, len(dataset), num_samples)
        samples = [dataset[idx] for idx in random_indices]

        fig, axes = plt.subplots(num_samples, args.timesteps, figsize=(3 * args.timesteps, 2 * num_samples))
        for row, (sample, _) in enumerate(samples):
            for col in range(args.timesteps):
                ax = axes[row, col] if num_samples > 1 else axes[col]
                ax.imshow(sample[col].add(1).div(2).permute(1,2,0).numpy())
                ax.axis('off')
                if col == 0:
                    ax.set_ylabel(f"Sample {row + 1}")
                if row == 0:
                    ax.set_title(f"Timestep {col + 1}")

        plt.tight_layout()
        plt.show()
