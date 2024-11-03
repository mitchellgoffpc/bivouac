import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset

class AtariDataset(Dataset):
    def __init__(self, data_dir: Path, timesteps: int):
        self.data_dir = data_dir
        self.timesteps = timesteps
        self.episode_data = []
        self.episode_lengths = []
        self.cumulative_lengths = [0]

        # Enumerate through all episode files, load data, and compute sequence counts
        for filename in tqdm(sorted(self.data_dir.glob("episode_*.npz")), desc="Loading episodes", leave=False):
            episode_data = np.load(filename)
            self.episode_data.append(episode_data['observations'])
            episode_length = len(episode_data['observations']) - timesteps + 1
            if episode_length > 0:
                self.episode_lengths.append(episode_length)
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + episode_length)

        print(f"Initialized dataset with {len(self.episode_data)} episodes / {len(self)} samples")

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        # Find the episode corresponding to the given idx
        episode_idx = np.searchsorted(self.cumulative_lengths, idx, side='right') - 1
        local_idx = idx - self.cumulative_lengths[episode_idx]

        # Get the episode data from the cached list
        obs_data = self.episode_data[episode_idx][local_idx:local_idx + self.timesteps]
        return torch.as_tensor(obs_data).permute(0,3,1,2)



if __name__ == "__main__":
    import time
    import argparse
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Benchmark or visualize the AtariDataset")
    parser.add_argument("data_path", type=str, help="Path to the Atari dataset")
    parser.add_argument("-m", "--mode", choices=["benchmark", "visualize"], default='benchmark', help="Testing mode")
    parser.add_argument("-t", "--timesteps", type=int, default=4, help="Number of timesteps in each sequence")
    args = parser.parse_args()

    dataset = AtariDataset(Path(args.data_path), timesteps=args.timesteps)

    if args.mode == "benchmark":
        num_samples = 1000
        indices = np.random.randint(0, len(dataset), num_samples)

        st = time.perf_counter()
        for idx in tqdm(indices, desc="Benchmarking", unit="sample"):
            _ = dataset[idx]
        et = time.perf_counter()

        total_time = et - st
        avg_time = total_time / num_samples
        print(f"Time taken for {num_samples} random samples: {total_time:.4f} seconds")
        print(f"Average time per sample: {avg_time:.6f} seconds")

    elif args.mode == "visualize":
        num_samples = 4
        random_indices = np.random.randint(0, len(dataset), num_samples)
        samples = [dataset[idx] for idx in random_indices]

        fig, axes = plt.subplots(args.timesteps, num_samples, figsize=(4 * num_samples, 4 * args.timesteps))
        for col, sample in enumerate(samples):
            for row in range(args.timesteps):
                ax = axes[row, col] if args.timesteps > 1 else axes[col]
                ax.imshow(sample[row].permute(1, 2, 0).numpy())
                ax.axis('off')
                if row == 0:
                    ax.set_title(f"Sample {col + 1}")
                if col == 0:
                    ax.set_ylabel(f"Timestep {row + 1}")

        plt.tight_layout()
        plt.show()
