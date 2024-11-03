import shutil
import argparse
import multiprocessing
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from pathlib import Path
from ale_py import ALEInterface, LoggerMode

ALEInterface.setLoggerMode(LoggerMode.Warning)

def run_episode(args):
    env_name, episode_num, data_dir = args
    env = gym.make(env_name)
    obs, _ = env.reset()
    done = trunc = False
    episode = []

    while not done and not trunc:
        action = env.action_space.sample()
        next_obs, reward, done, trunc, _ = env.step(action)
        episode.append((obs, action, reward, done))
        obs = next_obs

    env.close()
    episode_data = {
        'observations': np.array([step[0] for step in episode]),
        'actions': np.array([step[1] for step in episode]),
        'rewards': np.array([step[2] for step in episode]),
        'dones': np.array([step[3] for step in episode])}

    np.savez_compressed(data_dir / f"episode_{episode_num:05d}.npz", **episode_data)
    return len(episode)

def collect_data(env_name, num_episodes):
    data_dir = Path(__file__).parent / "atari" / env_name
    if data_dir.exists():
        shutil.rmtree(str(data_dir))
    data_dir.mkdir(parents=True)

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        episode_args = [(env_name, i, data_dir) for i in range(num_episodes)]
        total_steps = sum(tqdm(pool.imap(run_episode, episode_args), total=num_episodes, desc="Collecting episodes"))

    print(f"Collected {total_steps} steps in {num_episodes} episodes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Atari environment data")
    parser.add_argument("env", type=str, help="Name of the Atari environment")
    parser.add_argument('-n', '--num-episodes', type=int, default=10, help="Number of episodes to collect")
    args = parser.parse_args()

    collect_data(args.env, args.num_episodes)
