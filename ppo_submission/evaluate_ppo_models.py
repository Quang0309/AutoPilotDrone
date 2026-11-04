"""Evaluate a saved PPO stage checkpoint without training."""

import argparse
import json
from pathlib import Path
import random
import sys


SUBMISSION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SUBMISSION_DIR.parent


def positive_int(value):
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("episodes must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("episodes must be a positive integer")
    return parsed


def evaluate_model(difficulty, episodes=5, seed=1_001_000, gui=False):
    """Use the recorded observation stack, reset order and episode seeding."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecFrameStack

    from gym_pybullet_drones.utils.enums import ActionType, ObservationType
    from ppo_submission.runtime import NavigateAviary

    checkpoint = SUBMISSION_DIR / "models" / f"best_ppo_{difficulty}.zip"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing PPO checkpoint: {checkpoint}")
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if seed < 0 or seed + episodes - 1 >= 2**32:
        raise ValueError("episode seeds must fit NumPy's unsigned 32-bit range")

    env = make_vec_env(
        NavigateAviary,
        n_envs=1,
        seed=0,
        env_kwargs={
            "difficulty": difficulty,
            "gui": gui,
            "record": False,
            "obs": ObservationType.ALL,
            "act": ActionType.VEL,
        },
    )
    env = VecFrameStack(env, n_stack=4, channels_order="first")
    results = []
    try:
        model = PPO.load(str(checkpoint), device="cpu")
        for episode_seed in range(seed, seed + episodes):
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            env.seed(episode_seed)
            observation = env.reset()
            reward_sum = 0.0
            length = 0
            any_collision = False

            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, rewards, done, infos = env.step(action)
                reward_sum += float(rewards[0])
                length += 1
                info = infos[0]
                any_collision |= bool(info["collision"])
                if done[0]:
                    results.append({
                        "seed": episode_seed,
                        "reward": reward_sum,
                        "length": length,
                        "success": bool(info["success"]) and not any_collision,
                        "collision": any_collision,
                        "distance": float(info["distance"]),
                        "reason": info.get("truncation_reason"),
                    })
                    break
    finally:
        env.close()

    return {
        "difficulty": difficulty,
        "model": checkpoint.name,
        "episodes": len(results),
        "success_rate": float(np.mean([row["success"] for row in results])),
        "collision_rate": float(np.mean([row["collision"] for row in results])),
        "mean_reward": float(np.mean([row["reward"] for row in results])),
        "mean_length": float(np.mean([row["length"] for row in results])),
        "results": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difficulty", choices=("easy", "medium", "hard"), default="hard")
    parser.add_argument("--episodes", type=positive_int, default=5)
    parser.add_argument("--seed", type=int, default=1_001_000)
    parser.add_argument("--gui", action="store_true", help="Show the PyBullet window")
    args = parser.parse_args(argv)
    if args.seed < 0 or args.seed + args.episodes - 1 >= 2**32:
        parser.error("episode seeds must fit NumPy's unsigned 32-bit range")

    result = evaluate_model(args.difficulty, args.episodes, args.seed, args.gui)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
