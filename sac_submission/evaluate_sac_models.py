import argparse
import sys
import time
from pathlib import Path

SUBMISSION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SUBMISSION_DIR.parent
sys.path.insert(0, str(SUBMISSION_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import SAC

from reward_modify.runtime import SeededNavigate


MODEL_DIR = SUBMISSION_DIR / "models"
DEFAULT_MODELS = {
    "easy": MODEL_DIR / "best_sac_easy.zip",
    "medium": MODEL_DIR / "best_sac_medium.zip",
    "hard": MODEL_DIR / "best_sac_hard.zip",
}


def evaluate(model_path, episodes=5, difficulty="hard", gui=False, seed=7000):
    model = SAC.load(model_path, device="auto")
    env = SeededNavigate(difficulty, seed, gui=gui)
    successes = 0
    collisions = 0

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            done = False
            total_reward = 0.0
            steps = 0
            info = {}

            while not done:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                done = terminated or truncated
                if gui:
                    time.sleep(env.unwrapped.CTRL_TIMESTEP)

            success = bool(info.get("success", False))
            collision = bool(info.get("collision", False))
            successes += int(success)
            collisions += int(collision)
            print(
                f"Episode {episode + 1}: success={success}, collision={collision}, "
                f"steps={steps}, reward={total_reward:.2f}"
            )
    finally:
        env.close()

    print(f"Success rate: {successes / episodes:.1%}")
    print(f"Collision rate: {collisions / episodes:.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained SAC drone policy")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    model_path = args.model or DEFAULT_MODELS[args.difficulty]
    evaluate(model_path, args.episodes, args.difficulty, args.gui)
