"""
Benchmark and compare UTS Stage 1, Stage 2 (No Cage), and Stage 2 (Cage) models.

Evaluates performance across quantitative robotics metrics:
  1. Success Rate (%) - Target arrival within 0.25m threshold
  2. Collision Rate (%) - Hits obstacle, net wall, or floor
  3. Out of Bounds / Timeout Rate (%)
  4. Mean Episode Reward (Return +/- Std)
  5. Mean Time to Target (seconds / steps)
  6. Path Efficiency / SPL (Success weighted by Path Length)
  7. Minimum Obstacle Clearance (meters)

Usage:
  python gym_pybullet_drones/examples/benchmark_stages.py --episodes 30 --difficulty medium
  python gym_pybullet_drones/examples/benchmark_stages.py --episodes 20 --all_difficulties
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from gym_pybullet_drones.envs.NavigateAviary import NavigateAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_MODELS = {
    "Stage 1 (Open Air Baseline)": "results/navigate/uts_stage_1/best_model.zip",
    "Stage 2 (Obstacle, No Cage)": "results/navigate/uts_stage_2_no_cage/best_model.zip",
    "Stage 2 (Obstacle + Cage)":   "results/navigate/uts_stage_2_cage/best_model.zip",
}

def evaluate_single_model(model_name, model_path, difficulty, episodes=20, base_seed=42):
    """ Runs headless Monte Carlo evaluation on a single model checkpoint. """
    if not os.path.exists(model_path):
        print(f"[WARN] Checkpoint not found: {model_path} - Skipping.")
        return None

    print(f"\n========================================================")
    print(f"  Evaluating: {model_name}")
    print(f"  Checkpoint: {model_path}")
    print(f"  Difficulty: {difficulty.upper()} | Episodes: {episodes}")
    print(f"========================================================")

    model = PPO.load(model_path)

    # Initialize headless environment for ultra-fast simulation
    base_env = NavigateAviary(
        obs=ObservationType.ALL,
        act=ActionType.VEL,
        gui=False,
        record=False,
        difficulty=difficulty
    )
    test_env = DummyVecEnv([lambda: base_env])
    test_env = VecFrameStack(test_env, n_stack=4, channels_order="first")

    results = []

    for ep in range(episodes):
        ep_seed = base_seed + ep
        np.random.seed(ep_seed)
        obs = test_env.reset()

        start_pos = base_env.INIT_XYZS[0].copy()
        target_pos = base_env.TARGET_POS.copy()
        ideal_dist = float(np.linalg.norm(target_pos - start_pos))

        ep_reward = 0.0
        path_length = 0.0
        prev_pos = start_pos.copy()
        min_obs_clearance = 5.0
        steps = 0
        success = False
        collision = False
        out_of_bounds = False
        timeout = False
        termination_reason = "unknown"

        max_steps = int(base_env.EPISODE_LEN_SEC * base_env.CTRL_FREQ)

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            ep_reward += float(reward[0])
            steps += 1

            # Track drone physical metrics
            drone_pos = base_env._getDroneStateVector(0)[:3]
            step_disp = np.linalg.norm(drone_pos - prev_pos)
            path_length += float(step_disp)
            prev_pos = drone_pos.copy()

            # Track obstacle distance
            if len(base_env.obstacle_ids) > 0:
                dist_obs = base_env._get_min_obstacle_distance(max_search_dist=2.0)
                if dist_obs < min_obs_clearance:
                    min_obs_clearance = dist_obs

            if done[0]:
                inf = info[0]
                if inf.get("success", False):
                    success = True
                    termination_reason = "SUCCESS"
                else:
                    reason = inf.get("truncation_reason", "unknown")
                    termination_reason = reason
                    if reason == "collision":
                        collision = True
                    elif reason in ["out_of_bounds", "flipped"]:
                        out_of_bounds = True
                    elif reason == "timeout":
                        timeout = True
                break
        else:
            timeout = True
            termination_reason = "TIMEOUT"

        # Success weighted by Path Length (SPL)
        # SPL = S * (ideal_dist / max(path_length, ideal_dist))
        spl = (ideal_dist / max(path_length, ideal_dist)) if success else 0.0
        time_sec = steps * base_env.CTRL_TIMESTEP

        results.append({
            "episode": ep + 1,
            "success": success,
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "timeout": timeout,
            "reason": termination_reason,
            "reward": ep_reward,
            "steps": steps,
            "time_sec": time_sec,
            "path_length": path_length,
            "ideal_dist": ideal_dist,
            "spl": spl,
            "min_clearance": min_obs_clearance if len(base_env.obstacle_ids) > 0 else np.nan
        })

        tag = "[OK]  SUCCESS" if success else f"[FAIL] {termination_reason.upper()}"
        print(f"  Ep {ep+1:02d}: {tag:<18} | Rew: {ep_reward:+6.2f} | Time: {time_sec:4.2f}s | Path: {path_length:4.2f}m (SPL: {spl:.2f}) | MinObs: {min_obs_clearance:.2f}m")

    test_env.close()

    df = pd.DataFrame(results)

    summary = {
        "Model": model_name,
        "Difficulty": difficulty,
        "Episodes": episodes,
        "Success Rate (%)": (df["success"].mean()) * 100.0,
        "Collision Rate (%)": (df["collision"].mean()) * 100.0,
        "OOB Rate (%)": (df["out_of_bounds"].mean()) * 100.0,
        "Timeout Rate (%)": (df["timeout"].mean()) * 100.0,
        "Mean Reward": df["reward"].mean(),
        "Std Reward": df["reward"].std(),
        "Mean Time (s)": df[df["success"]]["time_sec"].mean() if df["success"].any() else np.nan,
        "Mean Path (m)": df[df["success"]]["path_length"].mean() if df["success"].any() else np.nan,
        "SPL": df["spl"].mean(),
        "Mean Min Clearance (m)": df["min_clearance"].mean() if not df["min_clearance"].isna().all() else np.nan
    }
    return summary, df

def run_benchmark(difficulties, episodes=20, models=None, output_csv="benchmark_results.csv"):
    if models is None:
        models = DEFAULT_MODELS

    all_summaries = []
    all_raw_dfs = []

    for diff in difficulties:
        for model_name, model_path in models.items():
            res = evaluate_single_model(model_name, model_path, difficulty=diff, episodes=episodes)
            if res is not None:
                summary, df_raw = res
                df_raw["model"] = model_name
                df_raw["difficulty"] = diff
                all_summaries.append(summary)
                all_raw_dfs.append(df_raw)

    if not all_summaries:
        print("[ERROR] No models evaluated.")
        return

    summary_df = pd.DataFrame(all_summaries)

    # Format Markdown Table
    print("\n" + "=" * 90)
    print("                     AUTONOMOUS NAVIGATION PERFORMANCE BENCHMARK")
    print("=" * 90)
    
    display_df = summary_df.copy()
    display_df["Reward (Mean +/- Std)"] = display_df.apply(lambda r: f"{r['Mean Reward']:+.1f} +/- {r['Std Reward']:.1f}", axis=1)
    display_df["Success Rate"] = display_df["Success Rate (%)"].apply(lambda v: f"{v:.1f}%")
    display_df["Collision Rate"] = display_df["Collision Rate (%)"].apply(lambda v: f"{v:.1f}%")
    display_df["SPL"] = display_df["SPL"].apply(lambda v: f"{v:.2f}")
    display_df["Mean Time"] = display_df["Mean Time (s)"].apply(lambda v: f"{v:.2f}s" if pd.notna(v) else "N/A")
    display_df["Min Clearance"] = display_df["Mean Min Clearance (m)"].apply(lambda v: f"{v:.2f}m" if pd.notna(v) else "N/A")

    cols_to_show = ["Model", "Difficulty", "Success Rate", "Collision Rate", "Reward (Mean +/- Std)", "Mean Time", "SPL", "Min Clearance"]
    print(display_df[cols_to_show].to_string(index=False))
    print("=" * 90)

    # Save CSV
    summary_df.to_csv(output_csv, index=False)
    print(f"\n[INFO] Benchmark results saved to: {output_csv}")

    # Save concatenated episode logs
    if all_raw_dfs:
        full_raw_df = pd.concat(all_raw_dfs, ignore_index=True)
        raw_csv = output_csv.replace(".csv", "_episodes.csv")
        full_raw_df.to_csv(raw_csv, index=False)
        print(f"[INFO] Detailed per-episode metrics saved to: {raw_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Stage 1, Stage 2 (No Cage), and Stage 2 (Cage) Models")
    parser.add_argument("--episodes", type=int, default=20, help="Number of test episodes per model (default: 20)")
    parser.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"], help="Environment difficulty (default: medium)")
    parser.add_argument("--all_difficulties", action="store_true", help="Benchmark across all 3 difficulties (easy, medium, hard)")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", help="Output summary CSV file")
    
    args = parser.parse_args()
    
    diffs = ["easy", "medium", "hard"] if args.all_difficulties else [args.difficulty]
    run_benchmark(difficulties=diffs, episodes=args.episodes, output_csv=args.output)
