#!/usr/bin/env python3
"""
plot_training_curves.py
=======================
Extracts, analyzes, and visualizes policy evaluation metrics from evaluations.npz
across sequential curriculum stages:
  - Stage 1: Easy (Free-space target navigation, open air)
  - Stage 2: Medium (Obstacle avoidance in empty simulation, no cage)
  - Stage 3: Hard (Obstacle avoidance with realistic cage netting/enclosure)

Key Metrics Analyzed:
  1. Mean Evaluation Return (+/- 1 Std Dev) -> Episodic performance & policy mastery
  2. Episode Length (Steps to Goal / Timeout) -> Flight efficiency & survival time

Supports:
  - 'timeline': Continuous sequential curriculum progression (0 -> 1.1M cumulative timesteps)
  - 'per_stage': 3-column faceted inspection on native stage timesteps
  - 'both': Generates both publication-ready figures

Usage:
  python gym_pybullet_drones/examples/plot_training_curves.py
  python gym_pybullet_drones/examples/plot_training_curves.py --view timeline --output eval_curriculum_timeline.png
  python gym_pybullet_drones/examples/plot_training_curves.py --view per_stage --output eval_per_stage_detailed.png
  python gym_pybullet_drones/examples/plot_training_curves.py --view both
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Default Curriculum Stage Definitions
# ---------------------------------------------------------------------------
CURRICULUM_STAGES = [
    {
        "id": "stage_1",
        "title": "Stage 1: Open Air",
        "subtitle": "Free-space Navigation",
        "path": "results/navigate/uts_stage_1/evaluations.npz",
        "color": "#1f77b4",       # Deep Blue
        "fill_color": "#e8f1f8",  # Light blue shading
        "edge_color": "#145280",
    },
    {
        "id": "stage_2_no_cage",
        "title": "Stage 2: Obstacle (No Cage)",
        "subtitle": "Single Pole Dodging",
        "path": "results/navigate/uts_stage_2_no_cage/evaluations.npz",
        "color": "#ff7f0e",       # Deep Orange
        "fill_color": "#fff3e6",  # Light orange shading
        "edge_color": "#b85500",
    },
    {
        "id": "stage_2_cage",
        "title": "Stage 3: Obstacle + Cage",
        "subtitle": "Pole + Netting / PVC Boundary",
        "path": "results/navigate/uts_stage_2_cage/evaluations.npz",
        "color": "#2ca02c",       # Deep Green
        "fill_color": "#eafaf1",  # Light green shading
        "edge_color": "#1d6f1d",
    },
]


def load_eval_data(file_path):
    """Loads and computes statistics from an evaluations.npz file."""
    if not os.path.exists(file_path):
        print(f"[WARN] File not found: {file_path}")
        return None
    data = np.load(file_path)
    timesteps = data['timesteps']
    results = data['results']        # shape: (n_evals, n_eval_episodes)
    ep_lengths = data['ep_lengths']  # shape: (n_evals, n_eval_episodes)

    mean_reward = results.mean(axis=1)
    std_reward = results.std(axis=1)

    mean_len = ep_lengths.mean(axis=1)
    std_len = ep_lengths.std(axis=1)

    return {
        "timesteps": timesteps,
        "results": results,
        "ep_lengths": ep_lengths,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_len": mean_len,
        "std_len": std_len,
        "final_mean_reward": float(mean_reward[-1]),
        "final_std_reward": float(std_reward[-1]),
        "final_mean_len": float(mean_len[-1]),
        "final_std_len": float(std_len[-1]),
        "max_reward": float(np.max(mean_reward)),
        "min_reward": float(np.min(mean_reward)),
    }


def format_thousands(x, pos):
    """Format large numbers with 'k' or 'M' suffix for publication-clean axes."""
    if x >= 1_000_000:
        return f"{x * 1e-6:.1f}M"
    elif x >= 1_000:
        return f"{x * 1e-3:.0f}k"
    else:
        return f"{x:.0f}"


def print_summary_table(all_data):
    """Print a clean academic summary table of the evaluation metrics across stages."""
    print("\n" + "=" * 90)
    print(f"{'POLICY EVALUATION CURRICULUM SUMMARY TABLE':^90}")
    print("=" * 90)
    header = f"{'Stage':<27} | {'Metric':<22} | {'Initial':>10} | {'Final':>10} | {'Min':>10} | {'Max':>10}"
    print(header)
    print("-" * 90)

    for item in all_data:
        stage = item["stage"]
        d = item["data"]
        
        # Reward row
        print(f"{stage['title']:<27} | {'Episodic Return':<22} | {d['mean_reward'][0]:>+10.2f} | {d['final_mean_reward']:>+10.2f} | {d['min_reward']:>+10.2f} | {d['max_reward']:>+10.2f}")
        # Length row
        print(f"{'':<27} | {'Episode Length (Steps)':<22} | {d['mean_len'][0]:>10.1f} | {d['final_mean_len']:>10.1f} | {np.min(d['mean_len']):>10.1f} | {np.max(d['mean_len']):>10.1f}")
        print("-" * 90)
    print("=" * 90 + "\n")


def plot_curriculum_timeline(all_data, output_path="eval_curriculum_timeline.png", show_fig=False):
    """
    Plot 1: Continuous Curriculum Timeline (Sequential Multi-Stage Progression)
    Shows cumulative training steps across all stages with shaded boundary backgrounds.
    2 vertically aligned panels with shared x-axis.
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 1, figsize=(15, 8.5), dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.26, top=0.89, bottom=0.08, left=0.07, right=0.96)

    # Clean header with generous top headroom to guarantee zero overlap
    fig.suptitle(
        "PPO Policy Evaluation Across Curriculum Stages",
        fontsize=16, fontweight='bold', y=0.965
    )

    ax_reward = axes[0]
    ax_length = axes[1]

    # Calculate cumulative step offsets for sequential stitching
    cumulative_offset = 0
    stage_boundaries = [0]
    stage_centers = []

    for item in all_data:
        d = item["data"]
        max_s = d["timesteps"][-1]
        stage_end = cumulative_offset + max_s
        stage_centers.append((cumulative_offset + stage_end) / 2.0)
        cumulative_offset = stage_end
        stage_boundaries.append(cumulative_offset)

    # Decorate stage backgrounds and vertical boundaries across both subplots
    for ax in [ax_reward, ax_length]:
        ax.xaxis.set_major_formatter(FuncFormatter(format_thousands))
        for i, item in enumerate(all_data):
            stage = item["stage"]
            b_start = stage_boundaries[i]
            b_end = stage_boundaries[i + 1]
            ax.axvspan(b_start, b_end, color=stage["fill_color"], alpha=0.7, zorder=1)
            if i > 0:
                ax.axvline(b_start, color="#555555", linestyle="--", linewidth=1.2, alpha=0.8, zorder=4)

    # ---------------------------------------------------------
    # Panel 1: Evaluation Episodic Return
    # ---------------------------------------------------------
    ax_reward.set_title("Evaluation Episodic Return", fontsize=13, fontweight='bold', pad=12)
    ax_reward.set_ylabel("Mean Evaluation Return", fontsize=11.5, fontweight='bold')
    ax_reward.axhline(0.0, color="#888888", linestyle=":", linewidth=1.1, alpha=0.8)
    ax_reward.grid(True, linestyle=":", alpha=0.6)

    cum_offset = 0
    max_rew = -np.inf
    min_rew = np.inf
    for item in all_data:
        stage = item["stage"]
        d = item["data"]
        ts = d["timesteps"] + cum_offset
        mu = d["mean_reward"]
        std = d["std_reward"]
        max_rew = max(max_rew, float(np.max(mu + std)))
        min_rew = min(min_rew, float(np.min(mu - std)))

        # Plot shaded confidence interval and solid mean curve
        ax_reward.fill_between(ts, mu - std, mu + std, color=stage["color"], alpha=0.18, zorder=2)
        ax_reward.plot(ts, mu, color=stage["color"], linewidth=2.4, zorder=3)
        ax_reward.scatter(ts[::4], mu[::4], color=stage["color"], alpha=0.45, s=16, zorder=3)
        cum_offset += d["timesteps"][-1]

    # Expand upper y-limit for stage badges
    y_top = max_rew + (max_rew - min_rew) * 0.22
    y_bot = min_rew - (max_rew - min_rew) * 0.05
    ax_reward.set_ylim(y_bot, y_top)

    # Annotate stage badges with final return callouts
    for i, item in enumerate(all_data):
        stage = item["stage"]
        d = item["data"]
        center = stage_centers[i]
        final_ret = d["final_mean_reward"]
        ax_reward.text(
            center, y_top * 0.94 if y_top > 0 else y_top * 1.06,
            f"{stage['title']}\n({stage['subtitle']} | Final: {final_ret:+.1f})",
            ha='center', va='top', fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor=stage['color'], alpha=0.94, linewidth=1.2),
            zorder=5
        )

    # ---------------------------------------------------------
    # Panel 2: Evaluation Episode Length (Survival / Efficiency)
    # ---------------------------------------------------------
    ax_length.set_title("Evaluation Episode Length", fontsize=13, fontweight='bold', pad=12)
    ax_length.set_xlabel("Cumulative Curriculum Timesteps", fontsize=11.5, fontweight='bold')
    ax_length.set_ylabel("Episode Length (Steps)", fontsize=11.5, fontweight='bold')
    ax_length.grid(True, linestyle=":", alpha=0.6)

    cum_offset = 0
    for item in all_data:
        stage = item["stage"]
        d = item["data"]
        ts = d["timesteps"] + cum_offset
        mu = d["mean_len"]
        std = d["std_len"]

        ax_length.fill_between(ts, np.maximum(0, mu - std), mu + std, color=stage["color"], alpha=0.18, zorder=2)
        ax_length.plot(ts, mu, color=stage["color"], linewidth=2.4, 
                       label=f"{stage['title']} (Final: {d['final_mean_len']:.0f} steps / {d['final_mean_len']/24:.1f}s)", zorder=3)
        ax_length.scatter(ts[::4], mu[::4], color=stage["color"], alpha=0.45, s=16, zorder=3)
        cum_offset += d["timesteps"][-1]

    ax_length.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9.5)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved Curriculum Timeline Figure to: {output_path}")
    if show_fig:
        plt.show()
    plt.close()


def plot_per_stage_comparison(all_data, output_path="eval_per_stage_detailed.png", show_fig=False):
    """
    Plot 2: 3-Column Faceted View (Stage 1 vs Stage 2 vs Stage 3).
    Displays each stage's evaluations on its native internal timestep axis (0 to MaxSteps)
    to allow fine-grained inspection without scale distortion.
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    n_stages = len(all_data)
    fig, axes = plt.subplots(2, n_stages, figsize=(5.5 * n_stages, 7.5), dpi=300, sharex='col')
    plt.subplots_adjust(hspace=0.22, wspace=0.22, top=0.88, bottom=0.10, left=0.07, right=0.96)

    fig.suptitle(
        "Faceted Policy Evaluation Dynamics by Curriculum Stage",
        fontsize=15, fontweight='bold', y=0.965
    )

    for col_idx, item in enumerate(all_data):
        stage = item["stage"]
        d = item["data"]
        ts = d["timesteps"]

        # Column header
        axes[0, col_idx].set_title(f"{stage['title']}\n({stage['subtitle']})", fontsize=12, fontweight='bold', pad=10, color=stage["edge_color"])

        # Row 0: Mean Evaluation Return
        ax_r = axes[0, col_idx]
        ax_r.xaxis.set_major_formatter(FuncFormatter(format_thousands))
        ax_r.set_facecolor(stage["fill_color"])
        ax_r.grid(True, linestyle=":", alpha=0.6)
        if col_idx == 0:
            ax_r.set_ylabel("Mean Return", fontsize=11, fontweight='bold')

        mu_r = d["mean_reward"]
        std_r = d["std_reward"]
        ax_r.fill_between(ts, mu_r - std_r, mu_r + std_r, color=stage["color"], alpha=0.20)
        ax_r.plot(ts, mu_r, color=stage["color"], linewidth=2.4, label=f"Return (Final: {d['final_mean_reward']:+.1f})")
        ax_r.scatter(ts[::4], mu_r[::4], color=stage["color"], alpha=0.45, s=18)
        ax_r.axhline(0.0, color="#888888", linestyle=":", linewidth=1.0)
        ax_r.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9.0)

        # Row 1: Mean Episode Length
        ax_l = axes[1, col_idx]
        ax_l.xaxis.set_major_formatter(FuncFormatter(format_thousands))
        ax_l.set_facecolor(stage["fill_color"])
        ax_l.grid(True, linestyle=":", alpha=0.6)
        ax_l.set_xlabel("Stage Timesteps", fontsize=11, fontweight='bold')
        if col_idx == 0:
            ax_l.set_ylabel("Episode Length (Steps)", fontsize=11, fontweight='bold')

        mu_l = d["mean_len"]
        std_l = d["std_len"]
        ax_l.fill_between(ts, np.maximum(0, mu_l - std_l), mu_l + std_l, color=stage["color"], alpha=0.20)
        ax_l.plot(ts, mu_l, color=stage["color"], linewidth=2.4, label=f"Length (Final: {d['final_mean_len']:.0f})")
        ax_l.scatter(ts[::4], mu_l[::4], color=stage["color"], alpha=0.45, s=18)
        ax_l.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9.0)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved Faceted Per-Stage Figure to: {output_path}")
    if show_fig:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot policy evaluation curves across curriculum stages.")
    parser.add_argument("--view", type=str, choices=["timeline", "per_stage", "both"], default="both",
                        help="Plotting mode: 'timeline' (continuous stitched stages), 'per_stage' (3-column faceted), or 'both'.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (defaults to eval_curriculum_timeline.png and/or eval_per_stage_detailed.png).")
    parser.add_argument("--show", action="store_true", help="Display interactive figure window.")
    args = parser.parse_args()

    all_data = []
    print("[INFO] Loading evaluations.npz across curriculum stages...")
    for stage in CURRICULUM_STAGES:
        data = load_eval_data(stage["path"])
        if data is None:
            continue
        all_data.append({
            "stage": stage,
            "data": data
        })

    if not all_data:
        print("[ERROR] No evaluations.npz files could be loaded. Exiting.")
        return

    # Print summary table
    print_summary_table(all_data)

    # Generate plots
    if args.view in ["timeline", "both"]:
        timeline_out = args.output if (args.output and args.view == "timeline") else "eval_curriculum_timeline.png"
        plot_curriculum_timeline(all_data, output_path=timeline_out, show_fig=args.show)

    if args.view in ["per_stage", "both"]:
        per_stage_out = args.output if (args.output and args.view == "per_stage") else "eval_per_stage_detailed.png"
        plot_per_stage_comparison(all_data, output_path=per_stage_out, show_fig=args.show)


if __name__ == "__main__":
    main()
