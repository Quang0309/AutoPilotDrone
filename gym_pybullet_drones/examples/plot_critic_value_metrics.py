#!/usr/bin/env python3
"""
plot_critic_value_metrics.py
============================
Extracts, analyzes, and visualizes Critic and Value Network metrics
from TensorBoard event files across sequential curriculum stages:
  - Stage 1: Easy (Free-space target navigation, open air)
  - Stage 2: Medium (Obstacle avoidance in empty simulation, no cage)
  - Stage 3: Hard (Obstacle avoidance with realistic cage netting/enclosure)

Key Metrics Analyzed:
  1. Value Loss (MSE: (V(s) - Returns)^2) -> Value function convergence
  2. Explained Variance (1 - Var(Returns - V(s)) / Var(Returns)) -> Predictive power (ideal ~1.0)
  3. Policy Entropy Loss (-H(pi)) -> Exploration annealing & action certainty
  4. PPO Approx KL Divergence -> Trust-region step stability (healthy < 0.02)
  5. PPO Clip Fraction -> Proportion of clipped policy updates (healthy ~0.05-0.20)

Usage:
  python gym_pybullet_drones/examples/plot_critic_value_metrics.py
  python gym_pybullet_drones/examples/plot_critic_value_metrics.py --view timeline --output critic_timeline.png
  python gym_pybullet_drones/examples/plot_critic_value_metrics.py --view per_stage --output critic_per_stage.png
  python gym_pybullet_drones/examples/plot_critic_value_metrics.py --view both
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Default Curriculum Run Definitions
# ---------------------------------------------------------------------------
CURRICULUM_STAGES = [
    {
        "id": "stage_1",
        "title": "Stage 1: Open Air",
        "subtitle": "Free-space Navigation",
        "dir": "results/navigate/uts_stage_1",
        "color": "#1f77b4",       # Deep Blue
        "fill_color": "#e8f1f8",  # Light blue shading
        "edge_color": "#145280",
        "max_steps": 500_000,
    },
    {
        "id": "stage_2_no_cage",
        "title": "Stage 2: Obstacle (No Cage)",
        "subtitle": "Single Pole Dodging",
        "dir": "results/navigate/uts_stage_2_no_cage",
        "color": "#ff7f0e",       # Deep Orange
        "fill_color": "#fff3e6",  # Light orange shading
        "edge_color": "#b85500",
        "max_steps": 400_000,
    },
    {
        "id": "stage_2_cage",
        "title": "Stage 3: Obstacle + Cage",
        "subtitle": "Pole + Netting / PVC Boundary",
        "dir": "results/navigate/uts_stage_2_cage",
        "color": "#2ca02c",       # Deep Green
        "fill_color": "#eafaf1",  # Light green shading
        "edge_color": "#1d6f1d",
        "max_steps": 200_000,
    },
]


def find_tfevent_file(stage_dir):
    """Locate the TensorBoard event file inside a stage's results directory."""
    tb_pattern = os.path.join(stage_dir, "tb", "*", "events.out.tfevents.*")
    files = glob.glob(tb_pattern)
    if not files:
        # Fallback to direct tb directory
        direct_pattern = os.path.join(stage_dir, "tb", "events.out.tfevents.*")
        files = glob.glob(direct_pattern)
    if not files:
        # Recursive search within stage_dir
        files = glob.glob(os.path.join(stage_dir, "**", "events.out.tfevents.*"), recursive=True)
    
    if files:
        # Return the newest or largest event file
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return files[0]
    return None


def load_tb_scalars(event_file, tags):
    """
    Load specified scalar metrics from a TensorBoard event file using EventAccumulator.
    Returns a dict: {tag: [(step, value), ...]}
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        raise ImportError(
            "tensorboard package not found in current environment. "
            "Please run with the project Python environment, e.g.:\n"
            "  /Users/quangnguyen/miniconda3/envs/tello_rl/bin/python " + __file__
        )

    ea = EventAccumulator(event_file, size_guidance={'scalars': 0})
    ea.Reload()
    available_tags = ea.Tags().get('scalars', [])

    results = {}
    for tag in tags:
        if tag in available_tags:
            scalars = ea.Scalars(tag)
            steps = np.array([s.step for s in scalars], dtype=np.int64)
            vals = np.array([s.value for s in scalars], dtype=np.float64)
            results[tag] = (steps, vals)
        else:
            results[tag] = None
    return results


def exponential_moving_average(values, alpha=0.3):
    """Compute exponential moving average for smoother trend visualization."""
    if len(values) == 0:
        return values
    smoothed = np.empty_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def format_thousands(x, pos):
    """Format large numbers with 'k' suffix for publication-clean axes."""
    if x >= 1_000_000:
        return f"{x * 1e-6:.1f}M"
    elif x >= 1_000:
        return f"{x * 1e-3:.0f}k"
    else:
        return f"{x:.0f}"


def print_summary_table(all_data):
    """Print an academic summary table of the extracted Critic & PPO metrics."""
    print("\n" + "=" * 90)
    print(f"{'CRITIC & VALUE NETWORK SUMMARY TABLE':^90}")
    print("=" * 90)
    header = f"{'Stage':<25} | {'Metric':<22} | {'Initial':>10} | {'Final':>10} | {'Min':>10} | {'Max':>10}"
    print(header)
    print("-" * 90)

    for item in all_data:
        stage_name = item["stage"]["title"]
        metrics = item["metrics"]
        first_line = True
        
        display_metrics = [
            ("train/value_loss", "Value Loss (MSE)"),
            ("train/explained_variance", "Explained Var (0-1)"),
            ("train/entropy_loss", "Entropy Loss"),
            ("train/approx_kl", "Approx KL Div"),
            ("train/clip_fraction", "PPO Clip Fraction")
        ]

        for tag, label in display_metrics:
            if tag in metrics and metrics[tag] is not None:
                _, vals = metrics[tag]
                s_name = stage_name if first_line else ""
                first_line = False
                print(f"{s_name:<25} | {label:<22} | {vals[0]:>10.4f} | {vals[-1]:>10.4f} | {np.min(vals):>10.4f} | {np.max(vals):>10.4f}")
        print("-" * 90)
    print("=" * 90 + "\n")


def plot_curriculum_timeline(all_data, output_path="critic_curriculum_timeline.png", show_fig=False):
    """
    Plot 1: Continuous Curriculum Timeline (Sequential Multi-Stage Transfer)
    Shows cumulative training steps across the stages with shaded boundary backgrounds.
    Focused on Critic Value Loss and Explained Variance.
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 1, figsize=(15, 8.5), dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.26, top=0.89, bottom=0.08, left=0.07, right=0.96)
    
    # Clean, concise suptitle positioned with generous top margin to prevent any overlap
    fig.suptitle(
        "PPO Critic Optimization Across Curriculum Stages",
        fontsize=16, fontweight='bold', y=0.965
    )

    ax_vloss = axes[0]
    ax_expvar = axes[1]

    # Calculate cumulative step offsets for sequential stitching
    cumulative_offset = 0
    stage_boundaries = [0]
    stage_centers = []

    for item in all_data:
        stage = item["stage"]
        metrics = item["metrics"]
        
        # Use maximum logged step or configured max
        if "train/value_loss" in metrics and metrics["train/value_loss"] is not None:
            max_s = metrics["train/value_loss"][0][-1]
        else:
            max_s = stage["max_steps"]
            
        stage_end = cumulative_offset + max_s
        stage_centers.append((cumulative_offset + stage_end) / 2.0)
        cumulative_offset = stage_end
        stage_boundaries.append(cumulative_offset)

    # Decorate stage backgrounds and vertical boundaries across both subplots
    for ax in [ax_vloss, ax_expvar]:
        ax.xaxis.set_major_formatter(FuncFormatter(format_thousands))
        for i, item in enumerate(all_data):
            stage = item["stage"]
            b_start = stage_boundaries[i]
            b_end = stage_boundaries[i + 1]
            ax.axvspan(b_start, b_end, color=stage["fill_color"], alpha=0.7, zorder=1)
            if i > 0:
                ax.axvline(b_start, color="#555555", linestyle="--", linewidth=1.2, alpha=0.8, zorder=4)

    # ---------------------------------------------------------
    # Panel 1: Value Loss
    # ---------------------------------------------------------
    ax_vloss.set_title("Critic Value Loss", fontsize=13, fontweight='bold', pad=12)
    ax_vloss.set_ylabel("Value Loss (MSE)", fontsize=11.5, fontweight='bold')
    ax_vloss.grid(True, linestyle=":", alpha=0.6)

    cum_offset = 0
    max_observed_loss = 0.0
    for item in all_data:
        stage = item["stage"]
        if "train/value_loss" in item["metrics"] and item["metrics"]["train/value_loss"] is not None:
            steps, vals = item["metrics"]["train/value_loss"]
            max_observed_loss = max(max_observed_loss, float(np.max(vals)))
            t_steps = steps + cum_offset
            smoothed = exponential_moving_average(vals, alpha=0.35)
            
            # Plot raw scatter points with lower opacity and smoothed trend line
            ax_vloss.scatter(t_steps, vals, color=stage["color"], alpha=0.30, s=22, zorder=2)
            ax_vloss.plot(t_steps, smoothed, color=stage["color"], linewidth=2.6, zorder=3)
            cum_offset += steps[-1]

    # Generous y-limit so stage badges sit comfortably above the curve peak
    ax_vloss.set_ylim(-5, max(380.0, max_observed_loss * 1.15))

    # Annotate stage badges with clear final metric callouts
    for i, item in enumerate(all_data):
        stage = item["stage"]
        center = stage_centers[i]
        final_vl = item["metrics"]["train/value_loss"][1][-1] if ("train/value_loss" in item["metrics"] and item["metrics"]["train/value_loss"] is not None) else 0.0
        ax_vloss.text(
            center, ax_vloss.get_ylim()[1] * 0.93,
            f"{stage['title']}\n({stage['subtitle']} | Final: {final_vl:.1f})",
            ha='center', va='top', fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor=stage['color'], alpha=0.94, linewidth=1.2),
            zorder=5
        )

    # ---------------------------------------------------------
    # Panel 2: Explained Variance
    # ---------------------------------------------------------
    ax_expvar.set_title("Critic Explained Variance", fontsize=13, fontweight='bold', pad=12)
    ax_expvar.set_xlabel("Cumulative Curriculum Timesteps", fontsize=11.5, fontweight='bold')
    ax_expvar.set_ylabel("Explained Variance", fontsize=11.5, fontweight='bold')
    ax_expvar.axhline(1.0, color="#2ca02c", linestyle=":", linewidth=1.4, label="Ideal (1.0)")
    ax_expvar.axhline(0.0, color="#d62728", linestyle=":", linewidth=1.4, label="Baseline (0.0)")
    ax_expvar.grid(True, linestyle=":", alpha=0.6)

    cum_offset = 0
    for item in all_data:
        stage = item["stage"]
        if "train/explained_variance" in item["metrics"] and item["metrics"]["train/explained_variance"] is not None:
            steps, vals = item["metrics"]["train/explained_variance"]
            t_steps = steps + cum_offset
            smoothed = exponential_moving_average(vals, alpha=0.35)
            
            ax_expvar.scatter(t_steps, vals, color=stage["color"], alpha=0.30, s=22, zorder=2)
            ax_expvar.plot(t_steps, smoothed, color=stage["color"], linewidth=2.6, 
                           label=f"{stage['title']} (Final: {vals[-1]:.2f})", zorder=3)
            cum_offset += steps[-1]

    ax_expvar.set_ylim(-0.4, 1.08)
    ax_expvar.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9.5)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved Curriculum Timeline Figure to: {output_path}")
    if show_fig:
        plt.show()
    plt.close()


def plot_per_stage_comparison(all_data, output_path="critic_per_stage_detailed.png", show_fig=False):
    """
    Plot 2: 3-Column Faceted View (Stage 1 vs Stage 2 vs Stage 3).
    Displays each stage's metrics on its native internal timestep axis (0 to MaxSteps)
    to avoid scale distortion and allow fine-grained inspection.
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    n_stages = len(all_data)
    fig, axes = plt.subplots(3, n_stages, figsize=(5.5 * n_stages, 10.5), dpi=300, sharex='col')
    plt.subplots_adjust(hspace=0.22, wspace=0.22, top=0.90, bottom=0.08, left=0.06, right=0.96)

    fig.suptitle(
        "Faceted Critic & Value Network Dynamics by Curriculum Stage",
        fontsize=15, fontweight='bold', y=0.97
    )

    row_metrics = [
        ("train/value_loss", "Value Loss (MSE)", "Value Loss (MSE)"),
        ("train/explained_variance", "Explained Variance", "Explained Variance"),
        ("train/approx_kl", "PPO Approx KL Divergence", "Approx KL Div")
    ]

    for col_idx, item in enumerate(all_data):
        stage = item["stage"]
        metrics = item["metrics"]
        
        # Column title
        axes[0, col_idx].set_title(f"{stage['title']}\n({stage['subtitle']})", fontsize=12, fontweight='bold', pad=10, color=stage["edge_color"])

        for row_idx, (tag, title, ylabel) in enumerate(row_metrics):
            ax = axes[row_idx, col_idx]
            ax.xaxis.set_major_formatter(FuncFormatter(format_thousands))
            ax.set_facecolor(stage["fill_color"])
            ax.grid(True, linestyle=":", alpha=0.6)

            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            if row_idx == 2:
                ax.set_xlabel("Stage Timesteps", fontsize=10.5, fontweight='bold')

            if tag in metrics and metrics[tag] is not None:
                steps, vals = metrics[tag]
                smoothed = exponential_moving_average(vals, alpha=0.35)
                
                # Scatter points + smoothed line
                ax.scatter(steps, vals, color=stage["color"], alpha=0.3, s=20)
                ax.plot(steps, smoothed, color=stage["color"], linewidth=2.4, label=f"Smoothed (Final: {vals[-1]:.3f})")

                if tag == "train/explained_variance":
                    ax.axhline(1.0, color="#2ca02c", linestyle=":", linewidth=1.2, alpha=0.8)
                    ax.axhline(0.0, color="#d62728", linestyle=":", linewidth=1.2, alpha=0.8)
                    ax.set_ylim(min(-0.35, np.min(vals) - 0.05), 1.05)
                elif tag == "train/approx_kl":
                    ax.axhline(0.015, color="#e41a1c", linestyle="--", linewidth=1.0, alpha=0.7, label="PPO Target KL (0.015)")

                ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8.5)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Saved Faceted Per-Stage Figure to: {output_path}")
    if show_fig:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Critic and Value Network metrics across curriculum stages.")
    parser.add_argument("--view", type=str, choices=["timeline", "per_stage", "both"], default="both",
                        help="Plotting mode: 'timeline' (continuous stitched stages), 'per_stage' (3-column faceted), or 'both'.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (defaults to critic_timeline.png and/or critic_per_stage.png).")
    parser.add_argument("--show", action="store_true", help="Display interactive figure window.")
    args = parser.parse_args()

    # Track data across all stages
    all_data = []
    required_tags = [
        "train/value_loss",
        "train/explained_variance",
        "train/entropy_loss",
        "train/approx_kl",
        "train/clip_fraction",
        "train/loss",
        "train/policy_gradient_loss"
    ]

    print("[INFO] Searching for TensorBoard event files in curriculum stages...")
    for stage in CURRICULUM_STAGES:
        event_path = find_tfevent_file(stage["dir"])
        if event_path is None or not os.path.exists(event_path):
            print(f"[WARN] No event file found for {stage['title']} in {stage['dir']}")
            continue
        
        print(f"[INFO] Loading: {stage['title']} -> {event_path}")
        metrics = load_tb_scalars(event_path, required_tags)
        all_data.append({
            "stage": stage,
            "event_file": event_path,
            "metrics": metrics
        })

    if not all_data:
        print("[ERROR] No valid stage event files could be loaded. Exiting.")
        return

    # Print formatted summary table
    print_summary_table(all_data)

    # Generate plots
    if args.view in ["timeline", "both"]:
        timeline_out = args.output if (args.output and args.view == "timeline") else "critic_curriculum_timeline.png"
        plot_curriculum_timeline(all_data, output_path=timeline_out, show_fig=args.show)

    if args.view in ["per_stage", "both"]:
        per_stage_out = args.output if (args.output and args.view == "per_stage") else "critic_per_stage_detailed.png"
        plot_per_stage_comparison(all_data, output_path=per_stage_out, show_fig=args.show)


if __name__ == "__main__":
    main()
