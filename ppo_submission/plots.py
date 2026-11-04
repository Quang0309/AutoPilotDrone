"""Render reference-style PPO learning curves from saved episode evaluations.

No environment, model loading or training is required. Timesteps in these logs
are already cumulative across curriculum stages; they must not be offset again.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np


ROOT = Path(__file__).resolve().parent
STAGES = [
    ("easy", "Stage 1: Open Air", "Free-space navigation", "#187bb5"),
    ("medium", "Stage 2: Obstacle (No Cage)", "Single-pole avoidance", "#f18a20"),
    ("hard", "Stage 3: Obstacle + Cage", "Pole, netting and PVC boundary", "#299640"),
]


def read_stage(run_dir: Path, stage: str):
    source = run_dir / stage / "evaluations.jsonl"
    config_path = run_dir / stage / "run_config.json"
    config = json.loads(config_path.read_text())
    origin = int(config["source_timesteps"])
    points = []
    dates = []
    for line_number, line in enumerate(source.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        episodes = row["results"]
        assert len(episodes) == row["episodes"] > 0
        rewards = np.asarray([episode["reward"] for episode in episodes], dtype=float)
        lengths = np.asarray([episode["length"] for episode in episodes], dtype=float)
        assert np.isfinite(rewards).all() and np.isfinite(lengths).all()
        assert np.isclose(rewards.mean(), row["mean_reward"])
        assert np.isclose(lengths.mean(), row["mean_length"])
        timestep = int(row["timesteps"])
        assert timestep - origin == int(row["stage_steps"])
        points.append({
            "stage": stage,
            "cumulative_timesteps": timestep,
            "stage_timesteps": int(row["stage_steps"]),
            "evaluation_episodes": len(episodes),
            "mean_return": float(rewards.mean()),
            "std_return": float(rewards.std(ddof=0)),
            "mean_episode_length": float(lengths.mean()),
            "std_episode_length": float(lengths.std(ddof=0)),
            "success_rate": float(np.mean([episode["success"] for episode in episodes])),
            "collision_rate": float(np.mean([episode["collision"] for episode in episodes])),
            "source_line": line_number,
        })
        dates.append(row["evaluated_at"])
    assert points, f"No saved evaluations for {stage}"
    timesteps = [point["cumulative_timesteps"] for point in points]
    assert timesteps == sorted(set(timesteps)), f"Repeated or unordered timesteps: {stage}"
    assert timesteps[0] == origin
    provenance = {
        "evaluations": str(source.relative_to(ROOT)),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "run_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_timesteps": origin,
        "evaluation_points": len(points),
        "first_evaluated_at": dates[0],
        "last_evaluated_at": dates[-1],
    }
    return points, provenance


def format_steps(value, _position):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    return f"{value / 1000:.0f}k" if value else "0"


def render(run_dir: Path, output: Path):
    data, sources = {}, {}
    for stage, *_ in STAGES:
        data[stage], sources[stage] = read_stage(run_dir, stage)
    bounds = [sources[stage]["source_timesteps"] for stage, *_ in STAGES]
    end = data["hard"][-1]["cumulative_timesteps"]
    bounds.append(end)
    for index, (stage, *_) in enumerate(STAGES[:-1]):
        assert data[stage][-1]["cumulative_timesteps"] == bounds[index + 1]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelcolor": "#273342",
        "axes.edgecolor": "#bac6d3",
        "axes.titleweight": "bold",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 1, figsize=(15, 10.5), sharex=True)
    fig.set_facecolor("white")
    fig.subplots_adjust(left=.09, right=.965, top=.79, bottom=.16, hspace=.35)
    fig.add_artist(Rectangle((.025, .897), .95, .079, transform=fig.transFigure,
                             facecolor="#eff6ff", edgecolor="#2864f0", linewidth=1.6))
    fig.text(.5, .936, "PPO LEARNING CURVES", ha="center", va="center",
             fontsize=27, fontweight="bold", color="#2563eb")
    fig.text(.5, .858, "PPO Policy Evaluation Across Curriculum Stages",
             ha="center", va="center", fontsize=16, fontweight="bold", color="#26313e")
    date = sources["easy"]["first_evaluated_at"][:10]
    fig.text(.5, .832, f"Saved training run: {date}  |  100 evaluation episodes per checkpoint",
             ha="center", va="center", fontsize=10.5, color="#64748b")

    metrics = [
        ("mean_return", "std_return", "Evaluation Episodic Return", "Mean evaluation return"),
        ("mean_episode_length", "std_episode_length", "Evaluation Episode Length", "Episode length (control steps)"),
    ]
    for axis, (mean_key, std_key, title, ylabel) in zip(axes, metrics):
        for index, (stage, _, _, color) in enumerate(STAGES):
            points = data[stage]
            x = np.asarray([p["cumulative_timesteps"] for p in points])
            mean = np.asarray([p[mean_key] for p in points])
            std = np.asarray([p[std_key] for p in points])
            axis.axvspan(bounds[index], bounds[index + 1], color=color, alpha=.065, zorder=0)
            if index:
                axis.axvline(bounds[index], color="#7c8792", linestyle="--", linewidth=1.1, zorder=2)
            axis.fill_between(x, mean - std, mean + std, color=color, alpha=.19,
                              linewidth=0, zorder=2)
            axis.plot(x, mean, color=color, linewidth=2.1, zorder=3)
        axis.set_title(title, fontsize=12.5, pad=11)
        axis.set_ylabel(ylabel, labelpad=10)
        axis.set_xlim(0, end * 1.012)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#c8d1db", linewidth=.7, linestyle=":", alpha=.8)
        axis.grid(axis="x", color="#dce3eb", linewidth=.6, alpha=.5)
        axis.tick_params(length=0, pad=7)
        for spine in axis.spines.values():
            spine.set_linewidth(.8)

    axes[0].axhline(0, color="#7c8792", linestyle=":", linewidth=1.1)
    extrema = [value for stage, *_ in STAGES for point in data[stage]
               for value in [point["mean_return"] - point["std_return"],
                             point["mean_return"] + point["std_return"]]]
    margin = .08 * (max(extrema) - min(extrema))
    axes[0].set_ylim(min(extrema) - margin, max(extrema) + 5.5 * margin)
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlabel("Cumulative curriculum training timesteps", fontweight="bold", labelpad=11)
    axes[1].xaxis.set_major_formatter(FuncFormatter(format_steps))
    axes[1].set_xticks([0, 200_000, 400_000, 600_000, 800_000, 1_000_000, 1_200_000, end])
    axes[1].set_xticklabels(["0", "200k", "400k", "600k", "800k", "1.0M", "1.2M", f"{end / 1_000_000:.3f}M"])
    axes[1].legend(handles=[Line2D([0], [0], color=color, lw=2,
                                  label=f"Stage {i + 1}: {stage.title()}")
                           for i, (stage, _, _, color) in enumerate(STAGES)],
                   loc="upper right", frameon=True, edgecolor="#d0d8e2", fontsize=10)

    # Extra return-axis headroom keeps stage labels clear of the data curves.
    for index, (stage, title, subtitle, color) in enumerate(STAGES):
        center = (bounds[index] + bounds[index + 1]) / 2
        axes[0].text(center, .97,
                     f"{title}\n{subtitle}\nFinal mean return: {data[stage][-1]['mean_return']:+.2f}",
                     transform=axes[0].get_xaxis_transform(), ha="center", va="top",
                     fontsize=10, fontweight="medium", linespacing=1.4,
                     bbox={"boxstyle": "round,pad=.4", "facecolor": "white",
                           "edgecolor": color, "linewidth": 1.1}, clip_on=False)

    fig.text(.09, .067, "Solid line: evaluation mean   ·   Shaded band: ±1 standard deviation across episodes   ·   No smoothing",
             fontsize=10, color="#475569")
    fig.text(.09, .041, f"Source: {run_dir.name}/{{easy,medium,hard}}/evaluations.jsonl. Task and reward settings differ between stages.",
             fontsize=9.5, color="#64748b")

    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(output.with_suffix("." + extension), dpi=200, facecolor="white")
    plt.close(fig)

    rows = [point for stage, *_ in STAGES for point in data[stage]]
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "sources": sources,
        "statistic": "Episode mean and population standard deviation (ddof=0)",
        "smoothing": None,
        "global_timesteps_used_without_additional_offsets": True,
        "stage_boundaries": bounds,
        "total_evaluation_checkpoints": len(rows),
        "final_stage_results": {stage: data[stage][-1] for stage, *_ in STAGES},
        "outputs": [str(output.with_suffix("." + ext)) for ext in ["png", "pdf", "svg", "csv"]],
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ["stage_boundaries", "total_evaluation_checkpoints", "final_stage_results", "outputs"]}, indent=2))


STAGE_NAMES = [stage[0] for stage in STAGES]
COLORS = [stage[3] for stage in STAGES]

def load(path, default):
    return json.loads(path.read_text()) if path.exists() else default

def read_history(result, critic=False):
    data={stage:[] for stage in STAGE_NAMES}
    for stage in STAGE_NAMES:
        if critic:
            paths=[result/stage/'progress.csv',*sorted((result/stage/'segments').glob('*/progress.csv'))]
            for path in paths:
                if not path.exists():continue
                with path.open() as handle:
                    for row in csv.DictReader(handle):
                        keys=['time/total_timesteps','train/value_loss','train/explained_variance']
                        if all(row.get(key) for key in keys):
                            data[stage].append(dict(stage=stage,timesteps=int(float(row[keys[0]])),
                                                   first=float(row[keys[1]]),second=float(row[keys[2]])))
        else:
            path=result/stage/'evaluations.jsonl'
            lines=path.read_text().splitlines() if path.exists() else []
            for index,line in enumerate(lines):
                if not line.strip():continue
                try:row=json.loads(line)
                except json.JSONDecodeError:
                    if index==len(lines)-1:break
                    raise
                rewards=np.array([episode['reward'] for episode in row['results']])
                lengths=np.array([episode['length'] for episode in row['results']])
                data[stage].append(dict(stage=stage,timesteps=row['timesteps'],
                    first=float(rewards.mean()),second=float(lengths.mean()),
                    first_std=float(rewards.std()),second_std=float(lengths.std())))
        data[stage].sort(key=lambda row:row['timesteps'])
    return data

def plot(result=ROOT / 'ppo_results', critic=False):
    result=Path(result);data=read_history(result,critic)
    config=load(result/'run_config.json',{})
    targets=config.get('steps',[500000,400000,300000])
    origin=data['easy'][0]['timesteps'] if data['easy'] else 0
    bounds=[0]
    for index,stage in enumerate(STAGE_NAMES):
        stage_config=load(result/stage/'run_config.json',{})
        start=stage_config.get('source_timesteps',bounds[-1])
        next_config=load(result/STAGE_NAMES[index+1]/'run_config.json',{}) if index<2 else {}
        end=next_config.get('source_timesteps',max(start+targets[index],max((r['timesteps'] for r in data[stage]),default=0)))
        bounds.append(end)
    bounds=[0]+[step-origin for step in bounds[1:]]
    fig,axes=plt.subplots(2,1,figsize=(15,8.5),sharex=True,gridspec_kw={'hspace':.32})
    fig.suptitle('PPO Critic Optimization Across Curriculum Stages' if critic else
                 'PPO Policy Evaluation Across Curriculum Stages',fontsize=17,fontweight='bold',y=.985)
    titles=['Critic Value Loss','Critic Explained Variance'] if critic else ['Evaluation Episodic Return','Evaluation Episode Length']
    for index,stage in enumerate(STAGE_NAMES):
        rows=data[stage]
        for axis,title,metric in zip(axes,titles,['first','second']):
            axis.set_title(title,fontweight='bold',pad=12)
            axis.axvspan(bounds[index],bounds[index+1],color=COLORS[index],alpha=.065)
            if index:axis.axvline(bounds[index],color='#777',linestyle='--',linewidth=1)
            axis.grid(alpha=.2,linestyle=':')
            if rows:
                x=np.array([row['timesteps']-origin for row in rows])
                y=np.array([row[metric] for row in rows])
                if critic:
                    smooth=np.array([y[max(0,i-4):i+1].mean() for i in range(len(y))])
                    axis.scatter(x,y,color=COLORS[index],s=12,alpha=.25)
                    axis.plot(x,smooth,color=COLORS[index],linewidth=2,label=f'{stage.title()} (last: {y[-1]:.3f})')
                else:
                    std=np.array([row[metric+'_std'] for row in rows])
                    axis.plot(x,y,color=COLORS[index],linewidth=1.8,marker='o' if len(rows)==1 else None,
                              label=f'{stage.title()} (last: {y[-1]:.2f})')
                    if len(rows)>1:
                        lower=y-std if metric=='first' else np.maximum(0,y-std)
                        axis.fill_between(x,lower,y+std,color=COLORS[index],alpha=.17)
                    else:axis.errorbar(x,y,yerr=std,color=COLORS[index],capsize=4)
        label=f'Stage {index+1}: {stage.title()}\n'+(f'Last: {rows[-1]["first"]:.2f}' if rows else 'Awaiting data')
        axes[0].text(np.mean(bounds[index:index+2]),.96,label,ha='center',va='top',
                     transform=axes[0].get_xaxis_transform(),fontweight='bold',
                     bbox=dict(boxstyle='round,pad=.35',fc='white',ec=COLORS[index]))
    axes[0].set_ylabel('Value Loss (Log Scale)' if critic else 'Mean Evaluation Return')
    axes[1].set_ylabel('Explained Variance' if critic else 'Episode Length (Steps)')
    if critic:
        if any(row['first']>0 for rows in data.values() for row in rows):axes[0].set_yscale('log')
        axes[1].axhline(1,color='green',linestyle=':',label='Ideal (1.0)')
        axes[1].axhline(0,color='red',linestyle=':',label='Baseline (0.0)')
    else:
        axes[0].axhline(0,color='#888',linestyle=':')
        axes[1].set_ylim(bottom=0)
    for axis in axes:
        axis.set_xlim(0,max(bounds[-1],1)*1.02);axis.margins(y=.3)
        if axis.get_legend_handles_labels()[0]:axis.legend(fontsize=9)
    axes[1].set_xlabel('Timesteps Since First Recorded Easy Step')
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v/1e6:.1f}M' if v>=1e6 else f'{v/1000:.0f}k'))
    caption='Dots: training logs; lines: trailing five-point means.' if critic else 'Evaluation mean ±1 SD across 100 episodes; 24 control steps/s.'
    fig.text(.5,.02,caption+' Future stage spans are provisional.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.08,right=.985,bottom=.10,top=.91)
    output=result/'plots';output.mkdir(parents=True,exist_ok=True)
    name='curriculum_critic' if critic else 'curriculum_evaluation'
    for extension in ['png','svg','pdf']:fig.savefig(output/f'{name}.{extension}',dpi=180,facecolor='white')
    rows=[row for group in data.values() for row in group]
    if rows:
        with (output/f'{name}_history.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return fig

def refresh(result):
    for critic in [False,True]:plt.close(plot(result,critic))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "ppo_results")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve() if args.output else ROOT / ".cache/ppo_learning_curves"
    render(run_dir, output)


if __name__ == "__main__":
    main()
