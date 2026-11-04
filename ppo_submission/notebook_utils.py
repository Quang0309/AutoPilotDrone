"""Display helpers and the optional training launcher for the PPO notebook."""

import contextlib
from html import escape
import io
from pathlib import Path
import signal
import subprocess
import sys

from IPython.display import HTML, Image, display


def show_table(headers, rows):
    """Display escaped values in a compact notebook table."""
    heading = "".join(f"<th>{escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    display(HTML(f"<table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table>"))


def show_learning_curves(run_dir, cache_dir):
    """Render curves into the cache and embed the resulting image."""
    from ppo_submission.plots import render

    output = Path(cache_dir) / "ppo_learning_curves"
    with contextlib.redirect_stdout(io.StringIO()):
        render(Path(run_dir), output)
    display(Image(filename=str(output.with_suffix(".png")), width=1100))


def show_figure(figure):
    """Embed a Matplotlib figure without writing to the recorded run."""
    import matplotlib.pyplot as plt

    with io.BytesIO() as buffer:
        figure.savefig(buffer, format="png", dpi=140, facecolor="white")
        display(Image(data=buffer.getvalue(), width=1050))
    plt.close(figure)


def train_curriculum(root, output_dir, resume=False):
    """Run the existing curriculum command and stream its output into the cell."""
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    recorded_run = root / "ppo_submission" / "ppo_results"
    if output_dir == recorded_run or recorded_run in output_dir.parents:
        raise ValueError("Choose a new directory to preserve the recorded run.")

    command = [
        sys.executable,
        "-u",
        "-m",
        "ppo_submission.train_ppo",
        "--output",
        str(output_dir),
    ]
    if resume:
        command.append("--resume")

    with subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
        except KeyboardInterrupt:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait()
            raise
