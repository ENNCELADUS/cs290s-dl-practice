from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = {
    "tiny": Path("hpc_logs/tiny/project2-pretrain-gpt"),
    "small": Path("hpc_logs/small/project2-pretrain-gpt"),
    "medium": Path("hpc_logs/medium/project2-pretrain-gpt"),
}


def read_scalar_series(run_dir: Path, tag: str) -> list[tuple[int, float]]:
    """Read and merge one scalar tag from all TensorBoard event files in a run."""
    series: dict[int, float] = {}
    for event_file in sorted(run_dir.glob("**/events.out.tfevents*")):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 10000})
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for event in accumulator.Scalars(tag):
            series[event.step] = event.value
    return sorted(series.items())


def plot_loss_curves(runs: dict[str, Path], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    for name, run_dir in runs.items():
        train_loss = read_scalar_series(run_dir, "train_loss")
        val_loss = read_scalar_series(run_dir, "val_loss")
        axes[0].plot(
            [step for step, _ in train_loss],
            [value for _, value in train_loss],
            label=name,
            linewidth=1.8,
        )
        axes[1].plot(
            [step for step, _ in val_loss],
            [value for _, value in val_loss],
            marker="o",
            label=name,
            linewidth=1.8,
        )

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].grid(alpha=0.25)

    for axis in axes:
        axis.legend()

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_perplexity_curves(runs: dict[str, Path], output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)

    for name, run_dir in runs.items():
        perplexity = read_scalar_series(run_dir, "val_perplexity")
        axis.plot(
            [step for step, _ in perplexity],
            [value for _, value in perplexity],
            marker="o",
            label=name,
            linewidth=1.8,
        )

    axis.set_title("Validation Perplexity")
    axis.set_xlabel("Step")
    axis.set_ylabel("Perplexity")
    axis.grid(alpha=0.25)
    axis.legend()

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(RUNS, output_dir / "basic_loss_curves.png")
    plot_perplexity_curves(RUNS, output_dir / "basic_perplexity_curves.png")


if __name__ == "__main__":
    main()
