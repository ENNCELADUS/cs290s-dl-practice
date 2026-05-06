from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = {
    "small baseline": Path("hpc_logs/small/project2-pretrain-gpt"),
    "medium baseline": Path("hpc_logs/medium/project2-pretrain-gpt"),
    "medium lr=6e-4": Path("hpc_logs/diagnostics/medium_lr_6e4/project2-pretrain-gpt"),
    "medium lr=4e-4": Path("hpc_logs/diagnostics/medium_lr_4e4/project2-pretrain-gpt"),
    "medium warmup=5%": Path("hpc_logs/diagnostics/medium_warmup_5pct/project2-pretrain-gpt"),
    "medium wd=0.2": Path("hpc_logs/diagnostics/medium_wd_02/project2-pretrain-gpt"),
}


def read_scalar_series(run_dir: Path, tag: str) -> list[tuple[int, float]]:
    series: dict[int, float] = {}
    for event_file in sorted(run_dir.glob("**/events.out.tfevents*")):
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 10000})
        accumulator.Reload()
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for event in accumulator.Scalars(tag):
            series[event.step] = event.value
    return sorted(series.items())


def plot_validation_curves(output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    for name, run_dir in RUNS.items():
        val_loss = read_scalar_series(run_dir, "val_loss")
        val_ppl = read_scalar_series(run_dir, "val_perplexity")
        axes[0].plot(
            [step for step, _ in val_loss],
            [value for _, value in val_loss],
            marker="o",
            linewidth=1.8,
            label=name,
        )
        axes[1].plot(
            [step for step, _ in val_ppl],
            [value for _, value in val_ppl],
            marker="o",
            linewidth=1.8,
            label=name,
        )

    axes[0].set_title("Medium Diagnostics: Validation Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Medium Diagnostics: Validation Perplexity")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Perplexity")
    axes[1].grid(alpha=0.25)

    for axis in axes:
        axis.legend(fontsize=8)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_validation_curves(output_dir / "medium_diagnostic_curves.png")


if __name__ == "__main__":
    main()
