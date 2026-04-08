"""Export publication-quality report figures from TensorBoard event logs.

Style targets ACL/EMNLP/NeurIPS figure aesthetics:
- Clean serif typography (LaTeX-compatible)
- Muted, journal-quality colour palette
- Consistent tick styling, tight layout, and high-DPI export
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from tensorboard.backend.event_processing import event_accumulator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_ROOT / "figures"

# ---------------------------------------------------------------------------
# Colour palette – perceptually distinct, print-safe, colour-blind accessible
# ---------------------------------------------------------------------------
PALETTE = {
    "blue": "#2B5A8E",
    "blue_light": "#A8C4E0",
    "orange": "#C05B1A",
    "orange_light": "#ECC8A8",
    "green": "#3D7A47",
    "green_light": "#A8D4B0",
    "gray": "#6B6B6B",
    "gray_light": "#D0D0D0",
}


@dataclass(frozen=True)
class RunSpec:
    """Metadata for one logged experiment run."""

    key: str
    label: str
    run_dir: Path
    color: str
    color_light: str


def _latest_run_dir(parent: Path) -> Path:
    """Return the most recently modified run directory under *parent*."""
    run_dirs = [path for path in parent.iterdir() if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {parent}")
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


RUNS: dict[str, RunSpec] = {
    "matched": RunSpec(
        key="matched",
        label="Controlled Frozen Baseline",
        run_dir=_latest_run_dir(PROJECT_ROOT / "runs/bert_classifier"),
        color=PALETTE["blue"],
        color_light=PALETTE["blue_light"],
    ),
    "advanced": RunSpec(
        key="advanced",
        label="Advanced BERT",
        run_dir=_latest_run_dir(PROJECT_ROOT / "runs/bert_classifier_advanced"),
        color=PALETTE["orange"],
        color_light=PALETTE["orange_light"],
    ),
}

CURVE_SPECS: tuple[tuple[str, str, str, bool], ...] = (
    ("loss/train_step", "Training Loss", "Loss", False),
    ("loss/val", "Validation Loss", "Loss", False),
    ("metrics/macro_f1", "Validation Macro F1", "Macro F1", True),
    ("metrics/accuracy", "Validation Accuracy", "Accuracy", True),
)


# ---------------------------------------------------------------------------
# Global style configuration
# ---------------------------------------------------------------------------

def configure_style() -> None:
    """Set a NeurIPS/ACL-compatible matplotlib style."""
    mpl.rcParams.update(
        {
            # Resolution
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Typography – serif for paper body compatibility
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "Palatino"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            # Axes
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#3A3A3A",
            "axes.linewidth": 0.9,
            "axes.facecolor": "#F9F9F8",
            "figure.facecolor": "#FFFFFF",
            "axes.titlelocation": "left",
            "axes.titlepad": 7,
            # Grid
            "axes.grid": True,
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.8,
            "grid.linestyle": "--",
            # Ticks
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#CCCCCC",
            "legend.borderpad": 0.4,
            "legend.labelspacing": 0.35,
            # Lines
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
        }
    )


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Load scalar series from one TensorBoard event directory."""
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event file found in {run_dir}")

    acc = event_accumulator.EventAccumulator(
        str(event_files[0]),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    acc.Reload()
    series: dict[str, list[tuple[int, float]]] = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        series[tag] = [(e.step, float(e.value)) for e in events]
    return series


def extract_xy(
    scalars: dict[str, list[tuple[int, float]]],
    tag: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) arrays for one scalar tag."""
    points = scalars.get(tag, [])
    if not points:
        return np.array([], dtype=float), np.array([], dtype=float)
    steps = np.array([s for s, _ in points], dtype=float)
    values = np.array([v for _, v in points], dtype=float)
    return steps, values


def ema(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Exponential moving average for noisy step-level loss curves."""
    if values.size == 0:
        return values
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, values.size):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def load_checkpoint_metrics(experiment_name: str) -> dict[str, float]:
    """Load best-checkpoint metrics for one experiment."""
    checkpoint_path = (
        PROJECT_ROOT / "checkpoints" / experiment_name / "best_model.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return {
        key: float(value)
        for key, value in checkpoint["metrics"].items()
    }


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

def _pad_ylim(ax: plt.Axes, arrays: list[np.ndarray], factor: float = 0.12) -> None:
    """Set y-limits with symmetric padding around the observed data range."""
    valid = [a for a in arrays if a.size > 0]
    if not valid:
        return
    lo = min(float(a.min()) for a in valid)
    hi = max(float(a.max()) for a in valid)
    pad = max((hi - lo) * factor, 0.008)
    ax.set_ylim(max(0.0, lo - pad), hi + pad)


def _annotate_best(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    maximize: bool,
    color: str,
) -> None:
    """Place a filled circle + value annotation at the best-checkpoint step."""
    if x.size == 0:
        return
    idx = int(np.nanargmax(y) if maximize else np.nanargmin(y))
    bx, by = float(x[idx]), float(y[idx])
    ax.scatter([bx], [by], s=52, color=color, edgecolors="white",
               linewidths=1.2, zorder=6)
    # Nudge the label toward the centre of the axis to avoid clipping
    xfrac = (bx - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0] + 1e-9)
    ha = "left" if xfrac < 0.85 else "right"
    offset_x = 6 if ha == "left" else -6
    ax.annotate(
        f"{by:.4f}",
        xy=(bx, by),
        xytext=(offset_x, 8),
        textcoords="offset points",
        fontsize=8.5,
        color=color,
        fontweight="bold",
        ha=ha,
    )


# ---------------------------------------------------------------------------
# Panel-level plotting
# ---------------------------------------------------------------------------

def _plot_panel(
    ax: plt.Axes,
    *,
    run_scalars: dict[str, dict[str, list[tuple[int, float]]]],
    tag: str,
    title: str,
    ylabel: str,
    maximize: bool,
) -> None:
    """Render one metric panel into *ax*."""
    drawn_values: list[np.ndarray] = []

    for spec in RUNS.values():
        xs, ys = extract_xy(run_scalars[spec.key], tag)
        if xs.size == 0:
            log.warning("Tag '%s' not found in run '%s'.", tag, spec.key)
            continue

        if tag == "loss/train_step":
            smoothed = ema(ys, alpha=0.08)
            # Ghost raw signal
            ax.plot(xs, ys, color=spec.color_light, lw=0.9, alpha=0.35, zorder=1)
            # EMA-smoothed foreground
            ax.plot(xs, smoothed, color=spec.color, lw=2.0, alpha=0.95, zorder=3)
            _annotate_best(ax, xs, smoothed, maximize=maximize, color=spec.color)
            drawn_values.append(smoothed)
        else:
            ax.plot(
                xs, ys,
                color=spec.color,
                lw=2.0,
                marker="o",
                markersize=4.5,
                markerfacecolor="white",
                markeredgewidth=1.2,
                zorder=3,
            )
            _annotate_best(ax, xs, ys, maximize=maximize, color=spec.color)
            drawn_values.append(ys)

    # Axis limits
    if drawn_values:
        all_xs = [
            extract_xy(run_scalars[spec.key], tag)[0]
            for spec in RUNS.values()
        ]
        valid_xs = [a for a in all_xs if a.size > 0]
        if valid_xs:
            ax.set_xlim(left=min(float(a.min()) for a in valid_xs))

    pad_factor = 0.10 if tag.startswith("metrics") else 0.12
    _pad_ylim(ax, drawn_values, factor=pad_factor)

    ax.set_title(title, fontweight="semibold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)


# ---------------------------------------------------------------------------
# Figure 1 – 2×2 training/validation curves
# ---------------------------------------------------------------------------

def export_main_curves(
    run_scalars: dict[str, dict[str, list[tuple[int, float]]]],
) -> list[Path]:
    """Export the 2×2 main-curve comparison figure."""
    fig, axes = plt.subplots(
        2, 2,
        figsize=(11.5, 8.4),
        layout="constrained",
    )
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.91))

    for ax, (tag, title, ylabel, maximize) in zip(axes.flat, CURVE_SPECS, strict=True):
        _plot_panel(
            ax,
            run_scalars=run_scalars,
            tag=tag,
            title=title,
            ylabel=ylabel,
            maximize=maximize,
        )

    # Shared legend above the grid – placed inside the reserved top margin
    legend_handles = [
        Line2D(
            [0], [0],
            color=spec.color,
            lw=2.2,
            marker="o",
            markersize=5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=spec.label,
        )
        for spec in RUNS.values()
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=2.0,
        handletextpad=0.6,
        borderpad=0.5,
    )
    fig.suptitle(
        "Controlled Frozen Baseline vs. Advanced BERT – Training Dynamics",
        y=1.0,
        fontsize=12,
        fontweight="semibold",
        va="bottom",
    )

    outputs = [
        FIGURE_DIR / "bert_controlled_vs_advanced_curves.pdf",
        FIGURE_DIR / "bert_controlled_vs_advanced_curves.png",
    ]
    _save(fig, outputs)
    return outputs


# ---------------------------------------------------------------------------
# Figure 2 – Learning-rate schedules
# ---------------------------------------------------------------------------

def export_learning_rates(
    run_scalars: dict[str, dict[str, list[tuple[int, float]]]],
) -> list[Path]:
    """Export the learning-rate schedule comparison."""
    fig, ax = plt.subplots(figsize=(9.5, 3.8), constrained_layout=True)

    # The baseline uses a single param group → logs lr/main only.
    # The advanced model uses two param groups → logs lr/adapter and lr/encoder.
    # The baseline LR and the advanced adapter LR share the same schedule, so
    # both lines are drawn with distinct styles (dashed vs. solid) to remain
    # visible on top of each other.
    curves: list[tuple[np.ndarray, np.ndarray, str, str, float, str]] = [
        (
            *extract_xy(run_scalars["matched"], "lr/main"),
            PALETTE["blue"],
            "--",
            2.5,
            "Baseline main LR",
        ),
        (
            *extract_xy(run_scalars["advanced"], "lr/adapter"),
            PALETTE["orange"],
            "-",
            2.0,
            "Advanced adapter / head LR",
        ),
        (
            *extract_xy(run_scalars["advanced"], "lr/encoder"),
            PALETTE["green"],
            "--",
            2.0,
            "Advanced encoder LR",
        ),
    ]

    for xs, ys, color, ls, lw, label in curves:
        if xs.size == 0:
            log.warning("Learning-rate tag not found for label '%s'.", label)
            continue
        ax.plot(xs, ys, color=color, lw=lw, linestyle=ls, label=label, zorder=3)
        # Peak marker
        peak_idx = int(np.argmax(ys))
        ax.scatter(
            [xs[peak_idx]], [ys[peak_idx]],
            s=40, color=color, edgecolors="white", lw=1.0, zorder=5,
        )

    ax.set_title("Learning-Rate Schedules", fontweight="semibold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-5, -3))
    ax.legend(loc="upper right")

    outputs = [
        FIGURE_DIR / "bert_learning_rate_schedules.pdf",
        FIGURE_DIR / "bert_learning_rate_schedules.png",
    ]
    _save(fig, outputs)
    return outputs


# ---------------------------------------------------------------------------
# Figure 3 – Grouped bar chart: headline metric comparison
# ---------------------------------------------------------------------------

def export_summary_bar_chart() -> list[Path]:
    """Export a publication-ready grouped bar chart of headline metrics."""
    baseline_metrics = load_checkpoint_metrics("bert_classifier")
    advanced_metrics = load_checkpoint_metrics("bert_classifier_advanced")

    # fmt: off
    metrics      = ["Accuracy",    "Macro F1",           "Validation Loss"]
    baseline_vals = np.array([
        baseline_metrics["accuracy"],
        baseline_metrics["macro_f1"],
        baseline_metrics["loss"],
    ])
    advanced_vals = np.array([
        advanced_metrics["accuracy"],
        advanced_metrics["macro_f1"],
        advanced_metrics["loss"],
    ])
    higher_better = [True, True, False]   # direction of improvement
    # fmt: on

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.6), constrained_layout=True)

    bar_width = 0.34
    x = np.array([0.0, 1.0])
    bar_colors = [PALETTE["blue"], PALETTE["orange"]]
    bar_labels = ["Frozen Baseline", "Advanced BERT"]

    for ax, metric, bval, aval, hi_better in zip(
        axes, metrics, baseline_vals, advanced_vals, higher_better, strict=True
    ):
        vals = [bval, aval]
        bars = ax.bar(
            x, vals,
            width=bar_width,
            color=bar_colors,
            alpha=0.90,
            zorder=3,
            edgecolor="white",
            linewidth=0.6,
        )

        # Numeric labels on top of bars
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + (max(vals) - min(vals)) * 0.015,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="semibold",
                color="#333333",
            )

        # Delta annotation: draw an arrow + Δ label between the two bars
        delta = aval - bval
        sign = "+" if delta >= 0 else ""
        delta_str = f"{sign}{delta:.4f}"
        # Colour the delta green if improvement, red if not
        improve = (delta > 0) if hi_better else (delta < 0)
        delta_color = PALETTE["green"] if improve else PALETTE["orange"]
        mid_x = 0.5
        bar_top = max(bval, aval)
        ax.text(
            mid_x,
            bar_top + (max(vals) - min(vals)) * 0.14,
            f"$\\Delta$ = {delta_str}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=delta_color,
            fontweight="bold",
        )

        # Y-axis limits with breathing room
        lo, hi = min(vals), max(vals)
        ax.set_ylim(lo - (hi - lo) * 0.30, hi + (hi - lo) * 0.45)
        ax.set_xticks(x, bar_labels, fontsize=9)
        ax.set_title(metric, fontweight="semibold")
        ax.grid(axis="y", zorder=0)

    fig.suptitle(
        "Checkpoint-Level Comparison: Controlled Frozen Baseline vs. Advanced BERT",
        fontsize=10.5,
        fontweight="semibold",
        y=1.04,
    )

    outputs = [
        FIGURE_DIR / "bert_controlled_vs_advanced_summary.pdf",
        FIGURE_DIR / "bert_controlled_vs_advanced_summary.png",
    ]
    _save(fig, outputs)
    return outputs


# ---------------------------------------------------------------------------
# Figure 4 – Parameter count comparison (horizontal bar)
# ---------------------------------------------------------------------------

def export_param_comparison() -> list[Path]:
    """Export a compact horizontal bar chart of trainable vs. total parameters."""
    models = ["BiLSTM", "Frozen Baseline\n(RoBERTa)", "Advanced BERT\n(RoBERTa + LoRA)"]
    total_params = np.array([260_546, 103_855_878, 105_730_571], dtype=float)
    trainable_params = np.array([260_546, 1_588_230, 3_565_835], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 3.2), constrained_layout=True)
    y = np.arange(len(models))
    bar_h = 0.32

    ax.barh(
        y + bar_h / 2, total_params / 1e6, height=bar_h,
        color=PALETTE["gray_light"], label="Total", zorder=3,
    )
    ax.barh(
        y - bar_h / 2, trainable_params / 1e6, height=bar_h,
        color=PALETTE["blue"], label="Trainable", zorder=3,
    )

    # Value labels
    def _fmt_m(v: float) -> str:
        return f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.1f}K"

    for i, (tv, tr) in enumerate(zip(total_params, trainable_params)):
        ax.text(tv / 1e6 + 0.3, i + bar_h / 2, _fmt_m(tv),
                va="center", fontsize=8, color=PALETTE["gray"])
        ax.text(tr / 1e6 + 0.3, i - bar_h / 2, _fmt_m(tr),
                va="center", fontsize=8, color=PALETTE["blue"])

    ax.set_yticks(y, models)
    ax.set_xlabel("Number of Parameters (millions)")
    ax.set_title("Parameter Counts by Model", fontweight="semibold")
    ax.legend(loc="lower right")

    outputs = [
        FIGURE_DIR / "param_comparison.pdf",
        FIGURE_DIR / "param_comparison.png",
    ]
    _save(fig, outputs)
    return outputs


# ---------------------------------------------------------------------------
# Shared save helper
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, paths: list[Path]) -> None:
    """Save *fig* to all listed *paths* and close it."""
    for p in paths:
        fig.savefig(p)
        log.info("Saved %s", p)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate all report figures."""
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading TensorBoard event files…")
    run_scalars = {key: load_scalars(spec.run_dir) for key, spec in RUNS.items()}

    outputs: list[Path] = []
    outputs.extend(export_main_curves(run_scalars))
    outputs.extend(export_learning_rates(run_scalars))
    outputs.extend(export_summary_bar_chart())
    outputs.extend(export_param_comparison())

    log.info("All figures exported (%d files).", len(outputs))


if __name__ == "__main__":
    main()
