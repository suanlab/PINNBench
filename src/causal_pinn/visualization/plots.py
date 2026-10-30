"""Plotting utilities for PINN experiment visualization.

All functions take save_path: str | None = None parameter.
If save_path is provided, save to file; otherwise display the plot.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from torch import Tensor

if TYPE_CHECKING:
    pass


# Default style configuration
DEFAULT_FIGSIZE = (10, 6)
DEFAULT_DPI = 150
DEFAULT_FONTSIZE = 12
STAGE_COLORS = {"stage1": "#1f77b4", "stage2": "#ff7f0e", "stage3": "#2ca02c"}


def _to_numpy(data: Tensor | np.ndarray | list | float) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(data, Tensor):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _save_or_show(fig: Figure, save_path: str | None) -> None:
    """Save figure to file or display it.

    Args:
        fig: The matplotlib figure to save/show.
        save_path: Path to save the figure. If None, display instead.
    """
    if save_path is not None:
        # Ensure directory exists
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


def plot_training_curves(
    history: dict[str, Any],
    title: str = "Training Curves",
    save_path: str | None = None,
) -> Figure:
    """Plot loss curves for all 3 training stages.

    Each stage is shown in a different color. Supports multiple histories
    overlaid (for seed comparison) when histories are provided as a list.

    Args:
        history: Training history dict with keys:
            - stage1_losses: list of dicts with 'total' key
            - stage2_losses: list of dicts with 'total' key
            - stage3_losses: list of dicts with 'total' key
            Or for seed comparison: dict mapping seed -> history dict
        title: Plot title.
        save_path: Path to save the figure. If None, display instead.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    # Check if this is a multi-seed comparison (dict of dicts)
    first_value = next(iter(history.values())) if history else None
    is_multi_seed = isinstance(first_value, dict) and "stage1_losses" in first_value

    if is_multi_seed:
        # Multiple seeds: plot each with different alpha
        seeds = list(history.keys())
        for idx, seed in enumerate(seeds):
            seed_history = history[seed]

            # Extract total losses for each stage
            stage1_losses = [loss.get("total", 0) for loss in seed_history.get("stage1_losses", [])]
            stage2_losses = [loss.get("total", 0) for loss in seed_history.get("stage2_losses", [])]
            stage3_losses = [loss.get("total", 0) for loss in seed_history.get("stage3_losses", [])]

            # Concatenate all stages
            all_losses = stage1_losses + stage2_losses + stage3_losses

            # Stage boundaries
            s1_end = len(stage1_losses)
            s2_end = s1_end + len(stage2_losses)

            # Plot with stage-based coloring
            if stage1_losses:
                ax.plot(
                    range(s1_end),
                    stage1_losses,
                    color=STAGE_COLORS["stage1"],
                    alpha=0.7,
                    linewidth=1.5,
                    label=f"Seed {seed} - Stage 1" if idx == 0 else None,
                )
            if stage2_losses:
                ax.plot(
                    range(s1_end, s2_end),
                    stage2_losses,
                    color=STAGE_COLORS["stage2"],
                    alpha=0.7,
                    linewidth=1.5,
                    label=f"Seed {seed} - Stage 2" if idx == 0 else None,
                )
            if stage3_losses:
                ax.plot(
                    range(s2_end, len(all_losses)),
                    stage3_losses,
                    color=STAGE_COLORS["stage3"],
                    alpha=0.7,
                    linewidth=1.5,
                    label=f"Seed {seed} - Stage 3" if idx == 0 else None,
                )
    else:
        # Single history: plot with solid lines
        stage1_losses = [loss.get("total", 0) for loss in history.get("stage1_losses", [])]
        stage2_losses = [loss.get("total", 0) for loss in history.get("stage2_losses", [])]
        stage3_losses = [loss.get("total", 0) for loss in history.get("stage3_losses", [])]

        # Plot each stage
        if stage1_losses:
            epochs1 = range(len(stage1_losses))
            ax.plot(epochs1, stage1_losses, color=STAGE_COLORS["stage1"],
                    linewidth=2, label="Stage 1: Pretrain")

        if stage2_losses:
            start_epoch = len(stage1_losses)
            epochs2 = range(start_epoch, start_epoch + len(stage2_losses))
            ax.plot(epochs2, stage2_losses, color=STAGE_COLORS["stage2"],
                    linewidth=2, label="Stage 2: Physics Refine")

        if stage3_losses:
            start_epoch = len(stage1_losses) + len(stage2_losses)
            epochs3 = range(start_epoch, start_epoch + len(stage3_losses))
            ax.plot(epochs3, stage3_losses, color=STAGE_COLORS["stage3"],
                    linewidth=2, label="Stage 3: Causal Calibrate")

    ax.set_xlabel("Epoch", fontsize=DEFAULT_FONTSIZE)
    ax.set_ylabel("Total Loss", fontsize=DEFAULT_FONTSIZE)
    ax.set_title(title, fontsize=DEFAULT_FONTSIZE + 2)
    _, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=DEFAULT_FONTSIZE - 1)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.tight_layout()
    _save_or_show(fig, save_path)

    return fig


def plot_convergence_comparison(
    histories: dict[str, dict[str, Any]],
    metric_key: str = "total",
    title: str = "Convergence Comparison",
    save_path: str | None = None,
) -> Figure:
    """Compare convergence of multiple runs (e.g., causal vs no-causal).

    One line per run configuration.

    Args:
        histories: Dict mapping run name -> history dict.
            Each history dict contains stage1_losses, stage2_losses, stage3_losses.
        metric_key: The loss key to plot from each loss dict (default: 'total').
        title: Plot title.
        save_path: Path to save the figure. If None, display instead.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    cmap = plt.colormaps.get_cmap("tab10")

    for idx, (name, history) in enumerate(histories.items()):
        # Extract metric from all stages
        stage1 = [loss.get(metric_key, 0) for loss in history.get("stage1_losses", [])]
        stage2 = [loss.get(metric_key, 0) for loss in history.get("stage2_losses", [])]
        stage3 = [loss.get(metric_key, 0) for loss in history.get("stage3_losses", [])]

        all_losses = stage1 + stage2 + stage3
        epochs = list(range(len(all_losses)))

        color = cmap(idx % 10)
        ax.plot(epochs, all_losses, color=color, linewidth=2, label=name)

    ax.set_xlabel("Epoch", fontsize=DEFAULT_FONTSIZE)
    ax.set_ylabel(f"{metric_key.replace('_', ' ').title()}", fontsize=DEFAULT_FONTSIZE)
    ax.set_title(title, fontsize=DEFAULT_FONTSIZE + 2)
    _, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=DEFAULT_FONTSIZE - 1)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.tight_layout()
    _save_or_show(fig, save_path)

    return fig


def plot_seed_variance(
    histories_by_seed: dict[int, dict[str, Any]],
    stage: str = "stage1",
    title: str = "Seed Variance",
    save_path: str | None = None,
) -> Figure:
    """Show mean +/- std across seeds with shaded confidence band.

    Args:
        histories_by_seed: Dict mapping seed -> history dict.
        stage: Which stage to plot ('stage1', 'stage2', or 'stage3').
        title: Plot title.
        save_path: Path to save the figure. If None, display instead.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    # Collect all loss sequences for the specified stage
    all_losses = []
    max_len = 0

    for _seed, history in histories_by_seed.items():
        losses_key = f"{stage}_losses"
        stage_losses = [loss.get("total", 0) for loss in history.get(losses_key, [])]
        all_losses.append(stage_losses)
        max_len = max(max_len, len(stage_losses))

    if not all_losses or max_len == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        _save_or_show(fig, save_path)
        return fig

    # Pad shorter sequences with NaN for alignment
    padded_losses = []
    for losses in all_losses:
        padded = losses + [float("nan")] * (max_len - len(losses))
        padded_losses.append(padded)

    loss_array = np.array(padded_losses)

    # Compute mean and std, ignoring NaN
    mean_loss = np.nanmean(loss_array, axis=0)
    std_loss = np.nanstd(loss_array, axis=0)
    epochs = np.arange(max_len)

    # Plot mean line
    ax.plot(epochs, mean_loss, color=STAGE_COLORS.get(stage, "#1f77b4"),
            linewidth=2, label=f"Mean (n={len(all_losses)})")

    # Plot shaded confidence band
    ax.fill_between(
        epochs,
        mean_loss - std_loss,
        mean_loss + std_loss,
        color=STAGE_COLORS.get(stage, "#1f77b4"),
        alpha=0.3,
        label="Mean +/- Std"
    )

    ax.set_xlabel("Epoch", fontsize=DEFAULT_FONTSIZE)
    ax.set_ylabel("Total Loss", fontsize=DEFAULT_FONTSIZE)
    ax.set_title(title, fontsize=DEFAULT_FONTSIZE + 2)
    _, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=DEFAULT_FONTSIZE - 1)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    fig.tight_layout()
    _save_or_show(fig, save_path)

    return fig


def plot_metric_comparison(
    results: dict[str, dict[str, float]],
    title: str = "Metric Comparison",
    save_path: str | None = None,
) -> Figure:
    """Create grouped bar chart comparing models across metrics.

    Args:
        results: Dict mapping model name -> dict of metric name -> value.
            Example: {"PINN": {"mse": 0.1, "mae": 0.2}, "Hybrid": {"mse": 0.05, "mae": 0.1}}
        title: Plot title.
        save_path: Path to save the figure. If None, display instead.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    if not results:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        _save_or_show(fig, save_path)
        return fig

    model_names = list(results.keys())
    # Get all unique metrics
    all_metrics: set[str] = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    metric_names = sorted(all_metrics)

    if not metric_names:
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        _save_or_show(fig, save_path)
        return fig

    n_models = len(model_names)
    n_metrics = len(metric_names)

    # Set up bar positions
    bar_width = 0.8 / n_models
    x = np.arange(n_metrics)

    cmap = plt.colormaps.get_cmap("tab10")

    for i, model_name in enumerate(model_names):
        model_results = results[model_name]
        values = [model_results.get(metric, 0) for metric in metric_names]
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width, label=model_name, color=cmap(i % 10))

    ax.set_xlabel("Metric", fontsize=DEFAULT_FONTSIZE)
    ax.set_ylabel("Value", fontsize=DEFAULT_FONTSIZE)
    ax.set_title(title, fontsize=DEFAULT_FONTSIZE + 2)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha="right", fontsize=DEFAULT_FONTSIZE - 1)
    ax.legend(fontsize=DEFAULT_FONTSIZE - 1)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save_or_show(fig, save_path)

    return fig


def plot_residual_distribution(
    residuals: list[float] | Tensor | np.ndarray,
    title: str = "PDE Residual Distribution",
    save_path: str | None = None,
    bins: int = 50,
) -> Figure:
    """Plot histogram of PDE residuals.

    Args:
        residuals: List, tensor, or array of residual values.
        title: Plot title.
        save_path: Path to save the figure. If None, display instead.
        bins: Number of histogram bins.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)

    residuals_np = _to_numpy(residuals).flatten()

    if len(residuals_np) == 0:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        _save_or_show(fig, save_path)
        return fig

    # Plot histogram
    ax.hist(residuals_np, bins=bins, color="#1f77b4", alpha=0.7, edgecolor="black", linewidth=0.5)

    # Add vertical line at 0
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Zero residual")

    # Add statistics annotation
    mean_val = np.mean(residuals_np)
    std_val = np.std(residuals_np)
    stats_text = f"Mean: {mean_val:.4e}\nStd: {std_val:.4e}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=DEFAULT_FONTSIZE - 1,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Residual Value", fontsize=DEFAULT_FONTSIZE)
    ax.set_ylabel("Count", fontsize=DEFAULT_FONTSIZE)
    ax.set_title(title, fontsize=DEFAULT_FONTSIZE + 2)
    ax.legend(fontsize=DEFAULT_FONTSIZE - 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_or_show(fig, save_path)

    return fig


def plot_gate_evolution(
    gate_stats_history: list[dict[str, float]],
    title: str = "Gate Evolution",
    save_path: str | None = None,
) -> Figure:
    """Plot gate mean and std over epochs.

    Args:
        gate_stats_history: List of dicts, one per epoch, containing gate statistics.
            Expected keys: 'mean', 'std' (from CausalGate.get_gate_stats())
        title: Plot title.
        save_path: Path to save the figure. If None, display instead.

    Returns:
        The matplotlib Figure object.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=DEFAULT_FIGSIZE, sharex=True)

    if not gate_stats_history:
        ax1.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax1.transAxes)
        fig.tight_layout()
        _save_or_show(fig, save_path)
        return fig

    epochs = list(range(len(gate_stats_history)))
    means = [_to_numpy(stats.get("mean", 0.5)).item() if hasattr(stats.get("mean", 0.5), "item")
             else stats.get("mean", 0.5) for stats in gate_stats_history]
    stds = [_to_numpy(stats.get("std", 0)).item() if hasattr(stats.get("std", 0), "item")
            else stats.get("std", 0) for stats in gate_stats_history]

    # Plot gate mean
    ax1.plot(epochs, means, color="#1f77b4", linewidth=2, label="Gate Mean")
    ax1.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Equal Weighting")
    ax1.fill_between(epochs, 0, 1, alpha=0.1, color="gray")
    ax1.set_ylabel("Gate Mean", fontsize=DEFAULT_FONTSIZE)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=DEFAULT_FONTSIZE - 1, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title, fontsize=DEFAULT_FONTSIZE + 2)

    # Plot gate std
    ax2.plot(epochs, stds, color="#ff7f0e", linewidth=2, label="Gate Std")
    ax2.set_xlabel("Epoch", fontsize=DEFAULT_FONTSIZE)
    ax2.set_ylabel("Gate Std", fontsize=DEFAULT_FONTSIZE)
    ax2.legend(fontsize=DEFAULT_FONTSIZE - 1, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_or_show(fig, save_path)

    return fig
