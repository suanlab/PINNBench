#!/usr/bin/env python3
"""Create a compact paper figure for budget-aware routing results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SELECTOR_LABELS = {
    "random_forest": "RF",
    "physics_final": "Phys.\nfinal",
    "training_prior": "Train\nprior",
    "linear_regression": "Linear",
    "random_expected": "Random",
}

SELECTOR_ORDER = [
    "random_forest",
    "physics_final",
    "training_prior",
    "linear_regression",
    "random_expected",
]

SELECTOR_COLORS = {
    "random_forest": "#1f77b4",
    "physics_final": "#ff7f0e",
    "training_prior": "#2ca02c",
    "linear_regression": "#9467bd",
    "random_expected": "#7f7f7f",
}


def load_json(path: Path) -> dict[str, object]:
    with open(path) as handle:
        return json.load(handle)


def selector_metrics(
    data: dict[str, object],
    *,
    window: str,
    protocol: str,
    selector: str,
) -> tuple[float, float, float]:
    summary = data["windows"][window]["routing"][protocol]["selectors"][selector]
    regret = float(summary["mean_regret"])
    accuracy = float(summary["selection_accuracy"])
    savings = float(summary["mean_budget_savings"])
    return regret, accuracy, savings


def add_regret_panel(
    ax: plt.Axes,
    data: dict[str, object],
    *,
    window: str,
    protocol: str,
    title: str,
) -> None:
    regrets = []
    accuracies = []
    colors = []
    labels = []
    for selector in SELECTOR_ORDER:
        regret, accuracy, _ = selector_metrics(
            data,
            window=window,
            protocol=protocol,
            selector=selector,
        )
        regrets.append(regret)
        accuracies.append(accuracy)
        colors.append(SELECTOR_COLORS[selector])
        labels.append(SELECTOR_LABELS[selector])

    x = np.arange(len(labels))
    bars = ax.bar(x, regrets, color=colors, width=0.72)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean regret")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ymax = max(regrets) if regrets else 0.0
    text_offset = max(0.01, ymax * 0.04)
    for bar, accuracy in zip(bars, accuracies, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + text_offset,
            f"{accuracy * 100:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylim(0.0, max(0.02, ymax * 1.35 + text_offset))


def add_budget_panel(
    ax: plt.Axes,
    residual_data: dict[str, object],
    rel_l2_data: dict[str, object],
) -> None:
    windows_residual = ["10", "20", "50", "100"]
    windows_rel_l2 = ["10", "50"]
    x_residual = [int(window) for window in windows_residual]
    x_rel_l2 = [int(window) for window in windows_rel_l2]

    residual_savings = [
        selector_metrics(
            residual_data,
            window=window,
            protocol="strict",
            selector="random_forest",
        )[2]
        for window in windows_residual
    ]
    rel_l2_savings = [
        selector_metrics(
            rel_l2_data,
            window=window,
            protocol="relaxed",
            selector="random_forest",
        )[2]
        for window in windows_rel_l2
    ]

    ax.plot(
        x_residual,
        residual_savings,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="Residual routing",
    )
    ax.plot(
        x_rel_l2,
        rel_l2_savings,
        marker="s",
        linewidth=2,
        color="#d62728",
        label="Rel. $L_2$ routing",
    )
    ax.set_xticks(x_residual)
    ax.set_xlabel("Probe epochs")
    ax.set_ylabel("Budget saved")
    ax.set_title("Stage-2 budget savings", fontsize=10)
    ax.set_ylim(0.45, 0.67)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower left")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    residual_path = root / "experiments/results/rq_early_predictor/results.json"
    rel_l2_path = root / "experiments/results/rq_early_predictor_rel_l2/results.json"
    output_dir = residual_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    residual_data = load_json(residual_path)
    rel_l2_data = load_json(rel_l2_path)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)
    add_regret_panel(
        axes[0],
        residual_data,
        window="10",
        protocol="strict",
        title="Residual routing (strict, 10 epochs)",
    )
    add_regret_panel(
        axes[1],
        rel_l2_data,
        window="10",
        protocol="relaxed",
        title="Rel. $L_2$ routing (suite, 10 epochs)",
    )
    add_budget_panel(axes[2], residual_data, rel_l2_data)

    png_path = output_dir / "rq_early_predictor_routing_summary.png"
    pdf_path = output_dir / "rq_early_predictor_routing_summary.pdf"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
