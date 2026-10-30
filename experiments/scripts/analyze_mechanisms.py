#!/usr/bin/env python3
"""Mechanism analysis: Why does causal loss hurt? What drives the lambda tradeoff?

Reads existing experiment results JSONs and produces diagnostic figures:
1. Training trajectory comparison (causal vs non-causal)
2. Loss component decomposition per stage
3. Stage 3 causal loss evolution
4. Cross-model comparison (pinn_only vs hybrid_no_gate vs hybrid_full)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def group_runs_by_condition(runs: list[dict]) -> dict[str, list[dict]]:
    """Group raw_results or runs by condition/model_name."""
    groups: dict[str, list[dict]] = {}
    for r in runs:
        key = r.get("condition") or r.get("model_name")
        if key is None:
            continue
        groups.setdefault(key, []).append(r)
    return groups


def extract_loss_series(
    history: dict[str, Any],
    stage: str,
    key: str,
) -> list[float]:
    """Extract a single loss component across all epochs of a stage."""
    losses = history.get(f"{stage}_losses", [])
    return [epoch.get(key, float("nan")) for epoch in losses]


def extract_total_loss_series(
    history: dict[str, Any],
    stage: str,
) -> list[float]:
    return history.get(f"{stage}_total_loss", [])


def get_run_metrics(run: dict[str, Any]) -> dict[str, Any]:
    return run.get("evaluation_metrics") or run.get("metrics") or {}


# ---------------------------------------------------------------------------
# Analysis 1: Training trajectory comparison (causal vs non-causal)
# ---------------------------------------------------------------------------

def plot_training_trajectories(
    runs: list[dict],
    condition_key: str,
    condition_labels: dict[str, str],
    stage_epochs: dict[str, int],
    title: str,
    output_path: Path,
    primary_metric: str = "total",
) -> None:
    """Plot total loss across all stages, comparing conditions."""
    groups = group_runs_by_condition(runs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full trajectory (log scale)
    ax = axes[0]
    colors = {"with_causal": "#d62728", "without_causal": "#1f77b4",
              "pinn_only": "#1f77b4", "hybrid_no_gate": "#ff7f0e",
              "hybrid_full": "#d62728"}

    for cond, cond_runs in groups.items():
        all_series = []
        max_len = 0
        for r in cond_runs:
            h = r["training_history"]
            if primary_metric == "total":
                series = (
                    extract_total_loss_series(h, "stage1")
                    + extract_total_loss_series(h, "stage2")
                    + extract_total_loss_series(h, "stage3")
                )
            else:
                series = (
                    extract_loss_series(h, "stage1", primary_metric)
                    + extract_loss_series(h, "stage2", primary_metric)
                    + extract_loss_series(h, "stage3", primary_metric)
                )
            all_series.append(series)
            max_len = max(max_len, len(series))

        padded = np.full((len(all_series), max_len), np.nan)
        for i, s in enumerate(all_series):
            padded[i, : len(s)] = s

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        epochs = np.arange(max_len)

        label = condition_labels.get(cond, cond)
        color = colors.get(cond)
        ax.plot(epochs, mean, label=label, color=color, linewidth=1.5)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)

    # Mark stage boundaries (use first condition's epoch counts)
    s1_end = stage_epochs.get("stage1", 200)
    s2_end = s1_end + stage_epochs.get("stage2", 500)
    ax.axvline(x=s1_end, color="gray", linestyle="--", alpha=0.5, label="Stage boundary")
    ax.axvline(x=s2_end, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{title} — Full Training Trajectory")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: stage 3 only (linear scale to see divergence)
    ax = axes[1]
    for cond, cond_runs in groups.items():
        all_series = []
        max_len = 0
        for r in cond_runs:
            h = r["training_history"]
            series = extract_total_loss_series(h, "stage3")
            if len(series) == 0:
                continue
            all_series.append(series)
            max_len = max(max_len, len(series))

        if not all_series:
            continue

        padded = np.full((len(all_series), max_len), np.nan)
        for i, s in enumerate(all_series):
            padded[i, : len(s)] = s

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        epochs = np.arange(max_len)

        label = condition_labels.get(cond, cond)
        color = colors.get(cond)
        ax.plot(epochs, mean, label=label, color=color, linewidth=1.5)
        ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Stage 3 Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title(f"{title} — Stage 3 (Causal Calibration) Detail")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Analysis 2: Loss component decomposition per stage
# ---------------------------------------------------------------------------

def plot_loss_decomposition(
    runs: list[dict],
    condition_key: str,
    condition_labels: dict[str, str],
    title: str,
    output_path: Path,
) -> None:
    """Plot individual loss components (data, physics, causal, bc) per stage."""
    groups = group_runs_by_condition(runs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    stage_configs = [
        ("stage1", "Stage 1: Pretrain"),
        ("stage2", "Stage 2: Physics Refinement"),
        ("stage3", "Stage 3: Causal Calibration"),
    ]

    loss_colors = {
        "data_loss": "#1f77b4",
        "physics_loss": "#d62728",
        "causal_loss": "#2ca02c",
        "bc_loss": "#ff7f0e",
        "total": "#333333",
    }

    for ax, (stage, stage_title) in zip(axes, stage_configs, strict=False):
        for cond, cond_runs in groups.items():
            # Collect all available keys from this stage
            all_keys: set[str] = set()
            for r in cond_runs:
                epoch_data = r["training_history"].get(f"{stage}_losses", [])
                if epoch_data:
                    all_keys.update(epoch_data[0].keys())

            for key in sorted(all_keys):
                if key in ("lambda_physics", "epsilon"):
                    continue  # Skip hyperparameters logged as loss keys

                all_series = []
                max_len = 0
                for r in cond_runs:
                    series = extract_loss_series(r["training_history"], stage, key)
                    if series:
                        all_series.append(series)
                        max_len = max(max_len, len(series))

                if not all_series:
                    continue

                padded = np.full((len(all_series), max_len), np.nan)
                for i, s in enumerate(all_series):
                    padded[i, : len(s)] = s

                mean = np.nanmean(padded, axis=0)
                epochs = np.arange(max_len)

                label = f"{condition_labels.get(cond, cond)}: {key}"
                color = loss_colors.get(key)
                ls = "-" if key == "total" else "--"
                ax.plot(epochs, mean, label=label, color=color, linestyle=ls,
                        linewidth=1.2, alpha=0.8)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(stage_title)
        ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Analysis 3: Causal loss evolution and interaction
# ---------------------------------------------------------------------------

def plot_causal_loss_dynamics(
    runs: list[dict],
    condition_key: str,
    condition_labels: dict[str, str],
    title: str,
    output_path: Path,
) -> None:
    """Analyze how causal loss interacts with other loss components in stage 3."""
    groups = group_runs_by_condition(runs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: causal loss magnitude over stage 3
    ax = axes[0]
    for cond, cond_runs in groups.items():
        all_series = []
        max_len = 0
        for r in cond_runs:
            series = extract_loss_series(r["training_history"], "stage3", "causal_loss")
            # Filter out NaN (runs without stage3 or without causal_loss key)
            series = [v for v in series if not np.isnan(v)]
            if series:
                all_series.append(series)
                max_len = max(max_len, len(series))

        if not all_series:
            continue

        padded = np.full((len(all_series), max_len), np.nan)
        for i, s in enumerate(all_series):
            padded[i, : len(s)] = s

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        epochs = np.arange(max_len)

        label = condition_labels.get(cond, cond)
        ax.plot(epochs, mean, label=label, linewidth=1.5)
        ax.fill_between(epochs, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("Stage 3 Epoch")
    ax.set_ylabel("Causal Loss")
    ax.set_title(f"{title} — Causal Loss Magnitude")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Causal loss vs total loss correlation (scatter)
    ax = axes[1]
    for cond, cond_runs in groups.items():
        causal_vals = []
        total_vals = []
        for r in cond_runs:
            cl = extract_loss_series(r["training_history"], "stage3", "causal_loss")
            tl = extract_total_loss_series(r["training_history"], "stage3")
            # Use last 50 epochs for correlation
            for cv, tv in zip(cl[-50:], tl[-50:], strict=False):
                if not np.isnan(cv) and not np.isnan(tv) and cv > 0 and tv > 0:
                    causal_vals.append(cv)
                    total_vals.append(tv)

        if causal_vals:
            label = condition_labels.get(cond, cond)
            ax.scatter(causal_vals, total_vals, alpha=0.3, s=10, label=label)

    ax.set_xlabel("Causal Loss")
    ax.set_ylabel("Total Loss (Stage 3)")
    ax.set_title(f"{title} — Causal vs Total Loss (last 50 epochs)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Analysis 4: Cross-model comparison (rq3_10seed style)
# ---------------------------------------------------------------------------

def plot_cross_model_comparison(
    data: dict[str, Any],
    title: str,
    output_path: Path,
) -> None:
    """Compare pinn_only vs hybrid_no_gate vs hybrid_full training dynamics."""
    groups = group_runs_by_condition(data.get("raw_results", []))

    model_labels = {
        "pinn_only": "PINN-only",
        "hybrid_no_gate": "Hybrid (no gate)",
        "hybrid_full": "Hybrid (full)",
    }
    model_colors = {
        "pinn_only": "#1f77b4",
        "hybrid_no_gate": "#ff7f0e",
        "hybrid_full": "#d62728",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Total loss across all stages
    ax = axes[0, 0]
    for model, runs in groups.items():
        all_series = []
        max_len = 0
        for r in runs:
            h = r["training_history"]
            series = (
                extract_total_loss_series(h, "stage1")
                + extract_total_loss_series(h, "stage2")
                + extract_total_loss_series(h, "stage3")
            )
            all_series.append(series)
            max_len = max(max_len, len(series))

        padded = np.full((len(all_series), max_len), np.nan)
        for i, s in enumerate(all_series):
            padded[i, : len(s)] = s

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        epochs = np.arange(max_len)

        ax.plot(epochs, mean, label=model_labels.get(model, model),
                color=model_colors.get(model), linewidth=1.5)
        ax.fill_between(epochs, mean - std, mean + std,
                        color=model_colors.get(model), alpha=0.1)

    ax.axvline(x=200, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(x=700, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss (All Stages)")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Top-right: Stage 1 comparison
    ax = axes[0, 1]
    for model, runs in groups.items():
        all_series = []
        max_len = 0
        for r in runs:
            series = extract_total_loss_series(r["training_history"], "stage1")
            if series:
                all_series.append(series)
                max_len = max(max_len, len(series))
        if not all_series:
            continue
        padded = np.full((len(all_series), max_len), np.nan)
        for i, s in enumerate(all_series):
            padded[i, : len(s)] = s
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        epochs = np.arange(max_len)
        ax.plot(epochs, mean, label=model_labels.get(model, model),
                color=model_colors.get(model), linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Stage 1: Operator Pretraining")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Stage 2 comparison (physics refinement)
    ax = axes[1, 0]
    for model, runs in groups.items():
        all_series = []
        max_len = 0
        for r in runs:
            series = extract_total_loss_series(r["training_history"], "stage2")
            if series:
                all_series.append(series)
                max_len = max(max_len, len(series))
        if not all_series:
            continue
        padded = np.full((len(all_series), max_len), np.nan)
        for i, s in enumerate(all_series):
            padded[i, : len(s)] = s
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        epochs = np.arange(max_len)
        ax.plot(epochs, mean, label=model_labels.get(model, model),
                color=model_colors.get(model), linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Stage 2: Physics Refinement")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Stage 3 comparison (causal calibration)
    ax = axes[1, 1]
    for model, runs in groups.items():
        all_series = []
        max_len = 0
        for r in runs:
            series = extract_total_loss_series(r["training_history"], "stage3")
            if series:
                all_series.append(series)
                max_len = max(max_len, len(series))
        if not all_series:
            continue
        padded = np.full((len(all_series), max_len), np.nan)
        for i, s in enumerate(all_series):
            padded[i, : len(s)] = s
        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        epochs = np.arange(max_len)
        ax.plot(epochs, mean, label=model_labels.get(model, model),
                color=model_colors.get(model), linewidth=1.5)
        ax.fill_between(epochs, mean - std, mean + std,
                        color=model_colors.get(model), alpha=0.1)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Stage 3: Causal Calibration")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Analysis 5: Quantitative mechanism summary
# ---------------------------------------------------------------------------

def compute_mechanism_summary(
    data: dict[str, Any],
    condition_key: str,
    condition_labels: dict[str, str],
) -> dict[str, Any]:
    """Compute quantitative metrics about training dynamics."""
    groups = group_runs_by_condition(
        data.get("raw_results", []) or data.get("runs", [])
    )

    summary: dict[str, Any] = {}

    for cond, runs in groups.items():
        stats: dict[str, Any] = {"num_seeds": len(runs)}

        # Stage 3 dynamics (only for runs with stage3)
        stage3_final_losses = []
        stage3_initial_losses = []
        causal_final_losses = []
        for r in runs:
            h = r["training_history"]
            s3_total = extract_total_loss_series(h, "stage3")
            s3_causal = extract_loss_series(h, "stage3", "causal_loss")
            if s3_total:
                stage3_initial_losses.append(s3_total[0])
                stage3_final_losses.append(s3_total[-1])
            if s3_causal:
                valid = [v for v in s3_causal if not np.isnan(v)]
                if valid:
                    causal_final_losses.append(valid[-1])

        if stage3_final_losses:
            stats["stage3_initial_loss_mean"] = float(np.mean(stage3_initial_losses))
            stats["stage3_final_loss_mean"] = float(np.mean(stage3_final_losses))
            stats["stage3_improvement_ratio"] = float(
                np.mean(stage3_final_losses) / np.mean(stage3_initial_losses)
                if np.mean(stage3_initial_losses) > 0 else float("nan")
            )
        if causal_final_losses:
            stats["causal_loss_final_mean"] = float(np.mean(causal_final_losses))

        # Final evaluation metrics
        if get_run_metrics(runs[0]):
            for metric_key in ["relative_l2_error", "pde_residual_norm", "mse"]:
                vals = [
                    get_run_metrics(r).get(metric_key, float("nan"))
                    for r in runs
                    if metric_key in get_run_metrics(r)
                ]
                valid = [v for v in vals if not np.isnan(v)]
                if valid:
                    stats[f"eval_{metric_key}_mean"] = float(np.mean(valid))
                    stats[f"eval_{metric_key}_std"] = float(np.std(valid))

        # Training time
        times = [
            r.get("total_training_time", r.get("training_time", 0))
            for r in runs
        ]
        if times:
            stats["training_time_mean"] = float(np.mean(times))

        summary[cond] = stats

    return summary


def print_mechanism_report(
    summaries: dict[str, dict[str, Any]],
    dataset_name: str,
) -> None:
    """Print a human-readable mechanism analysis report."""
    print(f"\n{'='*60}")
    print(f"MECHANISM ANALYSIS: {dataset_name}")
    print(f"{'='*60}")

    for cond, stats in summaries.items():
        print(f"\n  [{cond}] ({stats.get('num_seeds', 0)} seeds)")
        for k, v in sorted(stats.items()):
            if k == "num_seeds":
                continue
            if isinstance(v, float):
                if abs(v) < 0.001 or abs(v) > 1000:
                    print(f"    {k}: {v:.6e}")
                else:
                    print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mechanism analysis for PINN training dynamics"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/analysis/mechanism_figures",
        help="Directory to save analysis figures",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MECHANISM ANALYSIS")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Dataset 1: rq1b_10seed (Burgers1D causal ablation)
    # ------------------------------------------------------------------
    rq1b_path = Path("experiments/results/rq1b_10seed/results.json")
    if rq1b_path.exists():
        print("\n[1/4] Burgers1D causal ablation (rq1b_10seed)")
        rq1b = load_json(rq1b_path)

        plot_training_trajectories(
            runs=rq1b.get("runs", []),
            condition_key="condition",
            condition_labels={
                "with_causal": "With Causal Loss",
                "without_causal": "Without Causal Loss",
            },
            stage_epochs={"stage1": 200, "stage2": 500, "stage3": 200},
            title="Burgers1D (10 seeds)",
            output_path=output_dir / "burgers_training_trajectories.png",
        )

        plot_loss_decomposition(
            runs=rq1b.get("runs", []),
            condition_key="condition",
            condition_labels={
                "with_causal": "With Causal",
                "without_causal": "Without Causal",
            },
            title="Burgers1D Loss Decomposition",
            output_path=output_dir / "burgers_loss_decomposition.png",
        )

        plot_causal_loss_dynamics(
            runs=rq1b.get("runs", []),
            condition_key="condition",
            condition_labels={
                "with_causal": "With Causal",
                "without_causal": "Without Causal",
            },
            title="Burgers1D",
            output_path=output_dir / "burgers_causal_dynamics.png",
        )

        summary = compute_mechanism_summary(
            rq1b,
            condition_key="condition",
            condition_labels={
                "with_causal": "With Causal",
                "without_causal": "Without Causal",
            },
        )
        print_mechanism_report(summary, "Burgers1D (rq1b_10seed)")
    else:
        print(f"  SKIP: {rq1b_path} not found")

    # ------------------------------------------------------------------
    # Dataset 2: rq3_10seed (Heat1D hybrid comparison)
    # ------------------------------------------------------------------
    rq3_path = Path("experiments/results/rq3_10seed/results.json")
    if rq3_path.exists():
        print("\n[2/4] Heat1D hybrid comparison (rq3_10seed)")
        rq3 = load_json(rq3_path)

        plot_cross_model_comparison(
            data=rq3,
            title="Heat1D Model Comparison (10 seeds)",
            output_path=output_dir / "heat1d_cross_model.png",
        )

        summary = compute_mechanism_summary(
            rq3,
            condition_key="model_name",
            condition_labels={
                "pinn_only": "PINN-only",
                "hybrid_no_gate": "Hybrid (no gate)",
                "hybrid_full": "Hybrid (full)",
            },
        )
        print_mechanism_report(summary, "Heat1D (rq3_10seed)")
    else:
        print(f"  SKIP: {rq3_path} not found")

    # ------------------------------------------------------------------
    # Dataset 3: rq3_10seed loss decomposition
    # ------------------------------------------------------------------
    if rq3_path.exists():
        print("\n[3/4] Heat1D loss decomposition (rq3_10seed)")
        rq3 = load_json(rq3_path)

        plot_loss_decomposition(
            runs=rq3.get("raw_results", []),
            condition_key="model_name",
            condition_labels={
                "pinn_only": "PINN-only",
                "hybrid_no_gate": "Hybrid (no gate)",
                "hybrid_full": "Hybrid (full)",
            },
            title="Heat1D Loss Decomposition",
            output_path=output_dir / "heat1d_loss_decomposition.png",
        )

    # ------------------------------------------------------------------
    # Dataset 4: rq4_10seed (Wave2D hybrid comparison)
    # ------------------------------------------------------------------
    rq4_path = Path("experiments/results/rq4_10seed/results.json")
    if rq4_path.exists():
        print("\n[4/4] Wave2D hybrid comparison (rq4_10seed)")
        rq4 = load_json(rq4_path)

        plot_cross_model_comparison(
            data=rq4,
            title="Wave2D Model Comparison (10 seeds)",
            output_path=output_dir / "wave2d_cross_model.png",
        )

        summary = compute_mechanism_summary(
            rq4,
            condition_key="model_name",
            condition_labels={
                "pinn_only": "PINN-only",
                "hybrid_full": "Hybrid (full)",
            },
        )
        print_mechanism_report(summary, "Wave2D (rq4_10seed)")
    else:
        print(f"  SKIP: {rq4_path} not found")

    rq1e_path = Path("experiments/results/rq1e_kdv/results.json")
    if rq1e_path.exists():
        print("\n[5/7] KdV1D causal ablation (rq1e_kdv)")
        rq1e = load_json(rq1e_path)

        plot_training_trajectories(
            runs=rq1e.get("runs", []),
            condition_key="condition",
            condition_labels={
                "with_causal": "With Causal Loss",
                "without_causal": "Without Causal Loss",
            },
            stage_epochs={"stage1": 200, "stage2": 500, "stage3": 200},
            title="KdV1D (5 seeds)",
            output_path=output_dir / "kdv_training_trajectories.png",
        )

        summary = compute_mechanism_summary(
            rq1e,
            condition_key="condition",
            condition_labels={
                "with_causal": "With Causal",
                "without_causal": "Without Causal",
            },
        )
        print_mechanism_report(summary, "KdV1D (rq1e_kdv)")
    else:
        print(f"  SKIP: {rq1e_path} not found")

    rq1f_path = Path("experiments/results/rq1f_allencahn/results.json")
    if rq1f_path.exists():
        print("\n[6/7] AllenCahn1D causal ablation (rq1f_allencahn)")
        rq1f = load_json(rq1f_path)

        plot_training_trajectories(
            runs=rq1f.get("runs", []),
            condition_key="condition",
            condition_labels={
                "with_causal": "With Causal Loss",
                "without_causal": "Without Causal Loss",
            },
            stage_epochs={"stage1": 200, "stage2": 500, "stage3": 200},
            title="AllenCahn1D (5 seeds)",
            output_path=output_dir / "allencahn_training_trajectories.png",
        )

        summary = compute_mechanism_summary(
            rq1f,
            condition_key="condition",
            condition_labels={
                "with_causal": "With Causal",
                "without_causal": "Without Causal",
            },
        )
        print_mechanism_report(summary, "AllenCahn1D (rq1f_allencahn)")
    else:
        print(f"  SKIP: {rq1f_path} not found")

    rq_pino_path = Path("experiments/results/rq_pino_heat1d/results.json")
    if rq_pino_path.exists():
        print("\n[7/7] Heat1D PINO comparison (rq_pino_heat1d)")
        rq_pino = load_json(rq_pino_path)

        plot_cross_model_comparison(
            data=rq_pino,
            title="Heat1D PINO vs PINN vs Hybrid (5 seeds)",
            output_path=output_dir / "heat1d_pino_cross_model.png",
        )

        summary = compute_mechanism_summary(
            rq_pino,
            condition_key="model_name",
            condition_labels={
                "pinn_only": "PINN-only",
                "pino": "PINO",
                "hybrid": "Hybrid",
            },
        )
        print_mechanism_report(summary, "Heat1D PINO comparison (rq_pino_heat1d)")
    else:
        print(f"  SKIP: {rq_pino_path} not found")

    rq1e_hybrid_path = Path("experiments/results/rq1e_kdv_hybrid/results.json")
    if rq1e_hybrid_path.exists():
        print("\n[8/9] KdV1D hybrid comparison (rq1e_kdv_hybrid)")
        rq1e_hybrid = load_json(rq1e_hybrid_path)

        plot_cross_model_comparison(
            data=rq1e_hybrid,
            title="KdV1D Hybrid Comparison (5 seeds)",
            output_path=output_dir / "kdv_hybrid_cross_model.png",
        )

        summary = compute_mechanism_summary(
            rq1e_hybrid,
            condition_key="model_name",
            condition_labels={
                "pinn_only": "PINN-only",
                "hybrid_no_gate": "Hybrid (no gate)",
                "hybrid_full": "Hybrid (full)",
            },
        )
        print_mechanism_report(summary, "KdV1D hybrid comparison (rq1e_kdv_hybrid)")
    else:
        print(f"  SKIP: {rq1e_hybrid_path} not found")

    rq1f_hybrid_path = Path("experiments/results/rq1f_allencahn_hybrid/results.json")
    if rq1f_hybrid_path.exists():
        print("\n[9/9] AllenCahn1D hybrid comparison (rq1f_allencahn_hybrid)")
        rq1f_hybrid = load_json(rq1f_hybrid_path)

        plot_cross_model_comparison(
            data=rq1f_hybrid,
            title="AllenCahn1D Hybrid Comparison (5 seeds)",
            output_path=output_dir / "allencahn_hybrid_cross_model.png",
        )

        summary = compute_mechanism_summary(
            rq1f_hybrid,
            condition_key="model_name",
            condition_labels={
                "pinn_only": "PINN-only",
                "hybrid_no_gate": "Hybrid (no gate)",
                "hybrid_full": "Hybrid (full)",
            },
        )
        print_mechanism_report(summary, "AllenCahn1D hybrid comparison (rq1f_allencahn_hybrid)")
    else:
        print(f"  SKIP: {rq1f_hybrid_path} not found")

    # ------------------------------------------------------------------
    # Save combined summary
    # ------------------------------------------------------------------
    all_summaries = {}
    if rq1b_path.exists():
        all_summaries["burgers1d_causal"] = compute_mechanism_summary(
            load_json(rq1b_path), "condition",
            {"with_causal": "With Causal", "without_causal": "Without Causal"},
        )
    if rq3_path.exists():
        all_summaries["heat1d_hybrid"] = compute_mechanism_summary(
            load_json(rq3_path), "model_name",
            {"pinn_only": "PINN-only", "hybrid_no_gate": "Hybrid (no gate)",
             "hybrid_full": "Hybrid (full)"},
        )
    if rq4_path.exists():
        all_summaries["wave2d_hybrid"] = compute_mechanism_summary(
            load_json(rq4_path), "model_name",
            {"pinn_only": "PINN-only", "hybrid_full": "Hybrid (full)"},
        )
    if rq1e_path.exists():
        all_summaries["kdv1d_causal"] = compute_mechanism_summary(
            load_json(rq1e_path), "condition",
            {"with_causal": "With Causal", "without_causal": "Without Causal"},
        )
    if rq1f_path.exists():
        all_summaries["allencahn1d_causal"] = compute_mechanism_summary(
            load_json(rq1f_path), "condition",
            {"with_causal": "With Causal", "without_causal": "Without Causal"},
        )
    if rq_pino_path.exists():
        all_summaries["heat1d_pino"] = compute_mechanism_summary(
            load_json(rq_pino_path), "model_name",
            {"pinn_only": "PINN-only", "pino": "PINO", "hybrid": "Hybrid"},
        )
    if rq1e_hybrid_path.exists():
        all_summaries["kdv1d_hybrid"] = compute_mechanism_summary(
            load_json(rq1e_hybrid_path), "model_name",
            {"pinn_only": "PINN-only", "hybrid_no_gate": "Hybrid (no gate)",
             "hybrid_full": "Hybrid (full)"},
        )
    if rq1f_hybrid_path.exists():
        all_summaries["allencahn1d_hybrid"] = compute_mechanism_summary(
            load_json(rq1f_hybrid_path), "model_name",
            {"pinn_only": "PINN-only", "hybrid_no_gate": "Hybrid (no gate)",
             "hybrid_full": "Hybrid (full)"},
        )

    summary_path = output_dir / "mechanism_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved combined summary: {summary_path}")

    print(f"\nAll figures saved to: {output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
