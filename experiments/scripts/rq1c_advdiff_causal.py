#!/usr/bin/env python3
"""RQ1c Experiment: CausalLoss ON/OFF comparison on 1D Advection-Diffusion equation.

This experiment investigates the impact of the AdaptiveCausalLoss stage (Stage 3)
on training convergence and stability for the 1D Advection-Diffusion equation:

    ∂u/∂t + c * ∂u/∂x = α * ∂²u/∂x²

We compare:
1. "with_causal": Full 3-stage pipeline including AdaptiveCausalLoss calibration
2. "without_causal": Only Stage 1 + Stage 2 (skipping causal calibration)

Research Question: Does causal loss improve convergence for a PDE with strong
directional propagation (advection velocity c=1.0)?

Hypothesis: AdvectionDiffusion1D has a dominant advection term that creates strong
temporal ordering in the solution. Causal training should be more beneficial here
than for diffusion-dominated PDEs (Heat1D) or nonlinear PDEs (Burgers1D).

Key difference from RQ1/RQ1b: AdvectionDiffusion1D has exact_solution available,
so relative_l2_error can be computed.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from causal_pinn.evaluation.benchmark_runner import (
    BenchmarkConfig,
    BenchmarkRunner,
)
from causal_pinn.models.pinn.base import PINNBase
from causal_pinn.physics.pde.advection_diffusion_1d import AdvectionDiffusion1D
from causal_pinn.training.trainer import HybridTrainer
from causal_pinn.utils.config import TrainingConfig
from causal_pinn.utils.random import set_seed

# Experiment configuration
SEEDS = [42, 123, 456, 789, 1024]
CONDITIONS = ["with_causal", "without_causal"]

# Training hyperparameters (GPU full-scale)
STAGE1_EPOCHS = 200
STAGE1_LR = 1e-3
STAGE2_EPOCHS = 500
STAGE2_LR = 1e-4
STAGE3_EPOCHS_WITH_CAUSAL = 200
STAGE3_EPOCHS_WITHOUT_CAUSAL = 0
STAGE3_LR = 1e-5
CAUSAL_EPSILON = 1.0
BATCH_SIZE = 256
NUM_COLLOCATION_POINTS = 4000
NUM_BOUNDARY_POINTS = 500
DEVICE = "cuda"

# Evaluation configuration
EVAL_N_COLLOCATION = 2000
EVAL_N_BOUNDARY = 200
EVAL_N_INITIAL = 200

# Model architecture
MODEL_INPUT_DIM = 2
MODEL_OUTPUT_DIM = 1
MODEL_HIDDEN_DIMS = [64, 64, 64]

# PDE parameters
ADVDIFF_VELOCITY = 1.0
ADVDIFF_ALPHA = 0.1
ADVDIFF_LENGTH = 1.0
ADVDIFF_T_MAX = 1.0


@dataclass
class SingleRunResult:
    """Results from a single training run."""

    condition: str
    seed: int
    training_history: dict[str, Any]
    evaluation_metrics: dict[str, float]
    total_training_time: float
    final_total_loss: float


def create_model() -> PINNBase:
    """Create a fresh PINN model instance."""
    return PINNBase(
        input_dim=MODEL_INPUT_DIM,
        output_dim=MODEL_OUTPUT_DIM,
        hidden_dims=MODEL_HIDDEN_DIMS,
    )


def create_training_config(condition: str, seed: int) -> TrainingConfig:
    """Create training configuration for a specific condition and seed."""
    stage3_epochs = (
        STAGE3_EPOCHS_WITH_CAUSAL
        if condition == "with_causal"
        else STAGE3_EPOCHS_WITHOUT_CAUSAL
    )

    return TrainingConfig(
        stage1_epochs=STAGE1_EPOCHS,
        stage1_lr=STAGE1_LR,
        stage2_epochs=STAGE2_EPOCHS,
        stage2_lr=STAGE2_LR,
        stage3_epochs=stage3_epochs,
        stage3_lr=STAGE3_LR,
        causal_epsilon=CAUSAL_EPSILON,
        batch_size=BATCH_SIZE,
        num_collocation_points=NUM_COLLOCATION_POINTS,
        num_boundary_points=NUM_BOUNDARY_POINTS,
        seed=seed,
        device=DEVICE,
    )


def run_single_experiment(condition: str, seed: int) -> SingleRunResult:
    """Run a single training experiment for a given condition and seed."""
    print(f"  Running {condition} with seed={seed}...")

    set_seed(seed)

    model = create_model()
    pde_problem = AdvectionDiffusion1D(
        velocity=ADVDIFF_VELOCITY,
        alpha=ADVDIFF_ALPHA,
        length=ADVDIFF_LENGTH,
        t_max=ADVDIFF_T_MAX,
    )

    config = create_training_config(condition, seed)

    trainer = HybridTrainer(
        model=model,
        pde_problem=pde_problem,
        config=config,
        callbacks=None,
        device=DEVICE,
    )

    start_time = time.perf_counter()
    training_history = trainer.train()
    total_training_time = time.perf_counter() - start_time

    # Compute final total loss
    final_loss = 0.0
    if training_history.get("stage1_total_loss"):
        final_loss += training_history["stage1_total_loss"][-1]
    if training_history.get("stage2_total_loss"):
        final_loss += training_history["stage2_total_loss"][-1]
    if training_history.get("stage3_total_loss"):
        final_loss += training_history["stage3_total_loss"][-1]

    # Evaluate model
    benchmark_config = BenchmarkConfig(
        n_collocation=EVAL_N_COLLOCATION,
        n_boundary=EVAL_N_BOUNDARY,
        n_initial=EVAL_N_INITIAL,
        include_physics_consistency=False,
        device=DEVICE,
    )
    runner = BenchmarkRunner(config=benchmark_config)
    evaluation_metrics = runner.run_single(model, pde_problem)

    eval_metrics_serializable = {
        k: float(v) if isinstance(v, (int, float, torch.Tensor)) else v
        for k, v in evaluation_metrics.items()
    }

    return SingleRunResult(
        condition=condition,
        seed=seed,
        training_history=training_history,
        evaluation_metrics=eval_metrics_serializable,
        total_training_time=total_training_time,
        final_total_loss=final_loss,
    )


def aggregate_results(results: list[SingleRunResult]) -> dict[str, dict[str, Any]]:
    """Aggregate results by condition."""
    summary: dict[str, dict[str, Any]] = {}

    for condition in CONDITIONS:
        condition_results = [r for r in results if r.condition == condition]

        if not condition_results:
            continue

        rel_l2_errors = [
            r.evaluation_metrics.get("relative_l2_error", float("nan"))
            for r in condition_results
        ]
        mses = [
            r.evaluation_metrics.get("mse", float("nan"))
            for r in condition_results
        ]
        maes = [
            r.evaluation_metrics.get("mae", float("nan"))
            for r in condition_results
        ]
        max_errors = [
            r.evaluation_metrics.get("max_error", float("nan"))
            for r in condition_results
        ]
        pde_residuals = [
            r.evaluation_metrics.get("pde_residual_norm", float("nan"))
            for r in condition_results
        ]
        bc_violations = [
            r.evaluation_metrics.get("bc_violation", float("nan"))
            for r in condition_results
        ]
        ic_violations = [
            r.evaluation_metrics.get("ic_violation", float("nan"))
            for r in condition_results
        ]
        final_losses = [r.final_total_loss for r in condition_results]
        training_times = [r.total_training_time for r in condition_results]

        summary[condition] = {
            "n_runs": len(condition_results),
            "rel_l2_mean": float(np.nanmean(rel_l2_errors)),
            "rel_l2_std": float(np.nanstd(rel_l2_errors)),
            "mse_mean": float(np.nanmean(mses)),
            "mse_std": float(np.nanstd(mses)),
            "mae_mean": float(np.nanmean(maes)),
            "mae_std": float(np.nanstd(maes)),
            "max_error_mean": float(np.nanmean(max_errors)),
            "max_error_std": float(np.nanstd(max_errors)),
            "pde_residual_mean": float(np.nanmean(pde_residuals)),
            "pde_residual_std": float(np.nanstd(pde_residuals)),
            "bc_violation_mean": float(np.nanmean(bc_violations)),
            "bc_violation_std": float(np.nanstd(bc_violations)),
            "ic_violation_mean": float(np.nanmean(ic_violations)),
            "ic_violation_std": float(np.nanstd(ic_violations)),
            "final_loss_mean": float(np.mean(final_losses)),
            "final_loss_std": float(np.std(final_losses)),
            "training_time_mean": float(np.mean(training_times)),
            "training_time_std": float(np.std(training_times)),
            "rel_l2_values": [float(x) if not np.isnan(x) else None for x in rel_l2_errors],
            "mse_values": [float(x) if not np.isnan(x) else None for x in mses],
            "mae_values": [float(x) if not np.isnan(x) else None for x in maes],
            "max_error_values": [float(x) if not np.isnan(x) else None for x in max_errors],
            "pde_residual_values": [float(x) for x in pde_residuals],
            "final_loss_values": [float(x) for x in final_losses],
        }

    return summary


def plot_convergence_curves(
    results: list[SingleRunResult],
    output_path: Path,
) -> None:
    """Plot training loss curves (mean +/- std) for both conditions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"with_causal": "blue", "without_causal": "red"}

    for condition in CONDITIONS:
        condition_results = [r for r in results if r.condition == condition]

        if not condition_results:
            continue

        all_losses = []
        max_len = 0

        for r in condition_results:
            total_loss = (
                r.training_history.get("stage1_total_loss", [])
                + r.training_history.get("stage2_total_loss", [])
                + r.training_history.get("stage3_total_loss", [])
            )
            all_losses.append(total_loss)
            max_len = max(max_len, len(total_loss))

        padded_losses = np.full((len(all_losses), max_len), np.nan)
        for i, loss_seq in enumerate(all_losses):
            padded_losses[i, : len(loss_seq)] = loss_seq

        mean_loss = np.nanmean(padded_losses, axis=0)
        std_loss = np.nanstd(padded_losses, axis=0)
        epochs = np.arange(max_len)

        ax.plot(epochs, mean_loss, label=condition, color=colors[condition], linewidth=2)
        ax.fill_between(
            epochs,
            mean_loss - std_loss,
            mean_loss + std_loss,
            color=colors[condition],
            alpha=0.2,
        )

        if condition == "with_causal":
            stage1_end = STAGE1_EPOCHS
            stage2_end = STAGE1_EPOCHS + STAGE2_EPOCHS
            ax.axvline(
                x=stage1_end,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="Stage boundary",
            )
            ax.axvline(x=stage2_end, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Epoch (all stages combined)")
    ax.set_ylabel("Total Loss")
    ax.set_title(
        "RQ1c: Training Convergence Curves (AdvectionDiffusion1D)\n"
        "(Mean ± Std across seeds)"
    )
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_metrics_comparison(
    summary: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot grouped bar chart comparing final metrics between conditions."""
    metrics = ["rel_l2", "mse", "pde_residual", "bc_violation", "ic_violation"]
    metric_labels = ["Rel L2", "MSE", "PDE Residual", "BC Violation", "IC Violation"]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, condition in enumerate(CONDITIONS):
        cond_data = summary.get(condition, {})
        means = [cond_data.get(f"{m}_mean", float("nan")) for m in metrics]
        stds = [cond_data.get(f"{m}_std", 0) or 0 for m in metrics]

        # Replace nan with 0 for plotting
        means_plot = [m if not np.isnan(m) else 0 for m in means]
        stds_plot = [s if not np.isnan(s) else 0 for s in stds]

        color = "blue" if condition == "with_causal" else "red"
        offset = -width / 2 if i == 0 else width / 2
        ax.bar(
            x + offset,
            means_plot,
            width,
            label=condition,
            yerr=stds_plot,
            capsize=3,
            color=color,
            alpha=0.7,
        )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value (log scale)")
    ax.set_title("RQ1c: Final Metrics Comparison (AdvectionDiffusion1D)\n(Mean ± Std)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_seed_variance(
    summary: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot box plot showing distribution of rel_l2 across seeds."""
    fig, ax = plt.subplots(figsize=(8, 6))

    data = []
    labels = []

    for condition in CONDITIONS:
        cond_data = summary.get(condition, {})
        values = cond_data.get("rel_l2_values", [])
        valid_values = [v for v in values if v is not None and not np.isnan(v)]
        if valid_values:
            data.append(valid_values)
            labels.append(condition)

    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

        colors_map = {"with_causal": "lightblue", "without_causal": "lightcoral"}
        for patch, label in zip(bp["boxes"], labels, strict=True):
            patch.set_facecolor(colors_map.get(label, "white"))
            patch.set_alpha(0.7)

        for i, (d, label) in enumerate(zip(data, labels, strict=True)):
            x_jitter = np.random.normal(i + 1, 0.04, size=len(d))
            ax.scatter(x_jitter, d, alpha=0.6, color=colors_map.get(label, "gray"), s=30)

    ax.set_xlabel("Condition")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("RQ1c: Seed-to-Seed Variance in Relative L2 Error (AdvectionDiffusion1D)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary_table(summary: dict[str, dict[str, Any]]) -> None:
    """Print summary table to console."""
    print("\n" + "=" * 120)
    print("RQ1c EXPERIMENT RESULTS: CausalLoss ON/OFF Comparison (AdvectionDiffusion1D)")
    print("=" * 120)
    print()

    header = (
        f"{'Condition':<20} | {'Rel L2 (mean±std)':<25} | {'MSE (mean±std)':<25} | "
        f"{'PDE Res (mean±std)':<25} | {'Time (s)':<12}"
    )
    print(header)
    print("-" * 120)

    for condition in CONDITIONS:
        s = summary.get(condition, {})

        rel_l2_mean = s.get("rel_l2_mean", float("nan"))
        rel_l2_std = s.get("rel_l2_std", float("nan"))
        rel_l2 = f"{rel_l2_mean:.4e} ± {rel_l2_std:.4e}"

        mse_mean = s.get("mse_mean", float("nan"))
        mse_std = s.get("mse_std", float("nan"))
        mse = f"{mse_mean:.4e} ± {mse_std:.4e}"

        pde_res_mean = s.get("pde_residual_mean", float("nan"))
        pde_res_std = s.get("pde_residual_std", float("nan"))
        pde_res = f"{pde_res_mean:.4e} ± {pde_res_std:.4e}"

        time_mean = s.get("training_time_mean", 0.0)
        time_std = s.get("training_time_std", 0.0)
        time_s = f"{time_mean:.1f} ± {time_std:.1f}"

        print(
            f"{condition:<20} | {rel_l2:<25} | {mse:<25} | "
            f"{pde_res:<25} | {time_s:<12}"
        )

    print("-" * 120)
    print()

    # Additional metrics
    print(f"{'Condition':<20} | {'MAE (mean±std)':<25} | {'Max Error (mean±std)':<25} | "
          f"{'BC Viol (mean±std)':<25} | {'IC Viol (mean±std)':<25}")
    print("-" * 120)

    for condition in CONDITIONS:
        s = summary.get(condition, {})

        mae_mean = s.get("mae_mean", float("nan"))
        mae_std = s.get("mae_std", float("nan"))
        mae = f"{mae_mean:.4e} ± {mae_std:.4e}"

        max_err_mean = s.get("max_error_mean", float("nan"))
        max_err_std = s.get("max_error_std", float("nan"))
        max_err = f"{max_err_mean:.4e} ± {max_err_std:.4e}"

        bc_mean = s.get("bc_violation_mean", float("nan"))
        bc_std = s.get("bc_violation_std", float("nan"))
        bc = f"{bc_mean:.4e} ± {bc_std:.4e}"

        ic_mean = s.get("ic_violation_mean", float("nan"))
        ic_std = s.get("ic_violation_std", float("nan"))
        ic = f"{ic_mean:.4e} ± {ic_std:.4e}"

        print(
            f"{condition:<20} | {mae:<25} | {max_err:<25} | "
            f"{bc:<25} | {ic:<25}"
        )

    print("=" * 120)


def main() -> None:
    """Main experiment entry point."""
    parser = argparse.ArgumentParser(
        description="RQ1c Experiment: CausalLoss ON/OFF comparison on AdvectionDiffusion1D",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/rq1c",
        help="Output directory for results and plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RQ1c Experiment: CausalLoss ON/OFF on AdvectionDiffusion1D")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  PDE: AdvectionDiffusion1D (velocity={ADVDIFF_VELOCITY}, alpha={ADVDIFF_ALPHA})")
    print(f"  Seeds: {SEEDS}")
    print(f"  Conditions: {CONDITIONS}")
    print(f"  Total runs: {len(SEEDS) * len(CONDITIONS)}")
    print(f"  Output directory: {output_dir}")
    print()

    all_results: list[SingleRunResult] = []

    for condition in CONDITIONS:
        print(f"\n{'=' * 40}")
        print(f"Condition: {condition}")
        print(f"{'=' * 40}")

        for seed in SEEDS:
            result = run_single_experiment(condition, seed)
            all_results.append(result)
            rel_l2 = result.evaluation_metrics.get("relative_l2_error", "N/A")
            pde_res = result.evaluation_metrics.get("pde_residual_norm", "N/A")
            if isinstance(rel_l2, float):
                print(
                    f"    Completed: time={result.total_training_time:.2f}s, "
                    f"rel_l2={rel_l2:.4e}, pde_res={pde_res:.4e}"
                )
            else:
                print(f"    Completed: time={result.total_training_time:.2f}s")

    # Aggregate results
    print("\n" + "=" * 80)
    print("Aggregating results...")
    summary = aggregate_results(all_results)

    # Save results
    output = {
        "experiment": "rq1c_advdiff_causal_ablation",
        "description": "CausalLoss ON/OFF comparison on 1D Advection-Diffusion equation",
        "pde": "AdvectionDiffusion1D",
        "pde_config": {
            "velocity": ADVDIFF_VELOCITY,
            "alpha": ADVDIFF_ALPHA,
            "length": ADVDIFF_LENGTH,
            "t_max": ADVDIFF_T_MAX,
        },
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "training_config": {
            "stage1_epochs": STAGE1_EPOCHS,
            "stage1_lr": STAGE1_LR,
            "stage2_epochs": STAGE2_EPOCHS,
            "stage2_lr": STAGE2_LR,
            "stage3_epochs_with_causal": STAGE3_EPOCHS_WITH_CAUSAL,
            "stage3_epochs_without_causal": STAGE3_EPOCHS_WITHOUT_CAUSAL,
            "stage3_lr": STAGE3_LR,
            "causal_epsilon": CAUSAL_EPSILON,
            "batch_size": BATCH_SIZE,
            "num_collocation_points": NUM_COLLOCATION_POINTS,
            "num_boundary_points": NUM_BOUNDARY_POINTS,
            "device": DEVICE,
        },
        "model_config": {
            "input_dim": MODEL_INPUT_DIM,
            "output_dim": MODEL_OUTPUT_DIM,
            "hidden_dims": MODEL_HIDDEN_DIMS,
        },
        "evaluation_config": {
            "n_collocation": EVAL_N_COLLOCATION,
            "n_boundary": EVAL_N_BOUNDARY,
            "n_initial": EVAL_N_INITIAL,
        },
        "summary": summary,
        "runs": [
            {
                "condition": r.condition,
                "seed": r.seed,
                "evaluation_metrics": r.evaluation_metrics,
                "total_training_time": r.total_training_time,
                "final_total_loss": r.final_total_loss,
            }
            for r in all_results
        ],
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved results to: {results_path}")

    # Generate plots
    print("\nGenerating plots...")

    plot_convergence_curves(all_results, output_dir / "rq1c_convergence_curves.png")
    print("  Saved: rq1c_convergence_curves.png")

    plot_final_metrics_comparison(summary, output_dir / "rq1c_final_metrics_comparison.png")
    print("  Saved: rq1c_final_metrics_comparison.png")

    plot_seed_variance(summary, output_dir / "rq1c_seed_variance.png")
    print("  Saved: rq1c_seed_variance.png")

    # Print summary table
    print_summary_table(summary)

    print("\nExperiment complete!")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
