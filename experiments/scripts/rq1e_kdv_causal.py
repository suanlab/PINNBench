#!/usr/bin/env python3
"""RQ1e Experiment: CausalLoss ON/OFF comparison on 1D KdV equation."""

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

from causal_pinn.evaluation.benchmark_runner import BenchmarkConfig, BenchmarkRunner
from causal_pinn.models.pinn.base import PINNBase
from causal_pinn.physics.pde.kdv_1d import KdV1D
from causal_pinn.training.trainer import HybridTrainer
from causal_pinn.utils.config import TrainingConfig
from causal_pinn.utils.random import set_seed

DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
CONDITIONS = ["with_causal", "without_causal"]

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

EVAL_N_COLLOCATION = 2000
EVAL_N_BOUNDARY = 200
EVAL_N_INITIAL = 200

MODEL_INPUT_DIM = 2
MODEL_OUTPUT_DIM = 1
MODEL_HIDDEN_DIMS = [64, 64, 64, 64]

KDV_DISPERSION = 1.0
KDV_LENGTH = 20.0
KDV_T_MAX = 0.5
KDV_SOLITON_AMPLITUDE = 3.0
KDV_SOLITON_CENTER = 10.0


@dataclass
class SingleRunResult:
    condition: str
    seed: int
    training_history: dict[str, Any]
    evaluation_metrics: dict[str, float]
    total_training_time: float
    final_total_loss: float


@dataclass
class AggregatedResults:
    runs: list[SingleRunResult]
    summary: dict[str, dict[str, Any]]


def create_model() -> PINNBase:
    return PINNBase(
        input_dim=MODEL_INPUT_DIM,
        output_dim=MODEL_OUTPUT_DIM,
        hidden_dims=MODEL_HIDDEN_DIMS,
    )


def create_training_config(condition: str, seed: int) -> TrainingConfig:
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
    print(f"  Running {condition} with seed={seed}...")
    set_seed(seed)

    model = create_model()
    pde_problem = KdV1D(
        dispersion=KDV_DISPERSION,
        length=KDV_LENGTH,
        t_max=KDV_T_MAX,
        soliton_amplitude=KDV_SOLITON_AMPLITUDE,
        soliton_center=KDV_SOLITON_CENTER,
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

    final_loss = 0.0
    if training_history.get("stage1_total_loss"):
        final_loss += training_history["stage1_total_loss"][-1]
    if training_history.get("stage2_total_loss"):
        final_loss += training_history["stage2_total_loss"][-1]
    if training_history.get("stage3_total_loss"):
        final_loss += training_history["stage3_total_loss"][-1]

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


def aggregate_results(results: list[SingleRunResult]) -> AggregatedResults:
    summary = {}

    for condition in CONDITIONS:
        condition_results = [r for r in results if r.condition == condition]
        if not condition_results:
            continue

        rel_l2_errors = [
            r.evaluation_metrics.get("relative_l2_error", float("nan"))
            for r in condition_results
        ]
        mses = [r.evaluation_metrics.get("mse", float("nan")) for r in condition_results]
        maes = [r.evaluation_metrics.get("mae", float("nan")) for r in condition_results]
        max_errors = [
            r.evaluation_metrics.get("max_error", float("nan"))
            for r in condition_results
        ]
        pde_residuals = [
            r.evaluation_metrics.get("pde_residual_norm", float("nan"))
            for r in condition_results
        ]
        bc_errors = [
            r.evaluation_metrics.get("boundary_condition_error", float("nan"))
            for r in condition_results
        ]
        ic_errors = [
            r.evaluation_metrics.get("initial_condition_error", float("nan"))
            for r in condition_results
        ]
        final_losses = [r.final_total_loss for r in condition_results]
        training_times = [r.total_training_time for r in condition_results]

        rel_l2_has_values = not all(np.isnan(rel_l2_errors))
        mse_has_values = not all(np.isnan(mses))
        mae_has_values = not all(np.isnan(maes))
        max_error_has_values = not all(np.isnan(max_errors))

        summary[condition] = {
            "n_runs": len(condition_results),
            "rel_l2_mean": float(np.nanmean(rel_l2_errors)) if rel_l2_has_values else None,
            "rel_l2_std": float(np.nanstd(rel_l2_errors)) if rel_l2_has_values else None,
            "mse_mean": float(np.nanmean(mses)) if mse_has_values else None,
            "mse_std": float(np.nanstd(mses)) if mse_has_values else None,
            "mae_mean": float(np.nanmean(maes)) if mae_has_values else None,
            "mae_std": float(np.nanstd(maes)) if mae_has_values else None,
            "max_error_mean": float(np.nanmean(max_errors)) if max_error_has_values else None,
            "max_error_std": float(np.nanstd(max_errors)) if max_error_has_values else None,
            "pde_residual_mean": float(np.nanmean(pde_residuals)),
            "pde_residual_std": float(np.nanstd(pde_residuals)),
            "bc_error_mean": float(np.nanmean(bc_errors)) if not all(np.isnan(bc_errors)) else None,
            "bc_error_std": float(np.nanstd(bc_errors)) if not all(np.isnan(bc_errors)) else None,
            "ic_error_mean": float(np.nanmean(ic_errors)) if not all(np.isnan(ic_errors)) else None,
            "ic_error_std": float(np.nanstd(ic_errors)) if not all(np.isnan(ic_errors)) else None,
            "final_loss_mean": float(np.mean(final_losses)),
            "final_loss_std": float(np.std(final_losses)),
            "training_time_mean": float(np.mean(training_times)),
            "training_time_std": float(np.std(training_times)),
            "rel_l2_values": [float(x) if not np.isnan(x) else None for x in rel_l2_errors],
            "mse_values": [float(x) if not np.isnan(x) else None for x in mses],
            "mae_values": [float(x) if not np.isnan(x) else None for x in maes],
            "max_error_values": [float(x) if not np.isnan(x) else None for x in max_errors],
            "pde_residual_values": [float(x) for x in pde_residuals],
            "bc_error_values": [float(x) if not np.isnan(x) else None for x in bc_errors],
            "ic_error_values": [float(x) if not np.isnan(x) else None for x in ic_errors],
            "final_loss_values": [float(x) for x in final_losses],
        }

    return AggregatedResults(runs=results, summary=summary)


def plot_convergence_curves(results: list[SingleRunResult], output_path: Path) -> None:
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
            ax.axvline(x=stage1_end, color=colors[condition], linestyle="--", alpha=0.5)
            ax.axvline(x=stage2_end, color=colors[condition], linestyle="--", alpha=0.5)

    ax.set_xlabel("Epoch (all stages combined)")
    ax.set_ylabel("Total Loss")
    ax.set_title("RQ1e: Training Convergence Curves (KdV1D)\n(Mean +/- Std across seeds)")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_metrics_comparison(aggregated: AggregatedResults, output_path: Path) -> None:
    metrics = ["pde_residual", "bc_error", "ic_error"]
    metric_labels = ["PDE Residual", "BC Error", "IC Error"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    means_with = []
    stds_with = []
    means_without = []
    stds_without = []

    for metric in metrics:
        with_causal = aggregated.summary.get("with_causal", {})
        without_causal = aggregated.summary.get("without_causal", {})
        with_mean = with_causal.get(f"{metric}_mean")
        with_std = with_causal.get(f"{metric}_std", 0) or 0
        without_mean = without_causal.get(f"{metric}_mean")
        without_std = without_causal.get(f"{metric}_std", 0) or 0
        means_with.append(with_mean if with_mean is not None else 0)
        stds_with.append(with_std)
        means_without.append(without_mean if without_mean is not None else 0)
        stds_without.append(without_std)

    ax.bar(
        x - width / 2,
        means_with,
        width,
        label="with_causal",
        yerr=stds_with,
        capsize=3,
        color="blue",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        means_without,
        width,
        label="without_causal",
        yerr=stds_without,
        capsize=3,
        color="red",
        alpha=0.7,
    )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title("RQ1e: Final Metrics Comparison (KdV1D)\n(Mean +/- Std Error Bars)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_seed_variance(aggregated: AggregatedResults, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    data = []
    labels = []

    for condition in CONDITIONS:
        condition_summary = aggregated.summary.get(condition, {})
        values = condition_summary.get("pde_residual_values", [])
        if values:
            data.append(values)
            labels.append(condition)

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    colors = {"with_causal": "lightblue", "without_causal": "lightcoral"}
    for patch, label in zip(bp["boxes"], labels, strict=False):
        patch.set_facecolor(colors.get(label, "white"))
        patch.set_alpha(0.7)

    for i, (d, label) in enumerate(zip(data, labels, strict=False)):
        x_jitter = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x_jitter, d, alpha=0.6, color=colors.get(label, "gray"), s=30)

    ax.set_xlabel("Condition")
    ax.set_ylabel("PDE Residual Norm")
    ax.set_title("RQ1e: Seed-to-Seed Variance in Final PDE Residual (KdV1D)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary_table(aggregated: AggregatedResults) -> None:
    print("\n" + "=" * 90)
    print("RQ1e EXPERIMENT RESULTS: CausalLoss ON/OFF Comparison (KdV1D)")
    print("=" * 90)
    print()

    header = (
        f"{'Condition':<20} | {'PDE Residual (mean +/- std)':<30} | "
        f"{'Final Loss (mean +/- std)':<30}"
    )
    print(header)
    print("-" * 90)

    for condition in CONDITIONS:
        summary = aggregated.summary.get(condition, {})
        pde_mean = summary.get("pde_residual_mean", float("nan"))
        pde_std = summary.get("pde_residual_std", float("nan"))
        loss_mean = summary.get("final_loss_mean", float("nan"))
        loss_std = summary.get("final_loss_std", float("nan"))
        pde_str = f"{pde_mean:.4e} +/- {pde_std:.4e}"
        loss_str = f"{loss_mean:.4e} +/- {loss_std:.4e}"
        print(f"{condition:<20} | {pde_str:<30} | {loss_str:<30}")

    print("-" * 90)
    print()
    print("Boundary and Initial Condition Errors:")
    print(f"{'Condition':<20} | {'BC Error (mean +/- std)':<30} | {'IC Error (mean +/- std)':<30}")
    print("-" * 90)

    for condition in CONDITIONS:
        summary = aggregated.summary.get(condition, {})
        bc_mean = summary.get("bc_error_mean")
        bc_std = summary.get("bc_error_std", 0) or 0
        ic_mean = summary.get("ic_error_mean")
        ic_std = summary.get("ic_error_std", 0) or 0
        bc_str = f"{bc_mean:.4e} +/- {bc_std:.4e}" if bc_mean is not None else "N/A"
        ic_str = f"{ic_mean:.4e} +/- {ic_std:.4e}" if ic_mean is not None else "N/A"
        print(f"{condition:<20} | {bc_str:<30} | {ic_str:<30}")

    print("-" * 90)


def save_results(aggregated: AggregatedResults, output_path: Path, seeds: list[int]) -> None:
    output = {
        "experiment": "rq1e_kdv_causal_ablation",
        "description": "CausalLoss ON/OFF comparison on 1D KdV equation",
        "pde": "KdV1D",
        "pde_config": {
            "dispersion": KDV_DISPERSION,
            "length": KDV_LENGTH,
            "t_max": KDV_T_MAX,
            "soliton_amplitude": KDV_SOLITON_AMPLITUDE,
            "soliton_center": KDV_SOLITON_CENTER,
        },
        "seeds": seeds,
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
        "summary": aggregated.summary,
        "runs": [
            {
                "condition": r.condition,
                "seed": r.seed,
                "training_history": r.training_history,
                "evaluation_metrics": r.evaluation_metrics,
                "total_training_time": r.total_training_time,
                "final_total_loss": r.final_total_loss,
            }
            for r in aggregated.runs
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RQ1e Experiment: CausalLoss ON/OFF comparison on KdV1D"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/rq1e_kdv",
        help="Output directory for results and plots (default: experiments/results/rq1e_kdv)",
    )
    parser.add_argument("--num-seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit seed list")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS[:args.num_seeds]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RQ1e Experiment: CausalLoss ON/OFF Comparison on 1D KdV Equation")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  PDE: KdV1D (dispersion={KDV_DISPERSION}, length={KDV_LENGTH}, t_max={KDV_T_MAX})")
    print(f"  Seeds: {seeds}")
    print(f"  Conditions: {CONDITIONS}")
    print(f"  Total runs: {len(seeds) * len(CONDITIONS)}")
    print(f"  Output directory: {output_dir}")
    print()

    all_results: list[SingleRunResult] = []
    for condition in CONDITIONS:
        print(f"\n{'=' * 40}")
        print(f"Condition: {condition}")
        print(f"{'=' * 40}")

        for seed in seeds:
            result = run_single_experiment(condition, seed)
            all_results.append(result)
            pde_res = result.evaluation_metrics.get("pde_residual_norm", "N/A")
            if isinstance(pde_res, float):
                print(
                    f"    Completed: time={result.total_training_time:.2f}s, "
                    f"pde_residual={pde_res:.4e}"
                )
            else:
                print(f"    Completed: time={result.total_training_time:.2f}s")

    print("\n" + "=" * 80)
    print("Aggregating results...")
    aggregated = aggregate_results(all_results)

    results_path = output_dir / "results.json"
    save_results(aggregated, results_path, seeds)
    print(f"Saved results to: {results_path}")

    print("\nGenerating plots...")
    plot_convergence_curves(all_results, output_dir / "rq1e_convergence_curves.png")
    print("  Saved: rq1e_convergence_curves.png")
    plot_final_metrics_comparison(aggregated, output_dir / "rq1e_final_metrics_comparison.png")
    print("  Saved: rq1e_final_metrics_comparison.png")
    plot_seed_variance(aggregated, output_dir / "rq1e_seed_variance.png")
    print("  Saved: rq1e_seed_variance.png")

    print_summary_table(aggregated)

    print("\nExperiment complete!")
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
