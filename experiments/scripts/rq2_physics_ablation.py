#!/usr/bin/env python3
"""RQ2: Physics Weight Ablation Experiment.

Research Question: Under what conditions does physics constraint help/hinder
operator generalization?

This experiment investigates the effect of the physics loss weight (lambda_physics)
on model performance, finding the optimal balance between data fitting and
physics compliance.

Experiment Design:
- PDE: Heat1D
- Model: PINNBase(input_dim=2, output_dim=1, hidden_dims=[64, 64])
- Vary lambda_physics: [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]
- Seeds: [42, 123, 456] — 3 seeds per λ value
- Training: stage1_epochs=50, stage2_epochs=100, stage3_epochs=0
- Evaluation: BenchmarkRunner with standard metrics
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from causal_pinn.evaluation.benchmark_runner import BenchmarkConfig, BenchmarkRunner
from causal_pinn.models.pinn.base import PINNBase
from causal_pinn.physics.pde.heat_1d import Heat1D
from causal_pinn.training.trainer import HybridTrainer
from causal_pinn.utils.config import TrainingConfig
from causal_pinn.utils.random import set_seed


def run_single_experiment(
    lambda_physics: float,
    seed: int,
    training_config: TrainingConfig,
    pde: Heat1D,
    benchmark_runner: BenchmarkRunner,
) -> dict[str, Any]:
    """Run a single training and evaluation experiment.

    Args:
        lambda_physics: Physics loss weight.
        seed: Random seed.
        training_config: Training configuration.
        pde: PDE problem instance.
        benchmark_runner: Benchmark runner for evaluation.

    Returns:
        Dictionary containing experiment results.
    """
    # Set seed for reproducibility
    set_seed(seed)

    # Create model
    model = PINNBase(
        input_dim=2,
        output_dim=1,
        hidden_dims=[64, 64, 64],
    )

    # Update training config with lambda_physics
    config = TrainingConfig(
        stage1_epochs=training_config.stage1_epochs,
        stage1_lr=training_config.stage1_lr,
        stage2_epochs=training_config.stage2_epochs,
        stage2_lr=training_config.stage2_lr,
        stage3_epochs=training_config.stage3_epochs,
        stage3_lr=training_config.stage3_lr,
        batch_size=training_config.batch_size,
        num_collocation_points=training_config.num_collocation_points,
        num_boundary_points=training_config.num_boundary_points,
        seed=seed,
        device=training_config.device,
        lambda_physics=lambda_physics,
        lambda_data=training_config.lambda_data,
        causal_epsilon=training_config.causal_epsilon,
    )

    # Create trainer
    trainer = HybridTrainer(
        model=model,
        pde_problem=pde,
        config=config,
        device=config.device,
    )

    # Train
    start_time = time.time()
    try:
        training_history = trainer.train()
        training_time = time.time() - start_time
        training_success = True
    except Exception as e:
        print(f"  Training failed for lambda={lambda_physics}, seed={seed}: {e}")
        training_time = time.time() - start_time
        training_history = {}
        training_success = False

    # Evaluate
    try:
        eval_metrics = benchmark_runner.run_single(model, pde)
        eval_success = True
    except Exception as e:
        print(f"  Evaluation failed for lambda={lambda_physics}, seed={seed}: {e}")
        eval_metrics = {}
        eval_success = False

    return {
        "lambda_physics": lambda_physics,
        "seed": seed,
        "training_history": training_history,
        "training_time": training_time,
        "training_success": training_success,
        "evaluation_metrics": eval_metrics,
        "eval_success": eval_success,
    }


def aggregate_results(
    results: list[dict[str, Any]],
    lambda_values: list[float],
) -> dict[str, dict[str, float]]:
    """Aggregate results across seeds for each lambda value.

    Args:
        results: List of all experiment results.
        lambda_values: List of lambda physics values.

    Returns:
        Dictionary mapping lambda to aggregated metrics (mean, std).
    """
    aggregated = {}

    for lam in lambda_values:
        # Filter results for this lambda
        lambda_results = [r for r in results if r["lambda_physics"] == lam and r["eval_success"]]

        if not lambda_results:
            aggregated[str(lam)] = {
                "rel_l2_mean": float("nan"),
                "rel_l2_std": float("nan"),
                "pde_residual_mean": float("nan"),
                "pde_residual_std": float("nan"),
                "mse_mean": float("nan"),
                "mse_std": float("nan"),
            }
            continue

        # Extract metrics
        rel_l2_values = [
            r["evaluation_metrics"].get("relative_l2_error", float("nan"))
            for r in lambda_results
        ]
        pde_residual_values = [
            r["evaluation_metrics"].get("pde_residual_norm", float("nan"))
            for r in lambda_results
        ]
        mse_values = [
            r["evaluation_metrics"].get("mse", float("nan"))
            for r in lambda_results
        ]

        aggregated[str(lam)] = {
            "rel_l2_mean": float(np.nanmean(rel_l2_values)),
            "rel_l2_std": float(np.nanstd(rel_l2_values)),
            "pde_residual_mean": float(np.nanmean(pde_residual_values)),
            "pde_residual_std": float(np.nanstd(pde_residual_values)),
            "mse_mean": float(np.nanmean(mse_values)),
            "mse_std": float(np.nanstd(mse_values)),
        }

    return aggregated


def generate_plots(
    results: list[dict[str, Any]],
    aggregated: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    """Generate visualization plots for RQ2.

    Args:
        results: Raw experiment results.
        aggregated: Aggregated metrics per lambda.
        output_dir: Directory to save plots.
    """
    lambda_values = sorted([float(k) for k in aggregated])

    # Extract means and stds
    rel_l2_means = [aggregated[str(lam)]["rel_l2_mean"] for lam in lambda_values]
    rel_l2_stds = [aggregated[str(lam)]["rel_l2_std"] for lam in lambda_values]
    pde_residual_means = [aggregated[str(lam)]["pde_residual_mean"] for lam in lambda_values]
    pde_residual_stds = [aggregated[str(lam)]["pde_residual_std"] for lam in lambda_values]

    # Plot 1: Lambda vs Accuracy (log scale)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter out nan values for plotting
    valid_idx = [
        i
        for i, (m, s) in enumerate(zip(rel_l2_means, rel_l2_stds, strict=False))
        if not (np.isnan(m) or np.isnan(s))
    ]

    if valid_idx:
        plot_lambdas = [lambda_values[i] for i in valid_idx]
        plot_means = [rel_l2_means[i] for i in valid_idx]
        plot_stds = [rel_l2_stds[i] for i in valid_idx]

        ax.errorbar(
            plot_lambdas,
            plot_means,
            yerr=plot_stds,
            fmt="o-",
            capsize=5,
            capthick=2,
            markersize=8,
            color="#2196F3",
            linewidth=2,
            label="Relative L2 Error",
        )

        # Mark the best lambda
        best_idx = np.nanargmin(plot_means)
        best_lambda = plot_lambdas[best_idx]
        ax.axvline(
            x=best_lambda,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Best λ={best_lambda}",
        )

        ax.set_xscale("log")
        ax.set_xlabel("Lambda Physics (log scale)", fontsize=12)
        ax.set_ylabel("Relative L2 Error (mean ± std)", fontsize=12)
        ax.set_title("RQ2: Physics Weight vs Accuracy", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "rq2_lambda_vs_accuracy.png", dpi=150)
    plt.close("all")

    # Plot 2: Lambda vs PDE Residual
    fig, ax = plt.subplots(figsize=(10, 6))

    valid_idx = [
        i
        for i, (m, s) in enumerate(zip(pde_residual_means, pde_residual_stds, strict=False))
        if not (np.isnan(m) or np.isnan(s))
    ]

    if valid_idx:
        plot_lambdas = [lambda_values[i] for i in valid_idx]
        plot_means = [pde_residual_means[i] for i in valid_idx]
        plot_stds = [pde_residual_stds[i] for i in valid_idx]

        ax.errorbar(
            plot_lambdas,
            plot_means,
            yerr=plot_stds,
            fmt="s-",
            capsize=5,
            capthick=2,
            markersize=8,
            color="#4CAF50",
            linewidth=2,
            label="PDE Residual Norm",
        )

        ax.set_xscale("log")
        ax.set_xlabel("Lambda Physics (log scale)", fontsize=12)
        ax.set_ylabel("PDE Residual Norm (mean ± std)", fontsize=12)
        ax.set_title("RQ2: Physics Weight vs PDE Residual", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "rq2_lambda_vs_residual.png", dpi=150)
    plt.close("all")

    # Plot 3: Pareto Trade-off Curve
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter valid points
    valid_idx = [
        i
        for i in range(len(lambda_values))
        if not (np.isnan(rel_l2_means[i]) or np.isnan(pde_residual_means[i]))
    ]

    if valid_idx:
        for i in valid_idx:
            lam = lambda_values[i]
            ax.scatter(
                rel_l2_means[i], pde_residual_means[i], s=150, label=f"λ={lam}", zorder=5
            )
            ax.annotate(
                f"λ={lam}",
                (rel_l2_means[i], pde_residual_means[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=10,
            )

        ax.set_xlabel("Relative L2 Error (Accuracy)", fontsize=12)
        ax.set_ylabel("PDE Residual Norm (Physics Compliance)", fontsize=12)
        ax.set_title("RQ2: Accuracy vs Physics Compliance Trade-off", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "rq2_tradeoff_curve.png", dpi=150)
    plt.close("all")


def print_summary(
    aggregated: dict[str, dict[str, float]],
    lambda_values: list[float],
) -> None:
    """Print summary table of results.

    Args:
        aggregated: Aggregated metrics per lambda.
        lambda_values: List of lambda physics values.
    """
    print("\n" + "=" * 80)
    print("RQ2: Physics Weight Ablation Summary")
    print("=" * 80)
    header = (
        f"{'Lambda':<10} | {'Rel L2 (mean±std)':<25} | "
        f"{'PDE Residual (mean±std)':<25} | {'MSE (mean±std)':<25}"
    )
    print(header)
    print("-" * 80)

    for lam in lambda_values:
        data = aggregated[str(lam)]
        rel_l2 = f"{data['rel_l2_mean']:.4e} ± {data['rel_l2_std']:.4e}"
        pde_res = f"{data['pde_residual_mean']:.4e} ± {data['pde_residual_std']:.4e}"
        mse = f"{data['mse_mean']:.4e} ± {data['mse_std']:.4e}"
        print(f"{lam:<10} | {rel_l2:<25} | {pde_res:<25} | {mse:<25}")

    print("=" * 80)


def main() -> None:
    """Main entry point for RQ2 experiment."""
    parser = argparse.ArgumentParser(description="RQ2: Physics Weight Ablation Experiment")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/rq2",
        help="Output directory for results and plots",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Experiment configuration
    lambda_values = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    seeds = [42, 123, 456, 789, 1024]

    # Training configuration (GPU full-scale)
    training_config = TrainingConfig(
        stage1_epochs=200,
        stage1_lr=1e-3,
        stage2_epochs=500,
        stage2_lr=1e-4,
        stage3_epochs=0,
        stage3_lr=1e-5,
        batch_size=256,
        num_collocation_points=4000,
        num_boundary_points=500,
        seed=42,
        device="cuda",
        lambda_physics=1.0,
        lambda_data=1.0,
        causal_epsilon=1.0,
    )

    # PDE problem
    pde = Heat1D(alpha=1.0, length=1.0, t_max=1.0)

    # Benchmark runner
    benchmark_config = BenchmarkConfig(
        n_collocation=2000,
        n_boundary=200,
        n_initial=200,
        include_physics_consistency=False,
        device="cuda",
    )
    benchmark_runner = BenchmarkRunner(config=benchmark_config)

    # Run experiments
    print("=" * 80)
    print("RQ2: Physics Weight Ablation Experiment")
    print("=" * 80)
    print(f"Lambda values: {lambda_values}")
    print(f"Seeds: {seeds}")
    print(f"Total experiments: {len(lambda_values) * len(seeds)}")
    print()

    all_results: list[dict[str, Any]] = []

    for lam in lambda_values:
        print(f"\n--- Lambda Physics = {lam} ---")
        for seed in seeds:
            print(f"  Running seed {seed}...", end=" ", flush=True)
            try:
                result = run_single_experiment(
                    lambda_physics=lam,
                    seed=seed,
                    training_config=training_config,
                    pde=pde,
                    benchmark_runner=benchmark_runner,
                )
                all_results.append(result)
                rel_l2 = result["evaluation_metrics"].get("relative_l2_error", float("nan"))
                print(f"Done. Rel L2: {rel_l2:.4e}")
            except Exception as e:
                print(f"Failed: {e}")
                all_results.append(
                    {
                        "lambda_physics": lam,
                        "seed": seed,
                        "training_history": {},
                        "training_time": 0.0,
                        "training_success": False,
                        "evaluation_metrics": {},
                        "eval_success": False,
                    }
                )

    # Aggregate results
    aggregated = aggregate_results(all_results, lambda_values)

    # Save results
    results_data = {
        "experiment": "rq2_physics_ablation",
        "lambda_values": lambda_values,
        "seeds": seeds,
        "config": {
            "stage1_epochs": training_config.stage1_epochs,
            "stage2_epochs": training_config.stage2_epochs,
            "stage3_epochs": training_config.stage3_epochs,
            "batch_size": training_config.batch_size,
            "num_collocation_points": training_config.num_collocation_points,
            "num_boundary_points": training_config.num_boundary_points,
            "device": training_config.device,
        },
        "raw_results": all_results,
        "aggregated_results": aggregated,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir / 'results.json'}")

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(all_results, aggregated, output_dir)
    print(f"Plots saved to {output_dir}")

    # Print summary
    print_summary(aggregated, lambda_values)


if __name__ == "__main__":
    main()
