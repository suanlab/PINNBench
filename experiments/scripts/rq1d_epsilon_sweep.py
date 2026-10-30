#!/usr/bin/env python3
"""RQ1d Experiment: Causal Epsilon Sweep on AdvectionDiffusion1D.

Research Question: Was ε=1.0 in prior experiments (RQ1, RQ1b) simply the wrong
hyperparameter setting? What is the optimal causal epsilon value?

This experiment sweeps through different causal_epsilon values to find the
optimal setting for causal training on the AdvectionDiffusion1D PDE.

Experiment Design:
- PDE: AdvectionDiffusion1D(velocity=1.0, alpha=0.1, length=1.0, t_max=1.0)
- Model: PINNBase(input_dim=2, output_dim=1, hidden_dims=[64, 64, 64])
- Epsilon values: [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
  - ε=0.0: Skip stage3 entirely (baseline without causal training)
  - Other ε: Full 3-stage pipeline with that epsilon value
- Seeds: [42, 123, 456, 789, 1024] — 5 seeds per epsilon
- Training: Stage1=200ep/1e-3, Stage2=500ep/1e-4, Stage3=200ep/1e-5
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
from causal_pinn.physics.pde.advection_diffusion_1d import AdvectionDiffusion1D
from causal_pinn.training.trainer import HybridTrainer
from causal_pinn.utils.config import TrainingConfig
from causal_pinn.utils.random import set_seed


def run_single_experiment(
    epsilon: float,
    seed: int,
    training_config: TrainingConfig,
    pde: AdvectionDiffusion1D,
    benchmark_runner: BenchmarkRunner,
) -> dict[str, Any]:
    """Run a single training and evaluation experiment.

    Args:
        epsilon: Causal epsilon value. If 0.0, skip stage3 entirely.
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

    # Determine stage3_epochs based on epsilon
    # ε=0.0: skip stage3 entirely (baseline)
    # Other ε: run full 3-stage pipeline
    stage3_epochs = 0 if epsilon == 0.0 else training_config.stage3_epochs

    # Update training config with epsilon
    config = TrainingConfig(
        stage1_epochs=training_config.stage1_epochs,
        stage1_lr=training_config.stage1_lr,
        stage2_epochs=training_config.stage2_epochs,
        stage2_lr=training_config.stage2_lr,
        stage3_epochs=stage3_epochs,
        stage3_lr=training_config.stage3_lr,
        batch_size=training_config.batch_size,
        num_collocation_points=training_config.num_collocation_points,
        num_boundary_points=training_config.num_boundary_points,
        seed=seed,
        device=training_config.device,
        lambda_physics=training_config.lambda_physics,
        lambda_data=training_config.lambda_data,
        causal_epsilon=(
            epsilon if epsilon > 0.0 else 1.0
        ),  # Value doesn't matter when stage3_epochs=0
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
        print(f"  Training failed for epsilon={epsilon}, seed={seed}: {e}")
        training_time = time.time() - start_time
        training_history = {}
        training_success = False

    # Evaluate
    try:
        eval_metrics = benchmark_runner.run_single(model, pde)
        eval_success = True
    except Exception as e:
        print(f"  Evaluation failed for epsilon={epsilon}, seed={seed}: {e}")
        eval_metrics = {}
        eval_success = False

    return {
        "epsilon": epsilon,
        "seed": seed,
        "training_history": training_history,
        "training_time": training_time,
        "training_success": training_success,
        "evaluation_metrics": eval_metrics,
        "eval_success": eval_success,
    }


def aggregate_results(
    results: list[dict[str, Any]],
    epsilon_values: list[float],
) -> dict[str, dict[str, float]]:
    """Aggregate results across seeds for each epsilon value.

    Args:
        results: List of all experiment results.
        epsilon_values: List of epsilon values.

    Returns:
        Dictionary mapping epsilon to aggregated metrics (mean, std).
    """
    aggregated = {}

    for eps in epsilon_values:
        # Filter results for this epsilon
        eps_results = [r for r in results if r["epsilon"] == eps and r["eval_success"]]

        if not eps_results:
            aggregated[str(eps)] = {
                "rel_l2_mean": float("nan"),
                "rel_l2_std": float("nan"),
                "pde_residual_mean": float("nan"),
                "pde_residual_std": float("nan"),
                "mse_mean": float("nan"),
                "mse_std": float("nan"),
                "mae_mean": float("nan"),
                "mae_std": float("nan"),
                "max_error_mean": float("nan"),
                "max_error_std": float("nan"),
                "bc_violation_mean": float("nan"),
                "bc_violation_std": float("nan"),
                "ic_violation_mean": float("nan"),
                "ic_violation_std": float("nan"),
                "training_time_mean": float("nan"),
                "training_time_std": float("nan"),
            }
            continue

        # Extract metrics
        rel_l2_values = [
            r["evaluation_metrics"].get("relative_l2_error", float("nan"))
            for r in eps_results
        ]
        pde_residual_values = [
            r["evaluation_metrics"].get("pde_residual_norm", float("nan"))
            for r in eps_results
        ]
        mse_values = [
            r["evaluation_metrics"].get("mse", float("nan"))
            for r in eps_results
        ]
        mae_values = [
            r["evaluation_metrics"].get("mae", float("nan"))
            for r in eps_results
        ]
        max_error_values = [
            r["evaluation_metrics"].get("max_error", float("nan"))
            for r in eps_results
        ]
        bc_violation_values = [
            r["evaluation_metrics"].get("boundary_condition_error", float("nan"))
            for r in eps_results
        ]
        ic_violation_values = [
            r["evaluation_metrics"].get("initial_condition_error", float("nan"))
            for r in eps_results
        ]
        training_time_values = [
            r["training_time"]
            for r in eps_results
        ]

        aggregated[str(eps)] = {
            "rel_l2_mean": float(np.nanmean(rel_l2_values)),
            "rel_l2_std": float(np.nanstd(rel_l2_values)),
            "pde_residual_mean": float(np.nanmean(pde_residual_values)),
            "pde_residual_std": float(np.nanstd(pde_residual_values)),
            "mse_mean": float(np.nanmean(mse_values)),
            "mse_std": float(np.nanstd(mse_values)),
            "mae_mean": float(np.nanmean(mae_values)),
            "mae_std": float(np.nanstd(mae_values)),
            "max_error_mean": float(np.nanmean(max_error_values)),
            "max_error_std": float(np.nanstd(max_error_values)),
            "bc_violation_mean": float(np.nanmean(bc_violation_values)),
            "bc_violation_std": float(np.nanstd(bc_violation_values)),
            "ic_violation_mean": float(np.nanmean(ic_violation_values)),
            "ic_violation_std": float(np.nanstd(ic_violation_values)),
            "training_time_mean": float(np.nanmean(training_time_values)),
            "training_time_std": float(np.nanstd(training_time_values)),
            "rel_l2_values": [float(x) if not np.isnan(x) else None for x in rel_l2_values],
            "pde_residual_values": [float(x) for x in pde_residual_values],
        }

    return aggregated


def generate_plots(
    results: list[dict[str, Any]],
    aggregated: dict[str, dict[str, float]],
    output_dir: Path,
) -> None:
    """Generate visualization plots for RQ1d.

    Args:
        results: Raw experiment results.
        aggregated: Aggregated metrics per epsilon.
        output_dir: Directory to save plots.
    """
    epsilon_values = sorted([float(k) for k in aggregated])

    # Extract means and stds
    rel_l2_means = [aggregated[str(eps)]["rel_l2_mean"] for eps in epsilon_values]
    rel_l2_stds = [aggregated[str(eps)]["rel_l2_std"] for eps in epsilon_values]
    pde_residual_means = [aggregated[str(eps)]["pde_residual_mean"] for eps in epsilon_values]
    pde_residual_stds = [aggregated[str(eps)]["pde_residual_std"] for eps in epsilon_values]

    # Plot 1: Epsilon vs Accuracy (log scale)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter out nan values for plotting
    valid_idx = [
        i
        for i, (m, s) in enumerate(zip(rel_l2_means, rel_l2_stds, strict=False))
        if not (np.isnan(m) or np.isnan(s))
    ]

    if valid_idx:
        plot_epsilons = [epsilon_values[i] for i in valid_idx]
        plot_means = [rel_l2_means[i] for i in valid_idx]
        plot_stds = [rel_l2_stds[i] for i in valid_idx]

        ax.errorbar(
            plot_epsilons, plot_means, yerr=plot_stds,
            fmt='o-', capsize=5, capthick=2, markersize=8,
            color='#2196F3', linewidth=2, label='Relative L2 Error'
        )

        # Mark the best epsilon (lowest error)
        best_idx = np.nanargmin(plot_means)
        best_epsilon = plot_epsilons[best_idx]
        ax.axvline(x=best_epsilon, color='red', linestyle='--', alpha=0.7,
                   label=f'Best ε={best_epsilon}')

        ax.set_xscale('log')
        ax.set_xlabel('Causal Epsilon (log scale)', fontsize=12)
        ax.set_ylabel('Relative L2 Error (mean ± std)', fontsize=12)
        ax.set_title('RQ1d: Causal Epsilon vs Accuracy (AdvectionDiffusion1D)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'rq1d_epsilon_vs_accuracy.png', dpi=150)
    plt.close('all')

    # Plot 2: Epsilon vs PDE Residual
    fig, ax = plt.subplots(figsize=(10, 6))

    valid_idx = [
        i
        for i, (m, s) in enumerate(zip(pde_residual_means, pde_residual_stds, strict=False))
        if not (np.isnan(m) or np.isnan(s))
    ]

    if valid_idx:
        plot_epsilons = [epsilon_values[i] for i in valid_idx]
        plot_means = [pde_residual_means[i] for i in valid_idx]
        plot_stds = [pde_residual_stds[i] for i in valid_idx]

        ax.errorbar(
            plot_epsilons, plot_means, yerr=plot_stds,
            fmt='s-', capsize=5, capthick=2, markersize=8,
            color='#4CAF50', linewidth=2, label='PDE Residual Norm'
        )

        ax.set_xscale('log')
        ax.set_xlabel('Causal Epsilon (log scale)', fontsize=12)
        ax.set_ylabel('PDE Residual Norm (mean ± std)', fontsize=12)
        ax.set_title('RQ1d: Causal Epsilon vs PDE Residual (AdvectionDiffusion1D)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'rq1d_epsilon_vs_residual.png', dpi=150)
    plt.close('all')

    # Plot 3: Trade-off Curve (Accuracy vs Physics Compliance)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Filter valid points
    valid_idx = [
        i for i in range(len(epsilon_values))
        if not (np.isnan(rel_l2_means[i]) or np.isnan(pde_residual_means[i]))
    ]

    if valid_idx:
        # Use different colors for different epsilon ranges
        colors = plt.cm.viridis(np.linspace(0, 1, len(valid_idx)))

        for idx, i in enumerate(valid_idx):
            eps = epsilon_values[i]
            ax.scatter(
                rel_l2_means[i], pde_residual_means[i],
                s=150, color=colors[idx], label=f'ε={eps}', zorder=5
            )
            ax.annotate(
                f'ε={eps}',
                (rel_l2_means[i], pde_residual_means[i]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10
            )

        ax.set_xlabel('Relative L2 Error (Accuracy)', fontsize=12)
        ax.set_ylabel('PDE Residual Norm (Physics Compliance)', fontsize=12)
        title = (
            'RQ1d: Accuracy vs Physics Compliance Trade-off '
            '(AdvectionDiffusion1D)'
        )
        ax.set_title(title, fontsize=14)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'rq1d_tradeoff_curve.png', dpi=150)
    plt.close('all')


def print_summary(
    aggregated: dict[str, dict[str, float]],
    epsilon_values: list[float],
) -> None:
    """Print summary table of results.

    Args:
        aggregated: Aggregated metrics per epsilon.
        epsilon_values: List of epsilon values.
    """
    print("\n" + "=" * 100)
    print("RQ1d: Causal Epsilon Sweep Summary (AdvectionDiffusion1D)")
    print("=" * 100)
    header = (
        f"{'Epsilon':<12} | {'Rel L2 (mean±std)':<25} | "
        f"{'PDE Residual (mean±std)':<25} | {'MSE (mean±std)':<25}"
    )
    print(header)
    print("-" * 100)

    for eps in epsilon_values:
        data = aggregated[str(eps)]
        rel_l2 = f"{data['rel_l2_mean']:.4e} ± {data['rel_l2_std']:.4e}"
        pde_res = f"{data['pde_residual_mean']:.4e} ± {data['pde_residual_std']:.4e}"
        mse = f"{data['mse_mean']:.4e} ± {data['mse_std']:.4e}"
        print(f"{eps:<12} | {rel_l2:<25} | {pde_res:<25} | {mse:<25}")

    print("=" * 100)

    # Print additional metrics
    print("\nAdditional Metrics:")
    header2 = (
        f"{'Epsilon':<12} | {'MAE (mean±std)':<25} | "
        f"{'BC Violation (mean±std)':<25} | {'IC Violation (mean±std)':<25}"
    )
    print(header2)
    print("-" * 100)

    for eps in epsilon_values:
        data = aggregated[str(eps)]
        mae = f"{data['mae_mean']:.4e} ± {data['mae_std']:.4e}"
        bc = f"{data['bc_violation_mean']:.4e} ± {data['bc_violation_std']:.4e}"
        ic = f"{data['ic_violation_mean']:.4e} ± {data['ic_violation_std']:.4e}"
        print(f"{eps:<12} | {mae:<25} | {bc:<25} | {ic:<25}")

    print("=" * 100)

    # Find and highlight best epsilon
    valid_epsilons = [
        (eps, aggregated[str(eps)]["rel_l2_mean"])
        for eps in epsilon_values
        if not np.isnan(aggregated[str(eps)]["rel_l2_mean"])
    ]

    if valid_epsilons:
        best_eps, best_rel_l2 = min(valid_epsilons, key=lambda x: x[1])
        print(f"\n*** Best Epsilon: {best_eps} with Relative L2 Error: {best_rel_l2:.4e} ***")


def main() -> None:
    """Main entry point for RQ1d experiment."""
    parser = argparse.ArgumentParser(
        description='RQ1d: Causal Epsilon Sweep Experiment'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/results/rq1d',
        help='Output directory for results and plots'
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Experiment configuration
    epsilon_values = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]
    seeds = [42, 123, 456, 789, 1024]

    # Training configuration (GPU full-scale)
    training_config = TrainingConfig(
        stage1_epochs=200,
        stage1_lr=1e-3,
        stage2_epochs=500,
        stage2_lr=1e-4,
        stage3_epochs=200,
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
    pde = AdvectionDiffusion1D(velocity=1.0, alpha=0.1, length=1.0, t_max=1.0)

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
    print("RQ1d: Causal Epsilon Sweep Experiment")
    print("=" * 80)
    print("PDE: AdvectionDiffusion1D (velocity=1.0, alpha=0.1)")
    print(f"Epsilon values: {epsilon_values}")
    print(f"Seeds: {seeds}")
    print(f"Total experiments: {len(epsilon_values) * len(seeds)}")
    print("Note: ε=0.0 skips Stage 3 (baseline without causal training)")
    print()

    all_results: list[dict[str, Any]] = []

    for eps in epsilon_values:
        print(f"\n--- Epsilon = {eps} ---")
        stage3_status = "SKIPPED (baseline)" if eps == 0.0 else "ACTIVE"
        print(f"Stage 3: {stage3_status}")
        for seed in seeds:
            print(f"  Running seed {seed}...", end=" ", flush=True)
            try:
                result = run_single_experiment(
                    epsilon=eps,
                    seed=seed,
                    training_config=training_config,
                    pde=pde,
                    benchmark_runner=benchmark_runner,
                )
                all_results.append(result)
                rel_l2 = result["evaluation_metrics"].get("relative_l2_error", float("nan"))
                pde_res = result["evaluation_metrics"].get("pde_residual_norm", float("nan"))
                print(f"Done. Rel L2: {rel_l2:.4e}, PDE Res: {pde_res:.4e}")
            except Exception as e:
                print(f"Failed: {e}")
                all_results.append({
                    "epsilon": eps,
                    "seed": seed,
                    "training_history": {},
                    "training_time": 0.0,
                    "training_success": False,
                    "evaluation_metrics": {},
                    "eval_success": False,
                })

    # Aggregate results
    aggregated = aggregate_results(all_results, epsilon_values)

    # Save results
    results_data = {
        "experiment": "rq1d_epsilon_sweep",
        "description": "Causal epsilon sweep on AdvectionDiffusion1D",
        "pde": "AdvectionDiffusion1D",
        "pde_config": {
            "velocity": 1.0,
            "alpha": 0.1,
            "length": 1.0,
            "t_max": 1.0,
        },
        "epsilon_values": epsilon_values,
        "seeds": seeds,
        "config": {
            "stage1_epochs": training_config.stage1_epochs,
            "stage1_lr": training_config.stage1_lr,
            "stage2_epochs": training_config.stage2_epochs,
            "stage2_lr": training_config.stage2_lr,
            "stage3_epochs": training_config.stage3_epochs,
            "stage3_lr": training_config.stage3_lr,
            "batch_size": training_config.batch_size,
            "num_collocation_points": training_config.num_collocation_points,
            "num_boundary_points": training_config.num_boundary_points,
            "device": training_config.device,
        },
        "aggregated_results": aggregated,
        "raw_results": all_results,
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir / 'results.json'}")

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(all_results, aggregated, output_dir)
    print(f"Plots saved to {output_dir}")

    # Print summary
    print_summary(aggregated, epsilon_values)


if __name__ == "__main__":
    main()
