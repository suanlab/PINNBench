#!/usr/bin/env python3
"""RQ3: Model Comparison Experiment.

Research Question: Can we evaluate hybrid models consistently across accuracy,
physics consistency, and verifiability?

This experiment compares three model configurations:
1. PINN-only: Pure physics-informed neural network
2. Hybrid (no gate): PINN + Neural Operator blend without causal gating
3. Hybrid (full): Complete framework with PINN, Operator, and Causal Gate

All models are trained using the 3-stage HybridTrainer pipeline:
- Stage 1: Operator pretraining (200 epochs, lr=1e-3)
- Stage 2: Physics refinement (500 epochs, lr=1e-4)
- Stage 3: Causal calibration (200 epochs, lr=1e-5)

Experiment Design:
- PDE: Heat1D
- Models:
  * pinn_only: PINNBase(input_dim=2, output_dim=1, hidden_dims=[64, 64, 64])
  * hybrid_no_gate: HybridModelWrapper(HybridPINNOperator(use_causal_gate=False))
  * hybrid_full: HybridModelWrapper(HybridPINNOperator(use_causal_gate=True))
- Seeds: [42, 123, 456, 789, 1024] — 5 seeds per model
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
import torch

from causal_pinn.evaluation.benchmark_runner import BenchmarkConfig, BenchmarkRunner
from causal_pinn.models.hybrid.causal_gate import CausalGate
from causal_pinn.models.hybrid.hybrid_model import (
    HybridConfig,
    HybridModelWrapper,
    HybridPINNOperator,
)
from causal_pinn.models.operators.fno import FNO
from causal_pinn.models.pinn.base import PINNBase
from causal_pinn.physics.pde.heat_1d import Heat1D
from causal_pinn.training.trainer import HybridTrainer
from causal_pinn.utils.config import TrainingConfig
from causal_pinn.utils.random import set_seed


# Keep SimpleBlendedHybrid for reference (not used in main experiment)
class SimpleBlendedHybrid(torch.nn.Module):
    """Custom hybrid model with simple average blending (no causal gate).

    This model combines PINN and Operator predictions using equal weighting,
    useful for testing the contribution of the causal gate.
    """

    def __init__(
        self,
        pinn_network: PINNBase,
        operator: FNO,
        pde_problem: Heat1D,
        pinn_weight: float = 0.5,
        operator_weight: float = 0.5,
    ) -> None:
        """Initialize the simple blended hybrid.

        Args:
            pinn_network: PINN network.
            operator: Neural operator (FNO).
            pde_problem: PDE problem definition.
            pinn_weight: Weight for PINN prediction.
            operator_weight: Weight for operator prediction.
        """
        super().__init__()
        self.pinn_network = pinn_network
        self.operator = operator
        self.pde_problem = pde_problem
        self.pinn_weight = pinn_weight
        self.operator_weight = operator_weight

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute blended prediction from operator and PINN.

        Args:
            t: Time coordinates, shape (N,) or (N, 1).
            x: Spatial coordinates, shape (N,) or (N, 1).

        Returns:
            Blended prediction tensor, shape (N, 1).
        """
        # Ensure 2D tensors
        t_2d = t.unsqueeze(-1) if t.dim() == 1 else t
        x_2d = x.unsqueeze(-1) if x.dim() == 1 else x

        # Get PINN prediction
        pinn_input = torch.cat([t_2d, x_2d], dim=-1)
        pinn_pred = self.pinn_network(pinn_input)

        # Get operator prediction
        combined = torch.cat([t_2d, x_2d], dim=-1)  # (N, 2)
        operator_input = combined.unsqueeze(-1)  # (N, 2, 1)
        operator_out = self.operator(operator_input)
        operator_pred = operator_out.squeeze(-1)
        if operator_pred.dim() == 1:
            operator_pred = operator_pred.unsqueeze(-1)

        # Simple weighted average
        return self.pinn_weight * pinn_pred + self.operator_weight * operator_pred

    def compute_hybrid_loss(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the hybrid loss with physics loss.

        Args:
            t: Time coordinates.
            x: Spatial coordinates.

        Returns:
            Dictionary with losses.
        """
        # Ensure gradients are tracked
        if not t.requires_grad:
            t = t.clone().requires_grad_(True)
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)

        # Get blended prediction
        blended_pred = self.forward(t, x)

        # Compute PDE residual for physics loss
        u = blended_pred.squeeze(-1) if blended_pred.dim() > 1 else blended_pred
        residual = self.pde_problem.residual(u, t, x)
        physics_loss = torch.mean(residual**2)

        # Total loss (just physics loss for this model)
        total_loss = physics_loss

        return {
            "physics_loss": physics_loss,
            "total": total_loss,
        }


def create_pinn_only_model() -> PINNBase:
    """Create a PINN-only model.

    Returns:
        PINNBase model instance.
    """
    return PINNBase(
        input_dim=2,
        output_dim=1,
        hidden_dims=[64, 64, 64],
    )


def create_hybrid_model(
    pde: Heat1D,
    use_causal_gate: bool,
) -> HybridModelWrapper:
    """Create a HybridModelWrapper around HybridPINNOperator.

    Args:
        pde: PDE problem instance.
        use_causal_gate: Whether to use causal gating.

    Returns:
        HybridModelWrapper instance.
    """
    # Create PINN component
    pinn_network = PINNBase(
        input_dim=2,
        output_dim=1,
        hidden_dims=[64, 64, 64],
    )

    # Create Operator component (FNO)
    operator = FNO(
        in_channels=2,
        out_channels=1,
        modes=8,
        width=32,
        num_layers=3,
        dim=1,
    )

    # Create Causal Gate
    causal_gate = CausalGate(
        out_dim=1,
        hidden_dim=32,
        causal_strength=1.0,
    )

    # Create hybrid config
    config = HybridConfig(
        use_operator=True,
        use_pinn=True,
        use_causal_gate=use_causal_gate,
        operator_weight=1.0,
        physics_weight=1.0,
        gate_weight=0.01,
    )

    # Create HybridPINNOperator
    hybrid_pinn = HybridPINNOperator(
        pinn_network=pinn_network,
        operator=operator,
        causal_gate=causal_gate,
        pde_problem=pde,
        config=config,
    )

    # Wrap for HybridTrainer compatibility
    return HybridModelWrapper(hybrid_pinn)


def create_simple_blended_model(
    pde: Heat1D,
) -> SimpleBlendedHybrid:
    """Create a simple blended hybrid model (no causal gate).

    Args:
        pde: PDE problem instance.

    Returns:
        SimpleBlendedHybrid instance.
    """
    pinn_network = PINNBase(
        input_dim=2,
        output_dim=1,
        hidden_dims=[64, 64],
    )

    operator = FNO(
        in_channels=2,
        out_channels=1,
        modes=8,
        width=16,
        num_layers=2,
        dim=1,
    )

    return SimpleBlendedHybrid(
        pinn_network=pinn_network,
        operator=operator,
        pde_problem=pde,
        pinn_weight=0.5,
        operator_weight=0.5,
    )


def train_with_hybrid_trainer(
    model: torch.nn.Module,
    pde: Heat1D,
    seed: int,
    device: str,
) -> tuple[dict[str, Any], float]:
    """Train model using HybridTrainer's 3-stage pipeline.

    Args:
        model: Neural network model (PINNBase or HybridModelWrapper).
        pde: PDE problem.
        seed: Random seed.
        device: Device to train on.

    Returns:
        Tuple of (training_history, training_time).
    """
    set_seed(seed)

    # Training configuration (GPU full-scale)
    config = TrainingConfig(
        stage1_epochs=200,
        stage1_lr=1e-3,
        stage2_epochs=500,
        stage2_lr=1e-4,
        stage3_epochs=200,
        stage3_lr=1e-5,
        batch_size=256,
        num_collocation_points=4000,
        num_boundary_points=500,
        seed=seed,
        device=device,
        lambda_physics=1.0,
        lambda_data=1.0,
        causal_epsilon=1.0,
    )

    trainer = HybridTrainer(
        model=model,
        pde_problem=pde,
        config=config,
        device=device,
    )

    start_time = time.time()
    history = trainer.train()
    training_time = time.time() - start_time

    return history, training_time


def run_single_experiment(
    model_name: str,
    seed: int,
    pde: Heat1D,
    benchmark_runner: BenchmarkRunner,
    device: str,
) -> dict[str, Any]:
    """Run a single model training and evaluation experiment.

    Args:
        model_name: Name of model configuration.
        seed: Random seed.
        pde: PDE problem instance.
        benchmark_runner: Benchmark runner for evaluation.
        device: Device to run on.

    Returns:
        Dictionary containing experiment results.
    """
    print(f"  Running {model_name} with seed {seed}...", end=" ", flush=True)

    try:
        # Create model based on configuration
        if model_name == "pinn_only":
            model = create_pinn_only_model()
            training_history, training_time = train_with_hybrid_trainer(
                model=model, pde=pde, seed=seed, device=device
            )
            eval_model = model
        elif model_name == "hybrid_no_gate":
            model = create_hybrid_model(pde=pde, use_causal_gate=False)
            training_history, training_time = train_with_hybrid_trainer(
                model=model, pde=pde, seed=seed, device=device
            )
            eval_model = model  # Already wrapped
        elif model_name == "hybrid_full":
            model = create_hybrid_model(pde=pde, use_causal_gate=True)
            training_history, training_time = train_with_hybrid_trainer(
                model=model, pde=pde, seed=seed, device=device
            )
            eval_model = model  # Already wrapped
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Evaluate
        metrics = benchmark_runner.run_single(eval_model, pde)

        rel_l2 = metrics.get("relative_l2_error", float("nan"))
        print(f"Done. Rel L2: {rel_l2:.4e}, Time: {training_time:.1f}s")

        return {
            "model_name": model_name,
            "seed": seed,
            "training_time": training_time,
            "training_history": training_history,
            "metrics": metrics,
            "success": True,
        }

    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "model_name": model_name,
            "seed": seed,
            "training_time": 0.0,
            "training_history": {},
            "metrics": {},
            "success": False,
        }


def aggregate_results(
    results: list[dict[str, Any]],
    model_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Aggregate results across seeds for each model.

    Args:
        results: List of all experiment results.
        model_names: List of model names.

    Returns:
        Dictionary mapping model name to aggregated metrics.
    """
    aggregated = {}

    metrics_to_aggregate = [
        "relative_l2_error",
        "mse",
        "mae",
        "max_error",
        "pde_residual_norm",
        "bc_violation",
        "ic_violation",
    ]

    for model_name in model_names:
        model_results = [r for r in results if r["model_name"] == model_name and r["success"]]

        if not model_results:
            aggregated[model_name] = {
                m: {"mean": float("nan"), "std": float("nan")}
                for m in metrics_to_aggregate
            }
            aggregated[model_name]["training_time"] = {"mean": float("nan"), "std": float("nan")}
            continue

        aggregated[model_name] = {}

        for metric in metrics_to_aggregate:
            values = [r["metrics"].get(metric, float("nan")) for r in model_results]
            aggregated[model_name][metric] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
            }

        # Training time
        times = [r["training_time"] for r in model_results]
        aggregated[model_name]["training_time"] = {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
        }

    return aggregated


def generate_plots(
    results: list[dict[str, Any]],
    aggregated: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate visualization plots for RQ3.

    Args:
        results: Raw experiment results.
        aggregated: Aggregated metrics per model.
        output_dir: Directory to save plots.
    """
    model_names = list(aggregated.keys())
    colors = ['#2196F3', '#4CAF50', '#FF9800']  # Blue, Green, Orange

    # Plot 1: Grouped Bar Chart - Metrics Comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    metrics_for_bar = [
        'relative_l2_error',
        'mse',
        'pde_residual_norm',
        'bc_violation',
        'ic_violation',
    ]
    metric_labels = ['Rel L2', 'MSE', 'PDE Residual', 'BC Violation', 'IC Violation']

    x = np.arange(len(metrics_for_bar))
    width = 0.25

    for i, model_name in enumerate(model_names):
        means = [
            aggregated[model_name].get(m, {}).get("mean", float("nan"))
            for m in metrics_for_bar
        ]
        stds = [aggregated[model_name].get(m, {}).get("std", 0.0) for m in metrics_for_bar]

        # Filter out nan values
        valid_mask = [not np.isnan(m) for m in means]
        if any(valid_mask):
            ax.bar(
                x + i * width,
                [m if not np.isnan(m) else 0 for m in means],
                width,
                yerr=[s if not np.isnan(means[j]) else 0 for j, s in enumerate(stds)],
                label=model_name,
                color=colors[i % len(colors)],
                capsize=3,
                alpha=0.8,
            )

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Value (log scale)', fontsize=12)
    ax.set_title('RQ3: Model Comparison Across Metrics', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'rq3_metrics_comparison.png', dpi=150)
    plt.close('all')

    # Plot 2: Training Efficiency Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, model_name in enumerate(model_names):
        model_results = [r for r in results if r["model_name"] == model_name and r["success"]]

        if model_results:
            times = [r["training_time"] for r in model_results]
            rel_l2s = [r["metrics"].get("relative_l2_error", float("nan")) for r in model_results]

            # Filter valid
            valid = [
                (time_value, rel_l2)
                for time_value, rel_l2 in zip(times, rel_l2s, strict=False)
                if not np.isnan(rel_l2)
            ]
            if valid:
                plot_times, plot_l2s = zip(*valid, strict=False)
                ax.scatter(
                    plot_times, plot_l2s,
                    s=150, c=colors[i % len(colors)],
                    label=model_name,
                    marker='o',
                    alpha=0.8,
                )

    ax.set_xlabel('Training Time (seconds)', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title('RQ3: Training Efficiency (Time vs Accuracy)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'rq3_training_efficiency.png', dpi=150)
    plt.close('all')

    # Plot 3: Radar Chart (simplified as grouped bar if radar is complex)
    # Using grouped bar for better readability
    fig, ax = plt.subplots(figsize=(12, 8))

    # Normalize metrics for radar-like comparison
    # Use 5 axes: accuracy (inverse of rel_l2), physics_residual (inverse),
    # bc_compliance (inverse), ic_compliance (inverse), training_time (inverse)

    radar_metrics = ['relative_l2_error', 'pde_residual_norm', 'bc_violation', 'ic_violation']
    radar_labels = ['Accuracy\n(inv Rel L2)', 'Physics\n(inv Residual)',
                    'BC Compliance\n(inv Violation)', 'IC Compliance\n(inv Violation)']

    # For inverse metrics, we'll use 1/(1+value) to get normalized scores in (0,1]
    x = np.arange(len(radar_metrics))
    width = 0.25

    for i, model_name in enumerate(model_names):
        scores = []
        for m in radar_metrics:
            mean_val = aggregated[model_name].get(m, {}).get("mean", float("nan"))
            score = 1.0 / (1.0 + mean_val) if not np.isnan(mean_val) and mean_val >= 0 else 0.0
            scores.append(score)

        ax.bar(
            x + i * width,
            scores,
            width,
            label=model_name,
            color=colors[i % len(colors)],
            alpha=0.8,
        )

    ax.set_xlabel('Normalized Score (higher is better)', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('RQ3: Normalized Performance Comparison', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(radar_labels)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'rq3_radar_chart.png', dpi=150)
    plt.close('all')


def print_summary(
    aggregated: dict[str, dict[str, Any]],
    model_names: list[str],
) -> None:
    """Print summary table of results.

    Args:
        aggregated: Aggregated metrics per model.
        model_names: List of model names.
    """
    print("\n" + "=" * 120)
    print("RQ3: Model Comparison Summary (3-Stage HybridTrainer)")
    print("=" * 120)

    header = (
        f"{'Model':<20} | {'Rel L2':<15} | {'MSE':<15} | {'PDE Res':<15} | "
        f"{'BC Viol':<15} | {'IC Viol':<15} | {'Time (s)':<12}"
    )
    print(header)
    print("-" * 120)

    for model_name in model_names:
        data = aggregated[model_name]
        rel_l2 = f"{data.get('relative_l2_error', {}).get('mean', float('nan')):.4e}"
        mse = f"{data.get('mse', {}).get('mean', float('nan')):.4e}"
        pde_res = f"{data.get('pde_residual_norm', {}).get('mean', float('nan')):.4e}"
        bc_viol = f"{data.get('bc_violation', {}).get('mean', float('nan')):.4e}"
        ic_viol = f"{data.get('ic_violation', {}).get('mean', float('nan')):.4e}"
        time_s = f"{data.get('training_time', {}).get('mean', 0.0):.1f}"

        print(
            f"{model_name:<20} | {rel_l2:<15} | {mse:<15} | {pde_res:<15} | "
            f"{bc_viol:<15} | {ic_viol:<15} | {time_s:<12}"
        )

    print("=" * 120)


def main() -> None:
    """Main entry point for RQ3 experiment."""
    parser = argparse.ArgumentParser(
        description='RQ3: Model Comparison Experiment (3-Stage HybridTrainer)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/results/rq3',
        help='Output directory for results and plots'
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Experiment configuration
    model_names = ["pinn_only", "hybrid_no_gate", "hybrid_full"]
    seeds = [42, 123, 456, 789, 1024]
    device = "cuda"

    # PDE problem
    pde = Heat1D(alpha=1.0, length=1.0, t_max=1.0)

    # Benchmark runner
    benchmark_config = BenchmarkConfig(
        n_collocation=2000,
        n_boundary=200,
        n_initial=200,
        include_physics_consistency=False,
        device=device,
    )
    benchmark_runner = BenchmarkRunner(config=benchmark_config)

    # Run experiments
    print("=" * 120)
    print("RQ3: Model Comparison Experiment (3-Stage HybridTrainer)")
    print("=" * 120)
    print(f"Models: {model_names}")
    print(f"Seeds: {seeds}")
    print(f"Total experiments: {len(model_names) * len(seeds)}")
    print("Training: Stage1=200ep/1e-3, Stage2=500ep/1e-4, Stage3=200ep/1e-5")
    print()

    all_results: list[dict[str, Any]] = []

    for model_name in model_names:
        print(f"\n--- Model: {model_name} ---")
        for seed in seeds:
            result = run_single_experiment(
                model_name=model_name,
                seed=seed,
                pde=pde,
                benchmark_runner=benchmark_runner,
                device=device,
            )
            all_results.append(result)

    # Aggregate results
    aggregated = aggregate_results(all_results, model_names)

    # Save results
    results_data = {
        "experiment": "rq3_model_comparison",
        "training_pipeline": "3-stage HybridTrainer",
        "model_names": model_names,
        "seeds": seeds,
        "config": {
            "device": device,
            "stage1_epochs": 200,
            "stage1_lr": "1e-3",
            "stage2_epochs": 500,
            "stage2_lr": "1e-4",
            "stage3_epochs": 200,
            "stage3_lr": "1e-5",
            "batch_size": 256,
            "num_collocation_points": 4000,
            "num_boundary_points": 500,
        },
        "raw_results": all_results,
        "aggregated_results": aggregated,
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir / 'results.json'}")

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(all_results, aggregated, output_dir)
    print(f"Plots saved to {output_dir}")

    # Print summary
    print_summary(aggregated, model_names)


if __name__ == "__main__":
    main()
