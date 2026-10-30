#!/usr/bin/env python3
"""RQ1e: Hybrid model comparison on 1D KdV.

Compares three model configurations with the 3-stage HybridTrainer pipeline:
1. PINN-only
2. Hybrid (no gate)
3. Hybrid (full causal gate)
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
from causal_pinn.physics.pde.kdv_1d import KdV1D
from causal_pinn.training.trainer import HybridTrainer
from causal_pinn.utils.config import TrainingConfig
from causal_pinn.utils.random import set_seed


def create_pinn_only_model() -> PINNBase:
    return PINNBase(
        input_dim=2,
        output_dim=1,
        hidden_dims=[64, 64, 64, 4],
    )


def create_hybrid_model(pde: KdV1D, use_causal_gate: bool) -> HybridModelWrapper:
    pinn_network = PINNBase(
        input_dim=2,
        output_dim=1,
        hidden_dims=[64, 64, 64, 4],
    )

    operator = FNO(
        in_channels=2,
        out_channels=1,
        modes=8,
        width=32,
        num_layers=4,
        dim=1,
    )

    causal_gate = CausalGate(
        out_dim=1,
        hidden_dim=32,
        causal_strength=1.0,
    )

    config = HybridConfig(
        use_operator=True,
        use_pinn=True,
        use_causal_gate=use_causal_gate,
        operator_weight=1.0,
        physics_weight=1.0,
        gate_weight=0.01,
    )

    hybrid_pinn = HybridPINNOperator(
        pinn_network=pinn_network,
        operator=operator,
        causal_gate=causal_gate,
        pde_problem=pde,
        config=config,
    )

    return HybridModelWrapper(hybrid_pinn)


def train_with_hybrid_trainer(
    model: torch.nn.Module,
    pde: KdV1D,
    seed: int,
    device: str,
) -> tuple[dict[str, Any], float]:
    set_seed(seed)

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
    pde: KdV1D,
    benchmark_runner: BenchmarkRunner,
    device: str,
) -> dict[str, Any]:
    print(f"  Running {model_name} with seed {seed}...", end=" ", flush=True)

    try:
        if model_name == "pinn_only":
            model = create_pinn_only_model()
            training_history, training_time = train_with_hybrid_trainer(
                model=model,
                pde=pde,
                seed=seed,
                device=device,
            )
            eval_model = model
        elif model_name == "hybrid_no_gate":
            model = create_hybrid_model(pde=pde, use_causal_gate=False)
            training_history, training_time = train_with_hybrid_trainer(
                model=model,
                pde=pde,
                seed=seed,
                device=device,
            )
            eval_model = model
        elif model_name == "hybrid_full":
            model = create_hybrid_model(pde=pde, use_causal_gate=True)
            training_history, training_time = train_with_hybrid_trainer(
                model=model,
                pde=pde,
                seed=seed,
                device=device,
            )
            eval_model = model
        else:
            raise ValueError(f"Unknown model name: {model_name}")

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
    except Exception as exc:
        print(f"Failed: {exc}")
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
    aggregated: dict[str, dict[str, Any]] = {}

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
        model_results = [
            r for r in results if r["model_name"] == model_name and r["success"]
        ]

        if not model_results:
            aggregated[model_name] = {
                metric: {"mean": float("nan"), "std": float("nan")}
                for metric in metrics_to_aggregate
            }
            aggregated[model_name]["training_time"] = {
                "mean": float("nan"),
                "std": float("nan"),
            }
            continue

        aggregated[model_name] = {}
        for metric in metrics_to_aggregate:
            values = [r["metrics"].get(metric, float("nan")) for r in model_results]
            aggregated[model_name][metric] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
            }

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
    model_names = list(aggregated.keys())
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for index, model_name in enumerate(model_names):
        model_results = [
            r for r in results if r["model_name"] == model_name and r["success"]
        ]
        if not model_results:
            continue

        all_losses: list[list[float]] = []
        max_len = 0
        for result in model_results:
            history = result["training_history"]
            loss_seq = (
                history.get("stage1_total_loss", [])
                + history.get("stage2_total_loss", [])
                + history.get("stage3_total_loss", [])
            )
            if loss_seq:
                all_losses.append(loss_seq)
                max_len = max(max_len, len(loss_seq))

        if not all_losses:
            continue

        padded_losses = np.full((len(all_losses), max_len), np.nan)
        for row_index, loss_seq in enumerate(all_losses):
            padded_losses[row_index, : len(loss_seq)] = loss_seq

        mean_loss = np.nanmean(padded_losses, axis=0)
        std_loss = np.nanstd(padded_losses, axis=0)
        epochs = np.arange(max_len)

        ax.plot(
            epochs,
            mean_loss,
            label=model_name,
            color=colors[index % len(colors)],
            linewidth=2,
        )
        ax.fill_between(
            epochs,
            mean_loss - std_loss,
            mean_loss + std_loss,
            color=colors[index % len(colors)],
            alpha=0.2,
        )

    ax.set_xlabel("Epoch (all stages combined)")
    ax.set_ylabel("Total Loss")
    ax.set_title("RQ1e: Training Convergence Curves (KdV1D)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "rq1e_convergence_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 8))

    metrics_for_bar = [
        "relative_l2_error",
        "mse",
        "pde_residual_norm",
        "bc_violation",
        "ic_violation",
    ]
    metric_labels = [
        "Rel L2",
        "MSE",
        "PDE Residual",
        "BC Violation",
        "IC Violation",
    ]

    x = np.arange(len(metrics_for_bar))
    width = 0.25

    for index, model_name in enumerate(model_names):
        means = [
            aggregated[model_name].get(metric, {}).get("mean", float("nan"))
            for metric in metrics_for_bar
        ]
        stds = [
            aggregated[model_name].get(metric, {}).get("std", 0.0)
            for metric in metrics_for_bar
        ]

        if any(not np.isnan(mean) for mean in means):
            ax.bar(
                x + index * width,
                [value if not np.isnan(value) else 0 for value in means],
                width,
                yerr=[
                    std if not np.isnan(means[item_idx]) else 0
                    for item_idx, std in enumerate(stds)
                ],
                label=model_name,
                color=colors[index % len(colors)],
                capsize=3,
                alpha=0.8,
            )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Value (log scale)", fontsize=12)
    ax.set_title("RQ1e: Model Comparison Across Metrics (KdV1D)", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "rq1e_metrics_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    data = []
    labels = []
    for model_name in model_names:
        model_results = [
            r for r in results if r["model_name"] == model_name and r["success"]
        ]
        values = [
            r["metrics"].get("pde_residual_norm", float("nan")) for r in model_results
        ]
        filtered_values = [value for value in values if not np.isnan(value)]
        if filtered_values:
            data.append(filtered_values)
            labels.append(model_name)

    if data:
        boxplot = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch, label in zip(boxplot["boxes"], labels, strict=False):
            color_idx = model_names.index(label) % len(colors)
            patch.set_facecolor(colors[color_idx])
            patch.set_alpha(0.4)
        for idx, values in enumerate(data, start=1):
            jitter = np.random.normal(idx, 0.04, size=len(values))
            ax.scatter(jitter, values, alpha=0.6, s=30, c="#555555")

    ax.set_xlabel("Model")
    ax.set_ylabel("PDE Residual Norm")
    ax.set_title("RQ1e: Seed-to-Seed Variance (KdV1D)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "rq1e_seed_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(
    aggregated: dict[str, dict[str, Any]],
    model_names: list[str],
) -> None:
    print("\n" + "=" * 120)
    print("RQ1e: Hybrid Model Comparison Summary (KdV1D)")
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
    parser = argparse.ArgumentParser(
        description="RQ1e: Hybrid comparison on KdV1D (PINN vs Hybrid no-gate vs full)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/rq1e_kdv_hybrid",
        help="Output directory for results and plots",
    )
    parser.add_argument("--num-seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit seed list")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda)")
    args = parser.parse_args()

    DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = ["pinn_only", "hybrid_no_gate", "hybrid_full"]
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS[:args.num_seeds]
    device = args.device

    pde = KdV1D(dispersion=1.0, length=20.0, t_max=0.5, soliton_amplitude=3.0, soliton_center=10.0)

    benchmark_config = BenchmarkConfig(
        n_collocation=2000,
        n_boundary=200,
        n_initial=200,
        include_physics_consistency=False,
        device=device,
    )
    benchmark_runner = BenchmarkRunner(config=benchmark_config)

    print("=" * 120)
    print("RQ1e: Hybrid Model Comparison Experiment (KdV1D)")
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

    aggregated = aggregate_results(all_results, model_names)

    results_data = {
        "experiment": "rq1e_kdv_hybrid",
        "model_names": model_names,
        "seeds": seeds,
        "raw_results": all_results,
        "aggregated_results": aggregated,
    }

    with open(output_dir / "results.json", "w") as file_handle:
        json.dump(results_data, file_handle, indent=2, default=str)

    print(f"\nResults saved to {output_dir / 'results.json'}")

    print("\nGenerating plots...")
    generate_plots(all_results, aggregated, output_dir)
    print(f"Plots saved to {output_dir}")

    print_summary(aggregated, model_names)


if __name__ == "__main__":
    main()
