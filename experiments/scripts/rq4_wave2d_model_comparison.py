#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from experiment_utils import create_hybrid_model

from causal_pinn.evaluation.benchmark_runner import BenchmarkConfig, BenchmarkRunner
from causal_pinn.models.hybrid import HybridConfig, HybridModelWrapper
from causal_pinn.models.pinn.base import PINNBase
from causal_pinn.physics.pde.wave_2d import Wave2D
from causal_pinn.training.trainer import HybridTrainer
from causal_pinn.utils.config import TrainingConfig
from causal_pinn.utils.random import set_seed

MODEL_NAMES = ["pinn_only", "hybrid_full"]
SEEDS = [42, 123, 456, 789, 1024]


def create_pinn_only_model() -> PINNBase:
    return PINNBase(input_dim=3, output_dim=1, hidden_dims=[64, 64, 64])


def create_hybrid_full_model(pde: Wave2D) -> HybridModelWrapper:
    hybrid = create_hybrid_model(
        pinn_dims=[64, 64, 64],
        fno_modes=8,
        fno_width=32,
        pde_problem=pde,
        config=HybridConfig(
            use_operator=True,
            use_pinn=True,
            use_causal_gate=True,
            operator_weight=1.0,
            physics_weight=1.0,
            gate_weight=0.01,
        ),
        fno_num_layers=3,
        input_dim=3,
        output_dim=1,
    )
    return HybridModelWrapper(hybrid)


def create_training_config(seed: int, device: str) -> TrainingConfig:
    return TrainingConfig(
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


def run_single_experiment(
    model_name: str,
    seed: int,
    pde: Wave2D,
    benchmark_runner: BenchmarkRunner,
    device: str,
) -> dict[str, Any]:
    print(f"  Running {model_name} with seed {seed}...", end=" ", flush=True)
    set_seed(seed)

    if model_name == "pinn_only":
        model = create_pinn_only_model()
    elif model_name == "hybrid_full":
        model = create_hybrid_full_model(pde)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    trainer = HybridTrainer(
        model=model,
        pde_problem=pde,
        config=create_training_config(seed, device),
        device=device,
    )

    try:
        start = time.perf_counter()
        history = trainer.train()
        training_time = time.perf_counter() - start
        metrics = benchmark_runner.run_single(model, pde)
        rel_l2 = metrics.get("relative_l2_error", float("nan"))
        pde_res = metrics.get("pde_residual_norm", float("nan"))
        print(
            f"done (rel_l2={rel_l2:.4e}, pde_res={pde_res:.4e}, time={training_time:.1f}s)"
        )
        return {
            "model_name": model_name,
            "seed": seed,
            "training_time": training_time,
            "training_history": history,
            "metrics": metrics,
            "success": True,
            "error_message": None,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"failed ({exc})")
        return {
            "model_name": model_name,
            "seed": seed,
            "training_time": 0.0,
            "training_history": {},
            "metrics": {},
            "success": False,
            "error_message": str(exc),
        }


def safe_mean_std(values: list[float]) -> tuple[float, float]:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan"), float("nan")
    arr = np.asarray(finite_values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    metrics_to_aggregate = [
        "relative_l2_error",
        "mse",
        "mae",
        "max_error",
        "pde_residual_norm",
        "bc_violation",
        "ic_violation",
    ]
    aggregated: dict[str, dict[str, dict[str, float]]] = {}

    for model_name in MODEL_NAMES:
        model_results = [
            row for row in results if row["model_name"] == model_name and row["success"]
        ]
        aggregated[model_name] = {}

        for metric in metrics_to_aggregate:
            values = [row["metrics"].get(metric, float("nan")) for row in model_results]
            mean_value, std_value = safe_mean_std(values)
            aggregated[model_name][metric] = {
                "mean": mean_value,
                "std": std_value,
            }

        times = [row["training_time"] for row in model_results]
        mean_time, std_time = safe_mean_std(times)
        aggregated[model_name]["training_time"] = {
            "mean": mean_time,
            "std": std_time,
        }

    return aggregated


def generate_plots(
    results: list[dict[str, Any]],
    aggregated: dict[str, dict[str, dict[str, float]]],
    output_dir: Path,
) -> None:
    model_names = list(aggregated.keys())
    colors = ["#1f77b4", "#d62728"]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    rel_l2_means = [aggregated[name]["relative_l2_error"]["mean"] for name in model_names]
    rel_l2_stds = [aggregated[name]["relative_l2_error"]["std"] for name in model_names]
    ax.bar(x, rel_l2_means, yerr=rel_l2_stds, color=colors, capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("RQ4 Wave2D: Hybrid vs PINN-only")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "rq4_wave2d_rel_l2.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    pde_means = [aggregated[name]["pde_residual_norm"]["mean"] for name in model_names]
    pde_stds = [aggregated[name]["pde_residual_norm"]["std"] for name in model_names]
    ax.bar(x, pde_means, yerr=pde_stds, color=colors, capsize=5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("PDE Residual Norm")
    ax.set_title("RQ4 Wave2D: Physics Residual Comparison")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "rq4_wave2d_pde_residual.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, model_name in enumerate(model_names):
        rows = [row for row in results if row["model_name"] == model_name and row["success"]]
        times = [row["training_time"] for row in rows]
        rel_l2 = [row["metrics"].get("relative_l2_error", float("nan")) for row in rows]
        valid_pairs = [(t, l2) for t, l2 in zip(times, rel_l2, strict=False) if np.isfinite(l2)]
        if valid_pairs:
            xs, ys = zip(*valid_pairs, strict=False)
            ax.scatter(xs, ys, color=colors[idx], label=model_name, s=120, alpha=0.85)
    ax.set_xlabel("Training Time (s)")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("RQ4 Wave2D: Training Time vs Accuracy")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "rq4_wave2d_training_efficiency.png", dpi=150)
    plt.close(fig)


def print_summary(aggregated: dict[str, dict[str, dict[str, float]]]) -> None:
    print("\n" + "=" * 96)
    print("RQ4 Wave2D Model Comparison Summary")
    print("=" * 96)
    header = (
        f"{'Model':<16} | {'Rel L2':<15} | {'PDE Res':<15} | "
        f"{'BC Viol':<15} | {'IC Viol':<15} | {'Time (s)':<12}"
    )
    print(header)
    print("-" * 96)
    for model_name in MODEL_NAMES:
        row = aggregated[model_name]
        print(
            f"{model_name:<16} | "
            f"{row['relative_l2_error']['mean']:<15.4e} | "
            f"{row['pde_residual_norm']['mean']:<15.4e} | "
            f"{row['bc_violation']['mean']:<15.4e} | "
            f"{row['ic_violation']['mean']:<15.4e} | "
            f"{row['training_time']['mean']:<12.1f}"
        )
    print("=" * 96)


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ4 Wave2D model comparison")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/rq4_wave2d_model_comparison",
        help="Output directory for results and plots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for training and evaluation",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pde = Wave2D(wave_speed=1.0, length_x=1.0, length_y=1.0, t_max=1.0)
    benchmark_runner = BenchmarkRunner(
        BenchmarkConfig(
            n_collocation=2000,
            n_boundary=200,
            n_initial=200,
            include_physics_consistency=False,
            device=args.device,
        )
    )

    print("=" * 96)
    print("RQ4: Wave2D Model Comparison")
    print("=" * 96)
    print(f"Models: {MODEL_NAMES}")
    print(f"Seeds: {SEEDS}")
    print(f"Total experiments: {len(MODEL_NAMES) * len(SEEDS)}")

    all_results: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        print(f"\n--- Model: {model_name} ---")
        for seed in SEEDS:
            result = run_single_experiment(model_name, seed, pde, benchmark_runner, args.device)
            all_results.append(result)

    aggregated = aggregate_results(all_results)

    results_data = {
        "experiment": "rq4_wave2d_model_comparison",
        "pde": "Wave2D",
        "model_names": MODEL_NAMES,
        "seeds": SEEDS,
        "config": {
            "device": args.device,
            "stage1_epochs": 200,
            "stage2_epochs": 500,
            "stage3_epochs": 200,
            "batch_size": 256,
            "num_collocation_points": 4000,
            "num_boundary_points": 500,
        },
        "raw_results": all_results,
        "aggregated_results": aggregated,
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results_data, indent=2, default=str), encoding="utf-8")
    generate_plots(all_results, aggregated, output_dir)
    print(f"\nSaved results: {results_path}")
    print_summary(aggregated)


if __name__ == "__main__":
    main()
