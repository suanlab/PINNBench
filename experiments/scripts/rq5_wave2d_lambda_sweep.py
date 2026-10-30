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
from causal_pinn.physics.pde.wave_2d import Wave2D
from causal_pinn.training.trainer import HybridTrainer
from causal_pinn.utils.config import TrainingConfig
from causal_pinn.utils.random import set_seed

LAMBDA_VALUES = [0.0, 0.01, 0.1, 0.5, 1.0]
SEEDS = [42, 123, 456, 789, 1024]


def create_model(pde: Wave2D) -> HybridModelWrapper:
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


def create_training_config(seed: int, device: str, lambda_physics: float) -> TrainingConfig:
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
        lambda_physics=lambda_physics,
        lambda_data=1.0,
        causal_epsilon=1.0,
    )


def safe_mean_std(values: list[float]) -> tuple[float, float]:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan"), float("nan")
    arr = np.asarray(finite_values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def run_single_experiment(
    lambda_physics: float,
    seed: int,
    pde: Wave2D,
    benchmark_runner: BenchmarkRunner,
    device: str,
) -> dict[str, Any]:
    print(f"  seed={seed} ...", end=" ", flush=True)
    set_seed(seed)

    model = create_model(pde)
    trainer = HybridTrainer(
        model=model,
        pde_problem=pde,
        config=create_training_config(seed, device, lambda_physics),
        device=device,
    )

    try:
        start = time.perf_counter()
        training_history = trainer.train()
        training_time = time.perf_counter() - start
        metrics = benchmark_runner.run_single(model, pde)
        print(
            "done "
            f"(rel_l2={metrics.get('relative_l2_error', float('nan')):.4e}, "
            f"pde_res={metrics.get('pde_residual_norm', float('nan')):.4e})"
        )
        return {
            "lambda_physics": lambda_physics,
            "seed": seed,
            "training_success": True,
            "eval_success": True,
            "training_time": training_time,
            "training_history": training_history,
            "evaluation_metrics": metrics,
            "error_message": None,
        }
    except Exception as exc:  # noqa: BLE001
        print(f"failed ({exc})")
        return {
            "lambda_physics": lambda_physics,
            "seed": seed,
            "training_success": False,
            "eval_success": False,
            "training_time": 0.0,
            "training_history": {},
            "evaluation_metrics": {},
            "error_message": str(exc),
        }


def aggregate_results(all_results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}

    for lam in LAMBDA_VALUES:
        rows = [row for row in all_results if row["lambda_physics"] == lam and row["eval_success"]]
        rel_l2_vals = [
            row["evaluation_metrics"].get("relative_l2_error", float("nan")) for row in rows
        ]
        pde_vals = [
            row["evaluation_metrics"].get("pde_residual_norm", float("nan")) for row in rows
        ]
        mse_vals = [row["evaluation_metrics"].get("mse", float("nan")) for row in rows]
        bc_vals = [row["evaluation_metrics"].get("bc_violation", float("nan")) for row in rows]
        ic_vals = [row["evaluation_metrics"].get("ic_violation", float("nan")) for row in rows]
        time_vals = [row["training_time"] for row in rows]

        rel_l2_mean, rel_l2_std = safe_mean_std(rel_l2_vals)
        pde_mean, pde_std = safe_mean_std(pde_vals)
        mse_mean, mse_std = safe_mean_std(mse_vals)
        bc_mean, bc_std = safe_mean_std(bc_vals)
        ic_mean, ic_std = safe_mean_std(ic_vals)
        time_mean, time_std = safe_mean_std(time_vals)

        aggregated[str(lam)] = {
            "n_success": len(rows),
            "relative_l2_mean": rel_l2_mean,
            "relative_l2_std": rel_l2_std,
            "pde_residual_mean": pde_mean,
            "pde_residual_std": pde_std,
            "mse_mean": mse_mean,
            "mse_std": mse_std,
            "bc_violation_mean": bc_mean,
            "bc_violation_std": bc_std,
            "ic_violation_mean": ic_mean,
            "ic_violation_std": ic_std,
            "training_time_mean": time_mean,
            "training_time_std": time_std,
            "relative_l2_values": [float(v) for v in rel_l2_vals],
            "pde_residual_values": [float(v) for v in pde_vals],
        }

    return aggregated


def generate_plots(aggregated: dict[str, dict[str, Any]], output_dir: Path) -> None:
    lambda_values = sorted(float(key) for key in aggregated)
    rel_l2_mean = [aggregated[str(lam)]["relative_l2_mean"] for lam in lambda_values]
    rel_l2_std = [aggregated[str(lam)]["relative_l2_std"] for lam in lambda_values]
    pde_mean = [aggregated[str(lam)]["pde_residual_mean"] for lam in lambda_values]
    pde_std = [aggregated[str(lam)]["pde_residual_std"] for lam in lambda_values]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        lambda_values,
        rel_l2_mean,
        yerr=rel_l2_std,
        fmt="o-",
        linewidth=2,
        markersize=8,
        capsize=5,
        color="#1f77b4",
    )
    ax.set_xscale("log")
    ax.set_xlabel("lambda_physics")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("RQ5 Wave2D: lambda_physics vs Relative L2 Error")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "rq5_wave2d_lambda_vs_rel_l2.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        lambda_values,
        pde_mean,
        yerr=pde_std,
        fmt="s-",
        linewidth=2,
        markersize=8,
        capsize=5,
        color="#2ca02c",
    )
    ax.set_xscale("log")
    ax.set_xlabel("lambda_physics")
    ax.set_ylabel("PDE Residual Norm")
    ax.set_title("RQ5 Wave2D: lambda_physics vs PDE Residual")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "rq5_wave2d_lambda_vs_pde_residual.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for lam in lambda_values:
        ax.scatter(
            aggregated[str(lam)]["relative_l2_values"],
            aggregated[str(lam)]["pde_residual_values"],
            s=40,
            alpha=0.75,
            label=f"lambda={lam}",
        )
    ax.set_xlabel("Relative L2 Error")
    ax.set_ylabel("PDE Residual Norm")
    ax.set_title("RQ5 Wave2D: Accuracy-Physics Trade-off")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "rq5_wave2d_tradeoff_scatter.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ5 Wave2D lambda sweep")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/rq5_wave2d_lambda_sweep",
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
    print("RQ5: Wave2D lambda sweep")
    print("=" * 96)
    print(f"Lambdas: {LAMBDA_VALUES}")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {len(LAMBDA_VALUES) * len(SEEDS)}")

    all_results: list[dict[str, Any]] = []
    for lam in LAMBDA_VALUES:
        print(f"\n--- lambda_physics={lam} ---")
        for seed in SEEDS:
            all_results.append(run_single_experiment(lam, seed, pde, benchmark_runner, args.device))

    aggregated = aggregate_results(all_results)
    generate_plots(aggregated, output_dir)

    best_lambda_rel_l2 = min(
        LAMBDA_VALUES,
        key=lambda lam: aggregated[str(lam)]["relative_l2_mean"],
    )
    best_lambda_pde = min(
        LAMBDA_VALUES,
        key=lambda lam: aggregated[str(lam)]["pde_residual_mean"],
    )

    results = {
        "experiment": "rq5_wave2d_lambda_sweep",
        "pde": "Wave2D",
        "lambda_values": LAMBDA_VALUES,
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
        "best_lambda_by_relative_l2": best_lambda_rel_l2,
        "best_lambda_by_pde_residual": best_lambda_pde,
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved results: {results_path}")
    print(f"Best lambda by relative_l2: {best_lambda_rel_l2}")
    print(f"Best lambda by pde_residual: {best_lambda_pde}")


if __name__ == "__main__":
    main()
