#!/usr/bin/env python3

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
from causal_pinn.physics.pde.burgers_1d import Burgers1D
from causal_pinn.training.trainer import HybridTrainer
from causal_pinn.utils.config import TrainingConfig
from causal_pinn.utils.random import set_seed

LAMBDA_VALUES = [0.0, 0.01, 0.1, 0.5, 1.0]
SEEDS = [42, 123, 456, 789, 1024]


def safe_mean_std(values: list[float]) -> tuple[float, float]:
    finite_values = [v for v in values if np.isfinite(v)]
    if not finite_values:
        return float("nan"), float("nan")
    arr = np.asarray(finite_values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def run_single_experiment(
    lambda_physics: float,
    seed: int,
    pde: Burgers1D,
    benchmark_runner: BenchmarkRunner,
) -> dict[str, Any]:
    set_seed(seed)

    model = PINNBase(input_dim=2, output_dim=1, hidden_dims=[64, 64, 64])
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
        device="cuda",
        lambda_physics=lambda_physics,
        lambda_data=1.0,
        causal_epsilon=1.0,
    )

    trainer = HybridTrainer(
        model=model,
        pde_problem=pde,
        config=config,
        device=config.device,
    )

    start = time.perf_counter()
    try:
        history = trainer.train()
        training_success = True
        error_message = None
    except Exception as exc:  # noqa: BLE001
        history = {}
        training_success = False
        error_message = str(exc)
    training_time = time.perf_counter() - start

    try:
        metrics = benchmark_runner.run_single(model, pde)
        eval_success = True
    except Exception as exc:  # noqa: BLE001
        metrics = {}
        eval_success = False
        if error_message is None:
            error_message = str(exc)

    return {
        "lambda_physics": lambda_physics,
        "seed": seed,
        "training_success": training_success,
        "eval_success": eval_success,
        "training_time": training_time,
        "training_history": history,
        "evaluation_metrics": metrics,
        "error_message": error_message,
    }


def aggregate_results(
    all_results: list[dict[str, Any]],
    lambda_values: list[float],
) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}

    for lam in lambda_values:
        rows = [r for r in all_results if r["lambda_physics"] == lam and r["eval_success"]]
        pde_vals = [r["evaluation_metrics"].get("pde_residual_norm", float("nan")) for r in rows]
        mse_vals = [r["evaluation_metrics"].get("mse", float("nan")) for r in rows]
        mae_vals = [r["evaluation_metrics"].get("mae", float("nan")) for r in rows]
        bc_vals = [r["evaluation_metrics"].get("bc_violation", float("nan")) for r in rows]
        ic_vals = [r["evaluation_metrics"].get("ic_violation", float("nan")) for r in rows]

        pde_mean, pde_std = safe_mean_std(pde_vals)
        mse_mean, mse_std = safe_mean_std(mse_vals)
        mae_mean, mae_std = safe_mean_std(mae_vals)
        bc_mean, bc_std = safe_mean_std(bc_vals)
        ic_mean, ic_std = safe_mean_std(ic_vals)

        aggregated[str(lam)] = {
            "n_success": len(rows),
            "pde_residual_mean": pde_mean,
            "pde_residual_std": pde_std,
            "mse_mean": mse_mean,
            "mse_std": mse_std,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
            "bc_violation_mean": bc_mean,
            "bc_violation_std": bc_std,
            "ic_violation_mean": ic_mean,
            "ic_violation_std": ic_std,
            "pde_residual_values": [float(v) for v in pde_vals],
        }

    return aggregated


def generate_plots(aggregated: dict[str, dict[str, Any]], output_dir: Path) -> None:
    lambda_values = [float(k) for k in aggregated]
    lambda_values.sort()

    pde_mean = [aggregated[str(lam)]["pde_residual_mean"] for lam in lambda_values]
    pde_std = [aggregated[str(lam)]["pde_residual_std"] for lam in lambda_values]
    mse_mean = [aggregated[str(lam)]["mse_mean"] for lam in lambda_values]
    mse_std = [aggregated[str(lam)]["mse_std"] for lam in lambda_values]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        lambda_values,
        pde_mean,
        yerr=pde_std,
        fmt="o-",
        linewidth=2,
        markersize=8,
        capsize=5,
        color="#1f77b4",
    )
    ax.set_xscale("log")
    ax.set_xlabel("lambda_physics")
    ax.set_ylabel("PDE residual norm")
    ax.set_title("RQ2b Burgers1D: lambda_physics vs PDE residual")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "rq2b_lambda_vs_pde_residual.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(
        lambda_values,
        mse_mean,
        yerr=mse_std,
        fmt="s-",
        linewidth=2,
        markersize=8,
        capsize=5,
        color="#2ca02c",
    )
    ax.set_xscale("log")
    ax.set_xlabel("lambda_physics")
    ax.set_ylabel("MSE")
    ax.set_title("RQ2b Burgers1D: lambda_physics vs MSE")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "rq2b_lambda_vs_mse.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    for lam in lambda_values:
        vals = aggregated[str(lam)]["pde_residual_values"]
        x = np.full(len(vals), lam)
        ax.scatter(x, vals, alpha=0.7, s=30)
    ax.set_xscale("log")
    ax.set_xlabel("lambda_physics")
    ax.set_ylabel("PDE residual norm (seed-level)")
    ax.set_title("RQ2b Burgers1D: seed-level PDE residual distribution")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "rq2b_seed_distribution.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2b Burgers1D lambda sweep")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/rq2b",
        help="Output directory for results and plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pde = Burgers1D(viscosity=0.01, length=1.0, t_max=1.0)
    benchmark_runner = BenchmarkRunner(
        config=BenchmarkConfig(
            n_collocation=2000,
            n_boundary=200,
            n_initial=200,
            include_physics_consistency=False,
            device="cuda",
        )
    )

    all_results: list[dict[str, Any]] = []
    print("=" * 80)
    print("RQ2b: Burgers1D lambda sweep")
    print("=" * 80)
    print(f"Lambdas: {LAMBDA_VALUES}")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {len(LAMBDA_VALUES) * len(SEEDS)}")

    for lam in LAMBDA_VALUES:
        print(f"\n--- lambda_physics={lam} ---")
        for seed in SEEDS:
            print(f"  seed={seed} ...", end=" ", flush=True)
            result = run_single_experiment(
                lambda_physics=lam,
                seed=seed,
                pde=pde,
                benchmark_runner=benchmark_runner,
            )
            all_results.append(result)
            if result["eval_success"]:
                val = result["evaluation_metrics"].get("pde_residual_norm", float("nan"))
                print(f"done (pde_residual={val:.4e})")
            else:
                print("failed")

    aggregated = aggregate_results(all_results, LAMBDA_VALUES)
    generate_plots(aggregated, output_dir)

    best_lambda = min(
        LAMBDA_VALUES,
        key=lambda lam: aggregated[str(lam)]["pde_residual_mean"],
    )

    results_data = {
        "experiment": "rq2b_burgers_lambda_sweep",
        "pde": "Burgers1D",
        "lambda_values": LAMBDA_VALUES,
        "seeds": SEEDS,
        "config": {
            "stage1_epochs": 200,
            "stage2_epochs": 500,
            "stage3_epochs": 200,
            "batch_size": 256,
            "num_collocation_points": 4000,
            "num_boundary_points": 500,
            "device": "cuda",
        },
        "raw_results": all_results,
        "aggregated_results": aggregated,
        "best_lambda_by_pde_residual": best_lambda,
    }

    out_path = output_dir / "results.json"
    out_path.write_text(json.dumps(results_data, indent=2, default=str), encoding="utf-8")

    print(f"\nSaved results: {out_path}")
    print(f"Saved plots in: {output_dir}")
    print(f"Best lambda by PDE residual: {best_lambda}")


if __name__ == "__main__":
    main()
