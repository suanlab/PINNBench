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
TRAIN_WAVE_SPEED = 1.0
IN_DOMAIN_SPEEDS = [1.0]
OOD_SPEEDS = [1.25, 1.5]


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


def safe_mean_std(values: list[float]) -> tuple[float, float]:
    finite_values = [value for value in values if np.isfinite(value)]
    if not finite_values:
        return float("nan"), float("nan")
    arr = np.asarray(finite_values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def evaluate_on_speed(model: Any, wave_speed: float, device: str) -> dict[str, float]:
    pde = Wave2D(wave_speed=wave_speed, length_x=1.0, length_y=1.0, t_max=1.0)
    runner = BenchmarkRunner(
        BenchmarkConfig(
            n_collocation=2000,
            n_boundary=200,
            n_initial=200,
            include_physics_consistency=False,
            device=device,
        )
    )
    return runner.run_single(model, pde)


def run_single_experiment(model_name: str, seed: int, device: str) -> dict[str, Any]:
    print(f"  Running {model_name} with seed {seed}...", end=" ", flush=True)
    set_seed(seed)
    train_pde = Wave2D(wave_speed=TRAIN_WAVE_SPEED, length_x=1.0, length_y=1.0, t_max=1.0)

    if model_name == "pinn_only":
        model = create_pinn_only_model()
    elif model_name == "hybrid_full":
        model = create_hybrid_full_model(train_pde)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    trainer = HybridTrainer(
        model=model,
        pde_problem=train_pde,
        config=create_training_config(seed, device),
        device=device,
    )

    try:
        start = time.perf_counter()
        history = trainer.train()
        training_time = time.perf_counter() - start
        per_speed_metrics: dict[str, dict[str, float]] = {}
        for wave_speed in IN_DOMAIN_SPEEDS + OOD_SPEEDS:
            per_speed_metrics[str(wave_speed)] = evaluate_on_speed(model, wave_speed, device)

        print(
            "done "
            f"(id_rel_l2={per_speed_metrics['1.0'].get('relative_l2_error', float('nan')):.4e}, "
            f"ood15_rel_l2={per_speed_metrics['1.5'].get('relative_l2_error', float('nan')):.4e})"
        )
        return {
            "model_name": model_name,
            "seed": seed,
            "training_time": training_time,
            "training_history": history,
            "per_speed_metrics": per_speed_metrics,
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
            "per_speed_metrics": {},
            "success": False,
            "error_message": str(exc),
        }


def metric_values(
    results: list[dict[str, Any]],
    model_name: str,
    speeds: list[float],
    metric_name: str,
) -> list[float]:
    values: list[float] = []
    for row in results:
        if row["model_name"] != model_name or not row["success"]:
            continue
        for speed in speeds:
            metrics = row["per_speed_metrics"].get(str(speed), {})
            values.append(metrics.get(metric_name, float("nan")))
    return values


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    metrics = ["relative_l2_error", "pde_residual_norm", "bc_violation", "ic_violation"]

    for model_name in MODEL_NAMES:
        aggregated[model_name] = {"in_domain": {}, "ood": {}, "per_speed": {}}
        for metric_name in metrics:
            id_values = metric_values(results, model_name, IN_DOMAIN_SPEEDS, metric_name)
            ood_values = metric_values(results, model_name, OOD_SPEEDS, metric_name)
            id_mean, id_std = safe_mean_std(id_values)
            ood_mean, ood_std = safe_mean_std(ood_values)
            aggregated[model_name]["in_domain"][metric_name] = {"mean": id_mean, "std": id_std}
            aggregated[model_name]["ood"][metric_name] = {"mean": ood_mean, "std": ood_std}
            aggregated[model_name]["ood"][f"{metric_name}_gap"] = ood_mean - id_mean

        for speed in IN_DOMAIN_SPEEDS + OOD_SPEEDS:
            speed_key = str(speed)
            aggregated[model_name]["per_speed"][speed_key] = {}
            for metric_name in metrics:
                values = metric_values(results, model_name, [speed], metric_name)
                mean_value, std_value = safe_mean_std(values)
                aggregated[model_name]["per_speed"][speed_key][metric_name] = {
                    "mean": mean_value,
                    "std": std_value,
                }

        time_values = [
            row["training_time"]
            for row in results
            if row["model_name"] == model_name and row["success"]
        ]
        time_mean, time_std = safe_mean_std(time_values)
        aggregated[model_name]["training_time"] = {"mean": time_mean, "std": time_std}

    return aggregated


def generate_plots(aggregated: dict[str, dict[str, Any]], output_dir: Path) -> None:
    colors = ["#1f77b4", "#d62728"]
    model_names = list(aggregated.keys())
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    id_means = [aggregated[name]["in_domain"]["relative_l2_error"]["mean"] for name in model_names]
    ood_means = [aggregated[name]["ood"]["relative_l2_error"]["mean"] for name in model_names]
    ax.bar(x - width / 2, id_means, width, label="in-domain", color=colors[0], alpha=0.85)
    ax.bar(x + width / 2, ood_means, width, label="OOD", color=colors[1], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("RQ6 Wave2D: In-domain vs OOD Accuracy")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "rq6_ood_rel_l2.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    id_pde = [aggregated[name]["in_domain"]["pde_residual_norm"]["mean"] for name in model_names]
    ood_pde = [aggregated[name]["ood"]["pde_residual_norm"]["mean"] for name in model_names]
    ax.bar(x - width / 2, id_pde, width, label="in-domain", color=colors[0], alpha=0.85)
    ax.bar(x + width / 2, ood_pde, width, label="OOD", color=colors[1], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylabel("PDE Residual Norm")
    ax.set_title("RQ6 Wave2D: In-domain vs OOD Physics Residual")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "rq6_ood_pde_residual.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    speeds = [str(speed) for speed in IN_DOMAIN_SPEEDS + OOD_SPEEDS]
    speed_positions = np.arange(len(speeds))
    plot_width = 0.35
    for idx, model_name in enumerate(model_names):
        means = [
            aggregated[model_name]["per_speed"][speed]["relative_l2_error"]["mean"]
            for speed in speeds
        ]
        ax.bar(speed_positions + idx * plot_width, means, plot_width, label=model_name, alpha=0.85)
    ax.set_xticks(speed_positions + plot_width / 2)
    ax.set_xticklabels(speeds)
    ax.set_xlabel("wave_speed")
    ax.set_ylabel("Relative L2 Error")
    ax.set_title("RQ6 Wave2D: Per-coefficient Accuracy")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "rq6_ood_per_speed_rel_l2.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ6 Wave2D OOD parametric generalization")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/rq6_ood",
        help="Output directory for results and plots",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for training and evaluation",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help=f"Seeds to run (default: {SEEDS})",
    )
    args = parser.parse_args()
    seeds = args.seeds or SEEDS

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print("RQ6: Wave2D OOD parametric generalization")
    print("=" * 96)
    print(f"Train wave_speed: {TRAIN_WAVE_SPEED}")
    print(f"In-domain speeds: {IN_DOMAIN_SPEEDS}")
    print(f"OOD speeds: {OOD_SPEEDS}")
    print(f"Models: {MODEL_NAMES}")
    print(f"Seeds: {seeds}")

    all_results: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        print(f"\n--- Model: {model_name} ---")
        for seed in seeds:
            all_results.append(run_single_experiment(model_name, seed, args.device))

    aggregated = aggregate_results(all_results)
    generate_plots(aggregated, output_dir)

    results = {
        "experiment": "rq6_ood_parametric_generalization",
        "pde": "Wave2D",
        "train_wave_speed": TRAIN_WAVE_SPEED,
        "in_domain_speeds": IN_DOMAIN_SPEEDS,
        "ood_speeds": OOD_SPEEDS,
        "model_names": MODEL_NAMES,
        "seeds": seeds,
        "raw_results": all_results,
        "aggregated_results": aggregated,
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved results: {results_path}")


if __name__ == "__main__":
    main()
