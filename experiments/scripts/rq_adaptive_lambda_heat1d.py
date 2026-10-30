#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from experiment_utils import create_model_factory, run_single_experiment

from causal_pinn.models.hybrid import HybridConfig, HybridModelWrapper
from causal_pinn.physics.pde.heat_1d import Heat1D
from causal_pinn.utils.config import TrainingConfig


def build_config(seed: int, adaptive_lambda_enabled: bool, device: str) -> TrainingConfig:
    return TrainingConfig(
        stage1_epochs=200,
        stage1_lr=1e-3,
        stage2_epochs=500,
        stage2_lr=1e-4,
        stage3_epochs=200,
        stage3_lr=1e-5,
        lambda_physics=1.0,
        lambda_data=1.0,
        adaptive_lambda_enabled=adaptive_lambda_enabled,
        adaptive_lambda_warmup_epochs=100,
        adaptive_lambda_min_scale=0.75,
        adaptive_lambda_max_scale=1.25,
        adaptive_lambda_adjust_rate=0.02,
        adaptive_lambda_alignment_threshold=0.1,
        adaptive_lambda_conflict_threshold=-0.1,
        adaptive_lambda_dominance_threshold=2.5,
        adaptive_lambda_progress_threshold=0.01,
        adaptive_lambda_ema_beta=0.9,
        adaptive_lambda_persistence=5,
        batch_size=256,
        num_collocation_points=4000,
        num_boundary_points=500,
        causal_epsilon=1.0,
        seed=seed,
        device=device,
    )


def summarize(results_by_seed: dict[int, dict[str, Any]]) -> dict[str, dict[str, float]]:
    rel_l2 = []
    pde_residual = []
    training_time = []

    for result in results_by_seed.values():
        metrics = result["metrics"]
        rel_l2.append(float(metrics["relative_l2_error"]))
        pde_residual.append(float(metrics["pde_residual_norm"]))
        history = result["history"]
        total_epochs = (
            len(history["stage1_total_loss"])
            + len(history["stage2_total_loss"])
            + len(history["stage3_total_loss"])
        )
        training_time.append(float(total_epochs))

    return {
        "relative_l2_error": {
            "mean": float(np.mean(rel_l2)),
            "std": float(np.std(rel_l2)),
        },
        "pde_residual_norm": {
            "mean": float(np.mean(pde_residual)),
            "std": float(np.std(pde_residual)),
        },
        "effective_epochs": {
            "mean": float(np.mean(training_time)),
            "std": float(np.std(training_time)),
        },
    }


def create_wrapped_hybrid_factory(pde: Heat1D) -> Callable[[], HybridModelWrapper]:
    base_factory = create_model_factory(
        model_type="hybrid",
        pinn_dims=[64, 64, 64, 64],
        fno_modes=8,
        fno_width=32,
        pde_problem=pde,
        hybrid_config=HybridConfig(use_operator=True, use_pinn=True, use_causal_gate=True),
        input_dim=2,
        output_dim=1,
    )

    def factory() -> HybridModelWrapper:
        return HybridModelWrapper(base_factory())

    return factory


def run_seeded_comparison(
    pde: Heat1D,
    seeds: list[int],
    adaptive_lambda_enabled: bool,
    device: str,
) -> dict[int, dict[str, Any]]:
    factory = create_wrapped_hybrid_factory(pde)
    results_by_seed: dict[int, dict[str, Any]] = {}

    for seed in seeds:
        config = build_config(
            seed=seed,
            adaptive_lambda_enabled=adaptive_lambda_enabled,
            device=device,
        )
        results_by_seed[seed] = run_single_experiment(
            model=factory(),
            pde_problem=pde,
            config=config,
            seed=seed,
            evaluate=True,
        )

    return results_by_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/rq_adaptive_lambda_heat1d"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pde = Heat1D(alpha=1.0, length=1.0, t_max=1.0)
    seeds = [42, 123, 456, 789, 1024]
    comparisons = [
        ("hybrid_fixed_lambda", False),
        ("hybrid_adaptive_lambda", True),
    ]

    raw_results: list[dict[str, Any]] = []
    aggregated_results: dict[str, dict[str, dict[str, float]]] = {}

    for label, adaptive in comparisons:
        results_by_seed = run_seeded_comparison(
            pde=pde,
            seeds=seeds,
            adaptive_lambda_enabled=adaptive,
            device=args.device,
        )

        for seed, result in results_by_seed.items():
            raw_results.append(
                {
                    "model_name": label,
                    "seed": seed,
                    "metrics": result["metrics"],
                    "training_history": result["history"],
                }
            )

        aggregated_results[label] = summarize(results_by_seed)

    output = {
        "experiment": "rq_adaptive_lambda_heat1d",
        "model_names": [label for label, _ in comparisons],
        "seeds": seeds,
        "raw_results": raw_results,
        "aggregated_results": aggregated_results,
    }

    output_path = args.output_dir / "results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
