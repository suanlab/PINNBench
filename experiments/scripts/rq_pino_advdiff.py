#!/usr/bin/env python3
"""PINO comparison on AdvectionDiffusion1D.

Compares three model configurations:
1. PINN-only
2. PINO (physics-informed neural operator)
3. Hybrid (PINN + FNO + causal gate)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from experiment_utils import create_hybrid_model

from causal_pinn.evaluation.benchmark_runner import BenchmarkConfig, BenchmarkRunner
from causal_pinn.models.hybrid import HybridConfig, HybridModelWrapper
from causal_pinn.models.operators import FNO
from causal_pinn.models.pinn import PINNBase
from causal_pinn.models.pino import PINOModel
from causal_pinn.physics.pde import AdvectionDiffusion1D
from causal_pinn.training.pino_trainer import PINOTrainer
from causal_pinn.training.trainer import HybridTrainer
from causal_pinn.utils.config import TrainingConfig
from causal_pinn.utils.random import set_seed

DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]


def create_pinn_only_model() -> PINNBase:
    return PINNBase(input_dim=2, output_dim=1, hidden_dims=[64, 64, 64])


def create_pino_model(pde: AdvectionDiffusion1D) -> PINOModel:
    fno = FNO(
        in_channels=2,
        out_channels=1,
        modes=8,
        width=32,
        num_layers=4,
        dim=1,
    )
    return PINOModel(fno=fno, pde_problem=pde)


def create_hybrid_wrapped_model(pde: AdvectionDiffusion1D) -> HybridModelWrapper:
    hybrid = create_hybrid_model(
        pinn_dims=[64, 64, 64],
        fno_modes=8,
        fno_width=32,
        fno_num_layers=4,
        pde_problem=pde,
        config=HybridConfig(
            use_operator=True,
            use_pinn=True,
            use_causal_gate=True,
        ),
        input_dim=2,
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


def train_model(
    model_name: str,
    model: torch.nn.Module,
    pde: AdvectionDiffusion1D,
    config: TrainingConfig,
    device: str,
) -> tuple[dict[str, Any], float]:
    start = time.time()

    if model_name == "pino":
        trainer = PINOTrainer(
            model=model, pde_problem=pde, config=config, device=device,
        )
        history = trainer.train().to_dict()
    else:
        trainer = HybridTrainer(
            model=model, pde_problem=pde, config=config, device=device,
        )
        history = trainer.train()

    return history, time.time() - start


def run_single_experiment(
    model_name: str,
    seed: int,
    pde: AdvectionDiffusion1D,
    benchmark_runner: BenchmarkRunner,
    device: str,
) -> dict[str, Any]:
    print(f"  Running {model_name} with seed {seed}...", end=" ", flush=True)
    set_seed(seed)
    config = create_training_config(seed=seed, device=device)

    try:
        if model_name == "pinn_only":
            model = create_pinn_only_model()
        elif model_name == "pino":
            model = create_pino_model(pde)
        elif model_name == "hybrid":
            model = create_hybrid_wrapped_model(pde)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        history, training_time = train_model(
            model_name, model, pde, config, device,
        )
        metrics = benchmark_runner.run_single(model, pde)

        rel_l2 = metrics.get("relative_l2_error", float("nan"))
        print(f"Done. Rel L2: {rel_l2:.4e}, Time: {training_time:.1f}s")

        return {
            "model_name": model_name,
            "seed": seed,
            "training_time": training_time,
            "training_history": history,
            "metrics": metrics,
            "success": True,
        }
    except Exception as exc:
        print(f"Failed: {exc}")
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
    metrics_to_aggregate = [
        "relative_l2_error",
        "mse",
        "mae",
        "max_error",
        "pde_residual_norm",
        "bc_violation",
        "ic_violation",
    ]
    aggregated: dict[str, dict[str, Any]] = {}

    for model_name in model_names:
        model_results = [
            r for r in results
            if r["model_name"] == model_name and r["success"]
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
            values = [
                r["metrics"].get(metric, float("nan"))
                for r in model_results
            ]
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PINO comparison on AdvectionDiffusion1D",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/rq_pino_advdiff",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--num-seeds", type=int, default=10, help="Number of seeds",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Explicit seed list",
    )
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else DEFAULT_SEEDS[: args.num_seeds]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_names = ["pinn_only", "pino", "hybrid"]
    pde = AdvectionDiffusion1D(velocity=1.0, alpha=0.1, length=1.0, t_max=1.0)

    benchmark_runner = BenchmarkRunner(
        config=BenchmarkConfig(
            n_collocation=2000,
            n_boundary=200,
            n_initial=200,
            include_physics_consistency=False,
            device=args.device,
        )
    )

    all_results: list[dict[str, Any]] = []
    for model_name in model_names:
        print(f"\n--- Model: {model_name} ---")
        for seed in seeds:
            result = run_single_experiment(
                model_name=model_name,
                seed=seed,
                pde=pde,
                benchmark_runner=benchmark_runner,
                device=args.device,
            )
            all_results.append(result)

    aggregated = aggregate_results(all_results, model_names)

    result_payload = {
        "experiment": "rq_pino_advdiff",
        "pde": "AdvectionDiffusion1D",
        "pde_config": {
            "velocity": 1.0,
            "alpha": 0.1,
            "length": 1.0,
            "t_max": 1.0,
        },
        "model_names": model_names,
        "seeds": seeds,
        "raw_results": all_results,
        "aggregated_results": aggregated,
    }

    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, default=str)

    print(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
