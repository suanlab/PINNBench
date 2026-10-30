#!/usr/bin/env python3
"""Unified policy suite runner for Paper A routing benchmark.

Runs all (policy, seed) combinations for a given PDE and writes a
``results.json`` file whose schema is compatible with
:pymethod:`EarlyTrainingPredictor.build_dataset`.

Usage examples::

    # Quick 3-seed run on Heat1D with default policies
    python run_policy_suite.py --pde heat1d --num-seeds 3 \
        --output-dir experiments/results/paper_a_heat1d

    # Full 10-seed run on KdV with lambda sweep
    python run_policy_suite.py --pde kdv1d --num-seeds 10 \
        --policies pinn_only hybrid_no_gate hybrid_full hybrid_full_lambda0.1 \
        --output-dir experiments/results/paper_a_kdv1d

    # PINO policy on Burgers (low viscosity)
    python run_policy_suite.py --pde burgers1d_lownu \
        --policies pinn_only pino --num-seeds 5 \
        --output-dir experiments/results/paper_a_burgers_lownu
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Framework imports
# ---------------------------------------------------------------------------
from causal_pinn.evaluation.benchmark_runner import BenchmarkConfig, BenchmarkRunner
from causal_pinn.models.hybrid.causal_gate import CausalGate
from causal_pinn.models.hybrid.hybrid_model import (
    HybridConfig,
    HybridModelWrapper,
    HybridPINNOperator,
)
from causal_pinn.models.operators.fno import FNO
from causal_pinn.models.pinn.base import PINNBase
from causal_pinn.physics.pde import (
    AdvectionDiffusion1D,
    AllenCahn1D,
    Burgers1D,
    Heat1D,
    KdV1D,
    MultiscaleHeat2D,
    NavierStokes2D,
    ReactionDiffusion1D,
    Wave2D,
)
from causal_pinn.training.trainer import HybridTrainer
from causal_pinn.utils.config import TrainingConfig
from causal_pinn.utils.random import set_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

DEFAULT_POLICIES = ["pinn_only", "hybrid_no_gate", "hybrid_full"]

# PDE registry: name -> (class, default_kwargs, input_dim, spatial_dim)
# input_dim = spatial_dim + 1 (time)
PDE_REGISTRY: dict[str, tuple[type, dict[str, Any], int, int]] = {
    # --- 1-D problems (input_dim=2) ---
    "heat1d": (
        Heat1D,
        {"alpha": 1.0, "length": 1.0, "t_max": 1.0},
        2, 1,
    ),
    "burgers1d": (
        Burgers1D,
        {"viscosity": 0.01, "length": 1.0, "t_max": 1.0},
        2, 1,
    ),
    "burgers1d_lownu": (
        Burgers1D,
        {"viscosity": 0.001, "length": 1.0, "t_max": 1.0},
        2, 1,
    ),
    "burgers1d_verylow": (
        Burgers1D,
        {"viscosity": 0.0001, "length": 2.0, "t_max": 0.5},
        2, 1,
    ),
    "advdiff1d": (
        AdvectionDiffusion1D,
        {"velocity": 1.0, "alpha": 0.1, "length": 1.0, "t_max": 1.0},
        2, 1,
    ),
    "advdiff1d_highpe": (
        AdvectionDiffusion1D,
        {"velocity": 5.0, "alpha": 0.01, "length": 1.0, "t_max": 0.5},
        2, 1,
    ),
    "kdv1d": (
        KdV1D,
        {
            "dispersion": 1.0,
            "length": 20.0,
            "t_max": 0.5,
            "soliton_amplitude": 3.0,
            "soliton_center": 10.0,
        },
        2, 1,
    ),
    "allencahn1d": (
        AllenCahn1D,
        {"epsilon": 0.01, "length": 2.0, "t_max": 1.0, "front_center": 1.0},
        2, 1,
    ),
    "allencahn1d_sharp": (
        AllenCahn1D,
        {"epsilon": 0.001, "length": 2.0, "t_max": 0.5, "front_center": 1.0},
        2, 1,
    ),
    "reaction_diffusion1d": (
        ReactionDiffusion1D,
        {"diffusion": 0.01, "reaction_rate": 1.0, "length": 1.0, "t_max": 1.0},
        2, 1,
    ),
    "reaction_diffusion1d_stiff": (
        ReactionDiffusion1D,
        {"diffusion": 0.01, "reaction_rate": 10.0, "length": 1.0, "t_max": 0.5},
        2, 1,
    ),
    # --- 2-D problems (input_dim=3) ---
    "wave2d": (
        Wave2D,
        {"wave_speed": 1.0, "length_x": 1.0, "length_y": 1.0, "t_max": 1.0},
        3, 2,
    ),
    "multiscale_heat2d": (
        MultiscaleHeat2D,
        {
            "alpha": 0.01,
            "scale_ratio": 4,
            "fine_amplitude": 0.5,
            "length_x": 1.0,
            "length_y": 1.0,
            "t_max": 1.0,
        },
        3, 2,
    ),
    "navier_stokes2d": (
        NavierStokes2D,
        {
            "viscosity": 0.1,
            "length_x": 2.0 * math.pi,
            "length_y": 2.0 * math.pi,
            "t_max": 1.0,
        },
        3, 2,
    ),
}

# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

# Standard architecture constants (match existing experiment scripts)
_PINN_HIDDEN_DIMS = [64, 64, 64, 64]
_FNO_MODES = 8
_FNO_WIDTH = 32
_FNO_NUM_LAYERS = 4
_GATE_HIDDEN_DIM = 32


def _parse_policy(policy_name: str) -> tuple[str, float]:
    """Parse a policy name into (base_policy, lambda_physics).

    Supports names like ``"hybrid_full"``, ``"hybrid_full_lambda0.1"``,
    ``"hybrid_no_gate_lambda5.0"``.

    Returns:
        (base_policy, lambda_physics) where base_policy is one of
        ``"pinn_only"``, ``"hybrid_no_gate"``, ``"hybrid_full"``, ``"pino"``.
    """
    lambda_physics = 1.0
    if "_lambda" in policy_name:
        base, lam_str = policy_name.rsplit("_lambda", maxsplit=1)
        lambda_physics = float(lam_str)
        return base, lambda_physics
    return policy_name, lambda_physics


def create_model(
    policy_name: str,
    pde: Any,
    input_dim: int,
    output_dim: int = 1,
) -> tuple[torch.nn.Module, float]:
    """Create model based on *policy_name*.

    Returns:
        (model, lambda_physics) tuple.
    """
    base_policy, lambda_physics = _parse_policy(policy_name)

    if base_policy == "pinn_only":
        model = PINNBase(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=_PINN_HIDDEN_DIMS,
        )
        return model, lambda_physics

    if base_policy == "pino":
        # PINO uses only the FNO branch (no PINN, no gate)
        fno_dim = 2 if input_dim == 3 else 1
        pinn_network = PINNBase(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=_PINN_HIDDEN_DIMS,
        )
        operator = FNO(
            in_channels=input_dim,
            out_channels=output_dim,
            modes=_FNO_MODES,
            width=_FNO_WIDTH,
            num_layers=_FNO_NUM_LAYERS,
            dim=fno_dim,
        )
        causal_gate = CausalGate(
            out_dim=output_dim,
            hidden_dim=_GATE_HIDDEN_DIM,
            causal_strength=1.0,
        )
        config = HybridConfig(
            use_operator=True,
            use_pinn=False,
            use_causal_gate=False,
        )
        hybrid = HybridPINNOperator(
            pinn_network=pinn_network,
            operator=operator,
            causal_gate=causal_gate,
            pde_problem=pde,
            config=config,
        )
        return HybridModelWrapper(hybrid), lambda_physics

    # hybrid_no_gate or hybrid_full
    use_gate = base_policy == "hybrid_full"
    fno_dim = 2 if input_dim == 3 else 1

    pinn_network = PINNBase(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=_PINN_HIDDEN_DIMS,
    )
    operator = FNO(
        in_channels=input_dim,
        out_channels=output_dim,
        modes=_FNO_MODES,
        width=_FNO_WIDTH,
        num_layers=_FNO_NUM_LAYERS,
        dim=fno_dim,
    )
    causal_gate = CausalGate(
        out_dim=output_dim,
        hidden_dim=_GATE_HIDDEN_DIM,
        causal_strength=1.0,
    )
    config = HybridConfig(
        use_operator=True,
        use_pinn=True,
        use_causal_gate=use_gate,
        operator_weight=1.0,
        physics_weight=1.0,
        gate_weight=0.01,
    )
    hybrid = HybridPINNOperator(
        pinn_network=pinn_network,
        operator=operator,
        causal_gate=causal_gate,
        pde_problem=pde,
        config=config,
    )
    return HybridModelWrapper(hybrid), lambda_physics


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------


def run_single(
    pde_name: str,
    policy_name: str,
    seed: int,
    device: str,
    pde: Any,
    input_dim: int,
    benchmark_runner: BenchmarkRunner,
) -> dict[str, Any]:
    """Run a single (pde, policy, seed) experiment.

    Returns a result dict compatible with
    :pymethod:`EarlyTrainingPredictor.build_dataset` -- the important keys
    are ``model_name``, ``seed``, ``metrics``, and ``training_history``.
    """
    set_seed(seed)

    model, lambda_physics = create_model(policy_name, pde, input_dim)

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
        lambda_physics=lambda_physics,
        lambda_data=1.0,
        causal_epsilon=1.0,
    )

    trainer = HybridTrainer(
        model=model,
        pde_problem=pde,
        config=config,
        device=device,
    )

    start = time.time()
    history = trainer.train()
    training_time = time.time() - start

    metrics = benchmark_runner.run_single(model, pde)

    return {
        "pde": pde_name,
        "model_name": policy_name,
        "seed": seed,
        "training_time": training_time,
        "training_history": history,
        "metrics": metrics,
        "success": True,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

_METRICS_TO_AGGREGATE = [
    "relative_l2_error",
    "mse",
    "mae",
    "max_error",
    "pde_residual_norm",
    "bc_violation",
    "ic_violation",
]


def aggregate_results(
    results: list[dict[str, Any]],
    policies: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute per-policy mean/std over seeds for key metrics."""
    aggregated: dict[str, dict[str, Any]] = {}

    for policy in policies:
        runs = [r for r in results if r["model_name"] == policy and r["success"]]
        agg: dict[str, Any] = {}

        if not runs:
            for metric in _METRICS_TO_AGGREGATE:
                agg[metric] = {"mean": float("nan"), "std": float("nan")}
            agg["training_time"] = {"mean": float("nan"), "std": float("nan")}
        else:
            for metric in _METRICS_TO_AGGREGATE:
                values = [r["metrics"].get(metric, float("nan")) for r in runs]
                agg[metric] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
                }
            times = [r["training_time"] for r in runs]
            agg["training_time"] = {
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
            }

        aggregated[policy] = agg

    return aggregated


def print_summary(
    aggregated: dict[str, dict[str, Any]],
    policies: list[str],
    pde_name: str,
) -> None:
    """Print a compact summary table to stdout."""
    print()
    print("=" * 120)
    print(f"Paper A Policy Suite Summary -- {pde_name}")
    print("=" * 120)

    header = (
        f"{'Policy':<30} | {'Rel L2':<15} | {'MSE':<15} | {'PDE Res':<15} | "
        f"{'BC Viol':<15} | {'IC Viol':<15} | {'Time (s)':<12}"
    )
    print(header)
    print("-" * 120)

    for policy in policies:
        data = aggregated.get(policy, {})
        rel_l2 = f"{data.get('relative_l2_error', {}).get('mean', float('nan')):.4e}"
        mse = f"{data.get('mse', {}).get('mean', float('nan')):.4e}"
        pde_res = f"{data.get('pde_residual_norm', {}).get('mean', float('nan')):.4e}"
        bc_viol = f"{data.get('bc_violation', {}).get('mean', float('nan')):.4e}"
        ic_viol = f"{data.get('ic_violation', {}).get('mean', float('nan')):.4e}"
        time_s = f"{data.get('training_time', {}).get('mean', 0.0):.1f}"
        print(
            f"{policy:<30} | {rel_l2:<15} | {mse:<15} | {pde_res:<15} | "
            f"{bc_viol:<15} | {ic_viol:<15} | {time_s:<12}"
        )

    print("=" * 120)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper A: Run policy suite for routing benchmark",
    )
    parser.add_argument(
        "--pde",
        type=str,
        required=True,
        choices=sorted(PDE_REGISTRY.keys()),
        help="PDE configuration to run (see PDE_REGISTRY for available names).",
    )
    parser.add_argument(
        "--policies",
        type=str,
        nargs="+",
        default=DEFAULT_POLICIES,
        help=(
            "Policy names to evaluate. "
            "Built-in: pinn_only, hybrid_no_gate, hybrid_full, pino. "
            "Append _lambda<value> for custom physics loss weight "
            "(e.g. hybrid_full_lambda0.1)."
        ),
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of seeds to use from DEFAULT_SEEDS (default: 10).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seed list (overrides --num-seeds).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default: cuda).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write results.json into (created if absent).",
    )
    args = parser.parse_args()

    # ----- resolve arguments -----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pde_cls, pde_kwargs, input_dim, spatial_dim = PDE_REGISTRY[args.pde]
    pde = pde_cls(**pde_kwargs)
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS[: args.num_seeds]
    policies = args.policies
    device = args.device

    benchmark_runner = BenchmarkRunner(
        config=BenchmarkConfig(
            n_collocation=2000,
            n_boundary=200,
            n_initial=200,
            include_physics_consistency=False,
            device=device,
        )
    )

    # ----- banner -----
    total = len(policies) * len(seeds)
    print("=" * 120)
    print(f"Paper A Policy Suite -- {args.pde}")
    print("=" * 120)
    print(f"PDE class : {pde_cls.__name__}")
    print(f"PDE kwargs: {pde_kwargs}")
    print(f"Policies  : {policies}")
    print(f"Seeds     : {seeds}")
    print(f"Total runs: {total}")
    print(f"Device    : {device}")
    print(f"Output    : {output_dir}")
    print("Training  : Stage1=200ep/1e-3, Stage2=500ep/1e-4, Stage3=200ep/1e-5")
    print()

    # ----- run loop -----
    all_results: list[dict[str, Any]] = []
    completed = 0

    for policy in policies:
        print(f"\n--- Policy: {policy} ---")
        for seed in seeds:
            completed += 1
            print(
                f"  [{completed}/{total}] {args.pde}/{policy}/seed={seed}...",
                end=" ",
                flush=True,
            )
            try:
                result = run_single(
                    pde_name=args.pde,
                    policy_name=policy,
                    seed=seed,
                    device=device,
                    pde=pde,
                    input_dim=input_dim,
                    benchmark_runner=benchmark_runner,
                )
                rel_l2 = result["metrics"].get("relative_l2_error", float("nan"))
                print(f"rel_l2={rel_l2:.4e}  time={result['training_time']:.1f}s")
            except Exception as exc:
                print(f"FAILED: {exc}")
                traceback.print_exc()
                result = {
                    "pde": args.pde,
                    "model_name": policy,
                    "seed": seed,
                    "training_time": 0.0,
                    "training_history": {},
                    "metrics": {},
                    "success": False,
                    "error": str(exc),
                }
            all_results.append(result)

    # ----- aggregate and save -----
    aggregated = aggregate_results(all_results, policies)

    payload = {
        "experiment": f"paper_a_{args.pde}",
        "pde": args.pde,
        "pde_config": pde_kwargs,
        "policies": policies,
        "seeds": seeds,
        "raw_results": all_results,
        "aggregated_results": aggregated,
    }

    results_path = output_dir / "results.json"
    with results_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\nSaved to {results_path}")

    # ----- summary -----
    print_summary(aggregated, policies, args.pde)


if __name__ == "__main__":
    main()
