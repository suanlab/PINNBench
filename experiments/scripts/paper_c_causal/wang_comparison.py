#!/usr/bin/env python3
"""Paper C: Wang et al. 2024 comparison — single-stage PINN vs. 3-stage hybrid.

Tests the hypothesis that causal loss helps in Wang et al.'s setting (single-stage
PINN, no operator pretraining) but becomes redundant with our staged pipeline.

3-way comparison per PDE:
  1. single_pinn_no_causal   — Stage 2 only, PINN-only (Wang et al. baseline)
  2. single_pinn_with_causal — Stage 2+3, PINN-only  (Wang et al. + causal)
  3. hybrid_full_pipeline    — Stage 1+2+3, full hybrid (our method)

Reference: Wang, Perdikaris & Teng (2024), "Respecting causality for training
physics-informed neural networks", JCP.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

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
    Heat1D,
    KdV1D,
)
from causal_pinn.training.trainer import HybridTrainer
from causal_pinn.utils.config import TrainingConfig
from causal_pinn.utils.random import set_seed

DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

# PDEs with exact solutions, spanning difficulty spectrum
PDE_REGISTRY: dict[str, tuple[type, dict[str, Any], int]] = {
    "heat1d": (Heat1D, {"alpha": 1.0, "length": 1.0, "t_max": 1.0}, 2),
    "advdiff1d": (
        AdvectionDiffusion1D,
        {"velocity": 1.0, "alpha": 0.1, "length": 1.0, "t_max": 1.0},
        2,
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
        2,
    ),
    "allencahn1d_sharp": (
        AllenCahn1D,
        {"epsilon": 0.001, "length": 2.0, "t_max": 0.5, "front_center": 1.0},
        2,
    ),
}

# (skip_stage1, skip_stage3, use_hybrid_model, label)
CONDITIONS: list[tuple[bool, bool, bool, str]] = [
    (True, True, False, "single_pinn_no_causal"),     # Stage 2 only, PINN
    (True, False, False, "single_pinn_with_causal"),   # Stage 2+3, PINN
    (False, False, True, "hybrid_full_pipeline"),      # Stage 1+2+3, hybrid
]


def create_pinn_model(input_dim: int) -> PINNBase:
    """Create a standalone PINN matching Wang et al.'s single-network setup."""
    return PINNBase(
        input_dim=input_dim,
        output_dim=1,
        hidden_dims=[64, 64, 64, 64],
    )


def create_hybrid_model(pde: Any, input_dim: int) -> HybridModelWrapper:
    """Create the full hybrid model (PINN + FNO + CausalGate)."""
    pinn = PINNBase(
        input_dim=input_dim, output_dim=1, hidden_dims=[64, 64, 64, 64],
    )
    fno = FNO(
        in_channels=input_dim, out_channels=1,
        modes=8, width=32, num_layers=4, dim=1,
    )
    gate = CausalGate(out_dim=1, hidden_dim=32, causal_strength=1.0)
    config = HybridConfig(
        use_operator=True, use_pinn=True, use_causal_gate=True,
    )
    hybrid = HybridPINNOperator(
        pinn_network=pinn, operator=fno,
        causal_gate=gate, pde_problem=pde, config=config,
    )
    return HybridModelWrapper(hybrid)


def run_single(
    pde_name: str,
    pde: Any,
    input_dim: int,
    condition_label: str,
    skip_stage1: bool,
    skip_stage3: bool,
    use_hybrid: bool,
    seed: int,
    device: str,
    benchmark_runner: BenchmarkRunner,
) -> dict[str, Any]:
    """Run one (condition, seed) trial."""
    set_seed(seed)

    model = create_hybrid_model(pde, input_dim) if use_hybrid else create_pinn_model(input_dim)

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
        skip_stage1=skip_stage1,
        skip_stage3=skip_stage3,
    )

    trainer = HybridTrainer(
        model=model, pde_problem=pde, config=config, device=device,
    )

    start = time.time()
    history = trainer.train()
    training_time = time.time() - start

    metrics = benchmark_runner.run_single(model, pde)

    return {
        "pde": pde_name,
        "condition": condition_label,
        "skip_stage1": skip_stage1,
        "skip_stage3": skip_stage3,
        "use_hybrid": use_hybrid,
        "seed": seed,
        "training_time": training_time,
        "training_history": history,
        "metrics": metrics,
        "success": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper C: Wang et al. 2024 comparison (single PINN vs hybrid)",
    )
    parser.add_argument(
        "--pde", type=str, required=True, choices=list(PDE_REGISTRY.keys()),
        help="PDE to test",
    )
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/results/paper_c/wang_comparison",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pde_cls, pde_kwargs, input_dim = PDE_REGISTRY[args.pde]
    pde = pde_cls(**pde_kwargs)
    seeds = DEFAULT_SEEDS[: args.num_seeds]

    runner = BenchmarkRunner(
        BenchmarkConfig(
            n_collocation=2000,
            n_boundary=200,
            n_initial=200,
            include_physics_consistency=False,
            device=args.device,
        )
    )

    print(f"Wang et al. 2024 Comparison: {args.pde}")
    print(f"Conditions: {[c[3] for c in CONDITIONS]}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {len(CONDITIONS) * len(seeds)}")
    print()

    all_results: list[dict[str, Any]] = []
    for skip_s1, skip_s3, use_hybrid, label in CONDITIONS:
        model_type = "hybrid" if use_hybrid else "PINN-only"
        stages = []
        if not skip_s1:
            stages.append("1")
        stages.append("2")
        if not skip_s3:
            stages.append("3")
        print(
            f"--- {label} ({model_type}, stages={'+'.join(stages)}) ---"
        )
        for seed in seeds:
            tag = f"  {args.pde}/{label}/seed={seed}"
            print(f"{tag}...", end=" ", flush=True)
            try:
                result = run_single(
                    args.pde, pde, input_dim, label,
                    skip_s1, skip_s3, use_hybrid,
                    seed, args.device, runner,
                )
                rl2 = result["metrics"].get("relative_l2_error", float("nan"))
                pde_res = result["metrics"].get(
                    "pde_residual_norm", float("nan"),
                )
                print(
                    f"rel_l2={rl2:.4e} pde_res={pde_res:.4e} "
                    f"time={result['training_time']:.1f}s"
                )
            except Exception as exc:
                print(f"FAILED: {exc}")
                result = {
                    "pde": args.pde,
                    "condition": label,
                    "skip_stage1": skip_s1,
                    "skip_stage3": skip_s3,
                    "use_hybrid": use_hybrid,
                    "seed": seed,
                    "training_time": 0.0,
                    "training_history": {},
                    "metrics": {},
                    "success": False,
                    "error": str(exc),
                }
            all_results.append(result)

    # ---- Aggregate summary ----
    summary: dict[str, dict[str, Any]] = {}
    metrics_list = [
        "relative_l2_error", "pde_residual_norm",
        "bc_violation", "ic_violation",
    ]
    for _s1, _s3, _hyb, label in CONDITIONS:
        cond_results = [
            r for r in all_results if r["condition"] == label and r["success"]
        ]
        summary[label] = {}
        for m in metrics_list:
            vals = [r["metrics"].get(m, float("nan")) for r in cond_results]
            summary[label][m] = {
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
                "values": [float(v) for v in vals],
            }
        times = [r["training_time"] for r in cond_results]
        summary[label]["training_time"] = {
            "mean": float(np.mean(times)) if times else 0.0,
            "std": float(np.std(times)) if times else 0.0,
        }

    # Causal benefit in Wang setting (single PINN)
    wang_base = summary.get("single_pinn_no_causal", {})
    wang_causal = summary.get("single_pinn_with_causal", {})
    hybrid_full = summary.get("hybrid_full_pipeline", {})

    causal_benefit: dict[str, Any] = {}
    base_mean = wang_base.get("relative_l2_error", {}).get("mean", float("nan"))
    causal_mean = wang_causal.get("relative_l2_error", {}).get("mean", float("nan"))
    hybrid_mean = hybrid_full.get("relative_l2_error", {}).get("mean", float("nan"))

    if base_mean > 0 and not np.isnan(base_mean):
        causal_benefit["wang_causal_improvement_pct"] = (
            (base_mean - causal_mean) / base_mean * 100
        )
    if base_mean > 0 and not np.isnan(base_mean):
        causal_benefit["hybrid_improvement_over_wang_pct"] = (
            (base_mean - hybrid_mean) / base_mean * 100
        )

    payload = {
        "experiment": f"paper_c_wang_comparison_{args.pde}",
        "pde": args.pde,
        "pde_config": pde_kwargs,
        "conditions": [c[3] for c in CONDITIONS],
        "seeds": seeds,
        "hypothesis": (
            "Causal loss helps in Wang et al. single-stage PINN setting "
            "(no operator pretraining) but is redundant in our full 3-stage "
            "hybrid pipeline where stage 1 provides implicit temporal curriculum."
        ),
        "raw_results": all_results,
        "summary": summary,
        "causal_benefit_analysis": causal_benefit,
    }

    results_path = output_dir / f"{args.pde}_results.json"
    results_path.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8",
    )
    print(f"\nSaved: {results_path}")

    # ---- Print summary table ----
    print(f"\n{'Condition':<30s} {'Rel L2':>20s} {'PDE Res':>20s} {'Time':>8s}")
    print("-" * 82)
    for label, data in summary.items():
        rl2 = data.get("relative_l2_error", {})
        pres = data.get("pde_residual_norm", {})
        t = data.get("training_time", {})
        print(
            f"{label:<30s} "
            f"{rl2.get('mean', 0):.4e}+/-{rl2.get('std', 0):.4e} "
            f"{pres.get('mean', 0):.4e}+/-{pres.get('std', 0):.4e} "
            f"{t.get('mean', 0):>6.1f}s"
        )

    if causal_benefit:
        print("\n--- Causal Benefit Analysis ---")
        for k, v in causal_benefit.items():
            print(f"  {k}: {v:+.1f}%")


if __name__ == "__main__":
    main()
