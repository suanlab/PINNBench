#!/usr/bin/env python3
"""Paper C: Stage ablation — 2x2 design (skip_stage1 x skip_stage3).

Tests whether causal loss (stage 3) helps when operator pretraining
(stage 1) is absent.  This is the central mechanistic experiment for
Paper C: if stage 1 provides implicit temporal curriculum, causal loss
should only help when stage 1 is skipped.
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
    Burgers1D,
    Heat1D,
    KdV1D,
    NavierStokes2D,
    ReactionDiffusion1D,
)
from causal_pinn.training.trainer import HybridTrainer
from causal_pinn.utils.config import TrainingConfig
from causal_pinn.utils.random import set_seed

DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

PDE_REGISTRY: dict[str, tuple[type, dict[str, Any], int]] = {
    "heat1d": (Heat1D, {"alpha": 1.0, "length": 1.0, "t_max": 1.0}, 2),
    "burgers1d": (Burgers1D, {"viscosity": 0.01, "length": 1.0, "t_max": 1.0}, 2),
    "burgers1d_lownu": (
        Burgers1D,
        {"viscosity": 0.001, "length": 1.0, "t_max": 1.0},
        2,
    ),
    "burgers1d_verylow": (
        Burgers1D,
        {"viscosity": 0.0001, "length": 2.0, "t_max": 0.5},
        2,
    ),
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
    "allencahn1d": (
        AllenCahn1D,
        {"epsilon": 0.01, "length": 2.0, "t_max": 1.0, "front_center": 1.0},
        2,
    ),
    "allencahn1d_sharp": (
        AllenCahn1D,
        {"epsilon": 0.001, "length": 2.0, "t_max": 0.5, "front_center": 1.0},
        2,
    ),
    "reaction_diffusion1d": (
        ReactionDiffusion1D,
        {"diffusion": 0.01, "reaction_rate": 1.0, "length": 1.0, "t_max": 1.0},
        2,
    ),
    "reaction_diffusion1d_stiff": (
        ReactionDiffusion1D,
        {"diffusion": 0.01, "reaction_rate": 10.0, "length": 1.0, "t_max": 0.5},
        2,
    ),
    "navier_stokes2d": (
        NavierStokes2D,
        {"viscosity": 0.1, "t_max": 1.0},
        3,
    ),
}

# 2x2 conditions: (skip_stage1, skip_stage3, label)
CONDITIONS = [
    (False, True, "full_pipeline_no_causal"),   # Stage 1 + 2 only
    (False, False, "full_pipeline_with_causal"), # Stage 1 + 2 + 3
    (True, True, "no_pretrain_no_causal"),       # Stage 2 only
    (True, False, "no_pretrain_with_causal"),    # Stage 2 + 3 only
]

# Controls for the confounds of the 2x2 design (``--condition-set controls``).
# Each entry: (skip_stage1, skip_stage3, label, TrainingConfig overrides).
#  * uniform_stage3: Stage 3 with epsilon fixed at 0, so every time bin has
#    weight 1 -- same 200 epochs, LR and loss terms as the causal arm but no
#    temporal reweighting.  Separates reweighting from extra training.
#  * long_stage2: no Stage 1, Stage 2 lengthened by 200 epochs so the total
#    epoch count matches the pretrained arms.  Separates pretraining from
#    training length.
CONTROL_CONDITIONS = [
    (False, False, "full_pipeline_uniform_stage3",
     {"causal_epsilon": 0.0, "causal_epsilon_init": 0.0}),
    (True, False, "no_pretrain_uniform_stage3",
     {"causal_epsilon": 0.0, "causal_epsilon_init": 0.0}),
    (True, True, "no_pretrain_long_stage2_no_causal", {"stage2_epochs": 700}),
    (True, False, "no_pretrain_long_stage2_with_causal", {"stage2_epochs": 700}),
]


def create_hybrid_model(
    pde: Any,
    input_dim: int,
) -> HybridModelWrapper:
    pinn = PINNBase(
        input_dim=input_dim, output_dim=1, hidden_dims=[64, 64, 64, 64],
    )
    fno_dim = 2 if input_dim == 3 else 1
    fno = FNO(
        in_channels=input_dim, out_channels=1,
        modes=8, width=32, num_layers=4, dim=fno_dim,
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
    seed: int,
    device: str,
    benchmark_runner: BenchmarkRunner,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    model = create_hybrid_model(pde, input_dim)

    settings: dict[str, Any] = dict(
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
    settings.update(overrides or {})
    config = TrainingConfig(**settings)

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
        "overrides": overrides or {},
        "seed": seed,
        "training_time": training_time,
        "training_history": history,
        "metrics": metrics,
        "success": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper C: 2x2 stage ablation (skip_stage1 x skip_stage3)",
    )
    parser.add_argument(
        "--pde", type=str, required=True, choices=list(PDE_REGISTRY.keys()),
    )
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--condition-set", choices=("original", "controls"), default="original",
        help="original: the 2x2 design; controls: epoch-matched control arms",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        help="Run only these condition labels (to split work across GPUs)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Explicit seeds (overrides --num-seeds)")
    args = parser.parse_args()
    conditions = (
        [(a, b, c, {}) for a, b, c in CONDITIONS]
        if args.condition_set == "original" else list(CONTROL_CONDITIONS)
    )
    if args.conditions:
        conditions = [c for c in conditions if c[2] in args.conditions]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pde_cls, pde_kwargs, input_dim = PDE_REGISTRY[args.pde]
    pde = pde_cls(**pde_kwargs)
    seeds = args.seeds or DEFAULT_SEEDS[: args.num_seeds]

    runner = BenchmarkRunner(
        BenchmarkConfig(
            n_collocation=2000,
            n_boundary=200,
            n_initial=200,
            include_physics_consistency=False,
            device=args.device,
        )
    )

    print(f"Paper C Stage Ablation: {args.pde}")
    print(f"Conditions: {[c[2] for c in conditions]}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {len(conditions) * len(seeds)}")
    print()

    all_results: list[dict[str, Any]] = []
    for skip_s1, skip_s3, label, overrides in conditions:
        print(f"--- {label} (skip_s1={skip_s1}, skip_s3={skip_s3}, {overrides}) ---")
        for seed in seeds:
            tag = f"  {args.pde}/{label}/seed={seed}"
            print(f"{tag}...", end=" ", flush=True)
            try:
                result = run_single(
                    args.pde, pde, input_dim, label,
                    skip_s1, skip_s3, seed, args.device, runner, overrides,
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
                    "overrides": overrides,
                    "seed": seed,
                    "training_time": 0.0,
                    "training_history": {},
                    "metrics": {},
                    "success": False,
                    "error": str(exc),
                }
            all_results.append(result)

    # Aggregate
    summary: dict[str, dict[str, Any]] = {}
    metrics_list = [
        "relative_l2_error", "pde_residual_norm",
        "bc_violation", "ic_violation",
    ]
    for _skip_s1, _skip_s3, label, _ov in conditions:
        cond_results = [
            r for r in all_results
            if r["condition"] == label and r["success"]
        ]
        summary[label] = {}
        for m in metrics_list:
            vals = [r["metrics"].get(m, float("nan")) for r in cond_results]
            summary[label][m] = {
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals)),
            }
        times = [r["training_time"] for r in cond_results]
        summary[label]["training_time"] = {
            "mean": float(np.mean(times)) if times else 0.0,
            "std": float(np.std(times)) if times else 0.0,
        }

    payload = {
        "experiment": f"paper_c_stage_ablation_{args.pde}",
        "pde": args.pde,
        "pde_config": pde_kwargs,
        "conditions": [c[2] for c in conditions],
        "condition_overrides": {c[2]: c[3] for c in conditions},
        "seeds": seeds,
        "raw_results": all_results,
        "summary": summary,
    }

    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8",
    )
    print(f"\nSaved: {results_path}")

    # Print summary table
    print(f"\n{'Condition':<35s} {'Rel L2':>12s} {'PDE Res':>12s} {'Time':>8s}")
    print("-" * 70)
    for label, data in summary.items():
        rl2 = data.get("relative_l2_error", {})
        pres = data.get("pde_residual_norm", {})
        t = data.get("training_time", {})
        print(
            f"{label:<35s} "
            f"{rl2.get('mean', 0):.4e}±{rl2.get('std', 0):.4e} "
            f"{pres.get('mean', 0):.4e}±{pres.get('std', 0):.4e} "
            f"{t.get('mean', 0):>6.1f}s"
        )


if __name__ == "__main__":
    main()
