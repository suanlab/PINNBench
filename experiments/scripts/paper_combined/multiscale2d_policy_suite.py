#!/usr/bin/env python3
"""Does any policy in the library recover the fine scale?

Follow-up to ``multiscale2d_probe.py`` (Track B1).  That probe ran only the
two-branch pretraining contrast and found that neither branch recovers the fine
mode of ``MultiscaleHeat2D`` -- the 2.68x advantage is earned entirely on the
coarse envelope.  That leaves the routing question open: the benchmark's premise
is that no single policy dominates, so a policy other than the two probed might
resolve the fine structure, in which case the multi-scale failure is something a
selection layer could route around rather than a property of the whole family.

This runs the full policy library on the same PDE and answers that directly.
Model construction and the training configuration are imported from
``run_policy_suite.py`` rather than duplicated, so the policies here are the
same objects the routing benchmark uses; the results schema matches too, which
keeps the artifact consumable by ``EarlyTrainingPredictor.build_dataset``.

Added on top of that schema: the per-mode projection.  Both eigenmodes are known
analytically, so each trained model's field is projected onto each and scored
separately, per policy.

Outputs ``{output_dir}/results.json`` and ``aggregate.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_a_routing"))

from run_policy_suite import (  # noqa: E402
    DEFAULT_SEEDS,
    PDE_REGISTRY,
    aggregate_results,
    create_model,
)

from causal_pinn.evaluation.benchmark_runner import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkRunner,
)
from causal_pinn.training.trainer import HybridTrainer  # noqa: E402
from causal_pinn.utils.config import TrainingConfig  # noqa: E402
from causal_pinn.utils.random import set_seed  # noqa: E402

PDE_NAME = "multiscale_heat2d"
DEFAULT_POLICIES = [
    "pinn_only",
    "hybrid_no_gate",
    "hybrid_full",
    "hybrid_full_lambda10.0",
    "pino",
]
PROJECTION_TIMES = [0.0, 0.1, 0.5, 1.0]
PROJECTION_GRID = 64


def _training_config(seed: int, device: str, lambda_physics: float) -> TrainingConfig:
    """Mirror of the configuration in ``run_policy_suite.run_single``.

    Kept identical on purpose: these runs are meant to be comparable to the rest
    of the routing benchmark, so if that schedule changes this must follow.
    """
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


@torch.no_grad()
def _scale_decomposition(model: torch.nn.Module, pde: Any, device: str) -> dict[str, Any]:
    """Project the learned field onto each eigenmode and score them separately.

    Uses the uniform ``model(concat([t, x]))`` calling convention that
    ``BenchmarkRunner`` relies on, so it works for the plain PINN and for the
    hybrid wrappers alike.
    """
    lin = torch.linspace(0.0, 1.0, PROJECTION_GRID, device=device)
    gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    x = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    coarse, fine = pde._modes(x)
    norm = float((coarse**2).mean())

    out: dict[str, Any] = {"times": PROJECTION_TIMES, "coarse": [], "fine": []}
    for tv in PROJECTION_TIMES:
        t = torch.full((x.shape[0], 1), tv, device=device)
        u: Tensor = model(torch.cat([t, x], dim=-1)).squeeze(-1)
        for mode_name, mode, exact in (
            ("coarse", coarse, float(np.exp(-pde.coarse_decay_rate() * tv))),
            (
                "fine",
                fine,
                pde.fine_amplitude * float(np.exp(-pde.fine_decay_rate() * tv)),
            ),
        ):
            predicted = float((u * mode).mean() / norm)
            out[mode_name].append(
                {
                    "t": tv,
                    "predicted": predicted,
                    "exact": exact,
                    "abs_error": abs(predicted - exact),
                    "rel_error": abs(predicted - exact) / max(abs(exact), 1e-12),
                }
            )
    return out


def run_single(policy: str, seed: int, device: str, runner: BenchmarkRunner) -> dict[str, Any]:
    pde_cls, pde_kwargs, input_dim, _spatial_dim = PDE_REGISTRY[PDE_NAME]
    pde = pde_cls(**pde_kwargs)
    set_seed(seed)

    model, lambda_physics = create_model(policy, pde, input_dim)
    trainer = HybridTrainer(
        model=model,
        pde_problem=pde,
        config=_training_config(seed, device, lambda_physics),
        device=device,
    )
    start = time.time()
    history = trainer.train()
    training_time = time.time() - start
    metrics = runner.run_single(model, pde)

    return {
        "pde": PDE_NAME,
        "model_name": policy,
        "seed": seed,
        "training_time": training_time,
        "training_history": history,
        "metrics": metrics,
        "scale_decomposition": _scale_decomposition(model, pde, device),
        "success": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--policies", nargs="+", default=DEFAULT_POLICIES)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--disable-cudnn",
        action="store_true",
        help=(
            "Run convolutions through PyTorch's non-cuDNN kernels.  Needed on "
            "hosts where cuDNN fails to initialise (a bare Conv2d raises "
            "CUDNN_STATUS_NOT_INITIALIZED with the GPU otherwise idle and "
            "~12 GB free), which takes out every FNO-bearing policy while "
            "leaving the MLP-only pinn_only policy working.  Results are "
            "numerically equivalent; the fallback kernels are slower."
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/paper_combined/multiscale2d_policy_suite",
    )
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("cuDNN disabled: convolutions use PyTorch's fallback kernels")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner(
        BenchmarkConfig(
            n_collocation=2000,
            n_boundary=240,
            n_initial=200,
            include_physics_consistency=False,
            device=args.device,
        )
    )
    seeds = DEFAULT_SEEDS[: args.num_seeds]
    rows: list[dict[str, Any]] = []
    total = len(args.policies) * len(seeds)
    k = 0
    for policy in args.policies:
        for seed in seeds:
            k += 1
            print(f"  [{k}/{total}] {PDE_NAME}/{policy}/seed={seed}...", end=" ", flush=True)
            try:
                r = run_single(policy, seed, args.device, runner)
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED ({exc})")
                rows.append(
                    {
                        "pde": PDE_NAME,
                        "model_name": policy,
                        "seed": seed,
                        "success": False,
                        "error_message": str(exc),
                        "metrics": {},
                        "training_history": {},
                        "training_time": 0.0,
                    }
                )
                continue
            print(
                f"rel_l2={r['metrics']['relative_l2_error']:.4e} "
                f"fine_rel_err(t=0)={r['scale_decomposition']['fine'][0]['rel_error']:.3f} "
                f"time={r['training_time']:.1f}s"
            )
            rows.append(r)

    aggregated = aggregate_results(rows, args.policies)
    for policy in args.policies:
        runs = [r for r in rows if r["model_name"] == policy and r.get("success")]
        if not runs:
            continue
        for mode in ("coarse", "fine"):
            vals = [r["scale_decomposition"][mode][0]["predicted"] for r in runs]
            errs = [r["scale_decomposition"][mode][0]["rel_error"] for r in runs]
            aggregated[policy][f"{mode}_mode_coefficient_t0"] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
            aggregated[policy][f"{mode}_mode_rel_error_t0"] = {
                "mean": float(np.mean(errs)),
                "std": float(np.std(errs)),
            }

    ok = [p for p in args.policies if aggregated[p].get("coarse_mode_coefficient_t0")]
    print("\n" + "=" * 88)
    print(f"  {'policy':<24} {'rel L2':<18} {'coarse coef':<16} {'fine coef':<16}")
    for p in ok:
        a = aggregated[p]
        print(
            f"  {p:<24} "
            f"{a['relative_l2_error']['mean']:.4f} +- {a['relative_l2_error']['std']:.4f}   "
            f"{a['coarse_mode_coefficient_t0']['mean']:.4f}          "
            f"{a['fine_mode_coefficient_t0']['mean']:.5f}"
        )
    if ok:
        best = min(ok, key=lambda p: aggregated[p]["relative_l2_error"]["mean"])
        best_fine = min(ok, key=lambda p: aggregated[p]["fine_mode_rel_error_t0"]["mean"])
        print(f"\n  oracle-best policy by relative L2 : {best}")
        print(f"  best policy on the fine mode      : {best_fine} "
              f"(rel err {aggregated[best_fine]['fine_mode_rel_error_t0']['mean']:.3f})")

    pde_cls, pde_kwargs, _input_dim, _sd = PDE_REGISTRY[PDE_NAME]
    payload = {
        "experiment": "multiscale2d_policy_suite",
        "pde": PDE_NAME,
        "pde_kwargs": pde_kwargs,
        "spatial_dim": 2,
        "policies": args.policies,
        "cudnn_disabled": bool(args.disable_cudnn),
        "seeds": seeds,
        "aggregated": aggregated,
        "exact_coarse_coefficient_t0": 1.0,
        "exact_fine_coefficient_t0": pde_kwargs["fine_amplitude"],
    }
    (out / "aggregate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps({**payload, "results": rows}, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nWrote {out}/results.json")


if __name__ == "__main__":
    main()
