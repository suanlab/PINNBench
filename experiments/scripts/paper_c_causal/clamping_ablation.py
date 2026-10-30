#!/usr/bin/env python3
"""Clamping ablation -- isolate weight magnitude vanishing from gradient direction conflict.

Mechanism question (Reviewer 1): when causal training fails on the no-pretrain
branch, is the failure driven by causal weights collapsing toward zero
(losing gradient magnitude on later times) or by gradient-direction conflict
between causal-weighted residuals and the data/BC objectives?

Procedure: replicate the no-pretrain causal regime on 9 PDEs x 10 seeds,
running two conditions per (PDE, seed):
  * unclamped baseline (CausalLoss with min_weight=0.0)
  * clamped (CausalLoss with min_weight=0.1)

If the clamped condition closes most of the gap, the failure is dominated by
weight collapse.  If the gap persists, gradient direction conflict is the
dominant factor.

Outputs:
  * ``{output_dir}/{pde}_results.json`` -- per-PDE results across seeds.
  * ``{output_dir}/aggregate.json`` -- summary across all PDEs.
  * ``{output_dir}/progress.json`` -- updated after each PDE.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam

from causal_pinn.evaluation.benchmark_runner import BenchmarkConfig, BenchmarkRunner
from causal_pinn.models.pinn.base import PINNBase
from causal_pinn.physics.losses import CausalLoss, PhysicsLoss
from causal_pinn.physics.pde import (
    AdvectionDiffusion1D,
    AllenCahn1D,
    Burgers1D,
    Heat1D,
    KdV1D,
    ReactionDiffusion1D,
)
from causal_pinn.physics.pde.base import PDEProblem
from causal_pinn.utils.random import set_seed

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
STAGE2_EPOCHS = 400
STAGE3_EPOCHS = 200
STAGE2_LR = 1e-4
STAGE3_LR = 1e-5
EPSILON = 1.0
MIN_WEIGHT_CLAMPED = 0.1
BATCH_SIZE = 256
NUM_COLLOCATION = 2000
NUM_BOUNDARY = 200
HIDDEN_DIMS = [64, 64, 64, 64]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

PDE_REGISTRY: dict[str, tuple[type, dict[str, Any]]] = {
    "heat1d": (Heat1D, {"alpha": 1.0, "length": 1.0, "t_max": 1.0}),
    "advdiff1d": (
        AdvectionDiffusion1D,
        {"velocity": 1.0, "alpha": 0.1, "length": 1.0, "t_max": 1.0},
    ),
    "burgers1d": (Burgers1D, {"viscosity": 0.01, "length": 1.0, "t_max": 1.0}),
    "burgers1d_verylow": (
        Burgers1D,
        {"viscosity": 0.0001, "length": 2.0, "t_max": 0.5},
    ),
    "allencahn1d": (
        AllenCahn1D,
        {"epsilon": 0.01, "length": 2.0, "t_max": 1.0, "front_center": 1.0},
    ),
    "allencahn1d_sharp": (
        AllenCahn1D,
        {"epsilon": 0.001, "length": 2.0, "t_max": 0.5, "front_center": 1.0},
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
    ),
    "reaction_diffusion1d": (
        ReactionDiffusion1D,
        {"diffusion": 0.01, "reaction_rate": 1.0, "length": 1.0, "t_max": 1.0},
    ),
    "reaction_diffusion1d_stiff": (
        ReactionDiffusion1D,
        {"diffusion": 0.01, "reaction_rate": 10.0, "length": 1.0, "t_max": 0.5},
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(device: str) -> PINNBase:
    return PINNBase(input_dim=2, output_dim=1, hidden_dims=HIDDEN_DIMS).to(device)


def _generate_collocation(
    pde: PDEProblem, n: int, device: str,
) -> tuple[Tensor, Tensor]:
    t_bounds, x_bounds = pde.domain_bounds
    t = torch.rand(n, device=device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
    x = torch.rand(n, device=device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    return t.requires_grad_(True), x.requires_grad_(True)


def _has_exact_solution(pde: PDEProblem, device: str) -> bool:
    try:
        t_sample = torch.tensor([0.5], device=device)
        x_sample = torch.tensor([0.5], device=device)
        result = pde.exact_solution(t_sample, x_sample)
        return result is not None
    except (NotImplementedError, AttributeError, TypeError):
        return False


def _generate_synthetic(
    pde: PDEProblem, n: int, device: str,
) -> tuple[Tensor, Tensor, Tensor]:
    t_bounds, x_bounds = pde.domain_bounds
    t = torch.rand(n, device=device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
    x = torch.rand(n, device=device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    u = pde.exact_solution(t, x)
    return t, x, u


def _data_loss(model: nn.Module, t: Tensor, x: Tensor, u_true: Tensor) -> Tensor:
    t_in = t.unsqueeze(-1) if t.dim() == 1 else t
    x_in = x.unsqueeze(-1) if x.dim() == 1 else x
    u_pred = model(torch.cat([t_in, x_in], dim=-1))
    if u_pred.shape != u_true.shape and u_true.dim() == 1:
        u_true = u_true.unsqueeze(-1)
    return torch.mean((u_pred - u_true) ** 2)


def _bc_loss(model: nn.Module, pde: PDEProblem, device: str) -> Tensor:
    t_bounds, x_bounds = pde.domain_bounds
    n_bc = NUM_BOUNDARY // 2

    t_left = torch.rand(n_bc, device=device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
    x_left = torch.full((n_bc,), x_bounds[0], device=device)
    t_right = torch.rand(n_bc, device=device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
    x_right = torch.full((n_bc,), x_bounds[1], device=device)

    t_bc = torch.cat([t_left, t_right])
    x_bc = torch.cat([x_left, x_right])
    u_bc = pde.boundary_condition(t_bc, x_bc)

    u_pred_bc = model(
        torch.cat([t_bc.unsqueeze(-1), x_bc.unsqueeze(-1)], dim=-1)
    ).squeeze(-1)
    bc = torch.mean((u_pred_bc - u_bc) ** 2)

    n_ic = NUM_BOUNDARY
    x_ic = torch.rand(n_ic, device=device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    u_ic = pde.initial_condition(x_ic)
    t_zero = torch.zeros(n_ic, device=device)
    u_pred_ic = model(
        torch.cat([t_zero.unsqueeze(-1), x_ic.unsqueeze(-1)], dim=-1)
    ).squeeze(-1)
    ic = torch.mean((u_pred_ic - u_ic) ** 2)
    return bc + ic


def run_stage2(model: nn.Module, pde: PDEProblem, device: str) -> list[float]:
    optimizer = Adam(model.parameters(), lr=STAGE2_LR)
    physics_loss_fn = PhysicsLoss(pde)
    has_exact = _has_exact_solution(pde, device)
    losses: list[float] = []

    for epoch in range(STAGE2_EPOCHS):
        optimizer.zero_grad()
        t_colloc, x_colloc = _generate_collocation(pde, NUM_COLLOCATION, device)

        total = torch.tensor(0.0, device=device)
        if has_exact:
            t_data, x_data, u_data = _generate_synthetic(pde, BATCH_SIZE, device)
            total = total + _data_loss(model, t_data, x_data, u_data)

        phys = physics_loss_fn(model, t_colloc, x_colloc)
        bc = _bc_loss(model, pde, device)

        progress = min(1.0, epoch / max(1, STAGE2_EPOCHS // 2))
        lam = 0.1 + 0.9 * progress
        total = total + lam * phys["physics_loss"] + bc

        if torch.isnan(total) or torch.isinf(total):
            losses.append(float("nan"))
            break
        total.backward()
        optimizer.step()
        losses.append(total.item())

    return losses


def run_stage3(
    model: nn.Module,
    pde: PDEProblem,
    device: str,
    min_weight: float,
) -> dict[str, Any]:
    optimizer = Adam(model.parameters(), lr=STAGE3_LR)
    causal = CausalLoss(
        pde, epsilon=EPSILON, num_time_bins=100, weight=1.0,
        min_weight=min_weight,
    )
    has_exact = _has_exact_solution(pde, device)
    losses: list[float] = []
    weight_stats: list[dict[str, float]] = []

    for _epoch in range(STAGE3_EPOCHS):
        optimizer.zero_grad()
        t_colloc, x_colloc = _generate_collocation(pde, NUM_COLLOCATION, device)

        total = torch.tensor(0.0, device=device)
        if has_exact:
            t_data, x_data, u_data = _generate_synthetic(pde, BATCH_SIZE, device)
            total = total + _data_loss(model, t_data, x_data, u_data)

        result = causal(model, t_colloc, x_colloc)
        bc = _bc_loss(model, pde, device)

        total = total + result["causal_loss"] + bc
        if torch.isnan(total) or torch.isinf(total):
            losses.append(float("nan"))
            break
        total.backward()
        optimizer.step()
        losses.append(total.item())

        w = result["causal_weights"].detach()
        weight_stats.append({
            "min": float(w.min().item()),
            "max": float(w.max().item()),
            "mean": float(w.mean().item()),
            "std": float(w.std().item()),
        })

    return {"losses": losses, "weight_stats": weight_stats}


def run_single(
    pde_name: str,
    pde: PDEProblem,
    min_weight: float,
    seed: int,
    device: str,
    runner: BenchmarkRunner,
) -> dict[str, Any]:
    set_seed(seed)
    model = _make_model(device)

    start = time.time()
    s2 = run_stage2(model, pde, device)
    s3 = run_stage3(model, pde, device, min_weight)
    train_time = time.time() - start

    metrics = runner.run_single(model, pde)

    return {
        "pde": pde_name,
        "min_weight": min_weight,
        "condition": "clamped" if min_weight > 0.0 else "unclamped",
        "seed": seed,
        "training_time": train_time,
        "stage2_losses": s2,
        "stage3_losses": s3["losses"],
        "stage3_weight_stats": s3["weight_stats"],
        "metrics": metrics,
        "success": True,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cond in ("unclamped", "clamped"):
        sub = [r for r in results if r["condition"] == cond and r["success"]]
        rl2 = [r["metrics"].get("relative_l2_error", float("nan")) for r in sub]
        pres = [r["metrics"].get("pde_residual_norm", float("nan")) for r in sub]
        end_min_w = [
            r["stage3_weight_stats"][-1]["min"]
            for r in sub if r["stage3_weight_stats"]
        ]
        times = [r["training_time"] for r in sub]
        out[cond] = {
            "n": len(sub),
            "relative_l2_error": {
                "mean": float(np.nanmean(rl2)) if rl2 else float("nan"),
                "std": float(np.nanstd(rl2)) if rl2 else float("nan"),
            },
            "pde_residual_norm": {
                "mean": float(np.nanmean(pres)) if pres else float("nan"),
                "std": float(np.nanstd(pres)) if pres else float("nan"),
            },
            "final_min_weight": {
                "mean": float(np.mean(end_min_w)) if end_min_w else float("nan"),
                "std": float(np.std(end_min_w)) if end_min_w else float("nan"),
            },
            "training_time": {
                "mean": float(np.mean(times)) if times else 0.0,
                "std": float(np.std(times)) if times else 0.0,
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal-loss clamping ablation")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/results/paper_combined/clamping_ablation",
    )
    parser.add_argument(
        "--pdes", type=str, nargs="+",
        default=list(PDE_REGISTRY.keys()),
        choices=list(PDE_REGISTRY.keys()),
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Single seed to override default seed list (debug only).",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        msg = "--device cuda requested but torch.cuda.is_available() is False"
        raise SystemExit(msg)
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed] if args.seed is not None else DEFAULT_SEEDS[: args.num_seeds]

    runner = BenchmarkRunner(
        BenchmarkConfig(
            n_collocation=2000, n_boundary=200, n_initial=200,
            include_physics_consistency=False, device=args.device,
        )
    )

    conditions = [("unclamped", 0.0), ("clamped", MIN_WEIGHT_CLAMPED)]
    total_runs = len(args.pdes) * len(conditions) * len(seeds)
    print(
        f"Clamping ablation: {len(args.pdes)} PDEs x "
        f"{len(conditions)} conditions x {len(seeds)} seeds = {total_runs} runs",
    )

    progress: dict[str, Any] = {"completed_pdes": [], "total_pdes": len(args.pdes)}
    progress_path = output_dir / "progress.json"

    aggregate_summary: dict[str, Any] = {}
    run_count = 0

    for pde_name in args.pdes:
        pde_cls, pde_kwargs = PDE_REGISTRY[pde_name]
        pde = pde_cls(**pde_kwargs)
        results: list[dict[str, Any]] = []

        for cond_label, min_w in conditions:
            for seed in seeds:
                run_count += 1
                tag = f"[{run_count}/{total_runs}] {pde_name}/{cond_label}/seed={seed}"
                print(f"  {tag}...", end=" ", flush=True)
                try:
                    r = run_single(pde_name, pde, min_w, seed, args.device, runner)
                    rl2 = r["metrics"].get("relative_l2_error", float("nan"))
                    pres = r["metrics"].get("pde_residual_norm", float("nan"))
                    print(
                        f"rel_l2={rl2:.4e} pde_res={pres:.4e} "
                        f"time={r['training_time']:.1f}s"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"FAILED: {exc}")
                    r = {
                        "pde": pde_name, "min_weight": min_w,
                        "condition": cond_label, "seed": seed,
                        "training_time": 0.0, "stage2_losses": [], "stage3_losses": [],
                        "stage3_weight_stats": [], "metrics": {},
                        "success": False, "error": str(exc),
                    }
                results.append(r)

        pde_summary = aggregate(results)
        payload = {
            "experiment": "clamping_ablation",
            "pde": pde_name,
            "pde_config": pde_kwargs,
            "seeds": seeds,
            "min_weight_clamped": MIN_WEIGHT_CLAMPED,
            "raw_results": results,
            "summary": pde_summary,
        }
        out_path = output_dir / f"{pde_name}_results.json"
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"  Saved {out_path}")

        aggregate_summary[pde_name] = pde_summary
        progress["completed_pdes"].append(pde_name)
        progress["last_completed"] = pde_name
        progress_path.write_text(
            json.dumps(progress, indent=2), encoding="utf-8",
        )

    aggregate_path = output_dir / "aggregate.json"
    aggregate_path.write_text(
        json.dumps({
            "experiment": "clamping_ablation",
            "pdes": args.pdes,
            "seeds": seeds,
            "by_pde": aggregate_summary,
        }, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nAggregate saved: {aggregate_path}")


if __name__ == "__main__":
    main()
