#!/usr/bin/env python3
"""Pretraining-budget sweep -- isolate Stage-1 pretraining as the dominant factor.

Reviewer 1 question: is the no-pretrain causal failure caused by *absence* of
operator pretraining, or merely *amount* of pretraining?  This sweep walks the
Stage-1 budget through {50, 100, 200, 400} epochs (vs 0 in the
no-pretrain branch from the main ablation) on three indicative PDEs, and
toggles Stage-3 (causal) on/off at each budget.  If causal helps at small
Stage-1 budgets but not large ones, the implicit-curriculum hypothesis is
supported.

Outputs:
  * ``{output_dir}/{pde}_{budget}_results.json``
  * ``{output_dir}/aggregate.json``
  * ``{output_dir}/progress.json`` updated per (PDE, budget).
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
from causal_pinn.physics.pde import AdvectionDiffusion1D, AllenCahn1D, Heat1D
from causal_pinn.physics.pde.base import PDEProblem
from causal_pinn.utils.random import set_seed

STAGE1_LR = 1e-3
STAGE2_EPOCHS = 400
STAGE3_EPOCHS = 200
STAGE2_LR = 1e-4
STAGE3_LR = 1e-5
EPSILON = 1.0
BATCH_SIZE = 256
NUM_COLLOCATION = 2000
NUM_BOUNDARY = 200
HIDDEN_DIMS = [64, 64, 64, 64]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
PRETRAIN_BUDGETS = [50, 100, 200, 400]

PDE_REGISTRY: dict[str, tuple[type, dict[str, Any]]] = {
    "heat1d": (Heat1D, {"alpha": 1.0, "length": 1.0, "t_max": 1.0}),
    "advdiff1d": (
        AdvectionDiffusion1D,
        {"velocity": 1.0, "alpha": 0.1, "length": 1.0, "t_max": 1.0},
    ),
    "allencahn1d": (
        AllenCahn1D,
        {"epsilon": 0.01, "length": 2.0, "t_max": 1.0, "front_center": 1.0},
    ),
}


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


def run_stage1(
    model: nn.Module, pde: PDEProblem, device: str, epochs: int,
) -> list[float]:
    if epochs == 0:
        return []
    optimizer = Adam(model.parameters(), lr=STAGE1_LR)
    physics_loss_fn = PhysicsLoss(pde)
    has_exact = _has_exact_solution(pde, device)
    losses: list[float] = []
    for _epoch in range(epochs):
        optimizer.zero_grad()
        t_colloc, x_colloc = _generate_collocation(pde, NUM_COLLOCATION, device)
        total = torch.tensor(0.0, device=device)
        if has_exact:
            t_data, x_data, u_data = _generate_synthetic(pde, BATCH_SIZE, device)
            total = total + _data_loss(model, t_data, x_data, u_data)
        phys = physics_loss_fn(model, t_colloc, x_colloc)
        bc = _bc_loss(model, pde, device)
        total = total + phys["physics_loss"] + bc
        if torch.isnan(total) or torch.isinf(total):
            losses.append(float("nan"))
            break
        total.backward()
        optimizer.step()
        losses.append(total.item())
    return losses


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


def run_stage3(model: nn.Module, pde: PDEProblem, device: str) -> list[float]:
    optimizer = Adam(model.parameters(), lr=STAGE3_LR)
    causal = CausalLoss(pde, epsilon=EPSILON, num_time_bins=100, weight=1.0)
    has_exact = _has_exact_solution(pde, device)
    losses: list[float] = []
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
    return losses


def run_single(
    pde_name: str,
    pde: PDEProblem,
    pretrain_epochs: int,
    use_causal: bool,
    seed: int,
    device: str,
    runner: BenchmarkRunner,
) -> dict[str, Any]:
    set_seed(seed)
    model = _make_model(device)
    start = time.time()
    s1 = run_stage1(model, pde, device, pretrain_epochs)
    s2 = run_stage2(model, pde, device)
    s3 = run_stage3(model, pde, device) if use_causal else []
    train_time = time.time() - start
    metrics = runner.run_single(model, pde)
    return {
        "pde": pde_name,
        "pretrain_epochs": pretrain_epochs,
        "use_causal": use_causal,
        "condition": ("causal" if use_causal else "no_causal"),
        "seed": seed,
        "training_time": train_time,
        "stage1_losses": s1,
        "stage2_losses": s2,
        "stage3_losses": s3,
        "metrics": metrics,
        "success": True,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cond in ("no_causal", "causal"):
        sub = [r for r in results if r["condition"] == cond and r["success"]]
        rl2 = [r["metrics"].get("relative_l2_error", float("nan")) for r in sub]
        pres = [r["metrics"].get("pde_residual_norm", float("nan")) for r in sub]
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
            "training_time": {
                "mean": float(np.mean(times)) if times else 0.0,
                "std": float(np.std(times)) if times else 0.0,
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-1 pretraining-budget sweep")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/results/paper_combined/pretrain_budget_sweep",
    )
    parser.add_argument(
        "--pdes", type=str, nargs="+",
        default=list(PDE_REGISTRY.keys()),
        choices=list(PDE_REGISTRY.keys()),
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=PRETRAIN_BUDGETS,
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Single seed override (debug only).",
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

    pde_names = args.pdes
    budgets = args.budgets
    conditions = [("no_causal", False), ("causal", True)]
    total_runs = len(pde_names) * len(budgets) * len(conditions) * len(seeds)
    print(
        f"Pretrain-budget sweep: {len(pde_names)} PDEs x {len(budgets)} budgets "
        f"x 2 conds x {len(seeds)} seeds = {total_runs} runs",
    )

    progress: dict[str, Any] = {"completed": [], "total_cells": len(pde_names) * len(budgets)}
    progress_path = output_dir / "progress.json"
    aggregate_summary: dict[str, dict[str, Any]] = {}
    run_count = 0

    for pde_name in pde_names:
        pde_cls, pde_kwargs = PDE_REGISTRY[pde_name]
        pde = pde_cls(**pde_kwargs)
        aggregate_summary.setdefault(pde_name, {})

        for budget in budgets:
            results: list[dict[str, Any]] = []
            for cond_label, use_causal in conditions:
                for seed in seeds:
                    run_count += 1
                    tag = (
                        f"[{run_count}/{total_runs}] "
                        f"{pde_name}/budget={budget}/{cond_label}/seed={seed}"
                    )
                    print(f"  {tag}...", end=" ", flush=True)
                    try:
                        r = run_single(
                            pde_name, pde, budget, use_causal,
                            seed, args.device, runner,
                        )
                        rl2 = r["metrics"].get("relative_l2_error", float("nan"))
                        print(
                            f"rel_l2={rl2:.4e} "
                            f"time={r['training_time']:.1f}s"
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"FAILED: {exc}")
                        r = {
                            "pde": pde_name, "pretrain_epochs": budget,
                            "use_causal": use_causal, "condition": cond_label,
                            "seed": seed, "training_time": 0.0,
                            "stage1_losses": [], "stage2_losses": [], "stage3_losses": [],
                            "metrics": {}, "success": False, "error": str(exc),
                        }
                    results.append(r)

            cell_summary = aggregate(results)
            payload = {
                "experiment": "pretrain_budget_sweep",
                "pde": pde_name,
                "pde_config": pde_kwargs,
                "pretrain_epochs": budget,
                "seeds": seeds,
                "raw_results": results,
                "summary": cell_summary,
            }
            out_path = output_dir / f"{pde_name}_{budget}_results.json"
            out_path.write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8",
            )
            print(f"  Saved {out_path}")

            aggregate_summary[pde_name][f"budget_{budget}"] = cell_summary
            progress["completed"].append({"pde": pde_name, "budget": budget})
            progress_path.write_text(
                json.dumps(progress, indent=2), encoding="utf-8",
            )

    aggregate_path = output_dir / "aggregate.json"
    aggregate_path.write_text(
        json.dumps({
            "experiment": "pretrain_budget_sweep",
            "pdes": pde_names,
            "budgets": budgets,
            "seeds": seeds,
            "by_pde": aggregate_summary,
        }, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nAggregate saved: {aggregate_path}")


if __name__ == "__main__":
    main()
