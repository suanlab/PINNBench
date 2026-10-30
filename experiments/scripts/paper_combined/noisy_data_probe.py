#!/usr/bin/env python3
"""Noisy-data probe -- Stage-1 reference data corrupted with relative Gaussian noise.

Reviewer 5 question: results are dominated by noiseless synthetic regimes.
This probe perturbs the Stage-1 reference signal with relative Gaussian noise
sigma in {0.0, 0.01, 0.05, 0.1, 0.2} and contrasts:
  * with_pretrain  -- the full hybrid pipeline (Stages 1+2+3)
  * pinn_only      -- skip Stage 1 (no operator pretraining)

If pretraining still helps under noise, the implicit-curriculum value is
robust; if pretraining hurts as sigma grows, results are noiseless-regime
specific.

Outputs:
  * ``{output_dir}/results.json``  -- raw rows for all (sigma, condition, seed)
  * ``{output_dir}/aggregate.json``  -- summary
  * ``{output_dir}/progress.json`` -- updated per (sigma, condition)
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
from causal_pinn.physics.pde import Heat1D
from causal_pinn.physics.pde.base import PDEProblem
from causal_pinn.utils.random import set_seed

STAGE1_EPOCHS = 200
STAGE2_EPOCHS = 400
STAGE3_EPOCHS = 200
STAGE1_LR = 1e-3
STAGE2_LR = 1e-4
STAGE3_LR = 1e-5
EPSILON = 1.0
BATCH_SIZE = 1024  # increased so data loss is comparable to physics in magnitude
NUM_COLLOCATION = 2000
NUM_BOUNDARY = 200
LAMBDA_DATA = 10.0  # data-loss weight; ensures noise actually moves the optimum
HIDDEN_DIMS = [64, 64, 64, 64]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.1, 0.2]


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


def _generate_synthetic_noisy(
    pde: PDEProblem, n: int, device: str, noise_sigma: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate synthetic data with relative Gaussian noise added to ``u``."""
    t_bounds, x_bounds = pde.domain_bounds
    t = torch.rand(n, device=device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
    x = torch.rand(n, device=device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    u_clean = pde.exact_solution(t, x)
    if noise_sigma > 0.0:
        scale = u_clean.detach().abs().mean().clamp(min=1e-6)
        noise = torch.randn_like(u_clean) * noise_sigma * scale
        u = u_clean + noise
    else:
        u = u_clean
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


def _sample_fixed_data(
    fixed: tuple[Tensor, Tensor, Tensor], n: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample ``n`` rows from a pre-generated fixed noisy dataset.

    If the dataset is exactly ``n`` rows, return it whole; otherwise
    draw a random uniform index without replacement (or with, if n>N).
    """
    t_full, x_full, u_full = fixed
    n_full = t_full.shape[0]
    if n >= n_full:
        return t_full, x_full, u_full
    idx = torch.randint(0, n_full, (n,), device=t_full.device)
    return t_full[idx], x_full[idx], u_full[idx]


def run_stage1(
    model: nn.Module, pde: PDEProblem, device: str, noise_sigma: float,
    fixed_data: tuple[Tensor, Tensor, Tensor] | None = None,
) -> list[float]:
    optimizer = Adam(model.parameters(), lr=STAGE1_LR)
    physics_loss_fn = PhysicsLoss(pde)
    has_exact = _has_exact_solution(pde, device)
    losses: list[float] = []
    for _epoch in range(STAGE1_EPOCHS):
        optimizer.zero_grad()
        t_colloc, x_colloc = _generate_collocation(pde, NUM_COLLOCATION, device)
        total = torch.tensor(0.0, device=device)
        if has_exact:
            if fixed_data is not None:
                t_data, x_data, u_data = _sample_fixed_data(fixed_data, BATCH_SIZE)
            else:
                t_data, x_data, u_data = _generate_synthetic_noisy(
                    pde, BATCH_SIZE, device, noise_sigma,
                )
            total = total + LAMBDA_DATA * _data_loss(model, t_data, x_data, u_data)
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


def run_stage2(
    model: nn.Module, pde: PDEProblem, device: str, noise_sigma: float,
    fixed_data: tuple[Tensor, Tensor, Tensor] | None = None,
) -> list[float]:
    optimizer = Adam(model.parameters(), lr=STAGE2_LR)
    physics_loss_fn = PhysicsLoss(pde)
    has_exact = _has_exact_solution(pde, device)
    losses: list[float] = []
    for epoch in range(STAGE2_EPOCHS):
        optimizer.zero_grad()
        t_colloc, x_colloc = _generate_collocation(pde, NUM_COLLOCATION, device)
        total = torch.tensor(0.0, device=device)
        if has_exact:
            if fixed_data is not None:
                t_data, x_data, u_data = _sample_fixed_data(fixed_data, BATCH_SIZE)
            else:
                t_data, x_data, u_data = _generate_synthetic_noisy(
                    pde, BATCH_SIZE, device, noise_sigma,
                )
            total = total + LAMBDA_DATA * _data_loss(model, t_data, x_data, u_data)
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
    losses: list[float] = []
    for _epoch in range(STAGE3_EPOCHS):
        optimizer.zero_grad()
        t_colloc, x_colloc = _generate_collocation(pde, NUM_COLLOCATION, device)
        result = causal(model, t_colloc, x_colloc)
        bc = _bc_loss(model, pde, device)
        total = result["causal_loss"] + bc
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
    noise_sigma: float,
    condition: str,
    seed: int,
    device: str,
    runner: BenchmarkRunner,
    fixed_dataset_size: int = BATCH_SIZE * 4,
) -> dict[str, Any]:
    set_seed(seed)
    model = _make_model(device)
    # Pre-generate ONE fixed noisy dataset reused across all stages and
    # epochs of this seed; this turns the probe from "noise resampled
    # per epoch with zero mean" (whose optimum collapses to the clean
    # signal) into a fixed-noisy-dataset experiment where sigma changes
    # the optimum.
    fixed_data: tuple[Tensor, Tensor, Tensor] | None = None
    if _has_exact_solution(pde, device):
        fixed_data = _generate_synthetic_noisy(
            pde, fixed_dataset_size, device, noise_sigma,
        )
    start = time.time()
    s1: list[float] = []
    if condition == "with_pretrain":
        s1 = run_stage1(model, pde, device, noise_sigma, fixed_data=fixed_data)
    s2 = run_stage2(model, pde, device, noise_sigma, fixed_data=fixed_data)
    s3 = run_stage3(model, pde, device)
    train_time = time.time() - start
    metrics = runner.run_single(model, pde)
    return {
        "pde": pde_name,
        "noise_sigma": noise_sigma,
        "condition": condition,
        "seed": seed,
        "training_time": train_time,
        "stage1_losses": s1,
        "stage2_losses": s2,
        "stage3_losses": s3,
        "metrics": metrics,
        "fixed_dataset_size": fixed_dataset_size,
        "success": True,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = sorted({(r["noise_sigma"], r["condition"]) for r in results})
    for noise_sigma, cond in keys:
        sub = [
            r for r in results
            if r["noise_sigma"] == noise_sigma
            and r["condition"] == cond
            and r["success"]
        ]
        rl2 = [r["metrics"].get("relative_l2_error", float("nan")) for r in sub]
        pres = [r["metrics"].get("pde_residual_norm", float("nan")) for r in sub]
        times = [r["training_time"] for r in sub]
        key = f"sigma={noise_sigma}/{cond}"
        out[key] = {
            "n": len(sub),
            "noise_sigma": noise_sigma,
            "condition": cond,
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
    parser = argparse.ArgumentParser(description="Noisy-data probe (Heat1D, sigma sweep)")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/results/paper_combined/noisy_data_probe",
    )
    parser.add_argument(
        "--noise-levels", type=float, nargs="+",
        default=NOISE_LEVELS,
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

    pde = Heat1D(alpha=1.0, length=1.0, t_max=1.0)
    runner = BenchmarkRunner(
        BenchmarkConfig(
            n_collocation=2000, n_boundary=200, n_initial=200,
            include_physics_consistency=False, device=args.device,
        )
    )

    conditions = ["with_pretrain", "pinn_only"]
    total_runs = len(args.noise_levels) * len(conditions) * len(seeds)
    print(
        f"Noisy probe: 1 PDE x {len(args.noise_levels)} sigmas x "
        f"{len(conditions)} conds x {len(seeds)} seeds = {total_runs} runs",
    )

    n_cells = len(args.noise_levels) * len(conditions)
    progress: dict[str, Any] = {"completed_cells": [], "total_cells": n_cells}
    progress_path = output_dir / "progress.json"
    all_results: list[dict[str, Any]] = []
    run_count = 0

    for sigma in args.noise_levels:
        for cond in conditions:
            for seed in seeds:
                run_count += 1
                tag = f"[{run_count}/{total_runs}] heat1d/sigma={sigma}/{cond}/seed={seed}"
                print(f"  {tag}...", end=" ", flush=True)
                try:
                    r = run_single("heat1d", pde, sigma, cond, seed, args.device, runner)
                    rl2 = r["metrics"].get("relative_l2_error", float("nan"))
                    print(
                        f"rel_l2={rl2:.4e} time={r['training_time']:.1f}s"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"FAILED: {exc}")
                    r = {
                        "pde": "heat1d", "noise_sigma": sigma, "condition": cond,
                        "seed": seed, "training_time": 0.0,
                        "stage1_losses": [], "stage2_losses": [], "stage3_losses": [],
                        "metrics": {}, "success": False, "error": str(exc),
                    }
                all_results.append(r)

            progress["completed_cells"].append({"sigma": sigma, "condition": cond})
            progress_path.write_text(
                json.dumps(progress, indent=2), encoding="utf-8",
            )

    summary = aggregate(all_results)
    payload = {
        "experiment": "noisy_data_probe",
        "pde": "heat1d",
        "noise_levels": args.noise_levels,
        "conditions": conditions,
        "seeds": seeds,
        "raw_results": all_results,
        "summary": summary,
    }
    out_path = output_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    aggregate_path = output_dir / "aggregate.json"
    aggregate_path.write_text(
        json.dumps({
            "experiment": "noisy_data_probe",
            "pde": "heat1d",
            "noise_levels": args.noise_levels,
            "conditions": conditions,
            "seeds": seeds,
            "summary": summary,
        }, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nResults: {out_path}\nAggregate: {aggregate_path}")


if __name__ == "__main__":
    main()
