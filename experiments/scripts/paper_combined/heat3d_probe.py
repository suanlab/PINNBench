#!/usr/bin/env python3
"""Does pretraining still dominate at three spatial dimensions?

Track B2 of the post-review roadmap.  Reviewers pointed out that 11 of the 13
benchmark configurations are 1D and that d>=3 was untested, so any statement
about how training policies behave in higher dimensions had no evidence behind
it.  ``Heat3D`` supplies a d=3 problem that still has an analytical solution,
which is what keeps relative-L2 (and therefore regret) well defined there.

This probe runs the paper's central contrast at d=3: the full pipeline with an
operator-pretraining stage against the same pipeline with that stage skipped.
It mirrors ``noisy_data_probe.py`` in structure but is written 3D-natively --
collocation, boundary sampling over all six faces, and a 4-input network.

Outputs ``{output_dir}/results.json`` and ``aggregate.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam

from causal_pinn.evaluation.benchmark_runner import BenchmarkConfig, BenchmarkRunner
from causal_pinn.models.pinn.base import PINNBase
from causal_pinn.physics.pde import Heat3D
from causal_pinn.utils.random import set_seed

HIDDEN_DIMS = [64, 64, 64, 64]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
STAGE1_EPOCHS = 200
STAGE2_EPOCHS = 400
STAGE1_LR = 1e-3
STAGE2_LR = 1e-4
NUM_COLLOCATION = 2000
NUM_BOUNDARY = 240  # divisible by the six faces of the box
BATCH_SIZE = 1024
LAMBDA_DATA = 10.0
# The 3D fundamental mode decays at 3*alpha*pi^2, three times the 1D rate, so
# alpha=1 would leave exp(-29.6) ~ 1e-13 by t=1 and relative L2 would be scored
# against a signal that has already vanished. alpha=0.1 puts the solution's RMS
# magnitude (0.151) alongside Heat1D at alpha=1 (0.156), which is the matched
# comparison we want.
ALPHA = 0.1


def _model(device: str) -> PINNBase:
    # (t, x, y, z) -> u
    return PINNBase(input_dim=4, output_dim=1, hidden_dims=HIDDEN_DIMS).to(device)


def _forward(model: PINNBase, t: Tensor, x: Tensor) -> Tensor:
    return model(torch.cat([t.reshape(-1, 1), x.reshape(-1, 3)], dim=-1)).squeeze(-1)


def _collocation(pde: Heat3D, n: int, device: str) -> tuple[Tensor, Tensor]:
    t_bounds, x_bounds = pde.domain_bounds
    t = torch.rand(n, 1, device=device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
    x = torch.empty(n, 3, device=device)
    for d in range(3):
        lo, hi = x_bounds[2 * d], x_bounds[2 * d + 1]
        x[:, d] = torch.rand(n, device=device) * (hi - lo) + lo
    return t.requires_grad_(True), x.requires_grad_(True)


def _reference_batch(pde: Heat3D, n: int, device: str) -> tuple[Tensor, Tensor, Tensor]:
    """Clean synthetic reference data for the pretraining stage."""
    t, x = _collocation(pde, n, device)
    return t.detach(), x.detach(), pde.exact_solution(t.detach(), x.detach())


def _boundary_batch(pde: Heat3D, n: int, device: str) -> tuple[Tensor, Tensor]:
    """Sample the six faces of the box uniformly."""
    _t_bounds, x_bounds = pde.domain_bounds
    per_face = n // 6
    ts, xs = [], []
    for face in range(6):
        axis, upper = divmod(face, 2)
        t_f = torch.rand(per_face, 1, device=device)
        coords = torch.empty(per_face, 3, device=device)
        for d in range(3):
            lo, hi = x_bounds[2 * d], x_bounds[2 * d + 1]
            if d == axis:
                coords[:, d] = hi if upper else lo
            else:
                coords[:, d] = torch.rand(per_face, device=device) * (hi - lo) + lo
        ts.append(t_f)
        xs.append(coords)
    return torch.cat(ts), torch.cat(xs)


def run_single(
    condition: str, seed: int, device: str, runner: BenchmarkRunner
) -> dict[str, Any]:
    set_seed(seed)
    pde = Heat3D(alpha=ALPHA)
    model = _model(device)
    start = time.time()

    stage1: list[float] = []
    if condition == "with_pretrain":
        t_d, x_d, u_d = _reference_batch(pde, BATCH_SIZE * 4, device)
        opt = Adam(model.parameters(), lr=STAGE1_LR)
        for _ in range(STAGE1_EPOCHS):
            opt.zero_grad()
            idx = torch.randint(0, t_d.shape[0], (BATCH_SIZE,), device=device)
            loss = torch.mean((_forward(model, t_d[idx], x_d[idx]) - u_d[idx]) ** 2)
            loss.backward()
            opt.step()
            stage1.append(float(loss.item()))

    # Physics refinement, identical for both branches.
    stage2: list[float] = []
    opt = Adam(model.parameters(), lr=STAGE2_LR)
    for _ in range(STAGE2_EPOCHS):
        opt.zero_grad()
        t_c, x_c = _collocation(pde, NUM_COLLOCATION, device)
        u_c = _forward(model, t_c, x_c)
        phys = torch.mean(pde.residual(u_c, t_c, x_c) ** 2)

        t_b, x_b = _boundary_batch(pde, NUM_BOUNDARY, device)
        bc = torch.mean(_forward(model, t_b, x_b) ** 2)

        x_i = torch.rand(256, 3, device=device)
        t_i = torch.zeros(256, 1, device=device)
        ic = torch.mean((_forward(model, t_i, x_i) - pde.initial_condition(x_i)) ** 2)

        loss = phys + LAMBDA_DATA * (bc + ic)
        loss.backward()
        opt.step()
        stage2.append(float(loss.item()))

    elapsed = time.time() - start
    metrics = runner.run_single(model, pde)
    return {
        "pde": "heat3d",
        "condition": condition,
        "seed": seed,
        "training_time": elapsed,
        "stage1_losses": stage1,
        "stage2_losses": stage2,
        "metrics": metrics,
        "success": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-seeds", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/paper_combined/heat3d_probe",
    )
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
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
    conditions = ["with_pretrain", "pinn_only"]
    rows: list[dict[str, Any]] = []
    total = len(conditions) * len(seeds)
    k = 0
    for cond in conditions:
        for seed in seeds:
            k += 1
            print(f"  [{k}/{total}] heat3d/{cond}/seed={seed}...", end=" ", flush=True)
            r = run_single(cond, seed, args.device, runner)
            print(
                f"rel_l2={r['metrics']['relative_l2_error']:.4e} "
                f"time={r['training_time']:.1f}s"
            )
            rows.append(r)

    summary: dict[str, Any] = {}
    for cond in conditions:
        sub = [r for r in rows if r["condition"] == cond]
        rl2 = [r["metrics"]["relative_l2_error"] for r in sub]
        res = [r["metrics"]["pde_residual_norm"] for r in sub]
        summary[cond] = {
            "n": len(sub),
            "relative_l2_error": {
                "mean": float(np.mean(rl2)),
                "std": float(np.std(rl2)),
            },
            "pde_residual_norm": {
                "mean": float(np.mean(res)),
                "std": float(np.std(res)),
            },
        }

    a = summary["with_pretrain"]["relative_l2_error"]["mean"]
    b = summary["pinn_only"]["relative_l2_error"]["mean"]
    print("\n" + "=" * 60)
    for cond, v in summary.items():
        print(
            f"  {cond:<15} rel_l2 = {v['relative_l2_error']['mean']:.4f} "
            f"+- {v['relative_l2_error']['std']:.4f}"
        )
    print(f"  pretraining advantage: {b / a:.1f}x" if a > 0 else "")

    payload = {
        "experiment": "heat3d_probe",
        "pde": "heat3d",
        "spatial_dim": 3,
        "alpha": ALPHA,
        "conditions": conditions,
        "seeds": seeds,
        "summary": summary,
        "advantage_ratio": (b / a) if a > 0 else None,
    }
    (out / "aggregate.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out / "results.json").write_text(
        json.dumps({**payload, "raw_results": rows}, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nWrote {out}/results.json")


if __name__ == "__main__":
    main()
