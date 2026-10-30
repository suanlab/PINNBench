#!/usr/bin/env python3
"""Does pretraining still dominate when the solution carries separated scales?

Track B1 of the post-review roadmap.  Reviewers noted that the suite's 2D cases
(Wave2D, NavierStokes2D) each carry a single dominant scale, so nothing in the
benchmark spoke to multi-scale problems -- the regime where PINNs are known to
struggle through spectral bias.

``MultiscaleHeat2D`` supplies one while keeping an analytical solution: a
superposition of two Dirichlet eigenmodes whose wavelengths differ by
``scale_ratio`` (4x) and whose decay rates therefore differ by its square
(16x).  The fine mode is half the signal at t=0 and a few percent of it by
t=1, so early and late dynamics are dominated by different structure.  That
makes this the sharpest available test of the early-dynamics assumption the
routing stack rests on.

Beyond the headline contrast, this probe decomposes the error by scale.  Since
both modes are known analytically, the learned field can be projected onto each
one and scored separately, which answers a question the aggregate relative-L2
cannot: whether pretraining helps because it captures the fine structure, or
merely because it fixes the coarse envelope.

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
from causal_pinn.physics.pde import MultiscaleHeat2D
from causal_pinn.utils.random import set_seed

HIDDEN_DIMS = [64, 64, 64, 64]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]
STAGE1_EPOCHS = 200
STAGE2_EPOCHS = 400
STAGE1_LR = 1e-3
STAGE2_LR = 1e-4
NUM_COLLOCATION = 2000
NUM_BOUNDARY = 240  # divisible by the four edges of the square
BATCH_SIZE = 1024
LAMBDA_DATA = 10.0
ALPHA = 0.01
SCALE_RATIO = 4
FINE_AMPLITUDE = 0.5
# Times at which the per-scale decomposition is evaluated. Spread across the
# window so the fine mode is sampled while it is still present (t=0, 0.1) and
# after it has largely decayed (t=0.5, 1.0).
PROJECTION_TIMES = [0.0, 0.1, 0.5, 1.0]
PROJECTION_GRID = 64


def _model(device: str) -> PINNBase:
    # (t, x, y) -> u
    return PINNBase(input_dim=3, output_dim=1, hidden_dims=HIDDEN_DIMS).to(device)


def _forward(model: PINNBase, t: Tensor, x: Tensor) -> Tensor:
    return model(torch.cat([t.reshape(-1, 1), x.reshape(-1, 2)], dim=-1)).squeeze(-1)


def _collocation(pde: MultiscaleHeat2D, n: int, device: str) -> tuple[Tensor, Tensor]:
    t_bounds, x_bounds = pde.domain_bounds
    t = torch.rand(n, 1, device=device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
    x = torch.empty(n, 2, device=device)
    for d in range(2):
        lo, hi = x_bounds[2 * d], x_bounds[2 * d + 1]
        x[:, d] = torch.rand(n, device=device) * (hi - lo) + lo
    return t.requires_grad_(True), x.requires_grad_(True)


def _reference_batch(
    pde: MultiscaleHeat2D, n: int, device: str
) -> tuple[Tensor, Tensor, Tensor]:
    """Clean synthetic reference data for the pretraining stage."""
    t, x = _collocation(pde, n, device)
    return t.detach(), x.detach(), pde.exact_solution(t.detach(), x.detach())


def _boundary_batch(pde: MultiscaleHeat2D, n: int, device: str) -> tuple[Tensor, Tensor]:
    """Sample the four edges of the square uniformly."""
    _t_bounds, x_bounds = pde.domain_bounds
    per_edge = n // 4
    ts, xs = [], []
    for edge in range(4):
        axis, upper = divmod(edge, 2)
        t_e = torch.rand(per_edge, 1, device=device)
        coords = torch.empty(per_edge, 2, device=device)
        for d in range(2):
            lo, hi = x_bounds[2 * d], x_bounds[2 * d + 1]
            if d == axis:
                coords[:, d] = hi if upper else lo
            else:
                coords[:, d] = torch.rand(per_edge, device=device) * (hi - lo) + lo
        ts.append(t_e)
        xs.append(coords)
    return torch.cat(ts), torch.cat(xs)


@torch.no_grad()
def _scale_decomposition(
    model: PINNBase, pde: MultiscaleHeat2D, device: str
) -> dict[str, Any]:
    """Project the learned field onto each eigenmode and score them separately.

    On the unit square the two Dirichlet modes are orthogonal, so the modal
    coefficient of a field u at time t is 2*2*<u, phi> under the uniform
    measure (<phi, phi> = 1/4 for either mode).  Comparing the recovered
    coefficient against its analytical value separates "did the model get the
    slow envelope" from "did it get the fine structure".
    """
    lin = torch.linspace(0.0, 1.0, PROJECTION_GRID, device=device)
    gx, gy = torch.meshgrid(lin, lin, indexing="ij")
    x = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
    coarse, fine = pde._modes(x)
    norm = float((coarse**2).mean())  # = 1/4 for a full-period sine product

    out: dict[str, Any] = {"times": PROJECTION_TIMES, "coarse": [], "fine": []}
    for tv in PROJECTION_TIMES:
        t = torch.full((x.shape[0], 1), tv, device=device)
        u = _forward(model, t, x)
        a_coarse = float((u * coarse).mean() / norm)
        a_fine = float((u * fine).mean() / norm)
        true_coarse = float(np.exp(-pde.coarse_decay_rate() * tv))
        true_fine = pde.fine_amplitude * float(np.exp(-pde.fine_decay_rate() * tv))
        out["coarse"].append(
            {
                "t": tv,
                "predicted": a_coarse,
                "exact": true_coarse,
                "abs_error": abs(a_coarse - true_coarse),
                "rel_error": abs(a_coarse - true_coarse) / max(abs(true_coarse), 1e-12),
            }
        )
        out["fine"].append(
            {
                "t": tv,
                "predicted": a_fine,
                "exact": true_fine,
                "abs_error": abs(a_fine - true_fine),
                "rel_error": abs(a_fine - true_fine) / max(abs(true_fine), 1e-12),
            }
        )
    return out


def run_single(
    condition: str, seed: int, device: str, runner: BenchmarkRunner
) -> dict[str, Any]:
    set_seed(seed)
    pde = MultiscaleHeat2D(
        alpha=ALPHA, scale_ratio=SCALE_RATIO, fine_amplitude=FINE_AMPLITUDE
    )
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

        x_i = torch.rand(256, 2, device=device)
        t_i = torch.zeros(256, 1, device=device)
        ic = torch.mean((_forward(model, t_i, x_i) - pde.initial_condition(x_i)) ** 2)

        loss = phys + LAMBDA_DATA * (bc + ic)
        loss.backward()
        opt.step()
        stage2.append(float(loss.item()))

    elapsed = time.time() - start
    metrics = runner.run_single(model, pde)
    return {
        "pde": "multiscale_heat2d",
        "condition": condition,
        "seed": seed,
        "training_time": elapsed,
        "stage1_losses": stage1,
        "stage2_losses": stage2,
        "metrics": metrics,
        "scale_decomposition": _scale_decomposition(model, pde, device),
        "success": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/paper_combined/multiscale2d_probe",
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
            print(f"  [{k}/{total}] multiscale_heat2d/{cond}/seed={seed}...", end=" ", flush=True)
            r = run_single(cond, seed, args.device, runner)
            fine_t0 = r["scale_decomposition"]["fine"][0]["rel_error"]
            print(
                f"rel_l2={r['metrics']['relative_l2_error']:.4e} "
                f"fine_rel_err(t=0)={fine_t0:.3f} time={r['training_time']:.1f}s"
            )
            rows.append(r)

    summary: dict[str, Any] = {}
    for cond in conditions:
        sub = [r for r in rows if r["condition"] == cond]
        rl2 = [r["metrics"]["relative_l2_error"] for r in sub]
        res = [r["metrics"]["pde_residual_norm"] for r in sub]
        coarse0 = [r["scale_decomposition"]["coarse"][0]["rel_error"] for r in sub]
        fine0 = [r["scale_decomposition"]["fine"][0]["rel_error"] for r in sub]
        summary[cond] = {
            "n": len(sub),
            "relative_l2_error": {"mean": float(np.mean(rl2)), "std": float(np.std(rl2))},
            "pde_residual_norm": {"mean": float(np.mean(res)), "std": float(np.std(res))},
            "coarse_mode_rel_error_t0": {
                "mean": float(np.mean(coarse0)),
                "std": float(np.std(coarse0)),
            },
            "fine_mode_rel_error_t0": {
                "mean": float(np.mean(fine0)),
                "std": float(np.std(fine0)),
            },
        }

    a = summary["with_pretrain"]["relative_l2_error"]["mean"]
    b = summary["pinn_only"]["relative_l2_error"]["mean"]
    print("\n" + "=" * 68)
    for cond, v in summary.items():
        print(
            f"  {cond:<15} rel_l2 = {v['relative_l2_error']['mean']:.4f} "
            f"+- {v['relative_l2_error']['std']:.4f}   "
            f"coarse_err = {v['coarse_mode_rel_error_t0']['mean']:.3f}  "
            f"fine_err = {v['fine_mode_rel_error_t0']['mean']:.3f}"
        )
    if a > 0:
        print(f"  pretraining advantage: {b / a:.2f}x")

    payload = {
        "experiment": "multiscale2d_probe",
        "pde": "multiscale_heat2d",
        "spatial_dim": 2,
        "alpha": ALPHA,
        "scale_ratio": SCALE_RATIO,
        "fine_amplitude": FINE_AMPLITUDE,
        "spatial_scale_separation": SCALE_RATIO,
        "temporal_scale_separation": SCALE_RATIO**2,
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
