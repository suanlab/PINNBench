#!/usr/bin/env python3
"""HyPINO matched-budget comparison (Reviewer 2 binding concern).

For each (PDE, seed) we run two same-budget conditions:

  * **HyPINO finetune** -- generate the target PINN weights from the HyPINO
    foundation checkpoint, then run 500 Adam steps of physics+data adaptation
    on the mapped task.  Wall-clock time T_h is recorded.
  * **Hybrid from-scratch matched-budget** -- our hybrid pipeline (PINNBase +
    3-stage trainer) trained from random initialization with Stage-2 capped at
    the same wall-clock budget T_h.

Wall-clock matching is used as a practical, hardware-honest stand-in for FLOP
matching: both branches see the same compute on the same GPU.  The HyPINO
finetune is the timing reference, so each (PDE, seed) cell is internally
consistent.

PDEs supported: Heat1D, AdvectionDiffusion1D, Burgers1D, AllenCahn1D.
Burgers1D and AllenCahn1D do not have closed-form exact solutions handled by
HyPINO's two-variable vocabulary, so for those PDEs we fall back to a
no-pretraining ``pinn_finetune`` baseline (random-init MLP fine-tuned with the
same number of Adam steps and budget) instead of the HyPINO branch.  This is
flagged in the result payload via ``hypino_branch_available``.

Outputs:
  * ``{output_dir}/{pde}_{seed}_results.json``
  * ``{output_dir}/aggregate.json``
  * ``{output_dir}/progress.json`` (updated per (pde, seed))
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam

from causal_pinn.evaluation.benchmark_runner import BenchmarkConfig, BenchmarkRunner
from causal_pinn.models.pinn.base import PINNBase
from causal_pinn.physics.losses import PhysicsLoss
from causal_pinn.physics.pde import (
    AdvectionDiffusion1D,
    AllenCahn1D,
    BurgersExact1D,
    Heat1D,
)
from causal_pinn.physics.pde.base import PDEProblem
from causal_pinn.utils.random import set_seed

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
HYPINO_STEPS = 500
HYPINO_LR = 1e-3
HYPINO_N_COLLOCATION = 512
HYPINO_N_DATA = 512
HYPINO_N_BOUNDARY = 512
HYPINO_GRID_SIZE = 224

HYBRID_STAGE1_EPOCHS = 200
HYBRID_STAGE3_EPOCHS = 200
HYBRID_STAGE1_LR = 1e-3
HYBRID_STAGE2_LR = 1e-4
HYBRID_STAGE3_LR = 1e-5
HYBRID_BATCH = 256
HYBRID_NUM_COLLOCATION = 2000
HYBRID_NUM_BOUNDARY = 200
HIDDEN_DIMS = [64, 64, 64, 64]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 5120, 6144]

# Pretraining + Stage-3 are time-bounded as a fraction of the matched budget.
STAGE1_FRAC = 0.4
STAGE3_FRAC = 0.2

PDE_REGISTRY: dict[str, dict[str, Any]] = {
    "heat1d": {
        "pde_cls": Heat1D,
        "pde_kwargs": {"alpha": 1.0, "length": 1.0, "t_max": 1.0},
        "hypino_supported": True,
    },
    "advdiff1d": {
        "pde_cls": AdvectionDiffusion1D,
        "pde_kwargs": {"velocity": 1.0, "alpha": 0.1, "length": 1.0, "t_max": 1.0},
        "hypino_supported": True,
    },
    "burgers1d": {
        # BurgersExact1D, not Burgers1D: the latter has no analytical solution,
        # so relative_l2_error came back NaN for every seed and this row carried
        # no information. The Cole-Hopf reference fixes that. HyPINO does not
        # cover Burgers, so this branch needs no external checkpoint.
        "pde_cls": BurgersExact1D,
        "pde_kwargs": {"viscosity": 0.01, "length": 1.0, "t_max": 1.0},
        "hypino_supported": False,
    },
    "allencahn1d": {
        "pde_cls": AllenCahn1D,
        "pde_kwargs": {"epsilon": 0.01, "length": 2.0, "t_max": 1.0, "front_center": 1.0},
        "hypino_supported": False,
    },
}

# ---------------------------------------------------------------------------
# HyPINO-mapped task helpers (Heat1D, AdvDiff1D)
# ---------------------------------------------------------------------------

def _heat_exact_xy(alpha: float, xy: torch.Tensor) -> torch.Tensor:
    x_unit = (xy[:, :1] + 1.0) / 2.0
    t_unit = (xy[:, 1:] + 1.0) / 2.0
    return torch.sin(torch.pi * x_unit) * torch.exp(-alpha * torch.pi**2 * t_unit)


def _heat_residual(alpha: float, net: nn.Module, xy: torch.Tensor) -> torch.Tensor:
    xy = xy.detach().clone().requires_grad_(True)
    pred = net(xy)
    grad = torch.autograd.grad(
        pred, xy, grad_outputs=torch.ones_like(pred), create_graph=True,
    )[0]
    u_x = grad[:, 0:1]
    grad_x = torch.autograd.grad(
        u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=True,
    )[0]
    u_xx = grad_x[:, 0:1]
    return grad[:, 1:2] - 2.0 * alpha * u_xx


def _advdiff_exact_xy(velocity: float, alpha: float, xy: torch.Tensor) -> torch.Tensor:
    x_unit = (xy[:, :1] + 1.0) / 2.0
    t_unit = (xy[:, 1:] + 1.0) / 2.0
    k = 2.0 * torch.pi
    return torch.sin(k * (x_unit - velocity * t_unit)) * torch.exp(-alpha * k**2 * t_unit)


def _advdiff_residual(
    velocity: float, alpha: float, net: nn.Module, xy: torch.Tensor,
) -> torch.Tensor:
    xy = xy.detach().clone().requires_grad_(True)
    pred = net(xy)
    grad = torch.autograd.grad(
        pred, xy, grad_outputs=torch.ones_like(pred), create_graph=True,
    )[0]
    u_x = grad[:, 0:1]
    grad_x = torch.autograd.grad(
        u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=True,
    )[0]
    u_xx = grad_x[:, 0:1]
    return grad[:, 1:2] + velocity * u_x - 2.0 * alpha * u_xx


def _sample_interior_xy(n: int, device: torch.device) -> torch.Tensor:
    return 2.0 * torch.rand(n, 2, device=device) - 1.0


def _sample_boundary_xy(n: int, device: torch.device) -> torch.Tensor:
    n_each = max(1, n // 3)
    y_left = 2.0 * torch.rand(n_each, 1, device=device) - 1.0
    y_right = 2.0 * torch.rand(n_each, 1, device=device) - 1.0
    x_initial = 2.0 * torch.rand(n - 2 * n_each, 1, device=device) - 1.0
    left = torch.cat([torch.full_like(y_left, -1.0), y_left], dim=1)
    right = torch.cat([torch.full_like(y_right, 1.0), y_right], dim=1)
    initial = torch.cat([x_initial, torch.full_like(x_initial, -1.0)], dim=1)
    return torch.cat([left, right, initial], dim=0)


def _build_heat_inputs(alpha: float, grid: int) -> tuple[str, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, grid)
    xx, yy = np.meshgrid(axis, axis)
    x_unit = (xx + 1.0) / 2.0
    t_unit = (yy + 1.0) / 2.0
    reference = (np.sin(np.pi * x_unit) * np.exp(-alpha * np.pi**2 * t_unit)).astype(np.float32)
    at_left = np.isclose(xx, -1.0)
    at_right = np.isclose(xx, 1.0)
    at_initial = np.isclose(yy, -1.0)
    dirichlet_mask = (at_left | at_right | at_initial).astype(np.float32)
    dirichlet_values = np.zeros_like(reference, dtype=np.float32)
    dirichlet_values[at_initial] = np.sin(np.pi * x_unit[at_initial]).astype(np.float32)
    neumann_mask = np.zeros_like(reference, dtype=np.float32)
    neumann_values = np.zeros_like(reference, dtype=np.float32)
    source = np.zeros_like(reference, dtype=np.float32)
    mat_inputs = np.stack(
        [dirichlet_mask, neumann_mask, dirichlet_values, neumann_values, source], axis=0,
    )
    pde_string = f"1.0*uy - {2.0 * alpha:.8f}*uxx"
    return pde_string, mat_inputs


def _build_advdiff_inputs(
    velocity: float, alpha: float, grid: int,
) -> tuple[str, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, grid)
    xx, yy = np.meshgrid(axis, axis)
    x_unit = (xx + 1.0) / 2.0
    t_unit = (yy + 1.0) / 2.0
    k = 2.0 * np.pi
    reference = (
        np.sin(k * (x_unit - velocity * t_unit)) * np.exp(-alpha * k**2 * t_unit)
    ).astype(np.float32)
    at_left = np.isclose(xx, -1.0)
    at_right = np.isclose(xx, 1.0)
    at_initial = np.isclose(yy, -1.0)
    dirichlet_mask = (at_left | at_right | at_initial).astype(np.float32)
    dirichlet_values = np.zeros_like(reference, dtype=np.float32)
    dirichlet_values[dirichlet_mask.astype(bool)] = reference[dirichlet_mask.astype(bool)]
    neumann_mask = np.zeros_like(reference, dtype=np.float32)
    neumann_values = np.zeros_like(reference, dtype=np.float32)
    source = np.zeros_like(reference, dtype=np.float32)
    mat_inputs = np.stack(
        [dirichlet_mask, neumann_mask, dirichlet_values, neumann_values, source], axis=0,
    )
    pde_string = f"1.0*uy + {velocity:.8f}*ux - {2.0 * alpha:.8f}*uxx"
    return pde_string, mat_inputs


# ---------------------------------------------------------------------------
# HyPINO finetune branch
# ---------------------------------------------------------------------------

def run_hypino_branch(
    pde_name: str,
    pde_kwargs: dict[str, Any],
    seed: int,
    device_name: str,
    hypino_root: Path,
    hypino_weights: Path,
) -> dict[str, Any]:
    """HyPINO-generated PINN + 500 Adam steps. Returns metrics + wall time."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # HyPINO's Conv2d stack triggers CUDNN_STATUS_NOT_INITIALIZED on this
    # cuDNN/driver pair; the eager fallback works fine and conv layers are
    # tiny relative to the rest of the cost.
    cudnn_was_enabled = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    sys.path.insert(0, str(hypino_root))
    from src.data.utils import encode_pde_str, to_tensor  # noqa: PLC0415
    from src.models import HyPINO  # noqa: PLC0415
    from src.models.utils import FNN  # noqa: PLC0415

    device = torch.device(device_name)
    old_cwd = Path.cwd()
    os.chdir(hypino_root)
    try:
        if pde_name == "heat1d":
            pde_string, mat_inputs = _build_heat_inputs(
                pde_kwargs["alpha"], HYPINO_GRID_SIZE,
            )
            exact_fn = lambda xy: _heat_exact_xy(pde_kwargs["alpha"], xy)  # noqa: E731
            residual_fn = lambda net, xy: _heat_residual(pde_kwargs["alpha"], net, xy)  # noqa: E731
        elif pde_name == "advdiff1d":
            pde_string, mat_inputs = _build_advdiff_inputs(
                pde_kwargs["velocity"], pde_kwargs["alpha"], HYPINO_GRID_SIZE,
            )
            exact_fn = lambda xy: _advdiff_exact_xy(  # noqa: E731
                pde_kwargs["velocity"], pde_kwargs["alpha"], xy,
            )
            residual_fn = lambda net, xy: _advdiff_residual(  # noqa: E731
                pde_kwargs["velocity"], pde_kwargs["alpha"], net, xy,
            )
        else:
            raise NotImplementedError(f"HyPINO branch not implemented for {pde_name}")

        pde_coeffs = encode_pde_str(pde_string)
        pde_coeffs_tensor = to_tensor([c for c in pde_coeffs.values()]).unsqueeze(0).to(device)
        mat_inputs_tensor = to_tensor(mat_inputs).unsqueeze(0).to(device)

        hypino = HyPINO.load_from_safetensors(str(hypino_weights)).eval().to(device)
        with torch.no_grad():
            _, w_gen, b_gen = hypino(
                pde_coeffs_tensor, mat_inputs_tensor, return_weights=True,
            )
        net = FNN.from_parameters(w_gen, b_gen, remove_batch_dim=True).to(device)

        optimizer = Adam(net.parameters(), lr=HYPINO_LR)
        start = time.perf_counter()

        for _step in range(HYPINO_STEPS):
            optimizer.zero_grad()
            xy_col = _sample_interior_xy(HYPINO_N_COLLOCATION, device)
            res_loss = torch.mean(residual_fn(net, xy_col) ** 2)
            xy_data = _sample_interior_xy(HYPINO_N_DATA, device)
            data_loss = F.mse_loss(net(xy_data), exact_fn(xy_data))
            xy_bc = _sample_boundary_xy(HYPINO_N_BOUNDARY, device)
            bc_loss = F.mse_loss(net(xy_bc), exact_fn(xy_bc))
            (res_loss + data_loss + bc_loss).backward()
            optimizer.step()

        wall_time = time.perf_counter() - start

        # Evaluate on uniform grid
        axis = np.linspace(-1.0, 1.0, HYPINO_GRID_SIZE)
        xx, yy = np.meshgrid(axis, axis)
        xy_np = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
        xy = torch.tensor(xy_np, device=device)
        with torch.no_grad():
            pred = net(xy).detach().cpu().numpy().reshape(HYPINO_GRID_SIZE, HYPINO_GRID_SIZE)
        x_unit = (xx + 1.0) / 2.0
        t_unit = (yy + 1.0) / 2.0
        if pde_name == "heat1d":
            ref = np.sin(np.pi * x_unit) * np.exp(-pde_kwargs["alpha"] * np.pi**2 * t_unit)
        else:
            k = 2.0 * np.pi
            ref = np.sin(k * (x_unit - pde_kwargs["velocity"] * t_unit)) * np.exp(
                -pde_kwargs["alpha"] * k**2 * t_unit,
            )
        rel_l2 = float(
            np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12),
        )
        return {
            "branch": "hypino_finetune",
            "wall_time": wall_time,
            "metrics": {
                "relative_l2_error": rel_l2,
                "mse": float(np.mean((pred - ref) ** 2)),
                "mae": float(np.mean(np.abs(pred - ref))),
                "max_error": float(np.max(np.abs(pred - ref))),
            },
        }
    finally:
        os.chdir(old_cwd)
        torch.backends.cudnn.enabled = cudnn_was_enabled


# ---------------------------------------------------------------------------
# Hybrid matched-budget branch
# ---------------------------------------------------------------------------

def _make_pinn(device: str) -> PINNBase:
    return PINNBase(input_dim=2, output_dim=1, hidden_dims=HIDDEN_DIMS).to(device)


def _generate_collocation(pde: PDEProblem, n: int, device: str) -> tuple[Tensor, Tensor]:
    t_bounds, x_bounds = pde.domain_bounds
    t = torch.rand(n, device=device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
    x = torch.rand(n, device=device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    return t.requires_grad_(True), x.requires_grad_(True)


def _has_exact(pde: PDEProblem, device: str) -> bool:
    try:
        t_sample = torch.tensor([0.5], device=device)
        x_sample = torch.tensor([0.5], device=device)
        return pde.exact_solution(t_sample, x_sample) is not None
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
    n_bc = HYBRID_NUM_BOUNDARY // 2
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
    n_ic = HYBRID_NUM_BOUNDARY
    x_ic = torch.rand(n_ic, device=device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    u_ic = pde.initial_condition(x_ic)
    t_zero = torch.zeros(n_ic, device=device)
    u_pred_ic = model(
        torch.cat([t_zero.unsqueeze(-1), x_ic.unsqueeze(-1)], dim=-1)
    ).squeeze(-1)
    ic = torch.mean((u_pred_ic - u_ic) ** 2)
    return bc + ic


def run_hybrid_branch(
    pde_name: str,
    pde: PDEProblem,
    seed: int,
    device: str,
    runner: BenchmarkRunner,
    matched_budget_seconds: float,
) -> dict[str, Any]:
    """Hybrid pipeline trained from scratch with wall-time matched to HyPINO."""
    set_seed(seed)
    model = _make_pinn(device)
    physics_loss_fn = PhysicsLoss(pde)
    has_exact = _has_exact(pde, device)

    stage1_budget = STAGE1_FRAC * matched_budget_seconds
    stage3_budget = STAGE3_FRAC * matched_budget_seconds

    # Stage 1 -- pretrain bounded by stage1_budget
    optimizer = Adam(model.parameters(), lr=HYBRID_STAGE1_LR)
    s1_start = time.perf_counter()
    s1_steps = 0
    while (time.perf_counter() - s1_start) < stage1_budget and s1_steps < HYBRID_STAGE1_EPOCHS:
        optimizer.zero_grad()
        t_colloc, x_colloc = _generate_collocation(pde, HYBRID_NUM_COLLOCATION, device)
        total = torch.tensor(0.0, device=device)
        if has_exact:
            t_data, x_data, u_data = _generate_synthetic(pde, HYBRID_BATCH, device)
            total = total + _data_loss(model, t_data, x_data, u_data)
        phys = physics_loss_fn(model, t_colloc, x_colloc)
        bc = _bc_loss(model, pde, device)
        total = total + phys["physics_loss"] + bc
        if torch.isnan(total) or torch.isinf(total):
            break
        total.backward()
        optimizer.step()
        s1_steps += 1
    s1_time = time.perf_counter() - s1_start

    # Stage 2 -- physics refinement bounded by remaining budget minus stage3_budget
    s2_budget = max(0.0, matched_budget_seconds - s1_time - stage3_budget)
    optimizer = Adam(model.parameters(), lr=HYBRID_STAGE2_LR)
    s2_start = time.perf_counter()
    s2_steps = 0
    while (time.perf_counter() - s2_start) < s2_budget:
        optimizer.zero_grad()
        t_colloc, x_colloc = _generate_collocation(pde, HYBRID_NUM_COLLOCATION, device)
        total = torch.tensor(0.0, device=device)
        if has_exact:
            t_data, x_data, u_data = _generate_synthetic(pde, HYBRID_BATCH, device)
            total = total + _data_loss(model, t_data, x_data, u_data)
        phys = physics_loss_fn(model, t_colloc, x_colloc)
        bc = _bc_loss(model, pde, device)
        progress = min(1.0, s2_steps / 200)
        lam = 0.1 + 0.9 * progress
        total = total + lam * phys["physics_loss"] + bc
        if torch.isnan(total) or torch.isinf(total):
            break
        total.backward()
        optimizer.step()
        s2_steps += 1
    s2_time = time.perf_counter() - s2_start

    # Stage 3 -- causal calibration bounded by stage3_budget
    from causal_pinn.physics.losses import CausalLoss  # noqa: PLC0415
    causal = CausalLoss(pde, epsilon=1.0, num_time_bins=100, weight=1.0)
    optimizer = Adam(model.parameters(), lr=HYBRID_STAGE3_LR)
    s3_start = time.perf_counter()
    s3_steps = 0
    while (time.perf_counter() - s3_start) < stage3_budget and s3_steps < HYBRID_STAGE3_EPOCHS:
        optimizer.zero_grad()
        t_colloc, x_colloc = _generate_collocation(pde, HYBRID_NUM_COLLOCATION, device)
        total = torch.tensor(0.0, device=device)
        if has_exact:
            t_data, x_data, u_data = _generate_synthetic(pde, HYBRID_BATCH, device)
            total = total + _data_loss(model, t_data, x_data, u_data)
        result = causal(model, t_colloc, x_colloc)
        bc = _bc_loss(model, pde, device)
        total = total + result["causal_loss"] + bc
        if torch.isnan(total) or torch.isinf(total):
            break
        total.backward()
        optimizer.step()
        s3_steps += 1
    s3_time = time.perf_counter() - s3_start

    metrics = runner.run_single(model, pde)
    return {
        "branch": "hybrid_matched",
        "wall_time": s1_time + s2_time + s3_time,
        "stage1_time": s1_time,
        "stage1_steps": s1_steps,
        "stage2_time": s2_time,
        "stage2_steps": s2_steps,
        "stage3_time": s3_time,
        "stage3_steps": s3_steps,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# PINN-only fallback for PDEs that HyPINO cannot represent
# ---------------------------------------------------------------------------

def run_pinn_only_branch(
    pde: PDEProblem,
    seed: int,
    device: str,
    runner: BenchmarkRunner,
    steps: int = HYPINO_STEPS,
) -> dict[str, Any]:
    set_seed(seed)
    model = _make_pinn(device)
    optimizer = Adam(model.parameters(), lr=HYPINO_LR)
    physics_loss_fn = PhysicsLoss(pde)
    has_exact = _has_exact(pde, device)
    start = time.perf_counter()
    for _step in range(steps):
        optimizer.zero_grad()
        t_colloc, x_colloc = _generate_collocation(pde, HYPINO_N_COLLOCATION, device)
        total = torch.tensor(0.0, device=device)
        if has_exact:
            t_data, x_data, u_data = _generate_synthetic(pde, HYPINO_N_DATA, device)
            total = total + _data_loss(model, t_data, x_data, u_data)
        phys = physics_loss_fn(model, t_colloc, x_colloc)
        bc = _bc_loss(model, pde, device)
        total = total + phys["physics_loss"] + bc
        if torch.isnan(total) or torch.isinf(total):
            break
        total.backward()
        optimizer.step()
    wall_time = time.perf_counter() - start
    metrics = runner.run_single(model, pde)
    return {
        "branch": "pinn_only_finetune",
        "wall_time": wall_time,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for branch in ("hypino_finetune", "pinn_only_finetune", "hybrid_matched"):
        sub = [r for r in rows if r.get("branch") == branch and r.get("success")]
        rl2 = [r["metrics"].get("relative_l2_error", float("nan")) for r in sub]
        times = [r.get("wall_time", 0.0) for r in sub]
        out[branch] = {
            "n": len(sub),
            "relative_l2_error": {
                "mean": float(np.nanmean(rl2)) if rl2 else float("nan"),
                "std": float(np.nanstd(rl2)) if rl2 else float("nan"),
            },
            "wall_time": {
                "mean": float(np.mean(times)) if times else 0.0,
                "std": float(np.std(times)) if times else 0.0,
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="HyPINO matched-budget comparison")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/results/paper_combined/hypino_matched_budget",
    )
    parser.add_argument(
        "--pdes", type=str, nargs="+",
        default=list(PDE_REGISTRY.keys()),
        choices=list(PDE_REGISTRY.keys()),
    )
    parser.add_argument(
        "--hypino-root", type=Path, default=Path("/tmp/hypino"),
    )
    parser.add_argument(
        "--hypino-weights", type=Path,
        default=Path("/tmp/hypino/models/hypino.safetensors"),
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

    progress: dict[str, Any] = {"completed_cells": [], "total_cells": len(args.pdes) * len(seeds)}
    progress_path = output_dir / "progress.json"
    aggregate_rows: list[dict[str, Any]] = []
    pde_summaries: dict[str, Any] = {}

    for pde_name in args.pdes:
        info = PDE_REGISTRY[pde_name]
        pde = info["pde_cls"](**info["pde_kwargs"])
        hypino_supported = info["hypino_supported"]
        pde_rows: list[dict[str, Any]] = []

        for seed in seeds:
            print(f"\n=== {pde_name}/seed={seed} ===")

            # Reference branch: HyPINO finetune (or PINN-only fallback)
            ref_branch_label = "hypino_finetune" if hypino_supported else "pinn_only_finetune"
            try:
                if hypino_supported:
                    ref = run_hypino_branch(
                        pde_name, info["pde_kwargs"], seed, args.device,
                        args.hypino_root.resolve(), args.hypino_weights.resolve(),
                    )
                else:
                    ref = run_pinn_only_branch(pde, seed, args.device, runner)
                ref["pde"] = pde_name
                ref["seed"] = seed
                ref["success"] = True
                rl2 = ref["metrics"].get("relative_l2_error", float("nan"))
                print(
                    f"  {ref_branch_label}: rel_l2={rl2:.4e} "
                    f"wall={ref['wall_time']:.1f}s"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {ref_branch_label} FAILED: {exc}")
                ref = {
                    "branch": ref_branch_label, "pde": pde_name, "seed": seed,
                    "wall_time": 0.0, "metrics": {},
                    "success": False, "error": str(exc),
                }

            # Matched-budget hybrid branch
            wall_budget = ref.get("wall_time", 0.0)
            if wall_budget < 1.0:
                wall_budget = 60.0  # safety floor
                print(f"  (using fallback {wall_budget:.1f}s budget)")
            try:
                hyb = run_hybrid_branch(
                    pde_name, pde, seed, args.device, runner, wall_budget,
                )
                hyb["pde"] = pde_name
                hyb["seed"] = seed
                hyb["success"] = True
                rl2 = hyb["metrics"].get("relative_l2_error", float("nan"))
                print(
                    f"  hybrid_matched: rel_l2={rl2:.4e} "
                    f"wall={hyb['wall_time']:.1f}s "
                    f"(s1={hyb['stage1_steps']} s2={hyb['stage2_steps']} s3={hyb['stage3_steps']})"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  hybrid_matched FAILED: {exc}")
                hyb = {
                    "branch": "hybrid_matched", "pde": pde_name, "seed": seed,
                    "wall_time": 0.0, "metrics": {},
                    "success": False, "error": str(exc),
                }

            pde_rows.extend([ref, hyb])
            aggregate_rows.extend([ref, hyb])

            cell_path = output_dir / f"{pde_name}_{seed}_results.json"
            cell_path.write_text(
                json.dumps({
                    "experiment": "hypino_matched_budget",
                    "pde": pde_name,
                    "pde_config": info["pde_kwargs"],
                    "hypino_supported": hypino_supported,
                    "seed": seed,
                    "matched_budget_seconds": wall_budget,
                    "rows": [ref, hyb],
                }, indent=2, default=str),
                encoding="utf-8",
            )
            progress["completed_cells"].append({"pde": pde_name, "seed": seed})
            progress_path.write_text(
                json.dumps(progress, indent=2), encoding="utf-8",
            )

        pde_summaries[pde_name] = aggregate(pde_rows)

    aggregate_path = output_dir / "aggregate.json"
    aggregate_path.write_text(
        json.dumps({
            "experiment": "hypino_matched_budget",
            "pdes": args.pdes,
            "seeds": seeds,
            "by_pde": pde_summaries,
            "overall": aggregate(aggregate_rows),
            "notes": [
                "HyPINO branch is run on PDEs with closed-form mappings to "
                "HyPINO's two-variable operator vocabulary (Heat1D, AdvDiff1D).",
                "Burgers1D and AllenCahn1D fall back to a PINN-only finetune "
                "baseline as the reference branch (HyPINO has no clean encoding).",
                "Matched budget is enforced via wall-clock time on the same GPU.",
            ],
        }, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nAggregate saved: {aggregate_path}")


if __name__ == "__main__":
    main()
