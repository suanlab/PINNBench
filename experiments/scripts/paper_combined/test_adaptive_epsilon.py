#!/usr/bin/env python3
"""Adaptive epsilon experiment: does residual-aware epsilon prevent causal weight collapse?

Compares 4 conditions on 4 PDEs (heat1d, advdiff1d, burgers1d, allencahn1d_sharp):
  1. no_causal          -- baseline, stages 1+2 only (no stage 3)
  2. fixed_epsilon      -- stages 1+2+3 with fixed epsilon=1.0
  3. adaptive_epsilon   -- stages 1+2+3 with adaptive epsilon
  4. no_pretrain_adaptive -- skip stage 1, stages 2+3 with adaptive epsilon
                            (critical test: does adaptive fix the collapse?)

For each PDE x condition x 5 seeds, trains a PINN (custom loop) and evaluates.

The adaptive epsilon is applied by computing mean PDE residual on collocation
points BEFORE each stage-3 epoch, then calling scheduler.compute_epsilon()
to set CausalLoss.epsilon for that epoch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam

from causal_pinn.evaluation.benchmark_runner import BenchmarkConfig, BenchmarkRunner
from causal_pinn.models.pinn.base import PINNBase
from causal_pinn.physics.losses import (
    AdaptiveEpsilonScheduler,
    CausalLoss,
    PhysicsLoss,
)
from causal_pinn.physics.pde.advection_diffusion_1d import AdvectionDiffusion1D
from causal_pinn.physics.pde.allen_cahn_1d import AllenCahn1D
from causal_pinn.physics.pde.base import PDEProblem
from causal_pinn.physics.pde.burgers_1d import Burgers1D
from causal_pinn.physics.pde.heat_1d import Heat1D
from causal_pinn.utils.random import set_seed

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
STAGE1_EPOCHS = 200
STAGE2_EPOCHS = 400
STAGE3_EPOCHS = 200
STAGE1_LR = 1e-3
STAGE2_LR = 1e-4
STAGE3_LR = 1e-5
FIXED_EPSILON = 1.0
BATCH_SIZE = 256
NUM_COLLOCATION = 2000
NUM_BOUNDARY = 200
HIDDEN_DIMS = [64, 64, 64, 64]
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]

# Adaptive epsilon defaults
ADAPTIVE_EPSILON_BASE = 1.0
ADAPTIVE_BETA = 10.0
ADAPTIVE_EPSILON_MIN = 0.01
ADAPTIVE_EPSILON_MAX = 10.0

PDE_REGISTRY: dict[str, tuple[type, dict[str, Any], int]] = {
    "heat1d": (
        Heat1D,
        {"alpha": 1.0, "length": 1.0, "t_max": 1.0},
        2,
    ),
    "advdiff1d": (
        AdvectionDiffusion1D,
        {"velocity": 1.0, "alpha": 0.1, "length": 1.0, "t_max": 1.0},
        2,
    ),
    "burgers1d": (
        Burgers1D,
        {"viscosity": 0.01, "length": 1.0, "t_max": 1.0},
        2,
    ),
    "allencahn1d_sharp": (
        AllenCahn1D,
        {"epsilon": 0.001, "length": 2.0, "t_max": 0.5, "front_center": 1.0},
        2,
    ),
}

# Condition definitions: (label, skip_stage1, skip_stage3, use_adaptive_epsilon)
CONDITIONS = [
    ("no_causal", False, True, False),
    ("fixed_epsilon", False, False, False),
    ("adaptive_epsilon", False, False, True),
    ("no_pretrain_adaptive", True, False, True),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(device: str) -> PINNBase:
    model = PINNBase(input_dim=2, output_dim=1, hidden_dims=HIDDEN_DIMS)
    return model.to(device)


def _generate_collocation(
    pde: PDEProblem, n: int, device: str,
) -> tuple[Tensor, Tensor]:
    t_bounds, x_bounds = pde.domain_bounds
    t = torch.rand(n, device=device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
    x = torch.rand(n, device=device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    return t.requires_grad_(True), x.requires_grad_(True)


def _has_exact_solution(pde: PDEProblem, device: str) -> bool:
    """Check if the PDE has a usable exact solution."""
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


def _compute_data_loss(
    model: nn.Module, t: Tensor, x: Tensor, u_true: Tensor,
) -> Tensor:
    t_in = t.unsqueeze(-1) if t.dim() == 1 else t
    x_in = x.unsqueeze(-1) if x.dim() == 1 else x
    inp = torch.cat([t_in, x_in], dim=-1)
    u_pred = model(inp)
    if u_pred.shape != u_true.shape and u_true.dim() == 1:
        u_true = u_true.unsqueeze(-1)
    return torch.mean((u_pred - u_true) ** 2)


def _compute_boundary_loss(
    model: nn.Module, pde: PDEProblem, device: str,
) -> Tensor:
    t_bounds, x_bounds = pde.domain_bounds
    n_bc = NUM_BOUNDARY // 2

    # Spatial boundaries
    t_left = torch.rand(n_bc, device=device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
    x_left = torch.full((n_bc,), x_bounds[0], device=device)
    t_right = torch.rand(n_bc, device=device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
    x_right = torch.full((n_bc,), x_bounds[1], device=device)

    t_bc = torch.cat([t_left, t_right])
    x_bc = torch.cat([x_left, x_right])
    u_bc = pde.boundary_condition(t_bc, x_bc)

    t_in = t_bc.unsqueeze(-1)
    x_in = x_bc.unsqueeze(-1)
    u_pred_bc = model(torch.cat([t_in, x_in], dim=-1)).squeeze(-1)
    bc_loss = torch.mean((u_pred_bc - u_bc) ** 2)

    # Initial condition
    n_ic = NUM_BOUNDARY
    x_ic = torch.rand(n_ic, device=device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    u_ic = pde.initial_condition(x_ic)
    t_zero = torch.zeros(n_ic, device=device)
    u_pred_ic = model(
        torch.cat([t_zero.unsqueeze(-1), x_ic.unsqueeze(-1)], dim=-1)
    ).squeeze(-1)
    ic_loss = torch.mean((u_pred_ic - u_ic) ** 2)

    return bc_loss + ic_loss


def _compute_mean_residual(
    model: nn.Module, pde: PDEProblem, device: str, n_points: int = 1000,
) -> float:
    """Compute mean squared PDE residual on fresh collocation points (no grad)."""
    with torch.no_grad():
        t_bounds, x_bounds = pde.domain_bounds
        t = torch.rand(n_points, device=device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
        x = torch.rand(n_points, device=device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]

    t = t.requires_grad_(True)
    x = x.requires_grad_(True)

    t_in = t.unsqueeze(-1)
    x_in = x.unsqueeze(-1)
    inp = torch.cat([t_in, x_in], dim=-1)
    u = model(inp).squeeze(-1)
    residual = pde.residual(u, t, x)
    mean_res = torch.mean(residual ** 2).item()
    return mean_res


# ---------------------------------------------------------------------------
# Training stages
# ---------------------------------------------------------------------------

def run_stage1(model: nn.Module, pde: PDEProblem, device: str) -> list[float]:
    """Stage 1: operator pretraining on synthetic data. Returns loss history."""
    optimizer = Adam(model.parameters(), lr=STAGE1_LR)
    physics_loss_fn = PhysicsLoss(pde)
    has_exact = _has_exact_solution(pde, device)
    losses: list[float] = []

    for _epoch in range(STAGE1_EPOCHS):
        optimizer.zero_grad()
        t_colloc, x_colloc = _generate_collocation(pde, NUM_COLLOCATION, device)

        total = torch.tensor(0.0, device=device)
        if has_exact:
            t_data, x_data, u_data = _generate_synthetic(pde, BATCH_SIZE, device)
            data_loss = _compute_data_loss(model, t_data, x_data, u_data)
            total = total + data_loss

        phys_result = physics_loss_fn(model, t_colloc, x_colloc)
        bc_loss = _compute_boundary_loss(model, pde, device)

        total = total + phys_result["physics_loss"] + bc_loss
        total.backward()
        optimizer.step()
        losses.append(total.item())

    return losses


def run_stage2(model: nn.Module, pde: PDEProblem, device: str) -> list[float]:
    """Stage 2: physics refinement with ramping lambda. Returns loss history."""
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
            data_loss = _compute_data_loss(model, t_data, x_data, u_data)
            total = total + data_loss

        phys_result = physics_loss_fn(model, t_colloc, x_colloc)
        bc_loss = _compute_boundary_loss(model, pde, device)

        progress = min(1.0, epoch / max(1, STAGE2_EPOCHS // 2))
        lam = 0.1 + 0.9 * progress

        total = total + lam * phys_result["physics_loss"] + bc_loss
        total.backward()
        optimizer.step()
        losses.append(total.item())

    return losses


def run_stage3_fixed(
    model: nn.Module, pde: PDEProblem, device: str, epsilon: float,
) -> dict[str, Any]:
    """Stage 3 with fixed epsilon. Returns loss history and weight stats."""
    optimizer = Adam(model.parameters(), lr=STAGE3_LR)
    causal_loss_fn = CausalLoss(pde, epsilon=epsilon, num_time_bins=100, weight=1.0)
    has_exact = _has_exact_solution(pde, device)
    losses: list[float] = []
    weight_stats: list[dict[str, float]] = []

    for _epoch in range(STAGE3_EPOCHS):
        optimizer.zero_grad()
        t_colloc, x_colloc = _generate_collocation(pde, NUM_COLLOCATION, device)

        total = torch.tensor(0.0, device=device)
        if has_exact:
            t_data, x_data, u_data = _generate_synthetic(pde, BATCH_SIZE, device)
            data_loss = _compute_data_loss(model, t_data, x_data, u_data)
            total = total + data_loss

        causal_result = causal_loss_fn(model, t_colloc, x_colloc)
        bc_loss = _compute_boundary_loss(model, pde, device)

        total = total + causal_result["causal_loss"] + bc_loss
        total.backward()
        optimizer.step()

        losses.append(total.item())

        # Record causal weight statistics
        w = causal_result["causal_weights"].detach()
        weight_stats.append({
            "min": w.min().item(),
            "max": w.max().item(),
            "mean": w.mean().item(),
            "std": w.std().item(),
            "epsilon": epsilon,
        })

    return {"losses": losses, "weight_stats": weight_stats}


def run_stage3_adaptive(
    model: nn.Module, pde: PDEProblem, device: str,
    scheduler: AdaptiveEpsilonScheduler,
) -> dict[str, Any]:
    """Stage 3 with adaptive epsilon. Returns loss history, weight stats, epsilon history."""
    optimizer = Adam(model.parameters(), lr=STAGE3_LR)
    causal_loss_fn = CausalLoss(pde, epsilon=FIXED_EPSILON, num_time_bins=100, weight=1.0)
    has_exact = _has_exact_solution(pde, device)
    losses: list[float] = []
    weight_stats: list[dict[str, float]] = []

    for _epoch in range(STAGE3_EPOCHS):
        # Compute adaptive epsilon from current residual state
        mean_res = _compute_mean_residual(model, pde, device)
        eps = scheduler.compute_epsilon(mean_res)
        causal_loss_fn.epsilon = eps

        optimizer.zero_grad()
        t_colloc, x_colloc = _generate_collocation(pde, NUM_COLLOCATION, device)

        total = torch.tensor(0.0, device=device)
        if has_exact:
            t_data, x_data, u_data = _generate_synthetic(pde, BATCH_SIZE, device)
            data_loss = _compute_data_loss(model, t_data, x_data, u_data)
            total = total + data_loss

        causal_result = causal_loss_fn(model, t_colloc, x_colloc)
        bc_loss = _compute_boundary_loss(model, pde, device)

        total = total + causal_result["causal_loss"] + bc_loss
        total.backward()
        optimizer.step()

        losses.append(total.item())

        # Record causal weight statistics
        w = causal_result["causal_weights"].detach()
        weight_stats.append({
            "min": w.min().item(),
            "max": w.max().item(),
            "mean": w.mean().item(),
            "std": w.std().item(),
            "epsilon": eps,
            "mean_residual": mean_res,
        })

    return {"losses": losses, "weight_stats": weight_stats}


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(
    pde_name: str,
    pde: PDEProblem,
    condition_label: str,
    skip_stage1: bool,
    skip_stage3: bool,
    use_adaptive: bool,
    seed: int,
    device: str,
    benchmark_runner: BenchmarkRunner,
) -> dict[str, Any]:
    """Run one condition for one seed and return metrics."""
    set_seed(seed)
    model = _make_model(device)

    start = time.time()
    stage_history: dict[str, Any] = {}

    # Stage 1
    if not skip_stage1:
        s1_losses = run_stage1(model, pde, device)
        stage_history["stage1_losses"] = s1_losses

    # Stage 2
    s2_losses = run_stage2(model, pde, device)
    stage_history["stage2_losses"] = s2_losses

    # Stage 3
    stage3_info: dict[str, Any] = {}
    if not skip_stage3:
        if use_adaptive:
            scheduler = AdaptiveEpsilonScheduler(
                epsilon_base=ADAPTIVE_EPSILON_BASE,
                beta=ADAPTIVE_BETA,
                epsilon_min=ADAPTIVE_EPSILON_MIN,
                epsilon_max=ADAPTIVE_EPSILON_MAX,
            )
            stage3_info = run_stage3_adaptive(model, pde, device, scheduler)
            stage3_info["epsilon_history"] = [
                h["epsilon"] for h in scheduler.history
            ]
            stage3_info["residual_history"] = [
                h["mean_residual"] for h in scheduler.history
            ]
        else:
            stage3_info = run_stage3_fixed(model, pde, device, epsilon=FIXED_EPSILON)

    stage_history["stage3_info"] = stage3_info
    training_time = time.time() - start

    # Evaluate
    metrics = benchmark_runner.run_single(model, pde)

    return {
        "pde": pde_name,
        "condition": condition_label,
        "skip_stage1": skip_stage1,
        "skip_stage3": skip_stage3,
        "use_adaptive": use_adaptive,
        "seed": seed,
        "training_time": training_time,
        "stage_history": stage_history,
        "metrics": metrics,
        "success": True,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_epsilon_and_weights(
    all_results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate figures showing epsilon adaptation and weight collapse prevention."""
    pde_names = sorted({r["pde"] for r in all_results})

    # Figure 1: Epsilon trajectory for adaptive conditions
    fig, axes = plt.subplots(1, len(pde_names), figsize=(5 * len(pde_names), 4), squeeze=False)
    for i, pde_name in enumerate(pde_names):
        ax = axes[0, i]
        for cond_label in ["adaptive_epsilon", "no_pretrain_adaptive"]:
            cond_results = [
                r for r in all_results
                if r["pde"] == pde_name and r["condition"] == cond_label
                and r["success"]
            ]
            if not cond_results:
                continue
            # Use first seed for trajectory plot
            r = cond_results[0]
            eps_hist = r.get("stage_history", {}).get("stage3_info", {}).get("epsilon_history", [])
            if eps_hist:
                ax.plot(eps_hist, label=cond_label, linewidth=1.5, alpha=0.85)
        ax.set_title(pde_name)
        ax.set_xlabel("Stage 3 Epoch")
        ax.set_ylabel("Epsilon")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Adaptive Epsilon Trajectories", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_epsilon_trajectories.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Causal weight min across epochs (collapse indicator)
    fig, axes = plt.subplots(1, len(pde_names), figsize=(5 * len(pde_names), 4), squeeze=False)
    colors = {
        "fixed_epsilon": "#F44336",
        "adaptive_epsilon": "#2196F3",
        "no_pretrain_adaptive": "#4CAF50",
    }
    for i, pde_name in enumerate(pde_names):
        ax = axes[0, i]
        for cond_label in ["fixed_epsilon", "adaptive_epsilon", "no_pretrain_adaptive"]:
            cond_results = [
                r for r in all_results
                if r["pde"] == pde_name and r["condition"] == cond_label
                and r["success"]
            ]
            if not cond_results:
                continue
            r = cond_results[0]
            ws = r.get("stage_history", {}).get("stage3_info", {}).get("weight_stats", [])
            if ws:
                mins = [w["min"] for w in ws]
                ax.plot(
                    mins, label=cond_label, linewidth=1.5, alpha=0.85,
                    color=colors.get(cond_label, "gray"),
                )
        ax.set_title(pde_name)
        ax.set_xlabel("Stage 3 Epoch")
        ax.set_ylabel("Min Causal Weight")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Causal Weight Collapse: Min Weight Over Training", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_weight_collapse.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Figures saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive epsilon experiment: prevent causal weight collapse",
    )
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/results/paper_combined/adaptive_epsilon",
    )
    parser.add_argument(
        "--pdes", type=str, nargs="+",
        default=list(PDE_REGISTRY.keys()),
        choices=list(PDE_REGISTRY.keys()),
        help="PDEs to test (default: all 4)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    pde_names = args.pdes
    total_runs = len(pde_names) * len(CONDITIONS) * len(seeds)

    print("=" * 70)
    print("Adaptive Epsilon Experiment")
    print(f"PDEs: {pde_names}")
    print(f"Conditions: {[c[0] for c in CONDITIONS]}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {total_runs}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    print()

    all_results: list[dict[str, Any]] = []
    run_count = 0

    for pde_name in pde_names:
        pde_cls, pde_kwargs, _ = PDE_REGISTRY[pde_name]
        pde = pde_cls(**pde_kwargs)

        print(f"\n{'='*50}")
        print(f"PDE: {pde_name}")
        print(f"{'='*50}")

        for cond_label, skip_s1, skip_s3, use_adaptive in CONDITIONS:
            print(f"\n--- {cond_label} (skip_s1={skip_s1}, skip_s3={skip_s3}, adaptive={use_adaptive}) ---")  # noqa: E501

            for seed in seeds:
                run_count += 1
                tag = f"[{run_count}/{total_runs}] {pde_name}/{cond_label}/seed={seed}"
                print(f"  {tag}...", end=" ", flush=True)

                try:
                    result = run_single(
                        pde_name, pde, cond_label,
                        skip_s1, skip_s3, use_adaptive,
                        seed, args.device, runner,
                    )
                    rl2 = result["metrics"].get("relative_l2_error", float("nan"))
                    pde_res = result["metrics"].get("pde_residual_norm", float("nan"))
                    t_sec = result["training_time"]
                    print(f"rel_l2={rl2:.4e} pde_res={pde_res:.4e} time={t_sec:.1f}s")
                except Exception as exc:
                    print(f"FAILED: {exc}")
                    result = {
                        "pde": pde_name,
                        "condition": cond_label,
                        "skip_stage1": skip_s1,
                        "skip_stage3": skip_s3,
                        "use_adaptive": use_adaptive,
                        "seed": seed,
                        "training_time": 0.0,
                        "stage_history": {},
                        "metrics": {},
                        "success": False,
                        "error": str(exc),
                    }
                all_results.append(result)

    # -----------------------------------------------------------------------
    # Aggregate summary
    # -----------------------------------------------------------------------
    summary: dict[str, dict[str, dict[str, Any]]] = {}
    metric_keys = ["relative_l2_error", "pde_residual_norm", "bc_violation", "ic_violation"]

    for pde_name in pde_names:
        summary[pde_name] = {}
        for cond_label, _, _, _ in CONDITIONS:
            cond_results = [
                r for r in all_results
                if r["pde"] == pde_name and r["condition"] == cond_label and r["success"]
            ]
            cond_summary: dict[str, Any] = {}
            for m in metric_keys:
                vals = [r["metrics"].get(m, float("nan")) for r in cond_results]
                cond_summary[m] = {
                    "mean": float(np.nanmean(vals)) if vals else float("nan"),
                    "std": float(np.nanstd(vals)) if vals else float("nan"),
                    "n": len(vals),
                }
            times = [r["training_time"] for r in cond_results]
            cond_summary["training_time"] = {
                "mean": float(np.mean(times)) if times else 0.0,
                "std": float(np.std(times)) if times else 0.0,
            }
            # Collapse metric: mean min-weight at end of stage 3
            end_min_weights = []
            for r in cond_results:
                ws = r.get("stage_history", {}).get("stage3_info", {}).get("weight_stats", [])
                if ws:
                    end_min_weights.append(ws[-1]["min"])
            cond_summary["final_min_causal_weight"] = {
                "mean": float(np.mean(end_min_weights)) if end_min_weights else float("nan"),
                "std": float(np.std(end_min_weights)) if end_min_weights else float("nan"),
            }
            summary[pde_name][cond_label] = cond_summary

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    # Strip bulky stage history for the JSON to keep it manageable
    slim_results = []
    for r in all_results:
        slim = {k: v for k, v in r.items() if k != "stage_history"}
        # Keep only weight_stats and epsilon_history from stage_history
        sh = r.get("stage_history", {})
        s3 = sh.get("stage3_info", {})
        slim["stage3_weight_stats"] = s3.get("weight_stats", [])
        slim["stage3_epsilon_history"] = s3.get("epsilon_history", [])
        slim["stage3_residual_history"] = s3.get("residual_history", [])
        slim_results.append(slim)

    payload = {
        "experiment": "adaptive_epsilon",
        "pdes": pde_names,
        "conditions": [c[0] for c in CONDITIONS],
        "seeds": seeds,
        "hyperparameters": {
            "stage1_epochs": STAGE1_EPOCHS,
            "stage2_epochs": STAGE2_EPOCHS,
            "stage3_epochs": STAGE3_EPOCHS,
            "stage1_lr": STAGE1_LR,
            "stage2_lr": STAGE2_LR,
            "stage3_lr": STAGE3_LR,
            "fixed_epsilon": FIXED_EPSILON,
            "adaptive_epsilon_base": ADAPTIVE_EPSILON_BASE,
            "adaptive_beta": ADAPTIVE_BETA,
            "adaptive_epsilon_min": ADAPTIVE_EPSILON_MIN,
            "adaptive_epsilon_max": ADAPTIVE_EPSILON_MAX,
            "num_collocation": NUM_COLLOCATION,
            "num_boundary": NUM_BOUNDARY,
            "batch_size": BATCH_SIZE,
            "hidden_dims": HIDDEN_DIMS,
        },
        "summary": summary,
        "raw_results": slim_results,
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {results_path}")

    # -----------------------------------------------------------------------
    # Generate figures
    # -----------------------------------------------------------------------
    try:
        plot_epsilon_and_weights(all_results, output_dir)
    except Exception as exc:
        print(f"Warning: plotting failed: {exc}")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    header = f"{'PDE':<20s} {'Condition':<25s} {'Rel L2':>14s} {'PDE Res':>14s} {'Min W (end)':>14s} {'Time':>8s}"  # noqa: E501
    print(header)
    print("-" * 100)

    for pde_name in pde_names:
        for cond_label, _, _, _ in CONDITIONS:
            s = summary.get(pde_name, {}).get(cond_label, {})
            rl2 = s.get("relative_l2_error", {})
            pres = s.get("pde_residual_norm", {})
            mw = s.get("final_min_causal_weight", {})
            t_info = s.get("training_time", {})
            print(
                f"{pde_name:<20s} "
                f"{cond_label:<25s} "
                f"{rl2.get('mean', 0):.4e}+/-{rl2.get('std', 0):.2e} "
                f"{pres.get('mean', 0):.4e}+/-{pres.get('std', 0):.2e} "
                f"{mw.get('mean', float('nan')):.4e} "
                f"{t_info.get('mean', 0):>6.1f}s"
            )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
