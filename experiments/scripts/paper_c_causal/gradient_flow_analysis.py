#!/usr/bin/env python3
"""Gradient flow analysis: causal vs physics loss gradient alignment.

Measures how causal loss gradients interact with physics loss gradients
during training, comparing two conditions:
  1. with_pretrain  -- full 3-stage pipeline (stage 1 -> 2 -> 3)
  2. without_pretrain -- skip stage 1, go directly to stage 2 -> 3

Hypothesis: with pretrain, causal and physics gradients are well-aligned
(cosine > 0), so causal loss provides a useful refinement.  Without
pretrain, they may conflict (cosine < 0), explaining why causal loss hurts.

Outputs:
  - gradient_flow_results.json   per-epoch alignment metrics
  - fig_gradient_alignment.pdf   cosine similarity over epochs
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam

from causal_pinn.models.pinn.base import PINNBase
from causal_pinn.physics.losses import AdaptiveCausalLoss, PhysicsLoss
from causal_pinn.physics.pde.advection_diffusion_1d import AdvectionDiffusion1D
from causal_pinn.physics.pde.allen_cahn_1d import AllenCahn1D
from causal_pinn.physics.pde.base import PDEProblem
from causal_pinn.physics.pde.burgers_1d import Burgers1D
from causal_pinn.physics.pde.heat_1d import Heat1D
from causal_pinn.physics.pde.kdv_1d import KdV1D
from causal_pinn.physics.pde.reaction_diffusion_1d import ReactionDiffusion1D
from causal_pinn.utils.random import set_seed

# ---------------------------------------------------------------------------
# Hyper-parameters (kept small for ~5 min per condition)
# ---------------------------------------------------------------------------
SEED = 42
STAGE1_EPOCHS = 200
STAGE2_EPOCHS = 300
STAGE3_EPOCHS = 200
STAGE1_LR = 1e-3
STAGE2_LR = 1e-4
STAGE3_LR = 1e-5
CAUSAL_EPSILON = 1.0
BATCH_SIZE = 256
NUM_COLLOCATION = 2000
NUM_BOUNDARY = 200
HIDDEN_DIMS = [64, 64, 64, 64]
PROBE_EVERY = 10  # measure gradient alignment every N epochs


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GradientProbe:
    """Single gradient alignment measurement."""
    stage: str
    epoch: int
    cosine_similarity: float
    grad_magnitude_ratio: float  # |grad_causal| / |grad_physics|
    physics_grad_norm: float
    causal_grad_norm: float
    physics_loss: float
    causal_loss: float


@dataclass
class ConditionResult:
    """Results for one experimental condition."""
    condition: str
    probes: list[dict[str, Any]] = field(default_factory=list)
    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Core gradient alignment computation
# ---------------------------------------------------------------------------

def _flatten_grads(
    grads: tuple[Tensor | None, ...],
    params: list[nn.Parameter],
) -> Tensor:
    """Flatten a tuple of per-parameter gradients into a single vector."""
    parts: list[Tensor] = []
    for g, p in zip(grads, params, strict=False):
        if g is None:
            parts.append(torch.zeros(p.numel(), device=p.device))
        else:
            parts.append(g.reshape(-1))
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


def compute_gradient_alignment(
    model: nn.Module,
    physics_loss: Tensor,
    causal_loss: Tensor,
) -> dict[str, float]:
    """Compute cosine similarity and magnitude ratio between two loss gradients.

    Both losses must still be part of the computation graph (no .backward()
    called yet).  We use torch.autograd.grad with retain_graph=True so the
    graph stays alive for the subsequent training step.

    Returns dict with keys: cosine_similarity, grad_magnitude_ratio,
    physics_grad_norm, causal_grad_norm.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    physics_grads = torch.autograd.grad(
        physics_loss, params, retain_graph=True, allow_unused=True,
    )
    causal_grads = torch.autograd.grad(
        causal_loss, params, retain_graph=True, allow_unused=True,
    )

    flat_physics = _flatten_grads(physics_grads, params)
    flat_causal = _flatten_grads(causal_grads, params)

    physics_norm = torch.linalg.vector_norm(flat_physics).item()
    causal_norm = torch.linalg.vector_norm(flat_causal).item()

    if physics_norm < 1e-12 or causal_norm < 1e-12:
        cos_sim = 0.0
    else:
        cos_sim = F.cosine_similarity(
            flat_physics.unsqueeze(0), flat_causal.unsqueeze(0),
        ).item()

    mag_ratio = causal_norm / max(physics_norm, 1e-12)

    return {
        "cosine_similarity": cos_sim,
        "grad_magnitude_ratio": mag_ratio,
        "physics_grad_norm": physics_norm,
        "causal_grad_norm": causal_norm,
    }


# ---------------------------------------------------------------------------
# Training with gradient probing
# ---------------------------------------------------------------------------

def _make_model(device: str) -> PINNBase:
    model = PINNBase(
        input_dim=2,
        output_dim=1,
        hidden_dims=HIDDEN_DIMS,
    )
    return model.to(device)


PDE_REGISTRY: dict[str, tuple[type, dict, int]] = {
    "heat1d": (
        Heat1D, {"alpha": 1.0, "length": 1.0, "t_max": 1.0}, 2,
    ),
    "advdiff1d": (
        AdvectionDiffusion1D,
        {"velocity": 1.0, "alpha": 0.1, "length": 1.0, "t_max": 1.0},
        2,
    ),
    "burgers1d": (
        Burgers1D, {"viscosity": 0.01, "length": 1.0, "t_max": 1.0}, 2,
    ),
    "allencahn1d_sharp": (
        AllenCahn1D,
        {"epsilon": 0.001, "length": 2.0, "t_max": 0.5,
         "front_center": 1.0},
        2,
    ),
    "kdv1d": (
        KdV1D,
        {"dispersion": 1.0, "length": 20.0, "t_max": 0.5,
         "soliton_amplitude": 3.0, "soliton_center": 10.0},
        2,
    ),
    "reaction_diffusion1d": (
        ReactionDiffusion1D,
        {"diffusion": 0.01, "reaction_rate": 1.0,
         "length": 1.0, "t_max": 1.0},
        2,
    ),
}


def _make_pde(pde_name: str = "heat1d") -> PDEProblem:
    if pde_name not in PDE_REGISTRY:
        raise ValueError(f"Unknown PDE: {pde_name}. Choose from {list(PDE_REGISTRY.keys())}")
    cls, kwargs, _ = PDE_REGISTRY[pde_name]
    return cls(**kwargs)


def _generate_collocation(pde: PDEProblem, n: int, device: str) -> tuple[Tensor, Tensor]:
    t_bounds, x_bounds = pde.domain_bounds
    t = torch.rand(n, device=device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
    x = torch.rand(n, device=device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    return t.requires_grad_(True), x.requires_grad_(True)


def _generate_synthetic(pde: Heat1D, n: int, device: str) -> tuple[Tensor, Tensor, Tensor]:
    t_bounds, x_bounds = pde.domain_bounds
    t = torch.rand(n, device=device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
    x = torch.rand(n, device=device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    u = pde.exact_solution(t, x)
    return t, x, u


def _compute_data_loss(model: nn.Module, t: Tensor, x: Tensor, u_true: Tensor) -> Tensor:
    t_in = t.unsqueeze(-1) if t.dim() == 1 else t
    x_in = x.unsqueeze(-1) if x.dim() == 1 else x
    inp = torch.cat([t_in, x_in], dim=-1)
    u_pred = model(inp)
    if u_pred.shape != u_true.shape and u_true.dim() == 1:
        u_true = u_true.unsqueeze(-1)
    return torch.mean((u_pred - u_true) ** 2)


def _compute_boundary_loss(model: nn.Module, pde: Heat1D, device: str) -> Tensor:
    """Simplified BC + IC loss for the analysis script."""
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


def run_stage1(model: nn.Module, pde: Heat1D, device: str) -> None:
    """Stage 1: operator pretraining on synthetic data."""
    optimizer = Adam(model.parameters(), lr=STAGE1_LR)
    physics_loss_fn = PhysicsLoss(pde)

    for _epoch in range(STAGE1_EPOCHS):
        optimizer.zero_grad()
        t_colloc, x_colloc = _generate_collocation(pde, NUM_COLLOCATION, device)
        t_data, x_data, u_data = _generate_synthetic(pde, BATCH_SIZE, device)

        data_loss = _compute_data_loss(model, t_data, x_data, u_data)
        phys_result = physics_loss_fn(model, t_colloc, x_colloc)
        bc_loss = _compute_boundary_loss(model, pde, device)

        total = data_loss + phys_result["physics_loss"] + bc_loss
        total.backward()
        optimizer.step()

    print(f"    Stage 1 done ({STAGE1_EPOCHS} epochs)")


def run_stage2_with_probing(
    model: nn.Module,
    pde: Heat1D,
    device: str,
) -> list[GradientProbe]:
    """Stage 2: physics refinement.  Probe gradient alignment every PROBE_EVERY epochs."""
    optimizer = Adam(model.parameters(), lr=STAGE2_LR)
    physics_loss_fn = PhysicsLoss(pde)
    causal_loss_fn = AdaptiveCausalLoss(
        pde,
        epsilon_init=0.01,
        epsilon_final=CAUSAL_EPSILON,
        num_time_bins=100,
        weight=1.0,
        annealing_steps=STAGE2_EPOCHS,
    )

    probes: list[GradientProbe] = []

    for epoch in range(STAGE2_EPOCHS):
        causal_loss_fn.step()
        optimizer.zero_grad()

        t_colloc, x_colloc = _generate_collocation(pde, NUM_COLLOCATION, device)

        # Compute physics loss (unweighted MSE residual)
        phys_result = physics_loss_fn(model, t_colloc, x_colloc)
        physics_loss = phys_result["physics_loss"]

        # Compute causal loss on the same collocation points
        causal_result = causal_loss_fn(model, t_colloc, x_colloc)
        causal_loss = causal_result["causal_loss"]

        # Data loss and BC for actual training objective
        t_data, x_data, u_data = _generate_synthetic(pde, BATCH_SIZE, device)
        data_loss = _compute_data_loss(model, t_data, x_data, u_data)
        bc_loss = _compute_boundary_loss(model, pde, device)

        # Linear lambda ramp
        progress = min(1.0, epoch / max(1, STAGE2_EPOCHS // 2))
        lam = 0.1 + 0.9 * progress

        total = data_loss + lam * physics_loss + bc_loss

        # Probe gradient alignment at selected epochs
        if epoch % PROBE_EVERY == 0:
            alignment = compute_gradient_alignment(model, physics_loss, causal_loss)
            probe = GradientProbe(
                stage="stage2",
                epoch=epoch,
                cosine_similarity=alignment["cosine_similarity"],
                grad_magnitude_ratio=alignment["grad_magnitude_ratio"],
                physics_grad_norm=alignment["physics_grad_norm"],
                causal_grad_norm=alignment["causal_grad_norm"],
                physics_loss=physics_loss.item(),
                causal_loss=causal_loss.item(),
            )
            probes.append(probe)

        total.backward()
        optimizer.step()

    print(f"    Stage 2 done ({STAGE2_EPOCHS} epochs, {len(probes)} probes)")
    return probes


def run_stage3_with_probing(
    model: nn.Module,
    pde: Heat1D,
    device: str,
) -> list[GradientProbe]:
    """Stage 3: causal calibration.  Probe gradient alignment every PROBE_EVERY epochs."""
    optimizer = Adam(model.parameters(), lr=STAGE3_LR)
    physics_loss_fn = PhysicsLoss(pde)
    causal_loss_fn = AdaptiveCausalLoss(
        pde,
        epsilon_init=0.01,
        epsilon_final=CAUSAL_EPSILON,
        num_time_bins=100,
        weight=1.0,
        annealing_steps=STAGE3_EPOCHS,
    )

    probes: list[GradientProbe] = []

    for epoch in range(STAGE3_EPOCHS):
        causal_loss_fn.step()
        optimizer.zero_grad()

        t_colloc, x_colloc = _generate_collocation(pde, NUM_COLLOCATION, device)

        # Compute physics loss (for probing only -- not in stage 3 objective)
        phys_result = physics_loss_fn(model, t_colloc, x_colloc)
        physics_loss = phys_result["physics_loss"]

        # Compute causal loss (part of stage 3 objective)
        causal_result = causal_loss_fn(model, t_colloc, x_colloc)
        causal_loss = causal_result["causal_loss"]

        # Data loss and BC
        t_data, x_data, u_data = _generate_synthetic(pde, BATCH_SIZE, device)
        data_loss = _compute_data_loss(model, t_data, x_data, u_data)
        bc_loss = _compute_boundary_loss(model, pde, device)

        total = data_loss + causal_loss + bc_loss

        # Probe gradient alignment at selected epochs
        if epoch % PROBE_EVERY == 0:
            alignment = compute_gradient_alignment(model, physics_loss, causal_loss)
            probe = GradientProbe(
                stage="stage3",
                epoch=epoch,
                cosine_similarity=alignment["cosine_similarity"],
                grad_magnitude_ratio=alignment["grad_magnitude_ratio"],
                physics_grad_norm=alignment["physics_grad_norm"],
                causal_grad_norm=alignment["causal_grad_norm"],
                physics_loss=physics_loss.item(),
                causal_loss=causal_loss.item(),
            )
            probes.append(probe)

        total.backward()
        optimizer.step()

    print(f"    Stage 3 done ({STAGE3_EPOCHS} epochs, {len(probes)} probes)")
    return probes


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_condition(condition: str, pde: Heat1D, device: str) -> ConditionResult:
    """Run one experimental condition and collect gradient probes."""
    set_seed(SEED)
    model = _make_model(device)

    t0 = time.perf_counter()

    if condition == "with_pretrain":
        print("  [with_pretrain] Running stage 1 ...")
        run_stage1(model, pde, device)
    else:
        print("  [without_pretrain] Skipping stage 1")

    print(f"  [{condition}] Running stage 2 with gradient probing ...")
    probes_s2 = run_stage2_with_probing(model, pde, device)

    print(f"  [{condition}] Running stage 3 with gradient probing ...")
    probes_s3 = run_stage3_with_probing(model, pde, device)

    wall_time = time.perf_counter() - t0

    all_probes = [asdict(p) for p in probes_s2 + probes_s3]

    return ConditionResult(
        condition=condition,
        probes=all_probes,
        wall_time_s=round(wall_time, 2),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_gradient_alignment(results: dict[str, Any], output_dir: Path) -> None:
    """Generate fig_gradient_alignment.pdf with two subplots:
      (a) Cosine similarity vs epoch
      (b) Gradient magnitude ratio vs epoch
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.0))

    colors = {"with_pretrain": "#2196F3", "without_pretrain": "#F44336"}
    labels = {
        "with_pretrain": "With pretrain (full pipeline)",
        "without_pretrain": "Without pretrain (skip stage 1)",
    }

    for cond_name, cond_data in results["conditions"].items():
        probes = cond_data["probes"]
        if not probes:
            continue

        # Separate stage 2 and stage 3 probes
        s2 = [p for p in probes if p["stage"] == "stage2"]
        s3 = [p for p in probes if p["stage"] == "stage3"]

        # Build a continuous epoch axis: stage2 epochs, then stage3 offset
        s2_epochs = [p["epoch"] for p in s2]
        s3_epochs = [STAGE2_EPOCHS + p["epoch"] for p in s3]
        all_epochs = s2_epochs + s3_epochs

        cos_sims = [p["cosine_similarity"] for p in s2] + [p["cosine_similarity"] for p in s3]
        mag_ratios = (
            [p["grad_magnitude_ratio"] for p in s2]
            + [p["grad_magnitude_ratio"] for p in s3]
        )

        color = colors[cond_name]
        label = labels[cond_name]

        axes[0].plot(all_epochs, cos_sims, color=color, label=label, linewidth=1.5, alpha=0.85)
        axes[1].plot(all_epochs, mag_ratios, color=color, label=label, linewidth=1.5, alpha=0.85)

    # Stage boundary
    for ax in axes:
        ax.axvline(STAGE2_EPOCHS, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(
            STAGE2_EPOCHS, ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 0.9,
            " stage 3", fontsize=8, color="gray", va="top",
        )

    # Subplot (a): cosine similarity — zoom in near 1.0 so the blue/red
    # lines (typically in [0.9997, 1.0000]) are visually separable.
    axes[0].set_xlabel("Epoch (stage 2 + stage 3)")
    axes[0].set_ylabel("Cosine similarity\n(physics vs causal)")
    axes[0].set_title("(a) Gradient alignment")
    axes[0].set_ylim(0.9995, 1.0002)
    axes[0].ticklabel_format(axis="y", useOffset=False)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Subplot (b): magnitude ratio — linear scale, tight range near 1.0,
    # so the blue/red lines (typically in [0.98, 1.00]) are separable.
    axes[1].set_xlabel("Epoch (stage 2 + stage 3)")
    axes[1].set_ylabel("|grad_causal| / |grad_physics|")
    axes[1].set_title("(b) Gradient magnitude ratio")
    axes[1].set_ylim(0.970, 1.005)
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout(pad=0.6)
    fig_path = output_dir / "fig_gradient_alignment.pdf"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Figure saved to {fig_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gradient flow analysis: causal vs physics loss alignment",
    )
    parser.add_argument(
        "--pde", type=str, default="heat1d",
        help="PDE to use (default: heat1d). Currently only heat1d is supported.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/results/paper_c_gradient_flow",
        help="Directory for output files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"Gradient flow analysis | PDE={args.pde} | device={device}")
    print(f"Output: {output_dir}")
    print(f"Probe every {PROBE_EVERY} epochs | "
          f"Stage1={STAGE1_EPOCHS}  Stage2={STAGE2_EPOCHS}  Stage3={STAGE3_EPOCHS}")
    print()

    pde = _make_pde(args.pde)

    # Run both conditions
    conditions: dict[str, Any] = {}
    for condition in ("with_pretrain", "without_pretrain"):
        print(f"=== Condition: {condition} ===")
        result = run_condition(condition, pde, device)
        conditions[condition] = asdict(result)
        print(f"  Wall time: {result.wall_time_s:.1f}s")
        print()

    # Assemble final results
    results = {
        "experiment": "gradient_flow_analysis",
        "pde": args.pde,
        "seed": SEED,
        "device": device,
        "hyperparameters": {
            "stage1_epochs": STAGE1_EPOCHS,
            "stage2_epochs": STAGE2_EPOCHS,
            "stage3_epochs": STAGE3_EPOCHS,
            "stage1_lr": STAGE1_LR,
            "stage2_lr": STAGE2_LR,
            "stage3_lr": STAGE3_LR,
            "causal_epsilon": CAUSAL_EPSILON,
            "num_collocation": NUM_COLLOCATION,
            "num_boundary": NUM_BOUNDARY,
            "batch_size": BATCH_SIZE,
            "hidden_dims": HIDDEN_DIMS,
            "probe_every": PROBE_EVERY,
        },
        "conditions": conditions,
    }

    # Save JSON
    json_path = output_dir / "gradient_flow_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")

    # Generate figure
    plot_gradient_alignment(results, output_dir)

    # Print summary
    print()
    print("=== Summary ===")
    for cond_name, cond_data in conditions.items():
        probes = cond_data["probes"]
        if not probes:
            print(f"  {cond_name}: no probes collected")
            continue
        cos_vals = [p["cosine_similarity"] for p in probes]
        s2_cos = [p["cosine_similarity"] for p in probes if p["stage"] == "stage2"]
        s3_cos = [p["cosine_similarity"] for p in probes if p["stage"] == "stage3"]
        print(f"  {cond_name}:")
        print(f"    Overall  cosine  mean={np.mean(cos_vals):+.4f}  "
              f"std={np.std(cos_vals):.4f}  "
              f"min={np.min(cos_vals):+.4f}  max={np.max(cos_vals):+.4f}")
        if s2_cos:
            print(f"    Stage 2  cosine  mean={np.mean(s2_cos):+.4f}  "
                  f"std={np.std(s2_cos):.4f}")
        if s3_cos:
            print(f"    Stage 3  cosine  mean={np.mean(s3_cos):+.4f}  "
                  f"std={np.std(s3_cos):.4f}")
        print(f"    Wall time: {cond_data['wall_time_s']:.1f}s")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
