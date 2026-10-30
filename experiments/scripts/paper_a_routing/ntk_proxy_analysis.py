#!/usr/bin/env python3
"""NTK proxy analysis: gradient alignment between early and late training.

Instead of computing the full Neural Tangent Kernel (expensive), uses a
practical proxy: cosine similarity between the physics-loss gradient at
early checkpoints and the gradient at the final checkpoint, evaluated on
a FIXED set of collocation points.

Hypothesis for Paper A: policies that are easy to route (where early
physics_loss predicts final performance) should have HIGH gradient
alignment.  The alignment should be higher for residual-predictive
features than for L2-predictive features.

For each (PDE, policy) pair:
  1. Train stage 1 + stage 2, saving model checkpoints at epochs
     10, 50, 100, 250, 500.
  2. At each checkpoint, compute the gradient of physics_loss w.r.t.
     model parameters on a FIXED set of collocation points.
  3. Measure cosine similarity between each early checkpoint's gradient
     and the final (epoch 500) gradient.

Outputs:
  - ntk_proxy_results.json
  - fig_ntk_proxy.pdf  (3-panel figure, one per PDE)
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
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam

from causal_pinn.models.hybrid.causal_gate import CausalGate
from causal_pinn.models.hybrid.hybrid_model import (
    HybridConfig,
    HybridModelWrapper,
    HybridPINNOperator,
)
from causal_pinn.models.operators.fno import FNO
from causal_pinn.models.pinn.base import PINNBase
from causal_pinn.physics.losses import PhysicsLoss
from causal_pinn.physics.pde.advection_diffusion_1d import AdvectionDiffusion1D
from causal_pinn.physics.pde.allen_cahn_1d import AllenCahn1D
from causal_pinn.physics.pde.base import PDEProblem
from causal_pinn.physics.pde.heat_1d import Heat1D
from causal_pinn.utils.random import set_seed

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
SEED = 42
STAGE1_EPOCHS = 200
STAGE2_EPOCHS = 500
HIDDEN_DIMS = [64, 64, 64, 64]
FNO_MODES = 8
FNO_WIDTH = 32
FNO_NUM_LAYERS = 4
GATE_HIDDEN_DIM = 32

STAGE1_LR = 1e-3
STAGE2_LR = 1e-4

NUM_COLLOCATION = 2000
NUM_BOUNDARY = 200
BATCH_SIZE = 256

# Epochs at which to save checkpoints and compute gradient
CHECKPOINT_EPOCHS = [10, 50, 100, 250, 500]

# ---------------------------------------------------------------------------
# PDE registry
# ---------------------------------------------------------------------------
PDE_REGISTRY: dict[str, tuple[type, dict[str, Any]]] = {
    "heat1d": (
        Heat1D,
        {"alpha": 1.0, "length": 1.0, "t_max": 1.0},
    ),
    "advdiff1d": (
        AdvectionDiffusion1D,
        {"velocity": 1.0, "alpha": 0.1, "length": 1.0, "t_max": 1.0},
    ),
    "allencahn1d": (
        AllenCahn1D,
        {"epsilon": 0.01, "length": 2.0, "t_max": 1.0, "front_center": 1.0},
    ),
}

# Policy names
DEFAULT_POLICIES = ["pinn_only", "hybrid_no_gate", "hybrid_full"]


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _make_pde(pde_name: str) -> PDEProblem:
    if pde_name not in PDE_REGISTRY:
        raise ValueError(f"Unknown PDE: {pde_name}. Choose from {list(PDE_REGISTRY.keys())}")
    cls, kwargs = PDE_REGISTRY[pde_name]
    return cls(**kwargs)


def _make_model(policy: str, pde: PDEProblem, device: str) -> nn.Module:
    """Create a model for the given policy.

    Returns a model whose forward(inputs) takes (N, 2) concatenated [t, x].
    """
    input_dim = 2
    output_dim = 1

    if policy == "pinn_only":
        model = PINNBase(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=HIDDEN_DIMS,
        )
        return model.to(device)

    # Hybrid policies need PINN + FNO + gate
    pinn_network = PINNBase(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=HIDDEN_DIMS,
    )
    operator = FNO(
        in_channels=input_dim,
        out_channels=output_dim,
        modes=FNO_MODES,
        width=FNO_WIDTH,
        num_layers=FNO_NUM_LAYERS,
        dim=1,
    )
    causal_gate = CausalGate(
        out_dim=output_dim,
        hidden_dim=GATE_HIDDEN_DIM,
        causal_strength=1.0,
    )

    if policy == "hybrid_no_gate":
        config = HybridConfig(
            use_operator=True,
            use_pinn=True,
            use_causal_gate=False,
        )
    elif policy == "hybrid_full":
        config = HybridConfig(
            use_operator=True,
            use_pinn=True,
            use_causal_gate=True,
        )
    else:
        raise ValueError(f"Unknown policy: {policy}")

    hybrid = HybridPINNOperator(
        pinn_network=pinn_network,
        operator=operator,
        causal_gate=causal_gate,
        pde_problem=pde,
        config=config,
    )
    wrapper = HybridModelWrapper(hybrid)
    return wrapper.to(device)


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------

def _generate_collocation(
    pde: PDEProblem, n: int, device: str,
) -> tuple[Tensor, Tensor]:
    t_bounds, x_bounds = pde.domain_bounds
    t = torch.rand(n, device=device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
    x = torch.rand(n, device=device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    return t.requires_grad_(True), x.requires_grad_(True)


def _generate_fixed_collocation(
    pde: PDEProblem, n: int, device: str, seed: int = 999,
) -> tuple[Tensor, Tensor]:
    """Generate a FIXED set of collocation points (same across checkpoints)."""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    t_bounds, x_bounds = pde.domain_bounds
    t = torch.rand(n, generator=rng) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
    x = torch.rand(n, generator=rng) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    return t.to(device), x.to(device)


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

    n_ic = NUM_BOUNDARY
    x_ic = torch.rand(n_ic, device=device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    u_ic = pde.initial_condition(x_ic)
    t_zero = torch.zeros(n_ic, device=device)
    u_pred_ic = model(
        torch.cat([t_zero.unsqueeze(-1), x_ic.unsqueeze(-1)], dim=-1),
    ).squeeze(-1)
    ic_loss = torch.mean((u_pred_ic - u_ic) ** 2)

    return bc_loss + ic_loss


# ---------------------------------------------------------------------------
# Gradient computation (NTK proxy)
# ---------------------------------------------------------------------------

def compute_physics_gradient(
    model: nn.Module,
    pde: PDEProblem,
    t_fixed: Tensor,
    x_fixed: Tensor,
    physics_loss_fn: PhysicsLoss,
) -> Tensor:
    """Compute flattened gradient of physics loss on fixed collocation points.

    Returns a detached 1-D parameter-gradient vector.
    """
    model.train()

    # We need fresh requires_grad on the fixed points for each call
    t = t_fixed.clone().detach().requires_grad_(True)
    x = x_fixed.clone().detach().requires_grad_(True)

    result = physics_loss_fn(model, t, x)
    loss = result["physics_loss"]

    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, allow_unused=True)

    parts: list[Tensor] = []
    for g, p in zip(grads, params, strict=False):
        if g is None:
            parts.append(torch.zeros(p.numel(), device=p.device))
        else:
            parts.append(g.reshape(-1))

    if not parts:
        return torch.zeros(1, device=t_fixed.device)
    return torch.cat(parts).detach()


# ---------------------------------------------------------------------------
# Training with checkpoint saving
# ---------------------------------------------------------------------------

def train_and_collect_gradients(
    policy: str,
    pde: PDEProblem,
    device: str,
) -> dict[str, Any]:
    """Train a model through stage 1+2, collecting gradient snapshots.

    Returns dict with per-checkpoint gradient cosine similarities.
    """
    set_seed(SEED)
    model = _make_model(policy, pde, device)
    physics_loss_fn = PhysicsLoss(pde)

    # Generate FIXED collocation points for gradient probing
    t_fixed, x_fixed = _generate_fixed_collocation(pde, NUM_COLLOCATION, device)

    # Check if exact solution is available for data loss
    has_exact = True
    try:
        t_test = torch.tensor([0.5], device=device)
        x_test = torch.tensor([0.5], device=device)
        result = pde.exact_solution(t_test, x_test)
        if result is None:
            has_exact = False
    except (NotImplementedError, AttributeError, TypeError):
        has_exact = False

    # ---- Stage 1: Operator pretraining ----
    optimizer = Adam(model.parameters(), lr=STAGE1_LR)
    for _epoch in range(STAGE1_EPOCHS):
        optimizer.zero_grad()
        t_colloc, x_colloc = _generate_collocation(pde, NUM_COLLOCATION, device)

        total = torch.tensor(0.0, device=device)
        if has_exact:
            t_data, x_data, u_data = _generate_synthetic(pde, BATCH_SIZE, device)
            total = total + _compute_data_loss(model, t_data, x_data, u_data)

        phys_result = physics_loss_fn(model, t_colloc, x_colloc)
        total = total + phys_result["physics_loss"]
        total = total + _compute_boundary_loss(model, pde, device)
        total.backward()
        optimizer.step()

    print(f"      Stage 1 done ({STAGE1_EPOCHS} epochs)")

    # ---- Stage 2: Physics refinement with checkpointing ----
    optimizer = Adam(model.parameters(), lr=STAGE2_LR)

    # We will collect gradient vectors at checkpoint epochs
    checkpoint_grads: dict[int, Tensor] = {}
    checkpoint_physics_loss: dict[int, float] = {}

    checkpoint_set = set(CHECKPOINT_EPOCHS)
    max_epoch = max(CHECKPOINT_EPOCHS)

    for epoch in range(1, max_epoch + 1):
        optimizer.zero_grad()
        t_colloc, x_colloc = _generate_collocation(pde, NUM_COLLOCATION, device)

        total = torch.tensor(0.0, device=device)
        if has_exact:
            t_data, x_data, u_data = _generate_synthetic(pde, BATCH_SIZE, device)
            total = total + _compute_data_loss(model, t_data, x_data, u_data)

        phys_result = physics_loss_fn(model, t_colloc, x_colloc)
        physics_loss = phys_result["physics_loss"]

        # Linear lambda ramp
        progress = min(1.0, epoch / max(1, max_epoch // 2))
        lam = 0.1 + 0.9 * progress

        total = total + lam * physics_loss
        total = total + _compute_boundary_loss(model, pde, device)
        total.backward()
        optimizer.step()

        # Checkpoint: compute gradient on FIXED points
        if epoch in checkpoint_set:
            grad_vec = compute_physics_gradient(
                model, pde, t_fixed, x_fixed, physics_loss_fn,
            )
            checkpoint_grads[epoch] = grad_vec

            # Also record the physics loss at this checkpoint
            with torch.no_grad():
                t_probe = t_fixed.clone().detach().requires_grad_(True)
                x_probe = x_fixed.clone().detach().requires_grad_(True)
            probe_result = physics_loss_fn(model, t_probe, x_probe)
            checkpoint_physics_loss[epoch] = probe_result["physics_loss"].item()

            print(f"      Epoch {epoch:4d}  physics_loss={checkpoint_physics_loss[epoch]:.6f}")

    print(f"      Stage 2 done ({max_epoch} epochs, {len(checkpoint_grads)} checkpoints)")

    # ---- Compute cosine similarities with final gradient ----
    final_epoch = max(CHECKPOINT_EPOCHS)
    final_grad = checkpoint_grads[final_epoch]
    final_norm = torch.linalg.vector_norm(final_grad).item()

    alignment_results: list[dict[str, Any]] = []
    for ep in CHECKPOINT_EPOCHS:
        grad = checkpoint_grads[ep]
        grad_norm = torch.linalg.vector_norm(grad).item()

        if final_norm < 1e-12 or grad_norm < 1e-12:
            cos_sim = 0.0
        else:
            cos_sim = F.cosine_similarity(
                grad.unsqueeze(0), final_grad.unsqueeze(0),
            ).item()

        alignment_results.append({
            "epoch": ep,
            "cosine_similarity": cos_sim,
            "grad_norm": grad_norm,
            "physics_loss": checkpoint_physics_loss[ep],
        })

    return {
        "policy": policy,
        "checkpoints": alignment_results,
        "final_grad_norm": final_norm,
        "final_physics_loss": checkpoint_physics_loss[final_epoch],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

POLICY_COLORS = {
    "pinn_only": "#2196F3",
    "hybrid_no_gate": "#FF9800",
    "hybrid_full": "#4CAF50",
}

POLICY_LABELS = {
    "pinn_only": "PINN only",
    "hybrid_no_gate": "Hybrid (no gate)",
    "hybrid_full": "Hybrid (full)",
}

PDE_TITLES = {
    "heat1d": "Heat 1D",
    "advdiff1d": "Advection-Diffusion 1D",
    "allencahn1d": "Allen-Cahn 1D",
}


def plot_ntk_proxy(results: dict[str, Any], output_path: Path) -> None:
    """Generate fig_ntk_proxy.pdf: 3-panel figure (one per PDE)."""
    pde_names = list(results["pde_results"].keys())
    n_pdes = len(pde_names)

    fig, axes = plt.subplots(1, n_pdes, figsize=(5 * n_pdes, 4.5), squeeze=False)
    axes = axes[0]

    for ax_idx, pde_name in enumerate(pde_names):
        ax = axes[ax_idx]
        pde_data = results["pde_results"][pde_name]

        for policy_data in pde_data["policies"]:
            policy = policy_data["policy"]
            checkpoints = policy_data["checkpoints"]

            epochs = [c["epoch"] for c in checkpoints]
            cos_sims = [c["cosine_similarity"] for c in checkpoints]

            color = POLICY_COLORS.get(policy, "gray")
            label = POLICY_LABELS.get(policy, policy)

            ax.plot(
                epochs, cos_sims,
                color=color, label=label,
                linewidth=2.0, marker="o", markersize=5, alpha=0.9,
            )

        ax.set_xlabel("Stage-2 epoch", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Cosine similarity with final gradient", fontsize=11)
        ax.set_title(PDE_TITLES.get(pde_name, pde_name), fontsize=12, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.5, linestyle="-", alpha=0.3)
        ax.axhline(1, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
        ax.set_ylim(-0.15, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        "NTK Proxy: Gradient Alignment with Final Checkpoint",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NTK proxy analysis: gradient alignment between early and late training",
    )
    parser.add_argument(
        "--pdes", type=str, nargs="+",
        default=["heat1d", "advdiff1d", "allencahn1d"],
        help="PDEs to analyse (default: heat1d advdiff1d allencahn1d).",
    )
    parser.add_argument(
        "--policies", type=str, nargs="+",
        default=DEFAULT_POLICIES,
        help="Policies to compare (default: pinn_only hybrid_no_gate hybrid_full).",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: cuda if available).",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/results/paper_a/ntk_proxy",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--figure-dir", type=str,
        default="experiments/analysis/figures",
        help="Directory for figure output.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"NTK Proxy Analysis | device={device}")
    print(f"PDEs: {args.pdes}")
    print(f"Policies: {args.policies}")
    print(f"Checkpoint epochs: {CHECKPOINT_EPOCHS}")
    print(f"Stage 1: {STAGE1_EPOCHS} epochs | Stage 2: {STAGE2_EPOCHS} epochs")
    print(f"Output: {output_dir}")
    print(f"Figure: {figure_dir}")
    print()

    all_results: dict[str, Any] = {
        "experiment": "ntk_proxy_analysis",
        "seed": SEED,
        "device": device,
        "checkpoint_epochs": CHECKPOINT_EPOCHS,
        "stage1_epochs": STAGE1_EPOCHS,
        "stage2_epochs": STAGE2_EPOCHS,
        "hyperparameters": {
            "stage1_lr": STAGE1_LR,
            "stage2_lr": STAGE2_LR,
            "num_collocation": NUM_COLLOCATION,
            "num_boundary": NUM_BOUNDARY,
            "batch_size": BATCH_SIZE,
            "hidden_dims": HIDDEN_DIMS,
        },
        "pde_results": {},
    }

    total_t0 = time.perf_counter()

    for pde_name in args.pdes:
        print(f"=== PDE: {pde_name} ===")
        pde = _make_pde(pde_name)

        pde_result: dict[str, Any] = {"pde": pde_name, "policies": []}

        for policy in args.policies:
            print(f"  --- Policy: {policy} ---")
            t0 = time.perf_counter()

            policy_result = train_and_collect_gradients(policy, pde, device)
            wall_time = time.perf_counter() - t0
            policy_result["wall_time_s"] = round(wall_time, 2)

            pde_result["policies"].append(policy_result)
            print(f"      Wall time: {wall_time:.1f}s")
            print()

        all_results["pde_results"][pde_name] = pde_result

    total_wall = time.perf_counter() - total_t0
    all_results["total_wall_time_s"] = round(total_wall, 2)

    # Save JSON
    json_path = output_dir / "ntk_proxy_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {json_path}")

    # Generate figure
    fig_path = figure_dir / "fig_ntk_proxy.pdf"
    plot_ntk_proxy(all_results, fig_path)

    # Print summary table
    print()
    print("=== Summary: Cosine Similarity with Final Gradient ===")
    print(f"{'PDE':<16} {'Policy':<20} " + "  ".join(f"ep{e:>4}" for e in CHECKPOINT_EPOCHS))
    print("-" * 80)
    for pde_name, pde_data in all_results["pde_results"].items():
        for policy_data in pde_data["policies"]:
            policy = policy_data["policy"]
            cos_vals = [c["cosine_similarity"] for c in policy_data["checkpoints"]]
            cos_str = "  ".join(f"{v:>6.3f}" for v in cos_vals)
            print(f"{pde_name:<16} {policy:<20} {cos_str}")
    print()
    print(f"Total wall time: {total_wall:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
