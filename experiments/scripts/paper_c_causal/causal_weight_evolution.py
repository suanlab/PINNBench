#!/usr/bin/env python3
"""Paper C: Causal weight evolution visualization.

Shows how the causal weights w_i = exp(-epsilon * sum_{j<i} MSE(residual_j))
evolve during stage-3 training, providing visual evidence for the "temporal
reweighting" mechanism.

Key insight:
  - With pretrain (stages 1+2 before 3): weights start relatively flat because
    the operator pretrain yields low residuals everywhere, so all weights ~ 1.
    Over training, mild temporal structure develops.
  - Without pretrain (stage 3 from scratch): residuals are large initially, so
    the exponential decay collapses weights for later time points to near zero.
    Only the earliest time bins receive meaningful gradient signal, distorting
    optimization.

Output: experiments/analysis/figures/fig_causal_weights.pdf
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from causal_pinn.models.hybrid.causal_gate import CausalGate
from causal_pinn.models.hybrid.hybrid_model import (
    HybridConfig,
    HybridModelWrapper,
    HybridPINNOperator,
)
from causal_pinn.models.operators.fno import FNO
from causal_pinn.models.pinn.base import PINNBase
from causal_pinn.physics.pde import Heat1D
from causal_pinn.training.trainer import HybridTrainer
from causal_pinn.utils.config import TrainingConfig
from causal_pinn.utils.random import set_seed

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "experiments" / "analysis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot style (matches other Paper C figures)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Snapshot epochs during stage 3 at which we record causal weights
SNAPSHOT_EPOCHS = [0, 50, 100, 150, 200]

# Causal weight epsilon for evaluation — must be large enough to reveal
# temporal structure given the typical residual magnitudes.  The adaptive
# causal loss anneals from 0.01 to causal_epsilon (1.0) during training,
# but the *visualisation* epsilon should amplify the contrast so the
# with/without-pretrain difference is clearly visible.
EVAL_EPSILON = 100.0

# Colour map for snapshot lines
SNAPSHOT_CMAP = plt.cm.viridis  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Model creation (same architecture as other Paper C scripts)
# ---------------------------------------------------------------------------
def create_hybrid_model(pde: Heat1D) -> HybridModelWrapper:
    """Create a fresh hybrid model for Heat1D."""
    pinn = PINNBase(input_dim=2, output_dim=1, hidden_dims=[64, 64, 64, 64])
    fno = FNO(in_channels=2, out_channels=1, modes=8, width=32, num_layers=4, dim=1)
    gate = CausalGate(out_dim=1, hidden_dim=32, causal_strength=1.0)
    config = HybridConfig(use_operator=True, use_pinn=True, use_causal_gate=True)
    hybrid = HybridPINNOperator(
        pinn_network=pinn, operator=fno,
        causal_gate=gate, pde_problem=pde, config=config,
    )
    return HybridModelWrapper(hybrid)


# ---------------------------------------------------------------------------
# Causal weight computation on a fixed evaluation grid
# ---------------------------------------------------------------------------
def compute_causal_weights_on_grid(
    model: HybridModelWrapper,
    pde: Heat1D,
    device: str,
    epsilon: float = EVAL_EPSILON,
    n_time: int = 100,
    n_spatial: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute causal weights w_i for evenly-spaced time fractions.

    We evaluate the model on a (n_time x n_spatial) grid, compute the PDE
    residual at each point (requires grad for autograd derivatives),
    then bin residuals by time and compute the causal weight per bin.

    Returns:
        t_fractions: (n_time,) array in [0, 1]
        weights:     (n_time,) array of causal weights
    """
    model.eval()

    t_bounds, x_bounds = pde.domain_bounds
    t_vals = torch.linspace(t_bounds[0], t_bounds[1], n_time, device=device)
    x_vals = torch.linspace(x_bounds[0] + 1e-6, x_bounds[1] - 1e-6, n_spatial, device=device)

    # Build a grid: each time value paired with all spatial values
    t_grid = t_vals.repeat_interleave(n_spatial)
    x_grid = x_vals.repeat(n_time)

    # We need gradients for the PDE residual computation (autograd through the net)
    t_g = t_grid.clone().requires_grad_(True)
    x_g = x_grid.clone().requires_grad_(True)

    # Forward pass — must keep the computation graph for pde.residual
    inp = torch.stack([t_g, x_g], dim=1)
    u_pred = model(inp).squeeze(-1)

    # Compute PDE residual (needs autograd graph for u_t, u_xx)
    residual = pde.residual(u_pred, t_g, x_g)

    # Now compute per-time-bin MSE and causal weights (detach to numpy)
    residual_np = residual.detach().cpu().numpy()
    t_grid.detach().cpu().numpy()

    # Bin by time index (we used linspace so indices are deterministic)
    bin_mse = np.zeros(n_time)
    bin_count = np.full(n_time, float(n_spatial))
    for i in range(n_time):
        start = i * n_spatial
        end = (i + 1) * n_spatial
        bin_mse[i] = np.mean(residual_np[start:end] ** 2)

    # Causal weights matching compute_causal_weights_binned:
    #   cumulative[i] = sum_{j<i} (bin_mse[j] * count[j]) / sum_{j<i} count[j]
    # This is the weighted cumulative mean of earlier-bin MSEs.
    cumulative_mse = np.zeros(n_time)
    for i in range(1, n_time):
        total_count = np.sum(bin_count[:i])
        cumulative_mse[i] = np.sum(bin_mse[:i] * bin_count[:i]) / (total_count + 1e-8)

    weights = np.exp(-epsilon * cumulative_mse)
    t_fractions = np.linspace(0.0, 1.0, n_time)

    return t_fractions, weights


# ---------------------------------------------------------------------------
# Training with snapshots
# ---------------------------------------------------------------------------
def train_with_snapshots(
    pde: Heat1D,
    device: str,
    skip_stage1: bool,
    seed: int = 42,
    stage1_epochs: int = 200,
    stage2_epochs: int = 300,
    stage3_epochs: int = 200,
    snapshot_epochs: list[int] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Train a model and record causal weight snapshots at given stage-3 epochs.

    Returns:
        {epoch: (t_fractions, weights)} for each requested snapshot epoch.
    """
    if snapshot_epochs is None:
        snapshot_epochs = SNAPSHOT_EPOCHS

    set_seed(seed)

    model = create_hybrid_model(pde)

    config = TrainingConfig(
        stage1_epochs=stage1_epochs,
        stage1_lr=1e-3,
        stage2_epochs=stage2_epochs,
        stage2_lr=1e-4,
        stage3_epochs=0,  # we drive stage 3 manually
        stage3_lr=1e-5,
        causal_epsilon=1.0,
        skip_stage1=skip_stage1,
        skip_stage3=True,  # we run stage 3 ourselves below
        batch_size=256,
        num_collocation_points=4096,
        num_boundary_points=512,
        seed=seed,
        device=device,
    )

    trainer = HybridTrainer(model=model, pde_problem=pde, config=config, device=device)

    # Run stages 1 & 2 via the trainer
    print(f"  [{'skip' if skip_stage1 else 'run':>4}] Stage 1 ({stage1_epochs} epochs)")
    print(f"  [ run] Stage 2 ({stage2_epochs} epochs)")
    trainer.train()

    # --- Manual stage 3 with snapshots ---
    print(f"  [ run] Stage 3 ({stage3_epochs} epochs) with snapshots at {snapshot_epochs}")
    model = trainer.model
    model.train()

    from causal_pinn.physics.losses import AdaptiveCausalLoss

    causal_loss_fn = AdaptiveCausalLoss(
        pde,
        epsilon_init=0.01,
        epsilon_final=1.0,
        num_time_bins=100,
        weight=1.0,
        annealing_steps=stage3_epochs,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Collocation points for stage 3
    t_bounds, x_bounds = pde.domain_bounds
    t_colloc = (
        torch.rand(config.num_collocation_points, device=device)
        * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
    ).requires_grad_(True)
    x_colloc = (
        torch.rand(config.num_collocation_points, device=device)
        * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    ).requires_grad_(True)

    # Boundary / IC helper (reuse trainer internals)
    bc_fn = trainer._compute_boundary_loss

    # Data generation helper
    has_exact = True  # Heat1D always has exact solution

    snapshots: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    max_epoch = max(snapshot_epochs) + 1

    for epoch in range(max_epoch):
        # Record snapshot BEFORE training this epoch
        if epoch in snapshot_epochs:
            t_frac, w = compute_causal_weights_on_grid(model, pde, device)
            snapshots[epoch] = (t_frac, w)
            model.train()

        causal_loss_fn.step()
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=device)

        # Data loss
        if has_exact:
            t_data, x_data, u_data = trainer._generate_synthetic_data(config.batch_size)
            data_loss = trainer._compute_data_loss(t_data, x_data, u_data)
            total_loss = total_loss + data_loss

        # Causal physics loss
        causal_result = causal_loss_fn(model, t_colloc, x_colloc)
        total_loss = total_loss + causal_result["causal_loss"]

        # Boundary loss
        total_loss = total_loss + bc_fn()

        total_loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(
                f"    epoch {epoch:4d}  loss={total_loss.item():.4e}  "
                f"eps={causal_loss_fn.epsilon:.4f}"
            )

    return snapshots


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_weight_evolution(
    snapshots_with_pretrain: dict[int, tuple[np.ndarray, np.ndarray]],
    snapshots_without_pretrain: dict[int, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Create a 2-panel figure showing causal weight evolution."""
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4), sharey=True)

    panels = [
        (axes[0], snapshots_with_pretrain, "With Operator Pretrain"),
        (axes[1], snapshots_without_pretrain, "Without Operator Pretrain"),
    ]

    sorted_epochs = sorted(snapshots_with_pretrain.keys())
    norm = plt.Normalize(vmin=min(sorted_epochs), vmax=max(sorted_epochs))  # type: ignore[attr-defined]

    for ax, snapshots, title in panels:
        for epoch in sorted_epochs:
            t_frac, w = snapshots[epoch]
            color = SNAPSHOT_CMAP(norm(epoch))
            ax.plot(
                t_frac, w,
                color=color,
                linewidth=1.2,
                label=f"Epoch {epoch}",
                alpha=0.85,
            )
        ax.set_xlabel("Time fraction $t / T$")
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.10)

    axes[0].set_ylabel("Causal weight $w_i$")

    # Single legend for both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(sorted_epochs),
        fontsize=7.5,
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.png')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize causal weight evolution during stage-3 training"
    )
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override results output directory",
    )
    parser.add_argument(
        "--stage1-epochs", type=int, default=200, help="Stage 1 epochs"
    )
    parser.add_argument(
        "--stage2-epochs", type=int, default=300, help="Stage 2 epochs"
    )
    parser.add_argument(
        "--stage3-epochs", type=int, default=200, help="Stage 3 epochs"
    )
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    results_dir = Path(args.output_dir) if args.output_dir else (
        ROOT / "experiments" / "results" / "paper_c" / "causal_weight_evolution"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    pde = Heat1D(alpha=1.0, length=1.0, t_max=1.0)

    print("=" * 60)
    print("Paper C: Causal Weight Evolution (Heat1D)")
    print(f"Device: {device} | Seed: {args.seed}")
    print(f"Schedule: {args.stage1_epochs}/{args.stage2_epochs}/{args.stage3_epochs} epochs")
    print("=" * 60)

    t0 = time.time()

    # --- Condition 1: With operator pretrain ---
    print("\n[1/2] Training WITH operator pretrain ...")
    snapshots_with = train_with_snapshots(
        pde, device,
        skip_stage1=False,
        seed=args.seed,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
    )

    # --- Condition 2: Without operator pretrain ---
    print("\n[2/2] Training WITHOUT operator pretrain ...")
    snapshots_without = train_with_snapshots(
        pde, device,
        skip_stage1=True,
        seed=args.seed,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs,
    )

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")

    # --- Save raw data ---
    raw_data: dict[str, Any] = {
        "config": {
            "seed": args.seed,
            "device": device,
            "stage1_epochs": args.stage1_epochs,
            "stage2_epochs": args.stage2_epochs,
            "stage3_epochs": args.stage3_epochs,
            "eval_epsilon": EVAL_EPSILON,
            "snapshot_epochs": SNAPSHOT_EPOCHS,
            "pde": "Heat1D",
        },
        "with_pretrain": {},
        "without_pretrain": {},
    }
    for epoch in sorted(snapshots_with.keys()):
        t_frac, w = snapshots_with[epoch]
        raw_data["with_pretrain"][str(epoch)] = {
            "t_fraction": t_frac.tolist(),
            "weights": w.tolist(),
        }
    for epoch in sorted(snapshots_without.keys()):
        t_frac, w = snapshots_without[epoch]
        raw_data["without_pretrain"][str(epoch)] = {
            "t_fraction": t_frac.tolist(),
            "weights": w.tolist(),
        }

    data_path = results_dir / "causal_weight_evolution.json"
    with open(data_path, "w") as f:
        json.dump(raw_data, f, indent=2)
    print(f"Saved data: {data_path}")

    # --- Plot ---
    plot_weight_evolution(
        snapshots_with, snapshots_without,
        OUT_DIR / "fig_causal_weights.pdf",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
