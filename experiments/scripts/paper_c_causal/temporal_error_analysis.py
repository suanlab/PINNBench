#!/usr/bin/env python3
"""Paper C: Temporal error distribution analysis.

Analyzes WHERE in the time domain causal loss helps or hurts.

Hypothesis: causal loss should reduce late-time errors (because it
upweights early-time training).  Without pretrain, this reweighting
may hurt early-time accuracy.

For each PDE with an exact solution, trains 4 models matching the 2x2
ablation design (skip_stage1 x skip_stage3), then evaluates the
prediction error at different time slices to compare temporal error
profiles across conditions.
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
from causal_pinn.physics.pde import (
    AdvectionDiffusion1D,
    AllenCahn1D,
    Heat1D,
    KdV1D,
)
from causal_pinn.training.trainer import HybridTrainer
from causal_pinn.utils.config import TrainingConfig
from causal_pinn.utils.random import set_seed

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# PDE registry — only PDEs with exact solutions
# (pde_class, kwargs, input_dim)
# ---------------------------------------------------------------------------
PDE_REGISTRY: dict[str, tuple[type, dict[str, Any], int]] = {
    "heat1d": (Heat1D, {"alpha": 1.0, "length": 1.0, "t_max": 1.0}, 2),
    "advdiff1d": (
        AdvectionDiffusion1D,
        {"velocity": 1.0, "alpha": 0.1, "length": 1.0, "t_max": 1.0},
        2,
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
        2,
    ),
    "allencahn1d_sharp": (
        AllenCahn1D,
        {"epsilon": 0.001, "length": 2.0, "t_max": 0.5, "front_center": 1.0},
        2,
    ),
}

# 2x2 conditions: (skip_stage1, skip_stage3, label)
CONDITIONS = [
    (False, True, "full_pipeline_no_causal"),
    (False, False, "full_pipeline_with_causal"),
    (True, True, "no_pretrain_no_causal"),
    (True, False, "no_pretrain_with_causal"),
]

# Readable labels for plotting
CONDITION_LABELS = {
    "full_pipeline_no_causal": "Pretrain, no causal",
    "full_pipeline_with_causal": "Pretrain + causal",
    "no_pretrain_no_causal": "No pretrain, no causal",
    "no_pretrain_with_causal": "No pretrain + causal",
}

CONDITION_STYLES = {
    "full_pipeline_no_causal": {"color": "#1f77b4", "ls": "-", "marker": "o"},
    "full_pipeline_with_causal": {"color": "#ff7f0e", "ls": "-", "marker": "s"},
    "no_pretrain_no_causal": {"color": "#2ca02c", "ls": "--", "marker": "^"},
    "no_pretrain_with_causal": {"color": "#d62728", "ls": "--", "marker": "D"},
}

PDE_DISPLAY = {
    "heat1d": "Heat 1D",
    "advdiff1d": "Advection-Diffusion 1D",
    "kdv1d": "KdV 1D (soliton)",
    "allencahn1d_sharp": "Allen-Cahn 1D (sharp)",
}


# ---------------------------------------------------------------------------
# Model creation (same architecture as stage_ablation.py)
# ---------------------------------------------------------------------------
def create_hybrid_model(pde: Any, input_dim: int) -> HybridModelWrapper:
    """Create a fresh hybrid model for a given PDE."""
    pinn = PINNBase(
        input_dim=input_dim, output_dim=1, hidden_dims=[64, 64, 64, 64],
    )
    fno_dim = 2 if input_dim == 3 else 1
    fno = FNO(
        in_channels=input_dim, out_channels=1,
        modes=8, width=32, num_layers=4, dim=fno_dim,
    )
    gate = CausalGate(out_dim=1, hidden_dim=32, causal_strength=1.0)
    config = HybridConfig(
        use_operator=True, use_pinn=True, use_causal_gate=True,
    )
    hybrid = HybridPINNOperator(
        pinn_network=pinn, operator=fno,
        causal_gate=gate, pde_problem=pde, config=config,
    )
    return HybridModelWrapper(hybrid)


# ---------------------------------------------------------------------------
# Temporal error evaluation
# ---------------------------------------------------------------------------
def evaluate_temporal_error(
    model: HybridModelWrapper,
    pde: Any,
    device: str,
    n_time_slices: int = 10,
    n_spatial: int = 200,
) -> list[dict[str, float]] | None:
    """Compute relative L2 error at each time slice.

    Returns a list of dicts with keys ``t_fraction``, ``t_value``,
    ``rel_l2`` — one entry per time slice.  Returns ``None`` if the
    PDE has no exact solution.
    """
    model.eval()

    # Determine spatial bounds
    if hasattr(pde, "domain_bounds"):
        x_lo, x_hi = pde.domain_bounds[1]
    else:
        x_lo, x_hi = 0.0, pde.length

    t_max = pde.t_max
    errors: list[dict[str, float]] = []

    for i in range(n_time_slices):
        t_val = (i + 1) / n_time_slices * t_max

        x = torch.linspace(x_lo, x_hi, n_spatial, device=device)
        t = torch.full((n_spatial,), t_val, device=device)

        # Exact solution (detached — no grad needed)
        u_exact = pde.exact_solution(t, x)
        if u_exact is None:
            return None
        u_exact = u_exact.detach()

        # Model prediction (no grad)
        with torch.no_grad():
            inp = torch.stack([t, x], dim=1)  # (n_spatial, 2)
            u_pred = model(inp).squeeze()

        rel_err = torch.norm(u_pred - u_exact) / (torch.norm(u_exact) + 1e-8)
        errors.append({
            "t_fraction": (i + 1) / n_time_slices,
            "t_value": float(t_val),
            "rel_l2": rel_err.item(),
        })

    return errors


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------
def train_and_evaluate(
    pde_name: str,
    pde: Any,
    input_dim: int,
    condition_label: str,
    skip_stage1: bool,
    skip_stage3: bool,
    seed: int,
    device: str,
    n_time_slices: int,
    n_spatial: int,
) -> dict[str, Any]:
    """Train one model and evaluate its temporal error profile."""
    set_seed(seed)
    model = create_hybrid_model(pde, input_dim)

    config = TrainingConfig(
        stage1_epochs=200,
        stage1_lr=1e-3,
        stage2_epochs=500,
        stage2_lr=1e-4,
        stage3_epochs=200,
        stage3_lr=1e-5,
        batch_size=256,
        num_collocation_points=4000,
        num_boundary_points=500,
        seed=seed,
        device=device,
        lambda_physics=1.0,
        lambda_data=1.0,
        causal_epsilon=1.0,
        skip_stage1=skip_stage1,
        skip_stage3=skip_stage3,
    )

    trainer = HybridTrainer(
        model=model, pde_problem=pde, config=config, device=device,
    )

    start = time.time()
    history = trainer.train()
    training_time = time.time() - start

    temporal_errors = evaluate_temporal_error(
        model, pde, device,
        n_time_slices=n_time_slices,
        n_spatial=n_spatial,
    )

    return {
        "pde": pde_name,
        "condition": condition_label,
        "skip_stage1": skip_stage1,
        "skip_stage3": skip_stage3,
        "seed": seed,
        "training_time": training_time,
        "training_history": history,
        "temporal_errors": temporal_errors,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_temporal_errors(
    all_results: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Generate a multi-panel figure of temporal error profiles.

    One panel per PDE, each panel has 4 lines (one per condition).
    """
    pde_names = [k for k in PDE_REGISTRY if k in all_results]
    n_panels = len(pde_names)
    if n_panels == 0:
        print("No results to plot.")
        return

    ncols = min(n_panels, 2)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4.5 * nrows),
        squeeze=False,
    )

    for idx, pde_name in enumerate(pde_names):
        ax = axes[idx // ncols][idx % ncols]
        results_for_pde = all_results[pde_name]

        for _skip_s1, _skip_s3, label in CONDITIONS:
            cond_results = [
                r for r in results_for_pde
                if r["condition"] == label and r["temporal_errors"] is not None
            ]
            if not cond_results:
                continue

            # Average across seeds (here just 1 seed, but handles multiple)
            t_fracs = [e["t_fraction"] for e in cond_results[0]["temporal_errors"]]
            all_errors = np.array([
                [e["rel_l2"] for e in r["temporal_errors"]]
                for r in cond_results
            ])
            mean_err = np.mean(all_errors, axis=0)
            std_err = np.std(all_errors, axis=0)

            style = CONDITION_STYLES[label]
            ax.plot(
                t_fracs, mean_err,
                color=style["color"], ls=style["ls"],
                marker=style["marker"], markersize=5,
                label=CONDITION_LABELS[label], linewidth=1.5,
            )
            if all_errors.shape[0] > 1:
                ax.fill_between(
                    t_fracs,
                    mean_err - std_err,
                    mean_err + std_err,
                    alpha=0.15, color=style["color"],
                )

        ax.set_xlabel("Time fraction ($t / t_{\\mathrm{max}}$)")
        ax.set_ylabel("Relative $L^2$ error")
        ax.set_title(PDE_DISPLAY.get(pde_name, pde_name))
        ax.set_yscale("log")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(
        "Temporal Error Distribution: Causal Loss Effect Across Time",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    fig_path = output_dir / "fig_temporal_error.pdf"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    print(f"Saved figure: {fig_path}")

    png_path = output_dir / "fig_temporal_error.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    print(f"Saved figure: {png_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper C: Temporal error distribution analysis",
    )
    parser.add_argument(
        "--pdes", nargs="+",
        default=list(PDE_REGISTRY.keys()),
        choices=list(PDE_REGISTRY.keys()),
        help="PDEs to evaluate (default: all with exact solutions)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-time-slices", type=int, default=10)
    parser.add_argument("--n-spatial", type=int, default=200)
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/results/paper_c/temporal_error_analysis",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Paper C: Temporal Error Distribution Analysis")
    print("=" * 70)
    print(f"PDEs:         {args.pdes}")
    print(f"Seed:         {args.seed}")
    print(f"Time slices:  {args.n_time_slices}")
    print(f"Spatial pts:  {args.n_spatial}")
    print(f"Device:       {args.device}")
    print(f"Output:       {output_dir}")
    print(f"Conditions:   {[c[2] for c in CONDITIONS]}")
    total_runs = len(args.pdes) * len(CONDITIONS)
    print(f"Total runs:   {total_runs}")
    print("=" * 70)
    print()

    # Collect all results grouped by PDE
    all_results: dict[str, list[dict[str, Any]]] = {}

    for pde_name in args.pdes:
        pde_cls, pde_kwargs, input_dim = PDE_REGISTRY[pde_name]
        pde = pde_cls(**pde_kwargs)

        # Quick sanity check: does this PDE have an exact solution?
        t_test = torch.tensor([0.1])
        x_test = torch.tensor([0.5])
        if pde.exact_solution(t_test, x_test) is None:
            print(f"SKIP {pde_name}: no exact solution available")
            continue

        print(f"\n--- PDE: {pde_name} ---")
        pde_results: list[dict[str, Any]] = []

        for skip_s1, skip_s3, label in CONDITIONS:
            tag = f"  {pde_name}/{label}"
            print(f"{tag} ... ", end="", flush=True)

            try:
                result = train_and_evaluate(
                    pde_name, pde, input_dim, label,
                    skip_s1, skip_s3, args.seed, args.device,
                    args.n_time_slices, args.n_spatial,
                )
                errs = result["temporal_errors"]
                if errs is not None:
                    early = np.mean([e["rel_l2"] for e in errs[:3]])
                    late = np.mean([e["rel_l2"] for e in errs[-3:]])
                    print(
                        f"early_avg={early:.4e}  late_avg={late:.4e}  "
                        f"time={result['training_time']:.1f}s"
                    )
                else:
                    print("no exact solution")
                pde_results.append(result)
            except Exception as exc:
                print(f"FAILED: {exc}")
                pde_results.append({
                    "pde": pde_name,
                    "condition": label,
                    "skip_stage1": skip_s1,
                    "skip_stage3": skip_s3,
                    "seed": args.seed,
                    "training_time": 0.0,
                    "training_history": {},
                    "temporal_errors": None,
                    "error": str(exc),
                })

        all_results[pde_name] = pde_results

    # ------------------------------------------------------------------
    # Save raw results
    # ------------------------------------------------------------------
    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "experiment": "paper_c_temporal_error_analysis",
                "pdes": args.pdes,
                "seed": args.seed,
                "n_time_slices": args.n_time_slices,
                "n_spatial": args.n_spatial,
                "conditions": [c[2] for c in CONDITIONS],
                "results": {
                    pde_name: [
                        {k: v for k, v in r.items() if k != "training_history"}
                        for r in runs
                    ]
                    for pde_name, runs in all_results.items()
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved results: {results_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TEMPORAL ERROR SUMMARY  (early = avg t/t_max <= 0.3, late = avg t/t_max >= 0.8)")
    print("=" * 90)
    header = f"{'PDE':<22s} {'Condition':<30s} {'Early Avg':>12s} {'Late Avg':>12s} {'Late/Early':>12s}"  # noqa: E501
    print(header)
    print("-" * 90)

    for pde_name, runs in all_results.items():
        for r in runs:
            errs = r.get("temporal_errors")
            if errs is None:
                continue
            n = len(errs)
            # Early: first 30%, Late: last 30%
            early_cut = max(1, int(0.3 * n))
            late_cut = max(1, int(0.3 * n))
            early_avg = np.mean([e["rel_l2"] for e in errs[:early_cut]])
            late_avg = np.mean([e["rel_l2"] for e in errs[-late_cut:]])
            ratio = late_avg / (early_avg + 1e-12)
            print(
                f"{pde_name:<22s} {r['condition']:<30s} "
                f"{early_avg:>12.4e} {late_avg:>12.4e} {ratio:>12.2f}"
            )

    # ------------------------------------------------------------------
    # Generate figure
    # ------------------------------------------------------------------
    plot_temporal_errors(all_results, output_dir)
    print(f"\nDone. All outputs in {output_dir}")


if __name__ == "__main__":
    main()
