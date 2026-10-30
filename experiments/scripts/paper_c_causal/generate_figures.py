#!/usr/bin/env python3
"""Generate all figures for Paper C (causal loss ablation) from experiment results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]  # /projects/PINN
PAPER_C_RESULTS = ROOT / "experiments" / "results" / "paper_c"
OUT_DIR = ROOT / "docs" / "paper_c_causal" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
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

PDE_DISPLAY = {
    "heat1d": "Heat 1D",
    "burgers1d": "Burgers 1D",
    "burgers1d_verylow": "Burgers (v. low $\\nu$)",
    "advdiff1d": "Adv-Diff 1D",
    "allencahn_sharp": "Allen-Cahn",
    "kdv1d": "KdV 1D",
    "reaction_diffusion_stiff": "React-Diff (stiff)",
    "reaction_diffusion1d_stiff": "React-Diff 1D (stiff)",
}


def load_paper_c_data():
    """Return {pde_name: {condition: {metric: {mean, std}}}}."""
    data = {}
    for ablation_dir in sorted(PAPER_C_RESULTS.iterdir()):
        rfile = ablation_dir / "results.json"
        if not rfile.exists():
            continue
        with open(rfile) as f:
            d = json.load(f)
        pde_name = d["pde"]
        data[pde_name] = d["summary"]
    return data


# ===================================================================
# Figure 1 — Stage ablation heatmap
# ===================================================================
def fig_stage_ablation_heatmap():
    data = load_paper_c_data()
    pde_names = list(data.keys())

    # Compute % improvement from causal loss in each pretrain condition
    # improvement = (no_causal - with_causal) / no_causal * 100
    conditions = [
        ("full_pipeline", "With Pretrain"),
        ("no_pretrain", "Without Pretrain"),
    ]

    matrix = np.full((len(pde_names), len(conditions)), np.nan)
    annotations = [[""]*len(conditions) for _ in range(len(pde_names))]

    for i, pde in enumerate(pde_names):
        for j, (prefix, _) in enumerate(conditions):
            no_causal_key = f"{prefix}_no_causal"
            with_causal_key = f"{prefix}_with_causal"
            if no_causal_key in data[pde] and with_causal_key in data[pde]:
                base = data[pde][no_causal_key]["pde_residual_norm"]["mean"]
                causal = data[pde][with_causal_key]["pde_residual_norm"]["mean"]
                if base > 0:
                    improvement = (base - causal) / base * 100
                    matrix[i, j] = improvement
                    annotations[i][j] = f"{improvement:+.1f}%"
                else:
                    annotations[i][j] = "N/A"
            else:
                annotations[i][j] = "N/A"

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Custom diverging colormap: red for degradation, white for zero, green for improvement
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    vmax = max(vmax, 1.0)  # minimum range

    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "red_white_green",
        ["#d62728", "#ffffff", "#2ca02c"],
        N=256,
    )

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(conditions)))
    ax.set_xticklabels([c[1] for c in conditions], fontsize=8)
    ax.set_yticks(np.arange(len(pde_names)))
    ax.set_yticklabels([PDE_DISPLAY.get(p, p) for p in pde_names], fontsize=8)

    # Annotate cells
    for i in range(len(pde_names)):
        for j in range(len(conditions)):
            val = matrix[i, j]
            text_color = "black" if abs(val) < vmax * 0.6 else "white"
            ax.text(j, i, annotations[i][j], ha="center", va="center",
                    fontsize=8, color=text_color, fontweight="bold")

    ax.set_title("PDE residual improvement (%) from causal loss", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Improvement (%)")

    fig.tight_layout()
    out = OUT_DIR / "fig_stage_ablation_heatmap.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===================================================================
# Figure 2 — Pretrain effect (paired bars)
# ===================================================================
def fig_pretrain_effect():
    data = load_paper_c_data()
    pde_names = list(data.keys())

    # Average residual over causal ON/OFF for each pretrain condition
    with_pretrain = []
    without_pretrain = []
    for pde in pde_names:
        s = data[pde]
        wp = np.mean([
            s["full_pipeline_no_causal"]["pde_residual_norm"]["mean"],
            s["full_pipeline_with_causal"]["pde_residual_norm"]["mean"],
        ])
        wop = np.mean([
            s["no_pretrain_no_causal"]["pde_residual_norm"]["mean"],
            s["no_pretrain_with_causal"]["pde_residual_norm"]["mean"],
        ])
        with_pretrain.append(wp)
        without_pretrain.append(wop)

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.arange(len(pde_names))
    bw = 0.35

    ax.bar(x - bw/2, with_pretrain, bw, color="#2ca02c", label="With pretrain",
           edgecolor="white", linewidth=0.3)
    ax.bar(x + bw/2, without_pretrain, bw, color="#d62728", label="Without pretrain",
           edgecolor="white", linewidth=0.3)

    ax.set_yscale("log")
    ax.set_ylabel("PDE Residual (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([PDE_DISPLAY.get(p, p) for p in pde_names],
                       rotation=35, ha="right", fontsize=7.5)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Stage 1 pretraining has dramatic impact on final residual", fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "fig_pretrain_effect.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===================================================================
# Figure 3 — Interaction plot
# ===================================================================
def fig_interaction_plot():
    data = load_paper_c_data()
    pde_names = list(data.keys())

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    colors = plt.cm.tab10(np.linspace(0, 1, len(pde_names)))

    for i, pde in enumerate(pde_names):
        s = data[pde]
        # Causal effect (%) for each pretrain condition
        # effect = (no_causal - with_causal) / no_causal * 100
        effects = []
        for prefix in ["full_pipeline", "no_pretrain"]:
            base = s[f"{prefix}_no_causal"]["pde_residual_norm"]["mean"]
            causal = s[f"{prefix}_with_causal"]["pde_residual_norm"]["mean"]
            effects.append((base - causal) / base * 100 if base > 0 else 0)

        ax.plot([0, 1], effects, "o-", color=colors[i], markersize=5,
                label=PDE_DISPLAY.get(pde, pde), linewidth=1.2)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["With Pretrain", "Without Pretrain"], fontsize=9)
    ax.set_ylabel("Causal loss effect (%)")
    ax.set_title("Causal loss interaction with pretraining", fontsize=10)
    ax.legend(fontsize=6.5, loc="best", frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "fig_interaction_plot.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===================================================================
# Figure 4 — Runtime tradeoff scatter
# ===================================================================
def fig_runtime_tradeoff():
    data = load_paper_c_data()
    pde_names = list(data.keys())

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    colors = plt.cm.tab10(np.linspace(0, 1, len(pde_names)))

    for i, pde in enumerate(pde_names):
        s = data[pde]
        # Use full_pipeline condition (with pretrain) for the main comparison
        base_res = s["full_pipeline_no_causal"]["pde_residual_norm"]["mean"]
        causal_res = s["full_pipeline_with_causal"]["pde_residual_norm"]["mean"]
        improvement = (base_res - causal_res) / base_res * 100

        base_time = s["full_pipeline_no_causal"]["training_time"]["mean"]
        causal_time = s["full_pipeline_with_causal"]["training_time"]["mean"]
        runtime_mult = causal_time / base_time if base_time > 0 else 1.0

        ax.scatter(improvement, runtime_mult, color=colors[i], s=60, zorder=5,
                   edgecolors="black", linewidth=0.4)
        ax.annotate(PDE_DISPLAY.get(pde, pde), (improvement, runtime_mult),
                    textcoords="offset points", xytext=(5, 5), fontsize=6.5)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--", label="Break-even runtime")
    ax.set_xlabel("Improvement from causal loss (%)")
    ax.set_ylabel("Runtime multiplier")
    ax.set_title("Runtime cost vs. accuracy gain from causal loss", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "fig_runtime_tradeoff.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating Paper C figures...")
    fig_stage_ablation_heatmap()
    fig_pretrain_effect()
    fig_interaction_plot()
    fig_runtime_tradeoff()
    print("Done.")
