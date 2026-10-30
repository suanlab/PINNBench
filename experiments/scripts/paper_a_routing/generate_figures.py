#!/usr/bin/env python3
"""Generate all figures for Paper A (routing) from experiment results."""

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
PAPER_A_RESULTS = ROOT / "experiments" / "results" / "paper_a"
EARLY_PRED_RES = ROOT / "experiments" / "results" / "rq_early_predictor_expanded" / "results.json"
EARLY_PRED_L2  = ROOT / "experiments" / "results" / "rq_early_predictor_l2" / "results.json"
OUT_DIR = ROOT / "docs" / "paper_a_routing" / "figures"
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

POLICY_COLORS = {
    "pinn_only": "#1f77b4",
    "hybrid_no_gate": "#ff7f0e",
    "hybrid_full": "#2ca02c",
    "hybrid_full_lambda0.1": "#d62728",
    "hybrid_full_lambda0.5": "#9467bd",
}

POLICY_LABELS = {
    "pinn_only": "PINN-only",
    "hybrid_no_gate": "Hybrid (no gate)",
    "hybrid_full": "Hybrid (full)",
    "hybrid_full_lambda0.1": r"Hybrid ($\lambda$=0.1)",
    "hybrid_full_lambda0.5": r"Hybrid ($\lambda$=0.5)",
}

PDE_DISPLAY = {
    "heat1d": "Heat 1D",
    "burgers1d": "Burgers 1D",
    "burgers1d_lownu": "Burgers (low $\\nu$)",
    "burgers1d_verylow": "Burgers (v. low $\\nu$)",
    "wave2d": "Wave 2D",
    "advdiff1d": "Adv-Diff 1D",
    "advdiff1d_highpe": "Adv-Diff (high Pe)",
    "kdv1d": "KdV 1D",
    "allencahn1d": "Allen-Cahn 1D",
    "allencahn1d_sharp": "Allen-Cahn (sharp)",
    "reaction_diffusion1d": "React-Diff 1D",
    "reaction_diffusion1d_stiff": "React-Diff (stiff)",
    "navier_stokes2d": "Navier-Stokes 2D",
}


def load_paper_a_data():
    """Return {pde: {policy: residual_mean}}."""
    data = {}
    for pde_dir in sorted(PAPER_A_RESULTS.iterdir()):
        rfile = pde_dir / "results.json"
        if not rfile.exists():
            continue
        with open(rfile) as f:
            d = json.load(f)
        if "pde" not in d or "aggregated_results" not in d:
            continue
        pde_name = d["pde"]
        agg = d["aggregated_results"]
        data[pde_name] = {}
        for policy, metrics in agg.items():
            data[pde_name][policy] = metrics["pde_residual_norm"]["mean"]
    return data


def load_early_predictor(path):
    with open(path) as f:
        return json.load(f)


# ===================================================================
# Figure 1 — No-dominance grouped bar chart
# ===================================================================
def fig_no_dominance():
    data = load_paper_a_data()
    pde_names = list(data.keys())
    all_policies = list(POLICY_COLORS.keys())
    # Only keep policies that actually appear
    policies = [p for p in all_policies if any(p in data[pde] for pde in pde_names)]
    n_pdes = len(pde_names)
    n_policies = len(policies)

    # Drawn at its printed size (full text width, 5.5in) so fonts render at
    # their nominal size; it was drawn at 10in and shrunk to ~45%, which put
    # the smallest labels near 5pt in print.
    fig, ax = plt.subplots(figsize=(5.5, 2.9))
    bar_width = 0.8 / n_policies
    x = np.arange(n_pdes)

    for i, policy in enumerate(policies):
        vals = [data[pde].get(policy, np.nan) for pde in pde_names]
        offset = (i - n_policies / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width * 0.9,
                      color=POLICY_COLORS[policy],
                      label=POLICY_LABELS.get(policy, policy),
                      edgecolor="white", linewidth=0.3)

    ax.set_yscale("log")
    ax.set_ylabel("PDE Residual (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([PDE_DISPLAY.get(p, p) for p in pde_names],
                       rotation=40, ha="right", fontsize=8)
    ax.legend(
        fontsize=8,
        ncol=min(n_policies, 3),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        borderaxespad=0.2,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT_DIR / "fig_no_dominance.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ===================================================================
# Figure 2 — Routing accuracy (residual vs L2)
# ===================================================================
def fig_routing_accuracy():
    res_data = load_early_predictor(EARLY_PRED_RES)
    l2_data  = load_early_predictor(EARLY_PRED_L2)

    # Selectors to display (mapped to short labels)
    selector_map = {
        "random_forest": "RF",
        "physics_final": "Physics (final)",
        "linear_regression": "Linear",
        "training_prior": "Training prior",
        "random_expected": "Random",
    }
    selectors = list(selector_map.keys())

    # Use window=10, strict grouping
    ws = "10"
    res_strict = res_data["windows"][ws]["routing"]["strict"]["selectors"]
    l2_strict  = l2_data["windows"][ws]["routing"]["strict"]["selectors"]

    res_accs = [res_strict[s]["selection_accuracy"] for s in selectors]
    l2_accs  = [l2_strict[s]["selection_accuracy"] for s in selectors]
    labels   = [selector_map[s] for s in selectors]

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharey=True)
    x = np.arange(len(selectors))
    bar_w = 0.55

    for ax, accs, title in zip(axes, [res_accs, l2_accs],
                                ["Residual routing (strict)", "L2 routing (strict)"], strict=False):
        colors = ["#2ca02c" if a >= 0.7 else "#ff7f0e" if a >= 0.4 else "#d62728"
                   for a in accs]
        ax.bar(x, accs, bar_w, color=colors, edgecolor="white", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # annotate values
        for xi, a in zip(x, accs, strict=False):
            ax.text(xi, a + 0.03, f"{a:.0%}", ha="center", fontsize=7)

    axes[0].set_ylabel("Selection accuracy")
    fig.tight_layout()
    out = OUT_DIR / "fig_routing_accuracy.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===================================================================
# Figure 3 — Feature importance (residual vs L2)
# ===================================================================
def fig_feature_importance():
    res_data = load_early_predictor(EARLY_PRED_RES)
    l2_data  = load_early_predictor(EARLY_PRED_L2)

    ws = "10"
    fi_res = res_data["windows"][ws]["feature_importance"]
    fi_l2  = l2_data["windows"][ws]["feature_importance"]

    # Top-10 for each
    top_res = sorted(fi_res.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top_l2  = sorted(fi_l2.items(), key=lambda kv: kv[1], reverse=True)[:10]

    def _pretty(name):
        return name.replace("_", " ").replace("physics loss", "phys. loss") \
                    .replace("total", "total loss").replace("bc loss", "BC loss") \
                    .replace("data loss", "data loss")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))

    for ax, top, title, color in zip(
        axes, [top_res, top_l2],
        ["Residual routing", "L2 routing"],
        ["#1f77b4", "#ff7f0e"], strict=False
    ):
        names = [_pretty(n) for n, _ in reversed(top)]
        vals  = [v for _, v in reversed(top)]
        y = np.arange(len(names))
        ax.barh(y, vals, color=color, edgecolor="white", linewidth=0.3, height=0.65)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel("Importance")
        ax.set_title(title, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUT_DIR / "fig_feature_importance.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===================================================================
# Figure 4 — Budget savings vs probe window
# ===================================================================
def fig_budget_savings():
    res_data = load_early_predictor(EARLY_PRED_RES)
    l2_data  = load_early_predictor(EARLY_PRED_L2)

    windows = [10, 20, 50, 100]
    savings_res = []
    savings_l2  = []

    for ws in windows:
        ws_str = str(ws)
        savings_res.append(
            res_data["windows"][ws_str]["routing"]["strict"]["selectors"]["random_forest"]["mean_budget_savings"]
        )
        savings_l2.append(
            l2_data["windows"][ws_str]["routing"]["strict"]["selectors"]["random_forest"]["mean_budget_savings"]
        )

    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.plot(windows, savings_res, "o-", color="#1f77b4", label="Residual routing", markersize=5)
    ax.plot(windows, savings_l2,  "s--", color="#ff7f0e", label="L2 routing", markersize=5)
    ax.set_xlabel("Probe window (epochs)")
    ax.set_ylabel("Budget savings")
    ax.set_xticks(windows)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Budget savings vs. probe window size", fontsize=10)

    fig.tight_layout()
    out = OUT_DIR / "fig_budget_savings.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--only", choices=["no_dominance"], default=None)
    args = ap.parse_args()
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.only == "no_dominance":
        fig_no_dominance()
        raise SystemExit(0)
    print("Generating Paper A figures...")
    fig_no_dominance()
    fig_routing_accuracy()
    fig_feature_importance()
    fig_budget_savings()
    print("Done.")
