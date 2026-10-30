#!/usr/bin/env python3
"""Generate combined 2x2 gradient flow figure for Paper C."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt

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

# 4 gradient flow PDEs in display order
GRADIENT_PDES = [
    ("gradient_flow_heat1d", "Heat 1D"),
    ("gradient_flow_advdiff1d", "Adv-Diff 1D"),
    ("gradient_flow_allencahn", "Allen-Cahn"),
    ("gradient_flow_kdv1d", "KdV 1D"),
]


def generate_combined_gradient_fig():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=False, sharey=True)
    axes_flat = axes.flatten()

    for idx, (dirname, display_name) in enumerate(GRADIENT_PDES):
        ax = axes_flat[idx]
        rfile = PAPER_C_RESULTS / dirname / "gradient_flow_results.json"
        if not rfile.exists():
            ax.set_title(f"{display_name} (no data)", fontsize=9)
            continue

        with open(rfile) as f:
            data = json.load(f)

        conditions = data["conditions"]

        for cond_key, color, label in [
            ("with_pretrain", "#1f77b4", "With pretrain"),
            ("without_pretrain", "#d62728", "Without pretrain"),
        ]:
            if cond_key not in conditions:
                continue
            probes = conditions[cond_key]["probes"]
            epochs = [p["epoch"] for p in probes]
            cosine_sim = [p["cosine_similarity"] for p in probes]
            ax.plot(epochs, cosine_sim, color=color, label=label, linewidth=1.2)

        ax.set_title(display_name, fontsize=9)
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(0, color="gray", linewidth=0.4, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if idx >= 2:
            ax.set_xlabel("Epoch")
        if idx % 2 == 0:
            ax.set_ylabel("Cosine similarity")

    # Single legend at top
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_DIR / "fig_gradient_combined.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    print("Generating combined gradient flow figure...")
    generate_combined_gradient_fig()
    print("Done.")
