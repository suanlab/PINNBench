"""Generate the accuracy-regret dissociation figure for Paper A.

Shows that physics_final achieves similar accuracy to RF for L2 routing
but has catastrophically worse regret (77x), revealing that accuracy alone
is a misleading metric for routing evaluation.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = ROOT / "experiments" / "results" / "paper_a"
FOLDS = ROOT / "experiments" / "results" / "paper_combined" / "routing_evaluation" / "results.json"

# ---------------------------------------------------------------------------
# Data -- recomputed from the released artifacts on every run.
#
# These values used to be typed in by hand, and drifted: after the Gradient
# Boosting residual accuracy was corrected in the table (53.8% -> 61.5%) the
# figure kept 54% and a regret of 0.0056, while the folds give 0.0031.
# ---------------------------------------------------------------------------


def _policy_means(metric: str) -> dict[tuple[str, str], float]:
    """Mean metric per (PDE, policy) over the runs the router was trained on."""
    runs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for d in sorted(RUNS_DIR.iterdir()):
        f = d / "results.json"
        if not d.is_dir() or d.name in {"bandit_evaluation", "routing_evaluation_full"}:
            continue
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        pde = data.get("pde", d.name)
        for r in data.get("raw_results", []):
            if not r.get("success", True):
                continue
            if len(r.get("training_history", {}).get("stage2_losses", [])) < 50:
                continue
            v = r.get("metrics", {}).get(metric)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            runs[(pde, r["model_name"])].append(float(v))
    return {k: sum(v) / len(v) for k, v in runs.items()}


def routing_scores(metric: str) -> dict[str, tuple[float, float]]:
    """(top-1 accuracy, mean regret) per selector; 'Random' is the expectation."""
    means = _policy_means(metric)
    folds = json.loads(FOLDS.read_text())["results"][metric]
    out: dict[str, tuple[float, float]] = {}
    for name, rec in folds.items():
        regrets = [means[(f["held_out_pde"], f["predicted_best"])]
                   - means[(f["held_out_pde"], f["actual_best"])] for f in rec["fold_details"]]
        out[name] = (rec["routing_accuracy"], sum(regrets) / len(regrets))
    accs, regs = [], []
    for f in next(iter(folds.values()))["fold_details"]:
        pols = [m for (p, _), m in means.items() if p == f["held_out_pde"]]
        accs.append(1 / len(pols))
        regs.append(sum(pols) / len(pols) - min(pols))
    out["Random"] = (sum(accs) / len(accs), sum(regs) / len(regs))
    return out


ORDER = [("physics_final", "physics_final"), ("RandomForest", "RF"),
         ("GradientBoosting", "GB"), ("Ridge", "Ridge"), ("Random", "Random")]
_res = routing_scores("pde_residual_norm")
_l2 = routing_scores("relative_l2_error")
selectors = [label for _, label in ORDER]
residual_acc = [_res[k][0] for k, _ in ORDER]
residual_reg = [_res[k][1] for k, _ in ORDER]
l2_acc = [_l2[k][0] for k, _ in ORDER]
l2_reg = [_l2[k][1] for k, _ in ORDER]
for (_k, label), ra, rr, la, lr in zip(ORDER, residual_acc, residual_reg, l2_acc, l2_reg,
                                     strict=True):
    print(f"{label:<14} residual {ra:6.1%} {rr:.4f} | L2 {la:6.1%} {lr:.4f}")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

color_res = "#3574b2"   # blue for residual
color_l2 = "#e8792b"    # orange for L2
highlight_edge = "#c22020"  # red for the key bar

fig, (ax_acc, ax_reg) = plt.subplots(1, 2, figsize=(7.2, 2.95))

x = np.arange(len(selectors))
w = 0.35

# ---------------------------------------------------------------------------
# Left panel: Accuracy
# ---------------------------------------------------------------------------
bars_res_a = ax_acc.bar(x - w / 2, [a * 100 for a in residual_acc], w,
                        label="Residual", color=color_res, edgecolor="white",
                        linewidth=0.5)
bars_l2_a = ax_acc.bar(x + w / 2, [a * 100 for a in l2_acc], w,
                       label="L2", color=color_l2, edgecolor="white",
                       linewidth=0.5)

ax_acc.set_ylabel("Accuracy (%)")
# Subpanel title placed BELOW the axes (as requested).
ax_acc.set_xlabel("(a)  Routing Accuracy", fontsize=11, fontweight="bold", labelpad=10)
ax_acc.set_xticks(x)
ax_acc.set_xticklabels(selectors, rotation=25, ha="right")
ax_acc.set_ylim(0, 100)
ax_acc.legend(frameon=False, fontsize=9)

# Remove grid lines and top/right spines
for sp in ("top", "right"):
    ax_acc.spines[sp].set_visible(False)

# Annotate the key observation: physics_final and RF have same L2 accuracy
ax_acc.annotate("same accuracy",
                xy=(0.175, l2_acc[0] * 100), xytext=(1.1, 85),
                fontsize=8.5, color="#555555", style="italic",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7),
                ha="center")

# ---------------------------------------------------------------------------
# Right panel: Regret (log scale)
# ---------------------------------------------------------------------------
bars_res_r = ax_reg.bar(x - w / 2, residual_reg, w,
                        label="Residual", color=color_res, edgecolor="white",
                        linewidth=0.5)
bars_l2_r = ax_reg.bar(x + w / 2, l2_reg, w,
                       label="L2", color=color_l2, edgecolor="white",
                       linewidth=0.5)

ax_reg.set_yscale("log")
ax_reg.set_ylabel("Regret (log scale)")
# Subpanel title placed BELOW the axes (as requested).
ax_reg.set_xlabel("(b)  Routing Regret", fontsize=11, fontweight="bold", labelpad=10)
ax_reg.set_xticks(x)
ax_reg.set_xticklabels(selectors, rotation=25, ha="right")
ax_reg.legend(frameon=False, fontsize=9)

for sp in ("top", "right"):
    ax_reg.spines[sp].set_visible(False)

# Highlight physics_final L2 regret bar with red border
bars_l2_r[0].set_edgecolor(highlight_edge)
bars_l2_r[0].set_linewidth(2.0)

# Annotate the 77x multiplier
# Ratio of the values as printed in Table 4 (four decimals), so the figure and
# the table agree on 77x; the unrounded ratio is 77.6x.
ratio = round(l2_reg[0], 4) / round(l2_reg[1], 4)
ax_reg.annotate(
    f"{ratio:.0f}$\\times$ worse",
    xy=(0 + w / 2, l2_reg[0]),
    xytext=(1.6, 0.12),
    fontsize=10, fontweight="bold", color=highlight_edge,
    arrowprops=dict(arrowstyle="-|>", color=highlight_edge, lw=1.2),
    ha="center", va="bottom",
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.tight_layout(w_pad=3)

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "paper_a_routing" / "figures")
out_dir = ap.parse_args().out_dir
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "fig_dissociation.pdf"
fig.savefig(out_path, bbox_inches="tight", dpi=300)
print(f"Saved: {out_path}")

# Also save PNG for quick preview
fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
print(f"Saved: {out_path.with_suffix('.png')}")
