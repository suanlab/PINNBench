#!/usr/bin/env python3
"""Statistics for the confound controls of the 2x2 stage ablation.

The 2x2 design (``stage_ablation.py``) compares arms that differ in more than
the named factor: the no-causal arms skip Stage 3 (200 fewer epochs), and the
no-pretraining arms skip Stage 1 (200 fewer epochs, no 1e-3 learning-rate
phase).  The control arms (``stage_ablation.py --condition-set controls``)
remove those confounds:

* E1 reweighting   causal Stage 3 vs uniform-weight Stage 3 (same epochs),
                   with and without pretraining.
* E1 extra training uniform-weight Stage 3 vs no Stage 3,
                   with and without pretraining.
* E2 pretraining   Stage 1 + 500 Stage-2 epochs vs no Stage 1 + 700 Stage-2
                   epochs (same total epochs), without and with Stage 3.

Each test pairs runs by seed on PDE residual and reports
delta% = (reference - treatment) / reference * 100, so positive means the
treatment (causal weighting / extra Stage 3 / pretraining) lowers the residual,
with an exact Wilcoxon test, a paired bootstrap CI and BH-FDR step-up over all
tests of this script.

Usage:
    PYTHONPATH=src python experiments/scripts/paper_c_causal/control_statistics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_statistics import (  # noqa: E402
    apply_bh_fdr,
    bootstrap_mean_ci,
    paired_cohens_d,
    wilcoxon_exact,
)

ROOT = Path(__file__).resolve().parents[3]
PAPER_C = ROOT / "experiments" / "results" / "paper_c"
OUT = PAPER_C / "control_statistics"
METRIC = "pde_residual_norm"

# PDE key -> directory of the original 2x2 ablation
ORIGINAL = {
    "heat1d": "heat1d_ablation",
    "advdiff1d": "advdiff1d_ablation",
    "burgers1d": "burgers1d_ablation",
    "burgers1d_verylow": "burgers1d_verylow_ablation",
    "kdv1d": "kdv1d_ablation",
    "allencahn1d": "allencahn1d_standard_ablation",
    "allencahn1d_sharp": "allencahn_sharp_ablation",
    "reaction_diffusion1d_stiff": "reaction_diffusion_stiff_ablation",
    "navier_stokes2d": "navier_stokes2d_ablation",
}

# (family, label, reference condition, treatment condition)
COMPARISONS = [
    ("reweighting", "with pretraining",
     "full_pipeline_uniform_stage3", "full_pipeline_with_causal"),
    ("reweighting", "without pretraining",
     "no_pretrain_uniform_stage3", "no_pretrain_with_causal"),
    ("extra_training", "with pretraining",
     "full_pipeline_no_causal", "full_pipeline_uniform_stage3"),
    ("extra_training", "without pretraining",
     "no_pretrain_no_causal", "no_pretrain_uniform_stage3"),
    ("pretraining_matched", "no Stage 3",
     "no_pretrain_long_stage2_no_causal", "full_pipeline_no_causal"),
    ("pretraining_matched", "with causal Stage 3",
     "no_pretrain_long_stage2_with_causal", "full_pipeline_with_causal"),
]


def runs_by_condition(pde: str) -> dict[str, dict[int, float]]:
    """{condition: {seed: residual}} from the original ablation and the controls."""
    out: dict[str, dict[int, float]] = {}
    files = [PAPER_C / ORIGINAL[pde] / "results.json"]
    files += sorted((PAPER_C / "controls").glob(f"{pde}__*/results.json"))
    for f in files:
        for r in json.loads(f.read_text())["raw_results"]:
            if not r.get("success", True) or METRIC not in r.get("metrics", {}):
                continue
            out.setdefault(r["condition"], {})[int(r["seed"])] = float(r["metrics"][METRIC])
    return out


LABEL = {
    "heat1d": "Heat1D", "advdiff1d": "AdvDiff1D", "burgers1d": r"Burgers1D ($\nu{=}0.01$)",
    "burgers1d_verylow": r"Burgers1D ($\nu{=}10^{-4}$)", "kdv1d": "KdV1D",
    "allencahn1d": r"AllenCahn1D ($\varepsilon{=}0.01$)",
    "allencahn1d_sharp": r"AllenCahn1D ($\varepsilon{=}0.001$)",
    "reaction_diffusion1d_stiff": "ReactDiff1D (stiff)", "navier_stokes2d": "NavierStokes2D",
}
COLUMNS = [("reweighting", "with pretraining"), ("reweighting", "without pretraining"),
           ("extra_training", "with pretraining"), ("extra_training", "without pretraining"),
           ("pretraining_matched", "no Stage 3")]


def latex_rows(results: list[dict]) -> str:
    """Rows of the appendix control table: delta% with a dagger when BH-significant."""
    def cell(r: dict) -> str:
        mark = r"$^{\ast}$" if r["bh_fdr_significant"] else ""
        return f"${r['delta_pct']:+.1f}${mark}"
    by = {(r["pde"], r["family"], r["comparison"]): r for r in results}
    lines = []
    for pde, label in LABEL.items():
        lines.append(f"{label:<38} & " + " & ".join(cell(by[(pde, f, c)]) for f, c in COLUMNS)
                     + r" \\")
    means = []
    for f, c in COLUMNS:
        rows = [by[(p, f, c)] for p in LABEL]
        means.append(f"${sum(r['delta_pct'] for r in rows) / len(rows):+.1f}$")
    lines.append(r"\midrule")
    lines.append(r"\textbf{Mean} & " + " & ".join(means) + r" \\")
    return "\n".join(lines)


def main() -> None:
    if "--latex" in sys.argv:
        print(latex_rows(json.loads((OUT / "results.json").read_text())))
        return
    results = []
    for pde in ORIGINAL:
        runs = runs_by_condition(pde)
        for family, label, ref, trt in COMPARISONS:
            if ref not in runs or trt not in runs:
                print(f"SKIP {pde} {family} {label}: missing {ref if ref not in runs else trt}")
                continue
            seeds = sorted(set(runs[ref]) & set(runs[trt]))
            a = np.array([runs[ref][s] for s in seeds])
            b = np.array([runs[trt][s] for s in seeds])
            diff = a - b
            _w, p, w_plus, n = wilcoxon_exact(diff)
            lo, hi = bootstrap_mean_ci(diff)
            results.append({
                "pde": pde, "family": family, "comparison": label,
                "reference": ref, "treatment": trt, "n_pairs": len(seeds),
                "mean_reference": float(a.mean()), "mean_treatment": float(b.mean()),
                "delta_pct": float(100 * (a.mean() - b.mean()) / a.mean()),
                "delta_pct_ci": [float(100 * lo / a.mean()), float(100 * hi / a.mean())],
                "cohens_d": paired_cohens_d(diff), "wilcoxon_p": float(p),
                "wilcoxon_w_plus": float(w_plus),
            })
    apply_bh_fdr(results)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {OUT / 'results.json'} ({len(results)} tests)")
    for family, label, _r, _t in COMPARISONS:
        rows = [r for r in results if r["family"] == family and r["comparison"] == label]
        if not rows:
            continue
        mean = sum(r["delta_pct"] for r in rows) / len(rows)
        pos = sum(r["delta_pct"] > 0 for r in rows)
        sig_pos = sum(r["bh_fdr_significant"] and r["delta_pct"] > 0 for r in rows)
        sig_neg = sum(r["bh_fdr_significant"] and r["delta_pct"] < 0 for r in rows)
        print(f"{family:<20} {label:<22} mean {mean:+7.2f}%  positive {pos}/{len(rows)}  "
              f"sig+ {sig_pos}  sig- {sig_neg}")


if __name__ == "__main__":
    main()
