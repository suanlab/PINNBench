#!/usr/bin/env python3
"""Print the LaTeX rows of the 2x2 stage-ablation table from released statistics.

Reads ``experiments/results/paper_c/ablation_statistics/results.json`` (written by
``ablation_statistics.py``) and prints, for each PDE, the change in PDE residual
from adding causal loss with and without operator pretraining:

    delta% = (no_causal - with_causal) / no_causal * 100   (positive = causal helps)

with the paired-bootstrap 95% CI rescaled to percent and a verdict from the
BH-FDR-adjusted Wilcoxon p-value (27 tests: 9 PDEs x 3 comparisons).

The table used to be typed in by hand.  It drifted: the without-pretraining mean
read -4.2% where the rows average -2.4%, the intervals did not match any
computation, and two verdicts disagreed with the tests.

Usage:
    python experiments/scripts/paper_c_causal/causal_ablation_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STATS = ROOT / "experiments" / "results" / "paper_c" / "ablation_statistics" / "results.json"

ROWS = [
    ("heat1d", "Heat1D"),
    ("advdiff1d", "AdvDiff1D"),
    ("burgers1d", r"Burgers1D ($\nu{=}0.01$)"),
    ("burgers1d_verylow", r"Burgers1D ($\nu{=}10^{-4}$)"),
    ("kdv1d", "KdV1D"),
    ("allencahn1d", r"AllenCahn1D ($\varepsilon{=}0.01$)"),
    ("allencahn1d_sharp", r"AllenCahn1D ($\varepsilon{=}0.001$)"),
    ("reaction_diffusion1d_stiff", r"ReactDiff1D (stiff)"),
    ("navier_stokes2d", "NavierStokes2D"),
]
CONDITIONS = ("Causal WITH pretrain", "Causal WITHOUT pretrain")


def cell(r: dict) -> tuple[float, str, str]:
    base = r["mean_a"]
    d = 100 * r["mean_diff"] / base
    lo, hi = (100 * v / base for v in r["bootstrap_95ci"])
    if not r["bh_fdr_significant"]:
        verdict = r"$\approx$"
    else:
        verdict = r"$\checkmark$" if d > 0 else r"$\times$"
    return d, f"${d:+.1f}$ $[{lo:+.1f}, {hi:+.1f}]$", verdict


def summarize() -> dict[str, dict[str, float | int]]:
    """Mean delta% and verdict counts per pretraining condition."""
    stats = json.loads(STATS.read_text())
    by = {(r["pde"], r["comparison"]): r for r in stats}
    out: dict[str, dict[str, float | int]] = {}
    for cond in CONDITIONS:
        ds = [cell(by[(k, cond)]) for k, _ in ROWS]
        out[cond] = {
            "mean": sum(d for d, _, _ in ds) / len(ds),
            "n_positive": sum(d > 0 for d, _, _ in ds),
            "n_negative": sum(d < 0 for d, _, _ in ds),
            "n_sig_better": sum(v == r"$\checkmark$" for _, _, v in ds),
            "n_sig_worse": sum(v == r"$\times$" for _, _, v in ds),
            "n_tests": len(stats),
        }
    return out


def main() -> None:
    stats = json.loads(STATS.read_text())
    by = {(r["pde"], r["comparison"]): r for r in stats}
    for key, label in ROWS:
        cells = [cell(by[(key, c)]) for c in CONDITIONS]
        print(f"{label:<38} & " + " & ".join(f"{t} & {v}" for _, t, v in cells) + r" \\")
    s = summarize()
    w, wo = s[CONDITIONS[0]], s[CONDITIONS[1]]
    print(r"\midrule")
    print(
        rf"\textbf{{Mean}} & ${w['mean']:+.1f}$ & {w['n_sig_better']}/9 $\checkmark$"
        rf" & ${wo['mean']:+.1f}$ & {wo['n_sig_worse']}/9 $\times$, "
        rf"{wo['n_sig_better']}/9 $\checkmark$ \\"
    )
    for cond, v in s.items():
        print(f"% {cond}: {v}")


if __name__ == "__main__":
    main()
