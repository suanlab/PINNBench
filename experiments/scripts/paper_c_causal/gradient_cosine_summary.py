"""Multi-PDE gradient cosine summary.

Aggregates per-PDE gradient-flow probe data from
``experiments/results/paper_c/gradient_flow_*`` and reports the min/max/mean
cosine similarity between physics-loss and causal-loss gradient directions
under each of the 2 pretraining conditions, across the 4 PDEs analysed
(Heat1D, AdvDiff1D, KdV1D, AllenCahn1D). Addresses Reviewer 1's concern
that the cosine ≈ 1.0 finding from Fig. 5 might not generalize beyond
Heat1D.

Usage:
    PYTHONPATH=src python experiments/scripts/paper_c_causal/gradient_cosine_summary.py
"""

from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

_PDES = ["heat1d", "advdiff1d", "kdv1d", "allencahn"]


def summarize() -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for pde in _PDES:
        pattern = f"experiments/results/paper_c/gradient_flow_{pde}/gradient_flow_results.json"
        files = glob.glob(pattern)
        if not files:
            continue
        data = json.loads(Path(files[0]).read_text())
        per_cond: dict[str, dict[str, float]] = {}
        for cond, payload in data.get("conditions", {}).items():
            probes = payload.get("probes", [])
            if not probes:
                continue
            cos = [p["cosine_similarity"] for p in probes]
            per_cond[cond] = {
                "min": min(cos),
                "max": max(cos),
                "mean": statistics.mean(cos),
                "n_probes": len(probes),
            }
        out[pde] = per_cond
    return out


def main() -> None:
    summary = summarize()
    print(
        f"{'PDE':<14s} {'condition':<22s} {'cos min':>9s} {'cos max':>9s} {'cos mean':>9s}"
    )
    for pde, conds in summary.items():
        for cond, stats in conds.items():
            print(
                f"  {pde:<12s} {cond:<22s} "
                f"{stats['min']:>9.4f} {stats['max']:>9.4f} {stats['mean']:>9.4f}"
            )


if __name__ == "__main__":
    main()
