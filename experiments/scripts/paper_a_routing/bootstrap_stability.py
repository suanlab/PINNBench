"""Benchmark-composition bootstrap for routing-method stability.

Resamples 9 of the 13 PDE leave-one-out folds with replacement (1000 reps,
seed=42) and reports the mean accuracy, standard deviation, and 95%
percentile interval per routing selector. Addresses Reviewer 2's concern
that routing rankings could be artifacts of which 13 PDEs were chosen.

Source artifact:
    experiments/results/paper_combined/routing_evaluation/results.json

Usage:
    PYTHONPATH=src python experiments/scripts/paper_a_routing/bootstrap_stability.py
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

_RESULTS_PATH = Path(
    "experiments/results/paper_combined/routing_evaluation/results.json"
)


def bootstrap_routing(
    n_reps: int = 1000, n_sample: int = 9, seed: int = 42
) -> dict[str, dict[str, float]]:
    random.seed(seed)
    data = json.loads(_RESULTS_PATH.read_text())
    out: dict[str, dict[str, float]] = {}
    for selector, payload in data["results"]["pde_residual_norm"].items():
        if not isinstance(payload, dict) or "fold_details" not in payload:
            continue
        folds = payload["fold_details"]
        accs = []
        for _ in range(n_reps):
            sample = random.choices(folds, k=n_sample)
            accs.append(sum(1 for f in sample if f["correct"]) / n_sample)
        mean = statistics.mean(accs)
        sd = statistics.stdev(accs)
        accs_sorted = sorted(accs)
        lo = accs_sorted[int(n_reps * 0.025)]
        hi = accs_sorted[int(n_reps * 0.975)]
        out[selector] = {"mean": mean, "sd": sd, "ci_lo": lo, "ci_hi": hi}
    return out


def main() -> None:
    summary = bootstrap_routing()
    print(
        f"{'selector':<20s} {'mean':>8s} {'sd':>8s} {'ci_lo':>8s} {'ci_hi':>8s}"
    )
    for selector, stats in summary.items():
        print(
            f"  {selector:<18s} "
            f"{stats['mean']*100:>7.1f}% "
            f"{stats['sd']*100:>7.1f} "
            f"{stats['ci_lo']*100:>7.1f}% "
            f"{stats['ci_hi']*100:>7.1f}%"
        )


if __name__ == "__main__":
    main()
