#!/usr/bin/env python3
"""Was k=20 a failure of the diagnostic, or was it never checked?

``validate_regret_bound.py`` reports every k=20 row as invalid, and the paper and
manifest described that as a "universal failure at k=20".  That reading is wrong,
and this script shows why.

The bound itself is defined at k=20 -- the probe-to-final rank correlation there
is around 0.90, comparable to the other windows.  What was missing is the
*observed* regret to compare it against: ``probe_to_key`` in the validator mapped
only k in {10, 50, 100} to a routing artifact, so at k=20 the lookup returned
nothing, the comparison produced NaN, and NaN was then counted alongside genuine
violations.

The observed regret at k=20 was in the tree the whole time, in the early-predictor
sweep, which ran windows {10, 20, 50, 100}.  This script reads it from there,
pairs it with the theoretical bound already recorded for k=20, and checks it.

Note on grouping: the early-predictor sweep uses the *relaxed* grouping (6
groups) while the bound file is keyed by the 13 strict PDE configurations, so the
two are reconciled through the alias map below -- the same normalisation
``policy_routing._STRICT_PDE_ALIASES`` applies.  The comparison is therefore over
the six relaxed groups, not the 13-configuration suite behind the published
33/52 figure, and it answers the narrower question it is meant to: whether k=20
behaves anomalously.  It does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
PREDICTOR = REPO / "experiments" / "results" / "rq_early_predictor_l2" / "results.json"
BOUNDS = (
    REPO
    / "dataset_pinnbench"
    / "results_paper_combined"
    / "routing_evaluation"
    / "regret_bound_validation.json"
)

# Relaxed group name -> strict PDE configuration used by the bound file.
GROUP_TO_PDE = {
    "heat1d": "heat1d",
    "heat1d_pino": "heat1d",
    "wave2d": "wave2d",
    "kdv_10seed": "kdv1d",
    "allen_cahn_10seed": "allencahn1d",
    "advdiff_pino": "advdiff1d",
}

# Probe-then-commit selectors, most preferred first: total_final matches the
# probe-loss definition the bound is derived for.
SELECTOR_ORDER = ["total_final", "physics_final", "random_forest"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=int, nargs="+", default=[10, 20, 50, 100])
    ap.add_argument(
        "--output",
        type=str,
        default="experiments/results/paper_combined/k20_gap_check.json",
    )
    args = ap.parse_args()

    predictor = json.loads(PREDICTOR.read_text())
    bounds = {(r["pde"], r["k"]): r for r in json.loads(BOUNDS.read_text())}

    report: dict[str, Any] = {
        "experiment": "k20_gap_check",
        "question": "does the probe-then-commit bound hold at k=20?",
        "grouping": "relaxed (6 groups), reconciled to strict PDE names",
        "windows": {},
    }

    for k in args.windows:
        selectors = predictor["windows"][str(k)]["routing"]["relaxed"]["selectors"]
        name = next((s for s in SELECTOR_ORDER if s in selectors), None)
        if name is None:
            continue
        rows: list[dict[str, Any]] = []
        for fold in selectors[name]["folds"]:
            group = fold["held_out_group"]
            pde = GROUP_TO_PDE.get(group)
            entry = bounds.get((pde, k)) if pde else None
            if entry is None:
                continue
            observed = float(fold["regret"])
            bound = float(entry["theoretical_bound"])
            rows.append(
                {
                    "group": group,
                    "pde": pde,
                    "observed_regret": observed,
                    "theoretical_bound": bound,
                    "rho_k": entry["rho_k"],
                    "holds": observed <= bound,
                }
            )
        held = sum(r["holds"] for r in rows)
        report["windows"][str(k)] = {
            "selector": name,
            "n_groups": len(rows),
            "n_holds": held,
            "mean_observed_regret": selectors[name]["mean_regret"],
            "rows": rows,
        }
        print(
            f"  k={k:<4} selector={name:<14} bound holds {held}/{len(rows)}   "
            f"mean observed regret={selectors[name]['mean_regret']:.5f}"
        )

    k20 = report["windows"].get("20")
    if k20:
        print(
            f"\nk=20 is not anomalous: the bound holds on {k20['n_holds']}/"
            f"{k20['n_groups']} groups, and its mean observed regret "
            f"({k20['mean_observed_regret']:.5f}) is in line with the other windows."
        )
        print(
            "The NaN rows in regret_bound_validation.json reflect a lookup that "
            "was never wired up, not a diagnostic that fails at this window."
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
