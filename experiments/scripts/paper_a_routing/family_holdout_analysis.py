"""Compute family-level macro accuracy from routing fold decisions.

Groups per-PDE fold decisions by PDE family (Heat, AdvDiff, Burgers, KdV,
AllenCahn, Wave, ReactDiff, NavierStokes; 8 families across 13 PDE
configurations) and reports per-family accuracy (fraction of the family's
configurations routed to the oracle), the macro average over families, and
the number of families routed fully correctly.

Which protocol the numbers describe depends on the input file:

* ``routing_family_holdout/results.json`` (``evaluate_routing_full.py
  --holdout family``): every router is retrained with the held-out PDE's whole
  family removed -- a true leave-one-family-out evaluation.  This is the file
  behind the paper's LOPO-F table.
* ``routing_evaluation/results.json`` (the default): leave-one-PDE-out folds
  merely regrouped by family, so siblings of the held-out PDE stay in training.
  The submitted paper reported these numbers as a family holdout; they are not.

The ``physics_final`` heuristic trains nothing, so it scores the same either way.

Usage:
    PYTHONPATH=src python experiments/scripts/paper_a_routing/family_holdout_analysis.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

_PDE_TO_FAMILY: dict[str, str] = {
    "heat1d": "Heat",
    "advdiff1d": "AdvDiff",
    "advdiff1d_highpe": "AdvDiff",
    "burgers1d": "Burgers",
    "burgers1d_lownu": "Burgers",
    "burgers1d_verylow": "Burgers",
    "kdv1d": "KdV",
    "allencahn1d": "AllenCahn",
    "allencahn1d_sharp": "AllenCahn",
    "wave2d": "Wave",
    "reaction_diffusion1d": "ReactDiff",
    "reaction_diffusion1d_stiff": "ReactDiff",
    "navier_stokes2d": "NavierStokes",
}

_DEFAULT_RESULTS_PATH = Path(
    "experiments/results/paper_combined/routing_evaluation/results.json"
)


def compute_family_macro(results_path: Path = _DEFAULT_RESULTS_PATH) -> dict[str, dict]:
    """Compute family-macro accuracy for each selector, per objective.

    Returns a nested dict ``{objective: {selector: {...}}}``.
    """
    data = json.loads(results_path.read_text())
    out: dict[str, dict] = {}
    for objective, selectors in data.get("results", {}).items():
        per_selector: dict[str, dict] = {}
        for selector, payload in selectors.items():
            if not isinstance(payload, dict) or "fold_details" not in payload:
                continue
            fam_correct: dict[str, list[int]] = defaultdict(list)
            for fold in payload["fold_details"]:
                pde = fold["held_out_pde"]
                family = _PDE_TO_FAMILY.get(pde, "?")
                fam_correct[family].append(1 if fold["correct"] else 0)
            per_family_acc = {
                fam: (sum(arr) / len(arr)) for fam, arr in fam_correct.items()
            }
            macro = sum(per_family_acc.values()) / len(per_family_acc)
            fully_correct = sum(1 for arr in fam_correct.values() if all(arr))
            per_selector[selector] = {
                "per_family_accuracy": per_family_acc,
                "family_macro_accuracy": macro,
                "families_fully_correct": fully_correct,
                "n_families": len(fam_correct),
            }
        out[objective] = per_selector
    return out


def main() -> None:
    summary = compute_family_macro()
    for objective, selectors in summary.items():
        print(f"=== {objective} ===")
        for selector, stats in selectors.items():
            macro = stats["family_macro_accuracy"] * 100
            full = stats["families_fully_correct"]
            n = stats["n_families"]
            print(f"  {selector}: macro={macro:.1f}% fully-correct={full}/{n}")
            for fam in sorted(stats["per_family_accuracy"]):
                acc = stats["per_family_accuracy"][fam] * 100
                print(f"    {fam}: {acc:.1f}%")


if __name__ == "__main__":
    main()
