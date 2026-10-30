"""Print the LaTeX rows of the paper's routing decision tables from released folds.

Three tables, all under the PDE-residual objective at probe window E_p = 50:

* ``--table perpde``: per held-out PDE, the oracle policy and whether each
  selector picked it (leave-one-PDE-out, ``routing_evaluation/results.json``).
* ``--table lopof``: the main-text summary.  For base and PDE-aware routers,
  PDE-level accuracy and family-macro accuracy under leave-one-PDE-out and
  under leave-one-family-out, plus families fully correct under the latter
  (base routers from ``routing_*``, PDE-aware ones from ``meta_router*``).
* ``--table family``: per PDE family, the fraction of its configurations each
  selector routes correctly when the whole family is held out of training
  (``routing_family_holdout/results.json``, written by
  ``evaluate_routing_full.py --holdout family``), plus the family-macro
  accuracy (mean of those fractions) and the count of fully correct families.

These tables used to be typed in.  The per-PDE table came from an older
8-group run with a different policy set, and most family cells matched no
artifact (every selector was marked correct on Burgers, where they route
0-67% of the configurations correctly).

Usage:
    PYTHONPATH=src python experiments/scripts/paper_a_routing/routing_tables.py --table perpde
    PYTHONPATH=src python experiments/scripts/paper_a_routing/routing_tables.py --table family
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from family_holdout_analysis import _PDE_TO_FAMILY  # type: ignore[import-not-found]

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / "experiments" / "results" / "paper_combined"
LOPO = EVAL / "routing_evaluation" / "results.json"
LOPO_F = EVAL / "routing_family_holdout" / "results.json"
META = EVAL / "meta_router" / "results.json"
META_F = EVAL / "meta_router_family_holdout" / "results.json"
META_ROWS = [("GBR+PDE", "GBR + PDE (top-5)"), ("RF+PDE-top5", "RF + PDE (top-5)")]
BASE_ROWS = [("Ridge", "Ridge"), ("RandomForest", "Random Forest"),
             ("GradientBoosting", "Gradient Boosting"),
             ("physics_final", r"\texttt{physics\_final} (heuristic)")]
METRIC = "pde_residual_norm"
SELECTORS = ["RandomForest", "GradientBoosting", "Ridge", "physics_final"]
FAMILIES = ["Heat", "AdvDiff", "Burgers", "KdV", "AllenCahn", "Wave", "ReactDiff",
            "NavierStokes"]
PDE_LABEL = {
    "heat1d": "Heat1D", "advdiff1d": "AdvDiff1D", "advdiff1d_highpe": "AdvDiff1D (high Pe)",
    "burgers1d": "Burgers1D", "burgers1d_lownu": r"Burgers1D (low $\nu$)",
    "burgers1d_verylow": "Burgers1D (very low)", "kdv1d": "KdV1D",
    "allencahn1d": "AllenCahn1D", "allencahn1d_sharp": "AllenCahn1D (sharp)",
    "wave2d": "Wave2D", "reaction_diffusion1d": "ReactDiff1D",
    "reaction_diffusion1d_stiff": "ReactDiff1D (stiff)", "navier_stokes2d": "NavierStokes2D",
}
MARK = {True: r"$\checkmark$", False: r"$\times$"}


def _tt(policy: str) -> str:
    return r"\texttt{" + policy.replace("_", r"\_") + "}"


def _folds(path: Path) -> dict[str, dict[str, dict]]:
    res = json.loads(path.read_text())["results"][METRIC]
    return {s: {f["held_out_pde"]: f for f in res[s]["fold_details"]} for s in SELECTORS}


def perpde() -> None:
    folds = _folds(LOPO)
    order = list(PDE_LABEL)
    for pde in order:
        oracle = folds[SELECTORS[0]][pde]["actual_best"]
        cells = " & ".join(MARK[folds[s][pde]["correct"]] for s in SELECTORS)
        print(f"{PDE_LABEL[pde]:<24} & {_tt(oracle):<28} & {cells} \\\\")
    print(r"\midrule")
    acc = [sum(f["correct"] for f in folds[s].values()) for s in SELECTORS]
    print(r"\textbf{Correct} & --- & " + " & ".join(f"{a}/13" for a in acc) + r" \\")


def family() -> None:
    folds = _folds(LOPO_F)
    by_family: dict[str, list[str]] = defaultdict(list)
    for pde in PDE_LABEL:
        by_family[_PDE_TO_FAMILY[pde]].append(pde)
    macro = dict.fromkeys(SELECTORS, 0.0)
    full = dict.fromkeys(SELECTORS, 0)
    for fam in FAMILIES:
        pdes = by_family[fam]
        oracles = sorted({folds[SELECTORS[0]][p]["actual_best"] for p in pdes})
        cells = []
        for s in SELECTORS:
            frac = Fraction(sum(folds[s][p]["correct"] for p in pdes), len(pdes))
            macro[s] += float(frac) / len(FAMILIES)
            full[s] += frac == 1
            cells.append(MARK[True] if frac == 1 else MARK[False] if frac == 0
                         else f"${frac.numerator}/{frac.denominator}$")
        print(f"{fam:<14} & {len(pdes)} & {', '.join(_tt(o) for o in oracles)} & "
              + " & ".join(cells) + r" \\")
    print(r"\midrule")
    print(r"\textbf{Family macro} & & & "
          + " & ".join(f"${100 * macro[s]:.1f}\\%$" for s in SELECTORS) + r" \\")
    print(r"\textbf{Fully correct} & & & " + " & ".join(f"{full[s]}/8" for s in SELECTORS)
          + r" \\")


def _scores(correct: dict[str, bool]) -> tuple[float, float, int]:
    """(PDE-level accuracy, family macro, families fully correct)."""
    fam: dict[str, list[bool]] = defaultdict(list)
    for pde, ok in correct.items():
        fam[_PDE_TO_FAMILY[pde]].append(ok)
    macro = sum(sum(v) / len(v) for v in fam.values()) / len(fam)
    return sum(correct.values()) / len(correct), macro, sum(all(v) for v in fam.values())


def _meta_correct(path: Path, router: str) -> dict[str, bool]:
    folds = json.loads(path.read_text())["fold_details"][router]
    return {f["held_out"]: bool(f["correct"]) for f in folds}


def lopof() -> None:
    rows = []
    lopo, lopo_f = _folds(LOPO), _folds(LOPO_F)
    for key, label in BASE_ROWS:
        a = {p: f["correct"] for p, f in lopo[key].items()}
        b = {p: f["correct"] for p, f in lopo_f[key].items()}
        rows.append((label, "No", _scores(a), _scores(b)))
    for key, label in META_ROWS:
        rows.append((label, "Yes", _scores(_meta_correct(META, key)),
                     _scores(_meta_correct(META_F, key))))
    for label, feats, (a1, m1, _), (a2, m2, f2) in rows:
        print(f"{label:<38} & {feats} & ${100 * a1:.1f}\\%$ & ${100 * a2:.1f}\\%$ & "
              f"${100 * m1:.1f}\\%$ & ${100 * m2:.1f}\\%$ & {f2}/8 \\\\")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", choices=("perpde", "family", "lopof"), required=True)
    {"perpde": perpde, "family": family, "lopof": lopof}[ap.parse_args().table]()


if __name__ == "__main__":
    main()
