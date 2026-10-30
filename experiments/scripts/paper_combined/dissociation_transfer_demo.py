#!/usr/bin/env python3
"""Does the accuracy-regret dissociation transfer outside PINNs?

Track E2 of the post-review roadmap.  The paper's central evaluation claim is
that top-1 routing accuracy is a poor stand-in for realized loss: two routers
can agree on how often they pick the best policy and still differ by orders of
magnitude in what their mistakes cost.  In the benchmark that shows up as a 77x
regret gap at equal 62% accuracy.

If that is a real evaluation principle rather than a quirk of PDE solvers, it
should reproduce on an ordinary model-selection problem with no physics in it.
This script runs the same protocol on classifier selection: several candidate
learners, a suite of tabular tasks, leave-one-task-out routing, and the same two
metrics.  Nothing here imports the PINN codebase.

The whole experiment is repeated over several seeds -- the seed controls
synthetic-task generation, train/test splits and the learners' own randomness --
so the headline numbers come with a spread rather than resting on one draw.

Outputs ``{output_dir}/results.json``.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_classification,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

PROBE_FRACTION = 0.1  # how much training data the cheap probe is allowed


def candidate_policies(seed: int) -> dict[str, Any]:
    """The 'policy library' -- deliberately heterogeneous in quality."""
    return {
        "logreg": LogisticRegression(max_iter=2000),
        "rbf_svm": SVC(),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "knn": KNeighborsClassifier(),
        "decision_tree": DecisionTreeClassifier(random_state=seed),
        "naive_bayes": GaussianNB(),
        "ridge": RidgeClassifier(),
        "lda": LinearDiscriminantAnalysis(),
    }


def task_suite(seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Tabular classification tasks spanning easy and hard regimes."""
    rng = np.random.RandomState(seed)
    tasks: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "iris": (lambda d: (d.data, d.target))(load_iris()),
        "wine": (lambda d: (d.data, d.target))(load_wine()),
        "breast_cancer": (lambda d: (d.data, d.target))(load_breast_cancer()),
        "digits": (lambda d: (d.data, d.target))(load_digits()),
    }
    # Synthetic tasks widen the spread of which learner wins.
    specs = [
        ("synth_linear", dict(n_features=20, n_informative=15, class_sep=2.0)),
        ("synth_nonlinear", dict(n_features=20, n_informative=5, class_sep=0.5)),
        ("synth_sparse", dict(n_features=50, n_informative=3, class_sep=1.0)),
        ("synth_noisy", dict(n_features=15, n_informative=10, flip_y=0.15)),
        ("synth_manyclass", dict(n_features=25, n_informative=20, n_classes=5)),
        ("synth_redundant", dict(n_features=30, n_informative=5, n_redundant=20)),
    ]
    for name, kw in specs:
        X, y = make_classification(
            n_samples=600, random_state=rng.randint(10_000), **kw
        )
        tasks[name] = (X, y)
    return tasks


def meta_features(X: np.ndarray, y: np.ndarray) -> list[float]:
    """Cheap descriptors of a task, analogous to the PDE feature registry."""
    n, d = X.shape
    _classes, counts = np.unique(y, return_counts=True)
    return [
        float(np.log(n)),
        float(np.log(d)),
        float(len(_classes)),
        float(counts.min() / counts.max()),
        float(np.mean(np.std(X, axis=0))),
        float(d / n),
    ]


def evaluate_task(
    X: np.ndarray, y: np.ndarray, seed: int
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (final error per policy, cheap probe error per policy)."""
    X = StandardScaler().fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    # Probe: same learners, small slice of the training data -- the analogue of
    # training each policy for k epochs before committing.
    n_probe = max(20, int(len(X_tr) * PROBE_FRACTION))
    final: dict[str, float] = {}
    probe: dict[str, float] = {}
    for name, model in candidate_policies(seed).items():
        final[name] = 1.0 - clone(model).fit(X_tr, y_tr).score(X_te, y_te)
        probe[name] = 1.0 - clone(model).fit(
            X_tr[:n_probe], y_tr[:n_probe]
        ).score(X_te, y_te)
    return final, probe


def run_experiment(seed: int) -> dict[str, dict[str, float]]:
    """One full replication; returns per-router {accuracy, mean/max regret}."""
    tasks = task_suite(seed)
    policies = list(candidate_policies(seed))
    names = list(tasks)

    finals: dict[str, dict[str, float]] = {}
    probes: dict[str, dict[str, float]] = {}
    feats: dict[str, list[float]] = {}
    for name, (X, y) in tasks.items():
        finals[name], probes[name] = evaluate_task(X, y, seed)
        feats[name] = meta_features(X, y)

    # ---- Router pool.  Each router maps task -> chosen policy without seeing
    # that task's final scores.  A wide pool is the point: with many routers we
    # can find pairs that tie on accuracy and compare what their mistakes cost.
    picks: dict[str, dict[str, str]] = {}
    picks["probe_heuristic"] = {t: min(probes[t], key=probes[t].get) for t in names}

    learned: dict[str, str] = {}
    for held in names:  # leave-one-task-out, as in the paper
        train = [t for t in names if t != held]
        Xtr = np.array([feats[t] for t in train])
        pred: dict[str, float] = {}
        for pol in policies:
            ytr = np.array([finals[t][pol] for t in train])
            reg = RandomForestRegressor(n_estimators=200, random_state=seed)
            reg.fit(Xtr, ytr)
            pred[pol] = float(reg.predict(np.array([feats[held]]))[0])
        learned[held] = min(pred, key=pred.get)
    picks["learned_router"] = learned

    for pol in policies:  # "always commit to X" -- the simplest routers there are
        picks[f"always_{pol}"] = dict.fromkeys(names, pol)

    rng = np.random.RandomState(seed + 9973)
    picks["random"] = {t: policies[rng.randint(len(policies))] for t in names}

    oracle = {t: min(finals[t], key=finals[t].get) for t in names}
    scores: dict[str, dict[str, float]] = {}
    for rname, mapping in picks.items():
        regrets = [finals[t][mapping[t]] - finals[t][oracle[t]] for t in names]
        scores[rname] = {
            "top1_accuracy": float(np.mean([mapping[t] == oracle[t] for t in names])),
            "mean_regret": float(np.mean(regrets)),
            "max_regret": float(np.max(regrets)),
        }

    # Dynamic range of policy quality on this replication -- the quantity that
    # should predict how severe the dissociation is.
    all_best = [finals[t][oracle[t]] for t in names]
    nz = [v for v in all_best if v > 0]
    scores["_meta"] = {
        "best_error_min": float(min(nz)) if nz else 0.0,
        "best_error_max": float(max(all_best)),
        "dynamic_range": float(max(all_best) / min(nz)) if nz else float("inf"),
    }
    return scores


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/paper_combined/dissociation_transfer",
    )
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict[str, dict[str, float]]] = []
    for s in range(args.n_seeds):
        print(f"  seed {s}...", end=" ", flush=True)
        per_seed.append(run_experiment(s))
        print("done")

    routers = [r for r in per_seed[0] if r != "_meta"]

    def col(router: str, metric: str) -> list[float]:
        return [rep[router][metric] for rep in per_seed]

    print("\n" + "=" * 78)
    print(f"{'router':<22}{'top-1 acc':>18}{'mean regret':>22}")
    print("-" * 78)
    summary: dict[str, Any] = {}
    for r in sorted(routers, key=lambda r: -float(np.mean(col(r, "top1_accuracy")))):
        a, g = col(r, "top1_accuracy"), col(r, "mean_regret")
        summary[r] = {
            "top1_accuracy_mean": float(np.mean(a)),
            "top1_accuracy_std": float(np.std(a)),
            "mean_regret_mean": float(np.mean(g)),
            "mean_regret_std": float(np.std(g)),
        }
        print(
            f"{r:<22}{np.mean(a):>12.1%} +- {np.std(a):<4.1%}"
            f"{np.mean(g):>14.4f} +- {np.std(g):<7.4f}"
        )

    # ---- Tie analysis, per replication, then pooled ------------------------
    ratios: list[float] = []
    tie_count = 0
    for rep in per_seed:
        for a, b in combinations(routers, 2):
            if rep[a]["top1_accuracy"] != rep[b]["top1_accuracy"]:
                continue
            tie_count += 1
            ra, rb = rep[a]["mean_regret"], rep[b]["mean_regret"]
            hi, lo = max(ra, rb), min(ra, rb)
            if lo > 0:
                ratios.append(hi / lo)

    corrs = [
        float(
            np.corrcoef(
                [rep[r]["top1_accuracy"] for r in routers],
                [rep[r]["mean_regret"] for r in routers],
            )[0, 1]
        )
        for rep in per_seed
    ]
    dyn = [rep["_meta"]["dynamic_range"] for rep in per_seed]

    print("\nAccuracy-tied router pairs, pooled over seeds:")
    print(f"  tied pairs: {tie_count} ({len(ratios)} with non-zero regret)")
    if ratios:
        print(
            f"  regret ratio at equal accuracy: median {np.median(ratios):.1f}x, "
            f"90th pct {np.percentile(ratios, 90):.1f}x, max {max(ratios):.1f}x"
        )
    print(
        f"\naccuracy vs regret correlation: {np.mean(corrs):.3f} +- {np.std(corrs):.3f} "
        f"(r = -1 would mean accuracy determines realized cost)"
    )
    print(
        f"policy-quality dynamic range on these tasks: median {np.median(dyn):.0f}x"
    )

    payload = {
        "experiment": "dissociation_transfer_demo",
        "domain": "tabular classifier selection (no PDE, no PINN)",
        "n_seeds": args.n_seeds,
        "protocol": "leave-one-task-out; probe router sees 10% of training data",
        "router_summary": summary,
        "tie_analysis": {
            "n_tied_pairs": tie_count,
            "n_ratios": len(ratios),
            "median_ratio": float(np.median(ratios)) if ratios else None,
            "p90_ratio": float(np.percentile(ratios, 90)) if ratios else None,
            "max_ratio": float(max(ratios)) if ratios else None,
        },
        "accuracy_regret_correlation": {
            "mean": float(np.mean(corrs)),
            "std": float(np.std(corrs)),
            "per_seed": corrs,
        },
        "policy_quality_dynamic_range": {
            "median": float(np.median(dyn)),
            "per_seed": dyn,
        },
        "per_seed_scores": per_seed,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}/results.json")


if __name__ == "__main__":
    main()
