#!/usr/bin/env python3
"""Evaluate PDE-aware MetaRouter vs RF vs baselines on the full Paper A benchmark.

Uses leave-one-PDE-out cross-validation on all 13 PDEs (450 decisions).
For each held-out PDE, trains each router on remaining PDEs and selects
the policy with the lowest predicted final metric.

Reports:
    - Accuracy: fraction of held-out PDEs where the selected policy matches oracle
    - Regret: mean (selected_metric - oracle_metric) / oracle_metric across PDEs

Routers evaluated:
    1. MetaRouter          -- MLP(16,8) with PDE features + top-5 training features
    2. MetaRouter-full     -- MLP(32,16) with PDE features + all training features
    3. RF+PDE              -- RF with PDE features concatenated (apples-to-apples)
    4. RF+PDE-top5         -- RF with PDE features + top-5 training features
    5. RF                  -- RF on all training features (no PDE features)
    6. RF-top3             -- RF on top-3 training features
    7. physics_final       -- heuristic: pick policy with lowest physics_loss at window end
    8. random              -- uniform random policy selection

Usage:
    python experiments/scripts/paper_combined/evaluate_meta_router.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from causal_pinn.training.early_predictor import (
    FeatureExtractor,
    PredictorDataset,
    PredictorSample,
)
from causal_pinn.training.meta_router import PDE_FEATURE_REGISTRY, MetaRouter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAPER_A_DIR = Path(__file__).resolve().parents[3] / "experiments" / "results" / "paper_a"
OUTPUT_DIR = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "results"
    / "paper_a"
    / "meta_router_evaluation"
)
SKIP_DIRS = {
    "bandit_evaluation",
    "routing_evaluation_full",
    "scaling_ablation",
    "holdout_validation",
    "meta_router_evaluation",
}
WINDOW_SIZE = 50
TARGET_METRIC = "pde_residual_norm"


# ---------------------------------------------------------------------------
# Data loading (mirrors scaling_and_ablation.py)
# ---------------------------------------------------------------------------


def load_all_samples(
    base_dir: Path,
    extractor: FeatureExtractor,
    target_metric: str,
) -> PredictorDataset:
    """Load all Paper A results into a PredictorDataset."""
    dataset = PredictorDataset()
    loaded = 0
    skipped = 0

    for pde_dir in sorted(base_dir.iterdir()):
        if not pde_dir.is_dir() or pde_dir.name in SKIP_DIRS:
            continue

        results_path = pde_dir / "results.json"
        if not results_path.exists():
            continue

        with open(results_path) as f:
            data = json.load(f)

        pde_name = data.get("pde", pde_dir.name)
        runs = data.get("raw_results", [])

        for run in runs:
            if not run.get("success", True):
                skipped += 1
                continue

            history = run.get("training_history", {})
            stage2 = history.get("stage2_losses", [])

            if len(stage2) < WINDOW_SIZE:
                skipped += 1
                continue

            metrics = run.get("metrics", {})
            target = metrics.get(target_metric)
            if target is None:
                skipped += 1
                continue

            features = extractor.extract(stage2)
            if not features:
                skipped += 1
                continue

            policy = run.get("model_name", "unknown")
            seed = run.get("seed", 0)

            dataset.samples.append(
                PredictorSample(
                    features=features,
                    target=float(target),
                    pde=pde_name,
                    policy=policy,
                    seed=seed,
                )
            )
            loaded += 1

    print(f"Loaded {loaded} samples, skipped {skipped}")
    return dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _global_feature_names(dataset: PredictorDataset) -> list[str]:
    """Get ordered union of all feature names."""
    names: list[str] = []
    seen: set[str] = set()
    for s in dataset.samples:
        for k in s.features:
            if k not in seen:
                names.append(k)
                seen.add(k)
    return names


def _best_policy(
    policies: list[str], values: NDArray[np.float64]
) -> tuple[str, float]:
    """Return (policy, mean_value) with lowest mean value."""
    policy_vals: dict[str, list[float]] = {}
    for p, v in zip(policies, values, strict=False):
        policy_vals.setdefault(p, []).append(float(v))
    policy_means = {p: float(np.mean(vs)) for p, vs in policy_vals.items()}
    best = min(policy_means, key=lambda k: policy_means[k])
    return best, policy_means[best]


def _get_rf_top_features(
    dataset: PredictorDataset,
    feature_names: list[str],
    top_k: int,
) -> list[str]:
    """Train RF on full dataset and return top-K feature names by importance."""
    X = dataset.feature_matrix(feature_names=feature_names)
    y = dataset.target_vector()
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X, y)
    importance_order = np.argsort(rf.feature_importances_)[::-1]
    return [feature_names[i] for i in importance_order[:top_k]]


def _pde_features_array(pde_names: list[str]) -> NDArray[np.float64]:
    """Convert PDE names to numeric feature matrix (N, 7)."""
    rows = []
    for name in pde_names:
        pf = PDE_FEATURE_REGISTRY.get(name)
        if pf is not None:
            rows.append(pf.to_array())
        else:
            rows.append(np.zeros(7, dtype=np.float64))
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Router wrappers
# ---------------------------------------------------------------------------


def _train_predict_meta(
    train_ds: PredictorDataset,
    test_ds: PredictorDataset,
    feature_names: list[str],
    hidden_sizes: tuple[int, ...],
    alpha: float,
) -> NDArray[np.float64]:
    """Train MetaRouter and predict on test set."""
    X_train = train_ds.feature_matrix(feature_names=feature_names)
    y_train = train_ds.target_vector()
    train_pdes = train_ds.pde_labels()

    router = MetaRouter(
        hidden_sizes=hidden_sizes,
        alpha=alpha,
        max_iter=2000,
        random_state=42,
    )
    router.fit(X_train, train_pdes, y_train)

    X_test = test_ds.feature_matrix(feature_names=feature_names)
    test_pdes = test_ds.pde_labels()
    return router.predict(X_test, test_pdes)


def _train_predict_rf(
    train_ds: PredictorDataset,
    test_ds: PredictorDataset,
    feature_names: list[str],
    *,
    append_pde_features: bool = False,
) -> NDArray[np.float64]:
    """Train RF and predict on test set, optionally appending PDE features."""
    X_train = train_ds.feature_matrix(feature_names=feature_names)
    y_train = train_ds.target_vector()
    X_test = test_ds.feature_matrix(feature_names=feature_names)

    if append_pde_features:
        pde_train = _pde_features_array(train_ds.pde_labels())
        pde_test = _pde_features_array(test_ds.pde_labels())
        X_train = np.hstack([X_train, pde_train])
        X_test = np.hstack([X_test, pde_test])

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf.predict(X_test)


def _train_predict_gbr(
    train_ds: PredictorDataset,
    test_ds: PredictorDataset,
    feature_names: list[str],
    *,
    append_pde_features: bool = False,
) -> NDArray[np.float64]:
    """Train GradientBoosting and predict, optionally with PDE features."""
    X_train = train_ds.feature_matrix(feature_names=feature_names)
    y_train = train_ds.target_vector()
    X_test = test_ds.feature_matrix(feature_names=feature_names)

    if append_pde_features:
        pde_train = _pde_features_array(train_ds.pde_labels())
        pde_test = _pde_features_array(test_ds.pde_labels())
        X_train = np.hstack([X_train, pde_train])
        X_test = np.hstack([X_test, pde_test])

    gbr = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )
    gbr.fit(X_train, y_train)
    return gbr.predict(X_test)


# ---------------------------------------------------------------------------
# Leave-one-PDE-out evaluation
# ---------------------------------------------------------------------------


def evaluate_all_routers(
    dataset: PredictorDataset, group_of: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run leave-one-PDE-out CV for all routers.

    With ``group_of`` (PDE -> family), each fold still tests one PDE but drops
    every PDE of its family from training (and from top-K feature selection):
    a true leave-one-family-out evaluation.

    Returns:
        Dict mapping router_name -> {accuracy, mean_regret, fold_details}.
    """
    global_names = _global_feature_names(dataset)
    unique_pdes = dataset.unique_pdes()
    pde_labels = dataset.pde_labels()

    print(
        f"\nFeature dimensionality: {len(global_names)} training features"
        f" + 7 PDE features"
    )
    print(f"PDEs in benchmark: {len(unique_pdes)}")
    print(f"Total samples: {len(dataset)}")

    # NOTE: top-K feature selection is now done INSIDE each CV fold
    # to prevent held-out PDE information from leaking into feature ranking.
    # Routers that use "top5" or "top3" compute fold-specific importances
    # from the train split only.

    def _fold_top_k(train_ds: PredictorDataset, k: int) -> list[str]:
        """Select top-K features from train split only (no leakage)."""
        X_tr = train_ds.feature_matrix(feature_names=global_names)
        y_tr = train_ds.target_vector()
        rf_sel = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )
        rf_sel.fit(X_tr, y_tr)
        order = np.argsort(rf_sel.feature_importances_)[::-1]
        return [global_names[i] for i in order[:k]]

    # Router registry: name -> callable(train_ds, test_ds) -> y_pred
    # Routers with "top5"/"top3" call _fold_top_k on train_ds inside the fold.
    router_registry: dict[str, Any] = {
        "MetaRouter": lambda tr, te: _train_predict_meta(
            tr, te, _fold_top_k(tr, 5), hidden_sizes=(16, 8), alpha=0.1,
        ),
        "MetaRouter-full": lambda tr, te: _train_predict_meta(
            tr, te, global_names, hidden_sizes=(32, 16), alpha=0.1,
        ),
        "GBR+PDE": lambda tr, te: _train_predict_gbr(
            tr, te, _fold_top_k(tr, 5), append_pde_features=True,
        ),
        "GBR+PDE-all": lambda tr, te: _train_predict_gbr(
            tr, te, global_names, append_pde_features=True,
        ),
        "RF+PDE": lambda tr, te: _train_predict_rf(
            tr, te, global_names, append_pde_features=True,
        ),
        "RF+PDE-top5": lambda tr, te: _train_predict_rf(
            tr, te, _fold_top_k(tr, 5), append_pde_features=True,
        ),
        "RF": lambda tr, te: _train_predict_rf(
            tr, te, global_names, append_pde_features=False,
        ),
        "RF-top3": lambda tr, te: _train_predict_rf(
            tr, te, _fold_top_k(tr, 3), append_pde_features=False,
        ),
    }

    # Add heuristic and random as special cases
    all_router_names = list(router_registry.keys()) + ["physics_final", "random"]

    results: dict[str, dict[str, Any]] = {}
    for name in all_router_names:
        results[name] = {
            "correct": 0,
            "regret_sum": 0.0,
            "n_folds": 0,
            "folds": [],
        }

    def _group(pde: str) -> str:
        return group_of[pde] if group_of else pde

    for held_out in unique_pdes:
        train_samples = [
            s for s, p in zip(dataset.samples, pde_labels, strict=False)
            if _group(p) != _group(held_out)
        ]
        test_samples = [
            s for s, p in zip(dataset.samples, pde_labels, strict=False) if p == held_out
        ]

        if not train_samples or not test_samples:
            continue

        train_ds = PredictorDataset(samples=train_samples)
        test_ds = PredictorDataset(samples=test_samples)

        test_policies = test_ds.policy_labels()
        y_test = test_ds.target_vector()

        # Oracle: actual best policy
        oracle_policy, oracle_val = _best_policy(test_policies, y_test)

        # Evaluate all ML-based routers
        for name, predict_fn in router_registry.items():
            y_pred = predict_fn(train_ds, test_ds)
            pred_policy, _ = _best_policy(test_policies, y_pred)

            is_correct = pred_policy == oracle_policy
            sel_vals = [
                float(t)
                for p, t in zip(test_policies, y_test, strict=False)
                if p == pred_policy
            ]
            selected_val = float(np.mean(sel_vals)) if sel_vals else oracle_val
            regret = (selected_val - oracle_val) / max(abs(oracle_val), 1e-10)

            results[name]["correct"] += int(is_correct)
            results[name]["regret_sum"] += regret
            results[name]["n_folds"] += 1
            results[name]["folds"].append({
                "held_out": held_out,
                "oracle_policy": oracle_policy,
                "oracle_val": oracle_val,
                "predicted_policy": pred_policy,
                "selected_val": selected_val,
                "correct": bool(is_correct),
                "regret": regret,
            })

        # physics_final heuristic
        y_proxy = np.array(
            [s.features.get("physics_loss_final", 0.0) for s in test_samples],
            dtype=np.float64,
        )
        pred_policy, _ = _best_policy(test_policies, y_proxy)
        sel_vals = [
            float(t) for p, t in zip(test_policies, y_test, strict=False) if p == pred_policy
        ]
        selected_val = float(np.mean(sel_vals)) if sel_vals else oracle_val
        regret = (selected_val - oracle_val) / max(abs(oracle_val), 1e-10)
        results["physics_final"]["correct"] += int(pred_policy == oracle_policy)
        results["physics_final"]["regret_sum"] += regret
        results["physics_final"]["n_folds"] += 1
        results["physics_final"]["folds"].append({
            "held_out": held_out,
            "oracle_policy": oracle_policy,
            "oracle_val": oracle_val,
            "predicted_policy": pred_policy,
            "selected_val": selected_val,
            "correct": bool(pred_policy == oracle_policy),
            "regret": regret,
        })

        # random baseline
        unique_test_policies = list(set(test_policies))
        exp_acc = (
            1.0 / len(unique_test_policies) if unique_test_policies else 0.0
        )
        all_policy_means: dict[str, list[float]] = {}
        for p, t in zip(test_policies, y_test, strict=False):
            all_policy_means.setdefault(p, []).append(float(t))
        mean_all = float(
            np.mean([np.mean(vs) for vs in all_policy_means.values()])
        )
        regret_rand = (mean_all - oracle_val) / max(abs(oracle_val), 1e-10)
        results["random"]["correct"] += exp_acc
        results["random"]["regret_sum"] += regret_rand
        results["random"]["n_folds"] += 1
        results["random"]["folds"].append({
            "held_out": held_out,
            "oracle_policy": oracle_policy,
            "oracle_val": oracle_val,
            "predicted_policy": "random",
            "correct": False,
            "regret": regret_rand,
        })

    # Summarize
    summary: dict[str, dict[str, Any]] = {}
    for name in all_router_names:
        n = results[name]["n_folds"]
        if n == 0:
            summary[name] = {
                "accuracy": 0.0,
                "mean_regret": 0.0,
                "n_folds": 0,
            }
            continue
        summary[name] = {
            "accuracy": results[name]["correct"] / n,
            "mean_regret": results[name]["regret_sum"] / n,
            "n_folds": n,
            "folds": results[name]["folds"],
        }

    return summary


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_results_table(summary: dict[str, dict[str, Any]]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 72)
    print("LEAVE-ONE-PDE-OUT ROUTING EVALUATION")
    print("=" * 72)
    print(f"{'Router':<22s} {'Accuracy':>10s} {'Mean Regret':>12s} {'Folds':>6s}")
    print("-" * 72)

    for name, data in summary.items():
        acc = data["accuracy"]
        regret = data["mean_regret"]
        n = data["n_folds"]
        print(f"{name:<22s} {acc:>9.1%} {regret:>11.3f} {n:>6d}")

    print("-" * 72)

    # Per-PDE breakdown: best ML router vs RF vs physics_final
    # Find best PDE-aware router
    pde_aware = [
        n for n in summary
        if n not in ("RF", "RF-top3", "physics_final", "random")
    ]
    best_pde_router = max(pde_aware, key=lambda n: summary[n]["accuracy"])

    print(f"\nPer-PDE breakdown ({best_pde_router} vs RF vs physics_final):")
    header = (
        f"{'PDE':<25s} {'Oracle':>18s}"
        f" {best_pde_router:>14s}"
        f" {'RF':>14s}"
        f" {'physics_final':>14s}"
    )
    print(header)
    print("-" * 88)

    best_folds = summary.get(best_pde_router, {}).get("folds", [])
    rf_folds = summary.get("RF", {}).get("folds", [])
    phys_folds = summary.get("physics_final", {}).get("folds", [])

    for bf, rf, pf in zip(best_folds, rf_folds, phys_folds, strict=False):
        pde = bf["held_out"]
        oracle = bf["oracle_policy"]

        def _fmt(fold: dict[str, Any]) -> str:
            p = fold["predicted_policy"]
            ok = " Y" if fold["correct"] else "  "
            return f"{p[:12]:>12s}{ok}"

        print(
            f"{pde:<25s} {oracle:>18s}"
            f" {_fmt(bf):>14s}"
            f" {_fmt(rf):>14s}"
            f" {_fmt(pf):>14s}"
        )

    print("-" * 88)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--holdout", choices=("pde", "family"), default="pde",
                    help="Hold out one PDE per fold (default) or its whole family")
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()
    group_of = None
    if args.holdout == "family":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper_a_routing"))
        from family_holdout_analysis import _PDE_TO_FAMILY  # type: ignore[import-not-found]

        group_of = _PDE_TO_FAMILY
    t_start = time.time()

    print("=" * 72)
    print("META-ROUTER EVALUATION: PDE-Aware vs Standard RF")
    print("=" * 72)

    extractor = FeatureExtractor(window_size=WINDOW_SIZE)
    dataset = load_all_samples(PAPER_A_DIR, extractor, TARGET_METRIC)

    if len(dataset) == 0:
        print("ERROR: No samples loaded. Check that Paper A results exist.")
        sys.exit(1)

    print(f"\nUnique PDEs: {dataset.unique_pdes()}")
    print(f"Unique policies: {sorted(set(dataset.policy_labels()))}")

    summary = evaluate_all_routers(dataset, group_of)
    print_results_table(summary)

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_data: dict[str, Any] = {
        "experiment": "meta_router_evaluation",
        "holdout": args.holdout,
        "target_metric": TARGET_METRIC,
        "window_size": WINDOW_SIZE,
        "n_samples": len(dataset),
        "n_pdes": len(dataset.unique_pdes()),
        "pdes": dataset.unique_pdes(),
        "results": {
            name: {
                "accuracy": data["accuracy"],
                "mean_regret": data["mean_regret"],
                "n_folds": data["n_folds"],
            }
            for name, data in summary.items()
        },
        "fold_details": {
            name: data.get("folds", []) for name, data in summary.items()
        },
    }

    output_path = args.output_dir / "results.json"
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {output_path}")

    elapsed = time.time() - t_start
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
