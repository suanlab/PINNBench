#!/usr/bin/env python3
"""Held-out PDE generalization and L2 feature ablation for Paper A.

Part 1 — Held-out PDE generalization: Train RF router on 10 PDEs, test on
3 completely unseen PDEs. Proves the router generalizes beyond the training
distribution.

Part 2 — Top-3 feature router on L2: Run the feature ablation study using
relative_l2_error as target metric.  Shows whether top-3 features also work
for L2, or if L2 routing needs more features.

Outputs:
    experiments/results/paper_a/holdout_validation/results.json

Usage:
    python experiments/scripts/paper_a_routing/holdout_validation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from sklearn.ensemble import RandomForestRegressor

from causal_pinn.training.early_predictor import (
    FeatureExtractor,
    PredictorDataset,
    PredictorSample,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAPER_A_DIR = Path(__file__).resolve().parents[3] / "experiments" / "results" / "paper_a"
OUTPUT_DIR = PAPER_A_DIR / "holdout_validation"
SKIP_DIRS = {"bandit_evaluation", "routing_evaluation_full", "scaling_ablation",
             "holdout_validation", "ntk_proxy"}
WINDOW_SIZE = 50

TRAIN_PDES = [
    "heat1d", "burgers1d", "advdiff1d", "kdv1d", "allencahn1d", "wave2d",
    "burgers1d_lownu", "advdiff1d_highpe", "reaction_diffusion1d",
    "reaction_diffusion1d_stiff",
]
TEST_PDES = ["allencahn1d_sharp", "burgers1d_verylow", "navier_stokes2d"]

ABLATION_K_VALUES = [3, 5, 10, 20]  # "all" added dynamically


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_samples(
    base_dir: Path,
    extractor: FeatureExtractor,
    target_metric: str,
    pde_filter: list[str] | None = None,
) -> PredictorDataset:
    """Load Paper A results into a PredictorDataset."""
    dataset = PredictorDataset()
    skipped = 0
    loaded = 0

    for pde_dir in sorted(base_dir.iterdir()):
        if not pde_dir.is_dir() or pde_dir.name in SKIP_DIRS:
            continue
        if pde_filter is not None and pde_dir.name not in pde_filter:
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

    return dataset


def load_dual_samples(
    base_dir: Path,
    extractor: FeatureExtractor,
) -> tuple[PredictorDataset, PredictorDataset]:
    """Load samples with both residual and L2 targets (same features).

    Returns:
        (dataset_residual, dataset_l2) with aligned samples.
    """
    ds_res = PredictorDataset()
    ds_l2 = PredictorDataset()

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
                continue

            history = run.get("training_history", {})
            stage2 = history.get("stage2_losses", [])
            if len(stage2) < WINDOW_SIZE:
                continue

            metrics = run.get("metrics", {})
            res_target = metrics.get("pde_residual_norm")
            l2_target = metrics.get("relative_l2_error")
            if res_target is None or l2_target is None:
                continue

            features = extractor.extract(stage2)
            if not features:
                continue

            policy = run.get("model_name", "unknown")
            seed = run.get("seed", 0)

            ds_res.samples.append(
                PredictorSample(
                    features=features, target=float(res_target),
                    pde=pde_name, policy=policy, seed=seed,
                )
            )
            ds_l2.samples.append(
                PredictorSample(
                    features=features, target=float(l2_target),
                    pde=pde_name, policy=policy, seed=seed,
                )
            )

    return ds_res, ds_l2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _global_feature_names(dataset: PredictorDataset) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for s in dataset.samples:
        for k in s.features:
            if k not in seen:
                names.append(k)
                seen.add(k)
    return names


def _best_policy(policies: list[str], values: NDArray[np.float64]) -> str:
    """Return policy with lowest mean value (lower is better)."""
    policy_vals: dict[str, list[float]] = {}
    for p, v in zip(policies, values, strict=False):
        policy_vals.setdefault(p, []).append(float(v))
    policy_means = {p: float(np.mean(vs)) for p, vs in policy_vals.items()}
    return min(policy_means, key=lambda k: policy_means[k])


def _policy_mean_target(
    samples: list[PredictorSample],
) -> dict[str, float]:
    """Compute mean target per policy."""
    grouped: dict[str, list[float]] = {}
    for s in samples:
        grouped.setdefault(s.policy, []).append(s.target)
    return {p: float(np.mean(vs)) for p, vs in grouped.items()}


# ---------------------------------------------------------------------------
# Part 1: Held-out PDE generalization
# ---------------------------------------------------------------------------

def run_holdout_generalization(
    base_dir: Path,
    extractor: FeatureExtractor,
) -> dict[str, object]:
    """Train RF on 10 training PDEs, test on 3 held-out PDEs."""
    print("=" * 70)
    print("PART 1: Held-out PDE Generalization")
    print("=" * 70)

    # Load training and test datasets for both metrics
    results: dict[str, object] = {}

    for metric_name in ["pde_residual_norm", "relative_l2_error"]:
        print(f"\n--- Metric: {metric_name} ---")

        train_ds = load_samples(base_dir, extractor, metric_name, pde_filter=TRAIN_PDES)
        test_ds = load_samples(base_dir, extractor, metric_name, pde_filter=TEST_PDES)

        train_pdes = train_ds.unique_pdes()
        test_pdes = test_ds.unique_pdes()
        print(f"  Train: {len(train_ds)} samples across {len(train_pdes)} PDEs: {train_pdes}")
        print(f"  Test:  {len(test_ds)} samples across {len(test_pdes)} PDEs: {test_pdes}")

        if len(train_ds) == 0 or len(test_ds) == 0:
            print("  WARNING: insufficient data, skipping")
            results[metric_name] = {"error": "insufficient data"}
            continue

        global_names = _global_feature_names(train_ds)
        X_train = train_ds.feature_matrix(feature_names=global_names)
        y_train = train_ds.target_vector()

        # Train RF on all training PDEs
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        # Evaluate per test PDE
        metric_details: list[dict] = []
        total_correct = 0
        total_regret = 0.0

        for test_pde in test_pdes:
            pde_samples = [s for s in test_ds.samples if s.pde == test_pde]
            pde_ds = PredictorDataset(samples=pde_samples)
            X_test = pde_ds.feature_matrix(feature_names=global_names)
            y_test = pde_ds.target_vector()
            y_pred = rf.predict(X_test)

            test_policies = pde_ds.policy_labels()
            actual_means = _policy_mean_target(pde_samples)
            actual_best = _best_policy(test_policies, y_test)
            predicted_best = _best_policy(test_policies, y_pred)
            oracle_target = actual_means[actual_best]
            selected_target = actual_means[predicted_best]
            regret = selected_target - oracle_target
            correct = int(actual_best == predicted_best)

            total_correct += correct
            total_regret += regret

            # Also compute physics_final heuristic
            y_phys = np.array(
                [s.features.get("physics_loss_final", 0.0) for s in pde_samples],
                dtype=np.float64,
            )
            phys_best = _best_policy(test_policies, y_phys)
            phys_target = actual_means[phys_best]
            phys_regret = phys_target - oracle_target
            phys_correct = int(actual_best == phys_best)

            # Random baseline
            unique_policies = list(set(test_policies))
            random_acc = 1.0 / len(unique_policies) if unique_policies else 0.0
            random_target = float(np.mean(list(actual_means.values())))
            random_regret = random_target - oracle_target

            detail = {
                "pde": test_pde,
                "n_samples": len(pde_samples),
                "n_policies": len(unique_policies),
                "policies": unique_policies,
                "actual_best_policy": actual_best,
                "oracle_target": oracle_target,
                "rf_predicted_best": predicted_best,
                "rf_selected_target": selected_target,
                "rf_regret": regret,
                "rf_correct": bool(correct),
                "physics_final_best": phys_best,
                "physics_final_target": phys_target,
                "physics_final_regret": phys_regret,
                "physics_final_correct": bool(phys_correct),
                "random_accuracy": random_acc,
                "random_expected_target": random_target,
                "random_regret": random_regret,
                "actual_policy_means": actual_means,
            }
            metric_details.append(detail)

            print(f"\n  {test_pde}:")
            print(f"    Oracle: {actual_best} ({oracle_target:.6f})")
            print(f"    RF:     {predicted_best} ({selected_target:.6f})  "
                  f"regret={regret:.6f}  {'CORRECT' if correct else 'WRONG'}")
            print(f"    Phys:   {phys_best} ({phys_target:.6f})  "
                  f"regret={phys_regret:.6f}  {'CORRECT' if phys_correct else 'WRONG'}")
            print(f"    Random: expected={random_target:.6f}  regret={random_regret:.6f}")

        n_test = len(test_pdes)
        rf_acc = total_correct / n_test if n_test > 0 else 0.0
        mean_regret = total_regret / n_test if n_test > 0 else 0.0

        phys_correct_total = sum(1 for d in metric_details if d["physics_final_correct"])
        phys_acc = phys_correct_total / n_test if n_test > 0 else 0.0
        mean_phys_regret = (
            sum(d["physics_final_regret"] for d in metric_details) / n_test
            if n_test > 0 else 0.0
        )
        mean_random_regret = (
            sum(d["random_regret"] for d in metric_details) / n_test
            if n_test > 0 else 0.0
        )

        print(f"\n  SUMMARY ({metric_name}):")
        print(f"    RF accuracy:           {rf_acc:.1%} ({total_correct}/{n_test})")
        print(f"    RF mean regret:        {mean_regret:.6f}")
        print(f"    physics_final acc:     {phys_acc:.1%}")
        print(f"    physics_final regret:  {mean_phys_regret:.6f}")
        print(f"    Random mean regret:    {mean_random_regret:.6f}")

        results[metric_name] = {
            "n_train_pdes": len(train_pdes),
            "n_test_pdes": n_test,
            "n_train_samples": len(train_ds),
            "n_test_samples": len(test_ds),
            "train_pdes": train_pdes,
            "test_pdes": test_pdes,
            "rf_accuracy": rf_acc,
            "rf_mean_regret": mean_regret,
            "physics_final_accuracy": phys_acc,
            "physics_final_mean_regret": mean_phys_regret,
            "random_mean_regret": mean_random_regret,
            "per_pde": metric_details,
        }

    return results


# ---------------------------------------------------------------------------
# Part 2: L2 feature ablation
# ---------------------------------------------------------------------------

def _leave_one_pde_out_accuracy(
    dataset: PredictorDataset,
    feature_names: list[str] | None = None,
) -> dict[str, object]:
    """Leave-one-PDE-out CV returning accuracy and importances."""
    global_names = feature_names if feature_names else _global_feature_names(dataset)
    unique_pdes = dataset.unique_pdes()
    pde_labels = dataset.pde_labels()

    rf_correct = 0
    physics_correct = 0
    random_acc_sum = 0.0
    n_folds = 0
    rf_importances: NDArray[np.float64] | None = None

    for held_out in unique_pdes:
        train_samples = [s for s, p in zip(dataset.samples, pde_labels, strict=False) if p != held_out]  # noqa: E501
        test_samples = [s for s, p in zip(dataset.samples, pde_labels, strict=False) if p == held_out]  # noqa: E501
        if not train_samples or not test_samples:
            continue

        train_ds = PredictorDataset(samples=train_samples)
        test_ds = PredictorDataset(samples=test_samples)

        X_train = train_ds.feature_matrix(feature_names=global_names)
        y_train = train_ds.target_vector()
        X_test = test_ds.feature_matrix(feature_names=global_names)
        y_test = test_ds.target_vector()

        rf = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        if rf_importances is None:
            rf_importances = rf.feature_importances_.copy()
        else:
            rf_importances += rf.feature_importances_

        test_policies = test_ds.policy_labels()
        actual_best = _best_policy(test_policies, y_test)
        predicted_best_rf = _best_policy(test_policies, y_pred_rf)
        rf_correct += int(actual_best == predicted_best_rf)

        y_pred_phys = np.array(
            [s.features.get("physics_loss_final", 0.0) for s in test_samples],
            dtype=np.float64,
        )
        predicted_best_phys = _best_policy(test_policies, y_pred_phys)
        physics_correct += int(actual_best == predicted_best_phys)

        unique_test_policies = set(test_policies)
        random_acc_sum += 1.0 / len(unique_test_policies) if unique_test_policies else 0.0
        n_folds += 1

    if n_folds == 0:
        return {"rf": 0.0, "physics_final": 0.0, "random": 0.0}

    if rf_importances is not None:
        rf_importances /= n_folds

    return {
        "rf": rf_correct / n_folds,
        "physics_final": physics_correct / n_folds,
        "random": random_acc_sum / n_folds,
        "_rf_importances": rf_importances,
        "_feature_names": global_names,
    }


def run_l2_feature_ablation(
    base_dir: Path,
    extractor: FeatureExtractor,
) -> dict[str, object]:
    """Feature ablation on L2 target, compared with residual."""
    print("\n" + "=" * 70)
    print("PART 2: L2 Feature Ablation (Top-K features)")
    print("=" * 70)

    results: dict[str, object] = {}

    for metric_name, label in [
        ("pde_residual_norm", "residual"),
        ("relative_l2_error", "l2"),
    ]:
        print(f"\n--- {label.upper()} metric: {metric_name} ---")
        dataset = load_samples(base_dir, extractor, metric_name)
        n_pdes = len(dataset.unique_pdes())
        print(f"  Loaded {len(dataset)} samples across {n_pdes} PDEs")

        if len(dataset) == 0:
            print("  WARNING: no samples, skipping")
            results[label] = {"error": "no samples"}
            continue

        # Full-feature pass to get importances
        full_result = _leave_one_pde_out_accuracy(dataset)
        rf_importances = full_result["_rf_importances"]
        feature_names = full_result["_feature_names"]
        n_all = len(feature_names)

        importance_order = np.argsort(rf_importances)[::-1]
        ranked_features = [
            (feature_names[i], float(rf_importances[i])) for i in importance_order
        ]

        print(f"\n  Top-10 features by RF importance ({label}):")
        for i, (name, imp) in enumerate(ranked_features[:10]):
            print(f"    {i + 1:2d}. {name:<35s} {imp:.4f}")

        # Ablation: top-K
        k_values = ABLATION_K_VALUES + [n_all]
        ablation_results: dict[str, dict] = {}

        for k in k_values:
            k_label = f"top-{k}" if k < n_all else f"all ({n_all})"
            top_k_names = [
                feature_names[importance_order[i]] for i in range(min(k, n_all))
            ]
            acc = _leave_one_pde_out_accuracy(dataset, feature_names=top_k_names)
            ablation_results[k_label] = {
                "k": min(k, n_all),
                "rf": acc["rf"],
                "physics_final": acc["physics_final"],
                "random": acc["random"],
                "features_used": top_k_names[:10],
            }
            print(
                f"  K={min(k, n_all):>3d}: RF={acc['rf']:.1%}  "
                f"physics_final={acc['physics_final']:.1%}  "
                f"random={acc['random']:.1%}"
            )

        results[label] = {
            "metric": metric_name,
            "n_pdes": n_pdes,
            "n_samples": len(dataset),
            "full_rf_accuracy": full_result["rf"],
            "ablation": ablation_results,
            "feature_ranking": ranked_features,
            "n_total_features": n_all,
        }

    # Summary comparison
    print("\n" + "-" * 70)
    print("  Comparison: Residual vs L2 at each K")
    print(f"  {'K':>6s}  {'Residual RF':>12s}  {'L2 RF':>12s}  {'Delta':>8s}")
    print("  " + "-" * 44)

    res_abl = results.get("residual", {})
    l2_abl = results.get("l2", {})
    if isinstance(res_abl, dict) and isinstance(l2_abl, dict):
        res_data = res_abl.get("ablation", {})
        l2_data = l2_abl.get("ablation", {})
        for k_label in res_data:
            res_rf = res_data[k_label]["rf"]
            l2_rf = l2_data.get(k_label, {}).get("rf", 0.0)
            delta = l2_rf - res_rf
            print(f"  {k_label:>6s}  {res_rf:>11.1%}  {l2_rf:>11.1%}  {delta:>+7.1%}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    extractor = FeatureExtractor(window_size=WINDOW_SIZE)

    # Part 1
    holdout_results = run_holdout_generalization(PAPER_A_DIR, extractor)

    # Part 2
    ablation_results = run_l2_feature_ablation(PAPER_A_DIR, extractor)

    # Save combined results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _sanitize(obj: object) -> object:
        """Make numpy types JSON-serializable."""
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, tuple):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, bool):
            return obj
        return obj

    save_data = {
        "experiment": "holdout_validation",
        "window_size": WINDOW_SIZE,
        "train_pdes": TRAIN_PDES,
        "test_pdes": TEST_PDES,
        "part1_holdout_generalization": _sanitize(holdout_results),
        "part2_l2_feature_ablation": _sanitize(ablation_results),
    }

    output_path = OUTPUT_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {output_path}")

    elapsed = time.time() - t_start
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
