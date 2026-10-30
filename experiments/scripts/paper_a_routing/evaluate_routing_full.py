#!/usr/bin/env python3
"""Full routing evaluation across ALL Paper A experiments (450 decisions, 13 PDEs).

Loads results from experiments/results/paper_a/*/results.json, builds a
PredictorDataset using FeatureExtractor(window_size=50), and runs
leave-one-PDE-out cross-validation with multiple models:

    - RandomForest
    - GradientBoosting
    - Ridge regression
    - physics_final heuristic (baseline: pick policy with lowest final physics_loss)

Reports both pde_residual_norm and relative_l2_error routing accuracy.

Usage:
    python experiments/scripts/paper_a_routing/evaluate_routing_full.py
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

from causal_pinn.training.early_predictor import (
    FeatureExtractor,
    PredictorDataset,
    PredictorSample,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAPER_A_DIR = Path(__file__).resolve().parents[3] / "experiments" / "results" / "paper_a"
OUTPUT_DIR = PAPER_A_DIR / "routing_evaluation_full"
SKIP_DIRS = {"bandit_evaluation", "routing_evaluation_full"}
WINDOW_SIZE = 50
TARGET_METRICS = ["pde_residual_norm", "relative_l2_error"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_results(
    base_dir: Path,
    extractor: FeatureExtractor,
    target_metric: str,
) -> PredictorDataset:
    """Load all Paper A results into a PredictorDataset."""
    dataset = PredictorDataset()
    skipped = 0
    loaded = 0

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

    print(f"  Loaded {loaded} samples, skipped {skipped} "
          f"(target_metric={target_metric})")
    return dataset


# ---------------------------------------------------------------------------
# Physics-final heuristic baseline
# ---------------------------------------------------------------------------

def physics_final_heuristic_predictions(dataset: PredictorDataset) -> NDArray[np.float64]:
    """Use the final physics_loss from features as a proxy for target.

    This is a simple baseline: rank policies by their physics_loss_final
    feature value. We return these values as 'predictions' so that the
    policy-selection logic works the same way (pick argmin).
    """
    return np.array(
        [s.features.get("physics_loss_final", 0.0) for s in dataset.samples],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Leave-one-PDE-out cross-validation
# ---------------------------------------------------------------------------

def leave_one_pde_out_cv(
    dataset: PredictorDataset,
    model_factories: dict[str, object],
    target_metric: str,
    group_of: dict[str, str] | None = None,
) -> dict[str, dict]:
    """Run leave-one-PDE-out CV for each model.

    With ``group_of`` (PDE -> family), each fold still tests one PDE but also
    drops every PDE of the same family from training: a true leave-one-family-out
    evaluation, scored per held-out PDE.

    Returns dict mapping model_name -> results dict.
    """
    def _group(pde: str) -> str:
        return group_of[pde] if group_of else pde

    unique_pdes = dataset.unique_pdes()
    pde_labels = dataset.pde_labels()

    # Global feature names for consistent matrix shape
    global_names: list[str] = []
    seen: set[str] = set()
    for s in dataset.samples:
        for k in s.features:
            if k not in seen:
                global_names.append(k)
                seen.add(k)

    all_results: dict[str, dict] = {}

    for model_name, factory in model_factories.items():
        fold_details = []
        policy_correct = 0
        r2_scores = []

        for held_out in unique_pdes:
            train_samples = [s for s, p in zip(dataset.samples, pde_labels, strict=False)
                             if _group(p) != _group(held_out)]
            test_samples = [s for s, p in zip(dataset.samples, pde_labels, strict=False) if p == held_out]  # noqa: E501

            if not train_samples or not test_samples:
                continue

            train_ds = PredictorDataset(samples=train_samples)
            test_ds = PredictorDataset(samples=test_samples)

            X_train = train_ds.feature_matrix(feature_names=global_names)
            y_train = train_ds.target_vector()
            X_test = test_ds.feature_matrix(feature_names=global_names)
            y_test = test_ds.target_vector()

            # Predict
            if model_name == "physics_final":
                y_pred = physics_final_heuristic_predictions(test_ds)
            else:
                model = factory()  # type: ignore[operator]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # R2
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
            r2_scores.append(r2)

            # Policy selection accuracy: predicted best vs actual best
            test_policies = test_ds.policy_labels()
            actual_best = _best_policy(test_policies, y_test)
            predicted_best = _best_policy(test_policies, y_pred)
            is_correct = actual_best == predicted_best
            policy_correct += int(is_correct)

            fold_details.append({
                "held_out_pde": held_out,
                "n_test": len(test_samples),
                "n_train_pdes": len({p for p in pde_labels if _group(p) != _group(held_out)}),
                "r2": round(r2, 4),
                "actual_best": actual_best,
                "predicted_best": predicted_best,
                "correct": is_correct,
            })

        n_folds = len(fold_details)
        routing_acc = policy_correct / n_folds if n_folds > 0 else 0.0
        mean_r2 = float(np.mean(r2_scores)) if r2_scores else 0.0

        all_results[model_name] = {
            "routing_accuracy": round(routing_acc, 4),
            "correct_folds": policy_correct,
            "total_folds": n_folds,
            "mean_r2": round(mean_r2, 4),
            "fold_details": fold_details,
        }

    return all_results


def _best_policy(policies: list[str], values: NDArray[np.float64]) -> str:
    """Return policy with lowest mean value (lower is better)."""
    policy_vals: dict[str, list[float]] = {}
    for p, v in zip(policies, values, strict=False):
        policy_vals.setdefault(p, []).append(float(v))
    policy_means = {p: float(np.mean(vs)) for p, vs in policy_vals.items()}
    return min(policy_means, key=lambda k: policy_means[k])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Full routing evaluation across 13 PDEs (450 decisions).",
    )
    parser.add_argument(
        "--window-size", type=int, default=WINDOW_SIZE,
        help=f"Probe window size in epochs (default: {WINDOW_SIZE})",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/results/paper_a/routing_evaluation_full",
        help="Output directory for results JSON",
    )
    parser.add_argument(
        "--holdout", choices=("pde", "family"), default="pde",
        help="Hold out one PDE per fold (default) or its whole family from training",
    )
    args = parser.parse_args()
    group_of = None
    if args.holdout == "family":
        from family_holdout_analysis import _PDE_TO_FAMILY  # type: ignore[import-not-found]

        group_of = _PDE_TO_FAMILY

    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge

    extractor = FeatureExtractor(window_size=args.window_size)

    model_factories = {
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        ),
        "GradientBoosting": lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=3,
            learning_rate=0.05, random_state=42,
        ),
        "Ridge": lambda: Ridge(alpha=1.0),
        "physics_final": None,  # heuristic, handled specially
    }

    all_metric_results: dict[str, dict] = {}

    for target_metric in TARGET_METRICS:
        print(f"\n{'='*70}")
        print(f"TARGET METRIC: {target_metric}")
        print(f"{'='*70}")

        t0 = time.time()
        dataset = load_all_results(PAPER_A_DIR, extractor, target_metric)
        print(f"  Dataset: {len(dataset)} samples, "
              f"{len(dataset.unique_pdes())} PDEs, "
              f"PDEs: {dataset.unique_pdes()}")

        results = leave_one_pde_out_cv(dataset, model_factories, target_metric, group_of)
        elapsed = time.time() - t0

        # Print comparison table
        print(f"\n  Leave-one-PDE-out routing results ({target_metric}):")
        print(f"  {'Model':<25} {'Routing Acc':>12} {'Correct':>8} {'Folds':>6} {'Mean R2':>9}")
        print(f"  {'-'*25} {'-'*12} {'-'*8} {'-'*6} {'-'*9}")
        for model_name, res in results.items():
            print(f"  {model_name:<25} {res['routing_accuracy']:>12.1%} "
                  f"{res['correct_folds']:>8} {res['total_folds']:>6} "
                  f"{res['mean_r2']:>9.4f}")

        # Per-fold details for best model
        best_model = max(results, key=lambda k: results[k]["routing_accuracy"])
        print(f"\n  Per-PDE breakdown ({best_model}):")
        print(f"  {'Held-out PDE':<30} {'Actual Best':<25} {'Predicted Best':<25} {'OK?':>4}")
        print(f"  {'-'*30} {'-'*25} {'-'*25} {'-'*4}")
        for fold in results[best_model]["fold_details"]:
            mark = "Y" if fold["correct"] else "N"
            print(f"  {fold['held_out_pde']:<30} {fold['actual_best']:<25} "
                  f"{fold['predicted_best']:<25} {mark:>4}")

        print(f"\n  Elapsed: {elapsed:.1f}s")
        all_metric_results[target_metric] = results

    # Save results
    # Honour --output-dir / --window-size: writing to the module-level OUTPUT_DIR
    # while recording the module-level WINDOW_SIZE would silently overwrite the
    # canonical w=50 artifact with results from a different probe window.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"

    save_data = {
        "experiment": "routing_evaluation_full",
        "description": (
            "Leave-one-family-out routing CV across 13 PDEs (450 decisions): each fold "
            "tests one PDE with its whole family removed from training"
            if args.holdout == "family"
            else "Leave-one-PDE-out routing CV across 13 PDEs (450 decisions)"
        ),
        "holdout": args.holdout,
        "window_size": args.window_size,
        "n_samples": 450,
        "n_pdes": 13,
        "target_metrics": TARGET_METRICS,
        "results": all_metric_results,
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Final summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Routing Accuracy (leave-one-PDE-out)")
    print(f"{'='*70}")
    print(f"{'Model':<25} {'PDE-residual':>14} {'Relative-L2':>14}")
    print(f"{'-'*25} {'-'*14} {'-'*14}")
    model_names = list(next(iter(all_metric_results.values())).keys())
    for mn in model_names:
        res_acc = all_metric_results["pde_residual_norm"][mn]["routing_accuracy"]
        l2_acc = all_metric_results["relative_l2_error"][mn]["routing_accuracy"]
        print(f"{mn:<25} {res_acc:>14.1%} {l2_acc:>14.1%}")


if __name__ == "__main__":
    main()
