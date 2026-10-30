#!/usr/bin/env python3
"""Scaling analysis and feature ablation for Paper A routing evaluation.

Part 1 — Benchmark scaling: How routing accuracy changes as PDE diversity
grows from 5 to 9 to 13 PDEs. Evaluates RF, physics_final heuristic, and
random baseline at each scale using leave-one-PDE-out cross-validation.

Part 2 — Feature ablation: How routing accuracy changes when restricting
to top-K features (K = 3, 5, 10, 20, all) ranked by RF feature importance.

Outputs:
    experiments/analysis/figures/fig_scaling.pdf
    experiments/analysis/figures/fig_feature_ablation.pdf
    experiments/results/paper_a/scaling_ablation/results.json

Usage:
    python experiments/scripts/paper_a_routing/scaling_and_ablation.py
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

import matplotlib

from causal_pinn.training.early_predictor import (
    FeatureExtractor,
    PredictorDataset,
    PredictorSample,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAPER_A_DIR = Path(__file__).resolve().parents[3] / "experiments" / "results" / "paper_a"
FIGURES_DIR = Path(__file__).resolve().parents[3] / "experiments" / "analysis" / "figures"
OUTPUT_DIR = PAPER_A_DIR / "scaling_ablation"
SKIP_DIRS = {"bandit_evaluation", "routing_evaluation_full", "scaling_ablation"}
WINDOW_SIZE = 50
TARGET_METRIC = "pde_residual_norm"

# PDE subsets for scaling analysis
PDE_SUBSETS = {
    "5 PDEs": ["heat1d", "burgers1d", "advdiff1d", "kdv1d", "wave2d"],
    "9 PDEs": [
        "heat1d", "burgers1d", "advdiff1d", "kdv1d", "wave2d",
        "allencahn1d", "burgers1d_lownu", "advdiff1d_highpe", "reaction_diffusion1d",
    ],
    "13 PDEs": [
        "heat1d", "burgers1d", "advdiff1d", "kdv1d", "wave2d",
        "allencahn1d", "burgers1d_lownu", "advdiff1d_highpe", "reaction_diffusion1d",
        "allencahn1d_sharp", "burgers1d_verylow", "reaction_diffusion1d_stiff",
        "navier_stokes2d",
    ],
}

# Feature ablation K values
ABLATION_K_VALUES = [3, 5, 10, 20]  # "all" is added dynamically


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_samples(
    base_dir: Path,
    extractor: FeatureExtractor,
    target_metric: str,
    pde_filter: list[str] | None = None,
) -> PredictorDataset:
    """Load Paper A results into a PredictorDataset.

    Args:
        base_dir: Path to experiments/results/paper_a/.
        extractor: FeatureExtractor for loss curve features.
        target_metric: Which metric to use as target.
        pde_filter: If given, only load these PDE names.

    Returns:
        PredictorDataset with one sample per (PDE, policy, seed).
    """
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _best_policy(policies: list[str], values: NDArray[np.float64]) -> str:
    """Return policy with lowest mean value (lower is better)."""
    policy_vals: dict[str, list[float]] = {}
    for p, v in zip(policies, values, strict=False):
        policy_vals.setdefault(p, []).append(float(v))
    policy_means = {p: float(np.mean(vs)) for p, vs in policy_vals.items()}
    return min(policy_means, key=lambda k: policy_means[k])


def _global_feature_names(dataset: PredictorDataset) -> list[str]:
    """Get ordered union of all feature names in dataset."""
    names: list[str] = []
    seen: set[str] = set()
    for s in dataset.samples:
        for k in s.features:
            if k not in seen:
                names.append(k)
                seen.add(k)
    return names


def leave_one_pde_out_accuracy(
    dataset: PredictorDataset,
    feature_names: list[str] | None = None,
) -> dict[str, float]:
    """Run leave-one-PDE-out CV and return accuracy for RF, physics_final, random.

    Args:
        dataset: Full dataset.
        feature_names: If given, restrict features to these columns.

    Returns:
        Dict with keys "rf", "physics_final", "random" mapping to accuracy.
    """
    from sklearn.ensemble import RandomForestRegressor

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

        # RF
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        # Accumulate importances
        if rf_importances is None:
            rf_importances = rf.feature_importances_.copy()
        else:
            rf_importances += rf.feature_importances_

        test_policies = test_ds.policy_labels()
        actual_best = _best_policy(test_policies, y_test)
        predicted_best_rf = _best_policy(test_policies, y_pred_rf)
        rf_correct += int(actual_best == predicted_best_rf)

        # Physics-final heuristic
        y_pred_phys = np.array(
            [s.features.get("physics_loss_final", 0.0) for s in test_samples],
            dtype=np.float64,
        )
        predicted_best_phys = _best_policy(test_policies, y_pred_phys)
        physics_correct += int(actual_best == predicted_best_phys)

        # Random baseline: 1/num_policies
        unique_test_policies = set(test_policies)
        random_acc_sum += 1.0 / len(unique_test_policies) if unique_test_policies else 0.0

        n_folds += 1

    if n_folds == 0:
        return {"rf": 0.0, "physics_final": 0.0, "random": 0.0}

    # Average importances
    if rf_importances is not None:
        rf_importances /= n_folds

    return {
        "rf": rf_correct / n_folds,
        "physics_final": physics_correct / n_folds,
        "random": random_acc_sum / n_folds,
        "_rf_importances": rf_importances,  # type: ignore[dict-item]
        "_feature_names": global_names,
    }


# ---------------------------------------------------------------------------
# Part 1: Scaling analysis
# ---------------------------------------------------------------------------

def run_scaling_analysis(
    base_dir: Path,
    extractor: FeatureExtractor,
) -> dict[str, dict]:
    """Evaluate routing accuracy at each PDE subset scale."""
    results: dict[str, dict] = {}

    for label, pde_list in PDE_SUBSETS.items():
        print(f"\n--- {label} ---")
        dataset = load_all_samples(base_dir, extractor, TARGET_METRIC, pde_filter=pde_list)
        actual_pdes = dataset.unique_pdes()
        print(f"  Loaded {len(dataset)} samples across {len(actual_pdes)} PDEs: {actual_pdes}")

        if len(dataset) == 0:
            print(f"  WARNING: No samples loaded for {label}, skipping")
            results[label] = {"n_pdes": 0, "n_samples": 0, "rf": 0.0, "physics_final": 0.0, "random": 0.0}  # noqa: E501
            continue

        acc = leave_one_pde_out_accuracy(dataset)
        results[label] = {
            "n_pdes": len(actual_pdes),
            "n_samples": len(dataset),
            "rf": acc["rf"],
            "physics_final": acc["physics_final"],
            "random": acc["random"],
            "pdes": actual_pdes,
        }
        print(f"  RF: {acc['rf']:.1%}  |  physics_final: {acc['physics_final']:.1%}  |  random: {acc['random']:.1%}")  # noqa: E501

    return results


# ---------------------------------------------------------------------------
# Part 2: Feature ablation
# ---------------------------------------------------------------------------

def run_feature_ablation(
    base_dir: Path,
    extractor: FeatureExtractor,
) -> dict[str, dict]:
    """Evaluate routing accuracy with top-K features."""
    print("\n--- Feature ablation (full 13-PDE dataset) ---")
    dataset = load_all_samples(base_dir, extractor, TARGET_METRIC)
    print(f"  Loaded {len(dataset)} samples across {len(dataset.unique_pdes())} PDEs")

    if len(dataset) == 0:
        print("  WARNING: No samples loaded, skipping feature ablation")
        return {}

    # Nested CV: feature selection happens INSIDE each fold to prevent
    # held-out PDE information from leaking into the feature ranking.
    # For each fold, train RF on the train split, extract importances,
    # select top-K, and evaluate on the held-out PDE.
    from sklearn.ensemble import RandomForestRegressor

    all_samples = dataset.samples
    pde_groups = sorted(set(s.pde for s in all_samples))
    # Get global feature names from the union of all sample keys
    all_keys: set[str] = set()
    for s in all_samples:
        all_keys.update(s.features.keys())
    global_names = sorted(all_keys)
    n_all = len(global_names)

    # Accumulate per-fold importances for reporting (post-hoc, not used for selection)
    avg_importances = np.zeros(n_all)

    k_values = ABLATION_K_VALUES + [n_all]
    fold_results: dict[int, dict[str, int]] = {k: {"correct": 0} for k in k_values}
    n_folds = 0

    for held_out in pde_groups:
        train_samples = [s for s in all_samples if s.pde != held_out]
        test_samples = [s for s in all_samples if s.pde == held_out]
        if not train_samples or not test_samples:
            continue

        train_ds = PredictorDataset(samples=train_samples)
        test_ds = PredictorDataset(samples=test_samples)

        X_train = train_ds.feature_matrix(feature_names=global_names)
        y_train = train_ds.target_vector()
        X_test = test_ds.feature_matrix(feature_names=global_names)
        y_test = test_ds.target_vector()

        # Train RF on train split to get fold-specific importances
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        fold_imp = rf.feature_importances_
        avg_importances += fold_imp
        fold_order = np.argsort(fold_imp)[::-1]

        test_policies = test_ds.policy_labels()
        actual_best = _best_policy(test_policies, y_test)

        for k in k_values:
            k_use = min(k, n_all)
            # Select top-K features from THIS fold's train-only importances
            selected_idx = fold_order[:k_use]
            X_train_k = X_train[:, selected_idx]
            X_test_k = X_test[:, selected_idx]

            rf_k = RandomForestRegressor(
                n_estimators=200, max_depth=8, min_samples_leaf=3,
                random_state=42, n_jobs=-1,
            )
            rf_k.fit(X_train_k, y_train)
            y_pred = rf_k.predict(X_test_k)
            predicted_best = _best_policy(test_policies, y_pred)
            if actual_best == predicted_best:
                fold_results[k]["correct"] += 1

        n_folds += 1

    if n_folds == 0:
        return {}

    avg_importances /= n_folds
    importance_order = np.argsort(avg_importances)[::-1]
    ranked_features = [
        (global_names[i], float(avg_importances[i])) for i in importance_order
    ]

    print("\n  Top-10 features (averaged across folds, post-hoc):")
    for i, (name, imp) in enumerate(ranked_features[:10]):
        print(f"    {i+1:2d}. {name:<35s} {imp:.4f}")

    results: dict[str, dict] = {}
    for k in k_values:
        k_label = f"top-{k}" if k < n_all else f"all ({n_all})"
        acc = fold_results[k]["correct"] / n_folds
        results[k_label] = {
            "k": min(k, n_all),
            "rf": acc,
            "physics_final": 0.0,  # heuristic doesn't use features
            "random": 0.0,
        }
        print(f"  K={min(k, n_all):>3d}: RF={acc:.1%} (nested CV, no leakage)")

    return {
        "ablation": results,
        "feature_ranking": ranked_features,
        "n_total_features": n_all,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scaling(scaling_results: dict[str, dict], output_path: Path) -> None:
    """Generate fig_scaling.pdf: accuracy vs number of PDEs."""
    labels = list(scaling_results.keys())
    n_pdes = [scaling_results[l]["n_pdes"] for l in labels]  # noqa: E741
    rf_acc = [scaling_results[l]["rf"] for l in labels]  # noqa: E741
    phys_acc = [scaling_results[l]["physics_final"] for l in labels]  # noqa: E741
    rand_acc = [scaling_results[l]["random"] for l in labels]  # noqa: E741

    fig, ax = plt.subplots(figsize=(5.5, 4))

    ax.plot(n_pdes, rf_acc, "o-", color="#2563eb", linewidth=2, markersize=8, label="Random Forest")
    ax.plot(n_pdes, phys_acc, "s--", color="#dc2626", linewidth=2, markersize=8, label="physics_final heuristic")  # noqa: E501
    ax.plot(n_pdes, rand_acc, "^:", color="#6b7280", linewidth=2, markersize=8, label="Random baseline")  # noqa: E501

    ax.set_xlabel("Number of PDEs in benchmark", fontsize=12)
    ax.set_ylabel("Routing accuracy", fontsize=12)
    ax.set_title("Routing accuracy vs. PDE diversity", fontsize=13)
    ax.set_xticks(n_pdes)
    ax.set_xticklabels([str(n) for n in n_pdes])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    # Annotate values
    for x, y in zip(n_pdes, rf_acc, strict=False):
        ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color="#2563eb")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {output_path}")


def plot_feature_ablation(ablation_results: dict[str, dict], output_path: Path) -> None:
    """Generate fig_feature_ablation.pdf: accuracy vs number of features."""
    ablation = ablation_results["ablation"]
    labels = list(ablation.keys())
    k_vals = [ablation[l]["k"] for l in labels]  # noqa: E741
    rf_acc = [ablation[l]["rf"] for l in labels]  # noqa: E741
    phys_acc = [ablation[l]["physics_final"] for l in labels]  # noqa: E741

    fig, ax = plt.subplots(figsize=(5.5, 4))

    ax.plot(k_vals, rf_acc, "o-", color="#2563eb", linewidth=2, markersize=8, label="Random Forest")
    ax.axhline(y=phys_acc[0], color="#dc2626", linestyle="--", linewidth=1.5,
               label="physics_final heuristic")
    ax.axhline(y=ablation[labels[0]]["random"], color="#6b7280", linestyle=":",
               linewidth=1.5, label="Random baseline")

    ax.set_xlabel("Number of features (ranked by importance)", fontsize=12)
    ax.set_ylabel("Routing accuracy", fontsize=12)
    ax.set_title("Feature ablation: routing accuracy vs. feature count", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    # Annotate RF values
    for x, y in zip(k_vals, rf_acc, strict=False):
        ax.annotate(f"{y:.0%}", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color="#2563eb")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    extractor = FeatureExtractor(window_size=WINDOW_SIZE)

    print("=" * 70)
    print("PART 1: Benchmark Scaling Analysis")
    print("=" * 70)
    scaling_results = run_scaling_analysis(PAPER_A_DIR, extractor)

    print("\n" + "=" * 70)
    print("PART 2: Feature Ablation Study")
    print("=" * 70)
    ablation_results = run_feature_ablation(PAPER_A_DIR, extractor)

    # --- Plots ---
    print("\n" + "=" * 70)
    print("Generating figures...")
    print("=" * 70)
    plot_scaling(scaling_results, FIGURES_DIR / "fig_scaling.pdf")
    plot_feature_ablation(ablation_results, FIGURES_DIR / "fig_feature_ablation.pdf")

    # --- Save JSON ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_data = {
        "experiment": "scaling_and_ablation",
        "target_metric": TARGET_METRIC,
        "window_size": WINDOW_SIZE,
        "scaling": {
            label: {
                "n_pdes": r["n_pdes"],
                "n_samples": r["n_samples"],
                "rf_accuracy": r["rf"],
                "physics_final_accuracy": r["physics_final"],
                "random_accuracy": r["random"],
                "pdes": r.get("pdes", []),
            }
            for label, r in scaling_results.items()
        },
        "feature_ablation": {
            label: {
                "k": v["k"],
                "rf_accuracy": v["rf"],
                "physics_final_accuracy": v["physics_final"],
                "random_accuracy": v["random"],
            }
            for label, v in ablation_results.get("ablation", {}).items()
        },
        "feature_ranking": [
            {"feature": name, "importance": imp}
            for name, imp in ablation_results.get("feature_ranking", [])
        ],
    }
    output_path = OUTPUT_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {output_path}")

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
