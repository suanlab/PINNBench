#!/usr/bin/env python3
"""Evaluate contextual bandit and successive halving routers against
existing RF and heuristic baselines for Paper A policy selection.

The script:
1. Loads all Paper A results from experiments/results/paper_a/
2. Constructs the routing dataset (same format as early_predictor)
3. Evaluates LinUCB, SuccessiveHalving, RF, and heuristic on
   leave-one-PDE-out cross-validation
4. Compares accuracy, regret, and budget savings
5. Saves results to experiments/results/paper_a/bandit_evaluation/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from causal_pinn.training.contextual_bandit import (  # noqa: E402
    LinUCBRouter,
    SuccessiveHalvingRouter,
)
from causal_pinn.training.early_predictor import (  # noqa: E402
    FeatureExtractor,
    PredictorDataset,
    PredictorSample,
    extract_runs,
    get_final_metrics,
    get_training_history,
    load_experiment_results,
)
from causal_pinn.training.policy_routing import (  # noqa: E402
    SelectorFoldResult,
    budget_ratio,
    budget_savings,
    choose_lowest_policy,
    dataset_feature_names,
    heuristic_policy_choice,
    normalize_policy_name,
    policy_mean_values,
    summarize_fold_results,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "paper_a"
OUTPUT_DIR = RESULTS_DIR / "bandit_evaluation"
OUTPUT_FILE = OUTPUT_DIR / "results.json"

TARGET_METRIC = "relative_l2_error"
WINDOW_SIZE = 50
TOTAL_EPOCHS = 500
PROBE_EPOCHS = 50

# LinUCB alpha sweep values
LINUCB_ALPHAS = [0.1, 0.5, 1.0, 2.0]


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def build_paper_a_dataset(
    results_dir: Path,
    target_metric: str = TARGET_METRIC,
    window_size: int = WINDOW_SIZE,
) -> PredictorDataset:
    """Load all Paper A experiment results and build a PredictorDataset.

    Scans every subdirectory of *results_dir* for a ``results.json`` file
    and extracts per-run features from the first *window_size* epochs of
    stage-2 loss history.
    """
    extractor = FeatureExtractor(window_size=window_size)
    dataset = PredictorDataset()

    for pde_dir in sorted(results_dir.iterdir()):
        results_path = pde_dir / "results.json"
        if not results_path.is_file():
            continue

        pde_name = pde_dir.name
        data = load_experiment_results(results_path)
        runs = extract_runs(data)

        for run in runs:
            history = get_training_history(run)
            stage2 = history.get("stage2_losses", [])
            if not stage2:
                continue

            metrics = get_final_metrics(run)
            target = metrics.get(target_metric)
            if target is None:
                continue

            policy = run.get("model_name") or run.get("condition") or "unknown"
            seed = run.get("seed", 0)

            features = extractor.extract(stage2)
            if not features:
                continue

            dataset.samples.append(
                PredictorSample(
                    features=features,
                    target=float(target),
                    pde=pde_name,
                    policy=policy,
                    seed=seed,
                )
            )

    return dataset


# ---------------------------------------------------------------------------
# Per-PDE loss trajectory helpers (for SuccessiveHalving)
# ---------------------------------------------------------------------------


def build_policy_trajectories(
    samples: list[PredictorSample],
    loss_key: str = "total_mean",
) -> dict[str, list[float]]:
    """Aggregate per-policy mean target across seeds for SH simulation.

    Since SuccessiveHalving needs epoch-level trajectories but we only
    store summary features, we synthesise a monotone proxy from the
    ``total_final`` feature (the stage-2 total loss at the window edge).
    For a proper evaluation the full loss curves would be loaded; here we
    approximate with a single-point trajectory so SH degenerates to a
    one-round comparison at the probe epoch, which is a fair baseline.
    """
    policy_targets: dict[str, list[float]] = {}
    for sample in samples:
        policy = normalize_policy_name(sample.policy)
        # Use the final total loss as a single-epoch "trajectory"
        value = sample.features.get("total_final", sample.target)
        policy_targets.setdefault(policy, []).append(float(value))
    # Average across seeds for each policy
    return {
        policy: [float(np.mean(values))]
        for policy, values in policy_targets.items()
    }


def build_full_trajectories_from_dir(
    results_dir: Path,
    pde_name: str,
) -> dict[str, dict[int, list[float]]]:
    """Load full stage-2 total loss trajectories from raw results.

    Returns:
        Nested dict: policy -> seed -> list of per-epoch total losses.
    """
    results_path = results_dir / pde_name / "results.json"
    if not results_path.is_file():
        return {}

    data = load_experiment_results(results_path)
    runs = extract_runs(data)
    trajectories: dict[str, dict[int, list[float]]] = {}

    for run in runs:
        policy = run.get("model_name") or run.get("condition") or "unknown"
        seed = run.get("seed", 0)
        history = get_training_history(run)
        stage2 = history.get("stage2_losses", [])
        if not stage2:
            continue
        totals = [ep.get("total", 0.0) for ep in stage2]
        trajectories.setdefault(policy, {})[seed] = totals

    return trajectories


def mean_trajectory(seed_trajectories: dict[int, list[float]]) -> list[float]:
    """Average trajectories across seeds, handling different lengths."""
    if not seed_trajectories:
        return []
    max_len = max(len(t) for t in seed_trajectories.values())
    result: list[float] = []
    for i in range(max_len):
        vals = [t[i] for t in seed_trajectories.values() if i < len(t)]
        result.append(float(np.mean(vals)))
    return result


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_linucb_fold(
    train_samples: list[PredictorSample],
    test_samples: list[PredictorSample],
    global_names: list[str],
    alpha: float,
    held_out_group: str,
    probe_epochs: int,
    total_epochs: int,
) -> SelectorFoldResult:
    """Run LinUCB on one leave-one-PDE-out fold."""
    n_features = len(global_names)
    router = LinUCBRouter(n_features=n_features, alpha=alpha)

    # Build training contexts and rewards
    for sample in train_samples:
        ctx = np.array([sample.features.get(k, 0.0) for k in global_names])
        arm = normalize_policy_name(sample.policy)
        reward = -sample.target  # lower metric = higher reward
        router.update(arm, ctx, reward)

    # Evaluate on test: for each sample get predicted reward, group by policy
    y_test = np.array([s.target for s in test_samples])
    actual_policy_targets = policy_mean_values(test_samples, y_test)
    actual_best_policy = choose_lowest_policy(actual_policy_targets)
    oracle_target = actual_policy_targets[actual_best_policy]
    candidate_policies = list(actual_policy_targets.keys())

    # Predict: average the exploitation values across test samples per policy
    policy_scores: dict[str, list[float]] = {}
    for sample in test_samples:
        ctx = np.array([sample.features.get(k, 0.0) for k in global_names])
        values = router.predict_values(ctx, candidate_policies)
        for policy, val in values.items():
            policy_scores.setdefault(policy, []).append(val)

    policy_mean_rewards = {
        p: float(np.mean(scores)) for p, scores in policy_scores.items()
    }
    # Highest predicted reward = best policy
    selected_policy = max(policy_mean_rewards, key=lambda k: policy_mean_rewards[k])
    selected_target = actual_policy_targets.get(selected_policy, float("inf"))

    fold_br = budget_ratio(len(candidate_policies), probe_epochs, total_epochs)
    fold_bs = budget_savings(len(candidate_policies), probe_epochs, total_epochs)

    return SelectorFoldResult(
        selector=f"linucb_alpha{alpha}",
        held_out_group=held_out_group,
        selected_policy=selected_policy,
        actual_best_policy=actual_best_policy,
        selected_target=selected_target,
        oracle_target=oracle_target,
        regret=selected_target - oracle_target,
        correct=float(selected_policy == actual_best_policy),
        num_policies=len(candidate_policies),
        probe_epochs=probe_epochs,
        total_epochs=total_epochs,
        budget_ratio=fold_br,
        budget_savings=fold_bs,
    )


def evaluate_successive_halving_fold(
    test_samples: list[PredictorSample],
    pde_name: str,
    results_dir: Path,
    held_out_group: str,
    total_epochs: int,
    probe_epochs: int,
) -> SelectorFoldResult:
    """Run successive halving on one leave-one-PDE-out fold."""
    y_test = np.array([s.target for s in test_samples])
    actual_policy_targets = policy_mean_values(test_samples, y_test)
    actual_best_policy = choose_lowest_policy(actual_policy_targets)
    oracle_target = actual_policy_targets[actual_best_policy]
    candidate_policies = list(actual_policy_targets.keys())

    # Load full trajectories for this PDE
    raw_trajectories = build_full_trajectories_from_dir(results_dir, pde_name)
    if raw_trajectories:
        policy_trajs = {
            normalize_policy_name(p): mean_trajectory(seeds)
            for p, seeds in raw_trajectories.items()
        }
    else:
        # Fallback to single-point trajectories
        policy_trajs = build_policy_trajectories(test_samples)

    sh = SuccessiveHalvingRouter(total_epochs=total_epochs, min_probe=10)
    selected_policy, sh_budget = sh.select_policy(policy_trajs)
    selected_policy = normalize_policy_name(selected_policy)

    if selected_policy not in actual_policy_targets:
        # Fallback to the closest normalized name
        for p in actual_policy_targets:
            if normalize_policy_name(p) == selected_policy:
                selected_policy = p
                break

    selected_target = actual_policy_targets.get(selected_policy, float("inf"))

    # SH budget savings: sh_budget vs exhaustive
    exhaustive = len(candidate_policies) * total_epochs
    sh_savings = 1.0 - (sh_budget / exhaustive) if exhaustive > 0 else 0.0

    budget_ratio(len(candidate_policies), probe_epochs, total_epochs)

    return SelectorFoldResult(
        selector="successive_halving",
        held_out_group=held_out_group,
        selected_policy=selected_policy,
        actual_best_policy=actual_best_policy,
        selected_target=selected_target,
        oracle_target=oracle_target,
        regret=selected_target - oracle_target,
        correct=float(selected_policy == actual_best_policy),
        num_policies=len(candidate_policies),
        probe_epochs=probe_epochs,
        total_epochs=total_epochs,
        budget_ratio=1.0 - sh_savings,
        budget_savings=sh_savings,
    )


def evaluate_random_forest_fold(
    train_samples: list[PredictorSample],
    test_samples: list[PredictorSample],
    global_names: list[str],
    held_out_group: str,
    probe_epochs: int,
    total_epochs: int,
) -> SelectorFoldResult:
    """Run Random Forest on one leave-one-PDE-out fold."""
    from sklearn.ensemble import RandomForestRegressor

    train_ds = PredictorDataset(samples=train_samples)
    test_ds = PredictorDataset(samples=test_samples)

    X_train = train_ds.feature_matrix(feature_names=global_names)
    y_train = train_ds.target_vector()
    X_test = test_ds.feature_matrix(feature_names=global_names)
    y_test = test_ds.target_vector()

    actual_policy_targets = policy_mean_values(test_samples, y_test)
    actual_best_policy = choose_lowest_policy(actual_policy_targets)
    oracle_target = actual_policy_targets[actual_best_policy]
    candidate_policies = list(actual_policy_targets.keys())

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    pred_policy_targets = policy_mean_values(test_samples, y_pred)
    rf_policy = choose_lowest_policy(pred_policy_targets)
    rf_target = actual_policy_targets.get(rf_policy, float("inf"))

    fold_br = budget_ratio(len(candidate_policies), probe_epochs, total_epochs)
    fold_bs = budget_savings(len(candidate_policies), probe_epochs, total_epochs)

    return SelectorFoldResult(
        selector="random_forest",
        held_out_group=held_out_group,
        selected_policy=rf_policy,
        actual_best_policy=actual_best_policy,
        selected_target=rf_target,
        oracle_target=oracle_target,
        regret=rf_target - oracle_target,
        correct=float(rf_policy == actual_best_policy),
        num_policies=len(candidate_policies),
        probe_epochs=probe_epochs,
        total_epochs=total_epochs,
        budget_ratio=fold_br,
        budget_savings=fold_bs,
    )


def evaluate_heuristic_fold(
    test_samples: list[PredictorSample],
    held_out_group: str,
    probe_epochs: int,
    total_epochs: int,
) -> SelectorFoldResult:
    """Run physics_final heuristic on one leave-one-PDE-out fold."""
    y_test = np.array([s.target for s in test_samples])
    actual_policy_targets = policy_mean_values(test_samples, y_test)
    actual_best_policy = choose_lowest_policy(actual_policy_targets)
    oracle_target = actual_policy_targets[actual_best_policy]
    candidate_policies = list(actual_policy_targets.keys())

    heuristic_policy = heuristic_policy_choice(test_samples, "physics_loss_final")
    if heuristic_policy == "unknown" or heuristic_policy not in actual_policy_targets:
        heuristic_policy = list(actual_policy_targets.keys())[0]

    heuristic_target = actual_policy_targets.get(heuristic_policy, float("inf"))

    fold_br = budget_ratio(len(candidate_policies), probe_epochs, total_epochs)
    fold_bs = budget_savings(len(candidate_policies), probe_epochs, total_epochs)

    return SelectorFoldResult(
        selector="physics_final",
        held_out_group=held_out_group,
        selected_policy=heuristic_policy,
        actual_best_policy=actual_best_policy,
        selected_target=heuristic_target,
        oracle_target=oracle_target,
        regret=heuristic_target - oracle_target,
        correct=float(heuristic_policy == actual_best_policy),
        num_policies=len(candidate_policies),
        probe_epochs=probe_epochs,
        total_epochs=total_epochs,
        budget_ratio=fold_br,
        budget_savings=fold_bs,
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(
    results_dir: Path = RESULTS_DIR,
    output_file: Path = OUTPUT_FILE,
    target_metric: str = TARGET_METRIC,
    window_size: int = WINDOW_SIZE,
    probe_epochs: int = PROBE_EPOCHS,
    total_epochs: int = TOTAL_EPOCHS,
) -> dict[str, object]:
    """Run the full bandit evaluation and save results."""
    print(f"Loading Paper A results from {results_dir} ...")
    dataset = build_paper_a_dataset(results_dir, target_metric, window_size)
    print(f"  Loaded {len(dataset)} samples across {len(dataset.unique_pdes())} PDEs")
    print(f"  PDEs: {dataset.unique_pdes()}")

    if len(dataset) == 0:
        print("ERROR: No samples loaded. Check that results exist.")
        return {}

    groups = dataset.pde_labels()
    unique_groups = list(dict.fromkeys(groups))
    global_names = dataset_feature_names(dataset)

    # Collect fold results per selector
    all_results: dict[str, list[SelectorFoldResult]] = {
        "random_forest": [],
        "physics_final": [],
        "successive_halving": [],
    }
    for alpha in LINUCB_ALPHAS:
        all_results[f"linucb_alpha{alpha}"] = []

    for held_out_group in unique_groups:
        train_samples = [
            s for s, g in zip(dataset.samples, groups, strict=False) if g != held_out_group
        ]
        test_samples = [
            s for s, g in zip(dataset.samples, groups, strict=False) if g == held_out_group
        ]
        if not train_samples or not test_samples:
            continue

        print(f"  Fold: held_out={held_out_group}  "
              f"train={len(train_samples)}  test={len(test_samples)}")

        # LinUCB for each alpha
        for alpha in LINUCB_ALPHAS:
            fold = evaluate_linucb_fold(
                train_samples, test_samples, global_names, alpha,
                held_out_group, probe_epochs, total_epochs,
            )
            all_results[f"linucb_alpha{alpha}"].append(fold)

        # Successive Halving
        sh_fold = evaluate_successive_halving_fold(
            test_samples, held_out_group, results_dir,
            held_out_group, total_epochs, probe_epochs,
        )
        all_results["successive_halving"].append(sh_fold)

        # Random Forest baseline
        rf_fold = evaluate_random_forest_fold(
            train_samples, test_samples, global_names,
            held_out_group, probe_epochs, total_epochs,
        )
        all_results["random_forest"].append(rf_fold)

        # Heuristic baseline
        heur_fold = evaluate_heuristic_fold(
            test_samples, held_out_group, probe_epochs, total_epochs,
        )
        all_results["physics_final"].append(heur_fold)

    # Summarise
    summaries: dict[str, dict] = {}
    for selector_name, fold_list in all_results.items():
        summaries[selector_name] = summarize_fold_results(fold_list)

    # Print comparison table
    print()
    print(f"{'Selector':<24s} {'Accuracy':>9s} {'Mean Regret':>12s} {'Budget Savings':>15s}")
    print("-" * 62)
    for name in sorted(summaries.keys()):
        s = summaries[name]
        acc = s["selection_accuracy"] * 100
        reg = s["mean_regret"]
        bsav = s["mean_budget_savings"] * 100
        print(f"{name:<24s} {acc:>8.1f}% {reg:>12.4f} {bsav:>14.1f}%")

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "config": {
            "target_metric": target_metric,
            "window_size": window_size,
            "probe_epochs": probe_epochs,
            "total_epochs": total_epochs,
            "linucb_alphas": LINUCB_ALPHAS,
            "n_samples": len(dataset),
            "n_pdes": len(unique_groups),
            "pdes": unique_groups,
        },
        "selectors": {
            name: {
                k: v for k, v in summary.items()
                if k != "folds"  # keep output compact
            }
            for name, summary in summaries.items()
        },
        "fold_details": {
            name: summary.get("folds", [])
            for name, summary in summaries.items()
        },
    }
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return output_data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()
