"""Budget-aware routing analysis for early-training policy selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from .early_predictor import PredictorDataset, PredictorSample

_POLICY_ALIASES = {
    "hybrid": "hybrid_full",
}

_STRICT_PDE_ALIASES = {
    "heat1d_pino": "heat1d",
    "advdiff_pino": "advdiff",
    "kdv_10seed": "kdv",
    "allen_cahn_10seed": "allen_cahn",
}


def normalize_policy_name(policy: str) -> str:
    return _POLICY_ALIASES.get(policy, policy)


def normalize_pde_group(pde: str, *, strict: bool) -> str:
    if not strict:
        return pde
    return _STRICT_PDE_ALIASES.get(pde, pde)


def budget_ratio(num_policies: int, probe_epochs: int, total_epochs: int) -> float:
    if num_policies <= 0 or total_epochs <= 0:
        raise ValueError("num_policies and total_epochs must be positive")
    if probe_epochs < 0 or probe_epochs > total_epochs:
        raise ValueError("probe_epochs must be in [0, total_epochs]")
    exhaustive_budget = num_policies * total_epochs
    routed_budget = num_policies * probe_epochs + (total_epochs - probe_epochs)
    return routed_budget / exhaustive_budget


def budget_savings(num_policies: int, probe_epochs: int, total_epochs: int) -> float:
    return 1.0 - budget_ratio(num_policies, probe_epochs, total_epochs)


def remap_dataset_groups(dataset: PredictorDataset, *, strict: bool) -> PredictorDataset:
    remapped_samples = [
        PredictorSample(
            features=sample.features,
            target=sample.target,
            pde=normalize_pde_group(sample.pde, strict=strict),
            policy=sample.policy,
            seed=sample.seed,
        )
        for sample in dataset.samples
    ]
    return PredictorDataset(samples=remapped_samples)


def policy_mean_values(
    samples: list[PredictorSample],
    values: NDArray[np.float64],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for sample, value in zip(samples, values, strict=True):
        policy = normalize_policy_name(sample.policy)
        grouped.setdefault(policy, []).append(float(value))
    return {policy: float(np.mean(policy_values)) for policy, policy_values in grouped.items()}


def choose_lowest_policy(policy_values: dict[str, float]) -> str:
    if not policy_values:
        return "unknown"
    return min(policy_values, key=lambda key: policy_values[key])


def expected_random_target(policy_actual_means: dict[str, float]) -> float:
    if not policy_actual_means:
        return 0.0
    return float(np.mean(list(policy_actual_means.values())))


def expected_random_accuracy(policy_actual_means: dict[str, float]) -> float:
    if not policy_actual_means:
        return 0.0
    return 1.0 / len(policy_actual_means)


def train_policy_prior(
    train_samples: list[PredictorSample],
    train_targets: NDArray[np.float64],
    candidate_policies: set[str],
) -> str:
    policy_targets = policy_mean_values(train_samples, train_targets)
    filtered = {
        policy: value
        for policy, value in policy_targets.items()
        if policy in candidate_policies
    }
    if not filtered:
        return "unknown"
    return choose_lowest_policy(filtered)


def heuristic_policy_choice(
    samples: list[PredictorSample],
    feature_name: str,
) -> str:
    policy_scores: dict[str, list[float]] = {}
    for sample in samples:
        if feature_name not in sample.features:
            continue
        policy = normalize_policy_name(sample.policy)
        policy_scores.setdefault(policy, []).append(sample.features[feature_name])
    if not policy_scores:
        return "unknown"
    mean_scores = {
        policy: float(np.mean(scores))
        for policy, scores in policy_scores.items()
    }
    return choose_lowest_policy(mean_scores)


def dataset_feature_names(dataset: PredictorDataset) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for sample in dataset.samples:
        for key in sample.features:
            if key not in seen:
                seen.add(key)
                names.append(key)
    return names


def tree_prediction_statistics(
    model: RandomForestRegressor,
    X: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    tree_predictions = np.array(
        [estimator.predict(X) for estimator in model.estimators_],
        dtype=np.float64,
    )
    return np.mean(tree_predictions, axis=0), np.std(tree_predictions, axis=0)


def policy_prediction_statistics(
    samples: list[PredictorSample],
    means: NDArray[np.float64],
    stds: NDArray[np.float64],
) -> dict[str, tuple[float, float]]:
    grouped_means: dict[str, list[float]] = {}
    grouped_stds: dict[str, list[float]] = {}
    for sample, mean_value, std_value in zip(samples, means, stds, strict=True):
        policy = normalize_policy_name(sample.policy)
        grouped_means.setdefault(policy, []).append(float(mean_value))
        grouped_stds.setdefault(policy, []).append(float(std_value))
    return {
        policy: (
            float(np.mean(grouped_means[policy])),
            float(np.mean(grouped_stds[policy])),
        )
        for policy in grouped_means
    }


def fallback_policy_choice(
    policy_stats: dict[str, tuple[float, float]],
    fallback_policy: str,
    margin_std_ratio: float,
) -> tuple[str, bool]:
    if fallback_policy not in policy_stats:
        return choose_lowest_policy({k: v[0] for k, v in policy_stats.items()}), False
    ranked = sorted(policy_stats.items(), key=lambda item: item[1][0])
    if len(ranked) < 2:
        return ranked[0][0], False
    best_policy, (best_mean, best_std) = ranked[0]
    second_mean = ranked[1][1][0]
    margin = second_mean - best_mean
    threshold = margin_std_ratio * max(best_std, 1e-8)
    if margin <= threshold:
        return fallback_policy, True
    return best_policy, False


@dataclass
class SelectorFoldResult:
    selector: str
    held_out_group: str
    selected_policy: str
    actual_best_policy: str
    selected_target: float
    oracle_target: float
    regret: float
    correct: float
    num_policies: int
    probe_epochs: int
    total_epochs: int
    budget_ratio: float
    budget_savings: float
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_fold_results(
    fold_results: list[SelectorFoldResult],
) -> dict[str, Any]:
    if not fold_results:
        return {
            "mean_regret": 0.0,
            "max_regret": 0.0,
            "selection_accuracy": 0.0,
            "mean_budget_ratio": 0.0,
            "mean_budget_savings": 0.0,
            "fallback_rate": 0.0,
            "folds": [],
        }
    return {
        "mean_regret": float(np.mean([fold.regret for fold in fold_results])),
        "max_regret": float(np.max([fold.regret for fold in fold_results])),
        "selection_accuracy": float(np.mean([fold.correct for fold in fold_results])),
        "mean_budget_ratio": float(np.mean([fold.budget_ratio for fold in fold_results])),
        "mean_budget_savings": float(np.mean([fold.budget_savings for fold in fold_results])),
        "fallback_rate": float(np.mean([float(fold.fallback_used) for fold in fold_results])),
        "folds": [fold.to_dict() for fold in fold_results],
    }


def evaluate_policy_routing(
    dataset: PredictorDataset,
    *,
    probe_epochs: int,
    total_epochs: int,
    strict_grouping: bool,
    margin_std_ratio: float = 1.0,
) -> dict[str, Any]:
    grouped_dataset = remap_dataset_groups(dataset, strict=strict_grouping)
    groups = grouped_dataset.pde_labels()
    unique_groups = list(dict.fromkeys(groups))
    global_names = dataset_feature_names(grouped_dataset)

    rf_results: list[SelectorFoldResult] = []
    rf_fallback_results: list[SelectorFoldResult] = []
    linear_results: list[SelectorFoldResult] = []
    gb_results: list[SelectorFoldResult] = []
    prior_results: list[SelectorFoldResult] = []
    physics_final_results: list[SelectorFoldResult] = []
    physics_slope_results: list[SelectorFoldResult] = []
    total_final_results: list[SelectorFoldResult] = []
    random_results: list[SelectorFoldResult] = []

    for held_out_group in unique_groups:
        train_samples = [
            sample
            for sample, group in zip(grouped_dataset.samples, groups, strict=True)
            if group != held_out_group
        ]
        test_samples = [
            sample
            for sample, group in zip(grouped_dataset.samples, groups, strict=True)
            if group == held_out_group
        ]
        train_dataset = PredictorDataset(samples=train_samples)
        test_dataset = PredictorDataset(samples=test_samples)
        if len(train_dataset) == 0 or len(test_dataset) == 0:
            continue

        y_train = train_dataset.target_vector()
        y_test = test_dataset.target_vector()
        actual_policy_targets = policy_mean_values(test_samples, y_test)
        actual_best_policy = choose_lowest_policy(actual_policy_targets)
        oracle_target = actual_policy_targets[actual_best_policy]
        candidate_policies = set(actual_policy_targets)
        fold_budget_ratio = budget_ratio(len(candidate_policies), probe_epochs, total_epochs)
        fold_budget_savings = budget_savings(len(candidate_policies), probe_epochs, total_epochs)

        X_train = train_dataset.feature_matrix(feature_names=global_names)
        X_test = test_dataset.feature_matrix(feature_names=global_names)

        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train)
        rf_mean, rf_std = tree_prediction_statistics(rf_model, X_test)
        rf_policy_stats = policy_prediction_statistics(test_samples, rf_mean, rf_std)
        rf_policy_means = {policy: values[0] for policy, values in rf_policy_stats.items()}
        rf_policy = choose_lowest_policy(rf_policy_means)
        rf_target = actual_policy_targets[rf_policy]
        rf_results.append(
            SelectorFoldResult(
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
                budget_ratio=fold_budget_ratio,
                budget_savings=fold_budget_savings,
            )
        )

        prior_policy = train_policy_prior(train_samples, y_train, candidate_policies)
        if prior_policy == "unknown":
            prior_policy = rf_policy
        fallback_policy, fallback_used = fallback_policy_choice(
            rf_policy_stats,
            prior_policy,
            margin_std_ratio,
        )
        fallback_target = actual_policy_targets[fallback_policy]
        rf_fallback_results.append(
            SelectorFoldResult(
                selector="random_forest_fallback",
                held_out_group=held_out_group,
                selected_policy=fallback_policy,
                actual_best_policy=actual_best_policy,
                selected_target=fallback_target,
                oracle_target=oracle_target,
                regret=fallback_target - oracle_target,
                correct=float(fallback_policy == actual_best_policy),
                num_policies=len(candidate_policies),
                probe_epochs=probe_epochs,
                total_epochs=total_epochs,
                budget_ratio=fold_budget_ratio,
                budget_savings=fold_budget_savings,
                fallback_used=fallback_used,
            )
        )

        linear_model = Ridge(alpha=1.0)
        linear_model.fit(X_train, y_train)
        linear_predictions = np.array(linear_model.predict(X_test), dtype=np.float64)
        linear_policy_targets = policy_mean_values(test_samples, linear_predictions)
        linear_policy = choose_lowest_policy(linear_policy_targets)
        linear_target = actual_policy_targets[linear_policy]
        linear_results.append(
            SelectorFoldResult(
                selector="linear_regression",
                held_out_group=held_out_group,
                selected_policy=linear_policy,
                actual_best_policy=actual_best_policy,
                selected_target=linear_target,
                oracle_target=oracle_target,
                regret=linear_target - oracle_target,
                correct=float(linear_policy == actual_best_policy),
                num_policies=len(candidate_policies),
                probe_epochs=probe_epochs,
                total_epochs=total_epochs,
                budget_ratio=fold_budget_ratio,
                budget_savings=fold_budget_savings,
            )
        )

        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        )
        gb_model.fit(X_train, y_train)
        gb_predictions = np.array(gb_model.predict(X_test), dtype=np.float64)
        gb_policy_targets = policy_mean_values(test_samples, gb_predictions)
        gb_policy = choose_lowest_policy(gb_policy_targets)
        gb_target = actual_policy_targets[gb_policy]
        gb_results.append(
            SelectorFoldResult(
                selector="gradient_boosting",
                held_out_group=held_out_group,
                selected_policy=gb_policy,
                actual_best_policy=actual_best_policy,
                selected_target=gb_target,
                oracle_target=oracle_target,
                regret=gb_target - oracle_target,
                correct=float(gb_policy == actual_best_policy),
                num_policies=len(candidate_policies),
                probe_epochs=probe_epochs,
                total_epochs=total_epochs,
                budget_ratio=fold_budget_ratio,
                budget_savings=fold_budget_savings,
            )
        )

        if prior_policy in actual_policy_targets:
            prior_target = actual_policy_targets[prior_policy]
            prior_results.append(
                SelectorFoldResult(
                    selector="training_prior",
                    held_out_group=held_out_group,
                    selected_policy=prior_policy,
                    actual_best_policy=actual_best_policy,
                    selected_target=prior_target,
                    oracle_target=oracle_target,
                    regret=prior_target - oracle_target,
                    correct=float(prior_policy == actual_best_policy),
                    num_policies=len(candidate_policies),
                    probe_epochs=probe_epochs,
                    total_epochs=total_epochs,
                    budget_ratio=fold_budget_ratio,
                    budget_savings=fold_budget_savings,
                )
            )

        for selector_name, feature_name, selector_results in [
            ("physics_final", "physics_loss_final", physics_final_results),
            ("physics_slope", "physics_loss_slope", physics_slope_results),
            ("total_final", "total_final", total_final_results),
        ]:
            heuristic_policy = heuristic_policy_choice(test_samples, feature_name)
            if heuristic_policy == "unknown" or heuristic_policy not in actual_policy_targets:
                continue
            heuristic_target = actual_policy_targets[heuristic_policy]
            selector_results.append(
                SelectorFoldResult(
                    selector=selector_name,
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
                    budget_ratio=fold_budget_ratio,
                    budget_savings=fold_budget_savings,
                )
            )

        random_target = expected_random_target(actual_policy_targets)
        random_results.append(
            SelectorFoldResult(
                selector="random_expected",
                held_out_group=held_out_group,
                selected_policy="expected_random",
                actual_best_policy=actual_best_policy,
                selected_target=random_target,
                oracle_target=oracle_target,
                regret=random_target - oracle_target,
                correct=expected_random_accuracy(actual_policy_targets),
                num_policies=len(candidate_policies),
                probe_epochs=probe_epochs,
                total_epochs=total_epochs,
                budget_ratio=fold_budget_ratio,
                budget_savings=fold_budget_savings,
            )
        )

    summaries = {
        "random_forest": summarize_fold_results(rf_results),
        "random_forest_fallback": summarize_fold_results(rf_fallback_results),
        "linear_regression": summarize_fold_results(linear_results),
        "gradient_boosting": summarize_fold_results(gb_results),
        "training_prior": summarize_fold_results(prior_results),
        "physics_final": summarize_fold_results(physics_final_results),
        "physics_slope": summarize_fold_results(physics_slope_results),
        "total_final": summarize_fold_results(total_final_results),
        "random_expected": summarize_fold_results(random_results),
    }
    return {
        "strict_grouping": strict_grouping,
        "probe_epochs": probe_epochs,
        "total_epochs": total_epochs,
        "groups": unique_groups,
        "selectors": summaries,
    }
