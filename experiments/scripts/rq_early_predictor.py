#!/usr/bin/env python3
"""Evaluate early-training prediction and budget-aware policy routing."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

_early_predictor_module = importlib.import_module("causal_pinn.training.early_predictor")
EarlyTrainingPredictor = _early_predictor_module.EarlyTrainingPredictor
FeatureExtractor = _early_predictor_module.FeatureExtractor
_policy_routing_module = importlib.import_module("causal_pinn.training.policy_routing")
evaluate_policy_routing = _policy_routing_module.evaluate_policy_routing
remap_dataset_groups = _policy_routing_module.remap_dataset_groups
RandomForestRegressor = importlib.import_module("sklearn.ensemble").RandomForestRegressor


def load_result_specs(results_dir: Path) -> tuple[list[Path], list[str]]:
    experiment_specs: list[tuple[str, str]] = [
        ("rq3_10seed/results.json", "heat1d"),
        ("rq4_10seed/results.json", "wave2d"),
        ("rq_pino_heat1d/results.json", "heat1d_pino"),
        ("rq1e_kdv_hybrid/results.json", "kdv"),
        ("rq1f_allencahn_hybrid/results.json", "allen_cahn"),
        # Extended suites (available after 10-seed reruns & PINO AdvDiff)
        ("rq1e_kdv_10seed/results.json", "kdv_10seed"),
        ("rq1f_allencahn_10seed/results.json", "allen_cahn_10seed"),
        ("rq_pino_advdiff/results.json", "advdiff_pino"),
    ]

    result_paths: list[Path] = []
    pde_labels: list[str] = []
    for rel_path, pde in experiment_specs:
        full_path = results_dir / rel_path
        if not full_path.exists():
            print(f"  [MISS] {rel_path}")
            continue
        print(f"  [OK]   {rel_path} -> {pde}")
        result_paths.append(full_path)
        pde_labels.append(pde)
    return result_paths, pde_labels


def build_predictor(
    window_size: int,
    target_metric: str,
    n_estimators: int,
) -> EarlyTrainingPredictor:
    extractor = FeatureExtractor(window_size=window_size)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    return EarlyTrainingPredictor(
        target_metric=target_metric,
        model=model,
        extractor=extractor,
    )


def print_prediction_summary(grouping_name: str, cv_results: dict[str, object]) -> None:
    print(
        f"  {grouping_name:7s} -> mean R^2={cv_results['mean_r2']:.4f}, "
        f"policy_acc={cv_results['policy_selection_accuracy']:.1%}"
    )


def print_selector_summary(label: str, selector_summary: dict[str, object]) -> None:
    print(
        f"    {label:24s} regret={selector_summary['mean_regret']:.4f}, "
        f"acc={selector_summary['selection_accuracy']:.1%}, "
        f"save={selector_summary['mean_budget_savings']:.1%}, "
        f"fallback={selector_summary['fallback_rate']:.1%}"
    )


def mean_regret(summary: object) -> float:
    typed_summary = cast(dict[str, object], summary)
    return float(cast(float, typed_summary["mean_regret"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Early-training prediction and budget-aware policy routing",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="Root directory for experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/rq_early_predictor",
        help="Output directory for routing results",
    )
    parser.add_argument(
        "--window-sizes",
        type=int,
        nargs="+",
        default=[10, 20, 50, 100],
        help="Observation windows in stage-2 epochs",
    )
    parser.add_argument(
        "--target-metric",
        type=str,
        default="pde_residual_norm",
        help="Evaluation metric to predict and route on",
    )
    parser.add_argument(
        "--secondary-target",
        type=str,
        default="relative_l2_error",
        help="Secondary metric for raw prediction only",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in RandomForest",
    )
    parser.add_argument(
        "--total-stage2-epochs",
        type=int,
        default=500,
        help="Total stage-2 epochs for budget accounting",
    )
    parser.add_argument(
        "--fallback-ratios",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0],
        help="Fallback trigger sweep: margin <= ratio * best-policy std",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading experiment files...")
    result_paths, pde_labels = load_result_specs(results_dir)
    if not result_paths:
        raise SystemExit("No experiment files found.")

    windows_output: dict[str, object] = {}
    output: dict[str, object] = {
        "experiment": "rq_early_predictor",
        "config": {
            "target_metric": args.target_metric,
            "secondary_target": args.secondary_target,
            "window_sizes": args.window_sizes,
            "n_estimators": args.n_estimators,
            "total_stage2_epochs": args.total_stage2_epochs,
            "fallback_ratios": args.fallback_ratios,
            "experiments_included": [str(path) for path in result_paths],
            "pde_labels": pde_labels,
        },
        "windows": windows_output,
    }

    for window_size in args.window_sizes:
        print(f"\n=== Window {window_size} ===")
        predictor = build_predictor(window_size, args.target_metric, args.n_estimators)
        dataset = predictor.build_dataset(result_paths=result_paths, pde_labels=pde_labels)
        relaxed_dataset = remap_dataset_groups(dataset, strict=False)
        strict_dataset = remap_dataset_groups(dataset, strict=True)

        print(
            f"  dataset: {len(dataset)} samples, "
            f"relaxed_groups={relaxed_dataset.unique_pdes()}, "
            f"strict_groups={strict_dataset.unique_pdes()}"
        )

        relaxed_prediction = predictor.cross_validate(relaxed_dataset, group_col="pde")
        strict_prediction = predictor.cross_validate(strict_dataset, group_col="pde")
        print_prediction_summary("relaxed", relaxed_prediction)
        print_prediction_summary("strict", strict_prediction)

        routing_relaxed = evaluate_policy_routing(
            dataset,
            probe_epochs=window_size,
            total_epochs=args.total_stage2_epochs,
            strict_grouping=False,
            margin_std_ratio=0.0,
        )
        routing_strict = evaluate_policy_routing(
            dataset,
            probe_epochs=window_size,
            total_epochs=args.total_stage2_epochs,
            strict_grouping=True,
            margin_std_ratio=0.0,
        )

        fallback_sweep: dict[str, dict[str, object]] = {"relaxed": {}, "strict": {}}
        for ratio in args.fallback_ratios:
            relaxed_with_ratio = evaluate_policy_routing(
                dataset,
                probe_epochs=window_size,
                total_epochs=args.total_stage2_epochs,
                strict_grouping=False,
                margin_std_ratio=ratio,
            )
            strict_with_ratio = evaluate_policy_routing(
                dataset,
                probe_epochs=window_size,
                total_epochs=args.total_stage2_epochs,
                strict_grouping=True,
                margin_std_ratio=ratio,
            )
            fallback_sweep["relaxed"][str(ratio)] = relaxed_with_ratio["selectors"][
                "random_forest_fallback"
            ]
            fallback_sweep["strict"][str(ratio)] = strict_with_ratio["selectors"][
                "random_forest_fallback"
            ]

        best_relaxed_ratio, best_relaxed_summary = min(
            fallback_sweep["relaxed"].items(),
            key=lambda item: mean_regret(item[1]),
        )
        best_strict_ratio, best_strict_summary = min(
            fallback_sweep["strict"].items(),
            key=lambda item: mean_regret(item[1]),
        )

        print("  routing (relaxed)")
        for selector_name in [
            "random_forest",
            "random_forest_fallback",
            "linear_regression",
            "training_prior",
            "physics_final",
            "physics_slope",
            "total_final",
            "random_expected",
        ]:
            print_selector_summary(
                selector_name,
                routing_relaxed["selectors"][selector_name],
            )

        print("  routing (strict)")
        for selector_name in [
            "random_forest",
            "random_forest_fallback",
            "linear_regression",
            "training_prior",
            "physics_final",
            "physics_slope",
            "total_final",
            "random_expected",
        ]:
            print_selector_summary(
                selector_name,
                routing_strict["selectors"][selector_name],
            )
        print(
            f"  best fallback ratio -> relaxed={best_relaxed_ratio} "
            f"(regret={mean_regret(best_relaxed_summary):.4f}), "
            f"strict={best_strict_ratio} "
            f"(regret={mean_regret(best_strict_summary):.4f})"
        )

        secondary_prediction: dict[str, object] | None = None
        if args.secondary_target:
            secondary_predictor = build_predictor(
                window_size,
                args.secondary_target,
                args.n_estimators,
            )
            secondary_dataset = secondary_predictor.build_dataset(
                result_paths=result_paths,
                pde_labels=pde_labels,
            )
            if len(secondary_dataset) > 0:
                secondary_prediction = {
                    "relaxed": secondary_predictor.cross_validate(
                        remap_dataset_groups(secondary_dataset, strict=False),
                        group_col="pde",
                    ),
                    "strict": secondary_predictor.cross_validate(
                        remap_dataset_groups(secondary_dataset, strict=True),
                        group_col="pde",
                    ),
                }

        feature_importance = predictor.feature_importance()
        sorted_importance = sorted(
            feature_importance.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        windows_output[str(window_size)] = {
            "dataset_summary": {
                "n_samples": len(dataset),
                "relaxed_groups": relaxed_dataset.unique_pdes(),
                "strict_groups": strict_dataset.unique_pdes(),
            },
            "prediction": {
                "relaxed": relaxed_prediction,
                "strict": strict_prediction,
                "secondary": secondary_prediction,
            },
            "routing": {
                "relaxed": routing_relaxed,
                "strict": routing_strict,
                "fallback_sweep": fallback_sweep,
                "best_fallback": {
                    "relaxed": {
                        "ratio": best_relaxed_ratio,
                        "summary": best_relaxed_summary,
                    },
                    "strict": {
                        "ratio": best_strict_ratio,
                        "summary": best_strict_summary,
                    },
                },
            },
            "feature_importance": dict(sorted_importance),
        }

    output_path = output_dir / "results.json"
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2, default=str)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
