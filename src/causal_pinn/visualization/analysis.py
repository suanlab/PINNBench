"""Result analysis and aggregation utilities for PINN experiments.

This module provides functions for aggregating results across seeds,
computing statistics, and formatting results for reporting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor


def _to_float(value: Any) -> float:
    """Convert a value to float, handling tensors."""
    if isinstance(value, Tensor):
        return value.detach().cpu().item()
    return float(value)


def aggregate_seed_results(
    results_by_seed: dict[int, dict[str, float]]
) -> dict[str, dict[str, float]]:
    """Compute mean, std, min, max across seeds for each metric.

    Args:
        results_by_seed: Dict mapping seed -> dict of metric name -> value.
            Example: {42: {"mse": 0.1, "mae": 0.2}, 123: {"mse": 0.15, "mae": 0.18}}

    Returns:
        Dict mapping metric name -> dict with 'mean', 'std', 'min', 'max' keys.
        Example: {"mse": {"mean": 0.125, "std": 0.025, "min": 0.1, "max": 0.15}, ...}
    """
    if not results_by_seed:
        return {}

    # Collect all metrics
    all_metrics: set[str] = set()
    for seed_results in results_by_seed.values():
        all_metrics.update(seed_results.keys())

    aggregated: dict[str, dict[str, float]] = {}

    for metric in sorted(all_metrics):
        values = []
        for seed_results in results_by_seed.values():
            if metric in seed_results:
                values.append(_to_float(seed_results[metric]))

        if values:
            values_array = np.array(values)
            aggregated[metric] = {
                "mean": float(np.mean(values_array)),
                "std": float(np.std(values_array)),
                "min": float(np.min(values_array)),
                "max": float(np.max(values_array)),
            }

    return aggregated


def compute_convergence_failure_rate(
    histories: dict[int, dict[str, Any]],
    threshold: float,
) -> float:
    """Compute fraction of seeds where final loss exceeds threshold.

    Args:
        histories: Dict mapping seed -> training history dict.
            Each history dict contains stage1_losses, stage2_losses, stage3_losses.
        threshold: Maximum acceptable final total loss.

    Returns:
        Failure rate as a float between 0 and 1.
    """
    if not histories:
        return 0.0

    failures = 0
    total = len(histories)

    for _seed, history in histories.items():
        # Get final loss from the last stage with data
        final_loss = None

        # Check stage 3 first, then stage 2, then stage 1
        for stage in ["stage3_losses", "stage2_losses", "stage1_losses"]:
            stage_losses = history.get(stage, [])
            if stage_losses:
                final_loss = stage_losses[-1].get("total", float("inf"))
                break

        if final_loss is None or final_loss > threshold:
            failures += 1

    return failures / total


def format_results_table(
    results: dict[str, dict[str, float]],
    precision: int = 6,
) -> str:
    """Pretty-print comparison table in markdown format.

    Args:
        results: Dict mapping model name -> dict of metric name -> value.
            Example: {"PINN": {"mse": 0.1, "mae": 0.2}, "Hybrid": {"mse": 0.05, "mae": 0.1}}
        precision: Number of decimal places for float formatting.

    Returns:
        Markdown-formatted string table.
    """
    if not results:
        return "No results to display."

    # Get all unique metrics
    all_metrics: set[str] = set()
    for model_results in results.values():
        all_metrics.update(model_results.keys())
    metric_names = sorted(all_metrics)

    if not metric_names:
        return "No metrics to display."

    model_names = list(results.keys())

    # Calculate column widths
    name_width = max(len(name) for name in model_names)
    name_width = max(name_width, len("Model"))

    metric_widths = {}
    for metric in metric_names:
        max_val_width = precision + 6  # for scientific notation like 1.234567e-10
        metric_widths[metric] = max(len(metric), max_val_width)

    # Build header
    header = f"| {'Model':<{name_width}} |"
    for metric in metric_names:
        header += f" {metric:>{metric_widths[metric]}} |"

    # Build separator
    separator = f"|{'-' * (name_width + 2)}|"
    for metric in metric_names:
        separator += f"{'-' * (metric_widths[metric] + 2)}|"

    # Build rows
    rows = []
    for name in model_names:
        model_results = results[name]
        row = f"| {name:<{name_width}} |"
        for metric in metric_names:
            value = model_results.get(metric, float("nan"))
            if isinstance(value, (int, float)):
                # Use scientific notation for very small/large numbers
                if abs(value) < 1e-3 or abs(value) >= 1e4:
                    formatted = f"{value:.{precision}e}"
                else:
                    formatted = f"{value:.{precision}f}"
            else:
                formatted = str(value)
            row += f" {formatted:>{metric_widths[metric]}} |"
        rows.append(row)

    # Combine into markdown table
    table = f"{header}\n{separator}\n" + "\n".join(rows)
    return table


def save_results_json(
    results: dict[str, Any],
    filepath: str,
) -> None:
    """Save results dict to JSON file.

    Handles Tensors by converting to Python floats/lists.

    Args:
        results: Dict of results to save.
        filepath: Path to save the JSON file.
    """
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    def convert_to_serializable(obj: Any) -> Any:
        """Recursively convert tensors and numpy arrays to serializable types."""
        if isinstance(obj, Tensor):
            return obj.detach().cpu().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    serializable_results = convert_to_serializable(results)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)


def load_results_json(filepath: str) -> dict[str, Any]:
    """Load results from JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        Dict of loaded results.
    """
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]


def compute_summary_statistics(
    values: list[float] | Tensor | np.ndarray,
) -> dict[str, float]:
    """Compute summary statistics for a list of values.

    Args:
        values: List, tensor, or array of numeric values.

    Returns:
        Dict with 'mean', 'std', 'min', 'max', 'median' keys.
    """
    if isinstance(values, Tensor):
        values = values.detach().cpu().numpy()
    elif isinstance(values, list):
        values = np.array(values)

    if len(values) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def compare_results(
    baseline: dict[str, float],
    comparison: dict[str, float],
    baseline_name: str = "Baseline",
    comparison_name: str = "Comparison",
) -> dict[str, dict[str, float]]:
    """Compare two result sets and compute relative improvements.

    Args:
        baseline: Baseline result dict (metric -> value).
        comparison: Comparison result dict (metric -> value).
        baseline_name: Name for baseline in output.
        comparison_name: Name for comparison in output.

    Returns:
        Dict with metric -> {baseline_name, comparison_name, 'improvement_pct'}.
    """
    comparison_results: dict[str, dict[str, float]] = {}

    # Get all metrics from both
    all_metrics = set(baseline.keys()) | set(comparison.keys())

    for metric in sorted(all_metrics):
        baseline_val = baseline.get(metric, 0.0)
        comparison_val = comparison.get(metric, 0.0)

        # Compute improvement percentage
        if baseline_val != 0:
            # For metrics where lower is better (like MSE, error)
            improvement = ((baseline_val - comparison_val) / baseline_val) * 100
        else:
            improvement = 0.0

        comparison_results[metric] = {
            baseline_name: baseline_val,
            comparison_name: comparison_val,
            "improvement_pct": improvement,
        }

    return comparison_results
