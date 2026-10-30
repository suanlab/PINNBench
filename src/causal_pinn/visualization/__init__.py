"""Visualization utilities for PINN experiment analysis.

This module provides plotting functions and result analysis utilities
for visualizing and comparing PINN training results.

Plotting Functions:
    - plot_training_curves: Loss curves for 3-stage training
    - plot_convergence_comparison: Compare multiple runs
    - plot_seed_variance: Mean +/- std across seeds
    - plot_metric_comparison: Grouped bar chart for metrics
    - plot_residual_distribution: Histogram of PDE residuals
    - plot_gate_evolution: Gate statistics over epochs

Analysis Functions:
    - aggregate_seed_results: Compute statistics across seeds
    - compute_convergence_failure_rate: Fraction of failed runs
    - format_results_table: Markdown table formatting
    - save_results_json: Save results to JSON
    - load_results_json: Load results from JSON
"""

from causal_pinn.visualization.analysis import (
    aggregate_seed_results,
    compare_results,
    compute_convergence_failure_rate,
    compute_summary_statistics,
    format_results_table,
    load_results_json,
    save_results_json,
)
from causal_pinn.visualization.plots import (
    plot_convergence_comparison,
    plot_gate_evolution,
    plot_metric_comparison,
    plot_residual_distribution,
    plot_seed_variance,
    plot_training_curves,
)

__all__ = [
    # Plotting functions
    "plot_training_curves",
    "plot_convergence_comparison",
    "plot_seed_variance",
    "plot_metric_comparison",
    "plot_residual_distribution",
    "plot_gate_evolution",
    # Analysis functions
    "aggregate_seed_results",
    "compute_convergence_failure_rate",
    "format_results_table",
    "save_results_json",
    "load_results_json",
    "compute_summary_statistics",
    "compare_results",
]
