<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# Visualization

## Purpose
Plotting and result-aggregation utilities for analyzing PINN/PINO experiment output. `plots.py` renders matplotlib figures (training curves, convergence/metric comparisons, seed variance bands, residual histograms, causal-gate evolution); `analysis.py` aggregates raw per-seed results into summary statistics, formats markdown comparison tables, and handles JSON (de)serialization of results (including `Tensor`/`numpy` payloads).

## Key Files
| File | Description |
|------|-------------|
| `plots.py` | `plot_training_curves` (3-stage loss curves, single run or multi-seed overlay), `plot_convergence_comparison` (one line per named run), `plot_seed_variance` (mean ± std shaded band across seeds for one stage), `plot_metric_comparison` (grouped bar chart across models/metrics), `plot_residual_distribution` (PDE residual histogram with mean/std annotation), `plot_gate_evolution` (causal-gate mean/std over epochs, two stacked subplots). All accept `save_path: str | None` (save to disk via `_save_or_show` if given, else `plt.show()`) and return the `Figure`. |
| `analysis.py` | `aggregate_seed_results` (mean/std/min/max per metric across seeds), `compute_convergence_failure_rate` (fraction of seeds whose final loss — checked stage3 → stage2 → stage1 — exceeds a threshold), `format_results_table` (markdown table), `save_results_json`/`load_results_json` (recursively converts `Tensor`/`np.ndarray`/`np.floating`/`np.integer` to JSON-safe types), `compute_summary_statistics` (mean/std/min/max/median for a single series), `compare_results` (baseline vs. comparison with `improvement_pct`). |
| `__init__.py` | Re-exports all plotting functions from `plots.py` and all analysis functions from `analysis.py`. |

## For AI Agents

### Working In This Directory
- Every plotting function follows the `save_path: str | None = None` convention — pass a path to persist the figure (parent directories are auto-created by `_save_or_show`) or `None` to display it interactively; always return the `Figure` object so callers/tests can inspect it without hitting the filesystem.
- `STAGE_COLORS = {"stage1": "#1f77b4", "stage2": "#ff7f0e", "stage3": "#2ca02c"}` is the fixed color mapping used across `plot_training_curves`, `plot_seed_variance`, etc. — reuse it rather than hardcoding new colors when adding stage-aware plots.
- Functions that can receive empty data (e.g. `plot_seed_variance`, `plot_metric_comparison`, `plot_residual_distribution`, `plot_gate_evolution`) render a "No data available" placeholder rather than raising — match this behavior in new plotting functions instead of letting matplotlib error out.

### Testing Requirements
Covered by `tests/unit/test_visualization.py`. Run with:
```
PYTHONPATH=src python -m pytest tests/unit/test_visualization.py -q
```

### Common Patterns
- `_to_numpy` (in `plots.py`) and `_to_float` (in `analysis.py`) normalize `Tensor`/`ndarray`/`list`/scalar inputs before computation — reuse these rather than adding new ad hoc conversions.
- `save_results_json`'s `convert_to_serializable` recursively walks dicts/lists/tuples — when adding new result payload types, extend this function rather than pre-converting at every call site.
- History dicts are consistently keyed `stage1_losses`/`stage2_losses`/`stage3_losses`, each a list of per-epoch `dict[str, float]` with at least a `"total"` key — new analysis/plot functions consuming training history should expect this shape.

## Dependencies

### Internal
None directly — consumes plain dicts/`Tensor`s produced by `causal_pinn.training` (e.g. `TrainingHistory.to_dict()`) and `causal_pinn.evaluation`.

### External
- `matplotlib` (`pyplot`, `Figure`)
- `numpy`
- `torch` (`Tensor` handling only)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
