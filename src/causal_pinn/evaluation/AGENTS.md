<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# Evaluation

## Purpose
Provides evaluation metrics, physics-conservation consistency checks, and automated benchmark execution for trained PINN/PINO models. `metrics.py` supplies the atomic error/residual functions; `physics_consistency.py` verifies that solutions respect energy/mass conservation and characterizes residual distributions; `benchmark_runner.py` ties these together into `BenchmarkRunner`, which generates collocation/boundary/initial points for a given `PDEProblem`, runs a model (or compares several), and formats results as a table.

## Key Files
| File | Description |
|------|-------------|
| `metrics.py` | `relative_l2_error`, `mae`, `mse`, `max_error`, `pde_residual_norm`, `bc_violation`, `ic_violation`, `convergence_failure_rate`, `seed_variance`, and `MetricSuite` (computes a named subset of the above at once, validating metric names against `_AVAILABLE_METRICS`). |
| `physics_consistency.py` | `PhysicsConsistencyChecker` — `check_energy_conservation` and `check_mass_conservation` (both integrate over 1D space via the trapezoidal rule at each unique time point and report deviation from the t=0 value), and `check_residual_distribution` (mean/std/min/max/percentiles of PDE residuals). |
| `benchmark_runner.py` | `BenchmarkConfig` (point counts, `metric_names`, device) and `BenchmarkRunner` — `run_single` evaluates one model (solution-error metrics only if `pde_problem.exact_solution` is available; PDE residual computed with `requires_grad_` inputs; physics-consistency checks only run when the domain is 1D), `run_comparison` runs several named models, `format_comparison_table` renders a fixed-width text table. |
| `__init__.py` | Re-exports all of the above: metric functions, `MetricSuite`, `PhysicsConsistencyChecker`, `BenchmarkConfig`, `BenchmarkRunner`. |

## For AI Agents

### Working In This Directory
- All metric functions in `metrics.py` `reshape(-1)` their inputs before computing — new metrics should follow this convention so they work regardless of input shape (`(N,)` vs `(N, 1)`).
- `BenchmarkRunner` infers spatial dimensionality from the length of `pde_problem.domain_bounds[1]` (2 elements → 1D, 4 → 2D `(x_min, x_max, y_min, y_max)`); physics-consistency checks (`check_energy_conservation`) are gated on `spatial_dim == 1` in `run_single` — 2D conservation checking is not implemented.
- `MetricSuite` raises `ValueError` on unknown metric names — when adding a new metric function, also register it in `_AVAILABLE_METRICS`.
- `run_single` only computes solution-error metrics (`relative_l2_error`, `mse`, `mae`, `max_error`) when `pde_problem.exact_solution(...)` returns non-`None`; PDEs without a closed-form solution will only get residual/BC/IC metrics.

### Testing Requirements
Covered by `tests/unit/test_evaluation.py`. Run with:
```
PYTHONPATH=src python -m pytest tests/unit/test_evaluation.py -q
```

### Common Patterns
- Functions take `(pred, true)` `Tensor` pairs and immediately flatten with `.reshape(-1)`.
- `BenchmarkConfig` and other config objects are `@dataclass`es using `field(default_factory=...)` for mutable list defaults.
- `PDEProblem` is imported only under `TYPE_CHECKING` in both `physics_consistency.py` and `benchmark_runner.py` to avoid circular imports — the actual PDE object is passed in at call time as a duck-typed argument.
- Metrics that can be undefined (e.g. `relative_l2_error` when `true_norm == 0`) explicitly special-case rather than raising, returning `0.0` or `inf`.

## Dependencies

### Internal
- `causal_pinn.physics.pde.base.PDEProblem` (TYPE_CHECKING only, in `physics_consistency.py` and `benchmark_runner.py`)

### External
- `torch` (tensor ops, autograd for PDE residuals)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
