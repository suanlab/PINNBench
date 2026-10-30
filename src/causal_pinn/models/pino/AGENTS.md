<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# pino

## Purpose
This directory implements the Physics-Informed Neural Operator (PINO) baseline: an `FNO` trained not with pure supervised input-output pairs but with a PDE-residual physics loss (plus boundary-condition and optional data losses), analogous to how a PINN is trained but using an operator backbone instead of a plain MLP. It serves as a comparison point against both the pure `FNO` operator and the `HybridPINNOperator` fusion model.

## Key Files
| File | Description |
|------|-------------|
| `pino_model.py` | Defines `PINOModel(nn.Module)`, wrapping an `FNO` (from `causal_pinn.models.operators.fno`) and a `PDEProblem`. `forward(inputs)` expects a single `(N, 2)` tensor of `[t, x]` (raises `ValueError` otherwise), reshapes it to the FNO's expected `(N, 2, 1)` 1D-channel format via `_prepare_operator_input`, and flattens the FNO's `(N, out_channels, 1)` output back to `(N, out_channels)` via `_flatten_operator_output`. `compute_pino_loss(t_colloc, x_colloc)` computes `physics_loss` (mean squared PDE residual via `pde_problem.residual`), `bc_loss` (MSE against `pde_problem.boundary_condition`), and an optional `data_loss` against `pde_problem.exact_solution` (silently skipped — left as zero — if the PDE problem raises `NotImplementedError`, `AttributeError`, or `TypeError`, i.e. doesn't provide an exact solution), returning all three plus their unweighted `total` sum. |

## For AI Agents

### Working In This Directory
- `PINOModel` is hard-restricted to 1D spatial problems: `_prepare_operator_input` raises `ValueError` if `t.shape[1] != 1 or x.shape[1] != 1`, and it always calls `self.fno` (not a generic `NeuralOperator`) — it is not written to be operator-agnostic like `HybridPINNOperator` is. If you need 2D PINO support, this class needs explicit extension, not just a different `dim` config.
- `compute_pino_loss` combines its three loss terms with equal (1.0) weights, unlike `HybridPINNOperator.compute_hybrid_loss` which applies configurable weights from `HybridConfig` — if you need weighted PINO losses, add weight parameters here rather than assuming parity with the hybrid model's config pattern.
- The `data_loss` fallback relies on catching three specific exception types from `pde_problem.exact_solution(...)` — when adding a new `PDEProblem` implementation, raise one of `NotImplementedError`/`AttributeError`/`TypeError` (or implement `exact_solution` properly) so this fallback continues to work as intended rather than crashing.
- Uses relative imports (`from ...physics.pde.base import PDEProblem`, `from ..operators.fno import FNO`) rather than absolute `causal_pinn.` imports used elsewhere in the models tree — match this file's existing relative-import style if editing it further.

### Testing Requirements
Covered by `tests/unit/test_pino.py`. Run with:
`PYTHONPATH=src python -m pytest tests/unit/test_pino.py -q`

### Common Patterns
- Same `requires_grad_(True)`-if-not-tracking pattern for collocation points (`t_colloc`, `x_colloc`) as seen in `hybrid_model.py` and `pinn_solver.py`.
- Shape coercion (`unsqueeze(-1)` for 1D inputs) is applied consistently before concatenation, matching the convention used throughout the `models` package.

## Dependencies

### Internal
`causal_pinn.models.operators.fno.FNO`, `causal_pinn.physics.pde.base.PDEProblem`.

### External
`torch`, `torch.nn`.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
