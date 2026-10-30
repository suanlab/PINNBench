<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# pinn

## Purpose
This directory contains the Physics-Informed Neural Network (PINN) components: the base MLP architecture that approximates the PDE solution `u(t, x)`, and the solver class that wraps that network together with a PDE problem to compute the physics/boundary/initial-condition losses needed for PINN training.

## Key Files
| File | Description |
|------|-------------|
| `base.py` | Defines `PINNBase(nn.Module)`: a configurable MLP taking `input_dim` (typically 2, for `(t, x)`) to `output_dim` (typically 1, for scalar `u`) through a list of `hidden_dims`, with GELU as the default activation, optional `LayerNorm` input normalization (`normalize_inputs`), and Xavier/Glorot weight initialization (`_initialize_weights`). Also provides `count_parameters()` and a descriptive `__repr__`. |
| `pinn_solver.py` | Defines `PDEProblemProtocol` (a `runtime_checkable` `Protocol` with `residual`, `boundary_condition`, `initial_condition` methods, used for duck-typing so `PINNSolver` doesn't need a hard import of `PDEProblem`) and `PINNSolver`, a plain class (not an `nn.Module`) wrapping a `PINNBase` network and a `pde_problem`. Provides `compute_physics_loss` (PDE residual MSE via autograd), `compute_bc_loss` (boundary condition MSE), `compute_ic_loss` (initial condition MSE at `t=0`), `predict`, `compute_derivatives` (computes `u`, `u_t`, `u_x`, `u_xx`, `u_tt`, `u_tx` via the `gradient` helper, with caching of `u_x` across `u_xx` if already computed), `train_mode`/`eval_mode`, and `parameters()` for optimizer construction. |

## For AI Agents

### Working In This Directory
- `PINNBase` is the base class every PINN network in this framework subclasses/instantiates directly — it is used as-is (not subclassed) by both `HybridPINNOperator` (`../hybrid/hybrid_model.py`) and `PINNSolver`. If you need a different architecture (e.g. Fourier features, SIREN), subclass `PINNBase`'s parent `nn.Module` directly or extend `PINNBase`, and keep the `input_dim`/`output_dim`/`hidden_dims`/`normalize_inputs` constructor signature stable since callers rely on it.
- `PINNSolver` is a plain Python class, not `nn.Module` — it does not register `network` as a submodule via `nn.Module.__setattr__`. If you add state that needs to move with `.to(device)`, remember callers must move `solver.network` directly.
- `compute_bc_loss` in `pinn_solver.py` (lines ~160-162) has its shape-matching guard commented out with unusual `#KB|`/`#JQ|`/`#PK|` prefixes (dead code, not currently executed) — unlike the equivalent live guard in `compute_ic_loss`. Be aware `compute_bc_loss` currently assumes `u_pred.shape == u_bc.shape` without correction; do not assume this comment block is intentional documentation.
- `compute_derivatives` takes a comma-separated `derivatives` string (e.g. `"u_t,u_x,u_xx,u_tt"`) rather than a list — follow this convention if extending it, and note it reuses an already-computed `u_x` for `u_xx` when both are requested to avoid a redundant autograd pass.
- All loss-computing methods clone-and-detach-then-`requires_grad_(True)` on `t`/`x` if not already tracking gradients, mirroring the pattern in `hybrid_model.py`.

### Testing Requirements
Covered by `tests/unit/test_pinn.py`. Run with:
`PYTHONPATH=src python -m pytest tests/unit/test_pinn.py -q`

### Common Patterns
- Consistent 1D→2D tensor coercion (`if t.dim() == 1: t = t.unsqueeze(-1)`) at the top of every public method that accepts `t`/`x`.
- `PDEProblemProtocol` uses `@runtime_checkable` so `isinstance(obj, PDEProblemProtocol)` works for duck-typed PDE objects without importing the concrete `PDEProblem` class — prefer this protocol-based typing over a concrete import when adding new solver-facing code that only needs `residual`/`boundary_condition`/`initial_condition`.
- Derivatives are computed via the shared `causal_pinn.utils.gradients.gradient(output, inputs, order=1, create_graph=True)` helper, not raw `torch.autograd.grad` calls — use the same helper for consistency.

## Dependencies

### Internal
`causal_pinn.utils.gradients.gradient` (autograd derivative helper used throughout `pinn_solver.py`).

### External
`torch`, `torch.nn`.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
