<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# losses

## Purpose
Implements the loss functions used to train PINNs against a PDE: standard physics-residual loss, boundary/initial-condition loss, and the causal-weighted losses that are the core research contribution of Paper C. All losses are plain callables that take a network and collocation coordinates, invoke the PDE's `residual(u, t, x)` method, and return a `dict[str, Tensor]` (always including a `"total"` key) so they can be composed and summed in a training loop.

## Key Files
| File | Description |
|------|-------------|
| `physics_loss.py` | `PhysicsLoss`: MSE of the PDE residual at collocation points via `torch.autograd`-based derivatives (`create_graph=True`). `HigherOrderPhysicsLoss`: subclass that additionally computes and returns first-order derivatives (`u`, `u_t`, `u_x`) alongside the residual, using `causal_pinn.utils.gradients.gradient`. Both declare a local structural `PDEProblemProtocol`. |
| `boundary_loss.py` | `BoundaryLoss`: MSE loss for Dirichlet-style boundary conditions (`compute_bc_loss`) and initial conditions (`compute_ic_loss`, evaluated at t=0). `DirichletBC`: thin subclass fixing `bc_weight == ic_weight`. `NeumannBC`: standalone class computing the normal derivative at boundary points via autograd and comparing to a target `du_dn_true`. |
| `causal_loss.py` | `CausalLoss`: implements the causal training strategy from Wang et al. 2022 — weights the residual loss at time t_n by `exp(-epsilon * sum_{m<n} MSE(residual_m))`, enforcing that earlier times are learned before later ones. Provides both an O(N²) per-point method (`compute_causal_weights`) and an O(N) time-binned approximation (`compute_causal_weights_binned`, default, `use_binning=True`). Supports a `min_weight` floor (default 0.0 = baseline Wang et al. behavior) to prevent causal weights from collapsing to zero at late times. `AdaptiveCausalLoss`: subclasses `CausalLoss`, linearly anneals `epsilon` from `epsilon_init` to `epsilon_final` over `annealing_steps` (via `step()`/`set_step()`) instead of using a fixed epsilon. `TimeDecayLoss`: simpler alternative that weights purely by `exp(-alpha * t_normalized)`, ignoring residual accumulation entirely. Declares its own `PDEProblemProtocol`. |
| `adaptive_epsilon.py` | `AdaptiveEpsilonScheduler`: computes `epsilon_adaptive = epsilon_base / (1 + beta * mean_residual)`, clamped to `[epsilon_min, epsilon_max]`. This is a *different* mechanism from `AdaptiveCausalLoss`'s step-based annealing — it reacts to the current residual magnitude (used to avoid causal-weight collapse when training without operator pretraining) rather than to a training-step schedule. Keeps a `history` list of `(mean_residual, epsilon)` pairs; not wired directly into `CausalLoss` itself (the caller is expected to call `compute_epsilon` and feed the result into `CausalLoss.epsilon`). |
| `__init__.py` | Re-exports all public loss classes: `PhysicsLoss`, `HigherOrderPhysicsLoss`, `BoundaryLoss`, `DirichletBC`, `NeumannBC`, `CausalLoss`, `AdaptiveCausalLoss`, `TimeDecayLoss`, `AdaptiveEpsilonScheduler`. |

## For AI Agents

### Working In This Directory
- Every loss couples to a PDE purely through the `PDEProblemProtocol` (a `@runtime_checkable` `Protocol` requiring `residual(self, u, t, x) -> Tensor`), independently redeclared in `physics_loss.py` and `causal_loss.py` — not imported from `pde/base.py`. This is a deliberate structural-typing choice: any object with a matching `residual` method works, whether or not it subclasses `PDEProblem`. If you add a new loss requiring PDE access, follow this pattern rather than importing the concrete `PDEProblem` base class.
- Invariant: before calling `network(...)`, `t` and `x` must have `requires_grad=True` set on the *original* tensors (not reshaped copies) — reshaping/unsqueezing breaks the autograd graph needed for `torch.autograd.grad` in `residual()`. Every loss's `__call__`/`compute_*` method clones-and-requires-grad only if not already tracking, then keeps a `_leaf` reference to the untouched tensor to pass into `residual`.
- `CausalLoss.__call__` and `TimeDecayLoss.__call__` accept `t`, `x` directly (not pre-squeezed) and internally unsqueeze to `(N, 1)` for the network's `cat([t, x], dim=-1)` input, but pass the original 1D leaf tensors to `pde_problem.residual`.
- `CausalLoss.compute_causal_weights` (unbinned) is O(N) per point via a Python loop (`for i in range(1, n)`) — intended for small batches / testing; `compute_causal_weights_binned` is the efficient default used by `__call__(use_binning=True)`.
- `min_weight` on `CausalLoss`/`AdaptiveCausalLoss` is an ablation knob (see `experiments/scripts/paper_c_causal/clamping_ablation.py`) — leave default `0.0` unless explicitly studying causal-weight collapse.

### Testing Requirements
- Covered by `tests/unit/test_losses.py`.
- Run: `PYTHONPATH=src python -m pytest tests/unit/test_losses.py -q`

### Common Patterns
- Constructor signature convention: `pde_problem` first positional arg, then loss-specific hyperparameters, then `weight: float = 1.0` last.
- All `__call__` methods return a dict containing raw unweighted quantities (e.g. `mse_residual`, `residual`) alongside the weighted component (e.g. `physics_loss`, `causal_loss`) and always a `"total"` key — this lets a training loop sum `"total"` across multiple loss objects while still logging the unweighted diagnostics.
- Shape handling convention: accept both `(N,)` and `(N, 1)` inputs; internally `.unsqueeze(-1)` before `torch.cat` for network input, `.reshape(-1)` / `.flatten()` before elementwise math.
- `from __future__ import annotations` used in every module (except `physics_loss.py`, which omits it despite otherwise matching style) for postponed evaluation of type hints; `PINNBase` is only imported under `TYPE_CHECKING` to avoid a runtime circular import.

## Dependencies

### Internal
- `causal_pinn.models.pinn.base.PINNBase` (type-hint only, `TYPE_CHECKING`).
- `causal_pinn.utils.gradients.gradient` (used only by `HigherOrderPhysicsLoss` in `physics_loss.py`).
- Structurally depends on `causal_pinn.physics.pde.*` classes via duck typing (no direct import).

### External
- `torch` (`torch.autograd.grad`, `Tensor`) — the sole external dependency across all five files.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
