<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# physics

## Purpose
The physics package is the domain layer of the Causal-Aware Hybrid PINN-Operator framework: it defines the PDE problems being solved (`pde/`) and the loss functions that train a network against them (`losses/`). The package's own `__init__.py` is empty and carries no re-exports — callers import directly from `causal_pinn.physics.pde` or `causal_pinn.physics.losses`. The central design pattern is the `residual(u, t, x)` method: every concrete PDE (`pde/`) implements it, and every loss (`losses/`) calls it, so a PDE is "plugged in" to training simply by passing its instance to a loss constructor.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Empty; no package-level exports. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `losses/` | PhysicsLoss, BoundaryLoss, CausalLoss/AdaptiveCausalLoss, TimeDecayLoss, and AdaptiveEpsilonScheduler (see `losses/AGENTS.md`) |
| `pde/` | `PDEProblem` abstract base and 9 concrete PDE definitions (heat, Burgers 1D/2D, advection-diffusion, KdV, Allen-Cahn, reaction-diffusion, wave, Navier-Stokes) (see `pde/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- This directory itself has no logic — do not add code here. New PDEs go in `pde/`, new loss types go in `losses/`.
- The coupling contract between the two subdirectories is the `residual(u, t, x) -> Tensor` signature. `pde/base.py` declares it as an abstract method on `PDEProblem`; `losses/causal_loss.py` and `losses/physics_loss.py` independently declare a structural `PDEProblemProtocol` with the same signature (duck-typed, not inheritance-based) so losses can accept any object exposing `residual`, not just `PDEProblem` subclasses.
- `u` passed into `residual` is always the *squeezed* network output (last singleton dim removed); `t` and `x` are the original leaf tensors with `requires_grad=True`, never reshaped copies, so `torch.autograd.grad` graphs stay intact.

### Testing Requirements
- No tests live directly under `physics/`. Coverage is provided by `tests/unit/test_losses.py` (losses) and `tests/unit/test_pde*.py`, `test_burgers1d.py`, `test_navier_stokes_2d.py`, `test_reaction_diffusion.py` (PDEs).
- Run: `PYTHONPATH=src python -m pytest tests/unit/test_pde.py tests/unit/test_losses.py -q`

### Common Patterns
- Every PDE class is a plain `@dataclass`-free `__init__`-based class storing physical coefficients as public float attributes (e.g. `self.viscosity`, `self.alpha`).
- Losses accept `pde_problem` in `__init__` and are callable (`__call__`) rather than exposing a `.forward()`/`.compute()` method, returning a `dict[str, Tensor]` with at least a `"total"` key.

## Dependencies

### Internal
- `causal_pinn.models.pinn.base.PINNBase` (TYPE_CHECKING-only import in losses, for type hints of the `network` argument).
- `causal_pinn.training.meta_router.PDE_FEATURE_REGISTRY` consumes PDE names/instances from `pde/` for routing; new PDEs that participate in routing must be registered there.

### External
- `torch` (autograd-based derivative computation is the core mechanism throughout).

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
