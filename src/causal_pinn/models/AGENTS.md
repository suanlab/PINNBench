<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# models

## Purpose
This directory is the top-level namespace package for all neural network architectures in the Causal-Aware Hybrid PINN-Operator framework. It has no code of its own (the package `__init__.py` is empty) — it exists purely to group the three model families (neural operators, PINNs, PINOs) and the hybrid model that fuses operator and PINN predictions via causal gating. All actual model logic lives in the four subdirectories below.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Empty package marker file; no exports, re-exports, or logic. |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `hybrid/` | `HybridPINNOperator` and `CausalGate` — blends a neural operator and a PINN with a learnable, time-biased gate (see `hybrid/AGENTS.md`). |
| `operators/` | Neural operator architectures (`FNO`, `DeepONet`) and the `NeuralOperator` abstract base class (see `operators/AGENTS.md`). |
| `pinn/` | Physics-Informed Neural Network base MLP (`PINNBase`) and the `PINNSolver` training-loss helper (see `pinn/AGENTS.md`). |
| `pino/` | Physics-Informed Neural Operator (`PINOModel`) — an FNO trained with a PDE-residual loss instead of pure supervised data (see `pino/AGENTS.md`). |

## For AI Agents

### Working In This Directory
- Do not add code directly to this `__init__.py` — it is intentionally empty. New models belong in one of the four subpackages, not here.
- When adding a new model family, create a new subdirectory with its own `__init__.py` and its own `AGENTS.md` following the pattern of the existing four.
- Import paths are always fully qualified from the package root, e.g. `from causal_pinn.models.hybrid.hybrid_model import HybridPINNOperator` — there is no re-export at this top level.

### Testing Requirements
No tests are collected against this directory directly (it has no logic). Tests for the model families live under `tests/unit/`: `test_hybrid.py`, `test_operators.py`, `test_deeponet.py`, `test_pinn.py`, `test_pino.py`. Run the full model suite with:
`PYTHONPATH=src python -m pytest tests/unit/test_hybrid.py tests/unit/test_operators.py tests/unit/test_deeponet.py tests/unit/test_pinn.py tests/unit/test_pino.py -q`

### Common Patterns
Every subpackage exposes its public API through its own `__init__.py` using explicit `__all__` lists — there is no wildcard re-export from this parent package.

## Dependencies

### Internal
None directly; subpackages depend on `causal_pinn.physics.pde` (PDE problem definitions) and `causal_pinn.utils.gradients` (autograd derivative helper).

### External
`torch` (all model subpackages build on `torch.nn.Module`).

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
