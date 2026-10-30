<!-- Parent: ../../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# causal_pinn (library root)

## Purpose
The installable library package. Provides the full stack for hybrid PINN-operator PDE solving: model architectures (`models/`), PDE definitions and physics losses (`physics/`), the multi-stage training pipeline and routing/policy-selection machinery (`training/`), evaluation metrics and benchmark harness (`evaluation/`), and shared utilities (`utils/`). Everything downstream — every experiment script and test — imports from here.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Package docstring and `__version__` ("0.1.0"); intentionally lightweight (no re-exports) |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `models/` | Operator, PINN, PINO, and hybrid model architectures (see `models/AGENTS.md`) |
| `physics/` | PDE problem definitions and physics/causal loss terms (see `physics/AGENTS.md`) |
| `training/` | 3-stage `HybridTrainer` and the routing/policy-selection stack (see `training/AGENTS.md`) |
| `evaluation/` | Metrics, benchmark runner, physics-consistency checks (see `evaluation/AGENTS.md`) |
| `utils/` | Config, seeding, gradient helpers (see `utils/AGENTS.md`) |
| `visualization/` | Plotting utilities for solutions and diagnostics (see `visualization/AGENTS.md`) |
| `data/` | Namespace/placeholder for data-generation helpers (see `data/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Import paths are absolute from the package root, e.g. `from causal_pinn.models.hybrid.hybrid_model import HybridPINNOperator`.
- The dependency direction is one-way: `physics` and `models` are consumed by `training`, which is consumed by `evaluation` and by experiment scripts. Avoid introducing back-edges (e.g. `physics` importing `training`).
- The single hook coupling a PDE to models and losses is `PDEProblem.residual(u, t, x)` — respect that contract when extending either side.

### Testing Requirements
Each subpackage has matching unit tests under `tests/unit/`. Run the whole library suite with `PYTHONPATH=src python -m pytest tests/ -q`.

### Common Patterns
- New functions require type annotations (mypy `disallow_untyped_defs`).
- All seeding goes through `utils.random.set_seed`; autograd/derivative helpers live in `utils.gradients`.
- Modules use `from __future__ import annotations` and `TYPE_CHECKING` guards to keep import graphs light.

## Dependencies

### External
- `torch` — all models, losses, and training
- `numpy`, `scipy`, `einops` — numerics
- `scikit-learn` — routing regressors in `training/`

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
