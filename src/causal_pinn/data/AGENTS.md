<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# Data

## Purpose
Currently a placeholder/namespace package. `__init__.py` is empty (no docstring, no exports) — the directory exists as a reserved location for future dataset loading/generation code but contains no implementation today. Synthetic training data generation currently lives inline in `causal_pinn.training.trainer.HybridTrainer._generate_synthetic_data` and PDE-specific `exact_solution`/domain definitions in `causal_pinn.physics.pde`, not here.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Empty file. Marks `causal_pinn.data` as an importable Python package; defines no symbols. |

## For AI Agents

### Working In This Directory
- Before adding code here, check whether equivalent functionality already exists elsewhere (e.g. `causal_pinn.physics.pde.*` for PDE domain/exact-solution definitions, or `causal_pinn.training.trainer` for synthetic collocation/data-point generation) to avoid duplicating logic.
- If populating this package, add real exports to `__init__.py` following the pattern used by sibling modules (explicit `from .module import X` plus an `__all__` list).

### Testing Requirements
No dedicated test file exists yet (there is no `tests/unit/test_data.py`). If this package gains functionality, add a corresponding test module under `tests/unit/` and run with:
```
PYTHONPATH=src python -m pytest tests/unit/test_data.py -q
```

### Common Patterns
Not applicable — no code exists in this directory yet.

## Dependencies

### Internal
None.

### External
None.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
