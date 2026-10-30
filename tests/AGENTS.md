<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# tests

## Purpose
Top-level test suite for the `causal_pinn` package. Split into `unit/` (component-level tests for PDEs, models, losses, training, evaluation, visualization) and `integration/` (full pipeline tests). All tests run on CPU with small models/few epochs so the whole suite stays fast enough for routine use.

## Key Files
| File | Description |
|------|-------------|
| `__init__.py` | Empty, marks `tests/` as a package for pytest discovery |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `unit/` | ~19 unit test files covering PDEs, operators, PINN/hybrid models, losses, training, evaluation, routing, visualization |
| `integration/` | End-to-end pipeline test (`test_end_to_end.py`): model creation → training → evaluation |

## For AI Agents

### Working In This Directory
- Add new tests under `unit/` if testing a single module/class in isolation, or `integration/` if the test exercises the full train/evaluate pipeline across multiple modules.
- `pyproject.toml` sets `pythonpath = ["src"]` and `testpaths = ["tests"]` for pytest, so tests import `causal_pinn` directly without manual path hacks.
- Keep tests CPU-safe and fast (small tensors, few epochs) — this suite is run routinely, not just before release.

### Testing Requirements
```bash
PYTHONPATH=src python -m pytest tests/ -q                                   # full suite
PYTHONPATH=src python -m pytest tests/unit/test_hybrid.py -q                # single file
PYTHONPATH=src python -m pytest tests/unit/test_hybrid.py::TestHybridPINNOperator::test_output_shape -q  # single test
make test                                                                    # equivalent Makefile target
```
Known flaky test on CPU (MKL FFT issue, environment-specific, does not affect CUDA path):
`tests/unit/test_operators.py::TestFNO::test_fno_gradient_flow_through_spectral_conv`.

### Common Patterns
- One `TestXxx` class per component/class under test; test methods named `test_<behavior>`.
- Module docstring states what the file verifies (shapes, gradients, edge cases, physics consistency).
- `from __future__ import annotations` at the top of most files.
- Seed via `causal_pinn.utils.random.set_seed` for reproducibility-sensitive tests.

## Dependencies

### Internal
`causal_pinn.models` (pinn, operators, hybrid), `causal_pinn.physics` (pde, losses), `causal_pinn.training`, `causal_pinn.evaluation`, `causal_pinn.utils`, `causal_pinn.visualization`.

### External
`pytest`, `torch`.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
