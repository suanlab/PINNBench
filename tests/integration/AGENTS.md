<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# tests/integration

## Purpose
End-to-end integration tests for the Causal-Aware Hybrid PINN-Operator framework: model creation, training, and evaluation exercised together as a full pipeline rather than in isolation. Uses small models and few epochs to keep runs fast.

## Key Files
| File | Description |
|------|-------------|
| `test_end_to_end.py` | Full pipeline test: builds `HybridPINNOperator`/`CausalGate`, trains with `HybridTrainer` (with `EarlyStoppingCallback`/`LoggingCallback`), evaluates with `BenchmarkRunner`/`MetricSuite`/`PhysicsConsistencyChecker`, across PDEs such as `Heat1D` and `AdvectionDiffusion1D`, using `CausalLoss`/`PhysicsLoss` |

## For AI Agents

### Working In This Directory
- Use this directory only for tests that span multiple modules (model + training + evaluation together); single-module tests belong in `tests/unit/`.
- Keep fixtures (e.g. `heat_pde`) and pipeline configs minimal — small models, few epochs — so the integration suite stays fast in CI.

### Testing Requirements
```bash
PYTHONPATH=src python -m pytest tests/integration/ -q
PYTHONPATH=src python -m pytest tests/integration/test_end_to_end.py -q
```

### Common Patterns
- Pytest fixtures (`@pytest.fixture`) build shared PDE/model objects reused across tests in the file.
- Imports use fully-qualified submodule paths (e.g. `causal_pinn.models.hybrid.hybrid_model`, `causal_pinn.physics.pde.heat_1d`) rather than the shorter package-level re-exports seen in some unit tests.
- `set_seed` from `causal_pinn.utils.random` is used to make pipeline runs reproducible.

## Dependencies

### Internal
`causal_pinn.evaluation.{benchmark_runner,metrics,physics_consistency}`, `causal_pinn.models.hybrid.{causal_gate,hybrid_model}`, `causal_pinn.models.operators.fno`, `causal_pinn.models.pinn.base`, `causal_pinn.physics.losses.{causal_loss,physics_loss}`, `causal_pinn.physics.pde.{advection_diffusion_1d,heat_1d}`, `causal_pinn.training.{callbacks,trainer}`, `causal_pinn.utils.{config,random}`.

### External
`pytest`, `torch`.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
