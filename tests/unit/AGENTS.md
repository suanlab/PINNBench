<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# tests/unit

## Purpose
Unit tests for individual `causal_pinn` components: PDE problem definitions, neural operators, PINN/hybrid models, physics losses, training loop pieces, evaluation metrics, policy routing, and visualization utilities. Each file targets one module and tests it in isolation with small tensors and no real training.

## Key Files
| File | Description |
|------|-------------|
| `test_adaptive_lambda.py` | Tests for adaptive physics-loss-weight (lambda) scheduling |
| `test_burgers1d.py` | Tests for the Burgers1D PDE implementation (residual, exact/reference solution behavior) |
| `test_deeponet.py` | Tests for the DeepONet operator implementation |
| `test_early_predictor.py` | Tests for the early-training predictor module (used for policy/routing decisions from early dynamics) |
| `test_evaluation.py` | Tests for evaluation metrics and the physics-consistency checker |
| `test_hybrid.py` | Tests for `CausalGate`, `HybridConfig`, `HybridModelWrapper`, `HybridPINNOperator` |
| `test_losses.py` | Tests for physics loss functions (physics loss, causal loss) |
| `test_navier_stokes_2d.py` | Tests for the 2D Navier-Stokes PDE definition |
| `test_operators.py` | Tests for neural operator layers: `SpectralConv1d`/`2d`, `FourierLayer`, `NeuralOperator`, `FNO` |
| `test_pde.py` | Tests for PDE problem definitions across equation families |
| `test_pde_allen_cahn.py` | Tests for the Allen-Cahn PDE definition |
| `test_pde_kdv.py` | Tests for the Korteweg-de Vries (KdV) PDE definition |
| `test_pinn.py` | Tests for PINN model implementations |
| `test_pino.py` | Tests for the Physics-Informed Neural Operator (PINO) model |
| `test_policy_routing.py` | Tests for budget-aware policy routing helpers |
| `test_reaction_diffusion.py` | Tests for the reaction-diffusion PDE definition |
| `test_trainer.py` | Tests for the `HybridTrainer` training pipeline |
| `test_utils.py` | Tests for utility modules (e.g. seeding/config helpers) |
| `test_visualization.py` | Tests that each plot function runs without error, saving output to `/tmp` |

## For AI Agents

### Working In This Directory
- Match a new unit test to the module it exercises; prefer extending an existing `test_*.py`/`TestXxx` class over creating a new file for a component that already has one.
- Keep tests fast and CPU-only: small tensor shapes, few epochs/iterations, no GPU-only code paths (except where explicitly guarded).

### Testing Requirements
```bash
PYTHONPATH=src python -m pytest tests/unit/ -q
PYTHONPATH=src python -m pytest tests/unit/test_hybrid.py::TestHybridPINNOperator::test_output_shape -q
```
Known flaky test on CPU (MKL FFT issue, environment-specific, does not affect CUDA path):
`tests/unit/test_operators.py::TestFNO::test_fno_gradient_flow_through_spectral_conv`.

### Common Patterns
- One `TestXxx` class per class/behavior group under test.
- Shape/gradient/edge-case assertions dominate; some files assert physics properties (e.g. residual magnitude, gate values in `[0, 1]`).
- Imports go through the public `causal_pinn.*` module paths (e.g. `from causal_pinn.models.hybrid import CausalGate`).

## Dependencies

### Internal
`causal_pinn.models.{pinn,operators,hybrid}`, `causal_pinn.physics.{pde,losses}`, `causal_pinn.training`, `causal_pinn.evaluation`, `causal_pinn.utils`, `causal_pinn.visualization`.

### External
`pytest`, `torch`.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
