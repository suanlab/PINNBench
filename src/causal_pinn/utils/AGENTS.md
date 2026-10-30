<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# Utils

## Purpose
Shared low-level utilities used throughout `causal_pinn`: dataclass-based configuration objects for model/training/evaluation/experiment settings (`config.py`), reproducibility seeding (`random.py`), and automatic-differentiation helpers for computing PDE derivatives (`gradients.py`). This is a leaf/foundational module with no internal `causal_pinn` dependencies — everything else depends on it, not the reverse.

## Key Files
| File | Description |
|------|-------------|
| `config.py` | `ModelConfig` (PINN MLP hidden dims/activation, FNO modes/width/layers — comments note the FNO defaults were realigned from legacy values of modes=16/width=64 to the paper protocol modes=8/width=32), `TrainingConfig` (3-stage epoch counts/LRs aligned to the paper protocol — stage1=200, stage2=500, stage3=200 — plus all `adaptive_lambda_*` hyperparameters and `skip_stage1`/`skip_stage3` ablation flags), `EvalConfig` (eval point count and metric list), `ExperimentConfig` (top-level, nests the three above plus `output_dir`). |
| `random.py` | `set_seed(seed=42)` — seeds Python `random`, `numpy`, and `torch` (CPU and CUDA), and sets `torch.backends.cudnn.deterministic = True` / `benchmark = False` when CUDA is available. |
| `gradients.py` | `gradient(outputs, inputs, order=1, create_graph=True)` — repeated `torch.autograd.grad` calls for nth-order derivatives; `jacobian_column(outputs, inputs, column, create_graph=True)` — partial derivative w.r.t. a single input dimension. Both underlie PDE residual computation elsewhere in `causal_pinn.physics`. |
| `__init__.py` | Re-exports `ExperimentConfig`, `ModelConfig`, `TrainingConfig`, `EvalConfig`, `set_seed`, `gradient`, `jacobian_column`. |

## For AI Agents

### Working In This Directory
- Always use `set_seed()` for any new seeding needs rather than calling `torch.manual_seed`/`np.random.seed` directly — it is the single place that keeps random, numpy, and torch (CPU+CUDA) in sync and toggles deterministic cuDNN behavior.
- `TrainingConfig` defaults are load-bearing for reproducing paper results (stage epoch counts, `lambda_physics`/`lambda_data`, adaptive-lambda thresholds) — changing a default changes every caller that doesn't explicitly override it. Prefer passing an override at the call site over editing the dataclass default.
- `gradient()`/`jacobian_column()` both default `create_graph=True` and `retain_graph=True` — this is required for higher-order PDE derivatives (e.g. second-order residuals) but means callers computing only first-order, single-use gradients may want `create_graph=False` for memory efficiency.

### Testing Requirements
Covered by `tests/unit/test_utils.py`. Run with:
```
PYTHONPATH=src python -m pytest tests/unit/test_utils.py -q
```

### Common Patterns
- Config classes are plain `@dataclass`es with `from __future__ import annotations`; mutable defaults (lists) use `field(default_factory=lambda: [...])`.
- Autograd helpers normalize `outputs` to shape `(N, 1)` via `unsqueeze(-1)` when given a 1D tensor before differentiating.

## Dependencies

### Internal
None — this is a foundational/leaf module imported by `training/`, `evaluation/`, `physics/`, and elsewhere.

### External
- `torch` (autograd, seeding)
- `numpy` (seeding)
- `random` (stdlib, seeding)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
