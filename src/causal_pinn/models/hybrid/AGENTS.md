<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# hybrid

## Purpose
This directory implements the framework's central contribution: the `HybridPINNOperator`, which combines a neural operator (e.g. FNO) with a PINN via a learnable, time-aware `CausalGate`. The gate blends the two predictions as `output = gate * operator_pred + (1 - gate) * pinn_pred`, with a `causal_strength` term that biases the gate toward the PINN branch as `t → 1`, enforcing a causal preference for the physics-constrained solution at later times. The directory also provides a thin wrapper (`HybridModelWrapper`) that adapts the two-argument `forward(t, x)` interface to the single-tensor `forward(inputs)` interface expected by the generic trainer.

## Key Files
| File | Description |
|------|-------------|
| `causal_gate.py` | Defines `CausalGate(nn.Module)`: a small MLP (`Linear → GELU → Linear → GELU → Linear` to a scalar logit) that takes `[operator_pred, pinn_pred, t]` and outputs a sigmoid gate in `[0, 1]`. Subtracts `causal_strength * t` from the logit before the sigmoid to bias toward PINN at high `t`. Final layer weights/biases are zero-initialized so the gate starts at ~0.5 (equal weighting). Also provides `get_gate_stats()` for monitoring (mean/std/min/max/frac_operator/frac_pinn). |
| `hybrid_model.py` | Defines `HybridConfig` (dataclass of ablation flags: `use_operator`, `use_pinn`, `use_causal_gate`, and loss weights `operator_weight`, `physics_weight`, `gate_weight`); `HybridPINNOperator(nn.Module)`, which owns a `pinn_network` (`PINNBase`), an `operator` (`NeuralOperator`), a `causal_gate` (`CausalGate`), and a `pde_problem` (`PDEProblem`), and exposes `forward(t, x)`, `compute_hybrid_loss(t, x)` (returns operator/physics/causal/gate-reg losses plus gate stats and a weighted `total`), and `get_component_predictions(t, x)` for analysis; `HybridModelWrapper(nn.Module)`, which adapts `HybridPINNOperator` to a single `inputs` tensor of shape `(N, 2)` for `HybridTrainer`, and manually computes a 50/50 blend for the `use_operator=True, use_pinn=True, use_causal_gate=False` ablation case (since `HybridPINNOperator.forward` returns zeros for that combination by design). |

## For AI Agents

### Working In This Directory
- `HybridPINNOperator.forward` returns all-zero predictions when `use_causal_gate=False` but both `use_operator` and `use_pinn` are `True` — this is a known, deliberate gap that `HybridModelWrapper.forward` detects (via `config.use_operator and config.use_pinn and not config.use_causal_gate`) and patches with a manual 50/50 blend. If you modify the ablation logic in `HybridPINNOperator`, update `HybridModelWrapper` in lockstep or this silent-zero case will resurface.
- `_prepare_operator_input` converts `(t, x)` into the operator's expected tensor shape by checking `getattr(self.operator, "dim", 1)`: 1D operators get `(N, 2, 1)`, 2D operators get `(N, 2, 1, 1)`. When wiring in a new operator type, make sure it exposes a `dim` attribute or this dispatch silently defaults to the 1D path.
- `compute_hybrid_loss` requires `t` and `x` to allow gradients (it clones and calls `.requires_grad_(True)` if not already tracking) because both the PINN physics residual and the operator's smoothness regularizer are computed via `torch.autograd.grad`.
- `CausalGate.causal_strength` is a plain float, not a learned parameter — set it to `0` to disable the causal bias entirely (pure learned gating).
- Gate MLP final layer is intentionally zero-initialized (`_initialize_weights`) so training starts from a neutral 50/50 blend; do not change this without understanding the effect on early-training dynamics documented in the class docstring.

### Testing Requirements
Covered by `tests/unit/test_hybrid.py`. Run with:
`PYTHONPATH=src python -m pytest tests/unit/test_hybrid.py -q`

### Common Patterns
- Both classes ensure input tensors are 2D (`t.unsqueeze(-1) if t.dim() == 1 else t`) at the start of every public method — follow this convention for any new method accepting `t`/`x`.
- Losses are always returned as `dict[str, Tensor]` (or nested dict for `gate_stats`), never as a single scalar, so callers can log components individually.
- `TYPE_CHECKING`-guarded imports (`NeuralOperator`, `PINNBase`, `PDEProblem`) are used to avoid import cycles — add new cross-package type hints the same way rather than importing eagerly.

## Dependencies

### Internal
`causal_pinn.models.operators.base.NeuralOperator` (type-only), `causal_pinn.models.pinn.base.PINNBase` (type-only), `causal_pinn.physics.pde.base.PDEProblem` (type-only, used at runtime via `pde_problem.residual(...)`).

### External
`torch`, `torch.nn`.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
