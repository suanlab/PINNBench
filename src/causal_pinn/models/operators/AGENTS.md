<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# operators

## Purpose
This directory contains the neural operator implementations used as the "fast approximate solution" branch of the hybrid model, plus the abstract base class they share. Neural operators learn resolution-invariant mappings between function spaces rather than pointwise functions, making them well suited for generalizing across discretizations of a PDE.

## Key Files
| File | Description |
|------|-------------|
| `base.py` | Defines `NeuralOperator(ABC, torch.nn.Module)`, the abstract base class all operators must subclass. Requires implementations of `forward(x) -> Tensor` and `get_num_params() -> int`. |
| `deeponet.py` | Implements `DeepONet(NeuralOperator)`: a branch-trunk architecture. `branch_net` encodes `x.mean(dim=-1)` (a pooled summary of the input function), `trunk_net` encodes per-point features (`x.transpose(1, 2)`), and the outputs are fused by elementwise product then projected by `output_layer` to `out_channels`. Currently restricted to `dim=1` and requires `branch_dim == trunk_dim` (both enforced by `ValueError` in `__init__`). Includes the private helper `_build_mlp(input_dim, output_dim, num_layers)` used to build both branch and trunk networks. |
| `fno.py` | Implements the Fourier Neural Operator stack: `SpectralConv1d` and `SpectralConv2d` (FFT-based spectral convolution layers that learn complex weights — stored as separate `weight_real`/`weight_imag` parameters — for a truncated set of low-frequency Fourier modes), `FourierLayer` (spectral conv + a 1x1 conv skip connection + GELU, dispatches to 1D or 2D spectral conv based on `dim`), and `FNO(NeuralOperator)` (lifting conv → stack of `num_layers` `FourierLayer`s → projection conv). Supports both 1D (`dim=1`, `nn.Conv1d`) and 2D (`dim=2`, `nn.Conv2d`) spatial domains, with `modes` accepted as an `int` or `(modes1, modes2)` tuple for 2D. |

## For AI Agents

### Working In This Directory
- Any new operator MUST subclass `NeuralOperator` from `base.py` and implement both abstract methods: `forward(x)` and `get_num_params()`. `get_num_params()` is conventionally implemented as `sum(p.numel() for p in self.parameters() if p.requires_grad)` in both existing subclasses — match this.
- `hybrid_model.py` (in `../hybrid/`) introspects `getattr(self.operator, "dim", 1)` to decide how to shape inputs (1D vs 2D). Any new operator intended for use inside `HybridPINNOperator` must expose a `dim` attribute (`1` or `2`).
- `DeepONet` currently hard-fails (`ValueError`) for `dim != 1` and for `branch_dim != trunk_dim` — these are real constraints, not placeholders; do not silently relax them without updating call sites.
- `SpectralConv1d`/`SpectralConv2d` store complex weights as separate real/imaginary `nn.Parameter` tensors (not `torch.complex` parameters directly, since PyTorch autograd support for complex parameters is limited) and reconstruct the complex weight via `torch.complex(weight_real, weight_imag)` inside `compl_mul1d`/`compl_mul2d`. Follow this pattern for any new spectral layer.
- Mode truncation is defensive: `modes_to_keep = min(self.modes, x_ft.shape[-1])` guards against requesting more Fourier modes than the input's Nyquist-limited spectrum actually has.

### Testing Requirements
Covered by `tests/unit/test_operators.py` (base class, FNO, spectral conv layers) and `tests/unit/test_deeponet.py` (DeepONet-specific). Run with:
`PYTHONPATH=src python -m pytest tests/unit/test_operators.py tests/unit/test_deeponet.py -q`

### Common Patterns
- All `forward` methods return raw tensors annotated `# type: ignore[no-any-return]` where PyTorch's `nn.Module.__call__` typing loses precision — follow this convention rather than adding wrapper casts.
- Every class-level docstring documents expected input/output tensor shapes explicitly (e.g. `(batch, in_channels, spatial_dim)`) — new operators should do the same since shape mismatches are the primary integration risk with `hybrid_model.py`.

## Dependencies

### Internal
None (this subpackage is self-contained; `base.py` has no dependency on other `causal_pinn` modules).

### External
`torch`, `torch.nn`, `torch.fft` (for `rfft`/`irfft`/`rfft2`/`irfft2` in the spectral conv layers).

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
