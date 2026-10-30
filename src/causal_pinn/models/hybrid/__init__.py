"""Hybrid PINN-Operator models with causal gating.

This module provides hybrid models that combine neural operators (e.g., FNO)
with Physics-Informed Neural Networks (PINNs) using learnable causal gates
for time-aware blending.
"""

from causal_pinn.models.hybrid.causal_gate import CausalGate
from causal_pinn.models.hybrid.hybrid_model import (
    HybridConfig,
    HybridModelWrapper,
    HybridPINNOperator,
)

__all__ = [
    "CausalGate",
    "HybridConfig",
    "HybridModelWrapper",
    "HybridPINNOperator",
]
