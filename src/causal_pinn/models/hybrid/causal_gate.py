"""CausalGate attention mechanism for hybrid PINN-operator models.

The CausalGate learns to blend predictions from a neural operator and a PINN
based on time coordinate, with a bias toward PINN for later times to enforce
causal consistency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CausalGate(nn.Module):
    """Learnable gating mechanism for blending operator and PINN predictions.

    The gate learns to combine predictions from two sources (neural operator
    and PINN) based on the predictions themselves and the time coordinate.
    A causal_strength parameter biases the gate toward PINN at later times.

    The blending formula is:
        output = gate * operator_pred + (1 - gate) * pinn_pred

    Where gate ∈ [0, 1] is learned via a small MLP with sigmoid activation.

    Attributes:
        out_dim: Output dimension of predictions.
        hidden_dim: Hidden dimension of the gate MLP.
        causal_strength: Strength of causal bias toward PINN at later times.
                         Higher values mean stronger preference for PINN at t→1.
    """

    def __init__(
        self,
        out_dim: int,
        hidden_dim: int = 32,
        causal_strength: float = 1.0,
    ) -> None:
        """Initialize the CausalGate.

        Args:
            out_dim: Dimension of output predictions (from operator/PINN).
            hidden_dim: Hidden layer dimension for the gate MLP.
            causal_strength: Bias strength toward PINN at later times.
                            When > 0, the gate tends toward 0 (PINN) as t → 1.
                            Set to 0 to disable causal bias.
        """
        super().__init__()

        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.causal_strength = causal_strength

        # Gate MLP: takes [operator_pred, pinn_pred, t] and outputs scalar gate
        # Input dim: out_dim (operator) + out_dim (pinn) + 1 (time) = 2 * out_dim + 1
        input_dim = 2 * out_dim + 1

        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize final layer to output ~0.5 initially (equal weighting)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize gate network for balanced initial gating."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Initialize final layer to give ~0.5 sigmoid output initially
        # sigmoid(0) = 0.5
        final_layer = self.gate_mlp[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(
        self,
        operator_pred: Tensor,
        pinn_pred: Tensor,
        t: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute the gated blend of operator and PINN predictions.

        Args:
            operator_pred: Predictions from neural operator, shape (N, out_dim).
            pinn_pred: Predictions from PINN, shape (N, out_dim).
            t: Time coordinates, shape (N, 1). Expected to be in [0, 1] range.

        Returns:
            Tuple of:
                - blended_pred: Gated combination, shape (N, out_dim).
                - gate_values: Gate values in [0, 1], shape (N, 1).
                              Values close to 1 favor operator, close to 0 favor PINN.
        """
        # Ensure t is 2D
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Concatenate inputs for gate MLP
        # Shape: (N, 2 * out_dim + 1)
        gate_input = torch.cat([operator_pred, pinn_pred, t], dim=-1)

        # Compute raw gate logits
        gate_logits = self.gate_mlp(gate_input)

        # Apply causal bias: subtract causal_strength * t to favor PINN at later times
        # Higher t → lower gate logits → lower gate value → more PINN
        if self.causal_strength > 0:
            gate_logits = gate_logits - self.causal_strength * t

        # Sigmoid to get gate values in [0, 1]
        gate_values = torch.sigmoid(gate_logits)

        # Compute blended prediction
        blended_pred = gate_values * operator_pred + (1 - gate_values) * pinn_pred

        return blended_pred, gate_values

    def get_gate_stats(self, gate_values: Tensor) -> dict[str, Tensor]:
        """Compute statistics about gate values for monitoring.

        Args:
            gate_values: Gate values from forward pass, shape (N, 1) or (N,).

        Returns:
            Dictionary with gate statistics:
                - mean: Mean gate value
                - std: Standard deviation
                - min: Minimum value
                - max: Maximum value
                - frac_operator: Fraction of predictions dominated by operator (gate > 0.5)
                - frac_pinn: Fraction dominated by PINN (gate < 0.5)
        """
        gate_flat = gate_values.flatten()

        return {
            "mean": gate_flat.mean(),
            "std": gate_flat.std() if gate_flat.numel() > 1 else torch.tensor(0.0),
            "min": gate_flat.min(),
            "max": gate_flat.max(),
            "frac_operator": (gate_flat > 0.5).float().mean(),
            "frac_pinn": (gate_flat < 0.5).float().mean(),
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CausalGate(out_dim={self.out_dim}, hidden_dim={self.hidden_dim}, "
            f"causal_strength={self.causal_strength})"
        )
