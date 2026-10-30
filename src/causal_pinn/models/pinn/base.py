"""PINN base network (MLP) with configurable architecture."""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn
from torch import Tensor


class PINNBase(nn.Module):
    """Multi-layer perceptron (MLP) base network for PINNs.

    Configurable architecture with support for:
    - Variable hidden layer dimensions
    - Multiple activation functions (default: GELU)
    - Optional input normalization

    Attributes:
        input_dim: Dimension of input features.
        output_dim: Dimension of output predictions.
        hidden_dims: List of hidden layer dimensions.
        activation: Activation function to use.
        normalize_inputs: Whether to apply input normalization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: Callable[[Tensor], Tensor] | None = None,
        normalize_inputs: bool = False,
    ) -> None:
        """Initialize the PINN base network.

        Args:
            input_dim: Dimension of input features (e.g., 2 for (t, x)).
            output_dim: Dimension of output predictions (e.g., 1 for scalar u).
            hidden_dims: List of hidden layer dimensions, e.g., [64, 64, 64].
            activation: Activation function. Defaults to nn.GELU().
            normalize_inputs: Whether to apply LayerNorm to inputs.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.normalize_inputs = normalize_inputs

        # Default activation to GELU
        if activation is None:
            self.activation = nn.GELU()
        else:
            self.activation = activation

        # Build network layers
        layers: list[nn.Module] = []

        # Optional input normalization
        if normalize_inputs:
            layers.append(nn.LayerNorm(input_dim))

        # Build hidden layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim

        # Output layer (no activation)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights using Xavier/Glorot initialization
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass through the network.

        Args:
            inputs: Input tensor of shape (batch_size, input_dim).
                   For PINNs, typically (N, 2) where columns are (t, x).

        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        return self.network(inputs)  # type: ignore[no-any-return]

    def count_parameters(self) -> int:
        """Count total number of trainable parameters.

        Returns:
            Total parameter count.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """String representation of the network."""
        return (
            f"PINNBase(input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"hidden_dims={self.hidden_dims}, normalize_inputs={self.normalize_inputs})"
        )
