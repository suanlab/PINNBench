"""Abstract base class for neural operators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class NeuralOperator(ABC, torch.nn.Module):
    """Abstract base class for neural operators.

    Neural operators learn mappings between function spaces, enabling
    resolution-invariant learning of PDE solutions. This base class
    defines the common interface for all neural operator implementations.

    Subclasses must implement:
        - forward(x): Apply the operator to input tensor
        - get_num_params(): Return total number of trainable parameters
    """

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Apply the neural operator to input tensor.

        Args:
            x: Input tensor. Shape depends on the operator type:
               - For 1D operators: (batch, channels, spatial_dim)
               - For 2D operators: (batch, channels, height, width)

        Returns:
            Output tensor with the same spatial dimensions but potentially
            different channel dimension based on the operator configuration.
        """
        ...

    @abstractmethod
    def get_num_params(self) -> int:
        """Return the total number of trainable parameters.

        Returns:
            Total count of trainable parameters in the model.
        """
        ...
