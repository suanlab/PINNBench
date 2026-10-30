"""Abstract base class for PDE problem definitions.

This module provides the foundational abstraction for defining partial differential
equations (PDEs) in a manner compatible with Physics-Informed Neural Networks (PINNs).
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class PDEProblem(ABC):
    """Abstract base class for PDE problem definitions.

    This class defines the interface for PDE problems that can be solved using
    Physics-Informed Neural Networks (PINNs). Each concrete implementation must
    define the governing equation residual, boundary conditions, and initial conditions.

    The residual method computes the PDE violation for a given solution field u(t, x),
    which is the core component used in PINN training to enforce physical constraints.

    Attributes:
        t_range: Tuple of (t_min, t_max) defining the temporal domain.
        x_range: Tuple of (x_min, x_max) for 1D or (x_min, x_max, y_min, y_max) for 2D
                 defining the spatial domain.
    """

    @property
    @abstractmethod
    def domain_bounds(self) -> tuple[tuple[float, float], tuple[float, ...]]:
        """Return the domain bounds for time and space.

        Returns:
            A tuple of (t_bounds, x_bounds) where:
            - t_bounds: (t_min, t_max) temporal domain
            - x_bounds: (x_min, x_max) for 1D or (x_min, x_max, y_min, y_max) for 2D

        Example:
            >>> pde.domain_bounds()
            ((0.0, 1.0), (0.0, 1.0))  # 1D case
            >>> pde.domain_bounds()
            ((0.0, 1.0), (0.0, 1.0, 0.0, 1.0))  # 2D case
        """
        pass

    @abstractmethod
    def residual(self, u: Tensor, t: Tensor, x: Tensor) -> Tensor:
        """Compute the PDE residual for the given solution field.

        The residual measures how much the solution u(t, x) violates the governing
        PDE. For a perfect solution, the residual should be zero everywhere.

        This method uses automatic differentiation to compute required spatial
        and temporal derivatives of u with respect to t and x.

        Args:
            u: Solution field tensor of shape (batch_size,) or (batch_size, 1).
               Must have requires_grad=True for automatic differentiation.
            t: Time coordinates of shape (batch_size,) or (batch_size, 1).
            x: Spatial coordinates. Shape (batch_size,) for 1D or (batch_size, 2)
               for 2D problems.

        Returns:
            Residual tensor of shape (batch_size,) or (batch_size, 1).
            Values close to zero indicate the PDE is satisfied.

        Note:
            The input tensors t and x should be used to compute derivatives via
            torch.autograd.grad, not the u tensor directly for complex PDEs.
        """
        pass

    @abstractmethod
    def boundary_condition(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute the boundary condition value at the given coordinates.

        Args:
            t: Time coordinates of shape (batch_size,) or (batch_size, 1).
            x: Spatial coordinates at the boundary. Shape (batch_size,) for 1D
               or (batch_size, 2) for 2D problems.

        Returns:
            Boundary condition values of shape (batch_size,) or (batch_size, 1).

        Note:
            The specific boundary (left, right, top, bottom, etc.) is determined
            by the spatial coordinates x. Implementations should handle all
            relevant boundary conditions for the problem.
        """
        pass

    @abstractmethod
    def initial_condition(self, x: Tensor) -> Tensor:
        """Compute the initial condition value at t=0.

        Args:
            x: Spatial coordinates. Shape (batch_size,) for 1D or (batch_size, 2)
               for 2D problems.

        Returns:
            Initial condition values of shape (batch_size,) or (batch_size, 1).
        """
        pass

    def exact_solution(self, t: Tensor, x: Tensor) -> Tensor | None:
        """Compute the exact analytical solution (if available).

        This method is optional and should be implemented when an analytical
        solution exists for validation and testing purposes.

        Args:
            t: Time coordinates of shape (batch_size,) or (batch_size, 1).
            x: Spatial coordinates. Shape (batch_size,) for 1D or (batch_size, 2)
               for 2D problems.

        Returns:
            Exact solution values of shape (batch_size,) or (batch_size, 1),
            or None if no analytical solution is available.

        Note:
            When implemented, this can be used to:
            - Verify PINN solutions against ground truth
            - Generate training data for operator learning
            - Test the residual computation accuracy
        """
        return None

    @staticmethod
    def _ensure_tensor(x: Tensor | float) -> Tensor:
        """Ensure input is a tensor with proper shape.

        Args:
            x: Input value, either a tensor or scalar.

        Returns:
            Tensor with shape preserved or converted from scalar.
        """
        if isinstance(x, (int, float)):
            return torch.tensor(x, dtype=torch.float32)
        return x

    @staticmethod
    def _compute_derivative(
        output: Tensor,
        input_var: Tensor,
        grad_outputs: Tensor | None = None,
        create_graph: bool = True,
    ) -> Tensor:
        """Compute derivative of output with respect to input_var.

        This is a convenience method for computing gradients using autograd.

        Args:
            output: Output tensor to differentiate.
            input_var: Input variable to differentiate with respect to.
            grad_outputs: Optional gradient outputs for backward pass.
            create_graph: Whether to create computational graph for higher-order
                         derivatives.

        Returns:
            Gradient tensor with same shape as input_var.
        """
        if grad_outputs is None:
            grad_outputs = torch.ones_like(output)

        gradient = torch.autograd.grad(
            outputs=output,
            inputs=input_var,
            grad_outputs=grad_outputs,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if gradient is None:
            return torch.zeros_like(input_var)

        return gradient
