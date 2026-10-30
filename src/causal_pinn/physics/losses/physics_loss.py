"""Physics residual loss computation for PINNs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import Tensor

from causal_pinn.utils.gradients import gradient

if TYPE_CHECKING:
    from causal_pinn.models.pinn.base import PINNBase


@runtime_checkable
class PDEProblemProtocol(Protocol):
    """Protocol for PDE problem interface."""

    def residual(self, u: Tensor, t: Tensor, x: Tensor) -> Tensor:
        """Compute PDE residual."""
        ...


class PhysicsLoss:
    """Physics-based loss using PDE residual.

    Computes the mean squared residual of the PDE at collocation points.
    Uses torch.autograd for computing derivatives with create_graph=True
    to enable gradient flow for optimization.

    The physics loss ensures the network learns to satisfy the PDE:
        L_physics = (1/N) * sum(|residual(u_θ(t, x))|²)

    Attributes:
        pde_problem: The PDE problem defining the residual.
        weight: Weight factor for the physics loss.
    """

    def __init__(
        self,
        pde_problem: PDEProblemProtocol,
        weight: float = 1.0,
    ) -> None:
        """Initialize the physics loss.

        Args:
            pde_problem: PDE problem with a residual(u, t, x) method.
            weight: Weight factor for the loss component.
        """
        self.pde_problem = pde_problem
        self.weight = weight

    def compute_residual(
        self,
        network: PINNBase,
        t: Tensor,
        x: Tensor,
    ) -> Tensor:
        """Compute PDE residual at given collocation points.

        Args:
            network: Neural network approximating the solution.
            t: Time coordinates with requires_grad=True, shape (N,) or (N, 1).
            x: Spatial coordinates with requires_grad=True, shape (N,) or (N, 1).

        Returns:
            PDE residual tensor of shape (N,) or (N, 1).
        """
        # Ensure gradients are tracked for autograd
        if not t.requires_grad:
            t = t.clone().requires_grad_(True)
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)

        # Store original references (leaf tensors) for autograd graph
        t_leaf = t
        x_leaf = x

        # Ensure 2D tensors for network input
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Concatenate inputs
        inputs = torch.cat([t, x], dim=-1)

        # Forward pass - network predicts u(t, x)
        u = network(inputs)

        # Compute PDE residual using the problem definition
        # Pass original leaf tensors (not squeezed copies) to preserve autograd graph
        residual = self.pde_problem.residual(u.squeeze(-1), t_leaf, x_leaf)

        return residual

    def __call__(
        self,
        network: PINNBase,
        t: Tensor,
        x: Tensor,
    ) -> dict[str, Tensor]:
        """Compute physics loss.

        Args:
            network: Neural network approximating the solution.
            t: Time coordinates at collocation points.
            x: Spatial coordinates at collocation points.

        Returns:
            Dictionary containing:
                - 'residual': Raw residual tensor
                - 'mse_residual': Mean squared residual
                - 'physics_loss': Weighted physics loss
                - 'total': Total weighted loss (same as physics_loss here)
        """
        residual = self.compute_residual(network, t, x)

        # Mean squared residual
        mse_residual = torch.mean(residual**2)

        # Weighted loss
        physics_loss = self.weight * mse_residual

        return {
            "residual": residual,
            "mse_residual": mse_residual,
            "physics_loss": physics_loss,
            "total": physics_loss,
        }


class HigherOrderPhysicsLoss(PhysicsLoss):
    """Physics loss with explicit derivative computation.

    Extends PhysicsLoss to explicitly compute and return derivatives
    for debugging and analysis purposes.
    """

    def compute_derivatives_and_residual(
        self,
        network: PINNBase,
        t: Tensor,
        x: Tensor,
    ) -> tuple[dict[str, Tensor], Tensor]:
        """Compute solution derivatives and PDE residual.

        Args:
            network: Neural network approximating the solution.
            t: Time coordinates with requires_grad=True.
            x: Spatial coordinates with requires_grad=True.

        Returns:
            Tuple of (derivatives_dict, residual_tensor).
            derivatives_dict contains: u, u_t, u_x (and u_tt, u_xx if needed).
        """
        # Ensure gradients are tracked
        if not t.requires_grad:
            t = t.clone().requires_grad_(True)
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)

        # Ensure 2D tensors
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Concatenate inputs
        inputs = torch.cat([t, x], dim=-1)

        # Forward pass
        u = network(inputs)

        derivatives: dict[str, Tensor] = {"u": u}

        # Compute first-order derivatives with create_graph=True
        # This allows gradients to flow through the derivative computation
        grad_u = gradient(u, inputs, order=1, create_graph=True)
        u_t = grad_u[:, 0:1]
        u_x = grad_u[:, 1:2]

        derivatives["u_t"] = u_t
        derivatives["u_x"] = u_x

        # Compute residual
        residual = self.pde_problem.residual(
            u.squeeze(-1), t.squeeze(-1), x.squeeze(-1)
        )

        return derivatives, residual

    def __call__(
        self,
        network: PINNBase,
        t: Tensor,
        x: Tensor,
    ) -> dict[str, Tensor]:
        """Compute physics loss with explicit derivatives.

        Args:
            network: Neural network approximating the solution.
            t: Time coordinates at collocation points.
            x: Spatial coordinates at collocation points.

        Returns:
            Dictionary containing:
                - 'u': Solution prediction
                - 'u_t': Time derivative
                - 'u_x': Spatial derivative
                - 'residual': Raw PDE residual
                - 'mse_residual': Mean squared residual
                - 'physics_loss': Weighted physics loss
                - 'total': Total weighted loss
        """
        derivatives, residual = self.compute_derivatives_and_residual(network, t, x)

        mse_residual = torch.mean(residual**2)
        physics_loss = self.weight * mse_residual

        result = {
            **derivatives,
            "residual": residual,
            "mse_residual": mse_residual,
            "physics_loss": physics_loss,
            "total": physics_loss,
        }

        return result
