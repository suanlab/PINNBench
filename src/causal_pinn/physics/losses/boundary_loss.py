"""Boundary condition and initial condition loss computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from causal_pinn.models.pinn.base import PINNBase


class BoundaryLoss:
    """Boundary condition loss for PINNs.

    Computes mean squared error between predicted and true boundary
    condition values. Supports both spatial boundary conditions (BC)
    and initial conditions (IC).

    Attributes:
        bc_weight: Weight for boundary condition loss.
        ic_weight: Weight for initial condition loss.
    """

    def __init__(
        self,
        bc_weight: float = 1.0,
        ic_weight: float = 1.0,
    ) -> None:
        """Initialize boundary loss.

        Args:
            bc_weight: Weight factor for boundary condition loss.
            ic_weight: Weight factor for initial condition loss.
        """
        self.bc_weight = bc_weight
        self.ic_weight = ic_weight

    def _prepare_inputs(
        self,
        t: Tensor | None,
        x: Tensor,
    ) -> Tensor:
        """Prepare inputs for network forward pass.

        Args:
            t: Time coordinates (can be None for IC), shape (N,) or (N, 1).
            x: Spatial coordinates, shape (N,) or (N, 1).

        Returns:
            Combined input tensor of shape (N, 2).
        """
        # Ensure 2D tensors
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        if t is None:
            batch_size = x.shape[0]
            t = torch.zeros(batch_size, 1, dtype=x.dtype, device=x.device)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)

        return torch.cat([t, x], dim=-1)

    def compute_bc_loss(
        self,
        network: PINNBase,
        t_bc: Tensor,
        x_bc: Tensor,
        u_bc: Tensor,
    ) -> Tensor:
        """Compute boundary condition loss.

        Args:
            network: Neural network approximating the solution.
            t_bc: Time coordinates at boundary, shape (N_bc,) or (N_bc, 1).
            x_bc: Spatial coordinates at boundary, shape (N_bc,) or (N_bc, 1).
            u_bc: True boundary values, shape (N_bc,) or (N_bc, 1).

        Returns:
            Scalar MSE loss for boundary conditions.
        """
        inputs = self._prepare_inputs(t_bc, x_bc)
        u_pred = network(inputs)

        # Ensure shapes match
        if u_pred.dim() == 2 and u_pred.shape[-1] == 1 and u_bc.dim() == 1:
            u_bc = u_bc.unsqueeze(-1)

        return torch.mean((u_pred - u_bc) ** 2)

    def compute_ic_loss(
        self,
        network: PINNBase,
        x_ic: Tensor,
        u_ic: Tensor,
    ) -> Tensor:
        """Compute initial condition loss at t=0.

        Args:
            network: Neural network approximating the solution.
            x_ic: Spatial coordinates at initial time, shape (N_ic,) or (N_ic, 1).
            u_ic: True initial values, shape (N_ic,) or (N_ic, 1).

        Returns:
            Scalar MSE loss for initial conditions.
        """
        inputs = self._prepare_inputs(None, x_ic)
        u_pred = network(inputs)

        # Ensure shapes match
        if u_pred.dim() == 2 and u_pred.shape[-1] == 1 and u_ic.dim() == 1:
            u_ic = u_ic.unsqueeze(-1)

        return torch.mean((u_pred - u_ic) ** 2)

    def __call__(
        self,
        network: PINNBase,
        t_bc: Tensor | None = None,
        x_bc: Tensor | None = None,
        u_bc: Tensor | None = None,
        x_ic: Tensor | None = None,
        u_ic: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute boundary and initial condition losses.

        Args:
            network: Neural network approximating the solution.
            t_bc: Time coordinates at boundary (optional if only computing IC).
            x_bc: Spatial coordinates at boundary (optional if only computing IC).
            u_bc: True boundary values (optional if only computing IC).
            x_ic: Spatial coordinates at initial time (optional if only computing BC).
            u_ic: True initial values (optional if only computing BC).

        Returns:
            Dictionary containing:
                - 'bc_loss': Weighted BC loss (if BC provided)
                - 'ic_loss': Weighted IC loss (if IC provided)
                - 'mse_bc': Raw BC MSE (if BC provided)
                - 'mse_ic': Raw IC MSE (if IC provided)
                - 'total': Combined weighted loss
        """
        result: dict[str, Tensor] = {}
        total = torch.tensor(0.0, dtype=torch.float32)

        # Compute BC loss if data provided
        if t_bc is not None and x_bc is not None and u_bc is not None:
            mse_bc = self.compute_bc_loss(network, t_bc, x_bc, u_bc)
            weighted_bc = self.bc_weight * mse_bc
            result["mse_bc"] = mse_bc
            result["bc_loss"] = weighted_bc
            total = total + weighted_bc

        # Compute IC loss if data provided
        if x_ic is not None and u_ic is not None:
            mse_ic = self.compute_ic_loss(network, x_ic, u_ic)
            weighted_ic = self.ic_weight * mse_ic
            result["mse_ic"] = mse_ic
            result["ic_loss"] = weighted_ic
            total = total + weighted_ic

        result["total"] = total

        return result


class DirichletBC(BoundaryLoss):
    """Dirichlet boundary condition loss.

    Specializes BoundaryLoss for Dirichlet BCs where the solution
    value is specified at the boundary: u(t, x_boundary) = g(t, x).
    """

    def __init__(self, weight: float = 1.0) -> None:
        """Initialize Dirichlet BC loss.

        Args:
            weight: Weight factor for the Dirichlet BC loss.
        """
        super().__init__(bc_weight=weight, ic_weight=weight)


class NeumannBC:
    """Neumann boundary condition loss.

    For Neumann BCs, the normal derivative is specified:
        ∂u/∂n = h(t, x) at the boundary.

    This requires computing spatial derivatives at boundary points.
    """

    def __init__(
        self,
        weight: float = 1.0,
        normal_direction: str = "x",
    ) -> None:
        """Initialize Neumann BC loss.

        Args:
            weight: Weight factor for the Neumann BC loss.
            normal_direction: Direction of the normal ('x' for 1D spatial).
        """
        self.weight = weight
        self.normal_direction = normal_direction

    def compute_normal_derivative(
        self,
        network: PINNBase,
        t_bc: Tensor,
        x_bc: Tensor,
    ) -> Tensor:
        """Compute normal derivative at boundary points.

        Args:
            network: Neural network.
            t_bc: Time coordinates at boundary.
            x_bc: Spatial coordinates at boundary.

        Returns:
            Normal derivative tensor.
        """
        # Ensure gradients are tracked
        if not t_bc.requires_grad:
            t_bc = t_bc.clone().requires_grad_(True)
        if not x_bc.requires_grad:
            x_bc = x_bc.clone().requires_grad_(True)

        # Prepare inputs
        if t_bc.dim() == 1:
            t_bc = t_bc.unsqueeze(-1)
        if x_bc.dim() == 1:
            x_bc = x_bc.unsqueeze(-1)

        inputs = torch.cat([t_bc, x_bc], dim=-1)
        u = network(inputs)

        # Compute gradient w.r.t inputs
        grad_u = torch.autograd.grad(
            u,
            inputs,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Extract spatial derivative (column 1 is x for inputs [t, x])
        return grad_u[:, 1:2]

    def __call__(
        self,
        network: PINNBase,
        t_bc: Tensor,
        x_bc: Tensor,
        du_dn_true: Tensor,
    ) -> dict[str, Tensor]:
        """Compute Neumann BC loss.

        Args:
            network: Neural network.
            t_bc: Time coordinates at boundary.
            x_bc: Spatial coordinates at boundary.
            du_dn_true: True normal derivative values.

        Returns:
            Dictionary with:
                - 'du_dn_pred': Predicted normal derivative
                - 'mse_neumann': Raw MSE
                - 'neumann_loss': Weighted Neumann BC loss
                - 'total': Total weighted loss
        """
        du_dn_pred = self.compute_normal_derivative(network, t_bc, x_bc)

        # Ensure shapes match
        if du_dn_pred.dim() == 2 and du_dn_pred.shape[-1] == 1 and du_dn_true.dim() == 1:
            du_dn_true = du_dn_true.unsqueeze(-1)

        mse = torch.mean((du_dn_pred - du_dn_true) ** 2)
        weighted_loss = self.weight * mse

        return {
            "du_dn_pred": du_dn_pred,
            "mse_neumann": mse,
            "neumann_loss": weighted_loss,
            "total": weighted_loss,
        }
