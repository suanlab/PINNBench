"""PINN Solver that combines network and PDE problem for training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import Tensor

from causal_pinn.models.pinn.base import PINNBase
from causal_pinn.utils.gradients import gradient

if TYPE_CHECKING:
    pass


@runtime_checkable
class PDEProblemProtocol(Protocol):
    """Protocol defining the interface expected from a PDE problem.

    This allows duck typing so PINNSolver doesn't need the actual PDEProblem import.
    """

    def residual(self, u: Tensor, t: Tensor, x: Tensor) -> Tensor:
        """Compute PDE residual.

        Args:
            u: Solution values.
            t: Time coordinates.
            x: Spatial coordinates.

        Returns:
            PDE residual tensor.
        """
        ...

    def boundary_condition(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute boundary condition values.

        Args:
            t: Time coordinates at boundary.
            x: Spatial coordinates at boundary.

        Returns:
            Boundary condition values.
        """
        ...

    def initial_condition(self, x: Tensor) -> Tensor:
        """Compute initial condition values.

        Args:
            x: Spatial coordinates at t=0.

        Returns:
            Initial condition values.
        """
        ...


class PINNSolver:
    """PINN Solver combining neural network with PDE constraints.

    This class wraps a PINNBase network and a PDE problem to provide
    methods for computing various loss components needed for training.

    Attributes:
        network: The neural network (PINNBase) to train.
        pde_problem: The PDE problem defining physics constraints.
    """

    def __init__(
        self,
        network: PINNBase,
        pde_problem: PDEProblemProtocol,
    ) -> None:
        """Initialize the PINN solver.

        Args:
            network: PINNBase neural network for approximating the solution.
            pde_problem: PDE problem defining the physics and boundary conditions.
        """
        self.network = network
        self.pde_problem = pde_problem

    def _prepare_inputs(self, t: Tensor, x: Tensor) -> Tensor:
        """Prepare inputs for network forward pass.

        Args:
            t: Time coordinates of shape (N,) or (N, 1).
            x: Spatial coordinates of shape (N,) or (N, 1).

        Returns:
            Combined input tensor of shape (N, 2).
        """
        # Ensure 2D tensors
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        return torch.cat([t, x], dim=-1)

    def compute_physics_loss(
        self,
        t: Tensor,
        x: Tensor,
    ) -> Tensor:
        """Compute physics-based loss from PDE residual.

        The network predicts u(t, x), then autograd computes derivatives
        needed for the PDE residual. The loss is the mean squared residual.

        Args:
            t: Time coordinates with requires_grad=True, shape (N,) or (N, 1).
            x: Spatial coordinates with requires_grad=True, shape (N,) or (N, 1).

        Returns:
            Scalar tensor containing mean squared PDE residual.
        """
        # Ensure gradients are tracked
        if not t.requires_grad:
            t = t.clone().requires_grad_(True)
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)

        # Forward pass to get u
        inputs = self._prepare_inputs(t, x)
        u = self.network(inputs)

        # Squeeze if needed for scalar output
        if u.shape[-1] == 1:
            u = u.squeeze(-1)

        # Compute PDE residual
        residual = self.pde_problem.residual(u, t, x)

        # Return mean squared residual
        return torch.mean(residual**2)

    def compute_bc_loss(
        self,
        t_bc: Tensor,
        x_bc: Tensor,
        u_bc: Tensor,
    ) -> Tensor:
        """Compute boundary condition loss.

        Args:
            t_bc: Time coordinates at boundary, shape (N_bc,) or (N_bc, 1).
            x_bc: Spatial coordinates at boundary, shape (N_bc,) or (N_bc, 1).
            u_bc: True boundary condition values, shape (N_bc,) or (N_bc, 1).

        Returns:
            Scalar tensor containing mean squared BC error.
        """
        inputs = self._prepare_inputs(t_bc, x_bc)
        u_pred = self.network(inputs)

        # Ensure shapes match. Without this, a 1-D u_bc broadcasts against the
        # (N, 1) prediction to an (N, N) difference and the loss silently becomes
        # a cross-product rather than an elementwise one. Homogeneous Dirichlet
        # data hides it, since every entry is zero either way, but any nonzero
        # boundary value gives a wrong number.
        if u_pred.dim() == 2 and u_pred.shape[-1] == 1 and u_bc.dim() == 1:
            u_bc = u_bc.unsqueeze(-1)

        return torch.mean((u_pred - u_bc) ** 2)

    def compute_ic_loss(
        self,
        x_ic: Tensor,
        u_ic: Tensor,
    ) -> Tensor:
        """Compute initial condition loss at t=0.

        Args:
            x_ic: Spatial coordinates at initial time, shape (N_ic,) or (N_ic, 1).
            u_ic: True initial condition values, shape (N_ic,) or (N_ic, 1).

        Returns:
            Scalar tensor containing mean squared IC error.
        """
        # Create zero time tensor for initial condition
        batch_size = x_ic.shape[0] if x_ic.dim() > 1 else x_ic.shape[0]
        t_zero = torch.zeros(batch_size, dtype=x_ic.dtype, device=x_ic.device)

        inputs = self._prepare_inputs(t_zero, x_ic)
        u_pred = self.network(inputs)

        # Ensure shapes match
        if u_pred.shape != u_ic.shape and u_ic.dim() == 1:
            u_ic = u_ic.unsqueeze(-1)

        return torch.mean((u_pred - u_ic) ** 2)

    def predict(
        self,
        t: Tensor,
        x: Tensor,
    ) -> Tensor:
        """Predict solution values at given coordinates.

        Args:
            t: Time coordinates, shape (N,) or (N, 1).
            x: Spatial coordinates, shape (N,) or (N, 1).

        Returns:
            Predicted solution u(t, x), shape (N, output_dim).
        """
        inputs = self._prepare_inputs(t, x)
        return self.network(inputs)  # type: ignore[no-any-return]

    def compute_derivatives(
        self,
        t: Tensor,
        x: Tensor,
        derivatives: str = "u_t,u_x,u_xx,u_tt",
    ) -> dict[str, Tensor]:
        """Compute derivatives of the solution w.r.t. inputs.

        Args:
            t: Time coordinates with requires_grad=True.
            x: Spatial coordinates with requires_grad=True.
            derivatives: Comma-separated list of derivatives to compute.
                        Options: u, u_t, u_x, u_xx, u_tt, u_tx.

        Returns:
            Dictionary mapping derivative names to computed values.
        """
        # Ensure gradients are tracked
        if not t.requires_grad:
            t = t.clone().requires_grad_(True)
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)

        inputs = self._prepare_inputs(t, x)
        u = self.network(inputs)

        result: dict[str, Tensor] = {}

        deriv_list = [d.strip() for d in derivatives.split(",")]

        if "u" in deriv_list:
            result["u"] = u

        # First-order derivatives
        if "u_t" in deriv_list or "u_tt" in deriv_list or "u_tx" in deriv_list:
            grad_u = gradient(u, inputs, order=1, create_graph=True)
            u_t = grad_u[:, 0:1]
            u_x = grad_u[:, 1:2]
            if "u_t" in deriv_list:
                result["u_t"] = u_t
            if "u_x" in deriv_list:
                result["u_x"] = u_x

            # Second-order derivatives
            if "u_tt" in deriv_list:
                u_tt = gradient(u_t, inputs, order=1, create_graph=True)[:, 0:1]
                result["u_tt"] = u_tt

            if "u_xx" in deriv_list:
                if "u_x" not in result:
                    grad_u = gradient(u, inputs, order=1, create_graph=True)
                    u_x = grad_u[:, 1:2]
                u_xx = gradient(u_x, inputs, order=1, create_graph=True)[:, 1:2]
                result["u_xx"] = u_xx

            if "u_tx" in deriv_list:
                u_tx = gradient(u_t, inputs, order=1, create_graph=True)[:, 1:2]
                result["u_tx"] = u_tx

        elif "u_x" in deriv_list or "u_xx" in deriv_list:
            grad_u = gradient(u, inputs, order=1, create_graph=True)
            u_x = grad_u[:, 1:2]
            if "u_x" in deriv_list:
                result["u_x"] = u_x

            if "u_xx" in deriv_list:
                u_xx = gradient(u_x, inputs, order=1, create_graph=True)[:, 1:2]
                result["u_xx"] = u_xx

        return result

    def train_mode(self) -> None:
        """Set network to training mode."""
        self.network.train()

    def eval_mode(self) -> None:
        """Set network to evaluation mode."""
        self.network.eval()

    def parameters(self) -> torch.nn.Parameter:
        """Get network parameters for optimizer.

        Returns:
            Iterator over network parameters.
        """
        return self.network.parameters()
