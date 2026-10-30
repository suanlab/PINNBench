"""1D Burgers equation PDE definition.

This module implements the 1D viscous Burgers equation:
    ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²

where ν is the kinematic viscosity coefficient.

The Burgers equation is a fundamental nonlinear PDE that combines
convection and diffusion, making it an excellent test case for PINNs
and causal training methods due to its tendency to develop shocks.
"""


import torch
from torch import Tensor

from .base import PDEProblem


class Burgers1D(PDEProblem):
    """1D Viscous Burgers equation with Dirichlet boundary conditions.

    The Burgers equation models nonlinear convection-diffusion and is given by:
        ∂u/∂t + u * ∂u/∂x = ν * ∂²u/∂x²

    This implementation uses the domain x ∈ [0, L], t ∈ [0, T] with:
        - Boundary conditions: u(0, t) = u(L, t) = 0
        - Initial condition: u(x, 0) = -sin(πx/L)

    Note:
        The Burgers equation is nonlinear due to the u * ∂u/∂x term,
        making it significantly harder for PINNs than linear PDEs.
        No analytical solution is available for this configuration.

    Attributes:
        viscosity: Kinematic viscosity coefficient ν.
        length: Spatial domain length L.
        t_max: Maximum time T.
    """

    def __init__(
        self,
        viscosity: float = 0.01,
        length: float = 1.0,
        t_max: float = 1.0,
    ) -> None:
        """Initialize the 1D Burgers equation.

        Args:
            viscosity: Kinematic viscosity coefficient. Default: 0.01
            length: Spatial domain length L. Default: 1.0
            t_max: Maximum simulation time. Default: 1.0
        """
        self.viscosity = viscosity
        self.length = length
        self.t_max = t_max

    @property
    def domain_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the domain bounds for time and space.

        Returns:
            Tuple of ((t_min, t_max), (x_min, x_max)).
        """
        return ((0.0, self.t_max), (0.0, self.length))

    def residual(self, u: Tensor, t: Tensor, x: Tensor) -> Tensor:
        """Compute the Burgers equation residual: ∂u/∂t + u * ∂u/∂x - ν * ∂²u/∂x².

        Args:
            u: Solution field of shape (batch_size,) or (batch_size, 1).
            t: Time coordinates of shape (batch_size,) or (batch_size, 1).
            x: Spatial coordinates of shape (batch_size,) or (batch_size, 1).

        Returns:
            Residual tensor of shape (batch_size,).
        """
        # Don't reshape t and x - it breaks the computational graph
        # Just ensure u is 1D
        u_flat = u.reshape(-1)

        # Compute ∂u/∂t using original t
        ones = torch.ones_like(u_flat)
        u_t = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        if u_t is None:
            u_t = torch.zeros_like(u_flat)
        u_t = u_t.reshape(-1)

        # Compute ∂u/∂x using original x
        u_x = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        if u_x is None:
            u_x = torch.zeros_like(u_flat)
        u_x = u_x.reshape(-1)

        # Compute ∂²u/∂x²
        ones_x = torch.ones_like(u_x)
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=ones_x,
            create_graph=True,
            retain_graph=True,
        )[0]
        if u_xx is None:
            u_xx = torch.zeros_like(u_flat)
        u_xx = u_xx.reshape(-1)

        # Burgers equation residual: u_t + u * u_x - ν * u_xx = 0
        residual = u_t + u_flat * u_x - self.viscosity * u_xx

        return residual

    def boundary_condition(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute Dirichlet boundary condition: u(0, t) = u(L, t) = 0.

        Args:
            t: Time coordinates at the boundary.
            x: Spatial coordinates at the boundary (0 or L).

        Returns:
            Zero tensor with same batch shape as input.
        """
        batch_size = t.reshape(-1).shape[0]
        return torch.zeros(batch_size, dtype=t.dtype, device=t.device)

    def initial_condition(self, x: Tensor) -> Tensor:
        """Compute initial condition: u(x, 0) = -sin(πx/L).

        Args:
            x: Spatial coordinates.

        Returns:
            Initial condition values -sin(πx/L).
        """
        x = x.reshape(-1)
        return -torch.sin(torch.pi * x / self.length)

    def exact_solution(self, t: Tensor, x: Tensor) -> Tensor | None:
        """Return None as no analytical solution is available for Burgers equation.

        Args:
            t: Time coordinates.
            x: Spatial coordinates.

        Returns:
            None - no analytical solution available.
        """
        return None
