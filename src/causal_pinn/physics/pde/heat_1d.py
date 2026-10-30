"""1D Heat equation PDE definition.

This module implements the 1D heat (diffusion) equation:
    ∂u/∂t = α * ∂²u/∂x²

where α is the thermal diffusivity coefficient.
"""


import torch
from torch import Tensor

from .base import PDEProblem


class Heat1D(PDEProblem):
    """1D Heat equation with Dirichlet boundary conditions.

    The heat equation models diffusion processes and is given by:
        ∂u/∂t = α * ∂²u/∂x²

    This implementation uses the exact solution:
        u(x, t) = sin(πx/L) * exp(-α(π/L)²t)

    for the domain x ∈ [0, L], t ∈ [0, T] with boundary conditions:
        u(0, t) = 0
        u(L, t) = 0

    and initial condition:
        u(x, 0) = sin(πx/L)

    Attributes:
        alpha: Thermal diffusivity coefficient.
        length: Spatial domain length L.
        t_max: Maximum time T.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        length: float = 1.0,
        t_max: float = 1.0,
    ) -> None:
        """Initialize the 1D heat equation.

        Args:
            alpha: Thermal diffusivity coefficient. Default: 1.0
            length: Spatial domain length L. Default: 1.0
            t_max: Maximum simulation time. Default: 1.0
        """
        self.alpha = alpha
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
        """Compute the heat equation residual: ∂u/∂t - α * ∂²u/∂x².

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
        u_xx = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        if u_xx is None:
            u_xx = torch.zeros_like(u_flat)
        u_xx = u_xx.reshape(-1)

        # Heat equation residual: u_t - α * u_xx = 0
        residual = u_t - self.alpha * u_xx

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
        """Compute initial condition: u(x, 0) = sin(πx/L).

        Args:
            x: Spatial coordinates.

        Returns:
            Initial condition values sin(πx/L).
        """
        x = x.reshape(-1)
        return torch.sin(torch.pi * x / self.length)

    def exact_solution(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute exact analytical solution.

        The exact solution is:
            u(x, t) = sin(πx/L) * exp(-α(π/L)²t)

        Args:
            t: Time coordinates.
            x: Spatial coordinates.

        Returns:
            Exact solution values.
        """
        t = t.reshape(-1)
        x = x.reshape(-1)

        # Spatial component
        spatial = torch.sin(torch.pi * x / self.length)

        # Temporal decay
        decay_rate = self.alpha * (torch.pi / self.length) ** 2
        temporal = torch.exp(-decay_rate * t)

        return spatial * temporal
