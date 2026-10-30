"""1D Advection-Diffusion equation PDE definition.

This module implements the 1D advection-diffusion equation:
    ∂u/∂t + c * ∂u/∂x = α * ∂²u/∂x²

where c is the advection velocity and α is the diffusion coefficient.
"""


import torch
from torch import Tensor

from .base import PDEProblem


class AdvectionDiffusion1D(PDEProblem):
    """1D Advection-Diffusion equation.

    The advection-diffusion equation combines transport and diffusion:
        ∂u/∂t + c * ∂u/∂x = α * ∂²u/∂x²

    This implementation provides an exact solution for the pure diffusion case
    (c=0) which reduces to the heat equation, and a traveling wave solution
    for the general case.

    For c ≠ 0, the exact solution on domain [0, L] with periodic BCs is:
        u(x, t) = sin(2π(x - ct)/L) * exp(-α(2π/L)²t)

    with boundary conditions:
        u(0, t) = u(L, t)  (periodic)

    and initial condition:
        u(x, 0) = sin(2πx/L)

    Attributes:
        velocity: Advection velocity c.
        alpha: Diffusion coefficient.
        length: Spatial domain length L.
        t_max: Maximum time T.
    """

    def __init__(
        self,
        velocity: float = 1.0,
        alpha: float = 0.1,
        length: float = 1.0,
        t_max: float = 1.0,
    ) -> None:
        """Initialize the 1D advection-diffusion equation.

        Args:
            velocity: Advection velocity c. Default: 1.0
            alpha: Diffusion coefficient. Default: 0.1
            length: Spatial domain length L. Default: 1.0
            t_max: Maximum simulation time. Default: 1.0
        """
        self.velocity = velocity
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
        """Compute the advection-diffusion residual: ∂u/∂t + c∂u/∂x - α∂²u/∂x².

        Args:
            u: Solution field of shape (batch_size,) or (batch_size, 1).
            t: Time coordinates of shape (batch_size,) or (batch_size, 1).
            x: Spatial coordinates of shape (batch_size,) or (batch_size, 1).

        Returns:
            Residual tensor of shape (batch_size,).
        """
        # Don't reshape t and x - it breaks the computational graph
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

        # Advection-diffusion residual: u_t + c*u_x - α*u_xx = 0
        residual = u_t + self.velocity * u_x - self.alpha * u_xx

        return residual

    def boundary_condition(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute periodic boundary condition: u(0, t) = u(L, t).

        For the exact solution, this returns the value at the boundary
        which automatically satisfies periodicity.

        Args:
            t: Time coordinates at the boundary.
            x: Spatial coordinates at the boundary (0 or L).

        Returns:
            Boundary condition values from exact solution.
        """
        return self.exact_solution(t, x)

    def initial_condition(self, x: Tensor) -> Tensor:
        """Compute initial condition: u(x, 0) = sin(2πx/L).

        Args:
            x: Spatial coordinates.

        Returns:
            Initial condition values sin(2πx/L).
        """
        x = x.reshape(-1)
        return torch.sin(2.0 * torch.pi * x / self.length)

    def exact_solution(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute exact analytical solution.

        The exact solution is a traveling wave with diffusion:
            u(x, t) = sin(2π(x - ct)/L) * exp(-α(2π/L)²t)

        This satisfies:
        - Advection at velocity c
        - Diffusion with coefficient α
        - Periodic boundary conditions

        Args:
            t: Time coordinates.
            x: Spatial coordinates.

        Returns:
            Exact solution values.
        """
        t = t.reshape(-1)
        x = x.reshape(-1)

        # Wave number
        k = 2.0 * torch.pi / self.length

        # Traveling wave with diffusion
        spatial = torch.sin(k * (x - self.velocity * t))

        # Diffusive decay
        decay_rate = self.alpha * k**2
        temporal = torch.exp(-decay_rate * t)

        return spatial * temporal
