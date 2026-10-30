"""1D Fisher-KPP Reaction-Diffusion equation PDE definition.

This module implements the 1D Fisher-KPP equation:
    du/dt = D * d^2u/dx^2 + r * u * (1 - u)

where D is the diffusion coefficient and r is the reaction rate.

The Fisher-KPP equation is a fundamental reaction-diffusion PDE that models
population dynamics and traveling wave phenomena. It combines diffusion with
logistic growth, making it a challenging test case for PINNs due to the
formation of sharp traveling fronts.
"""

import torch
from torch import Tensor

from .base import PDEProblem


class ReactionDiffusion1D(PDEProblem):
    """1D Fisher-KPP Reaction-Diffusion equation with Dirichlet BCs.

    The Fisher-KPP equation models reaction-diffusion dynamics and is given by:
        du/dt = D * d^2u/dx^2 + r * u * (1 - u)

    This implementation uses the domain x in [0, L], t in [0, T] with:
        - Boundary conditions: u(0, t) = 1, u(L, t) = 0
        - Initial condition: smooth step 0.5 * (1 - tanh(k * (x - L/4)))

    Note:
        No closed-form analytical solution is available for the Fisher-KPP
        equation with general initial conditions. When reaction_rate=0 the
        equation reduces to pure diffusion, but this class always returns
        None for exact_solution to keep behavior uniform.

    Attributes:
        diffusion: Diffusion coefficient D.
        reaction_rate: Reaction rate r.
        length: Spatial domain length L.
        t_max: Maximum time T.
    """

    def __init__(
        self,
        diffusion: float = 0.01,
        reaction_rate: float = 1.0,
        length: float = 1.0,
        t_max: float = 1.0,
    ) -> None:
        """Initialize the 1D Fisher-KPP Reaction-Diffusion equation.

        Args:
            diffusion: Diffusion coefficient D. Must be positive. Default: 0.01
            reaction_rate: Reaction rate r. Must be non-negative. Default: 1.0
            length: Spatial domain length L. Default: 1.0
            t_max: Maximum simulation time. Default: 1.0

        Raises:
            ValueError: If diffusion is not positive or reaction_rate is negative.
        """
        if diffusion <= 0:
            raise ValueError("diffusion must be positive")
        if reaction_rate < 0:
            raise ValueError("reaction_rate must be non-negative")
        self.diffusion = diffusion
        self.reaction_rate = reaction_rate
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
        """Compute the Fisher-KPP residual: du/dt - D*d^2u/dx^2 - r*u*(1-u).

        Args:
            u: Solution field of shape (batch_size,) or (batch_size, 1).
            t: Time coordinates of shape (batch_size,) or (batch_size, 1).
            x: Spatial coordinates of shape (batch_size,) or (batch_size, 1).

        Returns:
            Residual tensor of shape (batch_size,).
        """
        u_flat = u.reshape(-1)

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

        # Fisher-KPP residual: u_t - D * u_xx - r * u * (1 - u) = 0
        residual = u_t - self.diffusion * u_xx - self.reaction_rate * u_flat * (1.0 - u_flat)

        return residual  # type: ignore[no-any-return]

    def boundary_condition(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute Dirichlet boundary conditions: u(0, t) = 1, u(L, t) = 0.

        Args:
            t: Time coordinates at the boundary.
            x: Spatial coordinates at the boundary (0 or L).

        Returns:
            Boundary values: 1 where x=0, 0 where x=L.
        """
        x_flat = x.reshape(-1)
        # u(0, t) = 1, u(L, t) = 0
        # Points near x=0 get value 1, points near x=L get value 0
        bc = torch.where(
            x_flat < self.length / 2.0,
            torch.ones_like(x_flat),
            torch.zeros_like(x_flat),
        )
        return bc

    def initial_condition(self, x: Tensor) -> Tensor:
        """Compute initial condition: smooth step 0.5*(1 - tanh(k*(x - L/4))).

        The initial condition approximates a step function that is 1 for
        x < L/4 and 0 for x >= L/4, with a smooth tanh transition.

        Args:
            x: Spatial coordinates.

        Returns:
            Initial condition values.
        """
        x = x.reshape(-1)
        k = 20.0 / self.length  # steepness of the transition
        return 0.5 * (1.0 - torch.tanh(k * (x - self.length / 4.0)))  # type: ignore[no-any-return]

    def exact_solution(self, t: Tensor, x: Tensor) -> Tensor | None:
        """Return None as no analytical solution is available for Fisher-KPP.

        Args:
            t: Time coordinates.
            x: Spatial coordinates.

        Returns:
            None - no analytical solution available.
        """
        return None
