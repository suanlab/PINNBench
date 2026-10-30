"""3D Heat equation PDE definition.

This module implements the 3D heat (diffusion) equation:
    ∂u/∂t = α * (∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)

It exists to give the benchmark a d=3 configuration.  The suite is otherwise
1D with two 2D cases, so any claim about how routing or training policies
behave in higher dimensions previously had no evidence behind it.  The
separable sine mode keeps an analytical solution available at d=3, which is
what makes relative-L2 scoring (and therefore regret) well defined here.
"""

import torch
from torch import Tensor

from .base import PDEProblem


class Heat3D(PDEProblem):
    """3D Heat equation with homogeneous Dirichlet boundary conditions.

    The governing equation is
        ∂u/∂t = α ∇²u

    on the box x ∈ [0, Lx], y ∈ [0, Ly], z ∈ [0, Lz], t ∈ [0, T], with u = 0
    on every face.  Taking the fundamental sine mode as the initial condition,
        u(x, y, z, 0) = sin(πx/Lx) sin(πy/Ly) sin(πz/Lz),
    the solution stays separable for all time:
        u = sin(πx/Lx) sin(πy/Ly) sin(πz/Lz) exp(-α π² (Lx⁻² + Ly⁻² + Lz⁻²) t).

    Spatial coordinates are packed as a (batch, 3) tensor, matching the
    convention used by the 2D problems in this package.

    Attributes:
        alpha: Thermal diffusivity coefficient.
        length_x: Domain extent in x.
        length_y: Domain extent in y.
        length_z: Domain extent in z.
        t_max: Maximum time T.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        length_x: float = 1.0,
        length_y: float = 1.0,
        length_z: float = 1.0,
        t_max: float = 1.0,
    ) -> None:
        """Initialize the 3D heat equation.

        Args:
            alpha: Thermal diffusivity coefficient. Default: 1.0
            length_x: Domain extent in x. Default: 1.0
            length_y: Domain extent in y. Default: 1.0
            length_z: Domain extent in z. Default: 1.0
            t_max: Maximum simulation time. Default: 1.0
        """
        self.alpha = alpha
        self.length_x = length_x
        self.length_y = length_y
        self.length_z = length_z
        self.t_max = t_max

    @property
    def domain_bounds(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float, float, float, float, float]]:
        """Return the domain bounds for time and space.

        Returns:
            Tuple of ((t_min, t_max), (x_min, x_max, y_min, y_max, z_min, z_max)).
        """
        return (
            (0.0, self.t_max),
            (0.0, self.length_x, 0.0, self.length_y, 0.0, self.length_z),
        )

    def _decay_rate(self) -> float:
        """Exponential decay rate of the fundamental mode."""
        return self.alpha * (torch.pi**2) * (
            1.0 / self.length_x**2 + 1.0 / self.length_y**2 + 1.0 / self.length_z**2
        )

    def residual(self, u: Tensor, t: Tensor, x: Tensor) -> Tensor:
        """Compute the heat equation residual: ∂u/∂t - α ∇²u.

        Args:
            u: Solution field of shape (batch_size,) or (batch_size, 1).
            t: Time coordinates of shape (batch_size,) or (batch_size, 1).
            x: Spatial coordinates of shape (batch_size, 3), columns (x, y, z).

        Returns:
            Residual tensor of shape (batch_size,).
        """
        batch_size = t.reshape(-1).shape[0]
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

        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        if grad_u is None:
            grad_u = torch.zeros(batch_size, 3, dtype=u.dtype, device=u.device)
        grad_u = grad_u.reshape(batch_size, 3)

        laplacian = torch.zeros_like(u_flat)
        for axis in range(3):
            second = torch.autograd.grad(
                outputs=grad_u[:, axis],
                inputs=x,
                grad_outputs=ones,
                create_graph=True,
                retain_graph=True,
            )[0]
            if second is None:
                continue
            laplacian = laplacian + second.reshape(batch_size, 3)[:, axis]

        return u_t - self.alpha * laplacian

    def boundary_condition(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute the homogeneous Dirichlet condition u = 0 on every face.

        Args:
            t: Time coordinates at the boundary.
            x: Spatial coordinates at the boundary.

        Returns:
            Zero tensor with batch shape matching the input.
        """
        batch_size = t.reshape(-1).shape[0]
        return torch.zeros(batch_size, dtype=t.dtype, device=t.device)

    def initial_condition(self, x: Tensor) -> Tensor:
        """Compute u(x, y, z, 0) = sin(πx/Lx) sin(πy/Ly) sin(πz/Lz).

        Args:
            x: Spatial coordinates of shape (batch_size, 3).

        Returns:
            Initial condition values of shape (batch_size,).
        """
        x = x.reshape(-1, 3)
        return (
            torch.sin(torch.pi * x[:, 0] / self.length_x)
            * torch.sin(torch.pi * x[:, 1] / self.length_y)
            * torch.sin(torch.pi * x[:, 2] / self.length_z)
        )

    def exact_solution(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute the exact separable solution.

        Args:
            t: Time coordinates of shape (batch_size,) or (batch_size, 1).
            x: Spatial coordinates of shape (batch_size, 3).

        Returns:
            Exact solution values of shape (batch_size,).
        """
        t_flat = t.reshape(-1)
        spatial = self.initial_condition(x)
        return spatial * torch.exp(-self._decay_rate() * t_flat)
