"""2D Wave equation PDE definition.

This module implements the 2D wave equation:
    ∂²u/∂t² = c²∇²u

where c is the wave propagation speed and ∇² is the Laplacian operator.
"""


import torch
from torch import Tensor

from .base import PDEProblem


class Wave2D(PDEProblem):
    """2D Wave equation with Dirichlet boundary conditions.

    The wave equation models wave propagation phenomena:
        ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)

    This implementation uses the exact standing wave solution:
        u(x, y, t) = sin(πx/L_x)sin(πy/L_y)cos(ωt)

    where ω = cπ√(1/L_x² + 1/L_y²) is the angular frequency.

    Domain: (x, y) ∈ [0, L_x] × [0, L_y], t ∈ [0, T]

    Boundary conditions:
        u(0, y, t) = u(L_x, y, t) = 0
        u(x, 0, t) = u(x, L_y, t) = 0

    Initial conditions:
        u(x, y, 0) = sin(πx/L_x)sin(πy/L_y)
        ∂u/∂t(x, y, 0) = 0

    Attributes:
        wave_speed: Wave propagation speed c.
        length_x: Spatial domain length in x direction L_x.
        length_y: Spatial domain length in y direction L_y.
        t_max: Maximum time T.
    """

    def __init__(
        self,
        wave_speed: float = 1.0,
        length_x: float = 1.0,
        length_y: float = 1.0,
        t_max: float = 1.0,
    ) -> None:
        """Initialize the 2D wave equation.

        Args:
            wave_speed: Wave propagation speed c. Default: 1.0
            length_x: Spatial domain length in x. Default: 1.0
            length_y: Spatial domain length in y. Default: 1.0
            t_max: Maximum simulation time. Default: 1.0
        """
        self.wave_speed = wave_speed
        self.length_x = length_x
        self.length_y = length_y
        self.t_max = t_max

    @property
    def domain_bounds(self) -> tuple[tuple[float, float], tuple[float, float, float, float]]:
        """Return the domain bounds for time and space.

        Returns:
            Tuple of ((t_min, t_max), (x_min, x_max, y_min, y_max)).
        """
        return ((0.0, self.t_max), (0.0, self.length_x, 0.0, self.length_y))

    def residual(self, u: Tensor, t: Tensor, x: Tensor) -> Tensor:
        """Compute the wave equation residual: ∂²u/∂t² - c²∇²u.

        Args:
            u: Solution field of shape (batch_size,) or (batch_size, 1).
            t: Time coordinates of shape (batch_size,) or (batch_size, 1).
            x: Spatial coordinates of shape (batch_size, 2) where x[:, 0] is
               the x-coordinate and x[:, 1] is the y-coordinate.

        Returns:
            Residual tensor of shape (batch_size,).
        """
        # Get batch size
        batch_size = t.reshape(-1).shape[0]
        u_flat = u.reshape(-1)

        # Compute first temporal derivative using original t
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

        # Compute second temporal derivative
        u_tt = torch.autograd.grad(
            outputs=u_t,
            inputs=t,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        if u_tt is None:
            u_tt = torch.zeros_like(u_flat)
        u_tt = u_tt.reshape(-1)

        # Compute first spatial derivatives using original x
        # Gradient will have shape (batch, 2)
        grad_u = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        if grad_u is None:
            grad_u = torch.zeros(batch_size, 2, dtype=u.dtype, device=u.device)
        grad_u = grad_u.reshape(batch_size, 2)

        u_x = grad_u[:, 0]  # ∂u/∂x
        u_y = grad_u[:, 1]  # ∂u/∂y

        # Compute second spatial derivatives (Laplacian)
        # For ∂²u/∂x², we need gradient of u_x with respect to x
        grad_ux = torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        if grad_ux is None:
            grad_ux = torch.zeros(batch_size, 2, dtype=u.dtype, device=u.device)
        grad_ux = grad_ux.reshape(batch_size, 2)
        u_xx = grad_ux[:, 0]  # ∂²u/∂x²

        # For ∂²u/∂y², we need gradient of u_y with respect to x
        grad_uy = torch.autograd.grad(
            outputs=u_y,
            inputs=x,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        if grad_uy is None:
            grad_uy = torch.zeros(batch_size, 2, dtype=u.dtype, device=u.device)
        grad_uy = grad_uy.reshape(batch_size, 2)
        u_yy = grad_uy[:, 1]  # ∂²u/∂y²

        # Wave equation residual: u_tt - c²(u_xx + u_yy) = 0
        laplacian = u_xx + u_yy
        residual = u_tt - self.wave_speed**2 * laplacian

        return residual

    def boundary_condition(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute Dirichlet boundary condition: u = 0 on all boundaries.

        Args:
            t: Time coordinates at the boundary.
            x: Spatial coordinates at the boundary of shape (batch_size, 2).

        Returns:
            Zero tensor with same batch shape as input.
        """
        batch_size = t.reshape(-1).shape[0]
        return torch.zeros(batch_size, dtype=t.dtype, device=t.device)

    def initial_condition(self, x: Tensor) -> Tensor:
        """Compute initial condition: u(x, y, 0) = sin(πx/L_x)sin(πy/L_y).

        Args:
            x: Spatial coordinates of shape (batch_size, 2).

        Returns:
            Initial condition values.
        """
        x = x.reshape(-1, 2)
        x_coord = x[:, 0]
        y_coord = x[:, 1]

        return (
            torch.sin(torch.pi * x_coord / self.length_x)
            * torch.sin(torch.pi * y_coord / self.length_y)
        )

    def exact_solution(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute exact analytical solution.

        The exact standing wave solution is:
            u(x, y, t) = sin(πx/L_x)sin(πy/L_y)cos(ωt)

        where ω = cπ√(1/L_x² + 1/L_y²) is the angular frequency.

        Args:
            t: Time coordinates.
            x: Spatial coordinates of shape (batch_size, 2).

        Returns:
            Exact solution values.
        """
        t = t.reshape(-1)
        x = x.reshape(-1, 2)
        x_coord = x[:, 0]
        y_coord = x[:, 1]

        # Spatial component
        spatial = (
            torch.sin(torch.pi * x_coord / self.length_x)
            * torch.sin(torch.pi * y_coord / self.length_y)
        )

        # Angular frequency
        omega = self.wave_speed * torch.pi * torch.sqrt(
            torch.tensor(
                1.0 / self.length_x**2 + 1.0 / self.length_y**2,
                device=t.device,
                dtype=t.dtype,
            )
        )

        # Temporal component
        temporal = torch.cos(omega * t)

        return spatial * temporal
