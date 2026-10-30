"""2D Navier-Stokes PDE definition via Taylor-Green vorticity transport.

This module implements the 2D Navier-Stokes equations in vorticity form
using the Taylor-Green vortex, which has an exact analytical solution.

The vorticity transport equation is:
    omega_t + u * omega_x + v * omega_y = nu * (omega_xx + omega_yy)

where the velocity field (u, v) is prescribed from the exact Taylor-Green
solution, making this a linear advection-diffusion equation for omega.

Taylor-Green vortex exact solutions:
    u(x, y, t) = -cos(x) sin(y) exp(-2 nu t)
    v(x, y, t) =  sin(x) cos(y) exp(-2 nu t)
    omega(x, y, t) = -2 cos(x) cos(y) exp(-2 nu t)
"""

import math

import torch
from torch import Tensor

from .base import PDEProblem


class NavierStokes2D(PDEProblem):
    """2D Navier-Stokes via Taylor-Green vorticity transport.

    The PINN predicts the scalar vorticity field omega(t, x, y). The velocity
    field is prescribed from the known Taylor-Green exact solution, so the
    residual reduces to a linear advection-diffusion equation for omega.

    Domain: (x, y) in [0, L_x] x [0, L_y], t in [0, T]

    The default domain is [0, 2*pi] x [0, 2*pi] which is the natural period
    of the Taylor-Green vortex.

    Attributes:
        viscosity: Kinematic viscosity nu.
        length_x: Spatial domain length in x direction L_x.
        length_y: Spatial domain length in y direction L_y.
        t_max: Maximum time T.
    """

    def __init__(
        self,
        viscosity: float = 0.1,
        length_x: float = 2.0 * math.pi,
        length_y: float = 2.0 * math.pi,
        t_max: float = 1.0,
    ) -> None:
        """Initialize the 2D Navier-Stokes vorticity transport equation.

        Args:
            viscosity: Kinematic viscosity nu. Must be positive. Default: 0.1
            length_x: Spatial domain length in x. Default: 2*pi
            length_y: Spatial domain length in y. Default: 2*pi
            t_max: Maximum simulation time. Default: 1.0

        Raises:
            ValueError: If viscosity is not positive.
        """
        if viscosity <= 0.0:
            raise ValueError("viscosity must be positive")
        self.viscosity = viscosity
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
        """Compute the vorticity transport residual.

        Residual = omega_t + vel_x * omega_x + vel_y * omega_y
                   - nu * (omega_xx + omega_yy)

        where omega is the PINN-predicted vorticity (passed as ``u``), and
        vel_x, vel_y are the exact Taylor-Green velocity components.

        Args:
            u: Vorticity field of shape (batch_size,) or (batch_size, 1).
            t: Time coordinates of shape (batch_size,) or (batch_size, 1).
            x: Spatial coordinates of shape (batch_size, 2) where x[:, 0] is
               the x-coordinate and x[:, 1] is the y-coordinate.

        Returns:
            Residual tensor of shape (batch_size,).
        """
        batch_size = t.reshape(-1).shape[0]
        u_flat = u.reshape(-1)
        ones = torch.ones_like(u_flat)

        # --- Temporal derivative: omega_t ---
        omega_t = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        if omega_t is None:
            omega_t = torch.zeros_like(u_flat)
        omega_t = omega_t.reshape(-1)

        # --- Spatial first derivatives: omega_x, omega_y ---
        grad_omega = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        if grad_omega is None:
            grad_omega = torch.zeros(batch_size, 2, dtype=u.dtype, device=u.device)
        grad_omega = grad_omega.reshape(batch_size, 2)

        omega_x = grad_omega[:, 0]
        omega_y = grad_omega[:, 1]

        # --- Spatial second derivatives: omega_xx, omega_yy ---
        grad_omega_x = torch.autograd.grad(
            outputs=omega_x,
            inputs=x,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        if grad_omega_x is None:
            grad_omega_x = torch.zeros(batch_size, 2, dtype=u.dtype, device=u.device)
        grad_omega_x = grad_omega_x.reshape(batch_size, 2)
        omega_xx = grad_omega_x[:, 0]

        grad_omega_y = torch.autograd.grad(
            outputs=omega_y,
            inputs=x,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        if grad_omega_y is None:
            grad_omega_y = torch.zeros(batch_size, 2, dtype=u.dtype, device=u.device)
        grad_omega_y = grad_omega_y.reshape(batch_size, 2)
        omega_yy = grad_omega_y[:, 1]

        # --- Exact Taylor-Green velocity field ---
        t_flat = t.reshape(-1)
        x_coord = x.reshape(batch_size, 2)[:, 0]
        y_coord = x.reshape(batch_size, 2)[:, 1]

        decay = torch.exp(-2.0 * self.viscosity * t_flat)
        vel_x = -torch.cos(x_coord) * torch.sin(y_coord) * decay
        vel_y = torch.sin(x_coord) * torch.cos(y_coord) * decay

        # --- Residual: omega_t + vel_x * omega_x + vel_y * omega_y - nu * laplacian ---
        laplacian = omega_xx + omega_yy
        residual = omega_t + vel_x * omega_x + vel_y * omega_y - self.viscosity * laplacian

        return residual

    def exact_solution(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute the exact vorticity for the Taylor-Green vortex.

        omega(x, y, t) = -2 cos(x) cos(y) exp(-2 nu t)

        Args:
            t: Time coordinates.
            x: Spatial coordinates of shape (batch_size, 2).

        Returns:
            Exact vorticity values of shape (batch_size,).
        """
        t = t.reshape(-1)
        x = x.reshape(-1, 2)
        x_coord = x[:, 0]
        y_coord = x[:, 1]

        decay = torch.exp(-2.0 * self.viscosity * t)
        return -2.0 * torch.cos(x_coord) * torch.cos(y_coord) * decay

    def boundary_condition(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute boundary condition from exact solution (periodic-compatible).

        Since the Taylor-Green vortex is periodic with period 2*pi, the
        boundary condition is simply the exact solution evaluated at the
        boundary points.

        Args:
            t: Time coordinates at the boundary.
            x: Spatial coordinates at the boundary of shape (batch_size, 2).

        Returns:
            Boundary condition values of shape (batch_size,).
        """
        return self.exact_solution(t, x)

    def initial_condition(self, x: Tensor) -> Tensor:
        """Compute initial vorticity at t=0.

        omega(x, y, 0) = -2 cos(x) cos(y)

        Args:
            x: Spatial coordinates of shape (batch_size, 2).

        Returns:
            Initial condition values of shape (batch_size,).
        """
        x = x.reshape(-1, 2)
        x_coord = x[:, 0]
        y_coord = x[:, 1]

        return -2.0 * torch.cos(x_coord) * torch.cos(y_coord)

    def exact_velocity(self, t: Tensor, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the exact Taylor-Green velocity field.

        u(x, y, t) = -cos(x) sin(y) exp(-2 nu t)
        v(x, y, t) =  sin(x) cos(y) exp(-2 nu t)

        Args:
            t: Time coordinates.
            x: Spatial coordinates of shape (batch_size, 2).

        Returns:
            Tuple of (vel_x, vel_y) each of shape (batch_size,).
        """
        t = t.reshape(-1)
        x = x.reshape(-1, 2)
        x_coord = x[:, 0]
        y_coord = x[:, 1]

        decay = torch.exp(-2.0 * self.viscosity * t)
        vel_x = -torch.cos(x_coord) * torch.sin(y_coord) * decay
        vel_y = torch.sin(x_coord) * torch.cos(y_coord) * decay

        return vel_x, vel_y

    def exact_pressure(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute the exact Taylor-Green pressure field.

        p(x, y, t) = -(cos(2x) + cos(2y)) exp(-4 nu t) / 4

        Args:
            t: Time coordinates.
            x: Spatial coordinates of shape (batch_size, 2).

        Returns:
            Pressure values of shape (batch_size,).
        """
        t = t.reshape(-1)
        x = x.reshape(-1, 2)
        x_coord = x[:, 0]
        y_coord = x[:, 1]

        decay = torch.exp(-4.0 * self.viscosity * t)
        return -(torch.cos(2.0 * x_coord) + torch.cos(2.0 * y_coord)) * decay / 4.0
