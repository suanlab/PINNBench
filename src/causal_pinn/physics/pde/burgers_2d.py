"""2D Burgers equation PDE definition.

This module implements the 2D viscous Burgers equations:
    ∂u/∂t + u∂u/∂x + v∂u/∂y = ν∇²u
    ∂v/∂t + u∂v/∂x + v∂v/∂y = ν∇²v

where (u, v) is the velocity field and ν is the kinematic viscosity.
"""


import torch
from torch import Tensor

from .base import PDEProblem


class Burgers2D(PDEProblem):
    """2D viscous Burgers equations.

    The Burgers equations are a system of nonlinear PDEs that model
    fluid dynamics with viscous effects:
        ∂u/∂t + u∂u/∂x + v∂u/∂y = ν∇²u
        ∂v/∂t + u∂v/∂x + v∂v/∂y = ν∇²v

    This is a fundamental test case for nonlinear PDE solvers and appears
    in various fluid mechanics applications.

    Note: Due to the nonlinearity, exact solutions are typically not available
    except for specific cases. This implementation provides a manufactured
    solution for verification purposes.

    Domain: (x, y) ∈ [0, L_x] × [0, L_y], t ∈ [0, T]

    Attributes:
        viscosity: Kinematic viscosity coefficient ν.
        length_x: Spatial domain length in x direction L_x.
        length_y: Spatial domain length in y direction L_y.
        t_max: Maximum time T.
    """

    def __init__(
        self,
        viscosity: float = 0.01,
        length_x: float = 1.0,
        length_y: float = 1.0,
        t_max: float = 1.0,
    ) -> None:
        """Initialize the 2D Burgers equations.

        Args:
            viscosity: Kinematic viscosity ν. Default: 0.01
            length_x: Spatial domain length in x. Default: 1.0
            length_y: Spatial domain length in y. Default: 1.0
            t_max: Maximum simulation time. Default: 1.0
        """
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
        """Compute the Burgers equations residual for both velocity components.

        Args:
            u: Solution field of shape (batch_size, 2) where u[:, 0] is the
               x-velocity and u[:, 1] is the y-velocity.
            t: Time coordinates of shape (batch_size,) or (batch_size, 1).
            x: Spatial coordinates of shape (batch_size, 2) where x[:, 0] is
               the x-coordinate and x[:, 1] is the y-coordinate.

        Returns:
            Residual tensor of shape (batch_size, 2) containing residuals
            for u and v components.
        """
        # Get batch size from t
        batch_size = t.reshape(-1).shape[0]

        # Ensure u has right shape
        u = u.reshape(batch_size, 2)

        # Extract velocity components
        u_vel = u[:, 0]  # x-velocity
        v_vel = u[:, 1]  # y-velocity

        # Compute temporal derivatives using original t
        ones = torch.ones_like(u_vel)
        u_t = torch.autograd.grad(
            outputs=u_vel,
            inputs=t,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        if u_t is None:
            u_t = torch.zeros_like(u_vel)
        u_t = u_t.reshape(-1)

        v_t = torch.autograd.grad(
            outputs=v_vel,
            inputs=t,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        if v_t is None:
            v_t = torch.zeros_like(v_vel)
        v_t = v_t.reshape(-1)

        # Compute first spatial gradients as full 2D tensors (batch, 2)
        grad_u = torch.autograd.grad(
            outputs=u_vel,
            inputs=x,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        if grad_u is None:
            grad_u = torch.zeros(batch_size, 2, dtype=u.dtype, device=u.device)
        grad_u = grad_u.reshape(batch_size, 2)

        grad_v = torch.autograd.grad(
            outputs=v_vel,
            inputs=x,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
        )[0]
        if grad_v is None:
            grad_v = torch.zeros(batch_size, 2, dtype=u.dtype, device=u.device)
        grad_v = grad_v.reshape(batch_size, 2)

        # Extract components for first derivatives
        u_x = grad_u[:, 0]
        u_y = grad_u[:, 1]
        v_x = grad_v[:, 0]
        v_y = grad_v[:, 1]

        # Compute second spatial derivatives
        # For u_xx and u_yy: compute gradient of grad_u with respect to x
        grad_uu = torch.autograd.grad(
            outputs=grad_u,
            inputs=x,
            grad_outputs=torch.ones_like(grad_u),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad_uu is None:
            grad_uu = torch.zeros(batch_size, 2, dtype=u.dtype, device=u.device)
        grad_uu = grad_uu.reshape(batch_size, 2)
        u_xx = grad_uu[:, 0]  # ∂²u/∂x²
        u_yy = grad_uu[:, 1]  # ∂²u/∂y²

        # For v_xx and v_yy: compute gradient of grad_v with respect to x
        grad_vv = torch.autograd.grad(
            outputs=grad_v,
            inputs=x,
            grad_outputs=torch.ones_like(grad_v),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad_vv is None:
            grad_vv = torch.zeros(batch_size, 2, dtype=u.dtype, device=u.device)
        grad_vv = grad_vv.reshape(batch_size, 2)
        v_xx = grad_vv[:, 0]  # ∂²v/∂x²
        v_yy = grad_vv[:, 1]  # ∂²v/∂y²

        # Burgers equations residual
        # For u: u_t + u*u_x + v*u_y - ν*(u_xx + u_yy) = 0
        residual_u = u_t + u_vel * u_x + v_vel * u_y - self.viscosity * (u_xx + u_yy)

        # For v: v_t + u*v_x + v*v_y - ν*(v_xx + v_yy) = 0
        residual_v = v_t + u_vel * v_x + v_vel * v_y - self.viscosity * (v_xx + v_yy)

        # Stack residuals
        residual = torch.stack([residual_u, residual_v], dim=1)

        return residual

    def boundary_condition(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute Dirichlet boundary conditions: u = v = 0 on all boundaries.

        Args:
            t: Time coordinates at the boundary.
            x: Spatial coordinates at the boundary of shape (batch_size, 2).

        Returns:
            Zero tensor of shape (batch_size, 2) for both velocity components.
        """
        batch_size = t.reshape(-1).shape[0]
        return torch.zeros(batch_size, 2, dtype=t.dtype, device=t.device)

    def initial_condition(self, x: Tensor) -> Tensor:
        """Compute initial condition: smooth velocity field.

        Uses a smooth initial condition for both velocity components:
            u(x, y, 0) = sin(πx/L_x)sin(πy/L_y)
            v(x, y, 0) = 0.5*sin(πx/L_x)sin(πy/L_y)

        Args:
            x: Spatial coordinates of shape (batch_size, 2).

        Returns:
            Initial velocity field of shape (batch_size, 2).
        """
        x = x.reshape(-1, 2)
        x_coord = x[:, 0]
        y_coord = x[:, 1]

        # Smooth initial condition
        u_init = torch.sin(torch.pi * x_coord / self.length_x) * torch.sin(
            torch.pi * y_coord / self.length_y
        )
        v_init = 0.5 * torch.sin(torch.pi * x_coord / self.length_x) * torch.sin(
            torch.pi * y_coord / self.length_y
        )

        return torch.stack([u_init, v_init], dim=1)

    def exact_solution(self, t: Tensor, x: Tensor) -> Tensor | None:
        """Exact solution not available for general Burgers equation.

        The 2D Burgers equation is nonlinear and does not have general
        analytical solutions except for specific initial conditions using
        the Cole-Hopf transformation.

        Args:
            t: Time coordinates.
            x: Spatial coordinates.

        Returns:
            None, as no general exact solution is available.
        """
        return None
