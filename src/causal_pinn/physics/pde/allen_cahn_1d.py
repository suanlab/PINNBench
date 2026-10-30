"""1D Allen-Cahn equation PDE definition.

This module implements the 1D Allen-Cahn equation:
    ∂u/∂t = ε² ∂²u/∂x² + u − u³

When ``front_center`` is provided the class uses the stationary tanh-front
exact solution:

    u(x) = −tanh( (x − x₀) / (√2 ε) )

which satisfies u_t = 0 and ε²u_xx + u − u³ = 0 exactly.
"""

import math
from typing import cast, override

import torch
from torch import Tensor

from .base import PDEProblem


class AllenCahn1D(PDEProblem):
    """1D Allen-Cahn equation.

    **Default mode** (``front_center=None``):
        - Domain [0, L], Dirichlet BCs u = 0.
        - IC: sin(πx/L).
        - No exact solution.

    **Front mode** (``front_center`` is set):
        - Stationary tanh-front exact solution.
        - BCs are the exact solution evaluated at domain boundaries (≈ ±1).
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        length: float = 1.0,
        t_max: float = 1.0,
        front_center: float | None = None,
    ) -> None:
        """Initialize the 1D Allen-Cahn equation.

        Args:
            epsilon: Interface thickness parameter ε.  Default: 0.1
            length: Spatial domain length L.  Default: 1.0
            t_max: Maximum simulation time.  Default: 1.0
            front_center: If set, enables *front mode* with the
                stationary tanh-front centred at this x-coordinate.
                Default: None (legacy mode with zero-Dirichlet BCs).
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        self.epsilon: float = epsilon
        self.length: float = length
        self.t_max: float = t_max
        self.front_center: float | None = front_center
        self._inv_width: float = 1.0 / (math.sqrt(2.0) * epsilon)

    @property
    def domain_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((0.0, self.t_max), (0.0, self.length))

    @override
    def residual(self, u: Tensor, t: Tensor, x: Tensor) -> Tensor:
        u_flat = u.reshape(-1)

        ones = torch.ones_like(u_flat)
        u_t_opt = cast(
            Tensor | None,
            torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            )[0],
        )
        if u_t_opt is None:
            u_t_opt = torch.zeros_like(u_flat)
        u_t = u_t_opt.reshape(-1)

        u_x_opt = cast(
            Tensor | None,
            torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            )[0],
        )
        if u_x_opt is None:
            u_x_opt = torch.zeros_like(u_flat)
        u_x = u_x_opt.reshape(-1)

        ones_x = torch.ones_like(u_x)
        u_xx_opt = cast(
            Tensor | None,
            torch.autograd.grad(
            outputs=u_x,
            inputs=x,
            grad_outputs=ones_x,
            create_graph=True,
            retain_graph=True,
            )[0],
        )
        if u_xx_opt is None:
            u_xx_opt = torch.zeros_like(u_flat)
        u_xx = u_xx_opt.reshape(-1)

        return u_t - (self.epsilon ** 2) * u_xx - u_flat + u_flat**3  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Front helper
    # ------------------------------------------------------------------

    def _front(self, x: Tensor) -> Tensor:
        """Evaluate the stationary tanh-front at spatial positions *x*."""
        x_flat = x.reshape(-1)
        return -torch.tanh(
            (x_flat - self.front_center) * self._inv_width  # type: ignore[operator]
        )

    # ------------------------------------------------------------------
    # Interface methods
    # ------------------------------------------------------------------

    @override
    def boundary_condition(self, t: Tensor, x: Tensor) -> Tensor:
        """Boundary condition values.

        Default mode: u = 0.
        Front mode: exact front evaluated at boundary points.
        """
        if self.front_center is not None:
            return self._front(x)
        batch_size = t.reshape(-1).shape[0]
        return torch.zeros(batch_size, dtype=t.dtype, device=t.device)

    @override
    def initial_condition(self, x: Tensor) -> Tensor:
        """Initial condition.

        Default mode: sin(πx/L).
        Front mode: exact front at t = 0 (time-independent).
        """
        if self.front_center is not None:
            return self._front(x)
        x = x.reshape(-1)
        return torch.sin(torch.pi * x / self.length)

    @override
    def exact_solution(self, t: Tensor, x: Tensor) -> Tensor | None:
        """Return the exact analytical solution if available.

        Returns the stationary tanh-front in front mode, None otherwise.
        """
        if self.front_center is not None:
            return self._front(x)
        return None
