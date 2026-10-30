"""1D Korteweg-de Vries (KdV) equation PDE definition.

This module implements the 1D KdV equation:
    ∂u/∂t + u * ∂u/∂x + μ * ∂³u/∂x³ = 0

where μ is the dispersion coefficient.

When ``soliton_amplitude`` is provided the class switches to *soliton mode*:
the initial / boundary conditions and ``exact_solution`` are derived from
the one-soliton analytical solution

    u(x, t) = 3c · sech²( κ (x − c t − x₀) )

with speed c = soliton_amplitude / 3 and κ = √(soliton_amplitude / (4μ)).
"""

import math
from typing import cast, override

import torch
from torch import Tensor

from .base import PDEProblem


class KdV1D(PDEProblem):
    """1D KdV equation with Dirichlet boundary conditions.

    The KdV equation models nonlinear wave propagation and is given by:
        ∂u/∂t + u * ∂u/∂x + μ * ∂³u/∂x³ = 0

    This implementation uses the domain x ∈ [0, L], t ∈ [0, T].

    **Default mode** (``soliton_amplitude=None``):
        - Boundary conditions: u(0, t) = u(L, t) = 0
        - Initial condition: u(x, 0) = -sin(πx/L)
        - No closed-form exact solution.

    **Soliton mode** (``soliton_amplitude`` is set):
        - Uses the one-soliton analytical solution as IC / BC / exact.
        - Choose domain large enough so the soliton decays at boundaries.
    """

    def __init__(
        self,
        dispersion: float = 1.0,
        length: float = 2.0,
        t_max: float = 0.5,
        soliton_amplitude: float | None = None,
        soliton_center: float | None = None,
    ) -> None:
        """Initialize the 1D KdV equation.

        Args:
            dispersion: Dispersion coefficient μ. Default: 1.0
            length: Spatial domain length L. Default: 2.0
            t_max: Maximum simulation time. Default: 0.5
            soliton_amplitude: If set, enables soliton mode with this
                amplitude parameter *a*.  The soliton speed is a/3 and
                the peak value is *a*.  Default: None (legacy mode).
            soliton_center: Initial center x₀ of the soliton.
                Only used when ``soliton_amplitude`` is not None.
                Default: None (falls back to length / 2).
        """
        if dispersion <= 0:
            raise ValueError(f"dispersion must be positive, got {dispersion}")
        if soliton_amplitude is not None and soliton_amplitude <= 0:
            raise ValueError(
                f"soliton_amplitude must be positive, got {soliton_amplitude}"
            )
        self.dispersion: float = dispersion
        self.length: float = length
        self.t_max: float = t_max
        self.soliton_amplitude: float | None = soliton_amplitude

        if soliton_amplitude is not None:
            self.soliton_center: float = (
                soliton_center if soliton_center is not None
                else length / 2.0
            )
            self.soliton_speed: float = soliton_amplitude / 3.0
            self.soliton_kappa: float = math.sqrt(
                self.soliton_speed / (4.0 * dispersion)
            )
        else:
            self.soliton_center = 0.0
            self.soliton_speed = 0.0
            self.soliton_kappa = 0.0

    @property
    def domain_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the domain bounds for time and space."""
        return ((0.0, self.t_max), (0.0, self.length))

    @override
    def residual(self, u: Tensor, t: Tensor, x: Tensor) -> Tensor:
        """Compute the KdV residual: ∂u/∂t + u∂u/∂x + μ∂³u/∂x³."""
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

        ones_xx = torch.ones_like(u_xx)
        u_xxx_opt = cast(
            Tensor | None,
            torch.autograd.grad(
            outputs=u_xx,
            inputs=x,
            grad_outputs=ones_xx,
            create_graph=True,
            retain_graph=True,
            )[0],
        )
        if u_xxx_opt is None:
            u_xxx_opt = torch.zeros_like(u_flat)
        u_xxx = u_xxx_opt.reshape(-1)

        residual = u_t + u_flat * u_x + self.dispersion * u_xxx
        return residual

    # ------------------------------------------------------------------
    # Soliton helper (pure function, no autograd needed)
    # ------------------------------------------------------------------

    def _soliton(self, t: Tensor, x: Tensor) -> Tensor:
        """Evaluate the one-soliton solution at (t, x)."""
        t_flat = t.reshape(-1)
        x_flat = x.reshape(-1)
        eta = self.soliton_kappa * (
            x_flat - self.soliton_speed * t_flat - self.soliton_center
        )
        return self.soliton_amplitude * torch.pow(  # type: ignore[arg-type]
            1.0 / torch.cosh(eta), 2
        )

    # ------------------------------------------------------------------
    # Interface methods
    # ------------------------------------------------------------------

    @override
    def boundary_condition(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute boundary condition values.

        Default mode: u(0, t) = u(L, t) = 0.
        Soliton mode: exact soliton evaluated at the boundary points.
        """
        if self.soliton_amplitude is not None:
            return self._soliton(t, x)
        batch_size = t.reshape(-1).shape[0]
        return torch.zeros(batch_size, dtype=t.dtype, device=t.device)

    @override
    def initial_condition(self, x: Tensor) -> Tensor:
        """Compute initial condition.

        Default mode: u(x, 0) = -sin(πx/L).
        Soliton mode: exact soliton at t = 0.
        """
        if self.soliton_amplitude is not None:
            t_zero = torch.zeros_like(x.reshape(-1))
            return self._soliton(t_zero, x)
        x = x.reshape(-1)
        return -torch.sin(torch.pi * x / self.length)

    @override
    def exact_solution(self, t: Tensor, x: Tensor) -> Tensor | None:
        """Return the exact analytical solution if available.

        Returns the one-soliton solution in soliton mode, None otherwise.
        """
        if self.soliton_amplitude is not None:
            return self._soliton(t, x)
        return None
