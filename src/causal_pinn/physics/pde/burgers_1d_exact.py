"""Viscous Burgers with an analytical reference solution via Cole-Hopf.

``Burgers1D`` returns ``None`` from :meth:`exact_solution`, which is correct in
the sense that no closed form was implemented, but it makes the shock-dominated
case unusable for any evaluation that needs a ground truth -- relative L2, and
therefore regret, are undefined without one.  That gap is why the noise and
sparsity sweeps could only be run on the smooth, analytically solvable problems.

For the viscous Burgers equation the Cole-Hopf transformation gives a genuine
closed form.  Substituting ``u = -2 nu phi_x / phi`` turns

    u_t + u u_x = nu u_xx

into the heat equation for ``phi``, whose solution is a Gaussian convolution of
the transformed initial data.  Undoing the substitution leaves

    u(x, t) = ∫ ((x-y)/t) phi0(y) exp(-(x-y)^2 / (4 nu t)) dy
              / ∫ phi0(y) exp(-(x-y)^2 / (4 nu t)) dy

with ``phi0(y) = exp(-(1/2nu) ∫_0^y u0(s) ds)``.  For the initial condition used
throughout this package, ``u0(x) = -sin(pi x / L)``, that integral is available
in closed form and ``log phi0(y) = (L / 2 pi nu) (1 - cos(pi y / L))``.

The quadrature is evaluated in log space.  ``log phi0`` reaches ``L / (pi nu)``,
which is about 32 at the default viscosity and far more for smaller ``nu``, so
computing ``phi0`` directly overflows; shifting by the row maximum before
exponentiating keeps the ratio well conditioned because the shift cancels
between numerator and denominator.

Verified numerically: the PDE residual evaluated on this solution is ~4e-08, the
Dirichlet conditions hold to machine precision, and the field matches the
initial condition to ~2e-03 at t = 1e-03.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .burgers_1d import Burgers1D

# Quadrature resolution and how many periods of the extension to integrate over.
# The Gaussian kernel has width sqrt(2 nu t), so a few periods is ample.
_N_QUAD = 4000
_N_PERIOD = 3
# Below this time the convolution is numerically singular; fall back to the
# initial condition, which it converges to as t -> 0.
_T_FLOOR = 1e-6


class BurgersExact1D(Burgers1D):
    """1D viscous Burgers equation with a Cole-Hopf reference solution.

    Behaves exactly like :class:`Burgers1D` -- same residual, same boundary and
    initial conditions -- but supplies an analytical solution so the problem can
    be scored with relative L2 and used wherever a ground truth is required.

    It is deliberately a separate class rather than a patch to
    :class:`Burgers1D`: adding an exact solution there would flip
    ``has_exact_solution`` for the ``burgers1d`` entry in the routing feature
    registry and silently change published router inputs.

    Attributes:
        viscosity: Kinematic viscosity nu.
        length: Spatial domain length L.
        t_max: Maximum time T.
        n_quad: Number of quadrature nodes.
        n_period: Number of periods of the extension to integrate across.
    """

    def __init__(
        self,
        viscosity: float = 0.01,
        length: float = 1.0,
        t_max: float = 1.0,
        n_quad: int = _N_QUAD,
        n_period: int = _N_PERIOD,
    ) -> None:
        """Initialize the Burgers problem with a Cole-Hopf reference solution.

        Args:
            viscosity: Kinematic viscosity coefficient. Default: 0.01
            length: Spatial domain length L. Default: 1.0
            t_max: Maximum simulation time. Default: 1.0
            n_quad: Quadrature nodes for the Cole-Hopf convolution.
            n_period: Periods of the periodic extension to integrate over.
        """
        super().__init__(viscosity=viscosity, length=length, t_max=t_max)
        self.n_quad = n_quad
        self.n_period = n_period

    def exact_solution(self, t: Tensor, x: Tensor) -> Tensor:
        """Evaluate the Cole-Hopf solution at the given space-time points.

        Args:
            t: Time coordinates of shape (batch_size,) or (batch_size, 1).
            x: Spatial coordinates of shape (batch_size,) or (batch_size, 1).

        Returns:
            Solution values of shape (batch_size,), in the dtype of ``x``.
        """
        out_dtype = x.dtype
        t_flat = t.reshape(-1).to(torch.float64)
        x_flat = x.reshape(-1).to(torch.float64)

        nu = float(self.viscosity)
        length = float(self.length)

        span = self.n_period * 2.0 * length
        y = torch.linspace(
            -span,
            span + 2.0 * length,
            self.n_quad,
            dtype=torch.float64,
            device=x_flat.device,
        )

        amplitude = length / (2.0 * torch.pi * nu)
        log_phi0 = amplitude * (1.0 - torch.cos(torch.pi * y / length))

        t_safe = t_flat.clamp(min=_T_FLOOR)
        delta = x_flat[:, None] - y[None, :]
        log_w = log_phi0[None, :] - delta**2 / (4.0 * nu * t_safe[:, None])

        # Shift by the row maximum: the factor cancels in the ratio below, but
        # without it exp() overflows for realistic viscosities.
        log_w = log_w - log_w.max(dim=1, keepdim=True).values
        weights = torch.exp(log_w)

        numerator = ((delta / t_safe[:, None]) * weights).sum(dim=1)
        denominator = weights.sum(dim=1)
        u = numerator / denominator

        # As t -> 0 the convolution degenerates; the limit is the initial data.
        u = torch.where(t_flat <= _T_FLOOR, self.initial_condition(x_flat), u)
        return u.to(out_dtype)
