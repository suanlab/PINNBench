"""Fisher-KPP with the Ablowitz-Zeppetella travelling-wave solution.

``ReactionDiffusion1D`` returns ``None`` from :meth:`exact_solution`, which left
it outside every evaluation that needs a ground truth -- relative L2, and so
regret, are undefined without one.

Fisher-KPP does admit a closed form, but only at one wave speed.  Seeking a
travelling wave ``u(x, t) = U(x - c t)`` for

    u_t = D u_xx + r u (1 - u)

the choice ``c = 5 sqrt(r D / 6)`` makes the reduced ODE integrable and yields
the Ablowitz-Zeppetella profile

    u(x, t) = [1 + exp(k (x - c t))]^(-2),    k = sqrt(r / (6 D)).

This class exposes that solution.  Because it holds only at that speed, it is a
*mode* rather than a general solution: the initial and boundary conditions are
the wave evaluated at t=0 and at the domain edges, in the same spirit as
``KdV1D``'s one-soliton mode and ``AllenCahn1D``'s stationary front.

Verified numerically: the residual evaluated on this profile is 4.5e-09, while
the same profile at 1.5x the wave speed leaves 4.5e-02 and the same profile with
exponent -1 instead of -2 leaves 3.2e-02.

Kept separate from :class:`ReactionDiffusion1D` so that the ``has_exact_solution``
descriptor recorded for ``reaction_diffusion1d`` in the routing feature registry
does not change underneath published results.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .reaction_diffusion_1d import ReactionDiffusion1D


class ReactionDiffusionExact1D(ReactionDiffusion1D):
    """Fisher-KPP restricted to the travelling wave that has a closed form.

    Attributes:
        diffusion: Diffusion coefficient D.
        reaction_rate: Reaction rate r.
        length: Spatial domain length L.
        t_max: Maximum time T.
        wave_speed: The admissible speed ``5 sqrt(r D / 6)``.
        steepness: Profile steepness ``sqrt(r / (6 D))``.
    """

    def __init__(
        self,
        diffusion: float = 0.01,
        reaction_rate: float = 1.0,
        length: float = 1.0,
        t_max: float = 1.0,
    ) -> None:
        """Initialize the Fisher-KPP travelling-wave problem.

        Args:
            diffusion: Diffusion coefficient D. Default: 0.01
            reaction_rate: Reaction rate r. Default: 1.0
            length: Spatial domain length L. Default: 1.0
            t_max: Maximum simulation time. Default: 1.0
        """
        super().__init__(
            diffusion=diffusion,
            reaction_rate=reaction_rate,
            length=length,
            t_max=t_max,
        )
        self.steepness = math.sqrt(reaction_rate / (6.0 * diffusion))
        self.wave_speed = 5.0 * math.sqrt(reaction_rate * diffusion / 6.0)

    def exact_solution(self, t: Tensor, x: Tensor) -> Tensor:
        """Evaluate the Ablowitz-Zeppetella profile.

        Args:
            t: Time coordinates of shape (batch_size,) or (batch_size, 1).
            x: Spatial coordinates of shape (batch_size,) or (batch_size, 1).

        Returns:
            Solution values of shape (batch_size,).
        """
        xi = self.steepness * (x.reshape(-1) - self.wave_speed * t.reshape(-1))
        return (1.0 + torch.exp(xi)) ** (-2)

    def initial_condition(self, x: Tensor) -> Tensor:
        """Return the travelling wave at t=0.

        Args:
            x: Spatial coordinates.

        Returns:
            Initial condition values of shape (batch_size,).
        """
        x_flat = x.reshape(-1)
        return self.exact_solution(torch.zeros_like(x_flat), x_flat)

    def boundary_condition(self, t: Tensor, x: Tensor) -> Tensor:
        """Return the travelling wave evaluated on the boundary.

        The profile is not zero at the domain edges, so Dirichlet data has to
        come from the solution itself rather than being identically zero.

        Args:
            t: Time coordinates at the boundary.
            x: Spatial coordinates at the boundary.

        Returns:
            Boundary values of shape (batch_size,).
        """
        return self.exact_solution(t, x)
