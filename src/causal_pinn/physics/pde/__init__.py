"""PDE problem definitions for Physics-Informed Neural Networks.

This module provides abstract and concrete PDE problem definitions that can be
used with PINN frameworks. Each PDE class defines the governing equation,
boundary conditions, initial conditions, and (when available) exact solutions.

Available PDEs:
    - Heat1D: 1D heat/diffusion equation
    - AdvectionDiffusion1D: 1D advection-diffusion equation
    - Burgers1D: 1D viscous Burgers equation
    - BurgersExact1D: Burgers1D plus a Cole-Hopf reference solution
    - Burgers2D: 2D viscous Burgers equations
    - MultiscaleHeat2D: 2D heat equation with two separated scales
    - Wave2D: 2D wave equation

Example:
    >>> from causal_pinn.physics.pde import Heat1D
    >>> pde = Heat1D(alpha=1.0, length=1.0, t_max=1.0)
    >>> bounds = pde.domain_bounds
    >>> print(bounds)  # ((0.0, 1.0), (0.0, 1.0))
"""

from .advection_diffusion_1d import AdvectionDiffusion1D
from .allen_cahn_1d import AllenCahn1D
from .base import PDEProblem
from .burgers_1d import Burgers1D
from .burgers_1d_exact import BurgersExact1D
from .burgers_2d import Burgers2D
from .heat_1d import Heat1D
from .heat_3d import Heat3D
from .kdv_1d import KdV1D
from .multiscale_heat_2d import MultiscaleHeat2D
from .navier_stokes_2d import NavierStokes2D
from .reaction_diffusion_1d import ReactionDiffusion1D
from .reaction_diffusion_1d_exact import ReactionDiffusionExact1D
from .wave_2d import Wave2D

__all__ = [
    "PDEProblem",
    "Heat1D",
    "Heat3D",
    "AdvectionDiffusion1D",
    "AllenCahn1D",
    "Burgers1D",
    "BurgersExact1D",
    "KdV1D",
    "MultiscaleHeat2D",
    "NavierStokes2D",
    "ReactionDiffusion1D",
    "ReactionDiffusionExact1D",
    "Burgers2D",
    "Wave2D",
]
