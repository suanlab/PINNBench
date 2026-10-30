"""2D multiscale heat equation PDE definition.

This module implements the 2D heat equation carrying two widely separated
spatial modes:
    ∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)

It exists to give the benchmark a genuinely multi-scale configuration.  Every
other 2D case in the suite (Wave2D, NavierStokes2D) carries a single dominant
scale, so any claim about how routing or training policies behave under scale
separation previously had no evidence behind it.

The construction is a superposition of two Dirichlet eigenmodes.  Because each
eigenmode decays at its own rate, separation appears on *both* axes at once:

    space  -- wavelengths differ by ``scale_ratio`` (default 4x)
    time   -- decay rates differ by ``scale_ratio`` squared (default 16x)

The temporal separation is the point of interest here.  The fine mode is gone
long before the coarse one, so early dynamics and late dynamics look nothing
alike, which is precisely the assumption the early-dynamics router rests on.
The superposition stays analytic for all time, which keeps relative-L2 (and
therefore regret) well defined.
"""

import torch
from torch import Tensor

from .base import PDEProblem


class MultiscaleHeat2D(PDEProblem):
    """2D heat equation with two separated scales and Dirichlet boundaries.

    The governing equation is
        ∂u/∂t = α ∇²u

    on the box x ∈ [0, Lx], y ∈ [0, Ly], t ∈ [0, T], with u = 0 on every edge.
    Taking a two-mode initial condition,

        u(x, y, 0) = sin(πx/Lx) sin(πy/Ly)
                     + A sin(mπx/Lx) sin(mπy/Ly),

    with m = ``scale_ratio``, each term is a Dirichlet eigenmode of the
    Laplacian, so the solution is the superposition of their independent
    decays:

        u = sin(πx/Lx) sin(πy/Ly) exp(-α π² (Lx⁻² + Ly⁻²) t)
            + A sin(mπx/Lx) sin(mπy/Ly) exp(-α m² π² (Lx⁻² + Ly⁻²) t).

    The fine mode therefore decays m² times faster than the coarse one.

    Spatial coordinates are packed as a (batch, 2) tensor, matching the
    convention used by the other 2D problems in this package.

    Attributes:
        alpha: Thermal diffusivity coefficient.
        scale_ratio: Wavenumber ratio m between the fine and coarse modes.
        fine_amplitude: Amplitude A of the fine mode.
        length_x: Domain extent in x.
        length_y: Domain extent in y.
        t_max: Maximum time T.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        scale_ratio: int = 4,
        fine_amplitude: float = 0.5,
        length_x: float = 1.0,
        length_y: float = 1.0,
        t_max: float = 1.0,
    ) -> None:
        """Initialize the 2D multiscale heat equation.

        Args:
            alpha: Thermal diffusivity. Default 0.01, chosen so that both modes
                remain resolvable on t ∈ [0, 1]: the coarse mode retains most of
                its amplitude while the fine mode decays away within the window.
            scale_ratio: Wavenumber ratio m between fine and coarse modes.
                Must be an integer ≥ 2 so the fine mode is a distinct Dirichlet
                eigenmode. Default: 4.
            fine_amplitude: Amplitude A of the fine mode. Default: 0.5.
            length_x: Domain extent in x. Default: 1.0
            length_y: Domain extent in y. Default: 1.0
            t_max: Maximum simulation time. Default: 1.0

        Raises:
            ValueError: If scale_ratio is not an integer ≥ 2.
        """
        if int(scale_ratio) != scale_ratio or scale_ratio < 2:
            raise ValueError(
                f"scale_ratio must be an integer >= 2 for the fine mode to be a "
                f"distinct Dirichlet eigenmode, got {scale_ratio}"
            )
        self.alpha = alpha
        self.scale_ratio = int(scale_ratio)
        self.fine_amplitude = fine_amplitude
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

    def _base_rate(self) -> float:
        """Decay rate of the coarse (fundamental) mode."""
        return self.alpha * (torch.pi**2) * (1.0 / self.length_x**2 + 1.0 / self.length_y**2)

    def coarse_decay_rate(self) -> float:
        """Exponential decay rate of the coarse mode."""
        return self._base_rate()

    def fine_decay_rate(self) -> float:
        """Exponential decay rate of the fine mode (m² times the coarse rate)."""
        return self._base_rate() * float(self.scale_ratio**2)

    def residual(self, u: Tensor, t: Tensor, x: Tensor) -> Tensor:
        """Compute the heat equation residual: ∂u/∂t - α ∇²u.

        Args:
            u: Solution field of shape (batch_size,) or (batch_size, 1).
            t: Time coordinates of shape (batch_size,) or (batch_size, 1).
            x: Spatial coordinates of shape (batch_size, 2), columns (x, y).

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
            grad_u = torch.zeros(batch_size, 2, dtype=u.dtype, device=u.device)
        grad_u = grad_u.reshape(batch_size, 2)

        laplacian = torch.zeros_like(u_flat)
        for axis in range(2):
            second = torch.autograd.grad(
                outputs=grad_u[:, axis],
                inputs=x,
                grad_outputs=ones,
                create_graph=True,
                retain_graph=True,
            )[0]
            if second is None:
                continue
            laplacian = laplacian + second.reshape(batch_size, 2)[:, axis]

        return u_t - self.alpha * laplacian

    def boundary_condition(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute the homogeneous Dirichlet condition u = 0 on every edge.

        Both modes vanish on the boundary by construction, so the condition is
        exact for the superposition.

        Args:
            t: Time coordinates at the boundary.
            x: Spatial coordinates at the boundary.

        Returns:
            Zero tensor with batch shape matching the input.
        """
        batch_size = t.reshape(-1).shape[0]
        return torch.zeros(batch_size, dtype=t.dtype, device=t.device)

    def initial_condition(self, x: Tensor) -> Tensor:
        """Compute the two-mode initial condition at t = 0.

        Args:
            x: Spatial coordinates of shape (batch_size, 2).

        Returns:
            Initial condition values of shape (batch_size,).
        """
        coarse, fine = self._modes(x)
        return coarse + self.fine_amplitude * fine

    def _modes(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return the coarse and fine spatial modes evaluated at x.

        Args:
            x: Spatial coordinates of shape (batch_size, 2).

        Returns:
            Tuple of (coarse_mode, fine_mode), each of shape (batch_size,).
        """
        x = x.reshape(-1, 2)
        m = float(self.scale_ratio)
        coarse = torch.sin(torch.pi * x[:, 0] / self.length_x) * torch.sin(
            torch.pi * x[:, 1] / self.length_y
        )
        fine = torch.sin(m * torch.pi * x[:, 0] / self.length_x) * torch.sin(
            m * torch.pi * x[:, 1] / self.length_y
        )
        return coarse, fine

    def exact_solution(self, t: Tensor, x: Tensor) -> Tensor:
        """Compute the exact two-mode solution.

        Args:
            t: Time coordinates of shape (batch_size,) or (batch_size, 1).
            x: Spatial coordinates of shape (batch_size, 2).

        Returns:
            Exact solution values of shape (batch_size,).
        """
        t_flat = t.reshape(-1)
        coarse, fine = self._modes(x)
        return coarse * torch.exp(-self.coarse_decay_rate() * t_flat) + (
            self.fine_amplitude * fine * torch.exp(-self.fine_decay_rate() * t_flat)
        )
