"""Tests for the Cole-Hopf reference solution of the viscous Burgers equation."""

import torch

from causal_pinn.physics.pde import Burgers1D, BurgersExact1D


def _coords(n: int = 300) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    # Stay away from t=0, where the convolution degenerates by construction.
    t = (0.2 + 0.6 * torch.rand(n, 1)).requires_grad_(True)
    x = torch.rand(n, 1).requires_grad_(True)
    return t, x


class TestBurgersExact1D:
    def test_satisfies_the_pde(self) -> None:
        """The Cole-Hopf field must drive the Burgers residual to zero."""
        pde = BurgersExact1D(viscosity=0.01)
        t, x = _coords()
        r = pde.residual(pde.exact_solution(t, x), t, x)
        assert r.abs().max().item() < 1e-4

    def test_residual_rejects_a_perturbed_field(self) -> None:
        """Burgers is nonlinear, so a rescaled solution is not a solution."""
        pde = BurgersExact1D(viscosity=0.01)
        t, x = _coords()
        u = pde.exact_solution(t, x)
        assert pde.residual(1.5 * u + 0.2, t, x).abs().mean().item() > 1e-2

    def test_dirichlet_boundaries(self) -> None:
        pde = BurgersExact1D(viscosity=0.01, length=1.0)
        t = 0.2 + 0.6 * torch.rand(40, 1)
        for x_value in (0.0, 1.0):
            x = torch.full((40, 1), x_value)
            assert pde.exact_solution(t, x).abs().max().item() < 1e-8

    def test_recovers_initial_condition_at_t0(self) -> None:
        pde = BurgersExact1D(viscosity=0.01)
        x = torch.rand(200, 1)
        u0 = pde.exact_solution(torch.zeros(200, 1), x)
        torch.testing.assert_close(u0, pde.initial_condition(x), atol=1e-5, rtol=1e-4)

    def test_solution_stays_bounded_by_initial_amplitude(self) -> None:
        """Burgers does not amplify the initial extremum."""
        pde = BurgersExact1D(viscosity=0.01)
        t, x = _coords()
        assert pde.exact_solution(t, x).abs().max().item() <= 1.0 + 1e-6

    def test_steepens_at_lower_viscosity(self) -> None:
        """Lower viscosity produces a sharper interior gradient."""
        x = torch.linspace(0.01, 0.99, 400).reshape(-1, 1)
        t = torch.full_like(x, 0.5)
        grads = []
        for nu in (0.05, 0.005):
            u = BurgersExact1D(viscosity=nu).exact_solution(t, x)
            grads.append((u[1:] - u[:-1]).abs().max().item())
        assert grads[1] > grads[0]

    def test_does_not_change_the_base_class(self) -> None:
        """Burgers1D must keep reporting that it has no analytical solution."""
        assert Burgers1D().exact_solution(torch.rand(4), torch.rand(4)) is None

    def test_output_shape_and_dtype(self) -> None:
        pde = BurgersExact1D(viscosity=0.01)
        t, x = _coords(64)
        u = pde.exact_solution(t, x)
        assert u.shape == (64,)
        assert u.dtype == x.dtype
