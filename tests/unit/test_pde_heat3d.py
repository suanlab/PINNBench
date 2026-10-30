"""Tests for the 3D heat equation problem."""

import pytest
import torch

from causal_pinn.physics.pde import Heat3D


@pytest.fixture
def pde() -> Heat3D:
    return Heat3D(alpha=1.0, length_x=1.0, length_y=1.0, length_z=1.0, t_max=1.0)


def _coords(n: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    t = torch.rand(n, 1, requires_grad=True)
    x = torch.rand(n, 3, requires_grad=True)
    return t, x


class TestHeat3D:
    def test_domain_bounds(self, pde: Heat3D) -> None:
        t_bounds, x_bounds = pde.domain_bounds
        assert t_bounds == (0.0, 1.0)
        assert len(x_bounds) == 6  # (x_min, x_max, y_min, y_max, z_min, z_max)

    def test_exact_solution_satisfies_residual(self, pde: Heat3D) -> None:
        """The analytical solution must drive the PDE residual to zero."""
        t, x = _coords()
        r = pde.residual(pde.exact_solution(t, x), t, x)
        assert r.abs().max().item() < 1e-4

    def test_residual_is_nonzero_for_non_solutions(self, pde: Heat3D) -> None:
        """A field with the wrong time dependence must not satisfy the PDE."""
        t, x = _coords()
        spatial = (
            torch.sin(torch.pi * x[:, 0])
            * torch.sin(torch.pi * x[:, 1])
            * torch.sin(torch.pi * x[:, 2])
        )
        wrong = spatial * (1.0 + t.reshape(-1))
        assert pde.residual(wrong, t, x).abs().mean().item() > 1.0

    def test_residual_detects_wrong_diffusivity(self, pde: Heat3D) -> None:
        """Feeding an alpha=1 solution to an alpha=4 operator leaves a residual."""
        t, x = _coords()
        u = pde.exact_solution(t, x)
        assert Heat3D(alpha=4.0).residual(u, t, x).abs().mean().item() > 1e-2

    def test_decay_rate_matches_theory(self, pde: Heat3D) -> None:
        """Unit box: the fundamental mode decays at 3*alpha*pi^2."""
        expected = 3.0 * torch.pi**2
        assert pde._decay_rate() == pytest.approx(expected, rel=1e-6)

    def test_exact_solution_matches_initial_condition_at_t0(self, pde: Heat3D) -> None:
        x = torch.rand(64, 3)
        t0 = torch.zeros(64, 1)
        torch.testing.assert_close(
            pde.exact_solution(t0, x), pde.initial_condition(x)
        )

    def test_solution_vanishes_on_boundary_faces(self, pde: Heat3D) -> None:
        """Homogeneous Dirichlet data on every face of the box."""
        for axis in range(3):
            for value in (0.0, 1.0):
                x = torch.rand(32, 3)
                x[:, axis] = value
                t = torch.rand(32, 1)
                assert pde.exact_solution(t, x).abs().max().item() < 1e-5

    def test_solution_decays_in_time(self, pde: Heat3D) -> None:
        x = torch.rand(64, 3)
        early = pde.exact_solution(torch.zeros(64, 1), x).abs().sum()
        late = pde.exact_solution(torch.full((64, 1), 0.5), x).abs().sum()
        assert late.item() < early.item()

    def test_output_shape(self, pde: Heat3D) -> None:
        t, x = _coords(48)
        assert pde.exact_solution(t, x).shape == (48,)
        assert pde.residual(pde.exact_solution(t, x), t, x).shape == (48,)

    def test_registered_for_routing(self) -> None:
        """heat3d must be discoverable by the meta-router as a d=3 problem."""
        from causal_pinn.training.meta_router import PDE_FEATURE_REGISTRY

        assert PDE_FEATURE_REGISTRY["heat3d"].spatial_dim == 3
        assert PDE_FEATURE_REGISTRY["heat3d"].has_exact_solution
