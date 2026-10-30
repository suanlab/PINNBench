"""Tests for the 2D multiscale heat equation problem."""

import math

import pytest
import torch

from causal_pinn.physics.pde import MultiscaleHeat2D


@pytest.fixture
def pde() -> MultiscaleHeat2D:
    return MultiscaleHeat2D(alpha=0.01, scale_ratio=4, fine_amplitude=0.5)


def _coords(n: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    t = torch.rand(n, 1, requires_grad=True)
    x = torch.rand(n, 2, requires_grad=True)
    return t, x


def _two_mode_field(
    t: torch.Tensor,
    x: torch.Tensor,
    coarse_rate: float,
    fine_rate: float,
    amplitude: float = 0.5,
    m: float = 4.0,
) -> torch.Tensor:
    """Build a two-mode field with arbitrary decay rates, for use as a control."""
    xx = x.reshape(-1, 2)
    tt = t.reshape(-1)
    coarse = torch.sin(torch.pi * xx[:, 0]) * torch.sin(torch.pi * xx[:, 1])
    fine = torch.sin(m * torch.pi * xx[:, 0]) * torch.sin(m * torch.pi * xx[:, 1])
    return coarse * torch.exp(-coarse_rate * tt) + amplitude * fine * torch.exp(-fine_rate * tt)


class TestMultiscaleHeat2D:
    def test_domain_bounds(self, pde: MultiscaleHeat2D) -> None:
        t_bounds, x_bounds = pde.domain_bounds
        assert t_bounds == (0.0, 1.0)
        assert len(x_bounds) == 4  # (x_min, x_max, y_min, y_max)

    def test_exact_solution_satisfies_residual(self, pde: MultiscaleHeat2D) -> None:
        """The analytical superposition must drive the PDE residual to zero."""
        t, x = _coords()
        r = pde.residual(pde.exact_solution(t, x), t, x)
        assert r.abs().max().item() < 1e-4

    def test_each_mode_alone_is_a_solution(self, pde: MultiscaleHeat2D) -> None:
        """Superposition sanity: either eigenmode on its own also solves the PDE."""
        t, x = _coords()
        for amplitude in (0.0, 1.0):
            field = _two_mode_field(
                t, x, pde.coarse_decay_rate(), pde.fine_decay_rate(), amplitude=amplitude
            )
            assert pde.residual(field, t, x).abs().max().item() < 1e-4

    def test_wrong_decay_rate_violates_residual(self, pde: MultiscaleHeat2D) -> None:
        """Giving the fine mode the coarse mode's decay rate must be detected.

        This is the control that matters: the spatial profile is untouched and
        only the separation of time scales is destroyed, so passing it would
        mean the residual is blind to exactly the property under test.
        """
        t, x = _coords()
        rate = pde.coarse_decay_rate()
        field = _two_mode_field(t, x, rate, rate)
        assert pde.residual(field, t, x).abs().max().item() > 1e-2

    def test_swapped_rates_violate_residual(self, pde: MultiscaleHeat2D) -> None:
        t, x = _coords()
        field = _two_mode_field(t, x, pde.fine_decay_rate(), pde.coarse_decay_rate())
        assert pde.residual(field, t, x).abs().max().item() > 1e-2

    def test_wrong_fine_wavenumber_violates_residual(self, pde: MultiscaleHeat2D) -> None:
        """An m=3 spatial profile decaying at the m=4 rate is not a solution."""
        t, x = _coords()
        field = _two_mode_field(t, x, pde.coarse_decay_rate(), pde.fine_decay_rate(), m=3.0)
        assert pde.residual(field, t, x).abs().max().item() > 1e-2

    def test_decay_rates_separate_by_scale_ratio_squared(self, pde: MultiscaleHeat2D) -> None:
        """Fine mode must decay exactly m^2 times faster than the coarse mode."""
        ratio = pde.fine_decay_rate() / pde.coarse_decay_rate()
        assert ratio == pytest.approx(pde.scale_ratio**2)

    def test_coarse_rate_matches_closed_form(self, pde: MultiscaleHeat2D) -> None:
        assert pde.coarse_decay_rate() == pytest.approx(0.01 * math.pi**2 * 2.0)

    def test_both_scales_resolvable_in_the_time_window(self, pde: MultiscaleHeat2D) -> None:
        """The default alpha must leave the coarse mode alive and the fine one decayed.

        If either fails, the configuration is not exercising scale separation:
        a vanished coarse mode makes relative L2 meaningless, and a fine mode
        that never decays makes the problem single-scale.
        """
        assert math.exp(-pde.coarse_decay_rate() * pde.t_max) > 0.5
        assert math.exp(-pde.fine_decay_rate() * pde.t_max) < 0.1

    def test_boundary_condition_is_zero_and_solution_vanishes_on_edges(
        self, pde: MultiscaleHeat2D
    ) -> None:
        t = torch.rand(64, 1)
        edge = torch.zeros(64, 2)
        edge[:, 1] = torch.linspace(0.0, 1.0, 64)
        assert pde.boundary_condition(t, edge).abs().max().item() == 0.0
        assert pde.exact_solution(t, edge).abs().max().item() < 1e-6

    def test_initial_condition_matches_exact_solution_at_t0(
        self, pde: MultiscaleHeat2D
    ) -> None:
        x = torch.rand(128, 2)
        t0 = torch.zeros(128, 1)
        diff = pde.exact_solution(t0, x) - pde.initial_condition(x)
        assert diff.abs().max().item() < 1e-6

    def test_fine_structure_dominates_early_and_fades_late(
        self, pde: MultiscaleHeat2D
    ) -> None:
        """The property the early-dynamics router is being tested against."""
        torch.manual_seed(0)
        x = torch.rand(2048, 2)
        coarse, fine = pde._modes(x)
        def ratio(tv: float) -> float:
            c = (coarse * math.exp(-pde.coarse_decay_rate() * tv)).abs().mean()
            f = (pde.fine_amplitude * fine * math.exp(-pde.fine_decay_rate() * tv)).abs().mean()
            return float(f / c)
        assert ratio(0.0) > 0.4
        assert ratio(1.0) < 0.05

    def test_output_shapes(self, pde: MultiscaleHeat2D) -> None:
        t, x = _coords(32)
        assert pde.exact_solution(t, x).shape == (32,)
        assert pde.initial_condition(x).shape == (32,)
        assert pde.residual(pde.exact_solution(t, x), t, x).reshape(-1).shape == (32,)

    @pytest.mark.parametrize("bad", [1, 0, -2, 2.5])
    def test_rejects_invalid_scale_ratio(self, bad: float) -> None:
        with pytest.raises(ValueError, match="scale_ratio"):
            MultiscaleHeat2D(scale_ratio=bad)
