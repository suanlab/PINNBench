"""Tests for the Fisher-KPP travelling-wave reference solution."""

import torch

from causal_pinn.physics.pde import ReactionDiffusion1D, ReactionDiffusionExact1D


def _coords(n: int = 300) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return (
        torch.rand(n, 1, requires_grad=True),
        torch.rand(n, 1, requires_grad=True),
    )


class TestReactionDiffusionExact1D:
    def test_satisfies_the_pde(self) -> None:
        pde = ReactionDiffusionExact1D(diffusion=0.01, reaction_rate=1.0)
        t, x = _coords()
        r = pde.residual(pde.exact_solution(t, x), t, x)
        assert r.abs().max().item() < 1e-5

    def test_wrong_wave_speed_leaves_a_residual(self) -> None:
        """The closed form holds only at c = 5 sqrt(rD/6)."""
        pde = ReactionDiffusionExact1D(diffusion=0.01, reaction_rate=1.0)
        t, x = _coords()
        xi = pde.steepness * (x.reshape(-1) - 1.5 * pde.wave_speed * t.reshape(-1))
        wrong = (1.0 + torch.exp(xi)) ** (-2)
        assert pde.residual(wrong, t, x).abs().mean().item() > 1e-3

    def test_wrong_exponent_leaves_a_residual(self) -> None:
        pde = ReactionDiffusionExact1D(diffusion=0.01, reaction_rate=1.0)
        t, x = _coords()
        xi = pde.steepness * (x.reshape(-1) - pde.wave_speed * t.reshape(-1))
        wrong = (1.0 + torch.exp(xi)) ** (-1)
        assert pde.residual(wrong, t, x).abs().mean().item() > 1e-3

    def test_speed_and_steepness_match_theory(self) -> None:
        import math

        d, r = 0.02, 3.0
        pde = ReactionDiffusionExact1D(diffusion=d, reaction_rate=r)
        assert abs(pde.wave_speed - 5.0 * math.sqrt(r * d / 6.0)) < 1e-12
        assert abs(pde.steepness - math.sqrt(r / (6.0 * d))) < 1e-12

    def test_profile_is_monotone_and_bounded(self) -> None:
        pde = ReactionDiffusionExact1D(diffusion=0.01, reaction_rate=1.0)
        x = torch.linspace(-1.0, 2.0, 200).reshape(-1, 1)
        u = pde.exact_solution(torch.zeros_like(x), x)
        assert torch.all(u[1:] <= u[:-1] + 1e-12)  # decreasing in x
        assert u.min().item() >= 0.0 and u.max().item() <= 1.0

    def test_initial_condition_matches_solution_at_t0(self) -> None:
        pde = ReactionDiffusionExact1D(diffusion=0.01, reaction_rate=1.0)
        x = torch.rand(64, 1)
        torch.testing.assert_close(
            pde.initial_condition(x), pde.exact_solution(torch.zeros(64, 1), x)
        )

    def test_base_class_still_has_no_exact_solution(self) -> None:
        assert ReactionDiffusion1D().exact_solution(torch.rand(4), torch.rand(4)) is None
