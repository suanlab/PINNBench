import pytest
import torch

from causal_pinn.physics.pde import ReactionDiffusion1D


def test_residual_shape() -> None:
    pde = ReactionDiffusion1D(diffusion=0.01, reaction_rate=1.0, length=1.0, t_max=1.0)
    batch_size = 16
    t = torch.rand(batch_size, requires_grad=True)
    x = torch.rand(batch_size, requires_grad=True) * pde.length
    u = torch.sin(torch.pi * x / pde.length) * torch.cos(t)

    residual = pde.residual(u, t, x)

    assert residual.shape == (batch_size,)


def test_domain_bounds() -> None:
    pde = ReactionDiffusion1D(diffusion=0.05, length=2.0, t_max=0.5)
    assert pde.domain_bounds == ((0.0, 0.5), (0.0, 2.0))


def test_boundary_condition_values() -> None:
    pde = ReactionDiffusion1D(length=1.0)
    batch_size = 10
    t = torch.rand(batch_size)

    # Left boundary: x = 0 -> u = 1
    x_left = torch.zeros(batch_size)
    bc_left = pde.boundary_condition(t, x_left)
    assert bc_left.shape == (batch_size,)
    assert torch.allclose(bc_left, torch.ones(batch_size))

    # Right boundary: x = L -> u = 0
    x_right = torch.full((batch_size,), pde.length)
    bc_right = pde.boundary_condition(t, x_right)
    assert bc_right.shape == (batch_size,)
    assert torch.allclose(bc_right, torch.zeros(batch_size))


def test_initial_condition_shape() -> None:
    pde = ReactionDiffusion1D(length=2.0)
    x = torch.linspace(0.0, pde.length, 25)

    u0 = pde.initial_condition(x)

    assert u0.shape == (25,)


def test_residual_nonzero_for_random_inputs() -> None:
    pde = ReactionDiffusion1D()
    batch_size = 32
    t = torch.rand(batch_size, requires_grad=True) * 0.5 + 0.1
    x = torch.rand(batch_size, requires_grad=True) * pde.length
    u = torch.sin(2.0 * torch.pi * x / pde.length) * torch.exp(-t)

    residual = pde.residual(u, t, x)

    assert torch.abs(residual).sum() > 0.0


def test_reaction_term_effect() -> None:
    """Compare residuals with r=0 vs r>0 to confirm the reaction term matters."""
    pde_no_reaction = ReactionDiffusion1D(diffusion=0.01, reaction_rate=0.0, length=1.0)
    pde_with_reaction = ReactionDiffusion1D(diffusion=0.01, reaction_rate=1.0, length=1.0)

    batch_size = 24
    t = torch.rand(batch_size, requires_grad=True) * 0.5 + 0.1
    x = torch.rand(batch_size, requires_grad=True) * pde_no_reaction.length
    u = torch.sin(2.0 * torch.pi * x / pde_no_reaction.length) * torch.cos(t)

    residual_no = pde_no_reaction.residual(u, t, x)
    residual_yes = pde_with_reaction.residual(u, t, x)

    assert not torch.allclose(residual_no, residual_yes)


def test_exact_solution_returns_none() -> None:
    pde = ReactionDiffusion1D()
    t = torch.rand(10)
    x = torch.rand(10)
    assert pde.exact_solution(t, x) is None


def test_negative_diffusion_raises() -> None:
    with pytest.raises(ValueError, match="diffusion must be positive"):
        ReactionDiffusion1D(diffusion=-0.01)


def test_zero_diffusion_raises() -> None:
    with pytest.raises(ValueError, match="diffusion must be positive"):
        ReactionDiffusion1D(diffusion=0.0)


def test_negative_reaction_rate_raises() -> None:
    with pytest.raises(ValueError, match="reaction_rate must be non-negative"):
        ReactionDiffusion1D(reaction_rate=-1.0)


def test_initial_condition_monotone_decreasing() -> None:
    """IC should be ~1 near x=0 and ~0 near x=L."""
    pde = ReactionDiffusion1D(length=1.0)
    x = torch.linspace(0.0, pde.length, 100)
    u0 = pde.initial_condition(x)
    assert u0[0] > 0.9
    assert u0[-1] < 0.1
