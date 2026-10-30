import torch

from causal_pinn.physics.pde import AllenCahn1D


def test_residual_shape() -> None:
    pde = AllenCahn1D(epsilon=0.1, length=1.0, t_max=1.0)
    batch_size = 16
    t = torch.rand(batch_size, requires_grad=True)
    x = torch.rand(batch_size, requires_grad=True)
    u = torch.sin(torch.pi * x) * torch.cos(t)

    residual = pde.residual(u, t, x)

    assert residual.shape == (batch_size,)


def test_boundary_condition_shape_and_values() -> None:
    pde = AllenCahn1D()
    batch_size = 20
    t = torch.rand(batch_size)
    x = torch.cat([torch.zeros(batch_size // 2), torch.ones(batch_size // 2) * pde.length])

    bc = pde.boundary_condition(t, x)

    assert bc.shape == (batch_size,)
    assert torch.allclose(bc, torch.zeros_like(bc))


def test_initial_condition_shape() -> None:
    pde = AllenCahn1D(length=1.0)
    x = torch.linspace(0.0, pde.length, 25)

    u0 = pde.initial_condition(x)

    assert u0.shape == (25,)


def test_domain_bounds() -> None:
    pde = AllenCahn1D(epsilon=0.1, length=1.0, t_max=1.0)
    assert pde.domain_bounds == ((0.0, 1.0), (0.0, 1.0))


def test_residual_nonzero_for_random_inputs() -> None:
    pde = AllenCahn1D()
    batch_size = 32
    t = torch.rand(batch_size, requires_grad=True) * 0.8 + 0.1
    x = torch.rand(batch_size, requires_grad=True) * 0.8 + 0.1
    u = torch.sin(torch.pi * x) * torch.exp(-t)

    residual = pde.residual(u, t, x)

    assert torch.abs(residual).sum() > 0.0


def test_residual_includes_nonlinear_term() -> None:
    pde = AllenCahn1D(epsilon=0.1)
    batch_size = 12
    t = torch.rand(batch_size, requires_grad=True)
    x = torch.rand(batch_size, requires_grad=True)

    perturbation = 1e-6 * (t + x**2)
    u_zero = perturbation
    u_one = 1.0 + perturbation
    u_half = 0.5 + perturbation

    residual_zero = pde.residual(u_zero, t, x)
    residual_one = pde.residual(u_one, t, x)
    residual_half = pde.residual(u_half, t, x)

    assert torch.allclose(residual_zero, torch.zeros_like(residual_zero), atol=1e-5)
    assert torch.allclose(residual_one, torch.zeros_like(residual_one), atol=1e-5)
    assert torch.allclose(residual_half, torch.full_like(residual_half, -0.375), atol=1e-4)


def test_zero_epsilon_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="epsilon must be positive"):
        AllenCahn1D(epsilon=0.0)


def test_negative_epsilon_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="epsilon must be positive"):
        AllenCahn1D(epsilon=-0.1)


def test_exact_solution_returns_none() -> None:
    pde = AllenCahn1D()
    t = torch.rand(10)
    x = torch.rand(10)
    assert pde.exact_solution(t, x) is None


# --------------- Front mode tests ---------------


def _front_pde() -> AllenCahn1D:
    return AllenCahn1D(
        epsilon=0.01,
        length=2.0,
        t_max=1.0,
        front_center=1.0,
    )


def test_front_exact_solution_shape() -> None:
    pde = _front_pde()
    n = 32
    t = torch.rand(n) * pde.t_max
    x = torch.rand(n) * pde.length
    u = pde.exact_solution(t, x)
    assert u is not None
    assert u.shape == (n,)


def test_front_initial_condition_matches_exact() -> None:
    pde = _front_pde()
    x = torch.linspace(0.0, pde.length, 64)
    t_zero = torch.zeros_like(x)
    u_exact = pde.exact_solution(t_zero, x)
    u_ic = pde.initial_condition(x)
    assert u_exact is not None
    assert torch.allclose(u_ic, u_exact, atol=1e-7)


def test_front_exact_solution_satisfies_pde() -> None:
    """Verify autograd PDE residual of the stationary front is near zero."""
    pde = _front_pde()
    n = 64
    t = (torch.rand(n) * 0.8 + 0.1).requires_grad_(True)
    # Avoid domain edges where tanh is flat and gradients vanish
    x = (torch.rand(n) * 1.6 + 0.2).requires_grad_(True)

    # Build differentiable u from the analytical formula.
    # Add 0*t so autograd can compute du/dt (which should be zero).
    import math

    inv_w = 1.0 / (math.sqrt(2.0) * pde.epsilon)
    u = -torch.tanh((x - pde.front_center) * inv_w) + 0.0 * t

    residual = pde.residual(u, t, x)
    assert residual.abs().max() < 1e-4


def test_front_boundary_values() -> None:
    """Front-mode BCs should approach ±1 for small epsilon."""
    pde = _front_pde()
    n = 10
    t = torch.rand(n)
    x_left = torch.zeros(n)
    x_right = torch.full((n,), pde.length)
    bc_left = pde.boundary_condition(t, x_left)
    bc_right = pde.boundary_condition(t, x_right)
    # With eps=0.01, center=1.0: u(0) ≈ +1, u(2) ≈ -1
    assert torch.allclose(bc_left, torch.ones(n), atol=1e-4)
    assert torch.allclose(bc_right, -torch.ones(n), atol=1e-4)
