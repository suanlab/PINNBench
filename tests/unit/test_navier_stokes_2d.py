import math

import torch

from causal_pinn.physics.pde import NavierStokes2D


def test_residual_shape() -> None:
    pde = NavierStokes2D(viscosity=0.1)
    batch_size = 16
    t = torch.rand(batch_size, requires_grad=True)
    x = torch.rand(batch_size, 2, requires_grad=True) * 2.0 * math.pi

    # Build a differentiable vorticity field
    x_coord = x[:, 0]
    y_coord = x[:, 1]
    omega = torch.sin(x_coord) * torch.cos(y_coord) * torch.exp(-t)

    residual = pde.residual(omega, t, x)

    assert residual.shape == (batch_size,)


def test_domain_bounds() -> None:
    pde = NavierStokes2D(viscosity=0.1, length_x=2.0 * math.pi, length_y=2.0 * math.pi, t_max=1.0)
    t_bounds, x_bounds = pde.domain_bounds
    assert t_bounds == (0.0, 1.0)
    assert len(x_bounds) == 4
    assert x_bounds[0] == 0.0
    assert abs(x_bounds[1] - 2.0 * math.pi) < 1e-10
    assert x_bounds[2] == 0.0
    assert abs(x_bounds[3] - 2.0 * math.pi) < 1e-10


def test_exact_solution_shape() -> None:
    pde = NavierStokes2D()
    batch_size = 32
    t = torch.rand(batch_size)
    x = torch.rand(batch_size, 2) * 2.0 * math.pi

    omega = pde.exact_solution(t, x)

    assert omega.shape == (batch_size,)


def test_exact_solution_satisfies_pde() -> None:
    """Verify that the exact vorticity satisfies the transport equation."""
    pde = NavierStokes2D(viscosity=0.1)
    nu = pde.viscosity
    batch_size = 64

    t = (torch.rand(batch_size) * 0.8 + 0.1).requires_grad_(True)
    x = (torch.rand(batch_size, 2) * 2.0 * math.pi).requires_grad_(True)

    # Build differentiable exact vorticity: omega = -2 cos(x) cos(y) exp(-2 nu t)
    x_coord = x[:, 0]
    y_coord = x[:, 1]
    omega = -2.0 * torch.cos(x_coord) * torch.cos(y_coord) * torch.exp(-2.0 * nu * t)

    residual = pde.residual(omega, t, x)

    assert residual.abs().max() < 1e-4, f"Max residual: {residual.abs().max().item()}"


def test_initial_condition_matches_exact() -> None:
    pde = NavierStokes2D(viscosity=0.1)
    batch_size = 50
    x = torch.rand(batch_size, 2) * 2.0 * math.pi
    t_zero = torch.zeros(batch_size)

    ic = pde.initial_condition(x)
    exact_at_zero = pde.exact_solution(t_zero, x)

    assert torch.allclose(ic, exact_at_zero, atol=1e-7)


def test_boundary_condition_equals_exact() -> None:
    pde = NavierStokes2D(viscosity=0.1)
    batch_size = 20
    t = torch.rand(batch_size)
    x = torch.rand(batch_size, 2) * 2.0 * math.pi

    bc = pde.boundary_condition(t, x)
    exact = pde.exact_solution(t, x)

    assert torch.allclose(bc, exact, atol=1e-7)


def test_negative_viscosity_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="viscosity must be positive"):
        NavierStokes2D(viscosity=-0.1)


def test_zero_viscosity_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="viscosity must be positive"):
        NavierStokes2D(viscosity=0.0)


def test_exact_velocity_shape() -> None:
    pde = NavierStokes2D()
    batch_size = 16
    t = torch.rand(batch_size)
    x = torch.rand(batch_size, 2) * 2.0 * math.pi

    vel_x, vel_y = pde.exact_velocity(t, x)

    assert vel_x.shape == (batch_size,)
    assert vel_y.shape == (batch_size,)


def test_exact_pressure_shape() -> None:
    pde = NavierStokes2D()
    batch_size = 16
    t = torch.rand(batch_size)
    x = torch.rand(batch_size, 2) * 2.0 * math.pi

    p = pde.exact_pressure(t, x)

    assert p.shape == (batch_size,)
