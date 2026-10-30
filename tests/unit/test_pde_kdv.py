import torch

from causal_pinn.physics.pde import KdV1D


def test_residual_shape() -> None:
    pde = KdV1D(dispersion=1.0, length=2.0, t_max=0.5)
    batch_size = 16
    t = torch.rand(batch_size, requires_grad=True)
    x = torch.rand(batch_size, requires_grad=True) * pde.length
    u = torch.sin(torch.pi * x / pde.length) * torch.cos(t)

    residual = pde.residual(u, t, x)

    assert residual.shape == (batch_size,)


def test_boundary_condition_shape_and_values() -> None:
    pde = KdV1D()
    batch_size = 20
    t = torch.rand(batch_size)
    x = torch.cat([torch.zeros(batch_size // 2), torch.full((batch_size // 2,), pde.length)])

    bc = pde.boundary_condition(t, x)

    assert bc.shape == (batch_size,)
    assert torch.allclose(bc, torch.zeros_like(bc))


def test_initial_condition_shape() -> None:
    pde = KdV1D(length=2.0)
    x = torch.linspace(0.0, pde.length, 25)

    u0 = pde.initial_condition(x)

    assert u0.shape == (25,)


def test_domain_bounds() -> None:
    pde = KdV1D(dispersion=1.0, length=2.0, t_max=0.5)
    assert pde.domain_bounds == ((0.0, 0.5), (0.0, 2.0))


def test_residual_nonzero_for_random_inputs() -> None:
    pde = KdV1D()
    batch_size = 32
    t = torch.rand(batch_size, requires_grad=True) * 0.5 + 0.1
    x = torch.rand(batch_size, requires_grad=True) * pde.length
    u = torch.sin(2.0 * torch.pi * x / pde.length) * torch.exp(-t)

    residual = pde.residual(u, t, x)

    assert torch.abs(residual).sum() > 0.0


def test_third_derivative_contribution_changes_with_dispersion() -> None:
    pde_no_dispersion = KdV1D(dispersion=0.01, length=2.0, t_max=0.5)
    pde_with_dispersion = KdV1D(dispersion=1.0, length=2.0, t_max=0.5)

    batch_size = 24
    t = torch.rand(batch_size, requires_grad=True) * 0.5 + 0.1
    x = torch.rand(batch_size, requires_grad=True) * pde_no_dispersion.length
    u = torch.sin(2.0 * torch.pi * x / pde_no_dispersion.length) * torch.cos(t)

    residual_no = pde_no_dispersion.residual(u, t, x)
    residual_yes = pde_with_dispersion.residual(u, t, x)

    assert not torch.allclose(residual_no, residual_yes)


def test_exact_solution_returns_none() -> None:
    pde = KdV1D()
    t = torch.rand(10)
    x = torch.rand(10)
    assert pde.exact_solution(t, x) is None


# --------------- Input validation tests ---------------


def test_negative_soliton_amplitude_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="soliton_amplitude must be positive"):
        KdV1D(soliton_amplitude=-1.0)


def test_zero_soliton_amplitude_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="soliton_amplitude must be positive"):
        KdV1D(soliton_amplitude=0.0)


def test_negative_dispersion_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="dispersion must be positive"):
        KdV1D(dispersion=-1.0)


# --------------- Soliton mode tests ---------------


def _soliton_pde() -> KdV1D:
    return KdV1D(
        dispersion=1.0,
        length=20.0,
        t_max=0.5,
        soliton_amplitude=3.0,
        soliton_center=10.0,
    )


def test_soliton_exact_solution_shape() -> None:
    pde = _soliton_pde()
    n = 32
    t = torch.rand(n) * pde.t_max
    x = torch.rand(n) * pde.length
    u = pde.exact_solution(t, x)
    assert u is not None
    assert u.shape == (n,)


def test_soliton_initial_condition_matches_exact() -> None:
    pde = _soliton_pde()
    x = torch.linspace(0.0, pde.length, 64)
    t_zero = torch.zeros_like(x)
    u_exact = pde.exact_solution(t_zero, x)
    u_ic = pde.initial_condition(x)
    assert u_exact is not None
    assert torch.allclose(u_ic, u_exact, atol=1e-7)


def test_soliton_boundary_decay() -> None:
    pde = _soliton_pde()
    n = 50
    t = torch.linspace(0.0, pde.t_max, n)
    x_left = torch.zeros(n)
    x_right = torch.full((n,), pde.length)
    u_left = pde.exact_solution(t, x_left)
    u_right = pde.exact_solution(t, x_right)
    assert u_left is not None and u_right is not None
    assert u_left.abs().max() < 1e-3
    assert u_right.abs().max() < 1e-3


def test_soliton_exact_solution_satisfies_pde() -> None:
    """Verify autograd PDE residual of the soliton is near zero."""
    pde = _soliton_pde()
    n = 64
    t = (torch.rand(n) * 0.4 + 0.05).requires_grad_(True)
    x = (torch.rand(n) * 18.0 + 1.0).requires_grad_(True)

    # Build a differentiable u from the analytical formula
    eta = pde.soliton_kappa * (x - pde.soliton_speed * t - pde.soliton_center)
    u = pde.soliton_amplitude * torch.pow(1.0 / torch.cosh(eta), 2)

    residual = pde.residual(u, t, x)
    assert residual.abs().max() < 1e-4
