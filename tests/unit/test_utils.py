"""Unit tests for utility modules."""

from __future__ import annotations

import torch

from causal_pinn.utils.gradients import gradient, jacobian_column
from causal_pinn.utils.random import set_seed


def test_set_seed_deterministic() -> None:
    """Verify that set_seed produces deterministic results."""
    set_seed(123)
    a = torch.randn(5)
    set_seed(123)
    b = torch.randn(5)
    assert torch.allclose(a, b), "set_seed should produce identical tensors"


def test_gradient_first_order() -> None:
    """Verify first-order gradient computation: d(x^2)/dx = 2x."""
    x = torch.linspace(0.0, 1.0, 50, requires_grad=True).unsqueeze(-1)
    y = x**2
    dy_dx = gradient(y, x, order=1)
    expected = 2.0 * x
    assert torch.allclose(dy_dx, expected, atol=1e-5), "d(x^2)/dx should be 2x"


def test_gradient_second_order() -> None:
    """Verify second-order gradient: d^2(x^3)/dx^2 = 6x."""
    x = torch.linspace(0.1, 1.0, 50, requires_grad=True).unsqueeze(-1)
    y = x**3
    d2y_dx2 = gradient(y, x, order=2)
    expected = 6.0 * x
    assert torch.allclose(d2y_dx2, expected, atol=1e-4), "d^2(x^3)/dx^2 should be 6x"


def test_jacobian_column() -> None:
    """Verify partial derivative w.r.t. specific column."""
    # f(t, x) = t*x, df/dt = x, df/dx = t
    tx = torch.rand(30, 2, requires_grad=True)
    t, x = tx[:, 0:1], tx[:, 1:2]
    f = t * x
    df_dt = jacobian_column(f, tx, column=0)
    df_dx = jacobian_column(f, tx, column=1)
    assert torch.allclose(df_dt, x, atol=1e-5), "df/dt should be x"
    assert torch.allclose(df_dx, t, atol=1e-5), "df/dx should be t"
