"""Unit tests for Burgers1D PDE definition.

This module tests the Burgers1D implementation to verify:
1. Correct domain bounds
2. Residual shape correctness
3. Initial and boundary conditions
4. No exact solution returns None
"""

import pytest
import torch

from causal_pinn.physics.pde import Burgers1D


class TestBurgers1D:
    """Tests for the 1D Burgers equation."""

    @pytest.fixture
    def pde(self) -> Burgers1D:
        """Create a Burgers1D instance for testing."""
        return Burgers1D(viscosity=0.01, length=1.0, t_max=1.0)

    def test_domain_bounds(self, pde: Burgers1D) -> None:
        """Test that domain bounds are correctly set."""
        t_bounds, x_bounds = pde.domain_bounds
        assert t_bounds == (0.0, 1.0)
        assert x_bounds == (0.0, 1.0)

    def test_residual_shape(self, pde: Burgers1D) -> None:
        """Test that residual has correct shape."""
        batch_size = 10
        t = torch.rand(batch_size, requires_grad=True)
        x = torch.rand(batch_size, requires_grad=True)

        # Create a u that has non-trivial u_xx (must be nonlinear in x)
        u = torch.sin(x) * t

        residual = pde.residual(u, t, x)

        assert residual.shape == (batch_size,)

    def test_residual_values_nonzero(self, pde: Burgers1D) -> None:
        """Test that residual computation produces non-trivial values."""
        batch_size = 20
        t = torch.rand(batch_size, requires_grad=True) * 0.5 + 0.1
        x = torch.rand(batch_size, requires_grad=True) * 0.8 + 0.1

        # Create a simple test function u = sin(x) * cos(t)
        u = torch.sin(x) * torch.cos(t)

        residual = pde.residual(u, t, x)

        # For this arbitrary u, residual should be non-zero
        # (Burgers equation is not satisfied by this u)
        assert torch.abs(residual).sum() > 0.0

    def test_initial_condition(self, pde: Burgers1D) -> None:
        """Test initial condition at t=0: u(x, 0) = -sin(πx/L)."""
        x = torch.linspace(0, 1, 11)
        u_init = pde.initial_condition(x)

        # Expected: -sin(πx)
        expected = -torch.sin(torch.pi * x)

        assert torch.allclose(u_init, expected, atol=1e-6)

    def test_initial_condition_shape(self, pde: Burgers1D) -> None:
        """Test that initial condition has correct shape."""
        batch_size = 20
        x = torch.rand(batch_size)

        u_init = pde.initial_condition(x)

        assert u_init.shape == (batch_size,)

    def test_initial_condition_at_boundaries(self, pde: Burgers1D) -> None:
        """Test that initial condition is zero at boundaries."""
        x = torch.tensor([0.0, 1.0])
        u_init = pde.initial_condition(x)

        # u(0, 0) = -sin(0) = 0
        # u(L, 0) = -sin(π) = 0
        assert torch.allclose(u_init, torch.zeros_like(u_init), atol=1e-6)

    def test_boundary_condition(self, pde: Burgers1D) -> None:
        """Test boundary conditions at x=0 and x=L."""
        t = torch.rand(10)

        # Left boundary (x=0)
        x_left = torch.zeros(10)
        bc_left = pde.boundary_condition(t, x_left)
        assert torch.allclose(bc_left, torch.zeros_like(bc_left))

        # Right boundary (x=L)
        x_right = torch.ones(10)
        bc_right = pde.boundary_condition(t, x_right)
        assert torch.allclose(bc_right, torch.zeros_like(bc_right))

    def test_no_exact_solution(self, pde: Burgers1D) -> None:
        """Test that exact_solution returns None for Burgers equation."""
        t = torch.rand(10)
        x = torch.rand(10)

        assert pde.exact_solution(t, x) is None


class TestBurgers1DWithNetwork:
    """Tests using a simple neural network for more realistic residual testing."""

    @pytest.fixture
    def pde(self) -> Burgers1D:
        """Create a Burgers1D instance for testing."""
        return Burgers1D(viscosity=0.01, length=1.0, t_max=1.0)

    @pytest.fixture
    def simple_network(self) -> torch.nn.Module:
        """Create a simple network for testing."""
        return torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1),
        )

    def test_residual_with_network(self, pde: Burgers1D, simple_network: torch.nn.Module) -> None:
        """Test residual computation with a network output."""
        batch_size = 50
        t = torch.rand(batch_size, requires_grad=True)
        x = torch.rand(batch_size, requires_grad=True)

        # Network expects (batch, 2) input
        inputs = torch.stack([t, x], dim=1)
        u = simple_network(inputs).squeeze(-1)

        residual = pde.residual(u, t, x)

        assert residual.shape == (batch_size,)
        assert not torch.isnan(residual).any()
        assert not torch.isinf(residual).any()


class TestBurgers1DVaryingParameters:
    """Tests with different parameter values."""

    def test_different_viscosities(self) -> None:
        """Test Burgers1D with different viscosity values."""
        for viscosity in [0.001, 0.01, 0.1, 1.0]:
            pde = Burgers1D(viscosity=viscosity)

            assert pde.viscosity == viscosity

            # Test residual still works with a function that has u_xx
            batch_size = 10
            t = torch.rand(batch_size, requires_grad=True)
            x = torch.rand(batch_size, requires_grad=True)
            u = torch.sin(x) * torch.cos(t)  # Has non-trivial u_xx

            residual = pde.residual(u, t, x)
            assert residual.shape == (batch_size,)

    def test_different_domain_sizes(self) -> None:
        """Test Burgers1D with different domain sizes."""
        pde = Burgers1D(viscosity=0.01, length=2.0, t_max=5.0)

        t_bounds, x_bounds = pde.domain_bounds
        assert t_bounds == (0.0, 5.0)
        assert x_bounds == (0.0, 2.0)

        # Initial condition should scale with length
        x = torch.tensor([0.5, 1.0, 1.5])
        u_init = pde.initial_condition(x)

        # Expected: -sin(πx/2)
        expected = -torch.sin(torch.pi * x / 2.0)
        assert torch.allclose(u_init, expected, atol=1e-6)
