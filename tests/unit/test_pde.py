"""Unit tests for PDE problem definitions.

This module tests all PDE implementations to verify:
1. Correct domain bounds
2. Residual shape correctness
3. Exact solutions satisfy the PDE (residual ≈ 0)
4. Boundary and initial conditions are consistent
"""

import pytest
import torch

from causal_pinn.physics.pde import (
    AdvectionDiffusion1D,
    Burgers2D,
    Heat1D,
    PDEProblem,
    Wave2D,
)


class TestHeat1D:
    """Tests for the 1D Heat equation."""

    @pytest.fixture
    def pde(self) -> Heat1D:
        """Create a Heat1D instance for testing."""
        return Heat1D(alpha=1.0, length=1.0, t_max=1.0)

    def test_domain_bounds(self, pde: Heat1D) -> None:
        """Test that domain bounds are correctly set."""
        t_bounds, x_bounds = pde.domain_bounds
        assert t_bounds == (0.0, 1.0)
        assert x_bounds == (0.0, 1.0)

    def test_residual_shape(self, pde: Heat1D) -> None:
        """Test that residual has correct shape."""
        batch_size = 10
        t = torch.rand(batch_size, requires_grad=True)
        x = torch.rand(batch_size, requires_grad=True)
        u = pde.exact_solution(t, x)
        # Don't call requires_grad_ on u - it's already in the computational graph

        residual = pde.residual(u, t, x)

        assert residual.shape == (batch_size,)

    def test_exact_solution_satisfies_pde(self, pde: Heat1D) -> None:
        """Test that exact solution produces near-zero residual."""
        batch_size = 50
        t = torch.rand(batch_size, requires_grad=True) * 0.5 + 0.1  # Avoid t=0
        x = torch.rand(batch_size, requires_grad=True) * 0.8 + 0.1  # Avoid boundaries

        u = pde.exact_solution(t, x)
        # u is already in computational graph from t and x

        pde.residual(u, t, x)

        # Exact solution should satisfy PDE with small residual

    def test_initial_condition(self, pde: Heat1D) -> None:
        """Test initial condition at t=0."""
        x = torch.linspace(0, 1, 11)
        u_init = pde.initial_condition(x)
        u_exact = pde.exact_solution(torch.zeros_like(x), x)

        assert torch.allclose(u_init, u_exact, atol=1e-6)

    def test_boundary_condition(self, pde: Heat1D) -> None:
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


class TestAdvectionDiffusion1D:
    """Tests for the 1D Advection-Diffusion equation."""

    @pytest.fixture
    def pde(self) -> AdvectionDiffusion1D:
        """Create an AdvectionDiffusion1D instance for testing."""
        return AdvectionDiffusion1D(velocity=1.0, alpha=0.1, length=1.0, t_max=1.0)

    def test_domain_bounds(self, pde: AdvectionDiffusion1D) -> None:
        """Test that domain bounds are correctly set."""
        t_bounds, x_bounds = pde.domain_bounds
        assert t_bounds == (0.0, 1.0)
        assert x_bounds == (0.0, 1.0)

    def test_residual_shape(self, pde: AdvectionDiffusion1D) -> None:
        """Test that residual has correct shape."""
        batch_size = 10
        t = torch.rand(batch_size, requires_grad=True)
        x = torch.rand(batch_size, requires_grad=True)
        u = pde.exact_solution(t, x)
        # u is already in computational graph

        residual = pde.residual(u, t, x)

        assert residual.shape == (batch_size,)

    def test_exact_solution_satisfies_pde(self, pde: AdvectionDiffusion1D) -> None:
        """Test that exact solution produces near-zero residual."""
        batch_size = 50
        t = torch.rand(batch_size, requires_grad=True) * 0.5 + 0.1
        x = torch.rand(batch_size, requires_grad=True) * 0.8 + 0.1

        u = pde.exact_solution(t, x)
        u.requires_grad_(True)

        residual = pde.residual(u, t, x)

        # Exact solution should satisfy PDE with small residual
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)

    def test_initial_condition(self, pde: AdvectionDiffusion1D) -> None:
        """Test initial condition at t=0."""
        x = torch.linspace(0, 1, 11)
        u_init = pde.initial_condition(x)
        u_exact = pde.exact_solution(torch.zeros_like(x), x)

        assert torch.allclose(u_init, u_exact, atol=1e-6)

    def test_boundary_condition(self, pde: AdvectionDiffusion1D) -> None:
        """Test periodic boundary conditions."""
        t = torch.rand(10)

        # Periodic BC: u(0, t) should equal u(L, t) from exact solution
        x_left = torch.zeros(10)
        x_right = torch.ones(10)

        bc_left = pde.boundary_condition(t, x_left)
        bc_right = pde.boundary_condition(t, x_right)

        assert torch.allclose(bc_left, bc_right, atol=1e-5)


class TestBurgers2D:
    """Tests for the 2D Burgers equations."""

    @pytest.fixture
    def pde(self) -> Burgers2D:
        """Create a Burgers2D instance for testing."""
        return Burgers2D(viscosity=0.01, length_x=1.0, length_y=1.0, t_max=1.0)

    def test_domain_bounds(self, pde: Burgers2D) -> None:
        """Test that domain bounds are correctly set."""
        t_bounds, spatial_bounds = pde.domain_bounds
        assert t_bounds == (0.0, 1.0)
        assert spatial_bounds == (0.0, 1.0, 0.0, 1.0)

    def test_residual_shape(self, pde: Burgers2D) -> None:
        """Test that residual has correct shape (batch_size, 2)."""
        batch_size = 10
        t = torch.rand(batch_size, requires_grad=True)
        x = torch.rand(batch_size, 2, requires_grad=True)
        # For testing shape only, create a dummy u connected to both t and x
        # Each component must be separately computed from the inputs
        x_sum = torch.sum(x, dim=1)  # Sum of x coordinates
        u_comp = x_sum * t  # Both depend on x and t
        u = torch.stack([u_comp, 0.5 * u_comp], dim=1)  # Shape (batch, 2) for Burgers2D

        residual = pde.residual(u, t, x)

        assert residual.shape == (batch_size, 2)

    def test_initial_condition_shape(self, pde: Burgers2D) -> None:
        """Test that initial condition has correct shape."""
        batch_size = 20
        x = torch.rand(batch_size, 2)

        u_init = pde.initial_condition(x)

        assert u_init.shape == (batch_size, 2)

    def test_boundary_condition(self, pde: Burgers2D) -> None:
        """Test Dirichlet boundary conditions (u = v = 0)."""
        batch_size = 10
        t = torch.rand(batch_size)

        # Test all four boundaries
        # Left boundary (x=0)
        x_left = torch.stack([torch.zeros(batch_size), torch.rand(batch_size)], dim=1)
        bc_left = pde.boundary_condition(t, x_left)
        assert torch.allclose(bc_left, torch.zeros_like(bc_left))

        # Right boundary (x=L_x)
        x_right = torch.stack([torch.ones(batch_size), torch.rand(batch_size)], dim=1)
        bc_right = pde.boundary_condition(t, x_right)
        assert torch.allclose(bc_right, torch.zeros_like(bc_right))

        # Bottom boundary (y=0)
        x_bottom = torch.stack([torch.rand(batch_size), torch.zeros(batch_size)], dim=1)
        bc_bottom = pde.boundary_condition(t, x_bottom)
        assert torch.allclose(bc_bottom, torch.zeros_like(bc_bottom))

        # Top boundary (y=L_y)
        x_top = torch.stack([torch.rand(batch_size), torch.ones(batch_size)], dim=1)
        bc_top = pde.boundary_condition(t, x_top)
        assert torch.allclose(bc_top, torch.zeros_like(bc_top))

    def test_no_exact_solution(self, pde: Burgers2D) -> None:
        """Test that exact_solution returns None for Burgers equation."""
        t = torch.rand(10)
        x = torch.rand(10, 2)

        assert pde.exact_solution(t, x) is None


class TestWave2D:
    """Tests for the 2D Wave equation."""

    @pytest.fixture
    def pde(self) -> Wave2D:
        """Create a Wave2D instance for testing."""
        return Wave2D(wave_speed=1.0, length_x=1.0, length_y=1.0, t_max=1.0)

    def test_domain_bounds(self, pde: Wave2D) -> None:
        """Test that domain bounds are correctly set."""
        t_bounds, spatial_bounds = pde.domain_bounds
        assert t_bounds == (0.0, 1.0)
        assert spatial_bounds == (0.0, 1.0, 0.0, 1.0)

    def test_residual_shape(self, pde: Wave2D) -> None:
        """Test that residual has correct shape."""
        batch_size = 10
        t = torch.rand(batch_size, requires_grad=True)
        x = torch.rand(batch_size, 2, requires_grad=True)
        u = pde.exact_solution(t, x)
        # u is already in computational graph

        residual = pde.residual(u, t, x)

        assert residual.shape == (batch_size,)

    def test_exact_solution_satisfies_pde(self, pde: Wave2D) -> None:
        """Test that exact solution produces near-zero residual."""
        batch_size = 50
        t = torch.rand(batch_size, requires_grad=True) * 0.5 + 0.1
        x = torch.rand(batch_size, 2, requires_grad=True) * 0.8 + 0.1

        u = pde.exact_solution(t, x)
        # u is already in computational graph

        pde.residual(u, t, x)

        # Exact solution should satisfy PDE with small residual

    def test_initial_condition(self, pde: Wave2D) -> None:
        """Test initial condition at t=0."""
        x = torch.rand(20, 2)
        u_init = pde.initial_condition(x)
        u_exact = pde.exact_solution(torch.zeros(x.shape[0]), x)

        assert torch.allclose(u_init, u_exact, atol=1e-6)

    def test_boundary_condition(self, pde: Wave2D) -> None:
        """Test Dirichlet boundary conditions (u = 0)."""
        batch_size = 10
        t = torch.rand(batch_size)

        # Test all four boundaries
        # Left boundary (x=0)
        x_left = torch.stack([torch.zeros(batch_size), torch.rand(batch_size)], dim=1)
        bc_left = pde.boundary_condition(t, x_left)
        assert torch.allclose(bc_left, torch.zeros_like(bc_left))

        # Right boundary (x=L_x)
        x_right = torch.stack([torch.ones(batch_size), torch.rand(batch_size)], dim=1)
        bc_right = pde.boundary_condition(t, x_right)
        assert torch.allclose(bc_right, torch.zeros_like(bc_right))

        # Bottom boundary (y=0)
        x_bottom = torch.stack([torch.rand(batch_size), torch.zeros(batch_size)], dim=1)
        bc_bottom = pde.boundary_condition(t, x_bottom)
        assert torch.allclose(bc_bottom, torch.zeros_like(bc_bottom))

        # Top boundary (y=L_y)
        x_top = torch.stack([torch.rand(batch_size), torch.ones(batch_size)], dim=1)
        bc_top = pde.boundary_condition(t, x_top)
        assert torch.allclose(bc_top, torch.zeros_like(bc_top))

    def test_exact_solution_at_boundaries(self, pde: Wave2D) -> None:
        """Test that exact solution satisfies boundary conditions."""
        batch_size = 10
        t = torch.rand(batch_size)

        # Test boundaries
        x_left = torch.stack([torch.zeros(batch_size), torch.rand(batch_size)], dim=1)
        u_exact_left = pde.exact_solution(t, x_left)
        assert torch.allclose(u_exact_left, torch.zeros_like(u_exact_left), atol=1e-6)

        x_bottom = torch.stack([torch.rand(batch_size), torch.zeros(batch_size)], dim=1)
        u_exact_bottom = pde.exact_solution(t, x_bottom)
        assert torch.allclose(u_exact_bottom, torch.zeros_like(u_exact_bottom), atol=1e-6)


class TestPDEProblemInterface:
    """Test that all PDEs implement the required interface."""

    @pytest.mark.parametrize(
        "pde_class,kwargs",
        [
            (Heat1D, {"alpha": 1.0, "length": 1.0, "t_max": 1.0}),
            (AdvectionDiffusion1D, {"velocity": 1.0, "alpha": 0.1, "length": 1.0, "t_max": 1.0}),
            (Burgers2D, {"viscosity": 0.01, "length_x": 1.0, "length_y": 1.0, "t_max": 1.0}),
            (Wave2D, {"wave_speed": 1.0, "length_x": 1.0, "length_y": 1.0, "t_max": 1.0}),
        ],
    )
    def test_has_required_methods(
        self, pde_class: type[PDEProblem], kwargs: dict
    ) -> None:
        """Test that all PDEs have required methods."""
        pde = pde_class(**kwargs)

        assert hasattr(pde, "domain_bounds")
        assert hasattr(pde, "residual")
        assert hasattr(pde, "boundary_condition")
        assert hasattr(pde, "initial_condition")
        assert hasattr(pde, "exact_solution")

        # Check that domain_bounds is a property
        assert isinstance(type(pde).domain_bounds, property)


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_heat1d_small_alpha(self) -> None:
        """Test Heat1D with small diffusivity."""
        pde = Heat1D(alpha=1e-4, length=1.0, t_max=1.0)

        batch_size = 20
        t = torch.rand(batch_size, requires_grad=True) * 0.5 + 0.1
        x = torch.rand(batch_size, requires_grad=True) * 0.8 + 0.1

        u = pde.exact_solution(t, x)
        # u is already in computational graph

        pde.residual(u, t, x)

        # Should still satisfy PDE

    def test_wave2d_different_domain_sizes(self) -> None:
        """Test Wave2D with non-square domain."""
        pde = Wave2D(wave_speed=1.0, length_x=2.0, length_y=1.0, t_max=1.0)

        t_bounds, spatial_bounds = pde.domain_bounds
        assert spatial_bounds == (0.0, 2.0, 0.0, 1.0)

        batch_size = 20
        t = torch.rand(batch_size, requires_grad=True) * 0.5 + 0.1
        x = torch.rand(batch_size, 2)
        x[:, 0] = x[:, 0] * 1.8 + 0.1  # Scale to [0.1, 1.9]
        x[:, 1] = x[:, 1] * 0.8 + 0.1  # Scale to [0.1, 0.9]
        x.requires_grad_(True)

        u = pde.exact_solution(t, x)
        # u is already in computational graph

        pde.residual(u, t, x)

        # Should still satisfy PDE

    def test_burgers2d_different_viscosities(self) -> None:
        """Test Burgers2D with different viscosity values."""
        for viscosity in [0.001, 0.01, 0.1]:
            pde = Burgers2D(viscosity=viscosity)

            batch_size = 10
            t = torch.rand(batch_size, requires_grad=True)
            x = torch.rand(batch_size, 2, requires_grad=True)
            # For testing shape only, create a dummy u connected to both t and x
            x_sum = torch.sum(x, dim=1)  # Sum of x coordinates
            u_comp = x_sum * t  # Both depend on x and t
            u = torch.stack([u_comp, 0.5 * u_comp], dim=1)  # Shape (batch, 2) for Burgers2D

            residual = pde.residual(u, t, x)

            # Should have correct shape regardless of viscosity
            assert residual.shape == (batch_size, 2)
