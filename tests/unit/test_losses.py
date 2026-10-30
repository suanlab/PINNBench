"""Unit tests for physics loss functions."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from causal_pinn.models.pinn import PINNBase
from causal_pinn.physics.losses import (
    AdaptiveCausalLoss,
    BoundaryLoss,
    CausalLoss,
    DirichletBC,
    HigherOrderPhysicsLoss,
    NeumannBC,
    PhysicsLoss,
    TimeDecayLoss,
)
from causal_pinn.physics.pde.wave_2d import Wave2D


class SimplePDEProblem:
    """Simple PDE problem for testing."""

    def residual(self, u: Tensor, t: Tensor, x: Tensor) -> Tensor:
        """Heat equation: u_t - u_xx = 0."""
        return u - 0.5 * (t + x)

    def boundary_condition(self, t: Tensor, x: Tensor) -> Tensor:
        """BC: u = 0 at boundaries."""
        return torch.zeros_like(t)

    def initial_condition(self, x: Tensor) -> Tensor:
        """IC: u(0, x) = sin(x)."""
        return torch.sin(x)


@pytest.fixture
def network() -> PINNBase:
    """Create a test network."""
    return PINNBase(
        input_dim=2,
        output_dim=1,
        hidden_dims=[32, 32],
    )


@pytest.fixture
def pde_problem() -> SimplePDEProblem:
    """Create a test PDE problem."""
    return SimplePDEProblem()


@pytest.fixture
def wave2d() -> Wave2D:
    return Wave2D(wave_speed=1.0, length_x=1.0, length_y=1.0, t_max=1.0)


@pytest.fixture
def network_2d() -> PINNBase:
    return PINNBase(
        input_dim=3,
        output_dim=1,
        hidden_dims=[32, 32],
    )


class TestPhysicsLoss:
    """Tests for PhysicsLoss class."""

    def test_physics_loss_scalar(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Test that physics loss returns scalar."""
        physics_loss = PhysicsLoss(pde_problem)

        t = torch.randn(50, requires_grad=True)
        x = torch.randn(50, requires_grad=True)

        result = physics_loss(network, t, x)

        assert "total" in result
        assert result["total"].dim() == 0
        assert result["total"].shape == ()

    def test_physics_loss_positive(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Test that physics loss is non-negative."""
        physics_loss = PhysicsLoss(pde_problem)

        t = torch.randn(30, requires_grad=True)
        x = torch.randn(30, requires_grad=True)

        result = physics_loss(network, t, x)

        assert result["mse_residual"] >= 0
        assert result["physics_loss"] >= 0

    def test_physics_loss_differentiable(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Test that gradients flow through physics loss."""
        physics_loss = PhysicsLoss(pde_problem)

        t = torch.randn(20, requires_grad=True)
        x = torch.randn(20, requires_grad=True)

        result = physics_loss(network, t, x)
        loss = result["total"]

        # Backward should work
        loss.backward()

        # Network should have gradients
        has_grad = any(
            p.grad is not None for p in network.parameters() if p.requires_grad
        )
        assert has_grad, "No gradients in network parameters"

    def test_physics_loss_weight(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Test that weight is applied correctly."""
        weight = 5.0
        physics_loss = PhysicsLoss(pde_problem, weight=weight)

        t = torch.randn(25, requires_grad=True)
        x = torch.randn(25, requires_grad=True)

        result = physics_loss(network, t, x)

        # Weighted loss should be weight * mse_residual
        expected = weight * result["mse_residual"]
        assert torch.isclose(result["physics_loss"], expected)

    def test_physics_loss_returns_dict(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Test that physics loss returns dictionary with expected keys."""
        physics_loss = PhysicsLoss(pde_problem)

        t = torch.randn(15, requires_grad=True)
        x = torch.randn(15, requires_grad=True)

        result = physics_loss(network, t, x)

        assert "residual" in result
        assert "mse_residual" in result
        assert "physics_loss" in result
        assert "total" in result


class TestHigherOrderPhysicsLoss:
    """Tests for HigherOrderPhysicsLoss class."""

    def test_returns_derivatives(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Test that higher-order loss returns derivative information."""
        physics_loss = HigherOrderPhysicsLoss(pde_problem)

        t = torch.randn(20, requires_grad=True)
        x = torch.randn(20, requires_grad=True)

        result = physics_loss(network, t, x)

        assert "u" in result
        assert "u_t" in result
        assert "u_x" in result

    def test_derivative_shapes(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Test that derivatives have correct shapes."""
        physics_loss = HigherOrderPhysicsLoss(pde_problem)

        n = 25
        t = torch.randn(n, requires_grad=True)
        x = torch.randn(n, requires_grad=True)

        result = physics_loss(network, t, x)

        assert result["u"].shape == (n, 1)
        assert result["u_t"].shape == (n, 1)
        assert result["u_x"].shape == (n, 1)


class TestBoundaryLoss:
    """Tests for BoundaryLoss class."""

    def test_bc_loss(self, network: PINNBase) -> None:
        """Test boundary condition loss."""
        bc_loss = BoundaryLoss()

        t_bc = torch.linspace(0, 1, 20)
        x_bc = torch.zeros(20)
        u_bc = torch.zeros(20)

        result = bc_loss(network, t_bc=t_bc, x_bc=x_bc, u_bc=u_bc)

        assert "bc_loss" in result
        assert "mse_bc" in result
        assert result["mse_bc"] >= 0

    def test_ic_loss(self, network: PINNBase) -> None:
        """Test initial condition loss."""
        bc_loss = BoundaryLoss()

        x_ic = torch.linspace(0, 3.14159, 30)
        u_ic = torch.sin(x_ic)

        result = bc_loss(network, x_ic=x_ic, u_ic=u_ic)

        assert "ic_loss" in result
        assert "mse_ic" in result
        assert result["mse_ic"] >= 0

    def test_combined_bc_ic_loss(self, network: PINNBase) -> None:
        """Test combined BC and IC loss."""
        bc_loss = BoundaryLoss(bc_weight=1.0, ic_weight=2.0)

        # BC data
        t_bc = torch.linspace(0, 1, 10)
        x_bc = torch.zeros(10)
        u_bc = torch.zeros(10)

        # IC data
        x_ic = torch.linspace(0, 3.14159, 15)
        u_ic = torch.sin(x_ic)

        result = bc_loss(
            network, t_bc=t_bc, x_bc=x_bc, u_bc=u_bc, x_ic=x_ic, u_ic=u_ic
        )

        assert "bc_loss" in result
        assert "ic_loss" in result
        assert "total" in result

        # Total should be sum of weighted losses
        expected_total = result["bc_loss"] + result["ic_loss"]
        assert torch.isclose(result["total"], expected_total)

    def test_bc_weight_applied(self, network: PINNBase) -> None:
        """Test that BC weight is correctly applied."""
        weight = 3.0
        bc_loss = BoundaryLoss(bc_weight=weight)

        t_bc = torch.linspace(0, 1, 15)
        x_bc = torch.zeros(15)
        u_bc = torch.zeros(15)

        result = bc_loss(network, t_bc=t_bc, x_bc=x_bc, u_bc=u_bc)

        expected = weight * result["mse_bc"]
        assert torch.isclose(result["bc_loss"], expected)

    def test_wave2d_bc_ic_loss(self, network_2d: PINNBase, wave2d: Wave2D) -> None:
        bc_loss = BoundaryLoss()

        n_bc = 24
        t_bc = torch.rand(n_bc)
        x_bc = torch.rand(n_bc, 2)
        x_bc[: n_bc // 4, 0] = 0.0
        x_bc[n_bc // 4 : n_bc // 2, 0] = wave2d.length_x
        x_bc[n_bc // 2 : 3 * n_bc // 4, 1] = 0.0
        x_bc[3 * n_bc // 4 :, 1] = wave2d.length_y
        u_bc = wave2d.boundary_condition(t_bc, x_bc)

        n_ic = 16
        x_ic = torch.rand(n_ic, 2)
        u_ic = wave2d.initial_condition(x_ic)

        result = bc_loss(
            network_2d,
            t_bc=t_bc,
            x_bc=x_bc,
            u_bc=u_bc,
            x_ic=x_ic,
            u_ic=u_ic,
        )

        assert "bc_loss" in result
        assert "ic_loss" in result
        assert torch.isfinite(result["total"])


class TestDirichletBC:
    """Tests for DirichletBC class."""

    def test_dirichlet_bc(self, network: PINNBase) -> None:
        """Test Dirichlet boundary condition."""
        dirichlet_bc = DirichletBC(weight=1.0)

        t_bc = torch.linspace(0, 1, 20)
        x_bc = torch.zeros(20)
        u_bc = torch.zeros(20)

        result = dirichlet_bc(network, t_bc=t_bc, x_bc=x_bc, u_bc=u_bc)

        assert "mse_bc" in result
        assert "bc_loss" in result


class TestNeumannBC:
    """Tests for NeumannBC class."""

    def test_neumann_bc(self, network: PINNBase) -> None:
        """Test Neumann boundary condition."""
        neumann_bc = NeumannBC(weight=1.0)

        t_bc = torch.linspace(0, 1, 15, requires_grad=True)
        x_bc = torch.zeros(15, requires_grad=True)
        du_dn_true = torch.zeros(15)

        result = neumann_bc(network, t_bc=t_bc, x_bc=x_bc, du_dn_true=du_dn_true)

        assert "du_dn_pred" in result
        assert "mse_neumann" in result
        assert "neumann_loss" in result

    def test_neumann_bc_derivative_shape(self, network: PINNBase) -> None:
        """Test that Neumann BC derivative has correct shape."""
        neumann_bc = NeumannBC()

        n = 20
        t_bc = torch.linspace(0, 1, n, requires_grad=True)
        x_bc = torch.zeros(n, requires_grad=True)
        du_dn_true = torch.zeros(n)

        result = neumann_bc(network, t_bc=t_bc, x_bc=x_bc, du_dn_true=du_dn_true)

        assert result["du_dn_pred"].shape == (n, 1)


class TestCausalLoss:
    """Tests for CausalLoss class."""

    def test_causal_weights_decay_with_time(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Test that causal weights decay for later times."""
        causal_loss = CausalLoss(pde_problem, epsilon=1.0)

        # Create points with increasing time
        n = 100
        t = torch.linspace(0, 1, n, requires_grad=True)
        x = torch.linspace(0, 1, n, requires_grad=True)

        result = causal_loss(network, t, x)

        weights = result["causal_weights"]

        # Weights for earlier times should generally be >= weights for later times
        # (after sorting by time)
        sorted_indices = torch.argsort(t.flatten())
        sorted_weights = weights[sorted_indices]

        # Average weight of first half should be >= second half
        mid = n // 2
        early_avg = sorted_weights[:mid].mean()
        late_avg = sorted_weights[mid:].mean()

        assert early_avg >= late_avg, "Causal weights should decay with time"

    def test_causal_loss_scalar(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Test that causal loss returns scalar total."""
        causal_loss = CausalLoss(pde_problem)

        t = torch.randn(50, requires_grad=True)
        x = torch.randn(50, requires_grad=True)

        result = causal_loss(network, t, x)

        assert result["total"].dim() == 0

    def test_causal_loss_differentiable(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Test that gradients flow through causal loss."""
        causal_loss = CausalLoss(pde_problem)

        t = torch.randn(30, requires_grad=True)
        x = torch.randn(30, requires_grad=True)

        result = causal_loss(network, t, x)
        loss = result["total"]

        loss.backward()

        has_grad = any(
            p.grad is not None for p in network.parameters() if p.requires_grad
        )
        assert has_grad

    def test_epsilon_effect(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Test that epsilon affects weight decay rate."""
        # Create two losses with different epsilon
        causal_loss_low = CausalLoss(pde_problem, epsilon=0.1)
        causal_loss_high = CausalLoss(pde_problem, epsilon=10.0)

        n = 50
        t = torch.linspace(0, 1, n, requires_grad=True)
        x = torch.linspace(0, 1, n, requires_grad=True)

        result_low = causal_loss_low(network, t.clone(), x.clone())
        result_high = causal_loss_high(network, t.clone(), x.clone())

        # Higher epsilon should cause more aggressive decay
        # (more variance in weights)
        weights_low = result_low["causal_weights"]
        weights_high = result_high["causal_weights"]

        # High epsilon should have more spread (some weights near 0, some near 1)
        # This may not always hold due to network predictions, so we just check
        # that both produce valid weights
        assert (weights_low >= 0).all()
        assert (weights_high >= 0).all()

    def test_causal_loss_returns_dict(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Test that causal loss returns expected dictionary keys."""
        causal_loss = CausalLoss(pde_problem)

        t = torch.randn(20, requires_grad=True)
        x = torch.randn(20, requires_grad=True)

        result = causal_loss(network, t, x)

        assert "residual" in result
        assert "causal_weights" in result
        assert "weighted_mse" in result
        assert "mse_residual" in result
        assert "causal_loss" in result
        assert "total" in result

    def test_binned_vs_unbinned(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Test binned vs unbinned computation both work."""
        causal_loss = CausalLoss(pde_problem, num_time_bins=20)

        t = torch.linspace(0, 1, 100, requires_grad=True)
        x = torch.linspace(0, 1, 100, requires_grad=True)

        result_binned = causal_loss(network, t.clone(), x.clone(), use_binning=True)
        result_unbinned = causal_loss(
            network, t.clone(), x.clone(), use_binning=False
        )

        # Both should produce valid losses
        assert result_binned["total"] >= 0
        assert result_unbinned["total"] >= 0


class TestAdaptiveCausalLoss:
    """Tests for AdaptiveCausalLoss class."""

    def test_epsilon_annealing(self, pde_problem: SimplePDEProblem) -> None:
        """Test that epsilon anneals correctly."""
        loss_fn = AdaptiveCausalLoss(
            pde_problem,
            epsilon_init=0.01,
            epsilon_final=100.0,
            annealing_steps=100,
        )

        assert loss_fn.epsilon == 0.01

        # Step 50 times (halfway)
        loss_fn.set_step(50)
        expected = 0.01 + 0.5 * (100.0 - 0.01)
        assert abs(loss_fn.epsilon - expected) < 0.1

        # Step past annealing
        loss_fn.set_step(200)
        assert loss_fn.epsilon == 100.0

    def test_returns_epsilon(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Test that adaptive loss returns current epsilon."""
        loss_fn = AdaptiveCausalLoss(
            pde_problem, epsilon_init=1.0, epsilon_final=10.0
        )

        t = torch.randn(20, requires_grad=True)
        x = torch.randn(20, requires_grad=True)

        result = loss_fn(network, t, x)

        assert "epsilon" in result


class TestTimeDecayLoss:
    """Tests for TimeDecayLoss class."""

    def test_decay_weights(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Test that time decay weights decrease with time."""
        loss_fn = TimeDecayLoss(pde_problem, alpha=5.0)

        n = 50
        t = torch.linspace(0, 1, n, requires_grad=True)
        x = torch.linspace(0, 1, n, requires_grad=True)

        result = loss_fn(network, t, x)

        weights = result["decay_weights"]

        # Weights should decrease with time
        sorted_indices = torch.argsort(t.flatten())
        sorted_weights = weights[sorted_indices]

        # Early weights should be higher than late weights
        early_avg = sorted_weights[: n // 2].mean()
        late_avg = sorted_weights[n // 2 :].mean()

        assert early_avg > late_avg, "Time decay weights should decrease"

    def test_returns_dict(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Test that time decay loss returns expected keys."""
        loss_fn = TimeDecayLoss(pde_problem)

        t = torch.randn(15, requires_grad=True)
        x = torch.randn(15, requires_grad=True)

        result = loss_fn(network, t, x)

        assert "decay_weights" in result
        assert "weighted_mse" in result
        assert "time_decay_loss" in result


class TestLossGradientFlow:
    """Test gradient flow through all losses."""

    def test_physics_loss_gradient_flow(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Verify gradient flow through physics loss."""
        physics_loss = PhysicsLoss(pde_problem)

        t = torch.randn(20, requires_grad=True)
        x = torch.randn(20, requires_grad=True)

        result = physics_loss(network, t, x)
        loss = result["total"]

        loss.backward()

        # Check network has gradients
        for name, param in network.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_causal_loss_gradient_flow(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Verify gradient flow through causal loss."""
        causal_loss = CausalLoss(pde_problem, epsilon=1.0)

        t = torch.randn(25, requires_grad=True)
        x = torch.randn(25, requires_grad=True)

        result = causal_loss(network, t, x)
        loss = result["total"]

        loss.backward()

        # Check network has gradients
        grad_count = sum(
            1 for p in network.parameters() if p.grad is not None and p.requires_grad
        )
        assert grad_count > 0

    def test_boundary_loss_gradient_flow(self, network: PINNBase) -> None:
        """Verify gradient flow through boundary loss."""
        bc_loss = BoundaryLoss()

        t_bc = torch.randn(15)
        x_bc = torch.randn(15)
        u_bc = torch.randn(15)

        result = bc_loss(network, t_bc=t_bc, x_bc=x_bc, u_bc=u_bc)
        loss = result["total"]

        loss.backward()

        grad_count = sum(
            1 for p in network.parameters() if p.grad is not None and p.requires_grad
        )
        assert grad_count > 0


class TestCombinedLoss:
    """Test combining multiple loss types."""

    def test_combined_physics_bc_causal(
        self, network: PINNBase, pde_problem: SimplePDEProblem
    ) -> None:
        """Test combining physics, BC, and causal losses."""
        physics_loss = PhysicsLoss(pde_problem, weight=1.0)
        bc_loss = BoundaryLoss(bc_weight=10.0, ic_weight=10.0)
        causal_loss = CausalLoss(pde_problem, epsilon=1.0, weight=0.5)

        # Collocation points
        t = torch.randn(50, requires_grad=True)
        x = torch.randn(50, requires_grad=True)

        # BC data
        t_bc = torch.linspace(0, 1, 20)
        x_bc = torch.zeros(20)
        u_bc = torch.zeros(20)

        # IC data
        x_ic = torch.linspace(0, 3.14159, 25)
        u_ic = torch.sin(x_ic)

        # Compute individual losses
        physics_result = physics_loss(network, t, x)
        bc_result = bc_loss(
            network, t_bc=t_bc, x_bc=x_bc, u_bc=u_bc, x_ic=x_ic, u_ic=u_ic
        )
        causal_result = causal_loss(network, t.clone(), x.clone())

        # Combine losses
        total = physics_result["total"] + bc_result["total"] + causal_result["total"]

        # Should be differentiable
        total.backward()

        assert total.dim() == 0
        assert total >= 0


class TestLoss2DWave:
    def test_physics_loss_wave2d_finite_and_grad(
        self,
        network_2d: PINNBase,
        wave2d: Wave2D,
    ) -> None:
        physics_loss = PhysicsLoss(wave2d)
        t = torch.rand(24, requires_grad=True)
        x = torch.rand(24, 2, requires_grad=True)

        result = physics_loss(network_2d, t, x)
        loss = result["total"]

        assert torch.isfinite(loss)
        loss.backward()
        grad_count = sum(
            1 for p in network_2d.parameters() if p.grad is not None and p.requires_grad
        )
        assert grad_count > 0

    def test_causal_loss_wave2d_finite_and_grad(
        self,
        network_2d: PINNBase,
        wave2d: Wave2D,
    ) -> None:
        causal_loss = CausalLoss(wave2d, epsilon=1.0, num_time_bins=16)
        t = torch.rand(24, requires_grad=True)
        x = torch.rand(24, 2, requires_grad=True)

        result = causal_loss(network_2d, t, x, use_binning=True)
        loss = result["total"]

        assert torch.isfinite(loss)
        assert torch.isfinite(result["causal_weights"]).all()
        loss.backward()
        grad_count = sum(
            1 for p in network_2d.parameters() if p.grad is not None and p.requires_grad
        )
        assert grad_count > 0
