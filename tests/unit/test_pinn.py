"""Unit tests for PINN models."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from causal_pinn.models.pinn import PINNBase, PINNSolver


class SimplePDEProblem:
    """Simple PDE problem for testing: u_t = u_xx (heat equation)."""

    def residual(self, u: Tensor, t: Tensor, x: Tensor) -> Tensor:
        """Heat equation residual: u_t - alpha * u_xx."""
        # For testing, just return a simple residual
        # In practice, this would compute actual derivatives
        return u - t * torch.sin(x)

    def boundary_condition(self, t: Tensor, x: Tensor) -> Tensor:
        """Boundary condition: u = 0 at x = 0, pi."""
        return torch.zeros_like(t)

    def initial_condition(self, x: Tensor) -> Tensor:
        """Initial condition: u(0, x) = sin(x)."""
        return torch.sin(x)


class TestPINNBase:
    """Tests for PINNBase neural network."""

    def test_forward_shape(self) -> None:
        """Test that forward pass produces correct output shape."""
        # Create network
        network = PINNBase(
            input_dim=2,  # (t, x)
            output_dim=1,  # u(t, x)
            hidden_dims=[32, 32],
        )

        # Test input
        batch_size = 100
        inputs = torch.randn(batch_size, 2)

        # Forward pass
        output = network(inputs)

        # Check shape
        expected_shape = (batch_size, 1)
        assert output.shape == expected_shape, (
            f"Expected {expected_shape}, got {output.shape}"
        )

    def test_forward_shape_multi_dim_output(self) -> None:
        """Test with multi-dimensional output."""
        network = PINNBase(
            input_dim=3,
            output_dim=5,
            hidden_dims=[64, 64, 64],
        )

        batch_size = 50
        inputs = torch.randn(batch_size, 3)
        output = network(inputs)

        assert output.shape == (batch_size, 5)

    def test_configurable_hidden_dims(self) -> None:
        """Test that hidden dimensions are configurable."""
        hidden_dims = [16, 32, 64, 32, 16]
        network = PINNBase(
            input_dim=2,
            output_dim=1,
            hidden_dims=hidden_dims,
        )

        # Verify network has the right number of layers
        # Count Linear layers
        linear_count = sum(1 for m in network.modules() if isinstance(m, nn.Linear))
        assert linear_count == len(hidden_dims) + 1  # hidden + 1 output layer

    def test_default_activation_gelu(self) -> None:
        """Test that default activation is GELU."""
        network = PINNBase(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32],
        )

        assert isinstance(network.activation, nn.GELU)

    def test_custom_activation(self) -> None:
        """Test that custom activation can be set."""
        network = PINNBase(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32],
            activation=nn.Tanh(),
        )

        assert isinstance(network.activation, nn.Tanh)

    def test_input_normalization(self) -> None:
        """Test that input normalization option works."""
        network = PINNBase(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32],
            normalize_inputs=True,
        )

        # LayerNorm should be present
        assert any(isinstance(m, nn.LayerNorm) for m in network.modules())

    def test_count_parameters(self) -> None:
        """Test parameter counting."""
        # Small network with known parameter count
        network = PINNBase(
            input_dim=2,
            output_dim=1,
            hidden_dims=[10],  # 2*10 + 10 = 30, 10*1 + 1 = 11
        )

        # Input: 2*10 + 10 = 30 params
        # Output: 10*1 + 1 = 11 params
        # Total: 41
        expected = 41
        actual = network.count_parameters()
        assert actual == expected, f"Expected {expected} params, got {actual}"

    def test_gradients_flow(self) -> None:
        """Test that gradients flow through the network."""
        network = PINNBase(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
        )

        inputs = torch.randn(10, 2, requires_grad=True)
        output = network(inputs)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert inputs.grad is not None
        assert inputs.grad.shape == inputs.shape


class TestPINNSolver:
    """Tests for PINNSolver."""

    @staticmethod
    def _create_solver() -> PINNSolver:
        """Create a PINNSolver for testing."""
        network = PINNBase(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
        )
        pde_problem = SimplePDEProblem()
        return PINNSolver(network, pde_problem)

    def test_compute_physics_loss_scalar(self) -> None:
        """Test that physics loss is a scalar tensor."""
        solver = self._create_solver()

        t = torch.randn(50, requires_grad=True)
        x = torch.randn(50, requires_grad=True)

        loss = solver.compute_physics_loss(t, x)

        # Should be a scalar
        assert loss.dim() == 0, f"Expected scalar (0-dim), got {loss.dim()}-dim"
        assert loss.shape == (), f"Expected shape (), got {loss.shape}"

    def test_compute_physics_loss_differentiable(self) -> None:
        """Test that physics loss is differentiable."""
        solver = self._create_solver()

        t = torch.randn(20, requires_grad=True)
        x = torch.randn(20, requires_grad=True)

        loss = solver.compute_physics_loss(t, x)

        # Backward should work
        loss.backward()

        # Network parameters should have gradients
        has_grad = any(p.grad is not None for p in solver.network.parameters())
        assert has_grad, "Network parameters have no gradients"

    def test_compute_bc_loss(self) -> None:
        """Test boundary condition loss computation."""
        solver = self._create_solver()

        t_bc = torch.linspace(0, 1, 10)
        x_bc = torch.zeros(10)  # x = 0 boundary
        u_bc = torch.zeros(10)  # u = 0 at boundary

        loss = solver.compute_bc_loss(t_bc, x_bc, u_bc)

        assert loss.dim() == 0
        assert loss >= 0

    def test_compute_ic_loss(self) -> None:
        """Test initial condition loss computation."""
        solver = self._create_solver()

        x_ic = torch.linspace(0, 3.14159, 20)
        u_ic = torch.sin(x_ic)  # u(0, x) = sin(x)

        loss = solver.compute_ic_loss(x_ic, u_ic)

        assert loss.dim() == 0
        assert loss >= 0

    def test_predict_shape(self) -> None:
        """Test prediction output shape."""
        solver = self._create_solver()

        n_points = 100
        t = torch.randn(n_points)
        x = torch.randn(n_points)

        u_pred = solver.predict(t, x)

        assert u_pred.shape == (n_points, 1)

    def test_predict_with_2d_inputs(self) -> None:
        """Test prediction with 2D input tensors."""
        solver = self._create_solver()

        n_points = 50
        t = torch.randn(n_points, 1)
        x = torch.randn(n_points, 1)

        u_pred = solver.predict(t, x)

        assert u_pred.shape == (n_points, 1)

    def test_compute_derivatives(self) -> None:
        """Test derivative computation."""
        solver = self._create_solver()

        t = torch.randn(20, requires_grad=True)
        x = torch.randn(20, requires_grad=True)

        derivs = solver.compute_derivatives(t, x, derivatives="u,u_t,u_x")

        assert "u" in derivs
        assert "u_t" in derivs
        assert "u_x" in derivs

        # Check shapes
        assert derivs["u"].shape == (20, 1)
        assert derivs["u_t"].shape == (20, 1)
        assert derivs["u_x"].shape == (20, 1)

    def test_compute_second_derivatives(self) -> None:
        """Test second-order derivative computation."""
        solver = self._create_solver()

        t = torch.randn(15, requires_grad=True)
        x = torch.randn(15, requires_grad=True)

        derivs = solver.compute_derivatives(t, x, derivatives="u_xx,u_tt")

        assert "u_xx" in derivs
        assert "u_tt" in derivs

    def test_train_eval_modes(self) -> None:
        """Test train/eval mode switching."""
        solver = self._create_solver()

        solver.train_mode()
        assert solver.network.training

        solver.eval_mode()
        assert not solver.network.training

    def test_pde_problem_protocol_duck_typing(self) -> None:
        """Test that PINNSolver works with duck typing."""
        # Create a simple class with required methods
        class MinimalPDE:
            def residual(self, u: Tensor, t: Tensor, x: Tensor) -> Tensor:
                return u

            def boundary_condition(self, t: Tensor, x: Tensor) -> Tensor:
                return torch.zeros_like(t)

            def initial_condition(self, x: Tensor) -> Tensor:
                return torch.zeros_like(x)

        network = PINNBase(input_dim=2, output_dim=1, hidden_dims=[16])
        solver = PINNSolver(network, MinimalPDE())

        # Should work without errors
        t = torch.randn(10, requires_grad=True)
        x = torch.randn(10, requires_grad=True)
        loss = solver.compute_physics_loss(t, x)

        assert loss.dim() == 0


class TestPINNBaseSerialization:
    """Tests for PINNBase serialization and state dict."""

    def test_state_dict_keys(self) -> None:
        """Test that state dict has expected keys."""
        network = PINNBase(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
        )

        state_dict = network.state_dict()

        # Should have keys for each linear layer
        assert len(state_dict) > 0

    def test_load_state_dict(self) -> None:
        """Test loading state dict."""
        network1 = PINNBase(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
        )

        # Save state
        state_dict = network1.state_dict()

        # Create new network and load state
        network2 = PINNBase(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
        )
        network2.load_state_dict(state_dict)

        # Should produce same outputs
        inputs = torch.randn(10, 2)
        out1 = network1(inputs)
        out2 = network2(inputs)

        assert torch.allclose(out1, out2)


class TestPINNBaseDevice:
    """Tests for device handling."""

    def test_cpu_device(self) -> None:
        """Test network on CPU."""
        network = PINNBase(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32],
        )

        inputs = torch.randn(10, 2)
        output = network(inputs)

        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self) -> None:
        """Test network on CUDA."""
        network = PINNBase(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32],
        ).cuda()

        inputs = torch.randn(10, 2).cuda()
        output = network(inputs)

        assert output.device.type == "cuda"


class TestBCLossShapeHandling:
    """Regression: a 1-D u_bc must not broadcast against an (N, 1) prediction."""

    def _solver(self):
        from causal_pinn.models.pinn.base import PINNBase
        from causal_pinn.models.pinn.pinn_solver import PINNSolver
        from causal_pinn.physics.pde import Heat1D

        torch.manual_seed(0)
        net = PINNBase(input_dim=2, output_dim=1, hidden_dims=[16, 16])
        return PINNSolver(net, Heat1D())

    def test_bc_loss_is_shape_invariant(self) -> None:
        solver = self._solver()
        t = torch.rand(8, 1)
        x = torch.rand(8, 1)
        # Nonzero targets: homogeneous data would mask the broadcast.
        u_flat = torch.linspace(0.0, 1.0, 8)
        torch.testing.assert_close(
            solver.compute_bc_loss(t, x, u_flat),
            solver.compute_bc_loss(t, x, u_flat.unsqueeze(-1)),
        )

    def test_ic_loss_is_shape_invariant(self) -> None:
        solver = self._solver()
        x = torch.rand(8, 1)
        u_flat = torch.linspace(0.0, 1.0, 8)
        torch.testing.assert_close(
            solver.compute_ic_loss(x, u_flat),
            solver.compute_ic_loss(x, u_flat.unsqueeze(-1)),
        )
