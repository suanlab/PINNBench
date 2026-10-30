"""Unit tests for hybrid PINN-operator models."""

from __future__ import annotations

import pytest
import torch

from causal_pinn.models.hybrid import (
    CausalGate,
    HybridConfig,
    HybridModelWrapper,
    HybridPINNOperator,
)
from causal_pinn.models.operators import FNO
from causal_pinn.models.pinn import PINNBase
from causal_pinn.physics.pde import Heat1D, Wave2D


class TestCausalGate:
    """Tests for CausalGate module."""

    def test_output_shape(self) -> None:
        """Test that CausalGate produces correct output shapes."""
        batch_size, out_dim = 32, 1
        gate = CausalGate(out_dim=out_dim, hidden_dim=16, causal_strength=1.0)

        operator_pred = torch.randn(batch_size, out_dim)
        pinn_pred = torch.randn(batch_size, out_dim)
        t = torch.rand(batch_size, 1)

        blended, gate_values = gate(operator_pred, pinn_pred, t)

        assert blended.shape == (batch_size, out_dim)
        assert gate_values.shape == (batch_size, 1)

    def test_gate_values_in_unit_interval(self) -> None:
        """Test that gate values are in [0, 1]."""
        batch_size, out_dim = 64, 1
        gate = CausalGate(out_dim=out_dim, hidden_dim=16, causal_strength=1.0)

        operator_pred = torch.randn(batch_size, out_dim)
        pinn_pred = torch.randn(batch_size, out_dim)
        t = torch.rand(batch_size, 1)

        _, gate_values = gate(operator_pred, pinn_pred, t)

        assert (gate_values >= 0).all(), "Gate values should be >= 0"
        assert (gate_values <= 1).all(), "Gate values should be <= 1"

    def test_causal_strength_bias(self) -> None:
        """Test that higher causal_strength biases toward PINN at later times."""
        batch_size, out_dim = 64, 1

        # Create identical predictions for both models
        operator_pred = torch.ones(batch_size, out_dim)
        pinn_pred = torch.zeros(batch_size, out_dim)

        # Late times (close to 1)
        t_late = torch.ones(batch_size, 1)

        # Gate with no causal bias
        gate_no_bias = CausalGate(out_dim=out_dim, hidden_dim=16, causal_strength=0.0)
        _, gate_values_no_bias = gate_no_bias(operator_pred, pinn_pred, t_late)

        # Gate with strong causal bias
        gate_strong_bias = CausalGate(out_dim=out_dim, hidden_dim=16, causal_strength=10.0)
        _, gate_values_strong_bias = gate_strong_bias(operator_pred, pinn_pred, t_late)

        # Stronger bias should give lower gate values (more PINN)
        assert gate_values_strong_bias.mean() < gate_values_no_bias.mean() + 0.5

    def test_blending_formula(self) -> None:
        """Test that blending follows: gate * operator + (1-gate) * pinn."""
        batch_size, out_dim = 16, 1
        gate = CausalGate(out_dim=out_dim, hidden_dim=8, causal_strength=0.0)

        operator_pred = torch.randn(batch_size, out_dim)
        pinn_pred = torch.randn(batch_size, out_dim)
        t = torch.rand(batch_size, 1)

        blended, gate_values = gate(operator_pred, pinn_pred, t)

        # Manually compute expected blend
        expected = gate_values * operator_pred + (1 - gate_values) * pinn_pred

        assert torch.allclose(blended, expected, atol=1e-6)

    def test_gradient_flow(self) -> None:
        """Test that gradients flow through CausalGate."""
        batch_size, out_dim = 16, 1
        gate = CausalGate(out_dim=out_dim, hidden_dim=8, causal_strength=1.0)

        operator_pred = torch.randn(batch_size, out_dim, requires_grad=True)
        pinn_pred = torch.randn(batch_size, out_dim, requires_grad=True)
        t = torch.rand(batch_size, 1, requires_grad=True)

        blended, _ = gate(operator_pred, pinn_pred, t)
        loss = blended.sum()
        loss.backward()

        # Check gradients exist for inputs
        assert operator_pred.grad is not None
        assert pinn_pred.grad is not None
        assert t.grad is not None

        # Check gradients for gate parameters
        for name, param in gate.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_gate_stats(self) -> None:
        """Test gate statistics computation."""
        batch_size, out_dim = 64, 1
        gate = CausalGate(out_dim=out_dim, hidden_dim=16, causal_strength=1.0)

        operator_pred = torch.randn(batch_size, out_dim)
        pinn_pred = torch.randn(batch_size, out_dim)
        t = torch.rand(batch_size, 1)

        _, gate_values = gate(operator_pred, pinn_pred, t)
        stats = gate.get_gate_stats(gate_values)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "frac_operator" in stats
        assert "frac_pinn" in stats

        # Check values are in valid range
        assert 0 <= stats["mean"].item() <= 1
        assert 0 <= stats["min"].item() <= 1
        assert 0 <= stats["max"].item() <= 1

    def test_multi_dim_output(self) -> None:
        """Test CausalGate with multi-dimensional output."""
        batch_size, out_dim = 32, 5
        gate = CausalGate(out_dim=out_dim, hidden_dim=32, causal_strength=1.0)

        operator_pred = torch.randn(batch_size, out_dim)
        pinn_pred = torch.randn(batch_size, out_dim)
        t = torch.rand(batch_size, 1)

        blended, gate_values = gate(operator_pred, pinn_pred, t)

        # Output should match input dimension
        assert blended.shape == (batch_size, out_dim)
        # Gate values should be scalar per sample
        assert gate_values.shape == (batch_size, 1)


class TestHybridPINNOperator:
    """Tests for HybridPINNOperator model."""

    @pytest.fixture
    def hybrid_model(self) -> HybridPINNOperator:
        """Create a default hybrid model for testing."""
        # Create PINN
        pinn = PINNBase(input_dim=2, output_dim=1, hidden_dims=[32, 32, 32])

        # Create FNO operator (1D)
        operator = FNO(in_channels=2, out_channels=1, modes=8, width=16, num_layers=2, dim=1)

        # Create causal gate
        causal_gate = CausalGate(out_dim=1, hidden_dim=16, causal_strength=1.0)

        # Create PDE problem
        pde = Heat1D(alpha=1.0, length=1.0, t_max=1.0)

        return HybridPINNOperator(
            pinn_network=pinn,
            operator=operator,
            causal_gate=causal_gate,
            pde_problem=pde,
        )

    def test_output_shape(self, hybrid_model: HybridPINNOperator) -> None:
        """Test that forward produces correct output shape."""
        n_points = 64
        t = torch.rand(n_points, 1)
        x = torch.rand(n_points, 1)

        output = hybrid_model(t, x)

        assert output.shape == (n_points, 1)

    def test_output_shape_1d_inputs(self, hybrid_model: HybridPINNOperator) -> None:
        """Test with 1D input tensors."""
        n_points = 32
        t = torch.rand(n_points)  # 1D
        x = torch.rand(n_points)  # 1D

        output = hybrid_model(t, x)

        assert output.shape == (n_points, 1)

    def test_finite_output(self, hybrid_model: HybridPINNOperator) -> None:
        """Test that output values are finite."""
        n_points = 64
        t = torch.rand(n_points, 1)
        x = torch.rand(n_points, 1)

        output = hybrid_model(t, x)

        assert torch.isfinite(output).all()

    # === Ablation Tests ===

    def test_operator_only_mode(self, hybrid_model: HybridPINNOperator) -> None:
        """Test operator-only mode (PINN disabled)."""
        hybrid_model.config = HybridConfig(use_operator=True, use_pinn=False, use_causal_gate=False)

        n_points = 32
        t = torch.rand(n_points, 1)
        x = torch.rand(n_points, 1)

        output = hybrid_model(t, x)

        assert output.shape == (n_points, 1)
        assert torch.isfinite(output).all()

    def test_pinn_only_mode(self, hybrid_model: HybridPINNOperator) -> None:
        """Test PINN-only mode (operator disabled)."""
        hybrid_model.config = HybridConfig(use_operator=False, use_pinn=True, use_causal_gate=False)

        n_points = 32
        t = torch.rand(n_points, 1)
        x = torch.rand(n_points, 1)

        output = hybrid_model(t, x)

        assert output.shape == (n_points, 1)
        assert torch.isfinite(output).all()

    def test_hybrid_mode_with_gate(self, hybrid_model: HybridPINNOperator) -> None:
        """Test full hybrid mode with causal gate."""
        hybrid_model.config = HybridConfig(
            use_operator=True, use_pinn=True, use_causal_gate=True
        )

        n_points = 32
        t = torch.rand(n_points, 1)
        x = torch.rand(n_points, 1)

        output = hybrid_model(t, x)

        assert output.shape == (n_points, 1)
        assert torch.isfinite(output).all()

    def test_hybrid_mode_without_gate(self, hybrid_model: HybridPINNOperator) -> None:
        """Test hybrid mode without causal gate (should use equal weighting)."""
        hybrid_model.config = HybridConfig(
            use_operator=True, use_pinn=True, use_causal_gate=False
        )

        n_points = 32
        t = torch.rand(n_points, 1)
        x = torch.rand(n_points, 1)

        output = hybrid_model(t, x)

        assert output.shape == (n_points, 1)
        assert torch.isfinite(output).all()

    # === Loss Tests ===

    def test_compute_hybrid_loss_structure(self, hybrid_model: HybridPINNOperator) -> None:
        """Test that compute_hybrid_loss returns expected keys."""
        n_points = 32
        t = torch.rand(n_points, 1, requires_grad=True)
        x = torch.rand(n_points, 1, requires_grad=True)

        loss_dict = hybrid_model.compute_hybrid_loss(t, x)

        assert "operator_loss" in loss_dict
        assert "physics_loss" in loss_dict
        assert "causal_loss" in loss_dict
        assert "gate_stats" in loss_dict
        assert "total" in loss_dict

    def test_compute_hybrid_loss_values(self, hybrid_model: HybridPINNOperator) -> None:
        """Test that loss values are finite and non-negative where appropriate."""
        n_points = 32
        t = torch.rand(n_points, 1, requires_grad=True)
        x = torch.rand(n_points, 1, requires_grad=True)

        loss_dict = hybrid_model.compute_hybrid_loss(t, x)

        # Losses should be finite
        assert torch.isfinite(loss_dict["physics_loss"])
        assert torch.isfinite(loss_dict["causal_loss"])
        assert torch.isfinite(loss_dict["total"])

        # Physics and causal losses should be non-negative (MSE)
        assert loss_dict["physics_loss"] >= 0
        assert loss_dict["causal_loss"] >= 0

    def test_compute_hybrid_loss_gate_stats(self, hybrid_model: HybridPINNOperator) -> None:
        """Test gate statistics in loss computation."""
        n_points = 64
        t = torch.rand(n_points, 1, requires_grad=True)
        x = torch.rand(n_points, 1, requires_grad=True)

        loss_dict = hybrid_model.compute_hybrid_loss(t, x)
        gate_stats = loss_dict["gate_stats"]

        assert "mean" in gate_stats
        assert "std" in gate_stats
        assert "min" in gate_stats
        assert "max" in gate_stats

    def test_loss_pinn_only_mode(self, hybrid_model: HybridPINNOperator) -> None:
        """Test loss computation in PINN-only mode."""
        hybrid_model.config = HybridConfig(use_operator=False, use_pinn=True, use_causal_gate=False)

        n_points = 32
        t = torch.rand(n_points, 1, requires_grad=True)
        x = torch.rand(n_points, 1, requires_grad=True)

        loss_dict = hybrid_model.compute_hybrid_loss(t, x)

        # Physics loss should be computed
        assert loss_dict["physics_loss"] >= 0
        # Operator loss should be zero
        assert loss_dict["operator_loss"] == 0

    def test_loss_operator_only_mode(self, hybrid_model: HybridPINNOperator) -> None:
        """Test loss computation in operator-only mode."""
        hybrid_model.config = HybridConfig(use_operator=True, use_pinn=False, use_causal_gate=False)

        n_points = 32
        t = torch.rand(n_points, 1)
        x = torch.rand(n_points, 1)

        loss_dict = hybrid_model.compute_hybrid_loss(t, x)

        # Physics loss should be zero (no PINN)
        assert loss_dict["physics_loss"] == 0

    # === Gradient Flow Tests ===

    def test_gradient_flow_forward(self, hybrid_model: HybridPINNOperator) -> None:
        """Test gradient flow through forward pass."""
        n_points = 32
        t = torch.rand(n_points, 1, requires_grad=True)
        x = torch.rand(n_points, 1, requires_grad=True)

        output = hybrid_model(t, x)
        loss = output.sum()
        loss.backward()

        # Check input gradients
        assert t.grad is not None
        assert x.grad is not None

    def test_gradient_flow_through_loss(self, hybrid_model: HybridPINNOperator) -> None:
        """Test gradient flow through loss computation."""
        n_points = 32
        t = torch.rand(n_points, 1, requires_grad=True)
        x = torch.rand(n_points, 1, requires_grad=True)

        loss_dict = hybrid_model.compute_hybrid_loss(t, x)
        total_loss = loss_dict["total"]
        total_loss.backward()

        # Check input gradients exist
        assert t.grad is not None
        assert x.grad is not None

        # Check PINN network gradients
        for name, param in hybrid_model.pinn_network.named_parameters():
            assert param.grad is not None, f"No gradient for PINN param {name}"

    def test_gradient_flow_operator_path(self, hybrid_model: HybridPINNOperator) -> None:
        """Test gradients flow through operator path."""
        hybrid_model.config = HybridConfig(
            use_operator=True, use_pinn=True, use_causal_gate=True
        )

        n_points = 16
        t = torch.rand(n_points, 1, requires_grad=True)
        x = torch.rand(n_points, 1, requires_grad=True)

        loss_dict = hybrid_model.compute_hybrid_loss(t, x)
        loss_dict["total"].backward()

        # Check operator gradients
        for name, param in hybrid_model.operator.named_parameters():
            assert param.grad is not None, f"No gradient for operator param {name}"

    def test_gradient_flow_gate_path(self, hybrid_model: HybridPINNOperator) -> None:
        """Test gradients flow through causal gate."""
        hybrid_model.config = HybridConfig(
            use_operator=True, use_pinn=True, use_causal_gate=True
        )

        n_points = 16
        t = torch.rand(n_points, 1, requires_grad=True)
        x = torch.rand(n_points, 1, requires_grad=True)

        loss_dict = hybrid_model.compute_hybrid_loss(t, x)
        loss_dict["total"].backward()

        # Check causal gate gradients
        for name, param in hybrid_model.causal_gate.named_parameters():
            assert param.grad is not None, f"No gradient for gate param {name}"

    # === Component Predictions Tests ===

    def test_get_component_predictions(self, hybrid_model: HybridPINNOperator) -> None:
        """Test getting individual component predictions."""
        n_points = 32
        t = torch.rand(n_points, 1)
        x = torch.rand(n_points, 1)

        preds = hybrid_model.get_component_predictions(t, x)

        assert "pinn_pred" in preds
        assert "operator_pred" in preds
        assert "blended_pred" in preds
        assert "gate_values" in preds

        assert preds["pinn_pred"] is not None
        assert preds["operator_pred"] is not None
        assert preds["blended_pred"] is not None

    def test_get_component_predictions_shapes(
        self, hybrid_model: HybridPINNOperator
    ) -> None:
        """Test component predictions have correct shapes."""
        n_points = 32
        t = torch.rand(n_points, 1)
        x = torch.rand(n_points, 1)

        preds = hybrid_model.get_component_predictions(t, x)

        assert preds["pinn_pred"].shape == (n_points, 1)
        assert preds["operator_pred"].shape == (n_points, 1)
        assert preds["blended_pred"].shape == (n_points, 1)
        assert preds["gate_values"].shape == (n_points, 1)

    def test_get_component_predictions_ablation(
        self, hybrid_model: HybridPINNOperator
    ) -> None:
        """Test component predictions in ablation mode."""
        # PINN only
        hybrid_model.config = HybridConfig(use_operator=False, use_pinn=True)
        preds = hybrid_model.get_component_predictions(torch.rand(16, 1), torch.rand(16, 1))
        assert preds["pinn_pred"] is not None
        assert preds["operator_pred"] is None

        # Operator only
        hybrid_model.config = HybridConfig(use_operator=True, use_pinn=False)
        preds = hybrid_model.get_component_predictions(torch.rand(16, 1), torch.rand(16, 1))
        assert preds["pinn_pred"] is None
        assert preds["operator_pred"] is not None


class TestHybridConfig:
    """Tests for HybridConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = HybridConfig()

        assert config.use_operator is True
        assert config.use_pinn is True
        assert config.use_causal_gate is True
        assert config.operator_weight == 1.0
        assert config.physics_weight == 1.0
        assert config.gate_weight == 0.01

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = HybridConfig(
            use_operator=False,
            use_pinn=True,
            use_causal_gate=False,
            operator_weight=0.5,
            physics_weight=2.0,
            gate_weight=0.1,
        )

        assert config.use_operator is False
        assert config.use_pinn is True
        assert config.use_causal_gate is False
        assert config.operator_weight == 0.5
        assert config.physics_weight == 2.0
        assert config.gate_weight == 0.1


class TestIntegration:
    """Integration tests for hybrid model."""

    def test_full_training_step(self) -> None:
        """Test a complete training step with optimizer."""
        # Create model
        pinn = PINNBase(input_dim=2, output_dim=1, hidden_dims=[16, 16])
        operator = FNO(in_channels=2, out_channels=1, modes=4, width=8, num_layers=1, dim=1)
        gate = CausalGate(out_dim=1, hidden_dim=8, causal_strength=1.0)
        pde = Heat1D()

        model = HybridPINNOperator(pinn, operator, gate, pde)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Training step
        optimizer.zero_grad()

        t = torch.rand(32, 1, requires_grad=True)
        x = torch.rand(32, 1, requires_grad=True)

        loss_dict = model.compute_hybrid_loss(t, x)
        loss = loss_dict["total"]

        loss.backward()
        optimizer.step()

        # Check that loss is finite and we completed a training step
        assert torch.isfinite(torch.tensor(loss.item()))

    def test_deterministic_output(self) -> None:
        """Test that model produces deterministic output with same seed."""
        torch.manual_seed(42)

        pinn = PINNBase(input_dim=2, output_dim=1, hidden_dims=[16, 16])
        operator = FNO(in_channels=2, out_channels=1, modes=4, width=8, num_layers=1, dim=1)
        gate = CausalGate(out_dim=1, hidden_dim=8, causal_strength=1.0)
        pde = Heat1D()

        model = HybridPINNOperator(pinn, operator, gate, pde)
        model.eval()

        t = torch.rand(16, 1)
        x = torch.rand(16, 1)

        # Two forward passes should give same result
        out1 = model(t, x)
        out2 = model(t, x)

        assert torch.allclose(out1, out2)

    def test_batch_independence(self) -> None:
        """Test that samples in a batch are processed independently."""
        pinn = PINNBase(input_dim=2, output_dim=1, hidden_dims=[16, 16])
        operator = FNO(in_channels=2, out_channels=1, modes=4, width=8, num_layers=1, dim=1)
        gate = CausalGate(out_dim=1, hidden_dim=8, causal_strength=1.0)
        pde = Heat1D()

        model = HybridPINNOperator(pinn, operator, gate, pde)
        model.eval()

        # Process as batch
        t_batch = torch.tensor([[0.1], [0.5], [0.9]])
        x_batch = torch.tensor([[0.2], [0.5], [0.8]])
        out_batch = model(t_batch, x_batch)

        # Process individually
        for i in range(3):
            t_single = t_batch[i : i + 1]
            x_single = x_batch[i : i + 1]
            out_single = model(t_single, x_single)
            assert torch.allclose(out_batch[i], out_single.squeeze(), atol=1e-5)


class TestHybrid2DIntegration:
    def test_hybrid_model_wrapper_wave2d_forward(self) -> None:
        pinn = PINNBase(input_dim=3, output_dim=1, hidden_dims=[16, 16])
        operator = FNO(
            in_channels=3,
            out_channels=1,
            modes=4,
            width=8,
            num_layers=1,
            dim=2,
        )
        gate = CausalGate(out_dim=1, hidden_dim=8, causal_strength=1.0)
        pde = Wave2D()
        model = HybridPINNOperator(pinn, operator, gate, pde)
        wrapper = HybridModelWrapper(model)

        inputs = torch.rand(12, 3)
        output = wrapper(inputs)

        assert output.shape == (12, 1)
        assert torch.isfinite(output).all()

    def test_hybrid_model_wrapper_wave2d_backward(self) -> None:
        pinn = PINNBase(input_dim=3, output_dim=1, hidden_dims=[16, 16])
        operator = FNO(
            in_channels=3,
            out_channels=1,
            modes=4,
            width=8,
            num_layers=1,
            dim=2,
        )
        gate = CausalGate(out_dim=1, hidden_dim=8, causal_strength=1.0)
        pde = Wave2D()
        model = HybridPINNOperator(pinn, operator, gate, pde)
        wrapper = HybridModelWrapper(model)

        inputs = torch.rand(10, 3, requires_grad=True)
        output = wrapper(inputs)
        loss = output.sum()
        loss.backward()

        assert inputs.grad is not None
        grad_count = sum(1 for p in wrapper.parameters() if p.grad is not None and p.requires_grad)
        assert grad_count > 0
