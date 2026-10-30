"""Unit tests for neural operator implementations."""

from __future__ import annotations

import pytest
import torch

from causal_pinn.models.operators import (
    FNO,
    FourierLayer,
    NeuralOperator,
    SpectralConv1d,
    SpectralConv2d,
)


class TestSpectralConv1d:
    """Tests for SpectralConv1d layer."""

    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        batch, in_channels, spatial_dim = 4, 8, 64
        out_channels = 16
        modes = 12

        layer = SpectralConv1d(in_channels, out_channels, modes)
        x = torch.randn(batch, in_channels, spatial_dim)
        out = layer(x)

        assert out.shape == (batch, out_channels, spatial_dim)

    def test_modes_less_than_spatial_dim(self) -> None:
        """Test handling when modes < spatial_dim // 2."""
        batch, channels, spatial_dim = 2, 4, 128
        modes = 8  # Less than 64 (spatial_dim // 2)

        layer = SpectralConv1d(channels, channels, modes)
        x = torch.randn(batch, channels, spatial_dim)
        out = layer(x)

        assert out.shape == (batch, channels, spatial_dim)

    def test_modes_greater_than_available(self) -> None:
        """Test handling when requested modes exceed available FFT frequencies."""
        batch, channels, spatial_dim = 2, 4, 16
        modes = 20  # More than spatial_dim // 2 + 1 = 9

        layer = SpectralConv1d(channels, channels, modes)
        x = torch.randn(batch, channels, spatial_dim)
        out = layer(x)

        assert out.shape == (batch, channels, spatial_dim)


class TestSpectralConv2d:
    """Tests for SpectralConv2d layer."""

    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        batch, in_channels, height, width = 4, 8, 32, 32
        out_channels = 16
        modes1, modes2 = 12, 12

        layer = SpectralConv2d(in_channels, out_channels, modes1, modes2)
        x = torch.randn(batch, in_channels, height, width)
        out = layer(x)

        assert out.shape == (batch, out_channels, height, width)

    def test_rectangular_input(self) -> None:
        """Test with non-square spatial dimensions."""
        batch, channels = 2, 4
        height, width = 64, 32
        modes1, modes2 = 8, 6

        layer = SpectralConv2d(channels, channels, modes1, modes2)
        x = torch.randn(batch, channels, height, width)
        out = layer(x)

        assert out.shape == (batch, channels, height, width)


class TestFourierLayer:
    """Tests for FourierLayer."""

    def test_1d_fourier_layer_shape(self) -> None:
        """Test 1D Fourier layer preserves shape."""
        batch, width, spatial_dim = 4, 32, 64
        modes = 12

        layer = FourierLayer(width, modes, dim=1)
        x = torch.randn(batch, width, spatial_dim)
        out = layer(x)

        assert out.shape == (batch, width, spatial_dim)

    def test_2d_fourier_layer_shape(self) -> None:
        """Test 2D Fourier layer preserves shape."""
        batch, width, height, width_spatial = 4, 32, 32, 32
        modes = 10

        layer = FourierLayer(width, modes, dim=2)
        x = torch.randn(batch, width, height, width_spatial)
        out = layer(x)

        assert out.shape == (batch, width, height, width_spatial)

    def test_2d_fourier_layer_tuple_modes(self) -> None:
        """Test 2D Fourier layer with tuple modes."""
        batch, width, height, width_spatial = 2, 16, 64, 32
        modes = (8, 6)

        layer = FourierLayer(width, modes, dim=2)
        x = torch.randn(batch, width, height, width_spatial)
        out = layer(x)

        assert out.shape == (batch, width, height, width_spatial)


class TestFNO:
    """Tests for Fourier Neural Operator."""

    # === Output Shape Tests ===

    def test_fno_1d_output_shape(self) -> None:
        """Test FNO produces correct output shape for 1D inputs."""
        batch, in_channels, spatial_dim = 8, 1, 128
        out_channels = 1

        fno = FNO(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=16,
            width=64,
            num_layers=4,
            dim=1,
        )
        x = torch.randn(batch, in_channels, spatial_dim)
        out = fno(x)

        assert out.shape == (batch, out_channels, spatial_dim)

    def test_fno_2d_output_shape(self) -> None:
        """Test FNO produces correct output shape for 2D inputs."""
        batch, in_channels, height, width = 4, 3, 64, 64
        out_channels = 1

        fno = FNO(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=12,
            width=32,
            num_layers=3,
            dim=2,
        )
        x = torch.randn(batch, in_channels, height, width)
        out = fno(x)

        assert out.shape == (batch, out_channels, height, width)

    def test_fno_2d_tuple_modes_output_shape(self) -> None:
        """Test FNO with tuple modes for 2D."""
        batch, channels, height, width = 2, 2, 32, 48

        fno = FNO(
            in_channels=channels,
            out_channels=channels,
            modes=(8, 12),
            width=16,
            num_layers=2,
            dim=2,
        )
        x = torch.randn(batch, channels, height, width)
        out = fno(x)

        assert out.shape == (batch, channels, height, width)

    def test_fno_different_in_out_channels(self) -> None:
        """Test FNO with different input and output channel counts."""
        batch, in_channels, out_channels, spatial_dim = 4, 3, 5, 64

        fno = FNO(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=12,
            width=32,
            num_layers=3,
            dim=1,
        )
        x = torch.randn(batch, in_channels, spatial_dim)
        out = fno(x)

        assert out.shape == (batch, out_channels, spatial_dim)

    # === Parameter Count Tests ===

    def test_fno_parameter_count_reasonable(self) -> None:
        """Test that parameter count is reasonable."""
        fno = FNO(
            in_channels=1,
            out_channels=1,
            modes=16,
            width=64,
            num_layers=4,
            dim=1,
        )

        num_params = fno.get_num_params()

        # Should have some parameters
        assert num_params > 0

        # Check that get_num_params matches actual count
        actual_params = sum(p.numel() for p in fno.parameters())
        assert num_params == actual_params

        # Should not be excessively large for this configuration
        # Rough estimate: lifting (1*64) + projection (64*1) + 4 layers
        # Each layer: spectral (64*64*16*2 real/imag) + skip (64*64) ~ 140K
        assert num_params < 600_000

    def test_fno_2d_parameter_count(self) -> None:
        """Test parameter count for 2D FNO."""
        fno = FNO(
            in_channels=2,
            out_channels=1,
            modes=8,
            width=32,
            num_layers=2,
            dim=2,
        )

        num_params = fno.get_num_params()
        assert num_params > 0

        # Verify it's consistent with manual calculation
        actual_params = sum(p.numel() for p in fno.parameters())
        assert num_params == actual_params

    # === Finite Value Tests ===

    def test_fno_forward_produces_finite_values_1d(self) -> None:
        """Test that forward pass produces finite values for 1D."""
        fno = FNO(
            in_channels=1,
            out_channels=1,
            modes=16,
            width=64,
            num_layers=4,
            dim=1,
        )
        x = torch.randn(8, 1, 128)
        out = fno(x)

        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_fno_forward_produces_finite_values_2d(self) -> None:
        """Test that forward pass produces finite values for 2D."""
        fno = FNO(
            in_channels=1,
            out_channels=1,
            modes=12,
            width=32,
            num_layers=3,
            dim=2,
        )
        x = torch.randn(4, 1, 64, 64)
        out = fno(x)

        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_fno_forward_stability_small_input(self) -> None:
        """Test numerical stability with very small inputs."""
        fno = FNO(
            in_channels=1,
            out_channels=1,
            modes=8,
            width=16,
            num_layers=2,
            dim=1,
        )
        x = torch.randn(2, 1, 32) * 1e-6
        out = fno(x)

        assert torch.isfinite(out).all()

    # === Gradient Flow Tests ===

    def test_fno_gradient_flow_1d(self) -> None:
        """Test that gradients flow through 1D FNO."""
        fno = FNO(
            in_channels=1,
            out_channels=1,
            modes=16,
            width=32,
            num_layers=3,
            dim=1,
        )
        x = torch.randn(4, 1, 64, requires_grad=True)
        out = fno(x)

        # Compute loss and backprop
        loss = out.sum()
        loss.backward()

        # Check input gradient exists
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        # Check all parameters have gradients
        for name, param in fno.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_fno_gradient_flow_2d(self) -> None:
        """Test that gradients flow through 2D FNO."""
        fno = FNO(
            in_channels=1,
            out_channels=1,
            modes=8,
            width=16,
            num_layers=2,
            dim=2,
        )
        x = torch.randn(2, 1, 32, 32, requires_grad=True)
        out = fno(x)

        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

        for name, param in fno.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    @pytest.mark.xfail(
        reason=(
            "Known CPU MKL FFT issue: Intel oneMKL DFTI ERROR: "
            "Inconsistent configuration parameters during backward pass. "
            "This is an environment-specific failure; the CUDA experiment path is unaffected."
        ),
        strict=False,
        raises=RuntimeError,
    )
    def test_fno_gradient_flow_through_spectral_conv(self) -> None:
        """Test gradients specifically flow through spectral convolution."""
        layer = SpectralConv1d(4, 4, 8)
        x = torch.randn(2, 4, 32, requires_grad=True)
        out = layer(x)

        loss = out.sum()
        loss.backward()

        # Check spectral conv weights have gradients
        assert layer.weight_real.grad is not None
        assert layer.weight_imag.grad is not None
        assert torch.isfinite(layer.weight_real.grad).all()
        assert torch.isfinite(layer.weight_imag.grad).all()

    def test_fno_gradient_magnitude(self) -> None:
        """Test that gradients have reasonable magnitude (not vanishing/exploding)."""
        fno = FNO(
            in_channels=1,
            out_channels=1,
            modes=16,
            width=64,
            num_layers=4,
            dim=1,
        )
        x = torch.randn(8, 1, 128, requires_grad=True)
        out = fno(x)

        # MSE loss
        target = torch.randn_like(out)
        loss = ((out - target) ** 2).mean()
        loss.backward()

        # Check gradient magnitudes are reasonable
        for name, param in fno.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Gradients should not be too small (vanishing) or too large (exploding)
                assert grad_norm < 1e6, f"Exploding gradient for {name}: {grad_norm}"
                # Note: We don't check lower bound strictly as some params might have small grads


class TestNeuralOperatorABC:
    """Tests for NeuralOperator abstract base class."""

    def test_fno_is_neural_operator(self) -> None:
        """Test that FNO is a NeuralOperator instance."""
        fno = FNO(
            in_channels=1,
            out_channels=1,
            modes=8,
            width=16,
            num_layers=2,
            dim=1,
        )
        assert isinstance(fno, NeuralOperator)
        assert isinstance(fno, torch.nn.Module)

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that NeuralOperator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            NeuralOperator()  # type: ignore[abstract]


class TestIntegration:
    """Integration tests for neural operators."""

    def test_fno_1d_full_forward_backward(self) -> None:
        """Test complete forward and backward pass for 1D FNO."""
        # Create model
        fno = FNO(
            in_channels=1,
            out_channels=1,
            modes=32,
            width=128,
            num_layers=4,
            dim=1,
        )

        # Create input
        batch_size = 16
        spatial_dim = 256
        x = torch.randn(batch_size, 1, spatial_dim, requires_grad=True)

        # Forward pass
        out = fno(x)
        assert out.shape == (batch_size, 1, spatial_dim)
        assert torch.isfinite(out).all()

        # Backward pass
        loss = out.mean()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_fno_2d_full_forward_backward(self) -> None:
        """Test complete forward and backward pass for 2D FNO."""
        fno = FNO(
            in_channels=3,
            out_channels=1,
            modes=16,
            width=64,
            num_layers=4,
            dim=2,
        )

        batch_size = 8
        height, width = 64, 64
        x = torch.randn(batch_size, 3, height, width, requires_grad=True)

        out = fno(x)
        assert out.shape == (batch_size, 1, height, width)
        assert torch.isfinite(out).all()

        loss = out.mean()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
