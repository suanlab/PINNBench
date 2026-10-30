"""Fourier Neural Operator (FNO) implementation."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from causal_pinn.models.operators.base import NeuralOperator


class SpectralConv1d(nn.Module):
    """1D Spectral convolution layer using FFT.

    Performs convolution in the Fourier domain by learning complex weights
    for low-frequency modes. This enables efficient global convolution
    operations with O(N log N) complexity.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes: Number of Fourier modes to keep (low-frequency modes).
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Scale factor for initialization
        scale = 1.0 / (in_channels * out_channels)

        # Complex weights for spectral convolution
        # Shape: (in_channels, out_channels, modes)
        # We store real and imaginary parts separately for compatibility
        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes)
        )

    def compl_mul1d(self, x: Tensor, weights_real: Tensor, weights_imag: Tensor) -> Tensor:
        """Complex multiplication for 1D inputs.

        Args:
            x: Input tensor of shape (batch, in_channels, modes), complex
            weights_real: Real part of weights (in_channels, out_channels, modes)
            weights_imag: Imaginary part of weights (in_channels, out_channels, modes)

        Returns:
            Output tensor of shape (batch, out_channels, modes), complex
        """
        # x is complex, weights are stored as real/imag parts
        # (batch, in_channels, modes) x (in_channels, out_channels, modes)
        # -> (batch, out_channels, modes)

        # Create complex weights
        weights = torch.complex(weights_real, weights_imag)

        # Use einsum for batched complex matrix multiplication
        # x: (batch, in_channels, modes)
        # weights: (in_channels, out_channels, modes)
        # output: (batch, out_channels, modes)
        return torch.einsum("bim,iom->bom", x, weights)

    def forward(self, x: Tensor) -> Tensor:
        """Apply spectral convolution.

        Args:
            x: Input tensor of shape (batch, in_channels, spatial_dim).

        Returns:
            Output tensor of shape (batch, out_channels, spatial_dim).
        """
        batch, _, spatial_dim = x.shape

        # Compute FFT (real to complex)
        # rfft returns shape (batch, in_channels, spatial_dim // 2 + 1)
        x_ft = torch.fft.rfft(x, dim=-1)

        # Allocate output in frequency domain
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            x_ft.shape[-1],
            dtype=x_ft.dtype,
            device=x.device,
        )

        # Multiply only the low-frequency modes
        # x_ft shape: (batch, in_channels, spatial_dim // 2 + 1)
        # We keep modes from index 0 to modes-1
        modes_to_keep = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :modes_to_keep] = self.compl_mul1d(
            x_ft[:, :, :modes_to_keep],
            self.weight_real[:, :, :modes_to_keep],
            self.weight_imag[:, :, :modes_to_keep],
        )

        # Inverse FFT (complex to real)
        return torch.fft.irfft(out_ft, n=spatial_dim, dim=-1)  # type: ignore[no-any-return]


class SpectralConv2d(nn.Module):
    """2D Spectral convolution layer using FFT.

    Performs convolution in the Fourier domain by learning complex weights
    for low-frequency modes in both spatial dimensions.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        modes1: Number of Fourier modes to keep in first spatial dimension.
        modes2: Number of Fourier modes to keep in second spatial dimension.
    """

    def __init__(
        self, in_channels: int, out_channels: int, modes1: int, modes2: int
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)

        # Complex weights for spectral convolution
        # Shape: (in_channels, out_channels, modes1, modes2)
        self.weight_real = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2)
        )

    def compl_mul2d(
        self,
        x: Tensor,
        weights_real: Tensor,
        weights_imag: Tensor,
    ) -> Tensor:
        """Complex multiplication for 2D inputs.

        Args:
            x: Input tensor of shape (batch, in_channels, modes1, modes2), complex
            weights_real: Real part of weights (in_channels, out_channels, modes1, modes2)
            weights_imag: Imaginary part of weights (in_channels, out_channels, modes1, modes2)

        Returns:
            Output tensor of shape (batch, out_channels, modes1, modes2), complex
        """
        weights = torch.complex(weights_real, weights_imag)
        # x: (batch, in_channels, modes1, modes2)
        # weights: (in_channels, out_channels, modes1, modes2)
        return torch.einsum("bimn,iomn->bomn", x, weights)

    def forward(self, x: Tensor) -> Tensor:
        """Apply spectral convolution.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width).

        Returns:
            Output tensor of shape (batch, out_channels, height, width).
        """
        batch, _, height, width = x.shape

        # Compute 2D FFT (real to complex)
        # rfft2 returns shape (batch, in_channels, height, width // 2 + 1)
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))

        # Allocate output in frequency domain
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            x_ft.shape[-2],
            x_ft.shape[-1],
            dtype=x_ft.dtype,
            device=x.device,
        )

        # Multiply only the low-frequency modes
        modes1_to_keep = min(self.modes1, x_ft.shape[-2])
        modes2_to_keep = min(self.modes2, x_ft.shape[-1])

        out_ft[:, :, :modes1_to_keep, :modes2_to_keep] = self.compl_mul2d(
            x_ft[:, :, :modes1_to_keep, :modes2_to_keep],
            self.weight_real[:, :, :modes1_to_keep, :modes2_to_keep],
            self.weight_imag[:, :, :modes1_to_keep, :modes2_to_keep],
        )

        # Inverse FFT (complex to real)
        return torch.fft.irfft2(out_ft, s=(height, width), dim=(-2, -1))  # type: ignore[no-any-return]


class FourierLayer(nn.Module):
    """A single Fourier layer with spectral convolution and skip connection.

    Each Fourier layer consists of:
    1. Spectral convolution in Fourier domain
    2. Local convolution (1x1 conv) for the skip connection
    3. Addition of both paths
    4. GELU activation

    Args:
        width: Channel dimension (number of features).
        modes: Number of Fourier modes to keep (for 1D) or tuple for 2D.
        dim: Spatial dimension (1 or 2).
    """

    def __init__(
        self,
        width: int,
        modes: int | tuple[int, int],
        dim: Literal[1, 2],
    ) -> None:
        super().__init__()
        self.dim = dim
        self.width = width

        if dim == 1:
            assert isinstance(modes, int), "modes must be int for 1D"
            self.spectral_conv = SpectralConv1d(width, width, modes)
        else:
            if isinstance(modes, int):
                modes = (modes, modes)
            self.spectral_conv = SpectralConv2d(width, width, modes[0], modes[1])

        # Local skip connection via 1x1 convolution
        self.skip_conv = nn.Conv1d(width, width, 1) if dim == 1 else nn.Conv2d(width, width, 1)

        self.activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        """Apply Fourier layer with skip connection.

        Args:
            x: Input tensor of shape (batch, width, *) where * is spatial dims.

        Returns:
            Output tensor of same shape as input.
        """
        # Spectral convolution path
        spectral_out = self.spectral_conv(x)

        # Skip connection path
        skip_out = self.skip_conv(x)

        # Combine and activate
        return self.activation(spectral_out + skip_out)  # type: ignore[no-any-return]


class FNO(NeuralOperator):
    """Fourier Neural Operator for learning PDE solutions.

    The FNO learns a mapping between function spaces using spectral convolutions
    in the Fourier domain. It operates in a resolution-invariant manner by
    truncating high-frequency modes and learning weights only for low frequencies.

    Architecture:
        1. Lifting layer: Project input channels to hidden width
        2. Fourier layers: [SpectralConv + Skip + GELU] x num_layers
        3. Projection layer: Project hidden width to output channels

    Args:
        in_channels: Number of input channels (e.g., 1 for scalar field).
        out_channels: Number of output channels (e.g., 1 for scalar field).
        modes: Number of Fourier modes to keep. For 2D, can be int or (modes1, modes2).
        width: Hidden channel dimension.
        num_layers: Number of Fourier layers.
        dim: Spatial dimension, either 1 or 2.

    Example:
        >>> # 1D FNO for time-series prediction
        >>> fno = FNO(in_channels=1, out_channels=1, modes=16, width=64, num_layers=4, dim=1)
        >>> x = torch.randn(8, 1, 128)  # batch=8, channels=1, spatial=128
        >>> y = fno(x)  # shape: (8, 1, 128)

        >>> # 2D FNO for image-based PDE
        >>> fno = FNO(in_channels=3, out_channels=1, modes=12, width=32, num_layers=3, dim=2)
        >>> x = torch.randn(4, 3, 64, 64)  # batch=4, channels=3, 64x64 spatial
        >>> y = fno(x)  # shape: (4, 1, 64, 64)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int | tuple[int, int],
        width: int,
        num_layers: int,
        dim: Literal[1, 2] = 1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        self.dim = dim

        # Lifting layer: project input to hidden width
        if dim == 1:
            self.lifting = nn.Conv1d(in_channels, width, 1)
            self.projection = nn.Conv1d(width, out_channels, 1)
        else:
            self.lifting = nn.Conv2d(in_channels, width, 1)
            self.projection = nn.Conv2d(width, out_channels, 1)

        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer(width, modes, dim) for _ in range(num_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """Apply the Fourier Neural Operator.

        Args:
            x: Input tensor of shape:
               - 1D: (batch, in_channels, spatial_dim)
               - 2D: (batch, in_channels, height, width)

        Returns:
            Output tensor of shape:
               - 1D: (batch, out_channels, spatial_dim)
               - 2D: (batch, out_channels, height, width)
        """
        # Lift input to hidden dimension
        x = self.lifting(x)

        # Apply Fourier layers
        for layer in self.fourier_layers:
            x = layer(x)

        # Project to output dimension
        x = self.projection(x)

        return x

    def get_num_params(self) -> int:
        """Return the total number of trainable parameters.

        Returns:
            Total count of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
