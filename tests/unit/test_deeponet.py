from __future__ import annotations

import pytest
import torch

from causal_pinn.models.operators import DeepONet


def test_deeponet_forward_shape_1d() -> None:
    model = DeepONet(
        in_channels=2,
        out_channels=1,
        branch_dim=64,
        trunk_dim=64,
        num_layers=3,
        dim=1,
    )
    x = torch.randn(8, 2, 1)

    out = model(x)

    assert out.shape == (8, 1, 1)


def test_deeponet_gradient_flow() -> None:
    model = DeepONet(in_channels=2, out_channels=1, dim=1)
    x = torch.randn(8, 2, 1, requires_grad=True)

    out = model(x)
    out.sum().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    for _, parameter in model.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("in_channels", [1, 2, 3])
def test_deeponet_supports_different_input_channels(in_channels: int) -> None:
    model = DeepONet(in_channels=in_channels, out_channels=1, dim=1)
    x = torch.randn(4, in_channels, 1)

    out = model(x)

    assert out.shape == (4, 1, 1)


def test_deeponet_output_requires_grad_when_input_requires_grad() -> None:
    model = DeepONet(in_channels=2, out_channels=1, dim=1)
    x = torch.randn(4, 2, 1, requires_grad=True)

    out = model(x)

    assert out.requires_grad


@pytest.mark.parametrize("batch_size", [1, 2, 8, 16])
def test_deeponet_batch_size_consistency(batch_size: int) -> None:
    model = DeepONet(in_channels=2, out_channels=1, dim=1)
    x = torch.randn(batch_size, 2, 1)

    out = model(x)

    assert out.shape[0] == batch_size
    assert out.shape == (batch_size, 1, 1)
