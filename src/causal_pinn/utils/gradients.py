"""Automatic differentiation utilities for PINN computations."""

from __future__ import annotations

import torch
from torch import Tensor


def gradient(
    outputs: Tensor,
    inputs: Tensor,
    order: int = 1,
    create_graph: bool = True,
) -> Tensor:
    """Compute gradient of outputs w.r.t. inputs using autograd.

    Args:
        outputs: Tensor of shape (N, 1) or (N,).
        inputs: Tensor of shape (N, D) with requires_grad=True.
        order: Order of differentiation (1 for first derivative, 2 for second, etc.).
        create_graph: Whether to create computation graph for higher-order derivatives.

    Returns:
        Gradient tensor of shape (N, D) for order=1.
    """
    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(-1)

    grad = outputs
    for _ in range(order):
        grad = torch.autograd.grad(
            grad,
            inputs,
            grad_outputs=torch.ones_like(grad),
            create_graph=create_graph,
            retain_graph=True,
        )[0]
    return grad


def jacobian_column(
    outputs: Tensor,
    inputs: Tensor,
    column: int,
    create_graph: bool = True,
) -> Tensor:
    """Compute partial derivative of outputs w.r.t. a specific input column.

    Args:
        outputs: Tensor of shape (N, 1) or (N,).
        inputs: Tensor of shape (N, D) with requires_grad=True.
        column: Index of the input dimension to differentiate w.r.t.
        create_graph: Whether to create computation graph.

    Returns:
        Partial derivative tensor of shape (N, 1).
    """
    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(-1)

    grads = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=True,
    )[0]
    return grads[:, column : column + 1]
