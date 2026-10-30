"""Evaluation metrics for PINN performance assessment.

This module provides a comprehensive suite of metrics for evaluating
Physics-Informed Neural Network (PINN) solutions against exact solutions
and physical constraints.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def relative_l2_error(pred: Tensor, true: Tensor) -> Tensor:
    """Compute relative L2 error between prediction and ground truth.

    ||pred - true||_2 / ||true||_2

    Args:
        pred: Predicted values tensor.
        true: Ground truth values tensor.

    Returns:
        Scalar tensor containing the relative L2 error.
    """
    pred = pred.reshape(-1)
    true = true.reshape(-1)

    error_norm = torch.norm(pred - true, p=2)
    true_norm = torch.norm(true, p=2)

    # Handle zero true_norm case
    if true_norm == 0:
        return torch.tensor(0.0 if error_norm == 0 else float('inf'),
                           dtype=pred.dtype, device=pred.device)

    return error_norm / true_norm  # type: ignore[no-any-return]


def mae(pred: Tensor, true: Tensor) -> Tensor:
    """Compute mean absolute error between prediction and ground truth.

    Args:
        pred: Predicted values tensor.
        true: Ground truth values tensor.

    Returns:
        Scalar tensor containing the MAE.
    """
    pred = pred.reshape(-1)
    true = true.reshape(-1)

    return torch.mean(torch.abs(pred - true))


def max_error(pred: Tensor, true: Tensor) -> Tensor:
    """Compute maximum pointwise error between prediction and ground truth.

    Args:
        pred: Predicted values tensor.
        true: Ground truth values tensor.

    Returns:
        Scalar tensor containing the maximum absolute error.
    """
    pred = pred.reshape(-1)
    true = true.reshape(-1)

    return torch.max(torch.abs(pred - true))


def mse(pred: Tensor, true: Tensor) -> Tensor:
    """Compute mean squared error between prediction and ground truth.

    Args:
        pred: Predicted values tensor.
        true: Ground truth values tensor.

    Returns:
        Scalar tensor containing the MSE.
    """
    pred = pred.reshape(-1)
    true = true.reshape(-1)

    return torch.mean((pred - true) ** 2)


def pde_residual_norm(residuals: Tensor) -> Tensor:
    """Compute RMS (root mean square) of PDE residuals.

    Args:
        residuals: Tensor of PDE residual values.

    Returns:
        Scalar tensor containing the RMS of residuals.
    """
    residuals = residuals.reshape(-1)

    return torch.sqrt(torch.mean(residuals ** 2))


def bc_violation(pred_bc: Tensor, true_bc: Tensor) -> Tensor:
    """Compute MSE at boundary conditions.

    Args:
        pred_bc: Predicted boundary values tensor.
        true_bc: True boundary values tensor.

    Returns:
        Scalar tensor containing the MSE at boundaries.
    """
    return mse(pred_bc, true_bc)


def ic_violation(pred_ic: Tensor, true_ic: Tensor) -> Tensor:
    """Compute MSE at initial conditions.

    Args:
        pred_ic: Predicted initial condition values tensor.
        true_ic: True initial condition values tensor.

    Returns:
        Scalar tensor containing the MSE at initial conditions.
    """
    return mse(pred_ic, true_ic)


def convergence_failure_rate(
    losses_per_seed: Sequence[Sequence[float]],
    threshold: float
) -> Tensor:
    """Compute fraction of seeds where final loss exceeds threshold.

    Args:
        losses_per_seed: List of loss histories, one per random seed.
                        Each inner list contains losses over training iterations.
        threshold: Maximum acceptable final loss value.

    Returns:
        Scalar tensor containing the failure rate (between 0 and 1).
    """
    if not losses_per_seed:
        return torch.tensor(0.0)

    failures = 0
    total = len(losses_per_seed)

    for loss_history in losses_per_seed:
        if not loss_history or loss_history[-1] > threshold:
            failures += 1

    return torch.tensor(failures / total, dtype=torch.float32)


def seed_variance(final_losses: Sequence[float]) -> Tensor:
    """Compute variance of final losses across random seeds.

    Args:
        final_losses: List of final loss values, one per random seed.

    Returns:
        Scalar tensor containing the variance across seeds.
    """
    if len(final_losses) < 2:
        return torch.tensor(0.0, dtype=torch.float32)

    losses_tensor = torch.tensor(final_losses, dtype=torch.float32)
    return torch.var(losses_tensor)


class MetricSuite:
    """Container for computing multiple metrics at once.

    Attributes:
        metric_names: List of metric names to compute.
        _metrics: Dictionary mapping metric names to functions.
    """

    _AVAILABLE_METRICS: dict[str, callable] = {
        'relative_l2_error': relative_l2_error,
        'mae': mae,
        'max_error': max_error,
        'mse': mse,
        'pde_residual_norm': pde_residual_norm,
        'bc_violation': bc_violation,
        'ic_violation': ic_violation,
    }

    def __init__(self, metric_names: list[str]) -> None:
        """Initialize the metric suite with specified metrics.

        Args:
            metric_names: List of metric names to include in the suite.

        Raises:
            ValueError: If any metric name is not recognized.
        """
        unknown = set(metric_names) - set(self._AVAILABLE_METRICS.keys())
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}. "
                           f"Available: {list(self._AVAILABLE_METRICS.keys())}")

        self.metric_names = metric_names
        self._metrics = {name: self._AVAILABLE_METRICS[name]
                        for name in metric_names}

    def compute(self, pred: Tensor, true: Tensor) -> dict[str, Tensor]:
        """Compute all configured metrics for given prediction and ground truth.

        Args:
            pred: Predicted values tensor.
            true: Ground truth values tensor.

        Returns:
            Dictionary mapping metric names to computed values.
        """
        results = {}
        for name, metric_fn in self._metrics.items():
            results[name] = metric_fn(pred, true)
        return results

    @classmethod
    def available_metrics(cls) -> list[str]:
        """Return list of all available metric names.

        Returns:
            List of metric names that can be used.
        """
        return list(cls._AVAILABLE_METRICS.keys())
