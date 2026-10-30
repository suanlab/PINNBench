"""Adaptive epsilon scheduling for causal loss.

Prevents causal weight collapse when training without operator pretraining.
When PDE residuals are high (no pretrain), epsilon shrinks so causal weights
stay near-uniform instead of collapsing to zero for late time slices.
When residuals are low (after pretrain), epsilon stays near epsilon_base
so temporal bias is active.

The key formula:
    epsilon_adaptive = epsilon_base / (1 + beta * mean_residual)
"""

from __future__ import annotations


class AdaptiveEpsilonScheduler:
    """Dynamically adjusts causal epsilon based on current residual magnitude.

    epsilon_adaptive = epsilon_base / (1 + beta * mean_residual)

    When residuals are high (no pretrain): epsilon shrinks -> weights stay near-uniform
    When residuals are low (after pretrain): epsilon stays near epsilon_base -> temporal bias active

    Attributes:
        epsilon_base: Base epsilon value used when residuals are near zero.
        beta: Sensitivity to residual magnitude. Higher beta = more aggressive scaling.
        epsilon_min: Floor on epsilon to prevent it from becoming negligibly small.
        epsilon_max: Ceiling on epsilon to prevent runaway causal weighting.
        history: List of dicts recording each (mean_residual, epsilon) pair.
    """

    def __init__(
        self,
        epsilon_base: float = 1.0,
        beta: float = 10.0,
        epsilon_min: float = 0.01,
        epsilon_max: float = 10.0,
    ) -> None:
        """Initialize adaptive epsilon scheduler.

        Args:
            epsilon_base: Base epsilon when residuals are near zero.
            beta: Sensitivity parameter. Higher values make epsilon respond
                  more aggressively to high residuals.
            epsilon_min: Minimum allowed epsilon value.
            epsilon_max: Maximum allowed epsilon value.
        """
        self.epsilon_base = epsilon_base
        self.beta = beta
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.history: list[dict[str, float]] = []

    def compute_epsilon(self, mean_residual: float) -> float:
        """Compute adaptive epsilon from current mean residual.

        Args:
            mean_residual: Mean squared PDE residual over collocation points.

        Returns:
            Adapted epsilon value, clamped to [epsilon_min, epsilon_max].
        """
        eps = self.epsilon_base / (1.0 + self.beta * mean_residual)
        eps = max(self.epsilon_min, min(self.epsilon_max, eps))
        self.history.append({"mean_residual": mean_residual, "epsilon": eps})
        return eps

    def reset(self) -> None:
        """Clear the history buffer."""
        self.history.clear()

    def __repr__(self) -> str:
        return (
            f"AdaptiveEpsilonScheduler(epsilon_base={self.epsilon_base}, "
            f"beta={self.beta}, epsilon_min={self.epsilon_min}, "
            f"epsilon_max={self.epsilon_max})"
        )
