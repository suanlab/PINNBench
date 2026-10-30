"""Causal-aware temporal loss for PINNs.

Implements the causal training strategy where temporal points are weighted
by exponential decay based on cumulative residual at earlier times. This
ensures earlier time points are optimized before later ones, following the
physical causality of time-dependent PDEs.

Reference:
    Wang et al., "Understanding and Mitigating Gradient Pathologies in
    Physics-Informed Neural Networks", 2022.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from torch import Tensor

if TYPE_CHECKING:
    from causal_pinn.models.pinn.base import PINNBase


@runtime_checkable
class PDEProblemProtocol(Protocol):
    """Protocol for PDE problem interface."""

    def residual(self, u: Tensor, t: Tensor, x: Tensor) -> Tensor:
        """Compute PDE residual."""
        ...


class CausalLoss:
    """Causal-aware temporal loss for PINNs.

    Implements the causal weighting strategy where the loss at time t
    is weighted by exp(-epsilon * cumulative_residual_before_t).

    The key insight is that for time-dependent PDEs, the solution at
    later times depends on the solution at earlier times. By weighting
    the loss according to how well earlier times are learned, we ensure
    proper temporal causality during training.

    The causal weight for time t_n is:
        w_n = exp(-epsilon * sum_{m < n} MSE(residual_m))

    Where residuals are computed at discrete time points and sorted
    chronologically.

    Attributes:
        pde_problem: The PDE problem defining the residual.
        epsilon: Causality strength parameter (larger = stronger causality).
        num_time_bins: Number of time bins for computing cumulative weights.
        weight: Overall weight for the causal loss.
    """

    def __init__(
        self,
        pde_problem: PDEProblemProtocol,
        epsilon: float = 1.0,
        num_time_bins: int = 100,
        weight: float = 1.0,
        min_weight: float = 0.0,
    ) -> None:
        """Initialize causal loss.

        Args:
            pde_problem: PDE problem with residual method.
            epsilon: Causality strength parameter. Higher values enforce
                    stricter causality (earlier times must be learned first).
                    Typical values: 0.1 to 10.0.
            num_time_bins: Number of discrete time bins for computing
                          cumulative weights. More bins = finer temporal resolution.
            weight: Overall weight multiplier for the causal loss.
            min_weight: Lower clamp on the per-point causal weight, applied
                    after the exponential decay. ``min_weight=0.0`` (default)
                    preserves the original Wang et al. behaviour. Values >0
                    isolate "weight magnitude vanishing" from "gradient
                    direction conflict" by guaranteeing a non-trivial
                    minimum gradient at later times.
        """
        self.pde_problem = pde_problem
        self.epsilon = epsilon
        self.num_time_bins = num_time_bins
        self.weight = weight
        self.min_weight = min_weight

    def compute_causal_weights(
        self,
        t: Tensor,
        residuals: Tensor,
    ) -> Tensor:
        """Compute causal weights for each sample point.

        The weight at time t_i is:
            w_i = exp(-epsilon * sum_{t_j < t_i} MSE(residual_j))

        Args:
            t: Time coordinates, shape (N,) or (N, 1).
            residuals: PDE residuals at each point, shape (N,) or (N, 1).

        Returns:
            Causal weights tensor of shape (N,).
        """
        # Flatten tensors
        t_flat = t.flatten()
        r_flat = residuals.flatten()

        # Sort by time
        sorted_indices = torch.argsort(t_flat)
        r_sorted = r_flat[sorted_indices]

        # Compute squared residuals
        r_squared = r_sorted**2

        # Compute cumulative mean squared residual (exclusive cumulative sum)
        # Use cumulative sum normalized by count
        n = r_squared.shape[0]

        # Exclusive cumulative sum: cumsum[i] = sum of elements 0 to i-1
        # We want: for point i, weight based on residuals of points j < i
        cumulative_mse = torch.zeros_like(r_squared)

        for i in range(1, n):
            # MSE of residuals at earlier times
            cumulative_mse[i] = torch.mean(r_squared[:i])

        # Compute causal weights
        causal_weights = torch.exp(-self.epsilon * cumulative_mse)

        # Apply lower clamp (default 0.0 preserves baseline behaviour)
        if self.min_weight > 0.0:
            causal_weights = torch.clamp(causal_weights, min=self.min_weight)

        # Restore original ordering
        unsorted_weights = torch.zeros_like(causal_weights)
        unsorted_weights[sorted_indices] = causal_weights

        return unsorted_weights

    def compute_causal_weights_binned(
        self,
        t: Tensor,
        residuals: Tensor,
    ) -> Tensor:
        """Compute causal weights using time binning for efficiency.

        Discretizes time into bins and computes cumulative MSE per bin.
        More efficient for large datasets.

        Args:
            t: Time coordinates, shape (N,) or (N, 1).
            residuals: PDE residuals at each point, shape (N,) or (N, 1).

        Returns:
            Causal weights tensor of shape (N,).
        """
        # Flatten tensors
        t_flat = t.flatten()
        r_flat = residuals.flatten()

        # Get time range
        t_min = t_flat.min()
        t_max = t_flat.max()

        # Handle edge case where all times are the same
        if torch.isclose(t_min, t_max):
            return torch.ones_like(t_flat)

        # Create time bins
        bin_edges = torch.linspace(t_min, t_max, self.num_time_bins + 1, device=t.device)

        # Compute MSE per bin
        bin_mse = torch.zeros(self.num_time_bins, device=t.device)
        bin_counts = torch.zeros(self.num_time_bins, device=t.device)

        for i in range(self.num_time_bins):
            mask = (t_flat >= bin_edges[i]) & (t_flat < bin_edges[i + 1])
            # Include right edge for last bin
            if i == self.num_time_bins - 1:
                mask = (t_flat >= bin_edges[i]) & (t_flat <= bin_edges[i + 1])

            if mask.sum() > 0:
                bin_mse[i] = torch.mean(r_flat[mask] ** 2)
                bin_counts[i] = mask.sum().float()

        # Compute cumulative MSE for each bin
        cumulative_mse = torch.zeros(self.num_time_bins, device=t.device)
        for i in range(1, self.num_time_bins):
            cumulative_mse[i] = torch.sum(bin_mse[:i] * bin_counts[:i]) / (
                bin_counts[:i].sum() + 1e-8
            )

        # Compute bin weights
        bin_weights = torch.exp(-self.epsilon * cumulative_mse)

        # Apply lower clamp (default 0.0 preserves baseline behaviour)
        if self.min_weight > 0.0:
            bin_weights = torch.clamp(bin_weights, min=self.min_weight)

        # Assign weights to points based on their time bin
        weights = torch.ones_like(t_flat)
        for i in range(self.num_time_bins):
            mask = (t_flat >= bin_edges[i]) & (t_flat < bin_edges[i + 1])
            if i == self.num_time_bins - 1:
                mask = (t_flat >= bin_edges[i]) & (t_flat <= bin_edges[i + 1])
            weights[mask] = bin_weights[i]

        return weights

    def __call__(
        self,
        network: PINNBase,
        t: Tensor,
        x: Tensor,
        use_binning: bool = True,
    ) -> dict[str, Tensor]:
        """Compute causal-weighted physics loss.

        Args:
            network: Neural network approximating the solution.
            t: Time coordinates at collocation points.
            x: Spatial coordinates at collocation points.
            use_binning: If True, use binned computation for efficiency.

        Returns:
            Dictionary containing:
                - 'residual': Raw PDE residual tensor
                - 'causal_weights': Causal weight per point
                - 'weighted_mse': Causal-weighted MSE
                - 'mse_residual': Unweighted MSE (for comparison)
                - 'causal_loss': Weighted causal loss
                - 'total': Total weighted loss
        """
        # Ensure gradients are tracked
        if not t.requires_grad:
            t = t.clone().requires_grad_(True)
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)

        t_leaf = t
        x_leaf = x

        # Ensure 2D tensors for network input
        t_2d = t.unsqueeze(-1) if t.dim() == 1 else t
        x_2d = x.unsqueeze(-1) if x.dim() == 1 else x

        # Concatenate inputs
        inputs = torch.cat([t_2d, x_2d], dim=-1)

        # Forward pass
        u = network(inputs)

        # Compute PDE residual
        residual = self.pde_problem.residual(
            u.squeeze(-1), t_leaf, x_leaf
        )

        # Compute causal weights
        if use_binning:
            causal_weights = self.compute_causal_weights_binned(t, residual)
        else:
            causal_weights = self.compute_causal_weights(t, residual)

        # Compute losses
        residual_squared = residual**2
        mse_residual = torch.mean(residual_squared)
        weighted_mse = torch.mean(causal_weights * residual_squared)

        # Apply overall weight
        causal_loss = self.weight * weighted_mse

        return {
            "residual": residual,
            "causal_weights": causal_weights,
            "weighted_mse": weighted_mse,
            "mse_residual": mse_residual,
            "causal_loss": causal_loss,
            "total": causal_loss,
        }


class AdaptiveCausalLoss(CausalLoss):
    """Causal loss with adaptive epsilon scheduling.

    The epsilon parameter is automatically adjusted during training:
    - Start with low epsilon (less causality) to allow exploration
    - Increase epsilon as training progresses to enforce causality

    This helps avoid local minima in early training while ensuring
    causal consistency in later stages.
    """

    def __init__(
        self,
        pde_problem: PDEProblemProtocol,
        epsilon_init: float = 0.01,
        epsilon_final: float = 100.0,
        num_time_bins: int = 100,
        weight: float = 1.0,
        annealing_steps: int = 10000,
        min_weight: float = 0.0,
    ) -> None:
        """Initialize adaptive causal loss.

        Args:
            pde_problem: PDE problem with residual method.
            epsilon_init: Initial (low) causality strength.
            epsilon_final: Final (high) causality strength.
            num_time_bins: Number of discrete time bins.
            weight: Overall weight multiplier.
            annealing_steps: Number of steps to anneal epsilon.
            min_weight: Lower clamp on per-point causal weight (passed to
                ``CausalLoss``). Default ``0.0`` preserves baseline behaviour.
        """
        super().__init__(
            pde_problem,
            epsilon_init,
            num_time_bins,
            weight,
            min_weight=min_weight,
        )
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.annealing_steps = annealing_steps
        self._current_step = 0

    def step(self) -> None:
        """Advance the annealing schedule by one step."""
        self._current_step += 1
        # Linear annealing
        progress = min(1.0, self._current_step / self.annealing_steps)
        self.epsilon = self.epsilon_init + progress * (
            self.epsilon_final - self.epsilon_init
        )

    def set_step(self, step: int) -> None:
        """Set the current annealing step directly.

        Args:
            step: Current training step.
        """
        self._current_step = step
        progress = min(1.0, step / self.annealing_steps)
        self.epsilon = self.epsilon_init + progress * (
            self.epsilon_final - self.epsilon_init
        )

    def __call__(
        self,
        network: PINNBase,
        t: Tensor,
        x: Tensor,
        use_binning: bool = True,
    ) -> dict[str, Tensor]:
        """Compute adaptive causal-weighted physics loss.

        Args:
            network: Neural network.
            t: Time coordinates.
            x: Spatial coordinates.
            use_binning: Use binned computation.

        Returns:
            Loss dictionary with additional 'epsilon' field.
        """
        result = super().__call__(network, t, x, use_binning)
        result["epsilon"] = torch.tensor(self.epsilon)
        return result


class TimeDecayLoss:
    """Alternative causal loss using direct time-based exponential decay.

    Simpler than CausalLoss, uses:
        w(t) = exp(-alpha * t)

    Where alpha controls the decay rate. This weights earlier times
    more heavily without considering residual accumulation.
    """

    def __init__(
        self,
        pde_problem: PDEProblemProtocol,
        alpha: float = 1.0,
        weight: float = 1.0,
    ) -> None:
        """Initialize time decay loss.

        Args:
            pde_problem: PDE problem with residual method.
            alpha: Time decay rate. Higher = stronger preference for early times.
            weight: Overall weight multiplier.
        """
        self.pde_problem = pde_problem
        self.alpha = alpha
        self.weight = weight

    def __call__(
        self,
        network: PINNBase,
        t: Tensor,
        x: Tensor,
    ) -> dict[str, Tensor]:
        """Compute time-decay weighted physics loss.

        Args:
            network: Neural network.
            t: Time coordinates.
            x: Spatial coordinates.

        Returns:
            Loss dictionary with time-decay weights.
        """
        # Ensure gradients are tracked
        if not t.requires_grad:
            t = t.clone().requires_grad_(True)
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)

        t_leaf = t
        x_leaf = x

        # Ensure 2D tensors
        t_2d = t.unsqueeze(-1) if t.dim() == 1 else t
        x_2d = x.unsqueeze(-1) if x.dim() == 1 else x

        # Concatenate inputs
        inputs = torch.cat([t_2d, x_2d], dim=-1)

        # Forward pass
        u = network(inputs)

        # Compute PDE residual
        residual = self.pde_problem.residual(
            u.squeeze(-1), t_leaf, x_leaf
        )

        # Compute time-based decay weights
        t_normalized = (t - t.min()) / (t.max() - t.min() + 1e-8)
        decay_weights = torch.exp(-self.alpha * t_normalized)

        # Compute losses
        residual_squared = residual**2
        mse_residual = torch.mean(residual_squared)
        weighted_mse = torch.mean(decay_weights * residual_squared)

        # Apply overall weight
        loss = self.weight * weighted_mse

        return {
            "residual": residual,
            "decay_weights": decay_weights,
            "weighted_mse": weighted_mse,
            "mse_residual": mse_residual,
            "time_decay_loss": loss,
            "total": loss,
        }
