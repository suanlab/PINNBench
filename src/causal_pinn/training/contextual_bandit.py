"""Contextual bandit and successive halving routers for policy selection.

LinUCB implements the Linear Upper Confidence Bound algorithm for
contextual bandits, where each PDE's early-training features form the
context and each training policy is an arm.  The reward is the negative
final metric (lower metric = higher reward).

SuccessiveHalving implements the successive-halving budget-allocation
strategy: run all policies for a short probe, discard the worst half,
repeat until one policy remains.

Both routers are designed to plug into the same leave-one-PDE-out
cross-validation framework used by the RF and heuristic selectors in
``policy_routing.py``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# LinUCB contextual bandit
# ---------------------------------------------------------------------------


class LinUCBRouter:
    """LinUCB contextual bandit for policy selection.

    Each PDE's early training features form the context.
    Each policy is an arm.
    The reward is the negative final metric (lower metric = higher reward).

    Attributes:
        n_features: Dimension of context (early training features).
        alpha: Exploration parameter (higher = more exploration).
        arms: Per-arm state dictionaries with keys ``A`` and ``b``.
    """

    def __init__(self, n_features: int, alpha: float = 1.0) -> None:
        """Initialise the router.

        Args:
            n_features: Dimension of context (early training features).
            alpha: Exploration parameter (higher = more exploration).
        """
        self.arms: dict[str, dict[str, NDArray[np.float64]]] = {}
        self.n_features = n_features
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_arm(self, arm: str) -> None:
        """Lazily initialise per-arm parameters on first encounter."""
        if arm not in self.arms:
            self.arms[arm] = {
                "A": np.eye(self.n_features, dtype=np.float64),
                "b": np.zeros(self.n_features, dtype=np.float64),
            }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def select_arm(self, context: NDArray[np.float64], available_arms: list[str]) -> str:
        """Select the best arm given *context* using the UCB criterion.

        For each arm *a* the UCB score is::

            p_a = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)

        where ``theta_a = A_a^{-1} b_a``.

        Args:
            context: 1-D feature vector of length ``n_features``.
            available_arms: List of arm (policy) names to choose from.

        Returns:
            Name of the selected arm.

        Raises:
            ValueError: If *available_arms* is empty.
        """
        if not available_arms:
            raise ValueError("available_arms must be non-empty")

        x = np.asarray(context, dtype=np.float64).ravel()
        if x.shape[0] != self.n_features:
            raise ValueError(
                f"Context dimension {x.shape[0]} != expected {self.n_features}"
            )

        best_arm = available_arms[0]
        best_score = -np.inf

        for arm in available_arms:
            self._ensure_arm(arm)
            A = self.arms[arm]["A"]
            b = self.arms[arm]["b"]
            A_inv = np.linalg.solve(A, np.eye(self.n_features))
            theta = A_inv @ b
            exploitation = float(theta @ x)
            exploration = self.alpha * float(np.sqrt(x @ A_inv @ x))
            score = exploitation + exploration
            if score > best_score:
                best_score = score
                best_arm = arm

        return best_arm

    def update(self, arm: str, context: NDArray[np.float64], reward: float) -> None:
        """Update arm parameters after observing *reward*.

        The update rules are::

            A_arm += x x^T
            b_arm += reward * x

        Args:
            arm: Name of the arm (policy) that was played.
            context: 1-D feature vector used when the arm was selected.
            reward: Observed reward (negative final metric).
        """
        x = np.asarray(context, dtype=np.float64).ravel()
        self._ensure_arm(arm)
        self.arms[arm]["A"] += np.outer(x, x)
        self.arms[arm]["b"] += reward * x

    def fit_batch(
        self,
        contexts: NDArray[np.float64],
        arms: list[str],
        rewards: NDArray[np.float64],
    ) -> None:
        """Batch fit from historical data.

        Iterates through all samples and calls :meth:`update` for each.

        Args:
            contexts: (N, n_features) array of context vectors.
            arms: Length-N list of arm names.
            rewards: Length-N array of observed rewards.
        """
        contexts = np.asarray(contexts, dtype=np.float64)
        rewards = np.asarray(rewards, dtype=np.float64)
        for ctx, arm, reward in zip(contexts, arms, rewards, strict=True):
            self.update(arm, ctx, float(reward))

    def predict_values(
        self,
        context: NDArray[np.float64],
        available_arms: list[str],
    ) -> dict[str, float]:
        """Return the estimated expected reward for each arm (exploitation only).

        Useful for ranking arms without the exploration bonus.

        Args:
            context: 1-D feature vector.
            available_arms: Arm names to score.

        Returns:
            Dict mapping arm name to predicted reward (theta^T x).
        """
        x = np.asarray(context, dtype=np.float64).ravel()
        values: dict[str, float] = {}
        for arm in available_arms:
            self._ensure_arm(arm)
            A = self.arms[arm]["A"]
            b = self.arms[arm]["b"]
            theta = np.linalg.solve(A, b)
            values[arm] = float(theta @ x)
        return values

    def reset(self) -> None:
        """Clear all learned arm parameters."""
        self.arms.clear()


# ---------------------------------------------------------------------------
# Successive Halving
# ---------------------------------------------------------------------------


class SuccessiveHalvingRouter:
    """Successive halving for policy selection.

    Given N policies and a budget of *total_epochs* training epochs:

    1. Run all N policies for ``T / (N * ceil(log2(N)))`` epochs.
    2. Keep the top N/2 policies (lowest metric).
    3. Repeat until one policy remains.

    The ``select_policy`` method simulates this process using the
    early-metric trajectories that are already available (i.e. it does
    not launch new training runs but reads from recorded loss curves).

    Attributes:
        total_epochs: Full training budget per policy.
        min_probe: Minimum epochs per halving round.
    """

    def __init__(self, total_epochs: int = 500, min_probe: int = 10) -> None:
        self.total_epochs = total_epochs
        self.min_probe = min_probe

    def select_policy(
        self,
        early_metrics: dict[str, list[float]],
    ) -> tuple[str, int]:
        """Simulate successive halving on recorded metric trajectories.

        Args:
            early_metrics: Mapping from policy name to a list of per-epoch
                metric values (e.g. total loss).  All lists should have the
                same length (== ``total_epochs``).

        Returns:
            Tuple of (best_policy_name, total_budget_used) where
            *total_budget_used* is the sum of epochs consumed across all
            policies during the halving process.
        """
        if not early_metrics:
            return ("unknown", 0)

        policies = list(early_metrics.keys())
        n = len(policies)

        if n == 1:
            return (policies[0], 0)

        n_rounds = max(1, math.ceil(math.log2(n)))
        budget_per_round = max(self.min_probe, self.total_epochs // (n * n_rounds))

        survivors = list(policies)
        epoch_cursor = 0
        total_budget = 0

        for _round_idx in range(n_rounds):
            epoch_cursor += budget_per_round
            epoch_cursor = min(epoch_cursor, self.total_epochs)

            # Budget consumed this round: epochs_this_round * n_survivors
            total_budget += budget_per_round * len(survivors)

            # Score each survivor by its metric at epoch_cursor.
            scores: dict[str, float] = {}
            for policy in survivors:
                trajectory = early_metrics[policy]
                idx = min(epoch_cursor - 1, len(trajectory) - 1)
                scores[policy] = trajectory[idx] if idx >= 0 else float("inf")

            # Keep top half (lowest scores).
            ranked = sorted(scores, key=lambda p: scores[p])
            keep = max(1, len(ranked) // 2)
            survivors = ranked[:keep]

            if len(survivors) == 1:
                break

        return (survivors[0], total_budget)

    def budget_savings(
        self,
        n_policies: int,
    ) -> float:
        """Estimate budget savings compared to exhaustive evaluation.

        The exhaustive budget is ``n_policies * total_epochs``.  Successive
        halving uses roughly ``total_epochs * (1 + 1/2 + 1/4 + ...)``
        which approaches ``2 * total_epochs`` for large N.

        Args:
            n_policies: Number of candidate policies.

        Returns:
            Fraction of budget saved (0.0 to 1.0).
        """
        if n_policies <= 1:
            return 0.0
        exhaustive = n_policies * self.total_epochs
        n_rounds = max(1, math.ceil(math.log2(n_policies)))
        budget_per_round = max(self.min_probe, self.total_epochs // (n_policies * n_rounds))

        estimated_budget = 0
        n_survivors = n_policies
        for _ in range(n_rounds):
            estimated_budget += budget_per_round * n_survivors
            n_survivors = max(1, n_survivors // 2)
            if n_survivors == 1:
                estimated_budget += budget_per_round
                break

        return 1.0 - estimated_budget / exhaustive
