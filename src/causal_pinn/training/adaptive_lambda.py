from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdaptiveLambdaMetrics:
    grad_cosine: float
    grad_norm_ratio: float
    physics_improvement: float
    objective_grad_norm: float
    physics_grad_norm: float


class AdaptiveLambdaController:

    def __init__(
        self,
        warmup_epochs: int,
        min_scale: float,
        max_scale: float,
        adjust_rate: float,
        alignment_threshold: float,
        conflict_threshold: float,
        dominance_threshold: float,
        progress_threshold: float,
        ema_beta: float,
        persistence: int,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.adjust_rate = adjust_rate
        self.alignment_threshold = alignment_threshold
        self.conflict_threshold = conflict_threshold
        self.dominance_threshold = dominance_threshold
        self.progress_threshold = progress_threshold
        self.ema_beta = ema_beta
        self.persistence = persistence

        self.current_scale = 1.0
        self.previous_physics_loss: float | None = None
        self.ema_grad_cosine: float | None = None
        self.ema_grad_norm_ratio: float | None = None
        self.ema_physics_improvement: float | None = None
        self.alignment_streak = 0
        self.conflict_streak = 0

    def update(self, epoch: int, base_lambda: float, metrics: AdaptiveLambdaMetrics) -> float:
        if epoch < self.warmup_epochs:
            self._reset_tracking()
            return base_lambda

        ema_grad_cosine = self._update_ema(self.ema_grad_cosine, metrics.grad_cosine)
        ema_grad_norm_ratio = self._update_ema(
            self.ema_grad_norm_ratio,
            metrics.grad_norm_ratio,
        )
        ema_physics_improvement = self._update_ema(
            self.ema_physics_improvement,
            metrics.physics_improvement,
        )

        self.ema_grad_cosine = ema_grad_cosine
        self.ema_grad_norm_ratio = ema_grad_norm_ratio
        self.ema_physics_improvement = ema_physics_improvement

        should_decrease = (
            ema_grad_cosine < self.conflict_threshold
            and ema_grad_norm_ratio > self.dominance_threshold
            and ema_physics_improvement < self.progress_threshold
        )
        should_increase = (
            ema_grad_cosine > self.alignment_threshold
            and ema_grad_norm_ratio <= self.dominance_threshold
            and ema_physics_improvement < self.progress_threshold
        )

        if should_decrease:
            self.conflict_streak += 1
            self.alignment_streak = 0
        elif should_increase:
            self.alignment_streak += 1
            self.conflict_streak = 0
        else:
            self.alignment_streak = 0
            self.conflict_streak = 0

        if self.conflict_streak >= self.persistence:
            self.current_scale *= 1.0 - self.adjust_rate
            self.conflict_streak = 0
        elif self.alignment_streak >= self.persistence:
            self.current_scale *= 1.0 + self.adjust_rate
            self.alignment_streak = 0

        self.current_scale = min(max(self.current_scale, self.min_scale), self.max_scale)
        return base_lambda * self.current_scale

    def compute_improvement(self, physics_loss: float) -> float:
        if self.previous_physics_loss is None:
            self.previous_physics_loss = physics_loss
            return 0.0

        baseline = max(abs(self.previous_physics_loss), 1e-8)
        improvement = (self.previous_physics_loss - physics_loss) / baseline
        self.previous_physics_loss = physics_loss
        return improvement

    def _update_ema(self, current: float | None, value: float) -> float:
        if current is None:
            return value
        return self.ema_beta * current + (1.0 - self.ema_beta) * value

    def _reset_tracking(self) -> None:
        self.previous_physics_loss = None
        self.ema_grad_cosine = None
        self.ema_grad_norm_ratio = None
        self.ema_physics_improvement = None
        self.alignment_streak = 0
        self.conflict_streak = 0
        self.current_scale = 1.0
