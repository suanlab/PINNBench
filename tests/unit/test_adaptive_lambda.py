from __future__ import annotations

from causal_pinn.training.adaptive_lambda import (
    AdaptiveLambdaController,
    AdaptiveLambdaMetrics,
)


def _metrics(
    grad_cosine: float,
    grad_norm_ratio: float,
    physics_improvement: float,
) -> AdaptiveLambdaMetrics:
    return AdaptiveLambdaMetrics(
        grad_cosine=grad_cosine,
        grad_norm_ratio=grad_norm_ratio,
        physics_improvement=physics_improvement,
        objective_grad_norm=1.0,
        physics_grad_norm=grad_norm_ratio,
    )


def test_controller_respects_warmup() -> None:
    controller = AdaptiveLambdaController(
        warmup_epochs=2,
        min_scale=0.5,
        max_scale=1.5,
        adjust_rate=0.5,
        alignment_threshold=0.1,
        conflict_threshold=-0.1,
        dominance_threshold=1.5,
        progress_threshold=0.01,
        ema_beta=0.9,
        persistence=2,
    )

    lambda_epoch0 = controller.update(epoch=0, base_lambda=2.0, metrics=_metrics(0.5, 1.0, 0.0))
    lambda_epoch1 = controller.update(epoch=1, base_lambda=2.0, metrics=_metrics(0.5, 1.0, 0.0))

    assert lambda_epoch0 == 2.0
    assert lambda_epoch1 == 2.0


def test_controller_increases_when_aligned_and_stalled() -> None:
    controller = AdaptiveLambdaController(
        warmup_epochs=0,
        min_scale=0.5,
        max_scale=1.5,
        adjust_rate=0.2,
        alignment_threshold=0.1,
        conflict_threshold=-0.1,
        dominance_threshold=1.5,
        progress_threshold=0.01,
        ema_beta=0.0,
        persistence=2,
    )

    first = controller.update(epoch=0, base_lambda=1.0, metrics=_metrics(0.6, 1.0, 0.0))
    second = controller.update(epoch=1, base_lambda=1.0, metrics=_metrics(0.6, 1.0, 0.0))

    assert first == 1.0
    assert second == 1.2


def test_controller_decreases_on_conflict() -> None:
    controller = AdaptiveLambdaController(
        warmup_epochs=0,
        min_scale=0.5,
        max_scale=1.5,
        adjust_rate=0.2,
        alignment_threshold=0.1,
        conflict_threshold=-0.1,
        dominance_threshold=1.5,
        progress_threshold=0.01,
        ema_beta=0.0,
        persistence=2,
    )
    controller.current_scale = 1.0

    first = controller.update(epoch=2, base_lambda=1.0, metrics=_metrics(-0.4, 2.0, 0.0))
    second = controller.update(epoch=3, base_lambda=1.0, metrics=_metrics(-0.4, 2.0, 0.0))

    assert first == 1.0
    assert second == 0.8


def test_compute_improvement_tracks_previous_loss() -> None:
    controller = AdaptiveLambdaController(
        warmup_epochs=0,
        min_scale=0.5,
        max_scale=1.5,
        adjust_rate=0.5,
        alignment_threshold=0.1,
        conflict_threshold=-0.1,
        dominance_threshold=1.5,
        progress_threshold=0.01,
        ema_beta=0.9,
        persistence=2,
    )

    first = controller.compute_improvement(2.0)
    second = controller.compute_improvement(1.0)

    assert first == 0.0
    assert second == 0.5
