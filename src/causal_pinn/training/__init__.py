"""Training module for the PINN framework.

This module provides the 3-stage training pipeline, callbacks,
and early-training prediction for training policy selection.
"""

from .adaptive_lambda import (
    AdaptiveLambdaController,
    AdaptiveLambdaMetrics,
)
from .callbacks import (
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    GradientClippingCallback,
    LoggingCallback,
    LRSchedulerCallback,
)
from .early_predictor import (
    EarlyTrainingPredictor,
    FeatureExtractor,
    PredictorDataset,
    PredictorSample,
)
from .policy_routing import (
    budget_ratio,
    budget_savings,
    evaluate_policy_routing,
    normalize_pde_group,
    normalize_policy_name,
)
from .trainer import HybridTrainer, TrainingHistory

__all__ = [
    # Trainer
    "HybridTrainer",
    "TrainingHistory",
    "AdaptiveLambdaController",
    "AdaptiveLambdaMetrics",
    # Early Predictor
    "EarlyTrainingPredictor",
    "FeatureExtractor",
    "PredictorDataset",
    "PredictorSample",
    "budget_ratio",
    "budget_savings",
    "evaluate_policy_routing",
    "normalize_pde_group",
    "normalize_policy_name",
    # Callbacks
    "Callback",
    "CheckpointCallback",
    "LoggingCallback",
    "EarlyStoppingCallback",
    "LRSchedulerCallback",
    "GradientClippingCallback",
]
