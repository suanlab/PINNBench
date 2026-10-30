"""Configuration utilities for experiment management."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    # PINN MLP
    pinn_hidden_dims: list[int] = field(default_factory=lambda: [64, 64, 64, 64])
    pinn_activation: str = "gelu"
    pinn_input_dim: int = 2  # (t, x) for 1D problems
    pinn_output_dim: int = 1

    # FNO — aligned with paper protocol (modes=8, width=32, 4 spectral layers).
    # Legacy defaults were modes=16, width=64; scripts overrode; now the
    # dataclass default itself matches the paper.
    fno_modes: int = 8
    fno_width: int = 32
    fno_num_layers: int = 4
    fno_dim: int = 1  # 1D or 2D


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""

    # Stage 1: Operator pretrain — aligned with paper protocol (200 epochs).
    stage1_epochs: int = 200
    stage1_lr: float = 1e-3

    # Stage 2: Physics refinement — aligned with paper protocol (500 epochs).
    stage2_epochs: int = 500
    stage2_lr: float = 1e-4
    lambda_physics: float = 1.0
    lambda_data: float = 1.0
    adaptive_lambda_enabled: bool = False
    adaptive_lambda_warmup_epochs: int = 25
    adaptive_lambda_min_scale: float = 0.5
    adaptive_lambda_max_scale: float = 1.5
    adaptive_lambda_adjust_rate: float = 0.02
    adaptive_lambda_alignment_threshold: float = 0.1
    adaptive_lambda_conflict_threshold: float = -0.1
    adaptive_lambda_dominance_threshold: float = 1.5
    adaptive_lambda_progress_threshold: float = 0.01
    adaptive_lambda_ema_beta: float = 0.9
    adaptive_lambda_persistence: int = 5

    # Stage 3: Causal calibration — aligned with paper protocol (200 epochs).
    stage3_epochs: int = 200
    stage3_lr: float = 1e-5
    causal_epsilon: float = 1.0
    # Starting value of the linear epsilon schedule in Stage 3.  With both this
    # and causal_epsilon at 0 every time bin gets weight 1, i.e. Stage 3 trains
    # on the plain (unweighted) residual: the epoch-matched control for causal
    # weighting.
    causal_epsilon_init: float = 0.01

    # Stage skip flags (for ablation studies)
    skip_stage1: bool = False
    skip_stage3: bool = False

    # General
    batch_size: int = 256
    num_collocation_points: int = 10000
    num_boundary_points: int = 1000
    seed: int = 42
    device: str = "cuda"


@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    num_eval_points: int = 10000
    eval_metrics: list[str] = field(
        default_factory=lambda: [
            "relative_l2",
            "mae",
            "pde_residual",
            "bc_violation",
            "ic_violation",
        ]
    )


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str = "default"
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "experiments/results"
