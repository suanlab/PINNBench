"""Shared experiment utilities for PINN research framework.

This module provides reusable helper functions for creating models,
configurations, and running experiments across multiple seeds.

Factory Functions:
    - create_pinn_model: Create a PINNBase model
    - create_hybrid_model: Create a HybridPINNOperator model
    - create_training_config: Create TrainingConfig with presets

Experiment Runners:
    - run_single_experiment: Train and evaluate a single model
    - run_multi_seed: Run experiment across multiple seeds
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from causal_pinn.evaluation.benchmark_runner import BenchmarkConfig, BenchmarkRunner
from causal_pinn.models.hybrid.causal_gate import CausalGate
from causal_pinn.models.hybrid.hybrid_model import HybridConfig, HybridPINNOperator
from causal_pinn.models.operators.fno import FNO
from causal_pinn.models.pinn.base import PINNBase
from causal_pinn.training.trainer import HybridTrainer
from causal_pinn.utils.config import TrainingConfig
from causal_pinn.utils.random import set_seed

if TYPE_CHECKING:
    from causal_pinn.physics.pde.base import PDEProblem


def create_pinn_model(
    hidden_dims: list[int],
    input_dim: int = 2,
    output_dim: int = 1,
) -> PINNBase:
    """Create a PINNBase model with specified architecture.

    Args:
        hidden_dims: List of hidden layer dimensions, e.g., [64, 64, 64].
        input_dim: Input dimension (default: 2 for (t, x) coordinates).
        output_dim: Output dimension (default: 1 for scalar u).

    Returns:
        PINNBase model instance.
    """
    return PINNBase(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims,
    )


def create_hybrid_model(
    pinn_dims: list[int],
    fno_modes: int,
    fno_width: int,
    pde_problem: PDEProblem,
    config: HybridConfig | None = None,
    fno_num_layers: int = 4,
    input_dim: int = 2,
    output_dim: int = 1,
) -> HybridPINNOperator:
    """Create a HybridPINNOperator combining PINN, FNO, and CausalGate.

    Args:
        pinn_dims: Hidden dimensions for PINN network.
        fno_modes: Number of Fourier modes for FNO.
        fno_width: Hidden width for FNO.
        pde_problem: PDE problem instance.
        config: HybridConfig for component enablement. Defaults to all enabled.
        fno_num_layers: Number of FNO layers.
        input_dim: Input dimension.
        output_dim: Output dimension.

    Returns:
        HybridPINNOperator instance.
    """
    config = config or HybridConfig()

    # Create PINN network
    pinn_network = PINNBase(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=pinn_dims,
    )

    # Create FNO operator (1D for typical PDE problems)
    fno_dim = 2 if input_dim == 3 else 1
    operator = FNO(
        in_channels=input_dim,
        out_channels=output_dim,
        modes=fno_modes,
        width=fno_width,
        num_layers=fno_num_layers,
        dim=fno_dim,
    )

    # Create CausalGate
    causal_gate = CausalGate(
        out_dim=output_dim,
        hidden_dim=32,
        causal_strength=1.0,
    )

    return HybridPINNOperator(
        pinn_network=pinn_network,
        operator=operator,
        causal_gate=causal_gate,
        pde_problem=pde_problem,
        config=config,
    )


def create_training_config(
    preset: str = "cpu_fast",
    seed: int = 42,
    device: str = "cpu",
) -> TrainingConfig:
    """Create TrainingConfig with predefined presets.

    Args:
        preset: Configuration preset name.
            - "cpu_fast": Small epochs for quick runs and debugging.
            - "cpu_full": Larger runs for paper-quality results.
        seed: Random seed for reproducibility.
        device: Device to train on ("cpu" or "cuda").

    Returns:
        TrainingConfig instance.

    Raises:
        ValueError: If preset name is not recognized.
    """
    presets = {
        "cpu_fast": {
            "stage1_epochs": 50,
            "stage1_lr": 1e-3,
            "stage2_epochs": 100,
            "stage2_lr": 5e-4,
            "stage3_epochs": 50,
            "stage3_lr": 1e-4,
            "batch_size": 64,
            "num_collocation_points": 512,
            "num_boundary_points": 64,
            "causal_epsilon": 1.0,
        },
        "cpu_full": {
            "stage1_epochs": 200,
            "stage1_lr": 1e-3,
            "stage2_epochs": 400,
            "stage2_lr": 1e-4,
            "stage3_epochs": 200,
            "stage3_lr": 1e-5,
            "batch_size": 128,
            "num_collocation_points": 2000,
            "num_boundary_points": 256,
            "causal_epsilon": 1.0,
        },
    }

    if preset not in presets:
        raise ValueError(
            f"Unknown preset: {preset}. Available: {list(presets.keys())}"
        )

    preset_config = presets[preset]
    preset_config["seed"] = seed
    preset_config["device"] = device

    return TrainingConfig(**preset_config)


def run_single_experiment(
    model: torch.nn.Module,
    pde_problem: PDEProblem,
    config: TrainingConfig,
    seed: int | None = None,
    callbacks: list | None = None,
    evaluate: bool = True,
) -> dict[str, Any]:
    """Run a single training experiment with evaluation.

    Args:
        model: Neural network model to train.
        pde_problem: PDE problem instance.
        config: Training configuration.
        seed: Random seed (overrides config.seed if provided).
        callbacks: Optional list of training callbacks.
        evaluate: Whether to run evaluation after training.

    Returns:
        Dict containing:
            - history: Training history from trainer
            - metrics: Evaluation metrics (if evaluate=True)
            - seed: Seed used for the experiment
    """
    # Set seed
    actual_seed = seed if seed is not None else config.seed
    set_seed(actual_seed)

    # Create trainer
    trainer = HybridTrainer(
        model=model,
        pde_problem=pde_problem,
        config=config,
        callbacks=callbacks or [],
        device=config.device,
    )

    # Train
    history = trainer.train()

    results: dict[str, Any] = {
        "history": history,
        "seed": actual_seed,
    }

    # Evaluate
    if evaluate:
        benchmark_config = BenchmarkConfig(
            n_collocation=500,
            n_boundary=50,
            n_initial=50,
            device=config.device,
        )
        runner = BenchmarkRunner(config=benchmark_config)
        metrics = runner.run_single(model, pde_problem)
        results["metrics"] = metrics

    return results


def run_multi_seed(
    model_factory: Callable[[], torch.nn.Module],
    pde_problem: PDEProblem,
    config: TrainingConfig,
    seeds: list[int],
    callbacks: list | None = None,
    evaluate: bool = True,
) -> dict[int, dict[str, Any]]:
    """Run the same experiment across multiple seeds.

    Args:
        model_factory: Callable that creates a fresh model instance.
            This is called for each seed to ensure independent models.
        pde_problem: PDE problem instance.
        config: Training configuration.
        seeds: List of random seeds to run.
        callbacks: Optional list of training callbacks.
        evaluate: Whether to run evaluation after training.

    Returns:
        Dict mapping seed -> experiment results dict.
    """
    results_by_seed: dict[int, dict[str, Any]] = {}

    for seed in seeds:
        # Create fresh model for each seed
        model = model_factory()

        # Run experiment with this seed
        results = run_single_experiment(
            model=model,
            pde_problem=pde_problem,
            config=config,
            seed=seed,
            callbacks=callbacks,
            evaluate=evaluate,
        )

        results_by_seed[seed] = results

    return results_by_seed


def create_model_factory(
    model_type: str = "hybrid",
    pinn_dims: list[int] | None = None,
    fno_modes: int = 16,
    fno_width: int = 64,
    pde_problem: PDEProblem | None = None,
    hybrid_config: HybridConfig | None = None,
    input_dim: int = 2,
    output_dim: int = 1,
) -> Callable[[], torch.nn.Module]:
    """Create a factory function that produces fresh model instances.

    Useful for multi-seed experiments where each seed needs an independent model.

    Args:
        model_type: Type of model to create ("pinn" or "hybrid").
        pinn_dims: Hidden dimensions for PINN.
        fno_modes: Fourier modes for FNO (hybrid only).
        fno_width: Hidden width for FNO (hybrid only).
        pde_problem: PDE problem (required for hybrid).
        hybrid_config: HybridConfig (hybrid only).
        input_dim: Input dimension.
        output_dim: Output dimension.

    Returns:
        Callable that creates a new model instance each time it's called.
    """
    pinn_dims = pinn_dims or [64, 64, 64]

    if model_type == "pinn":

        def factory() -> PINNBase:
            return create_pinn_model(
                hidden_dims=pinn_dims,
                input_dim=input_dim,
                output_dim=output_dim,
            )

    elif model_type == "hybrid":
        if pde_problem is None:
            raise ValueError("pde_problem is required for hybrid model type")

        def factory() -> HybridPINNOperator:
            return create_hybrid_model(
                pinn_dims=pinn_dims,
                fno_modes=fno_modes,
                fno_width=fno_width,
                pde_problem=pde_problem,
                config=hybrid_config,
                input_dim=input_dim,
                output_dim=output_dim,
            )

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'pinn' or 'hybrid'.")

    return factory
