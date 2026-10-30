"""Evaluation and verification tools for PINN research.

This module provides comprehensive evaluation metrics, physics consistency
checking, and benchmark automation for Physics-Informed Neural Networks.

Modules:
    metrics: Evaluation metrics (L2 error, MAE, MSE, PDE residuals, etc.)
    physics_consistency: Physics conservation law verification
    benchmark_runner: Automated benchmark execution and comparison
"""

from .benchmark_runner import BenchmarkConfig, BenchmarkRunner
from .metrics import (
    MetricSuite,
    bc_violation,
    convergence_failure_rate,
    ic_violation,
    mae,
    max_error,
    mse,
    pde_residual_norm,
    relative_l2_error,
    seed_variance,
)
from .physics_consistency import PhysicsConsistencyChecker

__all__ = [
    # Metrics
    'relative_l2_error',
    'mae',
    'max_error',
    'mse',
    'pde_residual_norm',
    'bc_violation',
    'ic_violation',
    'convergence_failure_rate',
    'seed_variance',
    'MetricSuite',
    # Physics consistency
    'PhysicsConsistencyChecker',
    # Benchmark runner
    'BenchmarkConfig',
    'BenchmarkRunner',
]
