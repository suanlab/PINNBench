"""Unit tests for evaluation module.

This module tests the evaluation metrics, physics consistency checker,
and benchmark runner components.
"""

import pytest
import torch
from torch import Tensor, nn

from causal_pinn.evaluation import (
    BenchmarkConfig,
    BenchmarkRunner,
    MetricSuite,
    PhysicsConsistencyChecker,
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
from causal_pinn.physics.pde.heat_1d import Heat1D
from causal_pinn.physics.pde.wave_2d import Wave2D

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_predictions():
    """Simple prediction tensors for testing."""
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    return pred, true


@pytest.fixture
def different_predictions():
    """Predictions with known differences for testing."""
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    true = torch.tensor([1.1, 2.2, 3.0, 3.8, 5.5])
    return pred, true


@pytest.fixture
def heat1d_problem():
    """Heat1D problem instance for testing."""
    return Heat1D(alpha=1.0, length=1.0, t_max=1.0)


@pytest.fixture
def simple_pinn():
    """Simple PINN model for testing."""
    return nn.Sequential(
        nn.Linear(2, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 1)
    )


@pytest.fixture
def wave2d_problem():
    return Wave2D(wave_speed=1.0, length_x=1.0, length_y=1.0, t_max=1.0)


@pytest.fixture
def simple_pinn_2d():
    return nn.Sequential(
        nn.Linear(3, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    )


# ============================================================================
# Metric Function Tests
# ============================================================================

class TestRelativeL2Error:
    """Tests for relative_l2_error metric."""

    def test_returns_scalar(self, simple_predictions):
        """Output should be a scalar tensor."""
        pred, true = simple_predictions
        result = relative_l2_error(pred, true)
        assert result.dim() == 0, "Result should be a scalar"

    def test_zero_when_equal(self, simple_predictions):
        """Relative L2 error should be zero when predictions match truth."""
        pred, true = simple_predictions
        result = relative_l2_error(pred, true)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_known_value(self, different_predictions):
        """Test with known values."""
        pred, true = different_predictions
        # L2 norm of difference: sqrt(0.01 + 0.04 + 0 + 0.04 + 0.25) = sqrt(0.34)
        # L2 norm of true: sqrt(1.21 + 4.84 + 9 + 14.44 + 30.25) = sqrt(59.74)
        # Expected: sqrt(0.34) / sqrt(59.74)
        expected = torch.sqrt(torch.tensor(0.34)) / torch.sqrt(torch.tensor(59.74))
        result = relative_l2_error(pred, true)
        assert torch.isclose(result, expected, rtol=1e-4)

    def test_handles_different_shapes(self):
        """Should handle different tensor shapes."""
        pred = torch.tensor([[1.0], [2.0], [3.0]])
        true = torch.tensor([1.0, 2.0, 3.0])
        result = relative_l2_error(pred, true)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_differentiable(self, simple_predictions):
        """Metric should be differentiable."""
        pred, true = simple_predictions
        pred = pred.requires_grad_(True)
        result = relative_l2_error(pred, true)
        result.backward()
        assert pred.grad is not None


class TestMAE:
    """Tests for mean absolute error metric."""

    def test_returns_scalar(self, simple_predictions):
        """Output should be a scalar tensor."""
        pred, true = simple_predictions
        result = mae(pred, true)
        assert result.dim() == 0

    def test_zero_when_equal(self, simple_predictions):
        """MAE should be zero when predictions match truth."""
        pred, true = simple_predictions
        result = mae(pred, true)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_known_value(self, different_predictions):
        """Test with known values."""
        pred, true = different_predictions
        # MAE: (0.1 + 0.2 + 0 + 0.2 + 0.5) / 5 = 0.2
        expected = torch.tensor(0.2)
        result = mae(pred, true)
        assert torch.isclose(result, expected)


class TestMaxError:
    """Tests for maximum error metric."""

    def test_returns_scalar(self, simple_predictions):
        """Output should be a scalar tensor."""
        pred, true = simple_predictions
        result = max_error(pred, true)
        assert result.dim() == 0

    def test_zero_when_equal(self, simple_predictions):
        """Max error should be zero when predictions match truth."""
        pred, true = simple_predictions
        result = max_error(pred, true)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_known_value(self, different_predictions):
        """Test with known values."""
        pred, true = different_predictions
        # Max abs error: max(0.1, 0.2, 0, 0.2, 0.5) = 0.5
        expected = torch.tensor(0.5)
        result = max_error(pred, true)
        assert torch.isclose(result, expected)


class TestMSE:
    """Tests for mean squared error metric."""

    def test_returns_scalar(self, simple_predictions):
        """Output should be a scalar tensor."""
        pred, true = simple_predictions
        result = mse(pred, true)
        assert result.dim() == 0

    def test_zero_when_equal(self, simple_predictions):
        """MSE should be zero when predictions match truth."""
        pred, true = simple_predictions
        result = mse(pred, true)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_known_value(self, different_predictions):
        """Test with known values."""
        pred, true = different_predictions
        # MSE: (0.01 + 0.04 + 0 + 0.04 + 0.25) / 5 = 0.068
        expected = torch.tensor(0.068)
        result = mse(pred, true)
        assert torch.isclose(result, expected)

    def test_differentiable(self, simple_predictions):
        """MSE should be differentiable."""
        pred, true = simple_predictions
        pred = pred.requires_grad_(True)
        result = mse(pred, true)
        result.backward()
        assert pred.grad is not None


class TestPDEResidualNorm:
    """Tests for PDE residual norm metric."""

    def test_returns_scalar(self):
        """Output should be a scalar tensor."""
        residuals = torch.tensor([0.1, 0.2, -0.1, 0.0])
        result = pde_residual_norm(residuals)
        assert result.dim() == 0

    def test_zero_residuals(self):
        """RMS should be zero for zero residuals."""
        residuals = torch.zeros(10)
        result = pde_residual_norm(residuals)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)

    def test_known_value(self):
        """Test with known values."""
        residuals = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # RMS: sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5)
        expected = torch.sqrt(torch.tensor(7.5))
        result = pde_residual_norm(residuals)
        assert torch.isclose(result, expected)


class TestBCViolation:
    """Tests for boundary condition violation metric."""

    def test_returns_scalar(self):
        """Output should be a scalar tensor."""
        pred = torch.zeros(10)
        true = torch.zeros(10)
        result = bc_violation(pred, true)
        assert result.dim() == 0

    def test_zero_when_satisfied(self):
        """BC violation should be zero when BC is satisfied."""
        pred = torch.zeros(10)
        true = torch.zeros(10)
        result = bc_violation(pred, true)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)


class TestICViolation:
    """Tests for initial condition violation metric."""

    def test_returns_scalar(self):
        """Output should be a scalar tensor."""
        pred = torch.zeros(10)
        true = torch.zeros(10)
        result = ic_violation(pred, true)
        assert result.dim() == 0

    def test_zero_when_satisfied(self):
        """IC violation should be zero when IC is satisfied."""
        pred = torch.zeros(10)
        true = torch.zeros(10)
        result = ic_violation(pred, true)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-6)


class TestConvergenceFailureRate:
    """Tests for convergence failure rate metric."""

    def test_returns_scalar(self):
        """Output should be a scalar tensor."""
        losses = [[0.1, 0.05], [0.2, 0.15], [0.1, 0.08]]
        result = convergence_failure_rate(losses, threshold=0.1)
        assert result.dim() == 0

    def test_all_converged(self):
        """Failure rate should be 0 when all seeds converge."""
        losses = [[0.1, 0.01], [0.1, 0.02], [0.1, 0.01]]
        result = convergence_failure_rate(losses, threshold=0.1)
        assert torch.isclose(result, torch.tensor(0.0))

    def test_none_converged(self):
        """Failure rate should be 1 when no seeds converge."""
        losses = [[0.5, 0.5], [0.5, 0.6]]
        result = convergence_failure_rate(losses, threshold=0.1)
        assert torch.isclose(result, torch.tensor(1.0))

    def test_partial_failure(self):
        """Test partial failure rate calculation."""
        losses = [[0.5, 0.05], [0.5, 0.5], [0.5, 0.02], [0.5, 0.5]]
        # 2 out of 4 failed
        result = convergence_failure_rate(losses, threshold=0.1)
        assert torch.isclose(result, torch.tensor(0.5))

    def test_empty_losses(self):
        """Empty losses should return 0."""
        result = convergence_failure_rate([], threshold=0.1)
        assert torch.isclose(result, torch.tensor(0.0))


class TestSeedVariance:
    """Tests for seed variance metric."""

    def test_returns_scalar(self):
        """Output should be a scalar tensor."""
        final_losses = [0.1, 0.2, 0.15, 0.12]
        result = seed_variance(final_losses)
        assert result.dim() == 0

    def test_single_value(self):
        """Single value should have zero variance."""
        result = seed_variance([0.5])
        assert torch.isclose(result, torch.tensor(0.0))

    def test_known_variance(self):
        """Test with known variance."""
        # Values [1, 2, 3, 4, 5] have variance 2.5
        final_losses = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = seed_variance(final_losses)
        expected = torch.tensor(2.5)
        assert torch.isclose(result, expected)


# ============================================================================
# MetricSuite Tests
# ============================================================================

class TestMetricSuite:
    """Tests for MetricSuite class."""

    def test_invalid_metric_name(self):
        """Should raise ValueError for unknown metrics."""
        with pytest.raises(ValueError, match="Unknown metrics"):
            MetricSuite(['invalid_metric'])

    def test_compute_all(self, different_predictions):
        """Should compute all configured metrics."""
        pred, true = different_predictions
        suite = MetricSuite(['mse', 'mae', 'relative_l2_error'])
        results = suite.compute(pred, true)

        assert 'mse' in results
        assert 'mae' in results
        assert 'relative_l2_error' in results
        assert all(isinstance(v, Tensor) for v in results.values())

    def test_available_metrics(self):
        """Should return list of available metrics."""
        available = MetricSuite.available_metrics()
        assert 'mse' in available
        assert 'mae' in available
        assert 'relative_l2_error' in available

    def test_zero_errors_for_equal_values(self, simple_predictions):
        """All error metrics should be zero for matching values."""
        pred, true = simple_predictions
        suite = MetricSuite(['relative_l2_error', 'mae', 'mse', 'max_error'])
        results = suite.compute(pred, true)

        for name, value in results.items():
            assert torch.isclose(value, torch.tensor(0.0), atol=1e-6), \
                f"{name} should be zero for equal values"


# ============================================================================
# PhysicsConsistencyChecker Tests
# ============================================================================

class TestPhysicsConsistencyChecker:
    """Tests for PhysicsConsistencyChecker class."""

    def test_residual_distribution_shape(self):
        """Residual distribution should return correct keys."""
        checker = PhysicsConsistencyChecker()
        residuals = torch.randn(100)
        dist = checker.check_residual_distribution(residuals)

        assert 'mean' in dist
        assert 'std' in dist
        assert 'max' in dist
        assert 'min' in dist
        assert 'percentiles' in dist
        assert '25' in dist['percentiles']
        assert '50' in dist['percentiles']
        assert '75' in dist['percentiles']
        assert '95' in dist['percentiles']

    def test_residual_distribution_zero_residuals(self):
        """Distribution of zero residuals should have zero stats."""
        checker = PhysicsConsistencyChecker()
        residuals = torch.zeros(100)
        dist = checker.check_residual_distribution(residuals)

        assert torch.isclose(dist['mean'], torch.tensor(0.0))
        assert torch.isclose(dist['max'], torch.tensor(0.0))

    def test_energy_conservation_returns_tensor(self, heat1d_problem):
        """Energy conservation should return tensor of errors."""
        checker = PhysicsConsistencyChecker()

        # Create test points over domain
        n = 50
        t = torch.linspace(0.01, 0.99, 10).repeat_interleave(n)
        x = torch.linspace(0.01, 0.99, n).repeat(10)

        # Use exact solution (which conserves energy correctly)
        u = heat1d_problem.exact_solution(t, x)

        errors = checker.check_energy_conservation(u, t, x, heat1d_problem)
        assert isinstance(errors, Tensor)
        assert errors.dim() == 1

    def test_mass_conservation_returns_tensor(self, heat1d_problem):
        """Mass conservation should return tensor of errors."""
        checker = PhysicsConsistencyChecker()

        # Create test points
        n = 50
        t = torch.linspace(0.01, 0.99, 10).repeat_interleave(n)
        x = torch.linspace(0.01, 0.99, n).repeat(10)

        u = heat1d_problem.exact_solution(t, x)

        errors = checker.check_mass_conservation(u, t, x, heat1d_problem)
        assert isinstance(errors, Tensor)
        assert errors.dim() == 1


# ============================================================================
# BenchmarkRunner Tests
# ============================================================================

class TestBenchmarkRunner:
    """Tests for BenchmarkRunner class."""

    def test_run_single_returns_dict(self, simple_pinn, heat1d_problem):
        """run_single should return a dictionary of metrics."""
        runner = BenchmarkRunner()
        results = runner.run_single(simple_pinn, heat1d_problem)

        assert isinstance(results, dict)
        assert 'pde_residual_norm' in results
        assert 'bc_violation' in results
        assert 'ic_violation' in results

    def test_run_comparison_multiple_models(self, simple_pinn, heat1d_problem):
        """run_comparison should handle multiple models."""
        # Create two different models
        model1 = simple_pinn
        model2 = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        runner = BenchmarkRunner()
        models = {'model1': model1, 'model2': model2}
        results = runner.run_comparison(models, heat1d_problem)

        assert 'model1' in results
        assert 'model2' in results
        assert set(results['model1'].keys()) == set(results['model2'].keys())

    def test_custom_config(self, simple_pinn, heat1d_problem):
        """Should use custom config when provided."""
        config = BenchmarkConfig(
            n_collocation=100,
            n_boundary=20,
            n_initial=20,
            include_physics_consistency=False,
        )
        runner = BenchmarkRunner(config)
        results = runner.run_single(simple_pinn, heat1d_problem)

        assert isinstance(results, dict)

    def test_format_comparison_table(self):
        """Should format results as readable table."""
        results = {
            'model_a': {'mse': 0.01, 'mae': 0.1},
            'model_b': {'mse': 0.02, 'mae': 0.15},
        }
        table = BenchmarkRunner.format_comparison_table(results)

        assert 'model_a' in table
        assert 'model_b' in table
        assert 'mse' in table or 'mae' in table

    def test_run_single_wave2d_returns_core_metrics(self, simple_pinn_2d, wave2d_problem):
        config = BenchmarkConfig(
            n_collocation=64,
            n_boundary=32,
            n_initial=32,
            include_physics_consistency=False,
            device='cpu',
        )
        runner = BenchmarkRunner(config)
        results = runner.run_single(simple_pinn_2d, wave2d_problem)

        assert 'pde_residual_norm' in results
        assert 'bc_violation' in results
        assert 'ic_violation' in results
        assert 'relative_l2_error' in results

    def test_wave2d_result_schema_matches_heat1d_metric_keys(
        self,
        simple_pinn,
        simple_pinn_2d,
        heat1d_problem,
        wave2d_problem,
    ):
        config = BenchmarkConfig(
            n_collocation=64,
            n_boundary=32,
            n_initial=32,
            include_physics_consistency=False,
            device='cpu',
        )
        runner = BenchmarkRunner(config)
        results_1d = runner.run_single(simple_pinn, heat1d_problem)
        results_2d = runner.run_single(simple_pinn_2d, wave2d_problem)

        assert set(results_1d.keys()) == set(results_2d.keys())


# ============================================================================
# Integration Tests with Heat1D
# ============================================================================

class TestHeat1DIntegration:
    """Integration tests using Heat1D problem."""

    def test_exact_solution_metrics(self, heat1d_problem):
        """Exact solution should have zero error metrics."""
        # Sample points
        n = 100
        t = torch.rand(n) * heat1d_problem.t_max
        x = torch.rand(n) * heat1d_problem.length

        # Get exact solution
        u_exact = heat1d_problem.exact_solution(t, x)

        # Test that exact solution has zero relative L2 error with itself
        error = relative_l2_error(u_exact, u_exact)
        assert torch.isclose(error, torch.tensor(0.0), atol=1e-6)

    def test_boundary_conditions(self, heat1d_problem):
        """Heat1D boundary conditions should be zero."""
        t = torch.rand(50)
        x_left = torch.zeros(50)
        x_right = torch.ones(50)

        bc_left = heat1d_problem.boundary_condition(t, x_left)
        bc_right = heat1d_problem.boundary_condition(t, x_right)

        assert torch.allclose(bc_left, torch.zeros_like(bc_left))
        assert torch.allclose(bc_right, torch.zeros_like(bc_right))

    def test_initial_condition(self, heat1d_problem):
        """Initial condition should match sin(pi*x)."""
        x = torch.linspace(0, 1, 100)
        ic = heat1d_problem.initial_condition(x)
        expected = torch.sin(torch.pi * x)

        assert torch.allclose(ic, expected, atol=1e-6)


# ============================================================================
# Differentiability Tests
# ============================================================================

class TestDifferentiability:
    """Tests for metric differentiability."""

    def test_mse_gradient_flow(self):
        """MSE should allow gradient flow."""
        pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        true = torch.tensor([1.5, 2.0, 2.5])

        loss = mse(pred, true)
        loss.backward()

        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_relative_l2_gradient_flow(self):
        """Relative L2 error should allow gradient flow."""
        pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        true = torch.tensor([1.5, 2.0, 2.5])

        loss = relative_l2_error(pred, true)
        loss.backward()

        assert pred.grad is not None

    def test_mae_gradient_flow(self):
        """MAE should allow gradient flow (via abs)."""
        pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        true = torch.tensor([1.5, 2.0, 2.5])

        loss = mae(pred, true)
        loss.backward()

        assert pred.grad is not None
