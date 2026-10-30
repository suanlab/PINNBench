"""Unit tests for visualization utilities.

Tests verify that each plot function runs without error (saves to /tmp),
and that analysis functions return expected types and shapes.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from causal_pinn.visualization import (
    aggregate_seed_results,
    compare_results,
    compute_convergence_failure_rate,
    compute_summary_statistics,
    format_results_table,
    load_results_json,
    plot_convergence_comparison,
    plot_gate_evolution,
    plot_metric_comparison,
    plot_residual_distribution,
    plot_seed_variance,
    plot_training_curves,
    save_results_json,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_history() -> dict:
    """Create a sample training history for testing."""
    return {
        "stage1_losses": [
            {"total": 1.0, "data_loss": 0.5, "physics_loss": 0.5},
            {"total": 0.8, "data_loss": 0.4, "physics_loss": 0.4},
            {"total": 0.6, "data_loss": 0.3, "physics_loss": 0.3},
        ],
        "stage2_losses": [
            {"total": 0.4, "physics_loss": 0.3, "bc_loss": 0.1},
            {"total": 0.3, "physics_loss": 0.2, "bc_loss": 0.1},
            {"total": 0.2, "physics_loss": 0.15, "bc_loss": 0.05},
        ],
        "stage3_losses": [
            {"total": 0.15, "causal_loss": 0.1, "bc_loss": 0.05},
            {"total": 0.1, "causal_loss": 0.07, "bc_loss": 0.03},
            {"total": 0.08, "causal_loss": 0.05, "bc_loss": 0.03},
        ],
        "stage1_total_loss": [1.0, 0.8, 0.6],
        "stage2_total_loss": [0.4, 0.3, 0.2],
        "stage3_total_loss": [0.15, 0.1, 0.08],
    }


@pytest.fixture
def sample_multi_seed_histories() -> dict[int, dict]:
    """Create sample histories for multiple seeds."""
    np.random.seed(42)
    histories = {}
    for seed in [42, 123, 456]:
        base_loss = 1.0 - np.random.random() * 0.2
        histories[seed] = {
            "stage1_losses": [
                {"total": base_loss * (0.9**i)} for i in range(5)
            ],
            "stage2_losses": [
                {"total": base_loss * (0.9**5) * (0.8**i)} for i in range(5)
            ],
            "stage3_losses": [
                {"total": base_loss * (0.9**5) * (0.8**5) * (0.7**i)} for i in range(5)
            ],
        }
    return histories


@pytest.fixture
def sample_results() -> dict[str, dict[str, float]]:
    """Create sample results for metric comparison."""
    return {
        "PINN": {"mse": 0.1, "mae": 0.2, "relative_l2": 0.15},
        "Hybrid": {"mse": 0.05, "mae": 0.1, "relative_l2": 0.08},
        "FNO": {"mse": 0.08, "mae": 0.15, "relative_l2": 0.12},
    }


@pytest.fixture
def sample_gate_stats() -> list[dict]:
    """Create sample gate statistics history."""
    stats = []
    for i in range(10):
        stats.append({
            "mean": 0.5 + 0.02 * i,
            "std": 0.3 - 0.02 * i,
            "min": 0.1,
            "max": 0.9,
        })
    return stats


@pytest.fixture
def temp_dir() -> Path:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Tests for Plotting Functions
# =============================================================================


class TestPlotTrainingCurves:
    """Tests for plot_training_curves function."""

    def test_runs_without_error_single_history(self, sample_history, temp_dir):
        """Test that function runs without error for single history."""
        save_path = str(temp_dir / "training_curves.png")
        plot_training_curves(sample_history, title="Test Training", save_path=save_path)

        assert Path(save_path).exists()
        plt.close("all")

    def test_returns_figure(self, sample_history):
        """Test that function returns a matplotlib Figure."""
        fig = plot_training_curves(sample_history, title="Test")
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_handles_empty_history(self, temp_dir):
        """Test that function handles empty history gracefully."""
        save_path = str(temp_dir / "empty_history.png")
        fig = plot_training_curves({}, title="Empty Test", save_path=save_path)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_multi_seed_histories(self, sample_multi_seed_histories, temp_dir):
        """Test plotting multiple seed histories."""
        save_path = str(temp_dir / "multi_seed.png")
        plot_training_curves(
            sample_multi_seed_histories,
            title="Multi-Seed Test",
            save_path=save_path
        )
        assert Path(save_path).exists()
        plt.close("all")


class TestPlotConvergenceComparison:
    """Tests for plot_convergence_comparison function."""

    def test_runs_without_error(self, sample_multi_seed_histories, temp_dir):
        """Test that function runs without error."""
        # Use the multi-seed histories as different runs
        save_path = str(temp_dir / "convergence.png")
        plot_convergence_comparison(
            sample_multi_seed_histories,
            metric_key="total",
            title="Convergence Test",
            save_path=save_path
        )
        assert Path(save_path).exists()
        plt.close("all")

    def test_returns_figure(self, sample_multi_seed_histories):
        """Test that function returns a matplotlib Figure."""
        fig = plot_convergence_comparison(
            {"Run1": sample_multi_seed_histories[42]},
            title="Test"
        )
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestPlotSeedVariance:
    """Tests for plot_seed_variance function."""

    def test_runs_without_error(self, sample_multi_seed_histories, temp_dir):
        """Test that function runs without error."""
        save_path = str(temp_dir / "seed_variance.png")
        plot_seed_variance(
            sample_multi_seed_histories,
            stage="stage1",
            title="Variance Test",
            save_path=save_path
        )
        assert Path(save_path).exists()
        plt.close("all")

    def test_all_stages(self, sample_multi_seed_histories, temp_dir):
        """Test plotting variance for all stages."""
        for stage in ["stage1", "stage2", "stage3"]:
            save_path = str(temp_dir / f"variance_{stage}.png")
            plot_seed_variance(
                sample_multi_seed_histories,
                stage=stage,
                title=f"Stage: {stage}",
                save_path=save_path
            )
            assert Path(save_path).exists()
            plt.close("all")


class TestPlotMetricComparison:
    """Tests for plot_metric_comparison function."""

    def test_runs_without_error(self, sample_results, temp_dir):
        """Test that function runs without error."""
        save_path = str(temp_dir / "metric_comparison.png")
        plot_metric_comparison(
            sample_results,
            title="Metric Test",
            save_path=save_path
        )
        assert Path(save_path).exists()
        plt.close("all")

    def test_handles_empty_results(self, temp_dir):
        """Test handling of empty results."""
        save_path = str(temp_dir / "empty_metrics.png")
        fig = plot_metric_comparison({}, title="Empty", save_path=save_path)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestPlotResidualDistribution:
    """Tests for plot_residual_distribution function."""

    def test_runs_with_list(self, temp_dir):
        """Test with list input."""
        residuals = [0.1, 0.05, -0.02, 0.08, -0.01] * 20
        save_path = str(temp_dir / "residuals_list.png")
        plot_residual_distribution(
            residuals,
            title="Residual Test",
            save_path=save_path
        )
        assert Path(save_path).exists()
        plt.close("all")

    def test_runs_with_tensor(self, temp_dir):
        """Test with tensor input."""
        residuals = torch.randn(100)
        save_path = str(temp_dir / "residuals_tensor.png")
        plot_residual_distribution(
            residuals,
            title="Tensor Residuals",
            save_path=save_path
        )
        assert Path(save_path).exists()
        plt.close("all")

    def test_runs_with_numpy(self, temp_dir):
        """Test with numpy array input."""
        residuals = np.random.randn(100)
        save_path = str(temp_dir / "residuals_numpy.png")
        plot_residual_distribution(
            residuals,
            title="Numpy Residuals",
            save_path=save_path
        )
        assert Path(save_path).exists()
        plt.close("all")


class TestPlotGateEvolution:
    """Tests for plot_gate_evolution function."""

    def test_runs_without_error(self, sample_gate_stats, temp_dir):
        """Test that function runs without error."""
        save_path = str(temp_dir / "gate_evolution.png")
        plot_gate_evolution(
            sample_gate_stats,
            title="Gate Test",
            save_path=save_path
        )
        assert Path(save_path).exists()
        plt.close("all")

    def test_handles_tensor_values(self, temp_dir):
        """Test with tensor values in stats."""
        stats = []
        for i in range(5):
            stats.append({
                "mean": torch.tensor(0.5 + 0.02 * i),
                "std": torch.tensor(0.3 - 0.02 * i),
            })
        save_path = str(temp_dir / "gate_tensors.png")
        plot_gate_evolution(stats, title="Tensor Gates", save_path=save_path)
        assert Path(save_path).exists()
        plt.close("all")


# =============================================================================
# Tests for Analysis Functions
# =============================================================================


class TestAggregateSeedResults:
    """Tests for aggregate_seed_results function."""

    def test_returns_correct_structure(self):
        """Test that function returns correct nested structure."""
        results = {
            42: {"mse": 0.1, "mae": 0.2},
            123: {"mse": 0.15, "mae": 0.18},
        }
        aggregated = aggregate_seed_results(results)

        assert "mse" in aggregated
        assert "mae" in aggregated
        assert "mean" in aggregated["mse"]
        assert "std" in aggregated["mse"]
        assert "min" in aggregated["mse"]
        assert "max" in aggregated["mse"]

    def test_computes_correct_statistics(self):
        """Test that computed statistics are correct."""
        results = {
            1: {"mse": 0.1},
            2: {"mse": 0.2},
            3: {"mse": 0.3},
        }
        aggregated = aggregate_seed_results(results)

        assert aggregated["mse"]["mean"] == pytest.approx(0.2)
        assert aggregated["mse"]["min"] == pytest.approx(0.1)
        assert aggregated["mse"]["max"] == pytest.approx(0.3)

    def test_handles_empty_input(self):
        """Test handling of empty input."""
        result = aggregate_seed_results({})
        assert result == {}

    def test_handles_tensor_values(self):
        """Test handling of tensor values."""
        results = {
            42: {"mse": torch.tensor(0.1)},
            123: {"mse": torch.tensor(0.2)},
        }
        aggregated = aggregate_seed_results(results)

        assert isinstance(aggregated["mse"]["mean"], float)
        assert aggregated["mse"]["mean"] == pytest.approx(0.15)


class TestComputeConvergenceFailureRate:
    """Tests for compute_convergence_failure_rate function."""

    def test_returns_float_between_0_and_1(self, sample_multi_seed_histories):
        """Test that function returns float in [0, 1]."""
        rate = compute_convergence_failure_rate(sample_multi_seed_histories, threshold=0.5)
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0

    def test_correct_failure_count(self, sample_multi_seed_histories):
        """Test correct failure counting."""
        # Set threshold such that some but not all fail
        rate = compute_convergence_failure_rate(sample_multi_seed_histories, threshold=0.1)
        assert isinstance(rate, float)

    def test_handles_empty_input(self):
        """Test handling of empty input."""
        rate = compute_convergence_failure_rate({}, threshold=0.1)
        assert rate == 0.0

    def test_all_fail(self, sample_multi_seed_histories):
        """Test when all runs fail."""
        rate = compute_convergence_failure_rate(
            sample_multi_seed_histories,
            threshold=1e-10  # Very low threshold
        )
        assert rate == 1.0

    def test_none_fail(self, sample_multi_seed_histories):
        """Test when no runs fail."""
        rate = compute_convergence_failure_rate(
            sample_multi_seed_histories,
            threshold=1e10  # Very high threshold
        )
        assert rate == 0.0


class TestFormatResultsTable:
    """Tests for format_results_table function."""

    def test_returns_string(self, sample_results):
        """Test that function returns a string."""
        table = format_results_table(sample_results)
        assert isinstance(table, str)

    def test_contains_model_names(self, sample_results):
        """Test that table contains model names."""
        table = format_results_table(sample_results)
        for model_name in sample_results:
            assert model_name in table

    def test_contains_metric_names(self, sample_results):
        """Test that table contains metric names."""
        table = format_results_table(sample_results)
        metrics = set()
        for model_results in sample_results.values():
            metrics.update(model_results.keys())
        for metric in metrics:
            assert metric in table

    def test_handles_empty_results(self):
        """Test handling of empty results."""
        table = format_results_table({})
        assert "No results" in table or table == ""

    def test_markdown_format(self, sample_results):
        """Test that output is in markdown format."""
        table = format_results_table(sample_results)
        assert "|" in table  # Markdown uses | for columns


class TestSaveLoadResultsJSON:
    """Tests for save_results_json and load_results_json functions."""

    def test_save_and_load_roundtrip(self, temp_dir):
        """Test that save/load preserves data."""
        results = {
            "model": "PINN",
            "metrics": {"mse": 0.1, "mae": 0.2},
            "nested": {"a": [1, 2, 3]}
        }
        filepath = str(temp_dir / "results.json")

        save_results_json(results, filepath)
        loaded = load_results_json(filepath)

        assert loaded["model"] == results["model"]
        assert loaded["metrics"]["mse"] == results["metrics"]["mse"]
        assert loaded["nested"]["a"] == results["nested"]["a"]

    def test_handles_tensors(self, temp_dir):
        """Test handling of tensor values."""
        results = {
            "tensor_value": torch.tensor(0.123),
            "tensor_list": torch.tensor([1.0, 2.0, 3.0]),
        }
        filepath = str(temp_dir / "tensor_results.json")

        save_results_json(results, filepath)
        loaded = load_results_json(filepath)

        assert loaded["tensor_value"] == pytest.approx(0.123)
        assert loaded["tensor_list"] == [1.0, 2.0, 3.0]

    def test_creates_directory(self, temp_dir):
        """Test that function creates directory if it doesn't exist."""
        filepath = str(temp_dir / "subdir" / "results.json")
        save_results_json({"test": 1}, filepath)
        assert Path(filepath).exists()


class TestComputeSummaryStatistics:
    """Tests for compute_summary_statistics function."""

    def test_returns_correct_keys(self):
        """Test that function returns expected keys."""
        stats = compute_summary_statistics([1.0, 2.0, 3.0])
        expected_keys = {"mean", "std", "min", "max", "median"}
        assert set(stats.keys()) == expected_keys

    def test_computes_correct_values(self):
        """Test that computed values are correct."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = compute_summary_statistics(values)

        assert stats["mean"] == pytest.approx(3.0)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)
        assert stats["median"] == pytest.approx(3.0)

    def test_handles_tensor_input(self):
        """Test handling of tensor input."""
        values = torch.tensor([1.0, 2.0, 3.0])
        stats = compute_summary_statistics(values)

        assert isinstance(stats["mean"], float)
        assert stats["mean"] == pytest.approx(2.0)

    def test_handles_empty_input(self):
        """Test handling of empty input."""
        stats = compute_summary_statistics([])
        for key in ["mean", "std", "min", "max", "median"]:
            assert stats[key] == 0.0


class TestCompareResults:
    """Tests for compare_results function."""

    def test_returns_correct_structure(self):
        """Test that function returns correct nested structure."""
        baseline = {"mse": 0.1, "mae": 0.2}
        comparison = {"mse": 0.05, "mae": 0.1}

        result = compare_results(baseline, comparison)

        assert "mse" in result
        assert "mae" in result
        assert "Baseline" in result["mse"]
        assert "Comparison" in result["mse"]
        assert "improvement_pct" in result["mse"]

    def test_computes_improvement_correctly(self):
        """Test that improvement percentage is computed correctly."""
        baseline = {"mse": 0.1}
        comparison = {"mse": 0.05}  # 50% improvement

        result = compare_results(baseline, comparison)

        assert result["mse"]["improvement_pct"] == pytest.approx(50.0)

    def test_handles_zero_baseline(self):
        """Test handling of zero baseline value."""
        baseline = {"mse": 0.0}
        comparison = {"mse": 0.05}

        result = compare_results(baseline, comparison)

        assert result["mse"]["improvement_pct"] == 0.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestVisualizationIntegration:
    """Integration tests combining multiple visualization functions."""

    def test_full_analysis_workflow(self, sample_multi_seed_histories, temp_dir):
        """Test a typical analysis workflow."""
        # 1. Plot training curves
        curves_path = str(temp_dir / "curves.png")
        plot_training_curves(
            sample_multi_seed_histories,
            title="Training Curves",
            save_path=curves_path
        )
        assert Path(curves_path).exists()

        # 2. Plot seed variance
        variance_path = str(temp_dir / "variance.png")
        plot_seed_variance(
            sample_multi_seed_histories,
            stage="stage1",
            title="Seed Variance",
            save_path=variance_path
        )
        assert Path(variance_path).exists()

        # 3. Compute aggregated metrics
        metrics_by_seed = {
            seed: {
                "final_loss": hist["stage3_losses"][-1]["total"]
                if hist["stage3_losses"] else float("inf")
            }
            for seed, hist in sample_multi_seed_histories.items()
        }
        aggregated = aggregate_seed_results(metrics_by_seed)
        assert "final_loss" in aggregated

        # 4. Compute failure rate
        failure_rate = compute_convergence_failure_rate(
            sample_multi_seed_histories,
            threshold=0.1
        )
        assert 0.0 <= failure_rate <= 1.0

        # 5. Save and load results
        results_path = str(temp_dir / "results.json")
        save_results_json({
            "aggregated": aggregated,
            "failure_rate": failure_rate,
        }, results_path)
        loaded = load_results_json(results_path)
        assert "aggregated" in loaded

        plt.close("all")
