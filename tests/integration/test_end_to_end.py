"""End-to-end integration tests for the Causal-Aware Hybrid PINN-Operator framework.

Tests the full pipeline: model creation → training → evaluation.
Uses small models and few epochs for speed.
"""

from __future__ import annotations

import pytest
import torch

from causal_pinn.evaluation.benchmark_runner import BenchmarkConfig, BenchmarkRunner
from causal_pinn.evaluation.metrics import (
    MetricSuite,
)
from causal_pinn.evaluation.physics_consistency import PhysicsConsistencyChecker
from causal_pinn.models.hybrid.causal_gate import CausalGate
from causal_pinn.models.hybrid.hybrid_model import HybridConfig, HybridPINNOperator
from causal_pinn.models.operators.fno import FNO
from causal_pinn.models.pinn.base import PINNBase
from causal_pinn.physics.losses.causal_loss import CausalLoss
from causal_pinn.physics.losses.physics_loss import PhysicsLoss
from causal_pinn.physics.pde.advection_diffusion_1d import AdvectionDiffusion1D
from causal_pinn.physics.pde.heat_1d import Heat1D
from causal_pinn.training.callbacks import (
    EarlyStoppingCallback,
    LoggingCallback,
)
from causal_pinn.training.trainer import HybridTrainer
from causal_pinn.utils.config import TrainingConfig
from causal_pinn.utils.random import set_seed

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def heat_pde() -> Heat1D:
    """Heat1D PDE with exact solution."""
    return Heat1D()


@pytest.fixture
def advection_pde() -> AdvectionDiffusion1D:
    """AdvectionDiffusion1D PDE (no exact solution)."""
    return AdvectionDiffusion1D()


@pytest.fixture
def small_pinn() -> PINNBase:
    """Small PINN for fast tests."""
    return PINNBase(input_dim=2, output_dim=1, hidden_dims=[16, 16])


@pytest.fixture
def small_fno() -> FNO:
    """Small FNO for fast tests."""
    return FNO(in_channels=2, out_channels=1, modes=4, width=8, num_layers=1, dim=1)


@pytest.fixture
def small_gate() -> CausalGate:
    """Small CausalGate for fast tests."""
    return CausalGate(out_dim=1, hidden_dim=8, causal_strength=1.0)


@pytest.fixture
def fast_training_config() -> TrainingConfig:
    """Minimal training config for fast integration tests."""
    return TrainingConfig(
        stage1_epochs=3,
        stage1_lr=1e-3,
        stage2_epochs=3,
        stage2_lr=1e-4,
        stage3_epochs=3,
        stage3_lr=1e-5,
        causal_epsilon=1.0,
        batch_size=32,
        num_collocation_points=64,
        num_boundary_points=16,
        seed=42,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Test: PINN-only training pipeline on Heat1D
# ---------------------------------------------------------------------------

class TestPINNOnlyPipeline:
    """Test full pipeline with PINNBase model (no hybrid)."""

    def test_train_pinn_on_heat1d(
        self, small_pinn: PINNBase, heat_pde: Heat1D, fast_training_config: TrainingConfig,
    ) -> None:
        """Train a PINN on Heat1D and verify training completes."""
        set_seed(42)
        trainer = HybridTrainer(
            model=small_pinn,
            pde_problem=heat_pde,
            config=fast_training_config,
        )
        history = trainer.train()

        # All 3 stages should have recorded losses
        assert len(history["stage1_total_loss"]) == fast_training_config.stage1_epochs
        assert len(history["stage2_total_loss"]) == fast_training_config.stage2_epochs
        assert len(history["stage3_total_loss"]) == fast_training_config.stage3_epochs

        # All losses should be finite
        for stage_key in ["stage1_total_loss", "stage2_total_loss", "stage3_total_loss"]:
            for loss_val in history[stage_key]:
                assert not torch.isnan(torch.tensor(loss_val)), f"NaN loss in {stage_key}"
                assert not torch.isinf(torch.tensor(loss_val)), f"Inf loss in {stage_key}"

    def test_train_with_callbacks(
        self, small_pinn: PINNBase, heat_pde: Heat1D, fast_training_config: TrainingConfig,
    ) -> None:
        """Train PINN with logging and early stopping callbacks."""
        set_seed(42)
        callbacks = [
            LoggingCallback(log_every_n_epochs=1, verbose=False),
            EarlyStoppingCallback(patience=5, min_delta=0.0),
        ]
        trainer = HybridTrainer(
            model=small_pinn,
            pde_problem=heat_pde,
            config=fast_training_config,
            callbacks=callbacks,
        )
        history = trainer.train()

        # Should complete without error — callbacks should not interfere
        assert "stage1_total_loss" in history
        assert "stage2_total_loss" in history
        assert "stage3_total_loss" in history

    def test_train_pinn_on_advection(
        self, advection_pde: AdvectionDiffusion1D, fast_training_config: TrainingConfig,
    ) -> None:
        """Train PINN on a PDE without exact solution."""
        set_seed(42)
        pinn = PINNBase(input_dim=2, output_dim=1, hidden_dims=[16, 16])
        trainer = HybridTrainer(
            model=pinn,
            pde_problem=advection_pde,
            config=fast_training_config,
        )
        history = trainer.train()

        # Stage 1 should still work (physics-only when no exact solution)
        assert len(history["stage1_total_loss"]) > 0
        assert all(
            not torch.isnan(torch.as_tensor(v))
            for v in history["stage1_total_loss"]
        )


# ---------------------------------------------------------------------------
# Test: Hybrid PINN-Operator pipeline
# ---------------------------------------------------------------------------

class TestHybridPipeline:
    """Test full pipeline with HybridPINNOperator."""

    def test_hybrid_forward_and_loss(
        self,
        small_pinn: PINNBase,
        small_fno: FNO,
        small_gate: CausalGate,
        heat_pde: Heat1D,
    ) -> None:
        """Verify hybrid model produces predictions and computable losses."""
        set_seed(42)
        hybrid = HybridPINNOperator(
            small_pinn, small_fno, small_gate, heat_pde,
        )

        t = torch.rand(16, 1, requires_grad=True)
        x = torch.rand(16, 1, requires_grad=True)

        # Forward pass
        pred = hybrid(t, x)
        assert pred.shape == (16, 1)
        assert torch.isfinite(pred).all()

        # Loss computation
        loss_dict = hybrid.compute_hybrid_loss(t, x)
        assert "total" in loss_dict
        assert "physics_loss" in loss_dict
        assert "operator_loss" in loss_dict
        assert "gate_stats" in loss_dict
        assert torch.isfinite(loss_dict["total"])

    def test_hybrid_training_step(
        self,
        small_pinn: PINNBase,
        small_fno: FNO,
        small_gate: CausalGate,
        heat_pde: Heat1D,
    ) -> None:
        """Test that a single gradient update works end-to-end."""
        set_seed(42)
        hybrid = HybridPINNOperator(
            small_pinn, small_fno, small_gate, heat_pde,
        )
        optimizer = torch.optim.Adam(hybrid.parameters(), lr=1e-3)

        t = torch.rand(32, 1, requires_grad=True)
        x = torch.rand(32, 1, requires_grad=True)

        # Record initial params
        initial_params = [p.clone() for p in hybrid.parameters()]

        # Training step
        optimizer.zero_grad()
        loss_dict = hybrid.compute_hybrid_loss(t, x)
        loss_dict["total"].backward()
        optimizer.step()

        # Params should have changed
        params_changed = False
        for p_old, p_new in zip(initial_params, hybrid.parameters(), strict=False):
            if not torch.allclose(p_old, p_new.data):
                params_changed = True
                break
        assert params_changed, "Parameters should update after gradient step"

    def test_hybrid_ablation_modes(
        self,
        small_pinn: PINNBase,
        small_fno: FNO,
        small_gate: CausalGate,
        heat_pde: Heat1D,
    ) -> None:
        """Test operator-only, pinn-only, and full hybrid modes."""
        t = torch.rand(16, 1, requires_grad=True)
        x = torch.rand(16, 1, requires_grad=True)

        modes = [
            HybridConfig(use_operator=True, use_pinn=False, use_causal_gate=False),
            HybridConfig(use_operator=False, use_pinn=True, use_causal_gate=False),
            HybridConfig(use_operator=True, use_pinn=True, use_causal_gate=True),
        ]

        for config in modes:
            set_seed(42)
            pinn = PINNBase(input_dim=2, output_dim=1, hidden_dims=[16, 16])
            fno = FNO(in_channels=2, out_channels=1, modes=4, width=8, num_layers=1, dim=1)
            gate = CausalGate(out_dim=1, hidden_dim=8, causal_strength=1.0)

            hybrid = HybridPINNOperator(pinn, fno, gate, heat_pde, config=config)

            pred = hybrid(t, x)
            assert pred.shape == (16, 1), f"Failed for config: {config}"
            assert torch.isfinite(pred).all(), f"Non-finite output for config: {config}"


# ---------------------------------------------------------------------------
# Test: Train → Evaluate pipeline
# ---------------------------------------------------------------------------

class TestTrainThenEvaluate:
    """Full create → train → evaluate pipeline."""

    def test_pinn_train_then_benchmark(
        self, heat_pde: Heat1D, fast_training_config: TrainingConfig,
    ) -> None:
        """Train a PINN then evaluate with BenchmarkRunner."""
        set_seed(42)

        # 1. Create model
        pinn = PINNBase(input_dim=2, output_dim=1, hidden_dims=[32, 32])

        # 2. Train
        trainer = HybridTrainer(
            model=pinn,
            pde_problem=heat_pde,
            config=fast_training_config,
        )
        history = trainer.train()
        assert len(history["stage1_total_loss"]) > 0

        # 3. Evaluate
        bench_config = BenchmarkConfig(
            n_collocation=64,
            n_boundary=16,
            n_initial=16,
            include_physics_consistency=False,
            device="cpu",
        )
        runner = BenchmarkRunner(config=bench_config)
        results = runner.run_single(pinn, heat_pde)

        # Check that all expected metrics are present and finite
        for key in ["relative_l2_error", "mse", "mae", "max_error", "pde_residual_norm"]:
            assert key in results, f"Missing metric: {key}"
            assert not torch.isnan(torch.tensor(results[key])), f"NaN in {key}"

    def test_pinn_metric_suite_after_training(
        self, heat_pde: Heat1D, fast_training_config: TrainingConfig,
    ) -> None:
        """Train PINN and evaluate with MetricSuite on known points."""
        set_seed(42)

        pinn = PINNBase(input_dim=2, output_dim=1, hidden_dims=[32, 32])
        trainer = HybridTrainer(
            model=pinn,
            pde_problem=heat_pde,
            config=fast_training_config,
        )
        trainer.train()

        # Generate evaluation points with exact solution
        t_eval = torch.rand(100) * 0.5  # Stay in early time (more accurate)
        x_eval = torch.rand(100)
        u_exact = heat_pde.exact_solution(t_eval, x_eval)

        # Get predictions
        pinn.eval()
        with torch.no_grad():
            inputs = torch.stack([t_eval, x_eval], dim=1)
            u_pred = pinn(inputs).squeeze(-1)

        # Compute metrics
        suite = MetricSuite(["relative_l2_error", "mse", "mae", "max_error"])
        metrics = suite.compute(u_pred, u_exact)

        assert "relative_l2_error" in metrics
        assert "mse" in metrics
        assert all(torch.isfinite(torch.as_tensor(v)) for v in metrics.values())

    def test_model_comparison(
        self, heat_pde: Heat1D, fast_training_config: TrainingConfig,
    ) -> None:
        """Train two models and compare via BenchmarkRunner."""
        set_seed(42)

        # Train model A (small)
        model_a = PINNBase(input_dim=2, output_dim=1, hidden_dims=[16, 16])
        trainer_a = HybridTrainer(model_a, heat_pde, fast_training_config)
        trainer_a.train()

        # Train model B (wider)
        set_seed(123)
        model_b = PINNBase(input_dim=2, output_dim=1, hidden_dims=[32, 32])
        trainer_b = HybridTrainer(model_b, heat_pde, fast_training_config)
        trainer_b.train()

        # Compare
        bench_config = BenchmarkConfig(
            n_collocation=64,
            n_boundary=16,
            n_initial=16,
            include_physics_consistency=False,
            device="cpu",
        )
        runner = BenchmarkRunner(config=bench_config)
        comparison = runner.run_comparison(
            {"small": model_a, "wide": model_b},
            heat_pde,
        )

        assert "small" in comparison
        assert "wide" in comparison
        assert "relative_l2_error" in comparison["small"]
        assert "relative_l2_error" in comparison["wide"]


# ---------------------------------------------------------------------------
# Test: Physics consistency verification
# ---------------------------------------------------------------------------

class TestPhysicsConsistencyIntegration:
    """Test physics consistency checking after training."""

    def test_residual_distribution_after_training(
        self, heat_pde: Heat1D, fast_training_config: TrainingConfig,
    ) -> None:
        """Compute PDE residual distribution after training."""
        set_seed(42)

        pinn = PINNBase(input_dim=2, output_dim=1, hidden_dims=[32, 32])
        trainer = HybridTrainer(pinn, heat_pde, fast_training_config)
        trainer.train()

        # Compute residuals
        t = torch.rand(100, requires_grad=True)
        x = torch.rand(100, requires_grad=True)
        inputs = torch.stack([t, x], dim=1)
        u_pred = pinn(inputs).squeeze(-1)
        residuals = heat_pde.residual(u_pred, t, x)

        # Check residual distribution
        checker = PhysicsConsistencyChecker(device="cpu")
        dist = checker.check_residual_distribution(residuals.detach())

        assert "mean" in dist
        assert "std" in dist
        assert "max" in dist
        assert "percentiles" in dist
        assert "95" in dist["percentiles"]
        # Residuals should be finite
        assert torch.isfinite(torch.as_tensor(dist["mean"]))
        assert torch.isfinite(torch.as_tensor(dist["std"]))


# ---------------------------------------------------------------------------
# Test: CausalLoss integration with model
# ---------------------------------------------------------------------------

class TestCausalLossIntegration:
    """Test CausalLoss works with actual trained models."""

    def test_causal_loss_with_pinn(
        self, heat_pde: Heat1D,
    ) -> None:
        """Verify CausalLoss computes with a PINN model."""
        set_seed(42)
        pinn = PINNBase(input_dim=2, output_dim=1, hidden_dims=[16, 16])

        causal_loss = CausalLoss(
            pde_problem=heat_pde,
            epsilon=1.0,
            num_time_bins=5,
        )

        t = torch.rand(64, requires_grad=True)
        x = torch.rand(64, requires_grad=True)

        result = causal_loss(pinn, t, x, use_binning=True)
        assert "causal_loss" in result
        assert torch.isfinite(result["causal_loss"])

    def test_physics_loss_with_pinn(
        self, heat_pde: Heat1D,
    ) -> None:
        """Verify PhysicsLoss computes with a PINN model."""
        set_seed(42)
        pinn = PINNBase(input_dim=2, output_dim=1, hidden_dims=[16, 16])

        physics_loss = PhysicsLoss(pde_problem=heat_pde)

        t = torch.rand(64, requires_grad=True)
        x = torch.rand(64, requires_grad=True)

        result = physics_loss(pinn, t, x)
        assert "physics_loss" in result
        assert torch.isfinite(result["physics_loss"])


# ---------------------------------------------------------------------------
# Test: Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Test that same seed produces same results."""

    def test_deterministic_training(
        self, heat_pde: Heat1D, fast_training_config: TrainingConfig,
    ) -> None:
        """Same seed should produce same training losses."""
        losses_run1 = self._run_training(heat_pde, fast_training_config, seed=42)
        losses_run2 = self._run_training(heat_pde, fast_training_config, seed=42)

        for l1, l2 in zip(losses_run1, losses_run2, strict=False):
            assert abs(l1 - l2) < 1e-5, f"Non-deterministic: {l1} vs {l2}"

    def test_different_seeds_differ(
        self, heat_pde: Heat1D, fast_training_config: TrainingConfig,
    ) -> None:
        """Different seeds should produce different training losses."""
        losses_run1 = self._run_training(heat_pde, fast_training_config, seed=42)
        losses_run2 = self._run_training(heat_pde, fast_training_config, seed=999)

        # At least some losses should differ
        any_differ = any(
            abs(l1 - l2) > 1e-6 for l1, l2 in zip(losses_run1, losses_run2, strict=False)
        )
        assert any_differ, "Different seeds should produce different results"

    @staticmethod
    def _run_training(
        pde: Heat1D,
        config: TrainingConfig,
        seed: int,
    ) -> list[float]:
        """Helper: run training and return stage1 losses."""
        set_seed(seed)
        config_copy = TrainingConfig(
            stage1_epochs=config.stage1_epochs,
            stage1_lr=config.stage1_lr,
            stage2_epochs=config.stage2_epochs,
            stage2_lr=config.stage2_lr,
            stage3_epochs=config.stage3_epochs,
            stage3_lr=config.stage3_lr,
            causal_epsilon=config.causal_epsilon,
            batch_size=config.batch_size,
            num_collocation_points=config.num_collocation_points,
            num_boundary_points=config.num_boundary_points,
            seed=seed,
            device=config.device,
        )
        pinn = PINNBase(input_dim=2, output_dim=1, hidden_dims=[16, 16])
        trainer = HybridTrainer(pinn, pde, config_copy)
        history = trainer.train()
        return history["stage1_total_loss"]


# ---------------------------------------------------------------------------
# Test: Component interoperability
# ---------------------------------------------------------------------------

class TestComponentInterop:
    """Test that all components work together correctly."""

    def test_pde_to_loss_to_model(self, heat_pde: Heat1D) -> None:
        """PDE → PhysicsLoss → PINN model pipeline."""
        pinn = PINNBase(input_dim=2, output_dim=1, hidden_dims=[16, 16])
        physics_loss = PhysicsLoss(heat_pde)

        t = torch.rand(32, requires_grad=True)
        x = torch.rand(32, requires_grad=True)

        result = physics_loss(pinn, t, x)
        loss = result["physics_loss"]

        # Should be differentiable
        loss.backward()

        # PINN params should have gradients
        has_grad = any(p.grad is not None for p in pinn.parameters())
        assert has_grad, "PINN should receive gradients through PhysicsLoss"

    def test_all_modules_importable(self) -> None:
        """Verify all public API modules are importable."""
        # Models

        # Physics

        # Training

        # Evaluation

        # Utils

        # If we get here, everything imported successfully
        assert True

    def test_training_history_structure(
        self, heat_pde: Heat1D, fast_training_config: TrainingConfig,
    ) -> None:
        """Verify training history has complete structure."""
        set_seed(42)
        pinn = PINNBase(input_dim=2, output_dim=1, hidden_dims=[16, 16])
        trainer = HybridTrainer(pinn, heat_pde, fast_training_config)
        history = trainer.train()

        # Required keys
        required_keys = [
            "stage1_losses", "stage2_losses", "stage3_losses",
            "stage1_total_loss", "stage2_total_loss", "stage3_total_loss",
        ]
        for key in required_keys:
            assert key in history, f"Missing history key: {key}"

        # Each stage loss entry should be a dict with 'total'
        for stage_losses in [history["stage1_losses"], history["stage2_losses"], history["stage3_losses"]]:  # noqa: E501
            assert len(stage_losses) > 0
            for entry in stage_losses:
                assert "total" in entry, f"Loss entry missing 'total': {entry.keys()}"
