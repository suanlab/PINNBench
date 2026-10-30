"""Unit tests for the training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor

from causal_pinn.models.pinn import PINNBase
from causal_pinn.physics.pde import Heat1D, Wave2D
from causal_pinn.training import (
    Callback,
    CheckpointCallback,
    EarlyStoppingCallback,
    HybridTrainer,
    LoggingCallback,
    TrainingHistory,
)
from causal_pinn.utils.config import TrainingConfig

if TYPE_CHECKING:
    pass


class MockCallback(Callback):
    """Mock callback for testing callback invocation."""

    def __init__(self) -> None:
        """Initialize mock callback with counters."""
        self.train_begin_count = 0
        self.train_end_count = 0
        self.stage_begin_count = 0
        self.stage_end_count = 0
        self.epoch_begin_count = 0
        self.epoch_end_count = 0
        self.stages_seen: list[str] = []
        self.epochs_seen: list[int] = []

    def on_train_begin(self, trainer: HybridTrainer) -> None:
        self.train_begin_count += 1

    def on_train_end(self, trainer: HybridTrainer) -> None:
        self.train_end_count += 1

    def on_stage_begin(self, trainer: HybridTrainer, stage: str) -> None:
        self.stage_begin_count += 1
        self.stages_seen.append(stage)

    def on_stage_end(self, trainer: HybridTrainer, stage: str) -> None:
        self.stage_end_count += 1

    def on_epoch_begin(self, trainer: HybridTrainer, epoch: int, stage: str) -> None:
        self.epoch_begin_count += 1

    def on_epoch_end(
        self,
        trainer: HybridTrainer,
        epoch: int,
        stage: str,
        losses: dict[str, float],
    ) -> None:
        self.epoch_end_count += 1
        self.epochs_seen.append(epoch)


class SimplePDEProblem:
    """Simple PDE problem for testing.

    This PDE has an exact solution for testing data-driven training.
    The residual is: u - sin(pi*x)*exp(-t) = 0 (which has known solution)
    """

    @property
    def domain_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return domain bounds."""
        return ((0.0, 1.0), (0.0, 1.0))

    def residual(self, u: Tensor, t: Tensor, x: Tensor) -> Tensor:
        """Simple residual: u - 0.5*(t + x).

        This doesn't use autograd internally, so it works with PhysicsLoss.
        """
        return u - 0.5 * (t + x)

    def boundary_condition(self, t: Tensor, x: Tensor) -> Tensor:
        """BC: u = 0 at boundaries."""
        return torch.zeros_like(t)

    def initial_condition(self, x: Tensor) -> Tensor:
        """IC: u(0, x) = sin(pi*x)."""
        return torch.sin(torch.pi * x)

    def exact_solution(self, t: Tensor, x: Tensor) -> Tensor:
        """Exact solution for testing: u(t, x) = 0.5*(t + x)."""
        return 0.5 * (t + x)


@pytest.fixture
def network() -> PINNBase:
    """Create a test network."""
    return PINNBase(
        input_dim=2,
        output_dim=1,
        hidden_dims=[32, 32],
    )


@pytest.fixture
def pde_problem() -> SimplePDEProblem:
    """Create a test PDE problem with exact solution."""
    return SimplePDEProblem()


@pytest.fixture
def heat_pde() -> Heat1D:
    """Create a Heat1D PDE for initialization tests only."""
    return Heat1D(alpha=1.0, length=1.0, t_max=1.0)


@pytest.fixture
def training_config() -> TrainingConfig:
    """Create a test training configuration with small number of epochs."""
    return TrainingConfig(
        stage1_epochs=5,
        stage1_lr=1e-3,
        stage2_epochs=5,
        stage2_lr=1e-4,
        stage3_epochs=5,
        stage3_lr=1e-5,
        batch_size=64,
        num_collocation_points=100,
        num_boundary_points=20,
        seed=42,
        device="cpu",
    )


class TestTrainingHistory:
    """Tests for TrainingHistory dataclass."""

    def test_history_initialization(self) -> None:
        """Test that history initializes with empty lists."""
        history = TrainingHistory()

        assert history.stage1_losses == []
        assert history.stage2_losses == []
        assert history.stage3_losses == []
        assert history.stage1_total_loss == []
        assert history.stage2_total_loss == []
        assert history.stage3_total_loss == []

    def test_add_stage1_loss(self) -> None:
        """Test adding stage 1 loss."""
        history = TrainingHistory()
        losses = {"total": 1.0, "physics_loss": 0.8}

        history.add_stage1_loss(losses)

        assert len(history.stage1_losses) == 1
        assert history.stage1_losses[0] == losses
        assert history.stage1_total_loss == [1.0]

    def test_add_stage2_loss(self) -> None:
        """Test adding stage 2 loss."""
        history = TrainingHistory()
        losses = {"total": 0.5, "physics_loss": 0.3, "data_loss": 0.2}

        history.add_stage2_loss(losses)

        assert len(history.stage2_losses) == 1
        assert history.stage2_total_loss == [0.5]

    def test_add_stage3_loss(self) -> None:
        """Test adding stage 3 loss."""
        history = TrainingHistory()
        losses = {"total": 0.2, "causal_loss": 0.1}

        history.add_stage3_loss(losses)

        assert len(history.stage3_losses) == 1
        assert history.stage3_total_loss == [0.2]

    def test_to_dict(self) -> None:
        """Test converting history to dictionary."""
        history = TrainingHistory()
        history.add_stage1_loss({"total": 1.0})
        history.add_stage2_loss({"total": 0.5})

        result = history.to_dict()

        assert "stage1_losses" in result
        assert "stage2_losses" in result
        assert "stage3_losses" in result
        assert len(result["stage1_total_loss"]) == 1


class TestHybridTrainer:
    """Tests for HybridTrainer class."""

    def test_trainer_initialization(
        self,
        network: PINNBase,
        pde_problem: SimplePDEProblem,
        training_config: TrainingConfig,
    ) -> None:
        """Test that trainer initializes correctly."""
        trainer = HybridTrainer(
            model=network,
            pde_problem=pde_problem,
            config=training_config,
        )

        assert trainer.model is network
        assert trainer.pde_problem is pde_problem
        assert trainer.config is training_config
        assert trainer.callbacks == []
        assert isinstance(trainer.history, TrainingHistory)

    def test_trainer_initialization_with_heat1d(
        self,
        network: PINNBase,
        heat_pde: Heat1D,
        training_config: TrainingConfig,
    ) -> None:
        """Test that trainer can be initialized with Heat1D PDE."""
        trainer = HybridTrainer(
            model=network,
            pde_problem=heat_pde,
            config=training_config,
        )

        assert trainer.model is network
        assert trainer.pde_problem is heat_pde
        assert isinstance(trainer.history, TrainingHistory)

    def test_trainer_with_callbacks(
        self,
        network: PINNBase,
        pde_problem: SimplePDEProblem,
        training_config: TrainingConfig,
    ) -> None:
        """Test that trainer accepts callbacks."""
        callback = LoggingCallback(log_every_n_epochs=1)
        trainer = HybridTrainer(
            model=network,
            pde_problem=pde_problem,
            config=training_config,
            callbacks=[callback],
        )

        assert len(trainer.callbacks) == 1
        assert trainer.callbacks[0] is callback

    def test_trainer_device(
        self,
        network: PINNBase,
        pde_problem: SimplePDEProblem,
        training_config: TrainingConfig,
    ) -> None:
        """Test that trainer sets device correctly."""
        trainer = HybridTrainer(
            model=network,
            pde_problem=pde_problem,
            config=training_config,
            device="cpu",
        )

        assert trainer.device == torch.device("cpu")

    def test_stage1_reduces_loss(
        self,
        network: PINNBase,
        pde_problem: SimplePDEProblem,
        training_config: TrainingConfig,
    ) -> None:
        """Test that stage 1 training reduces loss."""
        trainer = HybridTrainer(
            model=network,
            pde_problem=pde_problem,
            config=training_config,
            device="cpu",
        )

        # Run training
        trainer.train()

        # Check that stage 1 losses exist and generally decrease
        stage1_losses = trainer.history.stage1_total_loss
        assert len(stage1_losses) > 0

        # Compare first few epochs to last few epochs
        if len(stage1_losses) >= 4:
            early_avg = sum(stage1_losses[:2]) / 2
            late_avg = sum(stage1_losses[-2:]) / 2
            # Loss should generally decrease
            assert late_avg < early_avg or stage1_losses[-1] < stage1_losses[0]

    def test_training_history_populated(
        self,
        network: PINNBase,
        pde_problem: SimplePDEProblem,
        training_config: TrainingConfig,
    ) -> None:
        """Test that training history is populated after training."""
        trainer = HybridTrainer(
            model=network,
            pde_problem=pde_problem,
            config=training_config,
            device="cpu",
        )

        # Run training
        result = trainer.train()

        # Check that history is populated
        assert len(trainer.history.stage1_losses) > 0
        assert len(trainer.history.stage2_losses) > 0
        assert len(trainer.history.stage3_losses) > 0

        # Check result dictionary
        assert "stage1_losses" in result
        assert "stage2_losses" in result
        assert "stage3_losses" in result

    def test_callbacks_are_called(
        self,
        network: PINNBase,
        pde_problem: SimplePDEProblem,
        training_config: TrainingConfig,
    ) -> None:
        """Test that callbacks are invoked at the right times."""
        mock_callback = MockCallback()
        trainer = HybridTrainer(
            model=network,
            pde_problem=pde_problem,
            config=training_config,
            callbacks=[mock_callback],
            device="cpu",
        )

        # Run training
        trainer.train()

        # Check that callback methods were called
        assert mock_callback.train_begin_count == 1
        assert mock_callback.train_end_count == 1
        assert mock_callback.stage_begin_count == 3  # 3 stages
        assert mock_callback.stage_end_count == 3

        # Check that all stages were seen
        assert "stage1" in mock_callback.stages_seen
        assert "stage2" in mock_callback.stages_seen
        assert "stage3" in mock_callback.stages_seen

        # Check that epochs were processed
        expected_epochs = (
            training_config.stage1_epochs
            + training_config.stage2_epochs
            + training_config.stage3_epochs
        )
        assert mock_callback.epoch_end_count == expected_epochs

    def test_stage1_pretrain(
        self,
        network: PINNBase,
        pde_problem: SimplePDEProblem,
        training_config: TrainingConfig,
    ) -> None:
        """Test stage 1 pretraining runs correctly."""
        trainer = HybridTrainer(
            model=network,
            pde_problem=pde_problem,
            config=training_config,
            device="cpu",
        )

        # Run only stage 1
        trainer.stage1_pretrain()

        # Check history is populated for stage 1
        assert len(trainer.history.stage1_losses) > 0
        assert len(trainer.history.stage2_losses) == 0
        assert len(trainer.history.stage3_losses) == 0

        # Check loss keys
        first_loss = trainer.history.stage1_losses[0]
        assert "total" in first_loss

    def test_stage2_physics_refine(
        self,
        network: PINNBase,
        pde_problem: SimplePDEProblem,
        training_config: TrainingConfig,
    ) -> None:
        """Test stage 2 physics refinement runs correctly."""
        trainer = HybridTrainer(
            model=network,
            pde_problem=pde_problem,
            config=training_config,
            device="cpu",
        )

        # Run stage 2 (which requires stage 1 first)
        trainer.stage1_pretrain()
        trainer.stage2_physics_refine()

        # Check history is populated
        assert len(trainer.history.stage1_losses) > 0
        assert len(trainer.history.stage2_losses) > 0
        assert len(trainer.history.stage3_losses) == 0

        # Check loss keys for stage 2
        first_loss = trainer.history.stage2_losses[0]
        assert "total" in first_loss
        assert "physics_loss" in first_loss

    def test_stage2_adaptive_lambda_logs_metrics(
        self,
        network: PINNBase,
        pde_problem: SimplePDEProblem,
    ) -> None:
        config = TrainingConfig(
            stage1_epochs=2,
            stage1_lr=1e-3,
            stage2_epochs=2,
            stage2_lr=1e-4,
            stage3_epochs=0,
            stage3_lr=1e-5,
            batch_size=32,
            num_collocation_points=64,
            num_boundary_points=16,
            adaptive_lambda_enabled=True,
            adaptive_lambda_warmup_epochs=0,
            seed=42,
            device="cpu",
        )
        trainer = HybridTrainer(
            model=network,
            pde_problem=pde_problem,
            config=config,
            device="cpu",
        )

        trainer.stage1_pretrain()
        trainer.stage2_physics_refine()

        first_loss = trainer.history.stage2_losses[0]
        assert "lambda_physics" in first_loss
        assert "grad_cosine" in first_loss
        assert "grad_norm_ratio" in first_loss
        assert "physics_improvement" in first_loss
        assert first_loss["lambda_physics"] == pytest.approx(0.1 * config.lambda_physics)

        later_loss = trainer.history.stage2_losses[-1]
        max_lambda = config.lambda_physics * config.adaptive_lambda_max_scale
        assert later_loss["lambda_physics"] <= max_lambda

    def test_stage2_adaptive_lambda_can_change(
        self,
        network: PINNBase,
        pde_problem: SimplePDEProblem,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = TrainingConfig(
            stage1_epochs=1,
            stage1_lr=1e-3,
            stage2_epochs=3,
            stage2_lr=1e-4,
            stage3_epochs=0,
            stage3_lr=1e-5,
            batch_size=16,
            num_collocation_points=32,
            num_boundary_points=8,
            adaptive_lambda_enabled=True,
            adaptive_lambda_warmup_epochs=0,
            adaptive_lambda_adjust_rate=0.5,
            seed=42,
            device="cpu",
        )
        trainer = HybridTrainer(
            model=network,
            pde_problem=pde_problem,
            config=config,
            device="cpu",
        )

        def fake_metrics(*_args: object, **_kwargs: object) -> dict[str, float]:
            return {
                "grad_cosine": 0.6,
                "grad_norm_ratio": 1.0,
                "objective_grad_norm": 1.0,
                "physics_grad_norm": 1.0,
                "physics_improvement": 0.0,
            }

        monkeypatch.setattr(trainer, "_compute_adaptive_lambda_metrics", fake_metrics)

        trainer.stage1_pretrain()
        trainer.stage2_physics_refine()

        lambdas = [row["lambda_physics"] for row in trainer.history.stage2_losses]
        assert lambdas[1] > lambdas[0]
        assert lambdas[-1] >= lambdas[1]

    def test_stage3_causal_calibrate(
        self,
        network: PINNBase,
        pde_problem: SimplePDEProblem,
        training_config: TrainingConfig,
    ) -> None:
        """Test stage 3 causal calibration runs correctly."""
        trainer = HybridTrainer(
            model=network,
            pde_problem=pde_problem,
            config=training_config,
            device="cpu",
        )

        # Run all stages
        trainer.stage1_pretrain()
        trainer.stage2_physics_refine()
        trainer.stage3_causal_calibrate()

        # Check history is populated
        assert len(trainer.history.stage1_losses) > 0
        assert len(trainer.history.stage2_losses) > 0
        assert len(trainer.history.stage3_losses) > 0

        # Check loss keys for stage 3
        first_loss = trainer.history.stage3_losses[0]
        assert "total" in first_loss
        assert "causal_loss" in first_loss

    def test_training_with_exact_solution(
        self,
        network: PINNBase,
        pde_problem: SimplePDEProblem,
        training_config: TrainingConfig,
    ) -> None:
        """Test training with a PDE that has an exact solution."""
        trainer = HybridTrainer(
            model=network,
            pde_problem=pde_problem,
            config=training_config,
            device="cpu",
        )

        # Run training
        trainer.train()

        # Check that training completed
        assert len(trainer.history.stage1_losses) > 0
        assert len(trainer.history.stage2_losses) > 0
        assert len(trainer.history.stage3_losses) > 0

        # Check that data_loss is present (from exact solution)
        first_loss = trainer.history.stage1_losses[0]
        assert "data_loss" in first_loss

    def test_early_stopping_callback(
        self,
        network: PINNBase,
        pde_problem: SimplePDEProblem,
    ) -> None:
        """Test that early stopping callback works."""
        # Use many epochs but early stopping with low patience
        config = TrainingConfig(
            stage1_epochs=100,
            stage1_lr=1e-3,
            stage2_epochs=100,
            stage2_lr=1e-4,
            stage3_epochs=100,
            stage3_lr=1e-5,
            batch_size=64,
            num_collocation_points=100,
            num_boundary_points=20,
            seed=42,
            device="cpu",
        )

        early_stop = EarlyStoppingCallback(patience=3, min_delta=1e-4)
        trainer = HybridTrainer(
            model=network,
            pde_problem=pde_problem,
            config=config,
            callbacks=[early_stop],
            device="cpu",
        )

        # Run training
        trainer.train()

        # Early stopping should have triggered in at least one stage
        # or training should have completed
        total_epochs = (
            len(trainer.history.stage1_losses)
            + len(trainer.history.stage2_losses)
            + len(trainer.history.stage3_losses)
        )
        # Should be less than full 300 epochs if early stopping worked
        # (or training completed normally with fewer epochs configured)
        assert total_epochs < 300 or early_stop.stopped_epoch >= 0


class TestCheckpointCallback:
    """Tests for CheckpointCallback."""

    def test_checkpoint_callback_saves(
        self,
        network: PINNBase,
        pde_problem: SimplePDEProblem,
        training_config: TrainingConfig,
        tmp_path: Path,
    ) -> None:
        """Test that checkpoint callback saves model."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_callback = CheckpointCallback(
            checkpoint_dir=checkpoint_dir,
            save_every_n_epochs=2,
            save_best=True,
        )

        trainer = HybridTrainer(
            model=network,
            pde_problem=pde_problem,
            config=training_config,
            callbacks=[checkpoint_callback],
            device="cpu",
        )

        # Run training
        trainer.train()

        # Check that checkpoint files were created
        assert checkpoint_dir.exists()
        checkpoint_files = list(checkpoint_dir.glob("*.pt"))
        assert len(checkpoint_files) > 0


class TestLoggingCallback:
    """Tests for LoggingCallback."""

    def test_logging_callback_verbose(
        self,
        network: PINNBase,
        pde_problem: SimplePDEProblem,
        training_config: TrainingConfig,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that logging callback prints output when verbose."""
        logging_callback = LoggingCallback(
            log_every_n_epochs=2,
            verbose=True,
        )

        trainer = HybridTrainer(
            model=network,
            pde_problem=pde_problem,
            config=training_config,
            callbacks=[logging_callback],
            device="cpu",
        )

        # Run training
        trainer.train()

        # Check that output was printed
        captured = capsys.readouterr()
        assert "Starting stage1" in captured.out or "Epoch" in captured.out

    def test_logging_callback_quiet(
        self,
        network: PINNBase,
        pde_problem: SimplePDEProblem,
        training_config: TrainingConfig,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that logging callback is quiet when verbose=False."""
        logging_callback = LoggingCallback(
            log_every_n_epochs=1,
            verbose=False,
        )

        trainer = HybridTrainer(
            model=network,
            pde_problem=pde_problem,
            config=training_config,
            callbacks=[logging_callback],
            device="cpu",
        )

        # Run training
        trainer.train()

        # Check that no output was printed
        captured = capsys.readouterr()
        assert "Epoch" not in captured.out


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    def test_early_stopping_initialization(self) -> None:
        """Test early stopping callback initialization."""
        callback = EarlyStoppingCallback(patience=10, min_delta=1e-5)

        assert callback.patience == 10
        assert callback.min_delta == 1e-5
        assert callback.best_loss == float("inf")
        assert callback.wait == 0
        assert callback.stopped_epoch == -1
        assert not callback.should_stop


class TestTrainerDimensionAgnostic:
    """Tests for dimension-agnostic trainer behavior."""

    def test_generate_collocation_points_wave2d_shapes(self) -> None:
        model = PINNBase(input_dim=3, output_dim=1, hidden_dims=[16, 16])
        pde = Wave2D(wave_speed=1.0, length_x=1.0, length_y=1.0, t_max=1.0)
        config = TrainingConfig(
            stage1_epochs=1,
            stage1_lr=1e-3,
            stage2_epochs=0,
            stage2_lr=1e-4,
            stage3_epochs=0,
            stage3_lr=1e-5,
            batch_size=8,
            num_collocation_points=32,
            num_boundary_points=8,
            seed=42,
            device="cpu",
        )

        trainer = HybridTrainer(model=model, pde_problem=pde, config=config, device="cpu")
        t, x = trainer._generate_collocation_points(32)

        assert t.shape == (32,)
        assert x.shape == (32, 2)
        assert t.requires_grad
        assert x.requires_grad

    def test_generate_synthetic_data_wave2d_shapes(self) -> None:
        model = PINNBase(input_dim=3, output_dim=1, hidden_dims=[16, 16])
        pde = Wave2D(wave_speed=1.0, length_x=1.0, length_y=1.0, t_max=1.0)
        config = TrainingConfig(
            stage1_epochs=0,
            stage1_lr=1e-3,
            stage2_epochs=0,
            stage2_lr=1e-4,
            stage3_epochs=0,
            stage3_lr=1e-5,
            batch_size=8,
            num_collocation_points=16,
            num_boundary_points=8,
            seed=42,
            device="cpu",
        )

        trainer = HybridTrainer(model=model, pde_problem=pde, config=config, device="cpu")
        t, x, u = trainer._generate_synthetic_data(16)

        assert t.shape == (16,)
        assert x.shape == (16, 2)
        assert u.shape == (16,)

    def test_wave2d_stage1_smoke_one_epoch(self) -> None:
        model = PINNBase(input_dim=3, output_dim=1, hidden_dims=[16, 16])
        pde = Wave2D(wave_speed=1.0, length_x=1.0, length_y=1.0, t_max=1.0)
        config = TrainingConfig(
            stage1_epochs=1,
            stage1_lr=1e-3,
            stage2_epochs=0,
            stage2_lr=1e-4,
            stage3_epochs=0,
            stage3_lr=1e-5,
            batch_size=8,
            num_collocation_points=32,
            num_boundary_points=8,
            seed=42,
            device="cpu",
        )

        trainer = HybridTrainer(model=model, pde_problem=pde, config=config, device="cpu")
        history = trainer.train()

        assert len(history["stage1_total_loss"]) == 1
        assert len(history["stage2_total_loss"]) == 0
        assert len(history["stage3_total_loss"]) == 0

    def test_wave2d_boundary_loss_computable(self) -> None:
        model = PINNBase(input_dim=3, output_dim=1, hidden_dims=[16, 16])
        pde = Wave2D(wave_speed=1.0, length_x=1.0, length_y=1.0, t_max=1.0)
        config = TrainingConfig(
            stage1_epochs=0,
            stage1_lr=1e-3,
            stage2_epochs=0,
            stage2_lr=1e-4,
            stage3_epochs=0,
            stage3_lr=1e-5,
            batch_size=8,
            num_collocation_points=16,
            num_boundary_points=32,
            seed=42,
            device="cpu",
        )

        trainer = HybridTrainer(model=model, pde_problem=pde, config=config, device="cpu")
        bc_loss = trainer._compute_boundary_loss()

        assert torch.isfinite(bc_loss)

    def test_early_stopping_resets_on_stage_begin(self) -> None:
        """Test that early stopping resets at stage start."""
        callback = EarlyStoppingCallback(patience=5)
        callback.best_loss = 0.5
        callback.wait = 3
        callback._should_stop = True

        callback.on_stage_begin(None, "stage1")  # type: ignore

        assert callback.best_loss == float("inf")
        assert callback.wait == 0
        assert callback.stopped_epoch == -1
        assert not callback.should_stop
