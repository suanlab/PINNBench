"""Training callbacks for the PINN training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from causal_pinn.training.trainer import HybridTrainer


class Callback:
    """Base class for training callbacks.

    Callbacks are called at various points during training to enable
    custom functionality like logging, checkpointing, early stopping, etc.
    """

    def on_train_begin(self, trainer: HybridTrainer) -> None:
        """Called at the beginning of training.

        Args:
            trainer: The trainer instance.
        """
        pass

    def on_train_end(self, trainer: HybridTrainer) -> None:
        """Called at the end of training.

        Args:
            trainer: The trainer instance.
        """
        pass

    def on_stage_begin(self, trainer: HybridTrainer, stage: str) -> None:
        """Called at the beginning of each training stage.

        Args:
            trainer: The trainer instance.
            stage: Name of the stage ('stage1', 'stage2', 'stage3').
        """
        pass

    def on_stage_end(self, trainer: HybridTrainer, stage: str) -> None:
        """Called at the end of each training stage.

        Args:
            trainer: The trainer instance.
            stage: Name of the stage ('stage1', 'stage2', 'stage3').
        """
        pass

    def on_epoch_begin(self, trainer: HybridTrainer, epoch: int, stage: str) -> None:
        """Called at the beginning of each epoch.

        Args:
            trainer: The trainer instance.
            epoch: Current epoch number.
            stage: Name of the current stage.
        """
        pass

    def on_epoch_end(
        self,
        trainer: HybridTrainer,
        epoch: int,
        stage: str,
        losses: dict[str, float],
    ) -> None:
        """Called at the end of each epoch.

        Args:
            trainer: The trainer instance.
            epoch: Current epoch number.
            stage: Name of the current stage.
            losses: Dictionary of loss values for this epoch.
        """
        pass


class CheckpointCallback(Callback):
    """Callback to save model checkpoints during training.

    Saves the model state dict at specified intervals and at the end
    of each training stage.

    Attributes:
        checkpoint_dir: Directory to save checkpoints.
        save_every_n_epochs: Save checkpoint every N epochs. If 0, only saves at stage end.
        save_best: Whether to save the best model based on loss.
        best_loss: Best loss seen so far (for save_best).
    """

    def __init__(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        save_every_n_epochs: int = 0,
        save_best: bool = True,
    ) -> None:
        """Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints.
            save_every_n_epochs: Save checkpoint every N epochs. If 0, only saves at stage end.
            save_best: Whether to track and save the best model.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every_n_epochs = save_every_n_epochs
        self.save_best = save_best
        self.best_loss: float = float("inf")
        self._best_state_dict: dict[str, Any] | None = None

    def on_train_begin(self, trainer: HybridTrainer) -> None:
        """Create checkpoint directory at training start."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")
        self._best_state_dict = None

    def on_epoch_end(
        self,
        trainer: HybridTrainer,
        epoch: int,
        stage: str,
        losses: dict[str, float],
    ) -> None:
        """Save checkpoint if conditions are met."""
        current_loss = losses.get("total", float("inf"))

        # Track best model
        if self.save_best and current_loss < self.best_loss:
            self.best_loss = current_loss
            self._best_state_dict = trainer.model.state_dict()

        # Save periodic checkpoint
        if self.save_every_n_epochs > 0 and (epoch + 1) % self.save_every_n_epochs == 0:
            self._save_checkpoint(trainer, f"{stage}_epoch_{epoch + 1}.pt")

    def on_stage_end(self, trainer: HybridTrainer, stage: str) -> None:
        """Save checkpoint at end of each stage."""
        self._save_checkpoint(trainer, f"{stage}_final.pt")

        # Save best model from this stage
        if self.save_best and self._best_state_dict is not None:
            torch.save(self._best_state_dict, self.checkpoint_dir / f"{stage}_best.pt")

    def on_train_end(self, trainer: HybridTrainer) -> None:
        """Save final best model at training end."""
        if self.save_best and self._best_state_dict is not None:
            torch.save(self._best_state_dict, self.checkpoint_dir / "best_model.pt")

    def _save_checkpoint(self, trainer: HybridTrainer, filename: str) -> None:
        """Save a model checkpoint.

        Args:
            trainer: The trainer instance.
            filename: Name for the checkpoint file.
        """
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(trainer.model.state_dict(), checkpoint_path)


class LoggingCallback(Callback):
    """Callback for logging training progress.

    Prints loss summaries at specified intervals during training.

    Attributes:
        log_every_n_epochs: Print summary every N epochs.
        verbose: Whether to print to stdout.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 100,
        verbose: bool = True,
    ) -> None:
        """Initialize logging callback.

        Args:
            log_every_n_epochs: Print summary every N epochs.
            verbose: Whether to print to stdout.
        """
        self.log_every_n_epochs = log_every_n_epochs
        self.verbose = verbose

    def on_stage_begin(self, trainer: HybridTrainer, stage: str) -> None:
        """Log stage start."""
        if self.verbose:
            print(f"\n{'=' * 50}")
            print(f"Starting {stage}")
            print(f"{'=' * 50}")

    def on_epoch_end(
        self,
        trainer: HybridTrainer,
        epoch: int,
        stage: str,
        losses: dict[str, float],
    ) -> None:
        """Log epoch progress."""
        if not self.verbose:
            return

        if (epoch + 1) % self.log_every_n_epochs == 0:
            loss_str = " | ".join([f"{k}: {v:.6f}" for k, v in losses.items()])
            print(f"[{stage}] Epoch {epoch + 1}: {loss_str}")

    def on_stage_end(self, trainer: HybridTrainer, stage: str) -> None:
        """Log stage completion."""
        if self.verbose:
            print(f"Completed {stage}")


class EarlyStoppingCallback(Callback):
    """Callback for early stopping based on loss plateau.

    Monitors the total loss and stops training if it doesn't improve
    for a specified number of epochs.

    Attributes:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
        best_loss: Best loss seen so far.
        wait: Number of epochs waited without improvement.
        stopped_epoch: Epoch at which training was stopped (-1 if not stopped).
    """

    def __init__(
        self,
        patience: int = 100,
        min_delta: float = 1e-6,
        monitor: str = "total",
    ) -> None:
        """Initialize early stopping callback.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change in loss to qualify as an improvement.
            monitor: Loss key to monitor (default: 'total').
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_loss: float = float("inf")
        self.wait: int = 0
        self.stopped_epoch: int = -1
        self._should_stop: bool = False

    def on_stage_begin(self, trainer: HybridTrainer, stage: str) -> None:
        """Reset state at the beginning of each stage."""
        self.best_loss = float("inf")
        self.wait = 0
        self.stopped_epoch = -1
        self._should_stop = False

    def on_epoch_end(
        self,
        trainer: HybridTrainer,
        epoch: int,
        stage: str,
        losses: dict[str, float],
    ) -> None:
        """Check for early stopping condition."""
        current_loss = losses.get(self.monitor, float("inf"))

        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self._should_stop = True

    @property
    def should_stop(self) -> bool:
        """Return whether training should stop."""
        return self._should_stop


class LRSchedulerCallback(Callback):
    """Callback to update learning rate scheduler.

    Steps the learning rate scheduler at the end of each epoch.

    Attributes:
        scheduler: The learning rate scheduler.
    """

    def __init__(self, scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
        """Initialize LR scheduler callback.

        Args:
            scheduler: PyTorch learning rate scheduler.
        """
        self.scheduler = scheduler

    def on_epoch_end(
        self,
        trainer: HybridTrainer,
        epoch: int,
        stage: str,
        losses: dict[str, float],
    ) -> None:
        """Step the scheduler at the end of each epoch."""
        self.scheduler.step()


class GradientClippingCallback(Callback):
    """Callback to clip gradients during training.

    Clips gradients by norm after backward pass.

    Attributes:
        max_norm: Maximum gradient norm.
    """

    def __init__(self, max_norm: float = 1.0) -> None:
        """Initialize gradient clipping callback.

        Args:
            max_norm: Maximum gradient norm for clipping.
        """
        self.max_norm = max_norm

    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients of the model.

        Args:
            model: The model whose gradients to clip.

        Returns:
            The total gradient norm before clipping.
        """
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
        return float(norm)
