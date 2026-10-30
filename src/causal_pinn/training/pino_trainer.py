from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor
from torch.optim import Adam

from ..models.pino import PINOModel
from ..training.trainer import TrainingHistory
from ..utils.random import set_seed


class PINOCallback(Protocol):
    def on_train_begin(self, trainer: object) -> None: ...

    def on_train_end(self, trainer: object) -> None: ...

    def on_stage_begin(self, trainer: object, stage: str) -> None: ...

    def on_stage_end(self, trainer: object, stage: str) -> None: ...

    def on_epoch_begin(self, trainer: object, epoch: int, stage: str) -> None: ...

    def on_epoch_end(
        self,
        trainer: object,
        epoch: int,
        stage: str,
        losses: dict[str, float],
    ) -> None: ...


class PINOTrainer:
    def __init__(
        self,
        model: PINOModel,
        pde_problem,
        config,
        callbacks: list[PINOCallback] | None = None,
        device: str | None = None,
    ) -> None:
        self.model = model
        self.pde_problem = pde_problem
        self.config = config
        self.callbacks = callbacks or []

        if device is not None:
            self.device = torch.device(device)
        elif hasattr(config, "device"):
            self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)
        self.history = TrainingHistory()
        self._current_stage = ""
        self._current_epoch = 0

        set_seed(config.seed)

    def train(self) -> TrainingHistory:
        for callback in self.callbacks:
            callback.on_train_begin(self)

        self.stage1_pretrain()
        self.stage2_physics_refine()
        self.stage3_finetune()

        for callback in self.callbacks:
            callback.on_train_end(self)

        return self.history

    def stage1_pretrain(self) -> None:
        self._current_stage = "stage1"
        for callback in self.callbacks:
            callback.on_stage_begin(self, "stage1")

        optimizer = Adam(self.model.parameters(), lr=self.config.stage1_lr)
        t_colloc, x_colloc = self._generate_collocation_points(self.config.num_collocation_points)
        has_exact_solution = self._check_exact_solution()

        for epoch in range(self.config.stage1_epochs):
            self._current_epoch = epoch
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch, "stage1")

            optimizer.zero_grad()
            pino_losses = self.model.compute_pino_loss(t_colloc, x_colloc)

            total_loss = pino_losses["physics_loss"] + pino_losses["bc_loss"]
            losses = {
                "physics_loss": pino_losses["physics_loss"].item(),
                "bc_loss": pino_losses["bc_loss"].item(),
            }
            if has_exact_solution:
                total_loss = total_loss + pino_losses["data_loss"]
                losses["data_loss"] = pino_losses["data_loss"].item()

            losses["total"] = total_loss.item()
            total_loss.backward()
            optimizer.step()

            self.history.add_stage1_loss(losses)

            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, "stage1", losses)
            if self._check_early_stopping():
                break

        for callback in self.callbacks:
            callback.on_stage_end(self, "stage1")

    def stage2_physics_refine(self) -> None:
        self._current_stage = "stage2"
        for callback in self.callbacks:
            callback.on_stage_begin(self, "stage2")

        optimizer = Adam(self.model.parameters(), lr=self.config.stage2_lr)
        t_colloc, x_colloc = self._generate_collocation_points(self.config.num_collocation_points)
        has_exact_solution = self._check_exact_solution()

        for epoch in range(self.config.stage2_epochs):
            self._current_epoch = epoch
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch, "stage2")

            progress = min(1.0, epoch / (self.config.stage2_epochs / 2))
            current_lambda = (0.1 + 0.9 * progress) * self.config.lambda_physics

            optimizer.zero_grad()
            pino_losses = self.model.compute_pino_loss(t_colloc, x_colloc)

            total_loss = current_lambda * pino_losses["physics_loss"] + pino_losses["bc_loss"]
            losses = {
                "physics_loss": pino_losses["physics_loss"].item(),
                "bc_loss": pino_losses["bc_loss"].item(),
                "lambda_physics": current_lambda,
            }
            if has_exact_solution:
                total_loss = total_loss + self.config.lambda_data * pino_losses["data_loss"]
                losses["data_loss"] = pino_losses["data_loss"].item()

            losses["total"] = total_loss.item()
            total_loss.backward()
            optimizer.step()

            self.history.add_stage2_loss(losses)

            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, "stage2", losses)
            if self._check_early_stopping():
                break

        for callback in self.callbacks:
            callback.on_stage_end(self, "stage2")

    def stage3_finetune(self) -> None:
        self._current_stage = "stage3"
        for callback in self.callbacks:
            callback.on_stage_begin(self, "stage3")

        optimizer = Adam(self.model.parameters(), lr=self.config.stage3_lr)
        t_colloc, x_colloc = self._generate_collocation_points(self.config.num_collocation_points)

        for epoch in range(self.config.stage3_epochs):
            self._current_epoch = epoch
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch, "stage3")

            optimizer.zero_grad()
            pino_losses = self.model.compute_pino_loss(t_colloc, x_colloc)

            total_loss = (
                self.config.lambda_physics * pino_losses["physics_loss"] + pino_losses["bc_loss"]
            )
            losses = {
                "physics_loss": pino_losses["physics_loss"].item(),
                "bc_loss": pino_losses["bc_loss"].item(),
                "total": total_loss.item(),
            }

            total_loss.backward()
            optimizer.step()

            self.history.add_stage3_loss(losses)

            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, "stage3", losses)
            if self._check_early_stopping():
                break

        for callback in self.callbacks:
            callback.on_stage_end(self, "stage3")

    def _generate_collocation_points(self, num_points: int) -> tuple[Tensor, Tensor]:
        t_bounds, x_bounds = self.pde_problem.domain_bounds
        t = torch.rand(num_points, device=self.device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
        x = torch.rand(num_points, device=self.device) * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
        return t.requires_grad_(True), x.requires_grad_(True)

    def _check_exact_solution(self) -> bool:
        try:
            t_sample = torch.tensor([0.5], device=self.device)
            x_sample = torch.tensor([0.5], device=self.device)
            result = self.pde_problem.exact_solution(t_sample, x_sample)
            return result is not None
        except (NotImplementedError, AttributeError, TypeError):
            return False

    def _check_early_stopping(self) -> bool:
        return any(getattr(callback, "should_stop", False) for callback in self.callbacks)

    @property
    def current_stage(self) -> str:
        return self._current_stage

    @property
    def current_epoch(self) -> int:
        return self._current_epoch
