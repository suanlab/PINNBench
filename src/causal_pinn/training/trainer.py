"""3-Stage training pipeline for Causal-Aware Hybrid PINN-Operator framework.

This module implements the three-stage training strategy:
1. Stage 1: Operator pretraining on synthetic data
2. Stage 2: Physics refinement with increasing lambda_physics
3. Stage 3: Causal calibration with AdaptiveCausalLoss
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import Adam

from causal_pinn.physics.losses import AdaptiveCausalLoss, BoundaryLoss, PhysicsLoss
from causal_pinn.training.adaptive_lambda import (
    AdaptiveLambdaController,
    AdaptiveLambdaMetrics,
)
from causal_pinn.training.callbacks import (
    Callback,
    EarlyStoppingCallback,
)
from causal_pinn.utils.random import set_seed

if TYPE_CHECKING:
    from causal_pinn.physics.pde.base import PDEProblem
    from causal_pinn.utils.config import TrainingConfig


@dataclass
class TrainingHistory:
    """Stores per-epoch training losses for each stage.

    Attributes:
        stage1_losses: List of loss dictionaries for each epoch in stage 1.
        stage2_losses: List of loss dictionaries for each epoch in stage 2.
        stage3_losses: List of loss dictionaries for each epoch in stage 3.
        stage1_total_loss: List of total loss values for stage 1.
        stage2_total_loss: List of total loss values for stage 2.
        stage3_total_loss: List of total loss values for stage 3.
    """

    stage1_losses: list[dict[str, float]] = field(default_factory=list)
    stage2_losses: list[dict[str, float]] = field(default_factory=list)
    stage3_losses: list[dict[str, float]] = field(default_factory=list)
    stage1_total_loss: list[float] = field(default_factory=list)
    stage2_total_loss: list[float] = field(default_factory=list)
    stage3_total_loss: list[float] = field(default_factory=list)

    def add_stage1_loss(self, losses: dict[str, float]) -> None:
        """Add loss record for stage 1.

        Args:
            losses: Dictionary of loss values.
        """
        self.stage1_losses.append(losses)
        self.stage1_total_loss.append(losses.get("total", 0.0))

    def add_stage2_loss(self, losses: dict[str, float]) -> None:
        """Add loss record for stage 2.

        Args:
            losses: Dictionary of loss values.
        """
        self.stage2_losses.append(losses)
        self.stage2_total_loss.append(losses.get("total", 0.0))

    def add_stage3_loss(self, losses: dict[str, float]) -> None:
        """Add loss record for stage 3.

        Args:
            losses: Dictionary of loss values.
        """
        self.stage3_losses.append(losses)
        self.stage3_total_loss.append(losses.get("total", 0.0))

    def to_dict(self) -> dict[str, Any]:
        """Convert history to dictionary for serialization.

        Returns:
            Dictionary representation of the training history.
        """
        return {
            "stage1_losses": self.stage1_losses,
            "stage2_losses": self.stage2_losses,
            "stage3_losses": self.stage3_losses,
            "stage1_total_loss": self.stage1_total_loss,
            "stage2_total_loss": self.stage2_total_loss,
            "stage3_total_loss": self.stage3_total_loss,
        }


class HybridTrainer:
    """3-Stage trainer for Causal-Aware Hybrid PINN-Operator.

    Implements the complete training pipeline with three stages:
    1. Pretrain: Train on synthetic data generated from exact solution or random points
    2. Physics Refine: Add physics residual loss with increasing weight
    3. Causal Calibrate: Enable causal loss for temporal consistency

    Attributes:
        model: The neural network model (any nn.Module).
        pde_problem: The PDE problem definition.
        config: Training configuration.
        callbacks: List of training callbacks.
        device: Device to train on.
        history: Training history recording losses per epoch.
    """

    def __init__(
        self,
        model: nn.Module,
        pde_problem: PDEProblem,
        config: TrainingConfig,
        callbacks: list[Callback] | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the hybrid trainer.

        Args:
            model: Neural network model (any nn.Module).
            pde_problem: PDE problem defining physics and boundary conditions.
            config: Training configuration with stage hyperparameters.
            callbacks: Optional list of training callbacks.
            device: Device to train on. If None, uses config.device or 'cpu'.
        """
        self.model = model
        self.pde_problem = pde_problem
        self.config = config
        self.callbacks = callbacks or []

        # Set device
        if device is not None:
            self.device = torch.device(device)
        elif hasattr(config, "device"):
            self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        # Move model to device
        self.model = self.model.to(self.device)

        # Initialize history
        self.history = TrainingHistory()

        # Initialize loss functions
        self._physics_loss = PhysicsLoss(pde_problem)
        self._boundary_loss = BoundaryLoss()
        self._causal_loss: AdaptiveCausalLoss | None = None
        self._adaptive_lambda_controller: AdaptiveLambdaController | None = None

        # Training state
        self._current_stage: str = ""
        self._current_epoch: int = 0

        # Set random seed
        set_seed(config.seed)

    def train(self) -> dict[str, Any]:
        """Run the complete 3-stage training pipeline.

        Returns:
            Dictionary containing training history and final metrics.
        """
        # Notify callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)

        # Stage 1: Operator pretraining
        if not self.config.skip_stage1:
            self.stage1_pretrain()

        # Stage 2: Physics refinement
        self.stage2_physics_refine()

        # Stage 3: Causal calibration
        if not self.config.skip_stage3:
            self.stage3_causal_calibrate()

        # Notify callbacks
        for callback in self.callbacks:
            callback.on_train_end(self)

        return self.history.to_dict()

    def stage1_pretrain(self) -> None:
        """Stage 1: Pretrain on synthetic data.

        Generates synthetic data from the exact solution if available,
        otherwise uses random points with physics loss only.
        """
        self._current_stage = "stage1"

        # Notify callbacks
        for callback in self.callbacks:
            callback.on_stage_begin(self, "stage1")

        # Create optimizer for stage 1
        optimizer = Adam(self.model.parameters(), lr=self.config.stage1_lr)

        # Generate collocation points
        t_colloc, x_colloc = self._generate_collocation_points(
            self.config.num_collocation_points
        )

        # Generate synthetic training data if exact solution is available
        has_exact_solution = self._check_exact_solution()

        # Training loop
        for epoch in range(self.config.stage1_epochs):
            self._current_epoch = epoch

            # Notify callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch, "stage1")

            optimizer.zero_grad()

            losses: dict[str, float] = {}

            if has_exact_solution:
                # Use data loss from exact solution
                t_data, x_data, u_data = self._generate_synthetic_data(
                    self.config.batch_size
                )
                data_loss = self._compute_data_loss(t_data, x_data, u_data)
                losses["data_loss"] = data_loss.item()

                # Also include physics loss
                physics_result = self._physics_loss(self.model, t_colloc, x_colloc)
                physics_loss = physics_result["physics_loss"]
                losses["physics_loss"] = physics_loss.item()

                total_loss = data_loss + physics_loss
            else:
                # Use only physics loss
                physics_result = self._physics_loss(self.model, t_colloc, x_colloc)
                total_loss = physics_result["physics_loss"]
                losses["physics_loss"] = total_loss.item()

            # Add boundary and initial condition losses
            bc_loss = self._compute_boundary_loss()
            losses["bc_loss"] = bc_loss.item()

            total_loss = total_loss + bc_loss
            losses["total"] = total_loss.item()

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Record history
            self.history.add_stage1_loss(losses)

            # Notify callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, "stage1", losses)

            # Check early stopping
            if self._check_early_stopping(epoch, "stage1"):
                break

        # Notify callbacks
        for callback in self.callbacks:
            callback.on_stage_end(self, "stage1")

    def stage2_physics_refine(self) -> None:
        """Stage 2: Physics refinement with increasing lambda_physics.

        Trains the full hybrid model with both data and physics losses,
        gradually increasing the physics weight.
        """
        self._current_stage = "stage2"

        # Notify callbacks
        for callback in self.callbacks:
            callback.on_stage_begin(self, "stage2")

        # Create optimizer for stage 2
        optimizer = Adam(self.model.parameters(), lr=self.config.stage2_lr)

        # Generate collocation points
        t_colloc, x_colloc = self._generate_collocation_points(
            self.config.num_collocation_points
        )

        has_exact_solution = self._check_exact_solution()
        if self.config.adaptive_lambda_enabled:
            self._adaptive_lambda_controller = AdaptiveLambdaController(
                warmup_epochs=self.config.adaptive_lambda_warmup_epochs,
                min_scale=self.config.adaptive_lambda_min_scale,
                max_scale=self.config.adaptive_lambda_max_scale,
                adjust_rate=self.config.adaptive_lambda_adjust_rate,
                alignment_threshold=self.config.adaptive_lambda_alignment_threshold,
                conflict_threshold=self.config.adaptive_lambda_conflict_threshold,
                dominance_threshold=self.config.adaptive_lambda_dominance_threshold,
                progress_threshold=self.config.adaptive_lambda_progress_threshold,
                ema_beta=self.config.adaptive_lambda_ema_beta,
                persistence=self.config.adaptive_lambda_persistence,
            )
        else:
            self._adaptive_lambda_controller = None

        # Training loop with increasing lambda_physics
        for epoch in range(self.config.stage2_epochs):
            self._current_epoch = epoch

            # Notify callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch, "stage2")

            optimizer.zero_grad()

            losses: dict[str, float] = {}
            objective_loss = torch.tensor(0.0, device=self.device)

            # Data loss (if exact solution available)
            if has_exact_solution:
                t_data, x_data, u_data = self._generate_synthetic_data(
                    self.config.batch_size
                )
                data_loss = self._compute_data_loss(t_data, x_data, u_data)
                weighted_data_loss = self.config.lambda_data * data_loss
                losses["data_loss"] = data_loss.item()
                objective_loss = objective_loss + weighted_data_loss

            # Physics loss with increasing weight
            physics_result = self._physics_loss(self.model, t_colloc, x_colloc)
            physics_loss = physics_result["physics_loss"]
            losses["physics_loss"] = physics_loss.item()

            # Boundary and initial condition losses
            bc_loss = self._compute_boundary_loss()
            losses["bc_loss"] = bc_loss.item()
            objective_loss = objective_loss + bc_loss

            adaptive_metrics = self._compute_adaptive_lambda_metrics(
                objective_loss=objective_loss,
                physics_loss=physics_loss,
            )
            current_lambda = self._get_stage2_lambda(epoch, adaptive_metrics)
            weighted_physics_loss = current_lambda * physics_loss
            losses["lambda_physics"] = current_lambda
            losses.update(adaptive_metrics)

            total_loss = objective_loss + weighted_physics_loss

            losses["total"] = total_loss.item()

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Record history
            self.history.add_stage2_loss(losses)

            # Notify callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, "stage2", losses)

            # Check early stopping
            if self._check_early_stopping(epoch, "stage2"):
                break

        # Notify callbacks
        for callback in self.callbacks:
            callback.on_stage_end(self, "stage2")

    def stage3_causal_calibrate(self) -> None:
        """Stage 3: Causal calibration with AdaptiveCausalLoss.

        Enables causal loss for temporal consistency, fine-tuning
        the model with the adaptive causal weighting strategy.
        """
        self._current_stage = "stage3"

        # Notify callbacks
        for callback in self.callbacks:
            callback.on_stage_begin(self, "stage3")

        # Create optimizer for stage 3
        optimizer = Adam(self.model.parameters(), lr=self.config.stage3_lr)

        # Initialize adaptive causal loss
        self._causal_loss = AdaptiveCausalLoss(
            self.pde_problem,
            epsilon_init=self.config.causal_epsilon_init,
            epsilon_final=self.config.causal_epsilon,
            num_time_bins=100,
            weight=1.0,
            annealing_steps=self.config.stage3_epochs,
        )

        # Generate collocation points
        t_colloc, x_colloc = self._generate_collocation_points(
            self.config.num_collocation_points
        )

        has_exact_solution = self._check_exact_solution()

        # Training loop
        for epoch in range(self.config.stage3_epochs):
            self._current_epoch = epoch

            # Notify callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch, "stage3")

            # Update causal loss annealing
            self._causal_loss.step()

            optimizer.zero_grad()

            losses: dict[str, float] = {}
            total_loss = torch.tensor(0.0, device=self.device)

            # Data loss (if exact solution available)
            if has_exact_solution:
                t_data, x_data, u_data = self._generate_synthetic_data(
                    self.config.batch_size
                )
                data_loss = self._compute_data_loss(t_data, x_data, u_data)
                weighted_data_loss = self.config.lambda_data * data_loss
                losses["data_loss"] = data_loss.item()
                total_loss = total_loss + weighted_data_loss

            # Causal loss
            causal_result = self._causal_loss(self.model, t_colloc, x_colloc)
            causal_loss = causal_result["causal_loss"]
            losses["causal_loss"] = causal_loss.item()
            losses["epsilon"] = self._causal_loss.epsilon
            total_loss = total_loss + causal_loss

            # Boundary and initial condition losses
            bc_loss = self._compute_boundary_loss()
            losses["bc_loss"] = bc_loss.item()
            total_loss = total_loss + bc_loss

            losses["total"] = total_loss.item()

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Record history
            self.history.add_stage3_loss(losses)

            # Notify callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, "stage3", losses)

            # Check early stopping
            if self._check_early_stopping(epoch, "stage3"):
                break

        # Notify callbacks
        for callback in self.callbacks:
            callback.on_stage_end(self, "stage3")

    def _generate_collocation_points(
        self,
        num_points: int,
    ) -> tuple[Tensor, Tensor]:
        """Generate uniform random collocation points in the domain.

        Args:
            num_points: Number of collocation points to generate.

        Returns:
            Tuple of (t, x) coordinate tensors with requires_grad=True.
        """
        domain_bounds = self.pde_problem.domain_bounds
        t_bounds, x_bounds = domain_bounds

        # Generate random points in domain
        t = torch.rand(num_points, device=self.device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
        x = self._sample_spatial_points(num_points, x_bounds)

        # Enable gradients for physics loss
        t = t.requires_grad_(True)
        x = x.requires_grad_(True)

        return t, x

    def _generate_synthetic_data(
        self,
        num_points: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate synthetic training data from exact solution.

        Args:
            num_points: Number of data points to generate.

        Returns:
            Tuple of (t, x, u) tensors where u is the exact solution.
        """
        domain_bounds = self.pde_problem.domain_bounds
        t_bounds, x_bounds = domain_bounds

        # Generate random points in domain
        t = torch.rand(num_points, device=self.device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
        x = self._sample_spatial_points(num_points, x_bounds)

        # Get exact solution
        u = self.pde_problem.exact_solution(t, x)

        return t, x, u

    def _check_exact_solution(self) -> bool:
        """Check if the PDE problem has an exact solution.

        Returns:
            True if exact_solution is implemented and returns non-None.
        """
        # Try a small sample to check if exact solution is available
        try:
            spatial_dim = self._infer_spatial_dim(self.pde_problem.domain_bounds[1])
            t_sample = torch.tensor([0.5], device=self.device)
            if spatial_dim == 1:
                x_sample = torch.tensor([0.5], device=self.device)
            else:
                x_sample = torch.full((1, spatial_dim), 0.5, device=self.device)
            result = self.pde_problem.exact_solution(t_sample, x_sample)
            return result is not None
        except (NotImplementedError, AttributeError, TypeError):
            return False

    def _compute_data_loss(
        self,
        t: Tensor,
        x: Tensor,
        u_true: Tensor,
    ) -> Tensor:
        """Compute MSE loss between prediction and true values.

        Args:
            t: Time coordinates.
            x: Spatial coordinates.
            u_true: True solution values.

        Returns:
            Scalar MSE loss tensor.
        """
        # Prepare inputs
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        inputs = torch.cat([t, x], dim=-1)
        u_pred = self.model(inputs)

        # Handle shape mismatch
        if u_pred.shape != u_true.shape and u_true.dim() == 1:
            u_true = u_true.unsqueeze(-1)

        return torch.mean((u_pred - u_true) ** 2)

    def _compute_boundary_loss(self) -> Tensor:
        """Compute boundary and initial condition losses.

        Returns:
            Combined scalar loss tensor.
        """
        domain_bounds = self.pde_problem.domain_bounds
        t_bounds, x_bounds = domain_bounds
        spatial_dim = self._infer_spatial_dim(x_bounds)

        if spatial_dim not in (1, 2):
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)

        if spatial_dim == 1:
            # Generate boundary points (left and right boundaries)
            num_bc = self.config.num_boundary_points // 2

            # Left boundary (x = x_min)
            t_left = (
                torch.rand(num_bc, device=self.device) * (t_bounds[1] - t_bounds[0])
                + t_bounds[0]
            )
            x_left = torch.full((num_bc,), x_bounds[0], device=self.device)
            u_bc_left = self.pde_problem.boundary_condition(t_left, x_left)

            # Right boundary (x = x_max)
            t_right = (
                torch.rand(num_bc, device=self.device) * (t_bounds[1] - t_bounds[0])
                + t_bounds[0]
            )
            x_right = torch.full((num_bc,), x_bounds[1], device=self.device)
            u_bc_right = self.pde_problem.boundary_condition(t_right, x_right)

            # Combine boundary points
            t_bc = torch.cat([t_left, t_right])
            x_bc = torch.cat([x_left, x_right])
            u_bc = torch.cat([u_bc_left, u_bc_right])

            # Generate initial condition points (t = t_min)
            num_ic = self.config.num_boundary_points
            x_ic = (
                torch.rand(num_ic, device=self.device) * (x_bounds[1] - x_bounds[0])
                + x_bounds[0]
            )
            u_ic = self.pde_problem.initial_condition(x_ic)

            bc_result = self._boundary_loss(
                self.model,
                t_bc=t_bc,
                x_bc=x_bc,
                u_bc=u_bc,
                x_ic=x_ic,
                u_ic=u_ic,
            )
            return bc_result["total"]

        x_min, x_max, y_min, y_max = x_bounds
        edge_count = max(1, self.config.num_boundary_points // 4)
        remainder = self.config.num_boundary_points - 4 * edge_count

        per_edge = [edge_count, edge_count, edge_count, edge_count]
        for idx in range(remainder):
            per_edge[idx % 4] += 1

        t_edges: list[Tensor] = []
        x_edges: list[Tensor] = []

        # x = x_min edge
        n0 = per_edge[0]
        t0 = torch.rand(n0, device=self.device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
        y0 = torch.rand(n0, device=self.device) * (y_max - y_min) + y_min
        x0 = torch.stack([torch.full((n0,), x_min, device=self.device), y0], dim=-1)
        t_edges.append(t0)
        x_edges.append(x0)

        # x = x_max edge
        n1 = per_edge[1]
        t1 = torch.rand(n1, device=self.device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
        y1 = torch.rand(n1, device=self.device) * (y_max - y_min) + y_min
        x1 = torch.stack([torch.full((n1,), x_max, device=self.device), y1], dim=-1)
        t_edges.append(t1)
        x_edges.append(x1)

        # y = y_min edge
        n2 = per_edge[2]
        t2 = torch.rand(n2, device=self.device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
        x2_coord = torch.rand(n2, device=self.device) * (x_max - x_min) + x_min
        x2 = torch.stack([x2_coord, torch.full((n2,), y_min, device=self.device)], dim=-1)
        t_edges.append(t2)
        x_edges.append(x2)

        # y = y_max edge
        n3 = per_edge[3]
        t3 = torch.rand(n3, device=self.device) * (t_bounds[1] - t_bounds[0]) + t_bounds[0]
        x3_coord = torch.rand(n3, device=self.device) * (x_max - x_min) + x_min
        x3 = torch.stack([x3_coord, torch.full((n3,), y_max, device=self.device)], dim=-1)
        t_edges.append(t3)
        x_edges.append(x3)

        t_bc = torch.cat(t_edges, dim=0)
        x_bc = torch.cat(x_edges, dim=0)
        u_bc = self.pde_problem.boundary_condition(t_bc, x_bc)

        num_ic = self.config.num_boundary_points
        x_ic = torch.stack(
            [
                torch.rand(num_ic, device=self.device) * (x_max - x_min) + x_min,
                torch.rand(num_ic, device=self.device) * (y_max - y_min) + y_min,
            ],
            dim=-1,
        )
        u_ic = self.pde_problem.initial_condition(x_ic)

        bc_result = self._boundary_loss(
            self.model,
            t_bc=t_bc,
            x_bc=x_bc,
            u_bc=u_bc,
            x_ic=x_ic,
            u_ic=u_ic,
        )

        return bc_result["total"]

    def _infer_spatial_dim(self, x_bounds: tuple[float, ...]) -> int:
        """Infer spatial dimension from flat min/max boundary tuple."""
        if len(x_bounds) % 2 != 0:
            raise ValueError(f"Invalid spatial bounds length: {len(x_bounds)}")
        return len(x_bounds) // 2

    def _sample_spatial_points(self, num_points: int, x_bounds: tuple[float, ...]) -> Tensor:
        """Sample random spatial points for 1D or 2D domains."""
        spatial_dim = self._infer_spatial_dim(x_bounds)

        if spatial_dim == 1:
            x_min, x_max = x_bounds[0], x_bounds[1]
            return torch.rand(num_points, device=self.device) * (x_max - x_min) + x_min

        components: list[Tensor] = []
        for dim in range(spatial_dim):
            x_min = x_bounds[2 * dim]
            x_max = x_bounds[2 * dim + 1]
            values = torch.rand(num_points, device=self.device) * (x_max - x_min) + x_min
            components.append(values)
        return torch.stack(components, dim=-1)

    def _compute_adaptive_lambda_metrics(
        self,
        objective_loss: Tensor,
        physics_loss: Tensor,
    ) -> dict[str, float]:
        if not self.config.adaptive_lambda_enabled:
            return {}

        objective_grad_norm, physics_grad_norm, grad_cosine = self._compute_gradient_conflict(
            objective_loss=objective_loss,
            physics_loss=physics_loss,
        )
        grad_norm_ratio = physics_grad_norm / max(objective_grad_norm, 1e-8)

        physics_improvement = 0.0
        if self._adaptive_lambda_controller is not None:
            physics_improvement = self._adaptive_lambda_controller.compute_improvement(
                physics_loss.item()
            )

        return {
            "grad_cosine": grad_cosine,
            "grad_norm_ratio": grad_norm_ratio,
            "objective_grad_norm": objective_grad_norm,
            "physics_grad_norm": physics_grad_norm,
            "physics_improvement": physics_improvement,
        }

    def _get_stage2_lambda(self, epoch: int, adaptive_metrics: dict[str, float]) -> float:
        progress = min(1.0, epoch / (self.config.stage2_epochs / 2))
        base_lambda = (0.1 + 0.9 * progress) * self.config.lambda_physics

        if self._adaptive_lambda_controller is None:
            return base_lambda

        metrics = AdaptiveLambdaMetrics(
            grad_cosine=adaptive_metrics["grad_cosine"],
            grad_norm_ratio=adaptive_metrics["grad_norm_ratio"],
            physics_improvement=adaptive_metrics["physics_improvement"],
            objective_grad_norm=adaptive_metrics["objective_grad_norm"],
            physics_grad_norm=adaptive_metrics["physics_grad_norm"],
        )
        return self._adaptive_lambda_controller.update(
            epoch=epoch,
            base_lambda=base_lambda,
            metrics=metrics,
        )

    def _compute_gradient_conflict(
        self,
        objective_loss: Tensor,
        physics_loss: Tensor,
    ) -> tuple[float, float, float]:
        params = [param for param in self.model.parameters() if param.requires_grad]
        objective_grads = torch.autograd.grad(
            objective_loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        physics_grads = torch.autograd.grad(
            physics_loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )

        flat_objective = self._flatten_gradients(objective_grads, params)
        flat_physics = self._flatten_gradients(physics_grads, params)

        objective_norm = torch.linalg.vector_norm(flat_objective).item()
        physics_norm = torch.linalg.vector_norm(flat_physics).item()

        if objective_norm < 1e-8 or physics_norm < 1e-8:
            return objective_norm, physics_norm, 0.0

        grad_cosine = F.cosine_similarity(flat_objective, flat_physics, dim=0).item()
        return objective_norm, physics_norm, grad_cosine

    def _flatten_gradients(
        self,
        gradients: tuple[Tensor | None, ...],
        params: Sequence[Tensor],
    ) -> Tensor:
        flat_parts: list[Tensor] = []
        for grad, param in zip(gradients, params, strict=True):
            if grad is None:
                flat_parts.append(torch.zeros_like(param, device=self.device).reshape(-1))
            else:
                flat_parts.append(grad.reshape(-1))

        if not flat_parts:
            return torch.zeros(1, device=self.device)
        return torch.cat(flat_parts)

    def _check_early_stopping(self, epoch: int, stage: str) -> bool:
        """Check if any callback has triggered early stopping.

        Args:
            epoch: Current epoch.
            stage: Current training stage.

        Returns:
            True if training should stop.
        """
        for callback in self.callbacks:
            if isinstance(callback, EarlyStoppingCallback) and callback.should_stop:
                return True
        return False

    @property
    def current_stage(self) -> str:
        """Return the current training stage name."""
        return self._current_stage

    @property
    def current_epoch(self) -> int:
        """Return the current epoch number within the stage."""
        return self._current_epoch
