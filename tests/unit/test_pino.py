from __future__ import annotations

import torch

from causal_pinn.models.operators.fno import FNO
from causal_pinn.models.pino import PINOModel
from causal_pinn.physics.pde.heat_1d import Heat1D
from causal_pinn.training.pino_trainer import PINOTrainer
from causal_pinn.training.trainer import TrainingHistory
from causal_pinn.utils.config import TrainingConfig


def _make_model_and_pde() -> tuple[PINOModel, Heat1D]:
    pde = Heat1D(alpha=1.0, length=1.0, t_max=1.0)
    fno = FNO(in_channels=2, out_channels=1, modes=8, width=16, num_layers=2, dim=1)
    return PINOModel(fno=fno, pde_problem=pde), pde


def test_pino_forward_shape_1d() -> None:
    model, _ = _make_model_and_pde()
    inputs = torch.rand(32, 2)
    out = model(inputs)
    assert out.shape == (32, 1)


def test_compute_pino_loss_returns_expected_keys() -> None:
    model, _ = _make_model_and_pde()
    t = torch.rand(64, requires_grad=True)
    x = torch.rand(64, requires_grad=True)
    losses = model.compute_pino_loss(t, x)

    assert "physics_loss" in losses
    assert "data_loss" in losses
    assert "bc_loss" in losses
    assert "total" in losses


def test_pino_trainer_train_returns_history() -> None:
    model, pde = _make_model_and_pde()
    config = TrainingConfig(
        stage1_epochs=1,
        stage1_lr=1e-3,
        stage2_epochs=1,
        stage2_lr=1e-4,
        stage3_epochs=1,
        stage3_lr=1e-5,
        batch_size=16,
        num_collocation_points=64,
        num_boundary_points=16,
        seed=42,
        device="cpu",
    )
    trainer = PINOTrainer(model=model, pde_problem=pde, config=config, device="cpu")
    history = trainer.train()

    assert isinstance(history, TrainingHistory)
    assert len(history.stage1_losses) == 1
    assert len(history.stage2_losses) == 1
    assert len(history.stage3_losses) == 1


def test_heat1d_data_loss_is_non_zero() -> None:
    model, _ = _make_model_and_pde()
    t = torch.rand(128, requires_grad=True)
    x = torch.rand(128, requires_grad=True)
    losses = model.compute_pino_loss(t, x)
    assert losses["data_loss"].item() > 0.0


def test_gradient_flows_through_pino_loss() -> None:
    model, _ = _make_model_and_pde()
    t = torch.rand(64, requires_grad=True)
    x = torch.rand(64, requires_grad=True)
    losses = model.compute_pino_loss(t, x)
    losses["total"].backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
