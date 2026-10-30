"""Fine-tune a HyPINO-generated PINN on the paper's Heat1D task.

This script is a closer same-task comparison than the zero-shot adapter:
HyPINO generates target PINN weights for mapped Heat1D, then the target PINN is
adapted with Heat1D data, boundary, and PDE residual losses for a fixed number
of Adam steps.  It is still not a full same-method reproduction because HyPINO's
hypernetwork was pretrained on a different PDE distribution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from evaluate_hypino_heat1d_adapter import build_heat1d_inputs, relative_l2, smape


def exact_heat(alpha: float, xy: torch.Tensor) -> torch.Tensor:
    x_unit = (xy[:, :1] + 1.0) / 2.0
    t_unit = (xy[:, 1:] + 1.0) / 2.0
    return torch.sin(torch.pi * x_unit) * torch.exp(-alpha * torch.pi**2 * t_unit)


def sample_interior(n_points: int, device: torch.device) -> torch.Tensor:
    return 2.0 * torch.rand(n_points, 2, device=device) - 1.0


def sample_boundary(n_points: int, device: torch.device) -> torch.Tensor:
    n_each = max(1, n_points // 3)
    y_left = 2.0 * torch.rand(n_each, 1, device=device) - 1.0
    y_right = 2.0 * torch.rand(n_each, 1, device=device) - 1.0
    x_initial = 2.0 * torch.rand(n_points - 2 * n_each, 1, device=device) - 1.0
    left = torch.cat([torch.full_like(y_left, -1.0), y_left], dim=1)
    right = torch.cat([torch.full_like(y_right, 1.0), y_right], dim=1)
    initial = torch.cat([x_initial, torch.full_like(x_initial, -1.0)], dim=1)
    return torch.cat([left, right, initial], dim=0)


def heat_residual(alpha: float, net: torch.nn.Module, xy: torch.Tensor) -> torch.Tensor:
    xy = xy.detach().clone().requires_grad_(True)
    pred = net(xy)
    grad = torch.autograd.grad(
        pred,
        xy,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
    )[0]
    u_y = grad[:, 1:2]
    u_x = grad[:, 0:1]
    grad_x = torch.autograd.grad(
        u_x,
        xy,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
    )[0]
    u_xx = grad_x[:, 0:1]
    return u_y - 2.0 * alpha * u_xx


def evaluate_grid(alpha: float, net: torch.nn.Module, grid_size: int, device: torch.device) -> dict:
    axis = np.linspace(-1.0, 1.0, grid_size)
    xx, yy = np.meshgrid(axis, axis)
    xy_np = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
    xy = torch.tensor(xy_np, device=device)
    with torch.no_grad():
        pred = net(xy).detach().cpu().numpy().reshape(grid_size, grid_size)
    x_unit = (xx + 1.0) / 2.0
    t_unit = (yy + 1.0) / 2.0
    ref = np.sin(np.pi * x_unit) * np.exp(-alpha * np.pi**2 * t_unit)
    return {
        "relative_l2_error": relative_l2(ref, pred),
        "mse": float(np.mean((pred - ref) ** 2)),
        "mae": float(np.mean(np.abs(pred - ref))),
        "max_error": float(np.max(np.abs(pred - ref))),
        "smape": smape(ref, pred),
    }


def run(
    hypino_root: Path,
    weights: Path,
    output_dir: Path,
    alpha: float,
    steps: int,
    lr: float,
    n_collocation: int,
    n_data: int,
    n_boundary: int,
    grid_size: int,
    seed: int,
    device_name: str,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    sys.path.insert(0, str(hypino_root))
    from src.data.utils import encode_pde_str, to_tensor  # noqa: PLC0415
    from src.models import HyPINO  # noqa: PLC0415
    from src.models.utils import FNN  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    old_cwd = Path.cwd()
    os.chdir(hypino_root)
    try:
        pde_string, mat_inputs, _ = build_heat1d_inputs(alpha, 224)
        pde_coeffs = encode_pde_str(pde_string)
        pde_coeffs_tensor = to_tensor([c for c in pde_coeffs.values()]).unsqueeze(0).to(device)
        mat_inputs_tensor = to_tensor(mat_inputs).unsqueeze(0).to(device)

        hypino = HyPINO.load_from_safetensors(str(weights)).eval().to(device)
        with torch.no_grad():
            _, generated_weights, generated_biases = hypino(
                pde_coeffs_tensor,
                mat_inputs_tensor,
                return_weights=True,
            )
        net = FNN.from_parameters(
            generated_weights,
            generated_biases,
            remove_batch_dim=True,
        ).to(device)

        before = evaluate_grid(alpha, net, grid_size, device)
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        history: list[dict[str, float]] = []
        start_time = time.perf_counter()

        for step in range(steps):
            optimizer.zero_grad()

            xy_col = sample_interior(n_collocation, device)
            residual_loss = torch.mean(heat_residual(alpha, net, xy_col) ** 2)

            xy_data = sample_interior(n_data, device)
            data_loss = F.mse_loss(net(xy_data), exact_heat(alpha, xy_data))

            xy_boundary = sample_boundary(n_boundary, device)
            boundary_loss = F.mse_loss(net(xy_boundary), exact_heat(alpha, xy_boundary))

            loss = residual_loss + data_loss + boundary_loss
            loss.backward()
            optimizer.step()

            if step == 0 or (step + 1) % 50 == 0 or step == steps - 1:
                history.append(
                    {
                        "step": step + 1,
                        "loss": float(loss.detach().cpu()),
                        "residual_loss": float(residual_loss.detach().cpu()),
                        "data_loss": float(data_loss.detach().cpu()),
                        "boundary_loss": float(boundary_loss.detach().cpu()),
                    }
                )

        runtime_seconds = time.perf_counter() - start_time
        after = evaluate_grid(alpha, net, grid_size, device)
        result = {
            "scope": (
                "HyPINO-generated target PINN fine-tuned on mapped Heat1D; "
                "closer same-task adaptation, not a full same-method reproduction"
            ),
            "mapped_pde": pde_string,
            "alpha": alpha,
            "seed": seed,
            "device": device_name,
            "steps": steps,
            "lr": lr,
            "n_collocation": n_collocation,
            "n_data": n_data,
            "n_boundary": n_boundary,
            "grid_size": grid_size,
            "runtime_seconds": runtime_seconds,
            "before_finetune": before,
            "after_finetune": after,
            "history": history,
            "limitations": [
                "HyPINO hypernetwork is not retrained on this paper's task distribution.",
                "Only the generated target PINN is adapted.",
                "This uses Heat1D exact data during adaptation, matching the "
                "paper's exact-solution regime.",
                "CPU runtime is reported because CUDA is unavailable on this machine.",
            ],
        }
        with (output_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    finally:
        os.chdir(old_cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypino-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-collocation", type=int, default=512)
    parser.add_argument("--n-data", type=int, default=512)
    parser.add_argument("--n-boundary", type=int, default=512)
    parser.add_argument("--grid-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    run(
        hypino_root=args.hypino_root.resolve(),
        weights=args.weights.resolve(),
        output_dir=args.output_dir.resolve(),
        alpha=args.alpha,
        steps=args.steps,
        lr=args.lr,
        n_collocation=args.n_collocation,
        n_data=args.n_data,
        n_boundary=args.n_boundary,
        grid_size=args.grid_size,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
