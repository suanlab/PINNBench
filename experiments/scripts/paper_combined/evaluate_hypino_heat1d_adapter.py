"""Evaluate HyPINO zero-shot on a mapped Heat1D task.

This is a compatibility/scope probe, not a fair head-to-head training
baseline.  Heat1D on ``(x, t) in [0, 1]^2`` is mapped to HyPINO's
``(X, Y) in [-1, 1]^2`` coordinates with ``X=2x-1`` and ``Y=2t-1``.
The transformed PDE is ``uy - 2*alpha*uxx = 0``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def heat_solution(alpha: float, x_unit: np.ndarray, t_unit: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x_unit) * np.exp(-alpha * np.pi**2 * t_unit)


def relative_l2(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - reference) / (np.linalg.norm(reference) + 1e-12))


def smape(reference: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        100
        * np.mean(
            2.0
            * np.abs(prediction - reference)
            / (np.abs(reference) + np.abs(prediction) + 1e-8)
        )
    )


def build_heat1d_inputs(alpha: float, grid_size: int) -> tuple[str, np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, grid_size)
    xx, yy = np.meshgrid(axis, axis)
    x_unit = (xx + 1.0) / 2.0
    t_unit = (yy + 1.0) / 2.0
    reference = heat_solution(alpha, x_unit, t_unit).astype(np.float32)

    at_left = np.isclose(xx, -1.0)
    at_right = np.isclose(xx, 1.0)
    at_initial = np.isclose(yy, -1.0)
    dirichlet_mask = (at_left | at_right | at_initial).astype(np.float32)
    dirichlet_values = np.zeros_like(reference, dtype=np.float32)
    dirichlet_values[at_initial] = np.sin(np.pi * x_unit[at_initial]).astype(np.float32)

    neumann_mask = np.zeros_like(reference, dtype=np.float32)
    neumann_values = np.zeros_like(reference, dtype=np.float32)
    source = np.zeros_like(reference, dtype=np.float32)

    mat_inputs = np.stack(
        [dirichlet_mask, neumann_mask, dirichlet_values, neumann_values, source],
        axis=0,
    )
    pde_string = f"1.0*uy - {2.0 * alpha:.8f}*uxx"
    return pde_string, mat_inputs, reference


def evaluate(
    hypino_root: Path,
    weights: Path,
    output_dir: Path,
    alpha: float,
    device_name: str,
    grid_size: int = 224,
) -> None:
    sys.path.insert(0, str(hypino_root))
    from src.data.utils import encode_pde_str, to_tensor  # noqa: PLC0415
    from src.models import HyPINO  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    old_cwd = Path.cwd()
    os.chdir(hypino_root)
    try:
        device = torch.device(device_name)
        model = HyPINO.load_from_safetensors(str(weights))
        model.eval().to(device)

        pde_string, mat_inputs, reference = build_heat1d_inputs(alpha, grid_size)
        pde_coeffs = encode_pde_str(pde_string)
        pde_coeffs_tensor = to_tensor([c for c in pde_coeffs.values()]).unsqueeze(0).to(device)
        mat_inputs_tensor = to_tensor(mat_inputs).unsqueeze(0).to(device)

        axis = np.linspace(-1.0, 1.0, grid_size)
        xx, yy = np.meshgrid(axis, axis)
        xy = (
            torch.tensor(np.stack([xx, yy], axis=-1), dtype=torch.float32)
            .flatten(0, 1)
            .unsqueeze(0)
            .to(device)
        )

        with torch.no_grad():
            pred_tensor = model(pde_coeffs_tensor, mat_inputs_tensor, xy=xy)
        prediction = pred_tensor.detach().cpu().numpy().reshape(grid_size, grid_size)

        result = {
            "scope": "HyPINO zero-shot Heat1D adapter smoke test; not a fair head-to-head",
            "mapped_pde": pde_string,
            "alpha": alpha,
            "grid_size": grid_size,
            "device": device_name,
            "metrics": {
                "relative_l2_error": relative_l2(reference, prediction),
                "mse": float(np.mean((prediction - reference) ** 2)),
                "mae": float(np.mean(np.abs(prediction - reference))),
                "max_error": float(np.max(np.abs(prediction - reference))),
                "smape": smape(reference, prediction),
            },
            "limitations": [
                "HyPINO is not retrained or fine-tuned on this paper's task distribution.",
                "Only Heat1D maps cleanly to HyPINO's two-variable operator vocabulary.",
                "Boundary/source encoding follows HyPINO's grid protocol but is not "
                "guaranteed to match its training distribution.",
                "The result should be used as a compatibility probe, not as a "
                "same-budget baseline.",
            ],
        }

        with (output_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        np.savez_compressed(
            output_dir / "fields.npz",
            reference=reference,
            prediction=prediction,
            mat_inputs=mat_inputs,
        )
    finally:
        os.chdir(old_cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypino-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    evaluate(
        hypino_root=args.hypino_root.resolve(),
        weights=args.weights.resolve(),
        output_dir=args.output_dir.resolve(),
        alpha=args.alpha,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
