"""Run HyPINO's official evaluation suite with a compatibility loader.

The upstream ``evaluate.py`` currently forwards ``strict`` and
``ignore_size_mismatch`` into ``HyPINO.__init__`` through
``load_from_safetensors``.  This wrapper keeps the same benchmark files and
metrics, but calls the loader with only the weights path.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

BENCHMARKS = (
    "eval_data/heat.pkl",
    "eval_data/helmholtz.pkl",
    "eval_data/helmholtz_G.pkl",
    "eval_data/poisson_C.pkl",
    "eval_data/poisson_L.pkl",
    "eval_data/poisson_G.pkl",
    "eval_data/wave.pkl",
)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        100
        * np.mean(
            2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
        )
    )


def evaluate(hypino_root: Path, weights: Path, output_dir: Path, device_name: str) -> None:
    sys.path.insert(0, str(hypino_root))
    from src.data.datasets import EvalPDEDataset  # noqa: PLC0415
    from src.models import HyPINO  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)

    old_cwd = Path.cwd()
    os.chdir(hypino_root)
    try:
        model = HyPINO.load_from_safetensors(str(weights))
        model.eval().to(device)

        dataset = EvalPDEDataset(list(BENCHMARKS))
        results_path = output_dir / "results.txt"
        with results_path.open("w", encoding="utf-8") as f_out:
            f_out.write("# HyPINO official-suite evaluation\n")
            f_out.write(f"# hypino_root: {hypino_root}\n")
            f_out.write(f"# weights: {weights}\n")
            f_out.write(f"# device: {device}\n\n")

            for index, benchmark in enumerate(BENCHMARKS):
                data = dataset[index]
                pde_coeffs = data["pde_coeffs"].unsqueeze(0).to(device)
                grids = data["mat_inputs"].unsqueeze(0).to(device)
                ref = data["u_grid"].cpu().numpy().reshape(224, 224)

                x = np.linspace(-1, 1, 224)
                y = np.linspace(1, -1, 224)
                xy_np = np.stack(np.meshgrid(x, y), axis=-1)
                xy = (
                    torch.tensor(xy_np)
                    .flatten(0, 1)
                    .unsqueeze(0)
                    .to(device, dtype=torch.float32)
                )

                with torch.no_grad():
                    pred_tensor = model(pde_coeffs, grids, xy=xy)
                pred = (
                    pred_tensor.detach().cpu().numpy().reshape(224, 224)
                    * data["domain_mask"].numpy()
                )

                mse = float(np.mean((ref - pred) ** 2))
                mae = float(np.mean(np.abs(ref - pred)))
                max_error = float(np.max(np.abs(ref - pred)))
                smape_val = smape(ref, pred)

                benchmark_name = Path(benchmark).stem
                f_out.write(f"{benchmark_name}:\n")
                f_out.write(f"MSE: {mse:.4e}\n")
                f_out.write(f"MAE: {mae:.4e}\n")
                f_out.write(f"Max Error: {max_error:.4e}\n")
                f_out.write(f"SMAPE: {smape_val:.2f}%\n\n")
    finally:
        os.chdir(old_cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypino-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    evaluate(
        args.hypino_root.resolve(),
        args.weights.resolve(),
        args.output_dir.resolve(),
        args.device,
    )


if __name__ == "__main__":
    main()
