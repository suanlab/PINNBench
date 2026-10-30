"""Run a small executable PINNacle benchmark subset.

The goal is artifact-level benchmark alignment, not reproducing the full
PINNacle paper.  This script runs a configurable PINNacle PDE for a small
number of iterations and writes metrics in this repository's result format.
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

PDE_REGISTRY = {
    "burgers1d": ("src.pde.burgers", "Burgers1D"),
    "heat2d_multiscale": ("src.pde.heat", "Heat2D_Multiscale"),
    "helmholtz2d": ("src.pde.helmholtz", "Helmholtz2D"),
    "poisson1d": ("src.pde.poisson", "Poisson1D"),
    "wave1d": ("src.pde.wave", "Wave1D"),
}


def import_symbol(module_name: str, symbol_name: str):
    module = __import__(module_name, fromlist=[symbol_name])
    return getattr(module, symbol_name)


def run(
    pinnacle_root: Path,
    output_dir: Path,
    pde_name: str,
    iterations: int,
    hidden_width: int,
    hidden_depth: int,
    lr: float,
    seed: int,
    device_name: str,
) -> None:
    os.environ["DDEBACKEND"] = "pytorch"
    sys.path.insert(0, str(pinnacle_root))

    old_cwd = Path.cwd()
    os.chdir(pinnacle_root)
    try:
        import deepxde as dde  # noqa: PLC0415

        module_name, class_name = PDE_REGISTRY[pde_name]
        pde_cls = import_symbol(module_name, class_name)

        torch.manual_seed(seed)
        np.random.seed(seed)
        dde.config.set_random_seed(seed)
        if device_name == "cpu":
            torch.set_default_device("cpu")

        pde = pde_cls()
        if hasattr(pde, "training_points"):
            try:
                pde.training_points(domain=512, boundary=128, initial=128, test=512)
            except TypeError:
                pde.training_points(domain=512, boundary=128, test=512)

        net = dde.nn.FNN(
            [pde.input_dim] + [hidden_width] * hidden_depth + [pde.output_dim],
            "tanh",
            "Glorot normal",
        )
        net = net.float()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        model = pde.create_model(net)
        model.compile(optimizer, loss_weights=np.ones(pde.num_loss))

        output_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        loss_history, train_state = model.train(
            iterations=iterations,
            display_every=max(1, iterations // 2),
            model_save_path=str(output_dir / "model"),
            save_model=False,
        )
        runtime_seconds = time.perf_counter() - start

        if pde.ref_data is not None:
            ref = pde.ref_data[~np.isnan(pde.ref_data).any(axis=1)]
            x_test = ref[:, : pde.input_dim]
            y_test = ref[:, pde.input_dim :]
        elif pde.ref_sol is not None:
            geom = getattr(pde, "geomtime", None) or pde.geom
            x_test = geom.uniform_points(2048, boundary=False)
            y_test = pde.ref_sol(x_test)
        else:
            raise RuntimeError(f"{pde_name} does not expose reference data/solution")

        with torch.no_grad():
            y_pred = model.predict(x_test)
        mse = float(np.mean((y_pred - y_test) ** 2))
        mae = float(np.mean(np.abs(y_pred - y_test)))
        max_error = float(np.max(np.abs(y_pred - y_test)))
        rel_l2 = float(np.linalg.norm(y_pred - y_test) / (np.linalg.norm(y_test) + 1e-12))

        result = {
            "scope": "PINNacle executable subset smoke run; not full benchmark reproduction",
            "pinnacle_root": str(pinnacle_root),
            "pde": pde_name,
            "pde_class": class_name,
            "iterations": iterations,
            "hidden_width": hidden_width,
            "hidden_depth": hidden_depth,
            "lr": lr,
            "seed": seed,
            "device": device_name,
            "runtime_seconds": runtime_seconds,
            "metrics": {
                "relative_l2_error": rel_l2,
                "mse": mse,
                "mae": mae,
                "max_error": max_error,
            },
            "train_state": {
                "best_step": int(train_state.best_step),
                "best_loss_train": np.asarray(train_state.best_loss_train).tolist(),
                "best_loss_test": np.asarray(train_state.best_loss_test).tolist(),
            },
            "loss_steps": np.asarray(loss_history.steps).tolist(),
            "limitations": [
                "Runs one small PINNacle task rather than the full 20-task benchmark.",
                "Uses reduced collocation/test points and a small network for CPU feasibility.",
                "Used for executable benchmark alignment, not for ranking against "
                "the paper's methods.",
            ],
        }
        with (output_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    finally:
        os.chdir(old_cwd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pinnacle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pde", choices=sorted(PDE_REGISTRY), default="burgers1d")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    run(
        pinnacle_root=args.pinnacle_root.resolve(),
        output_dir=args.output_dir.resolve(),
        pde_name=args.pde,
        iterations=args.iterations,
        hidden_width=args.hidden_width,
        hidden_depth=args.hidden_depth,
        lr=args.lr,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
