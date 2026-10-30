"""Aggregate external-probe result JSON files for paper reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def infer_hypino_task(data: dict) -> str:
    if "velocity" in data:
        return "AdvectionDiffusion1D"
    return "Heat1D"


def aggregate_hypino(seed_dirs: list[Path]) -> dict:
    rows = []
    for seed_dir in seed_dirs:
        data = load_json(seed_dir / "results.json")
        rows.append(
            {
                "pde": infer_hypino_task(data),
                "seed": data["seed"],
                "before_relative_l2_error": data["before_finetune"]["relative_l2_error"],
                "after_relative_l2_error": data["after_finetune"]["relative_l2_error"],
                "after_mse": data["after_finetune"]["mse"],
                "after_mae": data["after_finetune"]["mae"],
                "after_max_error": data["after_finetune"]["max_error"],
                "after_smape": data["after_finetune"]["smape"],
                "runtime_seconds": data["runtime_seconds"],
            }
        )
    pdes = sorted({row["pde"] for row in rows})
    pde_scope = pdes[0] if len(pdes) == 1 else ", ".join(pdes)
    return {
        "scope": f"HyPINO {pde_scope} target-PINN adaptation aggregated over seeds",
        "pdes": pdes,
        "n_seeds": len(rows),
        "rows": rows,
        "summary": {
            "after_relative_l2_error": summarize(
                [row["after_relative_l2_error"] for row in rows]
            ),
            "after_mse": summarize([row["after_mse"] for row in rows]),
            "after_mae": summarize([row["after_mae"] for row in rows]),
            "after_max_error": summarize([row["after_max_error"] for row in rows]),
            "after_smape": summarize([row["after_smape"] for row in rows]),
            "runtime_seconds": summarize([row["runtime_seconds"] for row in rows]),
        },
    }


def aggregate_hypino_suite(aggregate_dirs: list[Path]) -> dict:
    rows = []
    for aggregate_dir in aggregate_dirs:
        data = load_json(aggregate_dir / "results.json")
        pde = data["pdes"][0] if len(data["pdes"]) == 1 else "+".join(data["pdes"])
        rows.append(
            {
                "pde": pde,
                "n_seeds": data["n_seeds"],
                "relative_l2_mean": data["summary"]["after_relative_l2_error"]["mean"],
                "relative_l2_std": data["summary"]["after_relative_l2_error"]["std"],
                "mse_mean": data["summary"]["after_mse"]["mean"],
                "mse_std": data["summary"]["after_mse"]["std"],
                "runtime_seconds_mean": data["summary"]["runtime_seconds"]["mean"],
            }
        )
    return {
        "scope": "HyPINO target-PINN adaptation aggregated over PDE-level aggregates",
        "n_pdes": len(rows),
        "rows": rows,
        "summary": {
            "relative_l2_mean_across_pdes": summarize(
                [row["relative_l2_mean"] for row in rows]
            ),
            "mse_mean_across_pdes": summarize([row["mse_mean"] for row in rows]),
            "runtime_seconds_mean_across_pdes": summarize(
                [row["runtime_seconds_mean"] for row in rows]
            ),
        },
    }


def aggregate_pinnacle(task_dirs: list[Path]) -> dict:
    rows = []
    for task_dir in task_dirs:
        data = load_json(task_dir / "results.json")
        rows.append(
            {
                "pde": data["pde"],
                "iterations": data["iterations"],
                "relative_l2_error": data["metrics"]["relative_l2_error"],
                "mse": data["metrics"]["mse"],
                "mae": data["metrics"]["mae"],
                "max_error": data["metrics"]["max_error"],
                "runtime_seconds": data["runtime_seconds"],
            }
        )
    return {
        "scope": "PINNacle executable subset aggregated over tasks",
        "n_tasks": len(rows),
        "rows": rows,
        "summary": {
            "relative_l2_error": summarize([row["relative_l2_error"] for row in rows]),
            "mse": summarize([row["mse"] for row in rows]),
            "mae": summarize([row["mae"] for row in rows]),
            "max_error": summarize([row["max_error"] for row in rows]),
            "runtime_seconds": summarize([row["runtime_seconds"] for row in rows]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("hypino", "hypino-suite", "pinnacle"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()

    if args.mode == "hypino":
        aggregate = aggregate_hypino(args.paths)
    elif args.mode == "hypino-suite":
        aggregate = aggregate_hypino_suite(args.paths)
    else:
        aggregate = aggregate_pinnacle(args.paths)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)


if __name__ == "__main__":
    main()
