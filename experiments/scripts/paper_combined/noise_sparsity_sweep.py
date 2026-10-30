#!/usr/bin/env python3
"""Noise x data-sparsity sweep -- how far do the policy conclusions travel?

Track A1 of the post-review roadmap.  The original ``noisy_data_probe.py``
varies only the corruption level of the Stage-1 reference signal and shows
that operator pretraining keeps roughly an 8x accuracy advantage all the way
to sigma=0.2.  Reviewers (and the AC) asked the harder question: does that
still hold when reference data is also *scarce*, as it is in any real
measurement setting?

This script crosses the two axes:

  * ``--noise-levels``  relative Gaussian noise on the Stage-1 reference
  * ``--data-sizes``    number of Stage-1 reference points (sparsity)

against the same two conditions as the original probe (``with_pretrain`` vs
``pinn_only``).  It deliberately reuses ``noisy_data_probe.run_single`` so the
per-run training pipeline is byte-for-byte the one behind the published noise
numbers; only the sweep grid is new.

Outputs:
  * ``{output_dir}/results.json``   -- raw rows for all (sigma, n_data, cond, seed)
  * ``{output_dir}/aggregate.json`` -- summary keyed by sigma/n_data/condition
  * ``{output_dir}/progress.json``  -- updated per completed cell
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from noisy_data_probe import (  # noqa: E402
    DEFAULT_SEEDS,
    run_single,
)

from causal_pinn.evaluation.benchmark_runner import (  # noqa: E402
    BenchmarkConfig,
    BenchmarkRunner,
)
from causal_pinn.physics.pde import (  # noqa: E402
    AdvectionDiffusion1D,
    AllenCahn1D,
    BurgersExact1D,
    Heat1D,
    KdV1D,
    ReactionDiffusionExact1D,
)

# Only PDEs with an analytical solution can be swept here: the probe corrupts a
# synthetic reference signal, and relative-L2 needs a ground truth to score
# against.  Heat1D and AdvectionDiffusion1D qualify directly.  Burgers needed the
# Cole-Hopf closed form (``BurgersExact1D``) and is the case worth stressing,
# since it is where the smooth-early-dynamics assumption breaks.  KdV1D and
# AllenCahn1D already carry closed forms, but only in their non-default modes --
# the one-soliton solution and the stationary tanh front respectively -- so they
# are constructed here with those modes switched on.  Burgers2D is now the only
# suite member without a reference; ReactionDiffusion1D gained one via the
# Ablowitz-Zeppetella travelling wave (``ReactionDiffusionExact1D``), which is
# exact only at the wave speed 5*sqrt(rD/6).
PDE_REGISTRY: dict[str, Any] = {
    "heat1d": (Heat1D, {"alpha": 1.0, "length": 1.0, "t_max": 1.0}),
    "advdiff1d": (
        AdvectionDiffusion1D,
        {"velocity": 1.0, "alpha": 0.1, "length": 1.0, "t_max": 1.0},
    ),
    "burgers1d": (
        BurgersExact1D,
        {"viscosity": 0.01, "length": 1.0, "t_max": 1.0},
    ),
    "kdv1d": (
        KdV1D,
        {"soliton_amplitude": 1.0, "dispersion": 1.0, "length": 20.0, "t_max": 1.0},
    ),
    "allencahn1d": (
        AllenCahn1D,
        {"epsilon": 0.01, "length": 2.0, "t_max": 1.0, "front_center": 1.0},
    ),
    "reaction_diffusion1d": (
        ReactionDiffusionExact1D,
        {"diffusion": 0.01, "reaction_rate": 1.0, "length": 1.0, "t_max": 1.0},
    ),
}

# Stage-1 reference-set sizes.  1024 is the original probe's default
# (BATCH_SIZE * 4 = 256 * 4); the smaller values emulate sparse measurements.
DEFAULT_DATA_SIZES = [1024, 256, 64, 16]
DEFAULT_NOISE_LEVELS = [0.0, 0.05, 0.2]

CONDITIONS = ["with_pretrain", "pinn_only"]


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise by (sigma, n_data, condition)."""
    out: dict[str, Any] = {}
    keys = sorted(
        {(r["noise_sigma"], r["n_data"], r["condition"]) for r in results}
    )
    for sigma, n_data, cond in keys:
        sub = [
            r
            for r in results
            if r["noise_sigma"] == sigma
            and r["n_data"] == n_data
            and r["condition"] == cond
            and r["success"]
        ]
        rl2 = [r["metrics"].get("relative_l2_error", float("nan")) for r in sub]
        pres = [r["metrics"].get("pde_residual_norm", float("nan")) for r in sub]
        out[f"sigma={sigma}/n={n_data}/{cond}"] = {
            "n": len(sub),
            "noise_sigma": sigma,
            "n_data": n_data,
            "condition": cond,
            "relative_l2_error": {
                "mean": float(np.nanmean(rl2)) if rl2 else float("nan"),
                "std": float(np.nanstd(rl2)) if rl2 else float("nan"),
            },
            "pde_residual_norm": {
                "mean": float(np.nanmean(pres)) if pres else float("nan"),
                "std": float(np.nanstd(pres)) if pres else float("nan"),
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/paper_combined/noise_sparsity_sweep",
    )
    parser.add_argument(
        "--noise-levels", type=float, nargs="+", default=DEFAULT_NOISE_LEVELS
    )
    parser.add_argument(
        "--data-sizes", type=int, nargs="+", default=DEFAULT_DATA_SIZES
    )
    parser.add_argument(
        "--pde", type=str, default="heat1d", choices=sorted(PDE_REGISTRY),
    )
    parser.add_argument(
        "--pde-kwargs",
        type=str,
        default=None,
        help=(
            "JSON dict merged over the registry defaults, e.g. "
            "'{\"diffusion\": 0.02, \"reaction_rate\": 2.0}'. Used to sweep a "
            "PDE's own parameters while holding the grid fixed."
        ),
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = DEFAULT_SEEDS[: args.num_seeds]

    pde_cls, pde_kwargs = PDE_REGISTRY[args.pde]
    pde_kwargs = dict(pde_kwargs)
    if args.pde_kwargs:
        pde_kwargs.update(json.loads(args.pde_kwargs))
        print(f"PDE kwargs override -> {pde_kwargs}")
    pde = pde_cls(**pde_kwargs)
    runner = BenchmarkRunner(
        BenchmarkConfig(
            n_collocation=2000,
            n_boundary=200,
            n_initial=200,
            include_physics_consistency=False,
            device=args.device,
        )
    )

    total = (
        len(args.noise_levels)
        * len(args.data_sizes)
        * len(CONDITIONS)
        * len(seeds)
    )
    print(
        f"Noise x sparsity sweep: {len(args.noise_levels)} sigmas x "
        f"{len(args.data_sizes)} data sizes x {len(CONDITIONS)} conds x "
        f"{len(seeds)} seeds = {total} runs"
    )

    progress: dict[str, Any] = {"completed_cells": [], "total_cells": total // len(seeds)}
    progress_path = output_dir / "progress.json"
    all_results: list[dict[str, Any]] = []
    count = 0
    t_start = time.time()

    for sigma in args.noise_levels:
        for n_data in args.data_sizes:
            for cond in CONDITIONS:
                for seed in seeds:
                    count += 1
                    tag = (
                        f"[{count}/{total}] sigma={sigma}/n={n_data}/"
                        f"{cond}/seed={seed}"
                    )
                    print(f"  {tag}...", end=" ", flush=True)
                    try:
                        r = run_single(
                            args.pde,
                            pde,
                            sigma,
                            cond,
                            seed,
                            args.device,
                            runner,
                            fixed_dataset_size=n_data,
                        )
                        rl2 = r["metrics"].get("relative_l2_error", float("nan"))
                        print(f"rel_l2={rl2:.4e} time={r['training_time']:.1f}s")
                    except Exception as exc:  # noqa: BLE001
                        print(f"FAILED: {exc}")
                        r = {
                            "pde": args.pde,
                            "noise_sigma": sigma,
                            "condition": cond,
                            "seed": seed,
                            "training_time": 0.0,
                            "metrics": {},
                            "success": False,
                            "error": str(exc),
                        }
                    r["n_data"] = n_data
                    all_results.append(r)

                progress["completed_cells"].append(
                    {"sigma": sigma, "n_data": n_data, "condition": cond}
                )
                progress_path.write_text(
                    json.dumps(progress, indent=2), encoding="utf-8"
                )

    summary = aggregate(all_results)
    meta = {
        "experiment": "noise_sparsity_sweep",
        "pde": args.pde,
        "noise_levels": args.noise_levels,
        "data_sizes": args.data_sizes,
        "pde_kwargs": pde_kwargs,
        "conditions": CONDITIONS,
        "seeds": seeds,
        "wall_time_seconds": time.time() - t_start,
        "summary": summary,
    }
    (output_dir / "results.json").write_text(
        json.dumps({**meta, "raw_results": all_results}, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "aggregate.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nWrote {output_dir}/results.json and aggregate.json")


if __name__ == "__main__":
    main()
