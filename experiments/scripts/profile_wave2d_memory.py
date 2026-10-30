#!/usr/bin/env python3
"""Wave 0a: Profile Wave2D memory usage on CUDA.

This script measures peak CUDA memory for Wave2D physics-loss backpropagation
at multiple collocation point counts and emits machine-readable and human-
readable reports for GO/NO-GO decisions.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from causal_pinn.models.pinn.base import PINNBase
from causal_pinn.physics.losses.physics_loss import PhysicsLoss
from causal_pinn.physics.pde.wave_2d import Wave2D
from causal_pinn.utils.random import set_seed


@dataclass
class ProfileResult:
    collocation_points: int
    iterations: int
    success: bool
    oom: bool
    peak_allocated_gb: float | None
    peak_reserved_gb: float | None
    mean_step_time_sec: float | None
    std_step_time_sec: float | None
    error_message: str | None


def make_points(num_points: int, device: torch.device) -> tuple[Tensor, Tensor]:
    t = torch.rand(num_points, device=device)
    x = torch.rand(num_points, 2, device=device)
    t = t.requires_grad_(True)
    x = x.requires_grad_(True)
    return t, x


def run_single_profile(
    num_points: int,
    iterations: int,
    warmup: int,
    device: torch.device,
) -> ProfileResult:
    set_seed(42)
    model = PINNBase(input_dim=3, output_dim=1, hidden_dims=[64, 64, 64]).to(device)
    pde = Wave2D(wave_speed=1.0, length_x=1.0, length_y=1.0, t_max=1.0)
    physics_loss = PhysicsLoss(pde)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    step_times: list[float] = []

    try:
        for _ in range(warmup):
            t, x = make_points(num_points, device)
            optimizer.zero_grad(set_to_none=True)
            loss = physics_loss(model, t, x)["physics_loss"]
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)

        for _ in range(iterations):
            t, x = make_points(num_points, device)
            optimizer.zero_grad(set_to_none=True)
            start = time.perf_counter()
            loss = physics_loss(model, t, x)["physics_loss"]
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            step_times.append(time.perf_counter() - start)

        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)

        mean_time = float(sum(step_times) / len(step_times))
        variance = sum((x - mean_time) ** 2 for x in step_times) / len(step_times)
        std_time = float(variance**0.5)

        return ProfileResult(
            collocation_points=num_points,
            iterations=iterations,
            success=True,
            oom=False,
            peak_allocated_gb=float(peak_allocated),
            peak_reserved_gb=float(peak_reserved),
            mean_step_time_sec=mean_time,
            std_step_time_sec=std_time,
            error_message=None,
        )
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return ProfileResult(
            collocation_points=num_points,
            iterations=iterations,
            success=False,
            oom=True,
            peak_allocated_gb=None,
            peak_reserved_gb=None,
            mean_step_time_sec=None,
            std_step_time_sec=None,
            error_message=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return ProfileResult(
            collocation_points=num_points,
            iterations=iterations,
            success=False,
            oom=False,
            peak_allocated_gb=None,
            peak_reserved_gb=None,
            mean_step_time_sec=None,
            std_step_time_sec=None,
            error_message=str(exc),
        )


def decide_go_no_go(results: list[ProfileResult]) -> tuple[str, str]:
    result_5k = next((r for r in results if r.collocation_points == 5000), None)
    if result_5k is None:
        return "NO-GO", "5000-point profile missing"

    if not result_5k.success or result_5k.oom:
        return "NO-GO", "OOM or failure at 5000 collocation points"

    if result_5k.peak_allocated_gb is None:
        return "NO-GO", "Missing peak memory measurement at 5000 collocation points"

    if result_5k.peak_allocated_gb >= 11.0:
        return "NO-GO", f"Peak allocated at 5000 is {result_5k.peak_allocated_gb:.2f} GB (>= 11 GB)"

    return (
        "GO",
        f"5000-point run stable with peak allocated {result_5k.peak_allocated_gb:.2f} GB (< 11 GB)",
    )


def write_markdown_report(
    output_md: Path,
    device_name: str,
    results: list[ProfileResult],
    decision: str,
    rationale: str,
) -> None:
    lines: list[str] = []
    lines.append("# Wave2D Memory Profile (Wave 0a)")
    lines.append("")
    lines.append(f"- Device: `{device_name}`")
    lines.append("- Rule: GO if 5000 collocation stable and peak_allocated < 11GB")
    lines.append(f"- Decision: **{decision}**")
    lines.append(f"- Rationale: {rationale}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append(
        "| Collocation | Success | OOM | Peak Allocated (GB) | "
        "Peak Reserved (GB) | Mean Step (s) | Std Step (s) |"
    )
    lines.append("|---:|:---:|:---:|---:|---:|---:|---:|")
    for r in results:
        peak_alloc = "N/A" if r.peak_allocated_gb is None else f"{r.peak_allocated_gb:.3f}"
        peak_res = "N/A" if r.peak_reserved_gb is None else f"{r.peak_reserved_gb:.3f}"
        mean_step = "N/A" if r.mean_step_time_sec is None else f"{r.mean_step_time_sec:.4f}"
        std_step = "N/A" if r.std_step_time_sec is None else f"{r.std_step_time_sec:.4f}"
        lines.append(
            "| "
            f"{r.collocation_points} | {r.success} | {r.oom} | "
            f"{peak_alloc} | {peak_res} | {mean_step} | {std_step} |"
        )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Wave2D CUDA memory usage")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/wave0",
        help="Directory to store Wave 0 memory profile artifacts",
    )
    parser.add_argument(
        "--collocation-points",
        type=int,
        nargs="+",
        default=[2000, 5000, 10000],
        help="Collocation point settings to profile",
    )
    parser.add_argument("--iterations", type=int, default=3, help="Measured iterations per setting")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per setting")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Wave 0a memory profiling, but CUDA is unavailable."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)

    all_results: list[ProfileResult] = []
    for num_points in args.collocation_points:
        print(f"Profiling Wave2D at collocation={num_points} ...", flush=True)
        result = run_single_profile(
            num_points=num_points,
            iterations=args.iterations,
            warmup=args.warmup,
            device=device,
        )
        all_results.append(result)
        print(
            "  "
            f"success={result.success}, oom={result.oom}, "
            f"peak_allocated={result.peak_allocated_gb}",
            flush=True,
        )

    decision, rationale = decide_go_no_go(all_results)

    report_json = {
        "experiment": "wave0_wave2d_memory_profile",
        "device": device_name,
        "collocation_points": args.collocation_points,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "decision": decision,
        "rationale": rationale,
        "results": [asdict(r) for r in all_results],
    }

    output_json = output_dir / "memory_profile_wave2d.json"
    output_md = output_dir / "memory_profile_wave2d.md"
    output_json.write_text(json.dumps(report_json, indent=2), encoding="utf-8")
    write_markdown_report(output_md, device_name, all_results, decision, rationale)

    print(f"Saved JSON: {output_json}")
    print(f"Saved report: {output_md}")
    print(f"Decision: {decision} ({rationale})")


if __name__ == "__main__":
    main()
