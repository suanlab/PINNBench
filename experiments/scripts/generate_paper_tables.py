#!/usr/bin/env python3
"""Generate LaTeX table fragments from experiment result JSONs.

This script generates table fragments (causal ablation, lambda sweep,
hybrid comparison, OOD sections) from result JSONs in
``experiments/results/``. The full claim-to-artifact mapping is published
alongside the benchmark dataset at
https://huggingface.co/datasets/suanlab/PINNBench (see ``MANIFEST.md``
in that dataset). Paper drafts are not shipped with this repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
OUTPUT = ROOT / "experiments" / "analysis" / "paper_tables.tex"


def load(name: str) -> dict:
    return json.loads((RESULTS / name / "results.json").read_text())


def fmt(mean: float, std: float, prec: int = 3) -> str:
    return f"${mean:.{prec}f} \\pm {std:.{prec}f}$"


def extract_paired(data: dict, key: str, field: str, metric: str, groups: list[str]) -> dict:
    rows = data.get(key, data.get("runs", []))
    out: dict[str, list[float]] = {g: [] for g in groups}
    for row in rows:
        g = row.get(field)
        if g in out:
            m = row.get("evaluation_metrics", row.get("metrics", {}))
            out[g].append(m.get(metric, float("nan")))
    return {g: (float(np.mean(v)), float(np.std(v, ddof=1)), len(v)) for g, v in out.items()}


def table_causal_ablation() -> str:
    lines = []
    specs = [
        ("Heat1D", "rq1", "relative_l2_error", "runs", "condition"),
        ("Burgers1D", "rq1b_10seed", "pde_residual_norm", "runs", "condition"),
        ("AdvDiff1D", "rq1c", "relative_l2_error", "runs", "condition"),
    ]
    metric_labels = {"relative_l2_error": "Rel.\\ $L_2$", "pde_residual_norm": "PDE Residual"}
    for pde_name, exp_name, metric, container, field in specs:
        data = load(exp_name)
        groups = ["with_causal", "without_causal"]
        stats = extract_paired(data, container, field, metric, groups)
        wc = stats["with_causal"]
        wo = stats["without_causal"]
        diff = np.array([
            r["evaluation_metrics"][metric]
            for r in data[container] if r[field] == "with_causal"
        ]) - np.array([
            r["evaluation_metrics"][metric]
            for r in data[container] if r[field] == "without_causal"
        ])
        d = float(np.mean(diff) / np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        lines.append(
            f"        {pde_name} & {metric_labels[metric]} & {fmt(wc[0], wc[1])} & "
            f"{fmt(wo[0], wo[1])} & ${d:.2f}$ \\\\"
        )
    return "\n".join(lines)


def table_lambda_sweep() -> str:
    data = load("rq5_wave2d_lambda_sweep")
    agg = data["aggregated_results"]
    lines = []
    for lam in ["0.0", "0.01", "0.1", "0.5", "1.0"]:
        r = agg[lam]
        rel = fmt(r["relative_l2_mean"], r["relative_l2_std"])
        pde = fmt(r["pde_residual_mean"], r["pde_residual_std"])
        mse = fmt(r["mse_mean"], r["mse_std"])
        lines.append(f"        ${lam}$ & {rel} & {pde} & {mse} \\\\")
    return "\n".join(lines)


def table_hybrid_comparison() -> str:
    lines = []
    experiments = [
        ("Heat1D", "rq3_10seed", "raw_results", "model_name"),
        ("Wave2D", "rq4_10seed", "raw_results", "model_name"),
    ]
    for pde_name, exp_name, container, field in experiments:
        data = load(exp_name)
        for model in ["pinn_only", "hybrid_full"]:
            rows = [
                r for r in data[container]
                if r[field] == model and r.get("success", True)
            ]
            rel_vals = [r["metrics"]["relative_l2_error"] for r in rows]
            pde_vals = [r["metrics"]["pde_residual_norm"] for r in rows]
            rel_m, rel_s = float(np.mean(rel_vals)), float(np.std(rel_vals, ddof=1))
            pde_m, pde_s = float(np.mean(pde_vals)), float(np.std(pde_vals, ddof=1))
            label = "PINN-only" if model == "pinn_only" else "Hybrid (full)"
            lines.append(
                f"        & {label} & {fmt(rel_m, rel_s)} & {fmt(pde_m, pde_s)} \\\\"
            )
        if pde_name != "Wave2D":
            lines.append("        \\midrule")
    return "\n".join(lines)


def table_ood() -> str:
    data = load("rq6_ood")
    agg = data["aggregated_results"]
    lines = []
    for model in ["pinn_only", "hybrid_full"]:
        m = agg[model]
        id_l2 = m["in_domain"]["relative_l2_error"]["mean"]
        ood_l2 = m["ood"]["relative_l2_error"]["mean"]
        id_pde = m["in_domain"]["pde_residual_norm"]["mean"]
        ood_pde = m["ood"]["pde_residual_norm"]["mean"]
        label = "PINN-only" if model == "pinn_only" else "Hybrid (full)"
        lines.append(
            f"        {label} & & ${id_l2:.3f}$ & ${ood_l2:.3f}$ & "
            f"${id_pde:.3f}$ & ${ood_pde:.3f}$ \\\\"
        )
    # degradation gaps
    p = agg["pinn_only"]
    h = agg["hybrid_full"]
    gap_l2_p = p["ood"]["relative_l2_error"]["mean"] - p["in_domain"]["relative_l2_error"]["mean"]
    gap_l2_h = h["ood"]["relative_l2_error"]["mean"] - h["in_domain"]["relative_l2_error"]["mean"]
    gap_pde_p = p["ood"]["pde_residual_norm"]["mean"] - p["in_domain"]["pde_residual_norm"]["mean"]
    gap_pde_h = h["ood"]["pde_residual_norm"]["mean"] - h["in_domain"]["pde_residual_norm"]["mean"]
    lines.append("        \\midrule")
    lines.append(
        f"        \\multicolumn{{2}}{{l}}{{Degradation gap}} & "
        f"$+{gap_l2_p:.3f}$ / $+{gap_l2_h:.3f}$ & & "
        f"$+{gap_pde_p:.3f}$ / $+{gap_pde_h:.3f}$ & \\\\"
    )
    return "\n".join(lines)


def main() -> None:
    sections = {
        "CAUSAL_ABLATION_ROWS": table_causal_ablation(),
        "LAMBDA_SWEEP_ROWS": table_lambda_sweep(),
        "HYBRID_COMPARISON_ROWS": table_hybrid_comparison(),
        "OOD_ROWS": table_ood(),
    }
    output_lines = [
        "% Auto-generated from experiments/scripts/generate_paper_tables.py",
        "% Do NOT edit manually — rerun the script instead.",
        "",
    ]
    for tag, content in sections.items():
        output_lines.append(f"% === {tag} ===")
        output_lines.append(content)
        output_lines.append("")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"Generated: {OUTPUT}")

    for tag, content in sections.items():
        print(f"\n--- {tag} ---")
        print(content)


if __name__ == "__main__":
    main()
