#!/usr/bin/env python3

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "experiments" / "results"
OUTPUT_MD = ROOT / "experiments" / "analysis" / "statistical_tests.md"
OUTPUT_JSON = ROOT / "experiments" / "analysis" / "statistical_tests.json"


@dataclass
class AnalysisSpec:
    label: str
    file_path: Path
    group_a: str
    group_b: str
    metric_key: str
    container_key: str
    group_field: str
    metric_field: str
    interpretation: str
    lower_is_better: bool = True


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_values(spec: AnalysisSpec) -> tuple[list[float], list[float]]:
    payload = read_json(spec.file_path)
    rows = payload[spec.container_key]

    values_a: dict[int, float] = {}
    values_b: dict[int, float] = {}

    for row in rows:
        group_name = row[spec.group_field]
        seed = int(row["seed"])
        metrics = row[spec.metric_field]
        value = float(metrics[spec.metric_key])
        if group_name == spec.group_a:
            values_a[seed] = value
        elif group_name == spec.group_b:
            values_b[seed] = value

    common_seeds = sorted(set(values_a) & set(values_b))
    return [values_a[s] for s in common_seeds], [values_b[s] for s in common_seeds]


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.zeros(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and np.isclose(values[order[j + 1]], values[order[i]]):
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def wilcoxon_exact(diff: np.ndarray) -> tuple[float, float, float, int]:
    nonzero = diff[~np.isclose(diff, 0.0)]
    if len(nonzero) == 0:
        return 0.0, 1.0, 0.0, 0

    ranks = average_ranks(np.abs(nonzero))
    positive_mask = nonzero > 0
    w_plus = float(np.sum(ranks[positive_mask]))
    total_rank = float(np.sum(ranks))
    w_minus = total_rank - w_plus
    observed = min(w_plus, w_minus)

    count = 0
    extreme = 0
    n = len(ranks)
    for bits in range(1 << n):
        signed_sum = 0.0
        for idx, rank in enumerate(ranks):
            if bits & (1 << idx):
                signed_sum += rank
        candidate = min(signed_sum, total_rank - signed_sum)
        count += 1
        if candidate <= observed + 1e-12:
            extreme += 1

    p_value = extreme / count
    return observed, p_value, w_plus, n


def bootstrap_mean_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple[float, float]:
    if len(diff) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(diff, size=(n_boot, len(diff)), replace=True)
    means = np.mean(samples, axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_cohens_d(diff: np.ndarray) -> float:
    if len(diff) < 2 or np.isclose(np.std(diff, ddof=1), 0.0):
        return 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))


def summarize(spec: AnalysisSpec) -> dict:
    values_a, values_b = extract_values(spec)
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    diff = a - b

    observed, p_value, w_plus, n_pairs = wilcoxon_exact(diff)
    ci_low, ci_high = bootstrap_mean_ci(diff)
    effect_size = paired_cohens_d(diff)

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    std_a = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    std_b = float(np.std(b, ddof=1)) if len(b) > 1 else 0.0

    if spec.lower_is_better:
        better = spec.group_a if mean_a < mean_b else spec.group_b
    else:
        better = spec.group_a if mean_a > mean_b else spec.group_b

    return {
        "label": spec.label,
        "file": str(spec.file_path.relative_to(ROOT)),
        "metric": spec.metric_key,
        "group_a": spec.group_a,
        "group_b": spec.group_b,
        "group_a_mean": mean_a,
        "group_a_std": std_a,
        "group_b_mean": mean_b,
        "group_b_std": std_b,
        "mean_difference_a_minus_b": float(np.mean(diff)),
        "bootstrap_95ci_a_minus_b": [ci_low, ci_high],
        "cohens_d_paired": effect_size,
        "wilcoxon_w": observed,
        "wilcoxon_w_plus": w_plus,
        "wilcoxon_p_two_sided": p_value,
        "n_pairs": n_pairs,
        "better_group": better,
        "interpretation": spec.interpretation,
    }


def apply_bh_fdr(results: list[dict], alpha: float = 0.05) -> None:
    """Apply the Benjamini-Hochberg step-up procedure to all results in-place.

    The adjusted p-value is the running minimum of p * m / rank taken from the
    largest p downwards, so it is monotone in p and tied p-values share one
    adjusted value; a test is significant when its adjusted p is <= alpha.
    (Comparing each p only to its own rank threshold is not step-up: it made
    the verdict on tied p-values depend on their sort order.)
    """
    m = len(results)
    if m == 0:
        return
    indexed = sorted(results, key=lambda row: row["wilcoxon_p_two_sided"])
    running = 1.0
    for rank_1 in range(m, 0, -1):
        row = indexed[rank_1 - 1]
        running = min(running, row["wilcoxon_p_two_sided"] * m / rank_1)
        row["bh_fdr_threshold"] = alpha * rank_1 / m
        row["bh_fdr_p_adjusted"] = running
        row["bh_fdr_significant"] = running <= alpha


def markdown_table(results: list[dict]) -> str:
    lines = []
    lines.append(
        "| Comparison | Metric | Group A mean+-std | Group B mean+-std | "
        "Mean diff (A-B) | 95% CI | Wilcoxon p | BH-FDR p_adj | Cohen's d | Better |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in results:
        ci_low, ci_high = row["bootstrap_95ci_a_minus_b"]
        p_adj = row.get("bh_fdr_p_adjusted", float("nan"))
        sig_mark = "" if row.get("bh_fdr_significant", True) else " (ns)"
        lines.append(
            "| "
            f"{row['label']} | {row['metric']} | "
            f"{row['group_a_mean']:.4f} +- {row['group_a_std']:.4f} | "
            f"{row['group_b_mean']:.4f} +- {row['group_b_std']:.4f} | "
            f"{row['mean_difference_a_minus_b']:.4f} | "
            f"[{ci_low:.4f}, {ci_high:.4f}] | "
            f"{row['wilcoxon_p_two_sided']:.4f} | "
            f"{p_adj:.4f}{sig_mark} | "
            f"{row['cohens_d_paired']:.4f} | "
            f"{row['better_group']} |"
        )
    return "\n".join(lines)


def write_markdown(results: list[dict]) -> None:
    lines: list[str] = []
    lines.append("# Statistical Tests")
    lines.append("")
    lines.append(
        "- Tests: exact Wilcoxon signed-rank, bootstrap 95% CI for paired mean difference, "
        "paired Cohen's d"
    )
    lines.append(
        "- Multiple-comparison correction: Benjamini-Hochberg FDR (q=0.05)"
    )
    lines.append("- Seeds: paired by identical seed across compared conditions/models")
    lines.append("- Convention: `mean_difference_a_minus_b = group_a - group_b`")
    lines.append("")
    lines.append("## Summary Table")
    lines.append("")
    lines.append(markdown_table(results))
    lines.append("")
    n_sig = sum(1 for r in results if r.get("bh_fdr_significant", True))
    n_total = len(results)
    lines.append(
        f"BH-FDR correction: {n_sig}/{n_total} tests remain significant at q=0.05."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for row in results:
        sig_note = "" if row.get("bh_fdr_significant", True) else " **[not significant after FDR]**"
        lines.append(
            f"- **{row['label']}**: {row['interpretation']} "
            f"(p={row['wilcoxon_p_two_sided']:.4f}, "
            f"p_adj={row.get('bh_fdr_p_adjusted', float('nan')):.4f}, "
            f"d={row['cohens_d_paired']:.4f}){sig_note}"
        )

    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    specs = [
        AnalysisSpec(
            label="RQ1 Heat1D causal ablation",
            file_path=RESULTS_DIR / "rq1" / "results.json",
            group_a="with_causal",
            group_b="without_causal",
            metric_key="relative_l2_error",
            container_key="runs",
            group_field="condition",
            metric_field="evaluation_metrics",
            interpretation="Heat1D causal loss should remain a null or near-null result.",
        ),
        AnalysisSpec(
            label="RQ1b Burgers1D causal ablation",
            file_path=RESULTS_DIR / "rq1b_10seed" / "results.json",
            group_a="with_causal",
            group_b="without_causal",
            metric_key="pde_residual_norm",
            container_key="runs",
            group_field="condition",
            metric_field="evaluation_metrics",
            interpretation=(
                "Burgers1D causal loss should remain a null or slightly negative result."
            ),
        ),
        AnalysisSpec(
            label="RQ1c AdvDiff1D causal ablation",
            file_path=RESULTS_DIR / "rq1c" / "results.json",
            group_a="with_causal",
            group_b="without_causal",
            metric_key="relative_l2_error",
            container_key="runs",
            group_field="condition",
            metric_field="evaluation_metrics",
            interpretation="AdvectionDiffusion1D causal loss should remain a weak/null effect.",
        ),
        AnalysisSpec(
            label="RQ3 Heat1D hybrid vs PINN-only",
            file_path=RESULTS_DIR / "rq3_10seed" / "results.json",
            group_a="hybrid_full",
            group_b="pinn_only",
            metric_key="relative_l2_error",
            container_key="raw_results",
            group_field="model_name",
            metric_field="metrics",
            interpretation="Heat1D hybrid should show a large accuracy gain over PINN-only.",
        ),
        AnalysisSpec(
            label="RQ4 Wave2D hybrid vs PINN-only",
            file_path=RESULTS_DIR / "rq4_10seed" / "results.json",
            group_a="hybrid_full",
            group_b="pinn_only",
            metric_key="relative_l2_error",
            container_key="raw_results",
            group_field="model_name",
            metric_field="metrics",
            interpretation="Wave2D hybrid should retain an in-domain accuracy gain over PINN-only.",
        ),
        AnalysisSpec(
            label="RQ6 Wave2D OOD hybrid vs PINN-only",
            file_path=RESULTS_DIR / "rq6_ood" / "results.json",
            group_a="hybrid_full",
            group_b="pinn_only",
            metric_key="relative_l2_error",
            container_key="ood_rows",
            group_field="model_name",
            metric_field="evaluation_metrics",
            interpretation=(
                "OOD coefficient shift should reduce the hybrid advantage relative to in-domain."
            ),
        ),
        AnalysisSpec(
            label="RQ1e KdV1D causal ablation (10-seed)",
            file_path=RESULTS_DIR / "rq1e_kdv_10seed_causal" / "results.json",
            group_a="with_causal",
            group_b="without_causal",
            metric_key="pde_residual_norm",
            container_key="runs",
            group_field="condition",
            metric_field="evaluation_metrics",
            interpretation=(
                "KdV1D causal loss should remain at most a weak directional "
                "effect relative to its runtime cost."
            ),
        ),
        AnalysisSpec(
            label="RQ1f AllenCahn1D causal ablation (10-seed)",
            file_path=RESULTS_DIR / "rq1f_allencahn_10seed_causal" / "results.json",
            group_a="with_causal",
            group_b="without_causal",
            metric_key="pde_residual_norm",
            container_key="runs",
            group_field="condition",
            metric_field="evaluation_metrics",
            interpretation=(
                "AllenCahn1D causal loss should remain a near-null result."
            ),
        ),
        AnalysisSpec(
            label="RQ1e-h KdV1D hybrid vs PINN-only (10-seed)",
            file_path=RESULTS_DIR / "rq1e_kdv_10seed" / "results.json",
            group_a="hybrid_full",
            group_b="pinn_only",
            metric_key="relative_l2_error",
            container_key="raw_results",
            group_field="model_name",
            metric_field="metrics",
            interpretation=(
                "KdV1D hybrid should improve accuracy over PINN-only."
            ),
        ),
        AnalysisSpec(
            label="RQ1f-h AllenCahn1D hybrid vs PINN-only (10-seed)",
            file_path=RESULTS_DIR / "rq1f_allencahn_10seed" / "results.json",
            group_a="hybrid_full",
            group_b="pinn_only",
            metric_key="relative_l2_error",
            container_key="raw_results",
            group_field="model_name",
            metric_field="metrics",
            interpretation=(
                "AllenCahn1D hybrid should improve accuracy over PINN-only."
            ),
        ),
        AnalysisSpec(
            label="RQ-PINO Heat1D hybrid vs PINO (accuracy, 10-seed)",
            file_path=RESULTS_DIR / "rq_pino_heat1d_10seed" / "results.json",
            group_a="hybrid",
            group_b="pino",
            metric_key="relative_l2_error",
            container_key="raw_results",
            group_field="model_name",
            metric_field="metrics",
            interpretation=(
                "Heat1D hybrid should strongly outperform PINO on solution accuracy."
            ),
        ),
        AnalysisSpec(
            label="RQ-PINO AdvDiff1D hybrid vs PINO (accuracy, 10-seed)",
            file_path=RESULTS_DIR / "rq_pino_advdiff" / "results.json",
            group_a="hybrid",
            group_b="pino",
            metric_key="relative_l2_error",
            container_key="raw_results",
            group_field="model_name",
            metric_field="metrics",
            interpretation=(
                "AdvDiff1D hybrid should outperform PINO on solution accuracy."
            ),
        ),
        AnalysisSpec(
            label="RQ-PINO Heat1D PINO vs hybrid (residual, 10-seed)",
            file_path=RESULTS_DIR / "rq_pino_heat1d_10seed" / "results.json",
            group_a="pino",
            group_b="hybrid",
            metric_key="pde_residual_norm",
            container_key="raw_results",
            group_field="model_name",
            metric_field="metrics",
            interpretation=(
                "Heat1D PINO should achieve lower PDE residual than the hybrid model."
            ),
        ),
    ]

    rq6_payload = read_json(RESULTS_DIR / "rq6_ood" / "results.json")
    # One pair per seed: the OOD metric is averaged over the shifted speeds.
    # (Earlier versions entered each (seed, speed) as its own pair, which treats
    # the two speeds of one trained model as independent replicates.)
    ood_rows = []
    for row in rq6_payload["raw_results"]:
        if not row["success"]:
            continue
        speeds = [str(sp) for sp in rq6_payload["ood_speeds"]]
        metrics = {
            key: sum(row["per_speed_metrics"][sp][key] for sp in speeds) / len(speeds)
            for key in row["per_speed_metrics"][speeds[0]]
        }
        ood_rows.append(
            {
                "model_name": row["model_name"],
                "seed": int(row["seed"]),
                "evaluation_metrics": metrics,
            }
        )
    rq6_payload["ood_rows"] = ood_rows
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    temp_rq6 = OUTPUT_JSON.parent / "_rq6_temp.json"
    temp_rq6.write_text(json.dumps(rq6_payload, indent=2), encoding="utf-8")
    for spec in specs:
        if spec.label == "RQ6 Wave2D OOD hybrid vs PINN-only":
            spec.file_path = temp_rq6
            break

    results = [summarize(spec) for spec in specs]
    for row in results:
        if row["label"] == "RQ6 Wave2D OOD hybrid vs PINN-only":
            row["file"] = "experiments/results/rq6_ood/results.json"
            break
    apply_bh_fdr(results, alpha=0.05)
    OUTPUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results)
    temp_rq6.unlink(missing_ok=True)

    print(f"Saved JSON: {OUTPUT_JSON}")
    print(f"Saved markdown: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
