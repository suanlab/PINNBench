#!/usr/bin/env python3
"""Statistical analysis for Paper C's 2x2 stage ablation results.

Runs paired Wilcoxon signed-rank tests on the stage ablation data across 8 PDEs
to determine which effects (causal, pretrain) are statistically significant.

Tests per PDE:
  1. Causal effect WITH pretrain: full_pipeline_no_causal vs full_pipeline_with_causal
  2. Causal effect WITHOUT pretrain: no_pretrain_no_causal vs no_pretrain_with_causal
  3. Pretrain effect: full_pipeline_no_causal vs no_pretrain_no_causal

All tests use BH-FDR correction across 8 PDEs x 3 tests = 24 tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "experiments" / "results" / "paper_c"
OUTPUT_DIR = RESULTS_DIR / "ablation_statistics"

ABLATION_DIRS = sorted(RESULTS_DIR.glob("*_ablation"))

COMPARISONS = [
    {
        "label": "Causal WITH pretrain",
        "group_a": "full_pipeline_no_causal",
        "group_b": "full_pipeline_with_causal",
        "interpretation": "Effect of adding causal loss when operator pretraining is present.",
    },
    {
        "label": "Causal WITHOUT pretrain",
        "group_a": "no_pretrain_no_causal",
        "group_b": "no_pretrain_with_causal",
        "interpretation": "Effect of adding causal loss when operator pretraining is skipped.",
    },
    {
        "label": "Pretrain effect",
        "group_a": "full_pipeline_no_causal",
        "group_b": "no_pretrain_no_causal",
        "interpretation": "Effect of operator pretraining (stage 1) without causal loss.",
    },
]

METRIC_KEY = "pde_residual_norm"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_paired_values(
    raw_results: list[dict], group_a: str, group_b: str
) -> tuple[list[float], list[float]]:
    """Extract paired metric values for two conditions, matched by seed."""
    values_a: dict[int, float] = {}
    values_b: dict[int, float] = {}

    for row in raw_results:
        condition = row["condition"]
        seed = int(row["seed"])
        value = float(row["metrics"][METRIC_KEY])
        if condition == group_a:
            values_a[seed] = value
        elif condition == group_b:
            values_b[seed] = value

    common_seeds = sorted(set(values_a) & set(values_b))
    if not common_seeds:
        raise ValueError(f"No common seeds between {group_a} and {group_b}")
    return [values_a[s] for s in common_seeds], [values_b[s] for s in common_seeds]


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Compute average ranks handling ties."""
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
    """Exact Wilcoxon signed-rank test.

    Returns: (W_observed, p_value, W_plus, n_nonzero_pairs)
    """
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


def bootstrap_mean_ci(
    diff: np.ndarray, n_boot: int = 10000, seed: int = 42
) -> tuple[float, float]:
    """Bootstrap 95% CI for paired mean difference."""
    if len(diff) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(diff, size=(n_boot, len(diff)), replace=True)
    means = np.mean(samples, axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def paired_cohens_d(diff: np.ndarray) -> float:
    """Paired Cohen's d = mean(diff) / std(diff)."""
    if len(diff) < 2 or np.isclose(np.std(diff, ddof=1), 0.0):
        return 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))


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
    indexed = sorted(results, key=lambda row: row["wilcoxon_p"])
    running = 1.0
    for rank_1 in range(m, 0, -1):
        row = indexed[rank_1 - 1]
        running = min(running, row["wilcoxon_p"] * m / rank_1)
        row["bh_fdr_threshold"] = alpha * rank_1 / m
        row["bh_fdr_p_adjusted"] = running
        row["bh_fdr_significant"] = running <= alpha


def run_test(
    pde_name: str,
    raw_results: list[dict],
    comparison: dict,
) -> dict:
    """Run a single paired test for one PDE and one comparison."""
    a_vals, b_vals = extract_paired_values(
        raw_results, comparison["group_a"], comparison["group_b"]
    )
    a = np.asarray(a_vals, dtype=float)
    b = np.asarray(b_vals, dtype=float)
    diff = a - b  # group_a - group_b

    observed, p_value, w_plus, n_pairs = wilcoxon_exact(diff)
    ci_low, ci_high = bootstrap_mean_ci(diff)
    d = paired_cohens_d(diff)

    return {
        "pde": pde_name,
        "comparison": comparison["label"],
        "group_a": comparison["group_a"],
        "group_b": comparison["group_b"],
        "metric": METRIC_KEY,
        "n_pairs": n_pairs,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_diff": float(np.mean(diff)),
        "bootstrap_95ci": [ci_low, ci_high],
        "wilcoxon_w": observed,
        "wilcoxon_w_plus": w_plus,
        "wilcoxon_p": p_value,
        "cohens_d": d,
        "interpretation": comparison["interpretation"],
    }


def format_markdown(results: list[dict]) -> str:
    """Generate markdown report."""
    lines: list[str] = []
    lines.append("# Paper C: Stage Ablation Statistical Analysis")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- **Test**: Exact Wilcoxon signed-rank (paired by seed)")
    lines.append("- **CI**: Bootstrap 95% CI for paired mean difference (10,000 resamples)")
    lines.append("- **Effect size**: Paired Cohen's d")
    lines.append(
        "- **Multiple-comparison correction**: Benjamini-Hochberg FDR (q=0.05) "
        "across all 24 tests"
    )
    lines.append(f"- **Metric**: `{METRIC_KEY}` (lower is better)")
    lines.append("- **Convention**: diff = group_a - group_b (negative means A is better)")
    lines.append("")
    lines.append("## Results Table")
    lines.append("")
    lines.append(
        "| PDE | Comparison | Mean diff | 95% CI | Wilcoxon p | BH p_adj "
        "| Cohen's d | Sig? |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")

    for row in results:
        ci_low, ci_high = row["bootstrap_95ci"]
        p_adj = row.get("bh_fdr_p_adjusted", float("nan"))
        sig = "yes" if row.get("bh_fdr_significant", False) else "no"
        lines.append(
            f"| {row['pde']} "
            f"| {row['comparison']} "
            f"| {row['mean_diff']:.6f} "
            f"| [{ci_low:.6f}, {ci_high:.6f}] "
            f"| {row['wilcoxon_p']:.4f} "
            f"| {p_adj:.4f} "
            f"| {row['cohens_d']:.3f} "
            f"| {sig} |"
        )

    lines.append("")
    lines.append("## Summary")
    lines.append("")

    for comp_label in ["Causal WITH pretrain", "Causal WITHOUT pretrain", "Pretrain effect"]:
        subset = [r for r in results if r["comparison"] == comp_label]
        n_sig = sum(1 for r in subset if r.get("bh_fdr_significant", False))
        n_total = len(subset)
        lines.append(f"- **{comp_label}**: {n_sig}/{n_total} significant")

    n_sig_total = sum(1 for r in results if r.get("bh_fdr_significant", False))
    lines.append(f"- **Overall**: {n_sig_total}/{len(results)} significant (BH-FDR q=0.05)")

    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("")
    for comp_label in ["Causal WITH pretrain", "Causal WITHOUT pretrain", "Pretrain effect"]:
        subset = [r for r in results if r["comparison"] == comp_label]
        sig_pdes = [r["pde"] for r in subset if r.get("bh_fdr_significant", False)]
        ns_pdes = [r["pde"] for r in subset if not r.get("bh_fdr_significant", False)]
        lines.append(f"### {comp_label}")
        if sig_pdes:
            lines.append(f"- Significant: {', '.join(sig_pdes)}")
        if ns_pdes:
            lines.append(f"- Not significant: {', '.join(ns_pdes)}")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    all_results: list[dict] = []

    for ablation_dir in ABLATION_DIRS:
        results_path = ablation_dir / "results.json"
        if not results_path.exists():
            print(f"SKIP: {results_path} not found")
            continue

        payload = read_json(results_path)
        pde_name = payload.get("pde", ablation_dir.name.replace("_ablation", ""))
        raw_results = payload["raw_results"]

        print(f"Processing {pde_name} ({len(raw_results)} raw results)")

        for comparison in COMPARISONS:
            try:
                result = run_test(pde_name, raw_results, comparison)
                all_results.append(result)
            except ValueError as e:
                print(f"  WARN: {pde_name} / {comparison['label']}: {e}")

    print(f"\nTotal tests: {len(all_results)}")
    apply_bh_fdr(all_results, alpha=0.05)

    # Save outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "results.json"
    json_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Saved JSON: {json_path}")

    md_path = OUTPUT_DIR / "results.md"
    md_content = format_markdown(all_results)
    md_path.write_text(md_content, encoding="utf-8")
    print(f"Saved markdown: {md_path}")

    # Print summary to stdout
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for comp_label in ["Causal WITH pretrain", "Causal WITHOUT pretrain", "Pretrain effect"]:
        subset = [r for r in all_results if r["comparison"] == comp_label]
        n_sig = sum(1 for r in subset if r.get("bh_fdr_significant", False))
        n_total = len(subset)
        print(f"  {comp_label}: {n_sig}/{n_total} significant")

    n_sig_total = sum(1 for r in all_results if r.get("bh_fdr_significant", False))
    print(f"  Overall: {n_sig_total}/{len(all_results)} significant (BH-FDR q=0.05)")


if __name__ == "__main__":
    main()
