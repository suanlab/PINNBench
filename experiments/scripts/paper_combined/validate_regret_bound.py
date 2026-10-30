#!/usr/bin/env python3
"""Validate the Probe-then-Commit regret bound against empirical routing data.

Theory (Theorem 1 in Paper A):
    Under sub-Gaussian probe noise and monotone probe-final correlation,
    the regret of the probe-then-commit router satisfies, with probability
    >= 1 - delta:

        Regret <= 2 * sigma * sqrt(2 * ln(2P / delta)) / rho_k

This script:
    1. Loads all Paper A routing data (450 runs across 13 PDEs).
    2. For each PDE and probe window k in {10, 20, 50, 100}, computes:
       - rho_k: Spearman correlation between probe loss and final loss
       - sigma: std of centred probe estimates across seeds
       - Delta: gap between best and second-best policy
    3. Computes the theoretical regret bound.
    4. Compares with the observed regret of the probe-then-commit heuristic it
       bounds: rank policies by their total training loss at the end of the
       k-epoch probe (``total_final``), commit to the lowest, and score the
       pick by PDE residual against the oracle policy, leave-one-PDE-out
       (``causal_pinn.training.policy_routing.evaluate_policy_routing``).
       Observed regret is recomputed at every k, including k=20.

       The released file before v1.3 took observed regret from a routing
       artifact keyed ``relaxed_w{k}`` that was never part of the release; its
       values equal this heuristic's regret at k=50 and were reused for k=10
       and k=100, while k=20 was left unevaluated.
    5. Generates fig_regret_bound.pdf.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[3]
PAPER_A_DIR = REPO / "experiments" / "results" / "paper_a"

PDES = [
    "advdiff1d",
    "advdiff1d_highpe",
    "allencahn1d",
    "allencahn1d_sharp",
    "burgers1d",
    "burgers1d_lownu",
    "burgers1d_verylow",
    "heat1d",
    "kdv1d",
    "navier_stokes2d",
    "reaction_diffusion1d",
    "reaction_diffusion1d_stiff",
    "wave2d",
]

PROBE_WINDOWS = [10, 20, 50, 100]
DELTA = 0.05  # confidence parameter


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_pde_runs(pde: str) -> list[dict[str, Any]]:
    """Load raw results for a single PDE."""
    path = PAPER_A_DIR / pde / "results.json"
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return data["raw_results"]


def observed_regrets_at(k: int) -> dict[str, float]:
    """Regret of the ``total_final`` probe-then-commit heuristic at probe window k."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_meta_router import load_all_samples  # type: ignore[import-not-found]

    from causal_pinn.training.early_predictor import FeatureExtractor
    from causal_pinn.training.policy_routing import evaluate_policy_routing

    dataset = load_all_samples(PAPER_A_DIR, FeatureExtractor(window_size=k), "pde_residual_norm")
    routing = evaluate_policy_routing(
        dataset, probe_epochs=k, total_epochs=500, strict_grouping=True,
    )
    folds = routing["selectors"]["total_final"]["folds"]
    return {f["held_out_group"]: float(f["regret"]) for f in folds}


# ---------------------------------------------------------------------------
# Probe and final loss extraction
# ---------------------------------------------------------------------------
def probe_loss(run: dict, k: int) -> float | None:
    """Average total loss over the first k stage-2 epochs."""
    s2 = run.get("training_history", {}).get("stage2_losses", [])
    if len(s2) < k:
        return None
    return float(np.mean([e["total"] for e in s2[:k]]))


def final_loss(run: dict) -> float | None:
    """Final total loss (last stage-2 epoch)."""
    s2 = run.get("training_history", {}).get("stage2_losses", [])
    if not s2:
        return None
    return float(s2[-1]["total"])


# ---------------------------------------------------------------------------
# Core statistics per PDE
# ---------------------------------------------------------------------------
def compute_pde_stats(pde: str, k: int) -> dict[str, Any] | None:
    """Compute rho_k, sigma, Delta for a PDE at probe window k."""
    runs = load_pde_runs(pde)
    if not runs:
        return None

    # Collect (policy, seed, probe_loss, final_loss) tuples
    records: list[tuple[str, int, float, float]] = []
    for r in runs:
        pl = probe_loss(r, k)
        fl = final_loss(r)
        if pl is not None and fl is not None:
            records.append((r["model_name"], r["seed"], pl, fl))

    if len(records) < 4:
        return None

    probe_vals = np.array([r[2] for r in records])
    final_vals = np.array([r[3] for r in records])

    # rho_k: Spearman correlation across all (policy x seed) runs
    rho_k, rho_p = stats.spearmanr(probe_vals, final_vals)

    # sigma: std of centred probe estimates per policy, then average
    policies = sorted(set(r[0] for r in records))
    sigmas: list[float] = []
    policy_means: dict[str, float] = {}
    for pol in policies:
        pol_probes = [r[2] for r in records if r[0] == pol]
        pol_finals = [r[3] for r in records if r[0] == pol]
        if len(pol_probes) >= 2:
            sigmas.append(float(np.std(pol_probes, ddof=1)))
        policy_means[pol] = float(np.mean(pol_finals))

    sigma = float(np.mean(sigmas)) if sigmas else 0.0

    # Delta: gap between best and second-best policy (by mean final loss)
    sorted_policies = sorted(policy_means.items(), key=lambda x: x[1])
    delta_gap = sorted_policies[1][1] - sorted_policies[0][1] if len(sorted_policies) >= 2 else 0.0

    P = len(policies)

    return {
        "pde": pde,
        "k": k,
        "n_runs": len(records),
        "P": P,
        "rho_k": float(rho_k),
        "rho_p": float(rho_p),
        "sigma": float(sigma),
        "delta_gap": float(delta_gap),
        "best_policy": sorted_policies[0][0] if sorted_policies else "N/A",
        "best_final": float(sorted_policies[0][1]) if sorted_policies else float("nan"),
    }


def theoretical_bound(sigma: float, P: int, rho_k: float, delta: float) -> float:
    """Compute the regret bound: 2*sigma*sqrt(2*ln(2P/delta)) / rho_k."""
    if rho_k <= 0:
        return float("inf")
    return 2.0 * sigma * math.sqrt(2.0 * math.log(2.0 * P / delta)) / rho_k


# ---------------------------------------------------------------------------
# Observed regret from routing evaluation
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate the probe-then-commit regret bound on 13 PDEs.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="experiments/results/paper_a",
        help="Directory containing routing evaluation results",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("Probe-then-Commit Regret Bound Validation")
    print("=" * 72)

    all_results: list[dict[str, Any]] = []

    for k in PROBE_WINDOWS:
        print(f"\n--- Probe window k = {k} ---")

        observed_regrets = observed_regrets_at(k)

        for pde in PDES:
            st = compute_pde_stats(pde, k)
            if st is None:
                continue

            bound = theoretical_bound(st["sigma"], st["P"], st["rho_k"], DELTA)
            obs_regret = observed_regrets.get(pde, float("nan"))

            bound_valid: bool | None = None
            if not math.isnan(obs_regret) and not math.isinf(bound):
                bound_valid = bool(obs_regret <= bound)

            tightness_ratio = float("nan")
            if obs_regret > 0 and not math.isnan(obs_regret) and not math.isinf(bound):
                tightness_ratio = float(bound / obs_regret)

            # Distinguish "the bound was checked and held/failed" from "there was
            # no observed regret to check against". Both used to surface as NaN,
            # which made an unevaluated probe window look like a bound failure.
            if math.isnan(obs_regret):
                check_status = "not_evaluated"
            elif bound_valid:
                check_status = "holds"
            else:
                check_status = "violated"

            result = {
                **st,
                "theoretical_bound": float(bound),
                "observed_regret": float(obs_regret),
                "bound_valid": bound_valid,
                "check_status": check_status,
                "tightness_ratio": tightness_ratio,
            }
            all_results.append(result)

            valid_str = (
                "VALID" if result["bound_valid"] is True
                else "VIOLATED" if result["bound_valid"] is False
                else "N/A"
            )
            print(
                f"  {pde:30s}  rho={st['rho_k']:.3f}  sigma={st['sigma']:.4f}  "
                f"P={st['P']}  bound={bound:.4f}  obs={obs_regret:.4f}  [{valid_str}]"
            )

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    for k in PROBE_WINDOWS:
        k_results = [r for r in all_results if r["k"] == k]
        valid_checks = [r for r in k_results if r["bound_valid"] is not None]
        n_valid = sum(1 for r in valid_checks if r["bound_valid"])
        n_total = len(valid_checks)

        tight = [
            r["tightness_ratio"]
            for r in k_results
            if not math.isnan(r["tightness_ratio"]) and not math.isinf(r["tightness_ratio"])
        ]
        mean_tight = float(np.mean(tight)) if tight else float("nan")

        rhos = [r["rho_k"] for r in k_results]
        mean_rho = float(np.mean(rhos)) if rhos else float("nan")

        print(
            f"  k={k:3d}: valid={n_valid}/{n_total}, "
            f"mean_tightness={mean_tight:.1f}x, "
            f"mean_rho_k={mean_rho:.3f}"
        )

    # ------------------------------------------------------------------
    # Overall assessment
    # ------------------------------------------------------------------
    valid_checks_all = [r for r in all_results if r["bound_valid"] is not None]
    n_valid_all = sum(1 for r in valid_checks_all if r["bound_valid"])
    n_total_all = len(valid_checks_all)

    tight_all = [
        r["tightness_ratio"]
        for r in all_results
        if not math.isnan(r["tightness_ratio"]) and not math.isinf(r["tightness_ratio"])
    ]
    mean_tight_all = float(np.mean(tight_all)) if tight_all else float("nan")

    print(f"\n  Overall: {n_valid_all}/{n_total_all} bounds valid")
    if mean_tight_all <= 10:
        print(f"  Tightness: {mean_tight_all:.1f}x (TIGHT -- theory is useful)")
    elif mean_tight_all <= 100:
        print(f"  Tightness: {mean_tight_all:.1f}x (MODERATE -- theory is conservative)")
    else:
        print(f"  Tightness: {mean_tight_all:.1f}x (LOOSE -- theory is valid but conservative)")

    # ------------------------------------------------------------------
    # Generate figure
    # ------------------------------------------------------------------
    generate_figure(all_results, Path(args.output_dir))

    # ------------------------------------------------------------------
    # Save JSON results
    # ------------------------------------------------------------------
    out_path = Path(args.output_dir) / "regret_bound_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _jsonify(obj: Any) -> Any:
        """Convert numpy/special types for JSON serialisation."""
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return str(v) if math.isnan(v) or math.isinf(v) else v
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return str(obj)
        return obj

    with open(out_path, "w") as f:
        json.dump(_jsonify(all_results), f, indent=2)
    print(f"\n  Results saved to {out_path}")


def generate_figure(results: list[dict[str, Any]], fig_dir: Path) -> None:
    """Generate fig_regret_bound.pdf: actual regret vs theoretical bound per PDE."""

    # Focus on k=10 (the probe window used in the paper) for the main figure
    k_main = 10
    k_results = [
        r for r in results
        if r["k"] == k_main
        and r["bound_valid"] is not None
        and not math.isinf(r["theoretical_bound"])
    ]

    if not k_results:
        print("  WARNING: No results for k=10 with valid bounds; skipping figure.")
        return

    # Sort by observed regret
    k_results.sort(key=lambda r: r["observed_regret"])

    pde_labels = [r["pde"].replace("_", " ").replace("1d", "").replace("2d", "").strip()
                  for r in k_results]
    observed = [r["observed_regret"] for r in k_results]
    bounds = [r["theoretical_bound"] for r in k_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [2, 1]})

    # --- Panel (a): Bar chart ---
    ax = axes[0]
    x = np.arange(len(pde_labels))
    width = 0.35

    ax.bar(x - width / 2, observed, width, label="Observed regret",
                      color="#2196F3", alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, bounds, width, label="Theoretical bound",
                      color="#FF9800", alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Regret", fontsize=12)
    ax.set_title(f"(a) Observed Regret vs Theoretical Bound  ($k={k_main}$, $\\delta=0.05$)",
                 fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(pde_labels, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Mark valid/violated
    for i, r in enumerate(k_results):
        marker_color = "#4CAF50" if r["bound_valid"] else "#F44336"
        ax.plot(x[i], max(observed[i], bounds[i]) * 1.08, marker="v" if r["bound_valid"] else "x",
                color=marker_color, markersize=7, zorder=5)

    # --- Panel (b): rho_k vs k ---
    ax2 = axes[1]
    for pde in PDES:
        rhos = []
        ks = []
        for k in PROBE_WINDOWS:
            matching = [r for r in results if r["pde"] == pde and r["k"] == k]
            if matching:
                rhos.append(matching[0]["rho_k"])
                ks.append(k)
        if rhos:
            ax2.plot(ks, rhos, "o-", alpha=0.6, markersize=4,
                     label=pde.replace("_", " ").replace("1d", "").replace("2d", "").strip())

    ax2.set_xlabel("Probe window $k$", fontsize=12)
    ax2.set_ylabel("Spearman $\\rho_k$", fontsize=12)
    ax2.set_title("(b) Probe-Final Correlation vs Window Size", fontsize=12, pad=10)
    ax2.set_ylim(-0.1, 1.05)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax2.legend(fontsize=6, ncol=2, loc="lower right")

    plt.tight_layout()
    fig_path = fig_dir / "fig_regret_bound.pdf"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    print(f"\n  Figure saved to {fig_path}")

    # Also save PNG for quick viewing
    fig.savefig(fig_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
