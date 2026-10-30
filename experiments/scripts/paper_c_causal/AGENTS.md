<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# paper_c_causal

## Purpose
Scripts for Paper C: causal-loss diagnosis. The central hypothesis under test is that Stage-1 operator pretraining acts as an implicit temporal curriculum, and that AdaptiveCausalLoss (Stage 3) only helps when that pretraining is *absent* — otherwise it can hurt. Every script here interrogates this mechanism from a different angle: the core 2x2 ablation (pretrain x causal), gradient-alignment diagnostics, causal-weight visualization, statistical significance testing, and a literature comparison against Wang et al. (2024).

## Key Files
| File | Description |
|------|-------------|
| `ablation_statistics.py` | Paired exact Wilcoxon signed-rank tests on the 2×2 stage-ablation results (BH-FDR step-up across 9 PDEs × 3 comparisons = 27 tests; picks up every `paper_c/*_ablation` dir), writing `paper_c/ablation_statistics/results.{json,md}` |
| `causal_ablation_table.py` | Prints the LaTeX rows of the paper's causal × pretraining table (Δ% residual, bootstrap CI, BH verdict) from `ablation_statistics/results.json`; rerun it instead of editing the table by hand |
| `causal_weight_evolution.py` | Visualizes how per-timepoint causal weights `w_i = exp(-epsilon * sum_{j<i} MSE(residual_j))` evolve during Stage-3 training, with vs without pretraining |
| `clamping_ablation.py` | Isolates weight-magnitude collapse from gradient-direction conflict by comparing unclamped (`min_weight=0.0`) vs clamped (`min_weight=0.1`) `CausalLoss` across 9 PDEs × 10 seeds |
| `generate_combined_gradient_fig.py` | Generates the combined 2×2 gradient-flow figure for the paper into `docs/paper_c_causal/figures/` |
| `generate_figures.py` | Generates all Paper C figures from `experiments/results/paper_c/` into `docs/paper_c_causal/figures/` |
| `gradient_cosine_summary.py` | Aggregates per-PDE gradient-flow cosine-similarity stats (min/max/mean) across 4 PDEs (Heat1D, AdvDiff1D, KdV1D, AllenCahn1D) to check the cosine≈1.0 finding generalizes beyond Heat1D |
| `gradient_flow_analysis.py` | Measures cosine-similarity alignment between physics-loss and causal-loss gradients during training, with vs without pretraining |
| `pretrain_budget_sweep.py` | Sweeps the Stage-1 epoch budget `{50, 100, 200, 400}` × causal on/off on 3 indicative PDEs, to distinguish "absence" from "amount" of pretraining as the causal factor |
| `stage_ablation.py` | The 2×2 stage ablation (`skip_stage1` × `skip_stage3`) across 9 PDEs. `--condition-set controls` runs the confound controls instead: Stage 3 with uniform weights (ε=0, same 200 epochs) and no-pretraining arms with Stage 2 lengthened to 700 epochs; `--conditions` and `--seeds` split work across GPUs. Note the no-causal arms of the original design skip Stage 3 entirely, so they also train 200 fewer epochs |
| `temporal_error_analysis.py` | Evaluates prediction error at different time slices to see *where* in the time domain causal loss helps or hurts, across the 2×2 ablation conditions |
| `wang_comparison.py` | Three-way comparison of single-stage PINN (Wang et al. 2024 setting) vs single-stage PINN + causal vs the full 3-stage hybrid pipeline |

## For AI Agents

### Working In This Directory
- Figures for this paper go to `docs/paper_c_causal/figures/`, **not** `experiments/analysis/` (that directory holds cross-cutting mechanism figures shared across papers, produced by `../analyze_mechanisms.py`).
- Most scripts read from `experiments/results/paper_c/*/` (per-PDE `results.json` plus `aggregate.json`) — run `make fetch-data` first, or produce them locally by running the corresponding ablation script.
- `src/causal_pinn/physics/losses/causal_loss.py` currently has local, uncommitted changes (see repo `git status`) — since every script here trains against `CausalLoss`/`AdaptiveCausalLoss`, re-run the relevant ablation (and the library's own loss unit tests) before trusting fresh numbers out of this directory while that file is in flux.
- Run with: `PYTHONPATH=src python experiments/scripts/paper_c_causal/<script>.py`; long-running ones (`clamping_ablation.py`, `pretrain_budget_sweep.py`) write `progress.json` incrementally so they can be resumed/monitored.

### Testing Requirements
No pytest coverage for this directory itself; `make lint` covers it. The `CausalLoss`/`AdaptiveCausalLoss` classes these scripts exercise are unit-tested under `tests/` — run `make test` after touching loss code before re-deriving Paper C numbers.

### Common Patterns
- The `skip_stage1 x skip_stage3` (equivalently `with_pretrain`/`without_pretrain` × causal on/off) 2×2 design recurs across `ablation_statistics.py`, `stage_ablation.py`, `clamping_ablation.py`, `pretrain_budget_sweep.py`, and `temporal_error_analysis.py` — new mechanism scripts should follow the same condition-naming scheme (`full_pipeline_no_causal`, `full_pipeline_with_causal`, `no_pretrain_no_causal`, `no_pretrain_with_causal`) so `ablation_statistics.py` and `generate_figures.py` can consume them without modification.
- Scripts build models manually from `HybridPINNOperator`/`HybridModelWrapper` + `CausalGate` + `FNO` + `PINNBase` (rather than `experiment_utils.create_hybrid_model`) because they need fine-grained control over which stage runs.

## Dependencies

### Internal
- `causal_pinn.physics.losses` (`CausalLoss`, `AdaptiveCausalLoss`, `PhysicsLoss`)
- `causal_pinn.models.hybrid.hybrid_model` (`HybridConfig`, `HybridPINNOperator`, `HybridModelWrapper`), `causal_pinn.models.hybrid.causal_gate.CausalGate`, `causal_pinn.models.operators.fno.FNO`, `causal_pinn.models.pinn.base.PINNBase`
- `causal_pinn.physics.pde` (`Heat1D`, `AdvectionDiffusion1D`, `Burgers1D`, `KdV1D`, `AllenCahn1D`, `NavierStokes2D`, `ReactionDiffusion1D`)
- `causal_pinn.evaluation.benchmark_runner` (`BenchmarkConfig`, `BenchmarkRunner`), `causal_pinn.utils.random.set_seed`

### External
- PyTorch (custom training loops with `torch.optim.Adam` rather than `HybridTrainer` in several scripts, for per-epoch gradient introspection), NumPy, matplotlib.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
