<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# scripts

## Purpose
Every experiment entrypoint in the repo lives here or in one of the three `paper_*` subdirectories. This top level holds the shared factory/runner module (`experiment_utils.py`), the original numbered research-question scripts (`rq*.py`, the precursor to the paper split), and a handful of cross-cutting analysis/plotting helpers that operate on already-produced result JSONs.

## Key Files
| File | Description |
|------|-------------|
| `experiment_utils.py` | Shared factories (`create_pinn_model`, `create_hybrid_model`, `create_training_config`) and multi-seed runners (`run_single_experiment`, `run_multi_seed`, `create_model_factory`) that `rq*.py` and some `paper_combined/` scripts import — the entry point for understanding how a script wires a PDE + model + trainer together |
| `analyze_mechanisms.py` | Reads existing result JSONs and produces mechanism-diagnostic figures (causal vs non-causal training trajectories, per-stage loss decomposition, cross-model comparison) into `experiments/analysis/mechanism_figures/` |
| `generate_paper_tables.py` | Generates LaTeX table fragments (causal ablation, lambda sweep, hybrid comparison, OOD sections) from `experiments/results/` into `experiments/analysis/paper_tables.tex` |
| `plot_routing_results.py` | Compact paper figure comparing budget-aware routing selectors (RF, physics-final heuristic, training-prior, linear, random) |
| `profile_wave2d_memory.py` | Profiles peak CUDA memory for Wave2D physics-loss backprop at multiple collocation-point counts; emits a GO/NO-GO report |
| `statistical_analysis.py` | Paired-comparison test runner driven by `AnalysisSpec` dataclasses (exact Wilcoxon, bootstrap CI, paired d, BH-FDR step-up); writes `experiments/analysis/statistical_tests.{md,json}`. The Wave2D OOD test uses one pair per seed, averaging the two shifted speeds |
| `rq1_causal_ablation.py` | RQ1: CausalLoss (Stage 3) ON/OFF convergence-stability comparison on Heat1D |
| `rq1b_burgers_causal.py` | RQ1b: same ON/OFF comparison on the nonlinear Burgers1D equation |
| `rq1c_advdiff_causal.py` | RQ1c: same ON/OFF comparison on AdvectionDiffusion1D (has an exact solution, so relative L2 error is available) |
| `rq1d_epsilon_sweep.py` | RQ1d: sweeps `causal_epsilon` over `[0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]` on AdvectionDiffusion1D to find the optimal value |
| `rq1e_kdv_causal.py` | RQ1e: CausalLoss ON/OFF on 1D KdV, 10 seeds |
| `rq1e_kdv_hybrid.py` | RQ1e: pinn-only vs hybrid-no-gate vs hybrid-full model comparison on 1D KdV |
| `rq1f_allencahn_causal.py` | RQ1f: CausalLoss ON/OFF on 1D Allen-Cahn, 10 seeds |
| `rq1f_allencahn_hybrid.py` | RQ1f: pinn-only vs hybrid-no-gate vs hybrid-full model comparison on 1D Allen-Cahn |
| `rq2_physics_ablation.py` | RQ2: sweeps `lambda_physics` over `[0.0, 0.1, 0.5, 1.0, 5.0, 10.0]` on Heat1D to find the data/physics balance |
| `rq2b_burgers_lambda_sweep.py` | RQ2b: `lambda_physics` sweep on Burgers1D |
| `rq3_model_comparison.py` | RQ3: pinn-only vs hybrid-no-gate vs hybrid-full comparison on Heat1D across accuracy, physics consistency, and verifiability |
| `rq4_wave2d_model_comparison.py` | RQ4: pinn-only vs hybrid-full comparison on Wave2D |
| `rq5_wave2d_lambda_sweep.py` | RQ5: `lambda_physics` sweep on the Wave2D hybrid model |
| `rq6_ood_advdiff.py` | RQ6: out-of-distribution generalization across advection velocities on AdvectionDiffusion1D |
| `rq6_ood_heat1d.py` | RQ6: out-of-distribution generalization across diffusivity `alpha` on Heat1D |
| `rq6_ood_parametric_generalization.py` | RQ6: Wave2D wave-speed OOD sweep (pinn_only vs hybrid_full); `--seeds` overrides the default 5 seeds |
| `rq_adaptive_lambda_heat1d.py` | Adaptive `lambda_physics` scaling (alignment/conflict/dominance-threshold driven) vs fixed lambda on Heat1D |
| `rq_early_predictor.py` | Early-training-signal prediction and budget-aware policy-routing evaluation (RandomForest on early features) — precursor to `paper_a_routing/` |
| `rq_pino_advdiff.py` | Three-way PINO vs PINN-only vs hybrid comparison on AdvectionDiffusion1D |
| `rq_pino_comparison.py` | Three-way PINO vs PINN-only vs hybrid comparison on Heat1D |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `paper_a_routing/` | Paper A (early-dynamics routing/policy selection) experiment and figure scripts (see `paper_a_routing/AGENTS.md`) |
| `paper_c_causal/` | Paper C (causal-loss diagnosis) experiment and figure scripts (see `paper_c_causal/AGENTS.md`) |
| `paper_combined/` | Cross-cutting and external-baseline (HyPINO, PINNacle) experiments spanning both papers (see `paper_combined/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- **`PYTHONPATH=src` is mandatory** to import `causal_pinn` — it is never set implicitly for script invocations (only for pytest).
- Scripts that `from experiment_utils import ...` rely on Python automatically adding the invoked script's own directory to `sys.path`; this works when you run `python experiments/scripts/rq_pino_advdiff.py` from anywhere, but breaks if you copy a script elsewhere without `experiment_utils.py` alongside it.
- Example full invocation: `PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python experiments/scripts/rq4_wave2d_model_comparison.py --output-dir experiments/results/rq4_10seed --device cuda`.

### Testing Requirements
No pytest coverage for this directory. `make lint` (`ruff check src tests experiments`) is the only automated gate. Sanity-check a script by running it with `--device cpu` and a scratch `--output-dir` before committing to a long GPU run.

### Common Patterns
- Nearly every `rq*.py` script follows the same shape: a module-level `SEEDS` list, a `run_single_experiment(...)` helper that calls `set_seed`, builds a model, trains with `HybridTrainer`, evaluates with `BenchmarkRunner`, then a `main()` that loops over conditions/seeds and `json.dump`s to `--output-dir`.
- `rq1*` scripts share the "CausalLoss ON/OFF" design; the letter suffix (`b`, `c`, `d`, `e`, `f`) denotes the same research question re-run on a different PDE or hyperparameter axis, not a different question.
- Scripts needing a hybrid model construct it directly via `HybridPINNOperator`/`HybridModelWrapper` + `CausalGate` + `FNO` + `PINNBase`, or go through `experiment_utils.create_hybrid_model` — prefer the latter for new scripts unless per-stage control over the components is required.

## Dependencies

### Internal
- `causal_pinn.evaluation.benchmark_runner` (`BenchmarkConfig`, `BenchmarkRunner`)
- `causal_pinn.models.pinn.base.PINNBase`, `causal_pinn.models.hybrid.*` (`HybridConfig`, `HybridPINNOperator`, `HybridModelWrapper`, `CausalGate`), `causal_pinn.models.operators.fno.FNO`, `causal_pinn.models.pino.PINOModel`
- `causal_pinn.physics.pde.*` (`Heat1D`, `Burgers1D`, `AdvectionDiffusion1D`, `KdV1D`, `AllenCahn1D`, `Wave2D`, ...)
- `causal_pinn.training.trainer.HybridTrainer`, `causal_pinn.training.pino_trainer.PINOTrainer`
- `causal_pinn.utils.config.TrainingConfig`, `causal_pinn.utils.random.set_seed`

### External
- PyTorch, NumPy, matplotlib — training, numerics, figures.
- scikit-learn (`RandomForestRegressor`) — `rq_early_predictor.py`.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
