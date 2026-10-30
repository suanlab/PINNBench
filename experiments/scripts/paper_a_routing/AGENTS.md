<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# paper_a_routing

## Purpose
Scripts for Paper A: early-dynamics routing / policy selection. These evaluate whether cheap early-training signals (probe windows of physics loss, feature-extracted training curves) can predict which model policy (`pinn_only`, `hybrid_no_gate`, `hybrid_full`, `pino`, ...) will win on a given PDE — and, if so, how reliably and at what regret. Scripts here consume the results produced by `run_policy_suite.py` (and `../rq_early_predictor.py`), run leave-one-PDE-out cross-validation across routers, and generate the paper's routing figures and robustness checks (bootstrap stability, family-level holdout, NTK-proxy gradient alignment, regret-bound validation lives in `../paper_combined/`).

## Key Files
| File | Description |
|------|-------------|
| `bootstrap_stability.py` | Resamples 9-of-13 PDE leave-one-out folds with replacement (1000 reps, seed=42), reporting mean/std/95% CI per routing selector — addresses a reviewer concern that rankings could be an artifact of which 13 PDEs were chosen |
| `evaluate_bandits.py` | Evaluates contextual bandit routers (`LinUCBRouter`, `SuccessiveHalvingRouter`) against RF and heuristic baselines via leave-one-PDE-out cross-validation |
| `evaluate_routing_full.py` | Full routing evaluation across all Paper A experiments (450 decisions, 13 PDEs) comparing RandomForest, GradientBoosting, Ridge, and the `physics_final` heuristic on both `pde_residual_norm` and `relative_l2_error` targets. `--holdout family` retrains each fold without the held-out PDE's whole family (true LOPO-F, written to `paper_combined/routing_family_holdout/`) |
| `family_holdout_analysis.py` | Groups fold decisions by PDE family and reports per-family accuracy, macro accuracy and fully-correct families. Only a true family holdout when fed `routing_family_holdout/results.json`; on the default LOPO file it just regroups leave-one-PDE-out folds |
| `generate_dissociation_fig.py` | Generates the accuracy-regret dissociation figure showing `physics_final` matches RF's accuracy but has far worse regret |
| `generate_figures.py` | Generates all Paper A figures from `experiments/results/paper_a/` and `rq_early_predictor_*` results into `docs/paper_a_routing/figures/` |
| `routing_tables.py` | Prints the LaTeX rows of the paper's routing decision tables (`--table perpde`, `family`, `lopof`) from the released fold files; rerun it instead of editing those tables |
| `holdout_validation.py` | Part 1: trains the RF router on 10 PDEs, tests on 3 unseen PDEs (out-of-distribution router generalization). Part 2: top-3-feature router ablation on the L2 target |
| `ntk_proxy_analysis.py` | NTK proxy: cosine similarity between the physics-loss gradient at early checkpoints (epochs 10/50/100/250) and at the final checkpoint (epoch 500), as a practical stand-in for the full Neural Tangent Kernel |
| `run_policy_suite.py` | Unified policy-suite runner — trains all (policy, seed) combinations for a given PDE and writes a `results.json` schema-compatible with `EarlyTrainingPredictor.build_dataset` |
| `scaling_and_ablation.py` | Part 1: routing-accuracy scaling as PDE diversity grows (5 → 9 → 13 PDEs). Part 2: feature ablation (top-K features by RF importance) |

## For AI Agents

### Working In This Directory
- Figures for this paper go to `docs/paper_a_routing/figures/`, **not** `experiments/analysis/` (that directory holds cross-cutting mechanism figures shared across papers).
- Most scripts read from `experiments/results/paper_a/*/results.json` and `experiments/results/paper_combined/routing_evaluation/results.json` — run `make fetch-data` (or `run_policy_suite.py`) first; these scripts do not generate their own training data.
- Several scripts (`evaluate_bandits.py`, `evaluate_routing_full.py`, `holdout_validation.py`, `scaling_and_ablation.py`) add `sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))` at the top as a self-contained fallback so they run without `PYTHONPATH` set — still prefer explicit `PYTHONPATH=src` for consistency with the rest of the repo.
- Run with: `PYTHONPATH=src python experiments/scripts/paper_a_routing/<script>.py`.

### Testing Requirements
No pytest coverage. `make lint` covers this directory. Verify a change by re-running the script against existing `experiments/results/` data and diffing the printed/summary output.

### Common Patterns
- Routing datasets are built via `causal_pinn.training.early_predictor.{FeatureExtractor, PredictorDataset, PredictorSample}` — this is the shared vocabulary across nearly every script here.
- Leave-one-PDE-out cross-validation is the standard evaluation protocol; "accuracy" = fraction of held-out PDEs where the selected policy matches the oracle, "regret" = relative gap between selected and oracle metric.
- Policy names (`pinn_only`, `hybrid_no_gate`, `hybrid_full`, `pino`) and router names (`random_forest`/RF, `physics_final`, `training_prior`, `linear_regression`, `random`) recur across scripts and figures — keep naming consistent when adding new ones.

## Dependencies

### Internal
- `causal_pinn.training.early_predictor` (`FeatureExtractor`, `PredictorDataset`, `PredictorSample`, `EarlyTrainingPredictor`, `extract_runs`, `get_final_metrics`, `get_training_history`, `load_experiment_results`)
- `causal_pinn.training.contextual_bandit` (`LinUCBRouter`, `SuccessiveHalvingRouter`)
- `causal_pinn.training.policy_routing` (`evaluate_policy_routing`, `remap_dataset_groups`)
- `causal_pinn.models.pinn.base.PINNBase` (used by `ntk_proxy_analysis.py` for checkpoint gradients)

### External
- NumPy, scikit-learn (`RandomForestRegressor`, `GradientBoostingRegressor`, `Ridge`) — router models.
- matplotlib (`Agg`/`pdf` backends) — figure generation.
- PyTorch — `ntk_proxy_analysis.py` gradient computation.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
