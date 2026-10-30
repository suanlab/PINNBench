<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# paper_combined

## Purpose
Cross-cutting and external-baseline experiments that don't belong exclusively to Paper A or Paper C: comparisons against the external HyPINO foundation model and the PINNacle benchmark suite, evaluation of the PDE-aware MetaRouter, robustness probes (noisy data, adaptive epsilon), and the theoretical regret-bound validation for Paper A's routing theorem. Several scripts here (`evaluate_meta_router.py`, `validate_regret_bound.py`) are best understood as Paper A extensions that needed the full combined benchmark; others (the `hypino_*`/`finetune_hypino_*` family, `run_pinnacle_subset.py`) are honesty-oriented reviewer-response probes explicitly scoped as *not* full method reproductions.

## Key Files
| File | Description |
|------|-------------|
| `aggregate_external_probes.py` | Aggregates external-probe result JSONs (e.g. per-seed HyPINO finetune runs) into mean/std/min/max summaries |
| `evaluate_hypino_heat1d_adapter.py` | Zero-shot HyPINO evaluation on a coordinate-mapped Heat1D task (`(x,t)->[-1,1]^2`); explicitly a compatibility/scope probe, not a fair head-to-head baseline |
| `evaluate_hypino_official.py` | Compatibility wrapper around HyPINO's own `evaluate.py` (fixes a `strict`/`ignore_size_mismatch` loader mismatch) across its 7 official benchmark `.pkl` files; requires `--hypino-root`/`--weights` pointing at an external HyPINO checkout |
| `evaluate_meta_router.py` | Evaluates the PDE-aware MetaRouter (MLP + PDE features) against RF/GB/heuristic/random baselines via leave-one-PDE-out CV on the full 13-PDE, 450-decision Paper A benchmark; `--holdout family` (with `--output-dir`) retrains each fold without the held-out PDE's family (true LOPO-F, `meta_router_family_holdout/`) |
| `finetune_hypino_advdiff1d.py` | Fine-tunes a HyPINO-generated target PINN on the mapped AdvectionDiffusion1D task; imports helpers from `evaluate_hypino_heat1d_adapter.py` |
| `finetune_hypino_heat1d.py` | Fine-tunes a HyPINO-generated target PINN on the mapped Heat1D task — closer to a same-task comparison, still not a full same-method reproduction |
| `hypino_matched_budget.py` | Wall-clock-matched-budget comparison: HyPINO finetune vs from-scratch hybrid pipeline on Heat1D/AdvDiff1D/Burgers1D/AllenCahn1D, timed on the same GPU (Reviewer 2 binding concern); falls back to a `pinn_finetune` baseline for PDEs HyPINO can't represent |
| `multiscale2d_policy_suite.py` | Follow-up to `multiscale2d_probe.py`: runs the **full policy library** (5 policies x 10 seeds) on `MultiscaleHeat2D` with the per-mode projection, to test whether routing could select a policy that resolves the fine scale. It cannot -- 0/50 runs recover 10% of the fine coefficient. Imports `create_model`/`PDE_REGISTRY` from `paper_a_routing/run_policy_suite.py` so the policies and schedule match the routing benchmark; carries a `--disable-cudnn` escape hatch for hosts with a broken cuDNN |
| `multiscale2d_probe.py` | Track B1: runs the `with_pretrain` vs `pinn_only` contrast on `MultiscaleHeat2D` (two Dirichlet eigenmodes, 4x wavelength and 16x decay-rate separation), 10 seeds. Beyond aggregate relative L2 it projects the learned field onto each eigenmode and scores them separately, which is what shows the pretraining advantage is earned entirely on the coarse envelope while neither branch recovers the fine mode |
| `noisy_data_probe.py` | Corrupts Stage-1 reference data with relative Gaussian noise (`sigma` in `{0.0, 0.01, 0.05, 0.1, 0.2}`) and compares `with_pretrain` vs `pinn_only`, testing whether the pretraining benefit is a noiseless-regime artifact |
| `run_pinnacle_subset.py` | Runs a small executable subset of the PINNacle benchmark (Burgers1D, Heat2D-multiscale, Helmholtz2D, Poisson1D, Wave1D) for artifact-level alignment; requires `--pinnacle-root` pointing at an external PINNacle checkout |
| `test_adaptive_epsilon.py` | Tests whether a residual-aware adaptive `causal_epsilon` schedule prevents causal-weight collapse, across 4 conditions × 4 PDEs (heat1d, advdiff1d, burgers1d, allencahn1d_sharp) |
| `validate_regret_bound.py` | Checks the probe-then-commit regret template (a diagnostic, not a theorem) on 13 PDEs × k∈{10,20,50,100}: computes ρ_k, σ_k and the bound from `paper_a/*` runs, and recomputes observed regret at every window as the regret of the `total_final` heuristic under PDE residual (`evaluate_policy_routing`, leave-one-PDE-out). `--output-dir` controls both the JSON and the figure (released at `paper_combined/regret_bound_v13/`) |

## For AI Agents

### Working In This Directory
- The `hypino_*`/`finetune_hypino_*`/`evaluate_hypino_*` scripts and `run_pinnacle_subset.py` depend on **external repositories not vendored in this codebase** (a HyPINO checkout + `.safetensors` weights, a PINNacle checkout) and take `--hypino-root`/`--weights`/`--pinnacle-root` CLI arguments accordingly — they cannot run standalone without those external checkouts present.
- `finetune_hypino_advdiff1d.py` and `finetune_hypino_heat1d.py` import shared helpers directly from `evaluate_hypino_heat1d_adapter.py` (`relative_l2`, `smape`, `build_heat1d_inputs`) via the same-directory `sys.path` auto-insertion — keep those signatures stable.
- `evaluate_meta_router.py` and `validate_regret_bound.py` read the same `experiments/results/paper_a/` and `experiments/results/paper_combined/routing_evaluation/results.json` inputs as `../paper_a_routing/` scripts — run `make fetch-data` first.
- Run with: `PYTHONPATH=src python experiments/scripts/paper_combined/<script>.py`, adding `CUDA_VISIBLE_DEVICES=0 --device cuda` for GPU-bound ones (`hypino_matched_budget.py`, `noisy_data_probe.py`, `test_adaptive_epsilon.py`).

### Testing Requirements
No pytest coverage. `make lint` covers this directory. HyPINO/PINNacle-dependent scripts cannot be smoke-tested without the external checkout — check argument parsing (`--help`) or a syntax/import check (`python -c "import ast; ast.parse(open('script.py').read())"`) style validation instead.

### Common Patterns
- Long-running sweep scripts (`hypino_matched_budget.py`, `noisy_data_probe.py`, `test_adaptive_epsilon.py`) write incremental `progress.json` plus a final `aggregate.json`, matching the pattern used in `../paper_c_causal/clamping_ablation.py` and `pretrain_budget_sweep.py`.
- Wall-clock matching (not FLOP matching) is the standard fairness protocol used when comparing against HyPINO, since exact FLOP-parity across a foundation-model hypernetwork and a from-scratch MLP isn't well-defined.
- Reviewer-response framing: several docstrings explicitly name which reviewer concern (e.g. "Reviewer 2 binding concern", "Reviewer 5 question") a script addresses — preserve that framing when extending a script since it maps directly to paper rebuttal text.

## Dependencies

### Internal
- `causal_pinn.models.hybrid` (`HybridConfig`, `HybridModelWrapper`), `causal_pinn.models.operators.FNO`, `causal_pinn.models.pinn.PINNBase`, `causal_pinn.models.pino.PINOModel`
- `causal_pinn.physics.pde` (`Heat1D`, `AdvectionDiffusion1D`, `Burgers1D`, `AllenCahn1D`), `causal_pinn.physics.losses` (`CausalLoss`, `PhysicsLoss`, `AdaptiveEpsilonScheduler`)
- `causal_pinn.training.trainer.HybridTrainer`, `causal_pinn.training.pino_trainer.PINOTrainer`, `causal_pinn.evaluation.benchmark_runner`, `causal_pinn.utils.random.set_seed`
- `experiment_utils.create_hybrid_model` (imported via same-directory `sys.path` insertion, from `../experiment_utils.py`)

### External
- PyTorch, NumPy, matplotlib, SciPy (`scipy.stats` in `validate_regret_bound.py`).
- scikit-learn (`RandomForestRegressor`, `GradientBoostingRegressor`) in `evaluate_meta_router.py`.
- HyPINO and PINNacle are external, non-pinned dependencies accessed via `--hypino-root`/`--pinnacle-root` paths, not installed packages.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
