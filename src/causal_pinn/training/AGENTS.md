<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# Training

## Purpose
Implements the 3-stage training pipeline for the Causal-Aware Hybrid PINN-Operator framework (`HybridTrainer` in `trainer.py`): (1) operator pretraining on synthetic data, (2) physics refinement with an increasing `lambda_physics` weight (optionally modulated by `AdaptiveLambdaController`), and (3) causal calibration using `AdaptiveCausalLoss`. Pretraining-as-stage-1 is a load-bearing research finding — the causal loss helps generalization *with* pretraining but hurts *without* it, so stage skip flags (`skip_stage1`/`skip_stage3` in `TrainingConfig`) exist specifically for ablation studies, not casual use. The module also houses the Paper A training-policy routing stack: early-training loss-curve feature extraction, a PDE-aware meta-router, budget-aware policy routing evaluation, and contextual-bandit/successive-halving baselines, all of which answer "given only the first N epochs of a run, which training policy should be used for the rest?".

## Key Files
| File | Description |
|------|-------------|
| `trainer.py` | `HybridTrainer` (3-stage pipeline) and `TrainingHistory`. Runs `stage1_pretrain` → `stage2_physics_refine` → `stage3_causal_calibrate`; stage 2 computes gradient-conflict metrics (`_compute_gradient_conflict`) between the data/BC objective and physics loss to feed the adaptive-lambda controller. |
| `pino_trainer.py` | `PINOTrainer` — analogous 3-stage trainer for `PINOModel` (pure operator architecture); stage 3 here still uses `lambda_physics` directly rather than a causal loss. |
| `adaptive_lambda.py` | `AdaptiveLambdaController` / `AdaptiveLambdaMetrics` — EMA-tracked gradient-cosine and norm-ratio signals that scale `lambda_physics` up or down when physics and data gradients conflict or align persistently. |
| `callbacks.py` | `Callback` base class (on_train/stage/epoch begin/end hooks) plus `CheckpointCallback`, `LoggingCallback`, `EarlyStoppingCallback`, `LRSchedulerCallback`, `GradientClippingCallback`. |
| `early_predictor.py` | `FeatureExtractor` (summary statistics — initial/final/mean/std/min/slope/ratio — over the first N epochs of stage-2 losses), `EarlyTrainingPredictor` (RandomForest by default, leave-one-PDE-out `cross_validate`), `PredictorDataset`/`PredictorSample`. |
| `meta_router.py` | `MetaRouter` and `PDEFeatures`/`PDE_FEATURE_REGISTRY` — augments early-training features with static, computable PDE characteristics (spatial dim, nonlinearity, derivative order, reaction term, domain volume) and fits a small `MLPRegressor` so the router generalizes to unseen PDE types. |
| `policy_routing.py` | Budget-aware routing analysis: `budget_ratio`/`budget_savings`, `evaluate_policy_routing` (compares RF, RF-with-fallback, Ridge, GradientBoosting, training-prior, and heuristic selectors via LOPO-CV), and `normalize_policy_name`/`normalize_pde_group` backed by `_POLICY_ALIASES`/`_STRICT_PDE_ALIASES`. |
| `contextual_bandit.py` | `LinUCBRouter` (LinUCB contextual bandit baseline over early-training feature context) and `SuccessiveHalvingRouter` (successive-halving budget-allocation baseline), both replaying recorded loss trajectories rather than launching new runs. |
| `__init__.py` | Package exports: `HybridTrainer`, `TrainingHistory`, `AdaptiveLambdaController`/`AdaptiveLambdaMetrics`, `EarlyTrainingPredictor`/`FeatureExtractor`/`PredictorDataset`/`PredictorSample`, `policy_routing` helpers, and all `Callback` subclasses. Note: `pino_trainer`, `meta_router`, and `contextual_bandit` are NOT re-exported here — import them directly from their submodules. |

## For AI Agents

### Working In This Directory
- Do not remove or bypass the stage-1 pretraining step without understanding it is a validated research finding, not incidental scaffolding — causal loss (stage 3) depends on it for the "helps with pretraining, hurts without" result.
- When adding new experiment result files or model/policy names, keep `_POLICY_ALIASES` and `_STRICT_PDE_ALIASES` in `policy_routing.py` in sync so name-normalization continues to reconcile experiment artifact names with canonical groups.
- New callbacks subclass `Callback` and override only the `on_*` hooks they need; `HybridTrainer._check_early_stopping` specifically looks for `EarlyStoppingCallback` instances, so a custom stopping callback must still be an `EarlyStoppingCallback` subclass (or update that check).
- `meta_router.PDE_FEATURE_REGISTRY` must be extended whenever a new PDE is added to the routing benchmark, or `MetaRouter` silently falls back to zero-valued PDE features for unknown names.

### Testing Requirements
Covered by `tests/unit/test_trainer.py` (HybridTrainer/TrainingHistory), `tests/unit/test_adaptive_lambda.py` (AdaptiveLambdaController), `tests/unit/test_early_predictor.py` (FeatureExtractor/EarlyTrainingPredictor), and `tests/unit/test_policy_routing.py` (budget_ratio, evaluate_policy_routing, normalization). Run with:
```
PYTHONPATH=src python -m pytest tests/unit/test_trainer.py -q
```
(substitute the other test files as needed for the module you touched).

### Common Patterns
- Trainers accept `callbacks: list[Callback] | None = None`, resolve `device` from an explicit arg, `config.device`, or fall back to CPU, and call `set_seed(config.seed)` in `__init__`.
- Per-epoch loop shape: `optimizer.zero_grad()` → compute per-component losses into a `dict[str, float]` → sum into `total_loss` → `backward()`/`step()` → `history.add_stageN_loss(losses)` → notify `on_epoch_end` callbacks → check early stopping.
- `TYPE_CHECKING`-guarded imports (e.g. `HybridTrainer` in `callbacks.py`, `PDEProblem`/`TrainingConfig` in `trainer.py`) are used to avoid circular imports; keep new cross-module type hints behind the same guard.
- Config/metrics objects are plain `@dataclass`es (`AdaptiveLambdaMetrics`, `PredictorSample`, `PredictorDataset`, `SelectorFoldResult`), several with `to_dict()`/`to_array()` convenience methods for serialization or model input.

## Dependencies

### Internal
- `causal_pinn.physics.losses` (`AdaptiveCausalLoss`, `BoundaryLoss`, `PhysicsLoss`)
- `causal_pinn.physics.pde.base.PDEProblem` (TYPE_CHECKING only)
- `causal_pinn.models.pino.PINOModel` (used by `pino_trainer.py`)
- `causal_pinn.utils.config.TrainingConfig` (TYPE_CHECKING only)
- `causal_pinn.utils.random.set_seed`

### External
- `torch` (autograd, optimizers, gradient-conflict computation)
- `numpy` (feature matrices, statistics)
- `scikit-learn` (`RandomForestRegressor`, `GradientBoostingRegressor`, `Ridge`, `MLPRegressor`, `StandardScaler`)

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
