<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# experiments

## Purpose
Top-level harness for reproducing every empirical claim in the two papers cut from this repo (Paper A: early-dynamics routing/policy selection; Paper C: causal-loss diagnosis). This directory holds no library code — it holds entrypoint scripts that wire `causal_pinn` models/PDEs/trainers together, the YAML configs that pin the canonical training protocol, generated analysis artifacts (tables, statistics, figures), and the (git-ignored) logs and result JSONs those scripts produce.

## Key Files
| File | Description |
|------|-------------|
| `results` | The git-ignored result tree (~397 MB, 1,539+ runs), populated via `make fetch-data` from the `suanlab/PINNBench` Hugging Face dataset |

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `scripts/` | All experiment entrypoints: shared `experiment_utils.py`, the original `rq*.py` research-question scripts, and the `paper_a_routing/`, `paper_c_causal/`, `paper_combined/` script sets (see `scripts/AGENTS.md`) |
| `configs/` | YAML configs (`default.yaml`, `paper_combined.yaml`) recording the canonical model/training protocol (see `configs/AGENTS.md`) |
| `analysis/` | Generated LaTeX table fragments, statistical test outputs, and cross-model mechanism figures (see `analysis/AGENTS.md`) |
| `logs/` | Git-ignored stdout/stderr captures from long-running GPU experiment invocations, timestamped (e.g. `clamping_ablation_20260506_075727.log`) |
| `results/` | Git-ignored symlink target holding per-experiment `results.json` (and related) files; not committed, fetched via `make fetch-data` |

## For AI Agents

### Working In This Directory
- **`PYTHONPATH=src` is mandatory** for every script under `scripts/` — `pyproject.toml` only sets `pythonpath=src` for pytest, not for `python experiments/scripts/...` invocations.
- GPU-bound scripts additionally take `CUDA_VISIBLE_DEVICES=0` plus `--device cuda --output-dir experiments/results/...`. Canonical example:
  ```
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python experiments/scripts/rq4_wave2d_model_comparison.py --output-dir experiments/results/rq4_10seed --device cuda
  ```
- Before analyzing or re-deriving numbers, run `make fetch-data` so `experiments/results/` is populated (it is not in git).
- Always invoke scripts from the repo root so relative `--output-dir` paths and `Path(__file__).resolve().parents[N]` root-detection logic resolve correctly.

### Testing Requirements
There is no pytest suite for `experiments/`; these are research scripts, not unit-tested code. Verification options:
- `make lint` runs `ruff check src tests experiments`, which does cover this tree.
- For a cheap smoke check of a script, run it with `--device cpu` and a throwaway `--output-dir` (most scripts accept both flags).
- Correctness of the underlying framework is covered by `tests/` (see `../tests/AGENTS.md`) and `make verify-claims`, not by anything in this directory.

### Common Patterns
- `scripts/experiment_utils.py` is the shared factory/runner module (`create_pinn_model`, `create_hybrid_model`, `create_training_config`, `run_single_experiment`, `run_multi_seed`, `create_model_factory`) that most `rq*.py` scripts and some `paper_combined/` scripts import.
- Naming convention: top-level `rq*.py` scripts are the original research questions; `paper_a_routing/` = Paper A experiments; `paper_c_causal/` = Paper C experiments; `paper_combined/` = cross-cutting and external-baseline (HyPINO, PINNacle) experiments spanning both papers.
- Every script writes a `results.json` (or per-condition JSON files plus an `aggregate.json`) under `experiments/results/<name>/`; `experiments/analysis/` and `docs/paper_*/figures/` are derived from those JSONs, never hand-edited.
- `experiments/configs/*.yaml` document the protocol but are not auto-loaded by most scripts — each script hardcodes its own hyperparameters and records them in its output JSON's `config` field; cross-check against the YAML rather than assuming it's read at runtime.

## Dependencies

### Internal
- `causal_pinn` (the whole library, via `PYTHONPATH=src`) — models, PDEs, losses, trainer, evaluation, utils.

### External
- PyTorch, NumPy, SciPy, matplotlib — core numerics and plotting.
- scikit-learn — routing/ablation regressors (RandomForest, GradientBoosting, Ridge).
- Optional `[experiment]` extras (hydra-core, omegaconf, wandb, tensorboard) are declared in `pyproject.toml` but not exercised by most scripts here.

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
