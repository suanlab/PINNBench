<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# causal-pinn

## Purpose
Research codebase for the **Causal-Aware Hybrid PINN-Operator Framework** — studying design choices in hybrid physics-informed neural networks (PINNs) and neural operators for PDE solving. The installable library is `causal_pinn` (`src/causal_pinn/`); the rest of the tree drives experiments and reproduces the manuscript's headline claims. Two papers are cut from this repo: **Paper A** (early-dynamics routing / policy selection) and **Paper C** (causal-loss diagnosis). See `CLAUDE.md` for the condensed operating guide.

## Key Files
| File | Description |
|------|-------------|
| `CLAUDE.md` | Condensed guidance for AI agents (commands, architecture, conventions) |
| `README.md` | Human-facing project overview, findings, and reproduction steps |
| `pyproject.toml` | Package metadata, deps, ruff/mypy/pytest config (Python 3.12) |
| `Makefile` | Single entrypoint: `test`, `lint`, `typecheck`, `verify-claims`, `fetch-data`, `release-check` |
| `requirements-lock.txt` | Pinned dependency versions |
| `research-state.yaml`, `research-log.md`, `findings.md` | Research planning/log notes |

Note: several large uppercase markdown files (`AUDIT.md`, `CHECK.md`, `REVIEW*.md`, `REVISION.md`, `PLAN.md`, `MODIFY.md`, `REFERENCE_AUDIT*.md`) are internal working notes, not code documentation.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `src/` | The `causal_pinn` library (see `src/causal_pinn/AGENTS.md`) |
| `experiments/` | Experiment entrypoints, configs, analysis outputs (see `experiments/AGENTS.md`) |
| `tests/` | Unit + integration test suites (see `tests/AGENTS.md`) |
| `scripts/` | Reproducibility helpers: `fetch_data.py`, `verify_claims.py` (see `scripts/AGENTS.md`) |
| `benchmarks/` | External + synthetic benchmark definitions (see `benchmarks/AGENTS.md`) |
| `dataset_pinnbench/` | Local staging mirror of the Hugging Face result dataset (see `dataset_pinnbench/AGENTS.md`) |
| `docs/` | Per-paper draft material and figures (see `docs/AGENTS.md`) |
| `literature/`, `to_human/` | Reference notes and human-oriented handoff material (not code) |

## For AI Agents

### Working In This Directory
- Always operate inside the venv: `source .venv/bin/activate`, install with `pip install -e .[dev]`.
- **`PYTHONPATH=src` is mandatory for running any experiment script** — `pyproject.toml` sets `pythonpath=src` for pytest only, not for `python experiments/scripts/...`.
- Experiment result JSONs (~555 MB, 3,545 runs) are **not in git**. They live on Hugging Face (`suanlab/PINNBench`) and land in `experiments/results/` via `make fetch-data`.
- New code in `src/causal_pinn/` must be typed (mypy `disallow_untyped_defs = true`) and pass ruff (line length 100, rules `E,F,W,I,UP,B,SIM`).

### Testing Requirements
```bash
make test            # full suite, PYTHONPATH=src, ~390 CPU-safe tests
make release-check   # test + lint + verify-claims
```
Known flaky test on CPU (MKL FFT, environment-specific): `tests/unit/test_operators.py::TestFNO::test_fno_gradient_flow_through_spectral_conv`.

### Common Patterns
- The core object is `HybridPINNOperator` = neural operator + PINN blended by a learnable `CausalGate`.
- Training uses a 3-stage pipeline (operator pretraining → physics refinement → causal calibration); pretraining-as-stage-1 is a load-bearing research finding.
- Seed via `causal_pinn.utils.random.set_seed`; never reimplement seeding or autograd helpers.

## Dependencies

### External
- PyTorch (`torch>=2.0`, target 2.5.1+cu121) — models, autograd, training
- NumPy, SciPy, einops — numerics and tensor ops
- scikit-learn — routing models (RandomForest, GradientBoosting, MLPRegressor) in `training/`
- matplotlib — figures; optional `[experiment]` extras add hydra-core, omegaconf, wandb, tensorboard

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
