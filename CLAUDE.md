# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase studying design choices in hybrid physics-informed neural networks (PINNs) and neural operators. The library is `causal_pinn` (installed as `causal-pinn`); everything else (`experiments/`, `scripts/`) drives experiments and reproduces the manuscript's headline claims. Two papers are cut from this repo: **Paper A** (early-dynamics routing / policy selection) and **Paper C** (causal-loss diagnosis). Experiment scripts and Hugging Face artifact names are grouped accordingly (`paper_a_routing/`, `paper_c_causal/`, `paper_combined/`).

## Commands

Requires Python 3.12. Always work inside the venv.

```bash
source .venv/bin/activate && pip install -e .[dev]   # one-time; add [experiment] for hydra/wandb
make test            # pytest, PYTHONPATH=src, ~390 CPU-safe tests
make lint            # ruff check src tests experiments
make typecheck       # mypy src/causal_pinn (best-effort, non-blocking)
make verify-claims   # recompute 9 headline claims from released JSONs (auto-fetches data)
make release-check   # test + lint + verify-claims
make fetch-data      # download ~555 MB result corpus into experiments/results/ (git-ignored)
```

Single test (note the mandatory `PYTHONPATH=src`):

```bash
PYTHONPATH=src python -m pytest tests/unit/test_hybrid.py::TestHybridPINNOperator::test_output_shape -q
```

**Running any experiment script requires `PYTHONPATH=src`** (pyproject sets `pythonpath=src` for pytest only, not for scripts), and scripts must be launched **from the repo root** — they locate the tree via `Path(__file__).resolve().parents[N]` and take relative `--output-dir`. GPU scripts also take `CUDA_VISIBLE_DEVICES` and `--device cuda`:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python experiments/scripts/rq4_wave2d_model_comparison.py --output-dir experiments/results/rq4_10seed --device cuda
```

Known flaky test: `tests/unit/test_operators.py::TestFNO::test_fno_gradient_flow_through_spectral_conv` can fail on CPU due to an MKL FFT issue; it is environment-specific and does not affect the CUDA path.

## Data / results boundary

Result JSONs, training histories, and figures (the 1,539-run corpus) are **not in git**. They live on Hugging Face (`suanlab/PINNBench`) and land in `experiments/results/` after `make fetch-data` (on this machine that path is a symlink to external storage, so it may already be populated). `scripts/verify_claims.py` auto-fetches if that directory is empty and recomputes headline numbers purely from JSON (no model calls). When adding or interpreting claims, the source of truth is the released artifacts under `experiments/results/`, not code.

Artifacts flow one way — **everything downstream of `experiments/results/` is generated, never hand-edited**:

```
experiments/scripts/**       ->  experiments/results/<run>/results.json (+ aggregate.json)
generate_paper_tables.py     ->  experiments/analysis/paper_tables.tex
statistical_analysis.py      ->  experiments/analysis/statistical_tests.json
paper_{a,c}_*/generate_figures.py -> docs/paper_{a_routing,c_causal}/figures/*.pdf
```

If a number in a table or figure looks wrong, fix the producing script and re-run it; do not patch the `.tex`/`.json`/`.pdf`.

### Verifying a claim

`scripts/verify_claims.py` is a flat list of check functions registered in the module-level `CHECKS` list of `(label, fn)` pairs. Each `fn(tol_rel)` loads JSON from `experiments/results/`, and returns `(status, computed, paper, note)` where status is `PASS` / `FAIL` / `PARTIAL` (structurally verified but not numerically comparable). Default tolerance is 2% relative (`--tol`). Adding a headline claim means adding a check function and registering it in `CHECKS` — a claim with no check is not considered released.

### Checking citations

`python scripts/audit_citations.py <references.bib>` verifies bibliographic metadata against primary records and must pass (0 fail) before a paper goes out. Every journal/proceedings entry needs a `doi`; entries without one need a `url` whose page actually carries the paper. The script cannot tell whether a sentence says what the cited paper says, and that is where the real errors were found: for any claim *about* another paper (a number in a positioning table, "X does not do Y"), open the cited text and find it. Search snippets and LLM-summarised pages are not evidence, and an empty `grep` over `pdftotext` output is not evidence of absence — those files often contain binary bytes that make grep print nothing; search them with Python or `grep -a`.

## Local-only files (git-ignored)

A large part of the working tree never reaches the public repo, and a fresh clone will not have it: `experiments/results/`, all paper drafts under `docs/paper*/` (LaTeX sources, figures, PDFs), the uppercase working notes (`AUDIT.md`, `CHECK.md`, `PLAN.md`, `MODIFY.md`, `REVIEW*.md`, `REVISION.md`, `REFERENCE_AUDIT*.md`, `findings.md`, `research-log.md`, `research-state.yaml`), plus `to_human/` (reviewer-response and handoff notes), `literature/`, `dataset_pinnbench/` (HF staging mirror), and `experiments/logs/`. Treat these as scratch/manuscript space: edit them freely, but never expect to commit them, and keep anything load-bearing for reproduction in code or in the released JSONs.

## Architecture

The central object is `HybridPINNOperator` (`src/causal_pinn/models/hybrid/hybrid_model.py`): a neural **operator** (FNO / DeepONet) for fast global structure, a **PINN** MLP for physics compliance, blended by a learnable **`CausalGate`** — `output = gate·operator + (1−gate)·pinn`, where the gate is biased toward the PINN at later times (`causal_strength`). `HybridConfig` flags (`use_operator`, `use_pinn`, `use_causal_gate`) exist to run ablations by disabling components.

Layered structure under `src/causal_pinn/`:

- **`models/`** — `operators/` (FNO, DeepONet, `NeuralOperator` base), `pinn/` (`PINNBase` MLP + solver), `pino/`, `hybrid/` (model + gate). All operators subclass `NeuralOperator`; all PINNs subclass `PINNBase`.
- **`physics/pde/`** — each PDE (heat, Burgers 1D/2D, advection-diffusion, KdV, Allen-Cahn, reaction-diffusion, wave 2D, Navier-Stokes 2D) subclasses `PDEProblem` and defines `residual(u, t, x)` + domain bounds. This residual is the single hook that couples a PDE to every loss and model.
- **`physics/losses/`** — `PhysicsLoss`, `BoundaryLoss`, `CausalLoss` (weights time `t_n` by `exp(-epsilon·Σ_{m<n} MSE(residual_m))`, enforcing earlier-time-first optimization), and `AdaptiveCausalLoss` / `adaptive_epsilon.py` (schedules epsilon to avoid causal-weight collapse). `CausalLoss` couples to PDEs via a structural `PDEProblemProtocol`, not the concrete base class.
- **`training/`** — `HybridTrainer` (`trainer.py`) runs the **3-stage pipeline**: (1) operator pretraining on synthetic data, (2) physics refinement with increasing `lambda_physics` (`adaptive_lambda.py`), (3) causal calibration with `AdaptiveCausalLoss`. Pretraining being stage 1 is a load-bearing finding: causal loss helps *with* pretraining and hurts without it.
- **`training/` routing stack** (Paper A) — `early_predictor.py` extracts early-training loss features; `meta_router.py` augments them with static `PDEFeatures` (spatial dim, nonlinearity, derivative order, …) via `PDE_FEATURE_REGISTRY` so the router generalizes to unseen PDEs; `policy_routing.py` does budget-aware probe-then-commit analysis; `contextual_bandit.py` is a baseline. Name normalization maps (`_POLICY_ALIASES`, `_STRICT_PDE_ALIASES` in `policy_routing.py`) reconcile experiment artifact names with canonical PDE/policy groups — keep these in sync when adding runs.
- **`evaluation/`** — `benchmark_runner.py` (`BenchmarkRunner` / `BenchmarkConfig`), `metrics.py` (relative L2, etc.), `physics_consistency.py`.

`experiments/scripts/experiment_utils.py` holds the factory functions (`create_pinn_model`, `create_hybrid_model`, `create_training_config`) and multi-seed runners (`run_single_experiment`, `run_multi_seed`) that all `rq*` scripts share — start here to understand how a script wires a PDE, model, and trainer together.

### Experiment script map

Top-level `experiments/scripts/rq*.py` are the original research-question scripts (the naming that the older result directories and README reproduction commands use). The three subdirectories are the paper cuts: `paper_a_routing/` (routing, holdout, regret, bandit baselines), `paper_c_causal/` (stage/clamping ablations, gradient-flow and causal-weight diagnostics), `paper_combined/` (cross-cutting work and external baselines — HyPINO, PINNacle, noise/sparsity, higher-dimensional probes).

`experiments/configs/*.yaml` document the canonical protocol but are **never read at runtime** — no script under `experiments/scripts/` references yaml/hydra/OmegaConf at all. Each script hardcodes its hyperparameters and echoes them into its output JSON's `config` field. Trust the JSON `config`, and cross-check the YAML rather than assuming it drove the run. There is no pytest coverage for `experiments/` — only `make lint` and the correctness of the library underneath it.

## Conventions

- Ruff line length 100, target py312, rules `E,F,W,I,UP,B,SIM`. Mypy runs with `disallow_untyped_defs = true` — new functions in `src/causal_pinn/` need type annotations.
- Seed control goes through `causal_pinn.utils.random.set_seed`; gradient helpers live in `utils/gradients.py`. Use these rather than reimplementing.
- Adding a PDE: subclass `PDEProblem`, implement `residual` + bounds, and register its static descriptors in `PDE_FEATURE_REGISTRY` (meta_router) if it should participate in routing.

## Per-directory notes

Most directories carry an `AGENTS.md` with a file-by-file breakdown (`src/causal_pinn/*/`, `experiments/`, `experiments/scripts/*/`, `tests/`, `scripts/`, `benchmarks/`, `docs/`). Read the nearest one before working in a subtree — it goes deeper than this file on local conventions and gotchas. They are generated documentation; if code moves, update the matching `AGENTS.md` too.
