# Causal-Aware Hybrid PINN-Operator Framework

This repository contains a research codebase for studying design choices in hybrid physics-informed neural networks (PINNs) and neural operators.

## Project Status

- Core framework implemented and tested
- Wave 0-2 experiment plan completed
- 1D and 2D experiment artifacts generated

## Main Findings So Far

The manuscript reports three main findings and two secondary contributions.

Main findings:

1. **Hybridization improves in-domain but not OOD.** Hybrid PINN-operator models reduce relative L2 error by Cohen's d = -3.1 to -8.3 across four PDEs in-domain, but the advantage weakens or disappears under distribution shift.
2. **Causal loss is conditional, and pretraining is the dominant factor.** Causal loss helps on 7/9 PDEs when preceded by operator pretraining (mean +5.9%), but hurts on 5/9 PDEs without pretraining (mean -4.2%). The mechanism is causal weight collapse, not gradient conflict.
3. **Early-dynamics routing is possible, but metric choice and regret matter.** The best nested PDE-aware residual router (5 selected training features + 7 PDE characteristics) achieves 84.6% policy-selection accuracy (11/13 held-out PDEs), while the full-feature RF+PDE variant reaches 61.5%. For K=3, E_p=50, and E_2=500, probe-then-commit saves 60% of Stage-2 policy-selection compute, excluding data generation and Stage-1 pretraining. A simple physics-loss heuristic matches the learned router in L2 accuracy but suffers ~77x worse regret.

Secondary contributions:

- Adaptive epsilon schedule mitigates causal weight collapse without pretraining.
- A probe-then-commit diagnostic regret check that holds on 44/52 PDE x probe-window pairs (84.6%; 11 of 13 PDEs at every window k in {10, 20, 50, 100}); it fails on two Burgers configurations, which keep rank transfer (one is a probe-spread calibration failure, the other is unexplained). It is presented as a heuristic diagnostic, not a formal guarantee.

## Environment

- Python 3.12
- PyTorch 2.5.1+cu121
- Primary hardware target: 4x NVIDIA TITAN Xp (12GB VRAM each)

## Repository Layout

- `src/causal_pinn/` - library code
- `tests/` - unit and integration tests
- `experiments/scripts/` - experiment entrypoints
- `experiments/configs/` - experiment configuration files
- `experiments/analysis/` - analysis output (generated table fragments, figures)
- `scripts/` - reproducibility helpers (`fetch_data.py`, `verify_claims.py`)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Benchmark Artifacts (Hugging Face)

The 3,545-run result corpus (~555 MB of JSONs, training histories, statistical
tests, and figures) is hosted on Hugging Face and **not** in this repository:

- Dataset: <https://huggingface.co/datasets/suanlab/PINNBench>
- Croissant metadata + claim-to-artifact manifest are bundled with the dataset.

Pull a local mirror once before running the verification pipeline:

```bash
make fetch-data       # downloads results/ from suanlab/PINNBench
```

The downloaded corpus is placed under `experiments/results/` and is ignored by
git so the source tree stays small.

## Verification

```bash
source .venv/bin/activate && pip install -e .[dev]   # one-time
make fetch-data                                       # one-time, ~555 MB
make test                                             # 390 tests, CPU-safe
make verify-claims                                    # recomputes 9 headline claims
make release-check                                    # test + lint + verify-claims
```

Known issue:

- `tests/unit/test_operators.py::TestFNO::test_fno_gradient_flow_through_spectral_conv` can fail on CPU due to an MKL FFT issue. Environment-specific; does not affect the CUDA experiment path.

## Reproducing Key Artifacts

### Wave 0

```bash
source .venv/bin/activate && PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python experiments/scripts/profile_wave2d_memory.py --output-dir experiments/results/wave0
source .venv/bin/activate && PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python experiments/scripts/rq2b_burgers_lambda_sweep.py --output-dir experiments/results/rq2b
```

### Wave 2 2D Experiments

```bash
source .venv/bin/activate && PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python experiments/scripts/rq4_wave2d_model_comparison.py --output-dir experiments/results/rq4_10seed --device cuda
source .venv/bin/activate && PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python experiments/scripts/rq5_wave2d_lambda_sweep.py --output-dir experiments/results/rq5_wave2d_lambda_sweep --device cuda
source .venv/bin/activate && PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python experiments/scripts/rq6_ood_parametric_generalization.py --output-dir experiments/results/rq6_ood --device cuda
```

### Statistical Analysis

```bash
source .venv/bin/activate && PYTHONPATH=src python experiments/scripts/statistical_analysis.py
source .venv/bin/activate && PYTHONPATH=src python experiments/scripts/generate_paper_tables.py
```

