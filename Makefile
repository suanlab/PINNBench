# Repo-root Makefile -- single entrypoint for code-side reproduction and
# verification.  Paper drafts and submission packages are kept outside this
# public repository (Hugging Face dataset + anonymous mirror).
#
# Usage:
#   make fetch-data        -- download benchmark JSONs from suanlab/PINNBench
#   make test              -- run pytest with PYTHONPATH=src
#   make lint              -- ruff check
#   make typecheck         -- mypy (best-effort, non-blocking)
#   make verify-claims     -- recompute headline numbers (auto-fetches if needed)
#   make reproduce-smoke   -- print reproduction instructions
#   make release-check     -- test + lint + verify-claims + summary

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
RUFF   ?= $(PYTHON) -m ruff
MYPY   ?= $(PYTHON) -m mypy

.PHONY: help fetch-data test lint typecheck verify-claims \
        reproduce-smoke release-check

help:
	@echo "Targets: fetch-data, test, lint, typecheck, verify-claims,"
	@echo "         reproduce-smoke, release-check"

fetch-data:
	$(PYTHON) scripts/fetch_data.py

test:
	PYTHONPATH=src $(PYTEST) tests/ -q --tb=short

lint:
	$(RUFF) check src tests experiments

typecheck:
	-$(MYPY) src/causal_pinn/

verify-claims:
	PYTHONPATH=src $(PYTHON) scripts/verify_claims.py

reproduce-smoke:
	@echo "Smoke reproduction instructions:"
	@echo "  See README.md 'Verification' section."
	@echo "  Minimal smoke path (CPU-safe):"
	@echo "    make fetch-data       # one-time, ~555 MB from Hugging Face"
	@echo "    make test"
	@echo "    make verify-claims"
	@echo "  Full GPU reproduction (~40-50 GPU-hours):"
	@echo "    See experiments/scripts/ for per-claim entry points."

release-check: test lint verify-claims
	@echo ""
	@echo "================================================================"
	@echo "RELEASE CHECK SUMMARY"
	@echo "================================================================"
	@echo "PASS: pytest suite"
	@echo "PASS: ruff lint"
	@echo "PASS: verify-claims"
	@echo ""
	@echo "Benchmark dataset: https://huggingface.co/datasets/suanlab/PINNBench"
	@echo "================================================================"
