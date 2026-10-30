<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# benchmarks

## Purpose
Placeholder location for external and synthetic benchmark definitions referenced by the research plan (e.g. PINNacle-style problems and synthetic PDE test suites). Both subdirectories are currently empty in this checkout — the actual benchmark evaluation logic and problem definitions used by the paper live under `experiments/` (see `experiments/scripts/paper_combined/run_pinnacle_subset.py` and `experiments/scripts/paper_a_routing/`) and `src/causal_pinn/physics/pde/`, not here.

## Subdirectories
| Directory | Purpose |
|-----------|---------|
| `pinnacle/` | Reserved for PINNacle external-benchmark definitions/assets; empty in this checkout |
| `synthetic/` | Reserved for synthetic PDE benchmark problem setups; empty in this checkout |

## For AI Agents

### Working In This Directory
- Check whether content actually belongs here before adding files — most benchmark/evaluation code in this repo currently lives under `experiments/scripts/paper_a_routing/`, `experiments/scripts/paper_combined/` (e.g. `run_pinnacle_subset.py`, `aggregate_external_probes.py`), and PDE definitions under `src/causal_pinn/physics/pde/`.
- If populating `pinnacle/` or `synthetic/`, keep external benchmark code (third-party problem definitions, adapters) separate from the paper's own experiment scripts to avoid duplicating logic that already exists in `experiments/`.

### Testing Requirements
No tests currently target this directory (it is empty). If code is added here, add corresponding tests under `tests/unit/` and run via `PYTHONPATH=src python -m pytest tests/ -q`.

### Common Patterns
None established yet — directory is currently empty.

## Dependencies

### Internal
None currently (empty directory).

### External
None currently (empty directory).

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
