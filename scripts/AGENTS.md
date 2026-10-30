<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-07-24 | Updated: 2026-07-24 -->

# scripts

## Purpose
Small reproducibility helper scripts for the public repo. These fetch the benchmark result corpus from Hugging Face and recompute the manuscript's headline claims directly from the released JSON artifacts, without needing to rerun any training. Both scripts are wired to `Makefile` targets so they're the standard entrypoint for verification rather than the larger per-experiment scripts under `experiments/scripts/`.

## Key Files
| File | Description |
|------|-------------|
| `fetch_data.py` | Downloads the ~397 MB `results/` prefix of the `suanlab/PINNBench` Hugging Face dataset into `experiments/results/`. Idempotent (skips if a `.hf_snapshot_complete` marker exists); pass `--force` to re-download. Requires `huggingface_hub`. |
| `audit_citations.py` | Checks every BibTeX entry against a primary record: Crossref by DOI (title, authors in order, print year, volume, pages; preprint/SSRN DOIs rejected), OpenReview by forum id, or the `url` page's own text (soft-404s fail). Exit 1 on any failure. Validated against a 12-case poisoned bib. Covers metadata only -- whether prose matches the cited work is a manual check (see `REFERENCE_AUDIT3.md`) |
| `verify_claims.py` | Recomputes every headline claim in `CLAIM_INVENTORY.md` directly from the JSON artifacts under `experiments/results/` and prints a PASS/FAIL summary. Makes no model calls, runs on CPU in seconds, and auto-fetches via `fetch_data.py` if `experiments/results/` is empty. Exit code 0 if all claims match within tolerance (default 2% relative), 1 otherwise. |

## For AI Agents

### Working In This Directory
- These scripts read released result JSONs; they do not train models or modify `dataset_pinnbench/` or `experiments/results/` contents (other than populating `experiments/results/` on first fetch).
- Run with `PYTHONPATH=src` set, since `verify_claims.py` imports from the `causal_pinn`-adjacent script namespace and repo root utilities.
- If you change a headline number in the paper or a results artifact schema, update `dataset_pinnbench/CLAIM_INVENTORY.md`/`MANIFEST.md` and keep `verify_claims.py`'s tolerance-based checks in sync — do not hand-edit JSON artifacts to make a check pass.

### Testing Requirements
```bash
make fetch-data       # python scripts/fetch_data.py
make verify-claims    # PYTHONPATH=src python scripts/verify_claims.py
python scripts/verify_claims.py --tol 0.02   # custom tolerance
```

### Common Patterns
- Both scripts are single-file, `argparse`-based CLIs with `if __name__ == "__main__": main()`.
- `REPO_ROOT = Path(__file__).resolve().parents[1]` is used to resolve paths relative to the repo root regardless of invocation directory.
- Prefer `sys.stderr.write(...)` for status/error messages over `print` when the message is diagnostic rather than primary output.

## Dependencies

### Internal
None from `src/causal_pinn/` directly; `verify_claims.py` imports `fetch_data._snapshot_from_hf` as a sibling-script fallback.

### External
`huggingface_hub` (snapshot download), standard library (`argparse`, `json`, `pathlib`).

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
