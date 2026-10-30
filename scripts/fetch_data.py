#!/usr/bin/env python3
"""Download the PINNBench result artifacts from Hugging Face into ``experiments/results/``.

The benchmark result JSONs (~397 MB) are hosted on the public Hugging Face
dataset ``suanlab/PINNBench`` under the ``results/`` prefix.  This script
fetches that prefix into ``<repo_root>/experiments/results/`` so that
``scripts/verify_claims.py`` and the per-experiment scripts can read them
locally without any network access on subsequent runs.

Usage:
    python scripts/fetch_data.py            # idempotent; skips already-present files
    python scripts/fetch_data.py --force    # re-download even if files exist

Requires ``huggingface_hub`` (already a dev dependency).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ID = "suanlab/PINNBench"
HF_PREFIX = "results"
LOCAL_TARGET_NAME = "results"
REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = REPO_ROOT / "experiments" / LOCAL_TARGET_NAME


def _snapshot_from_hf(force: bool) -> Path:
    """Download the ``results/`` prefix and return the snapshot directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        sys.stderr.write(
            "huggingface_hub is not installed.  Install via "
            "`pip install huggingface_hub` and retry.\n"
        )
        raise SystemExit(1) from exc

    cache_marker = TARGET_DIR / ".hf_snapshot_complete"
    if cache_marker.exists() and not force:
        print(f"[fetch_data] {TARGET_DIR} already populated; skipping download")
        print("[fetch_data] re-run with --force to re-download")
        return TARGET_DIR

    print(f"[fetch_data] downloading {REPO_ID}/{HF_PREFIX} -> {TARGET_DIR}")
    snap = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=[f"{HF_PREFIX}/**"],
        )
    )
    src = snap / HF_PREFIX
    if not src.exists():
        raise SystemExit(f"snapshot did not contain expected prefix {HF_PREFIX}/")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        dest = TARGET_DIR / entry.name
        if dest.exists() and not force:
            continue
        if entry.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(entry, dest)
        else:
            shutil.copy2(entry, dest)

    cache_marker.touch()
    print(f"[fetch_data] done; snapshot mirrored under {TARGET_DIR}")
    return TARGET_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if .hf_snapshot_complete already exists",
    )
    args = parser.parse_args()
    _snapshot_from_hf(force=args.force)


if __name__ == "__main__":
    main()
