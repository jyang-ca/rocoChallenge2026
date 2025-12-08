#!/usr/bin/env python3
"""Scan HF HDF5 logs and keep only those with non-zero scores.

Logic:
- List files in a HF dataset repo (default: rocochallenge2025/rocochallenge2025).
- Download each .hdf5 into a cache dir.
- Inspect the "score" dataset; if any value is non-zero, keep and record stats.
- If all zeros (or score missing), delete the local file to save space.
- Stop after finding up to --max-keep files with non-zero scores.

Usage:
    python scripts/find_nonzero_scores.py
    python scripts/find_nonzero_scores.py --max-keep 5 --repo your/repo --local-dir data/cache
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

try:
    import h5py  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("h5py is required. pip install h5py") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find HDF5 files with non-zero scores.")
    p.add_argument("--repo", default="rocochallenge2025/rocochallenge2025", help="HF dataset repo id.")
    p.add_argument("--local-dir", default="data/cache", help="Local cache directory.")
    p.add_argument("--max-keep", type=int, default=10, help="Stop after keeping this many files with non-zero score.")
    p.add_argument("--summary", default="data/nonzero_scores_summary.json", help="Where to write the summary JSON.")
    p.add_argument("--pretend", action="store_true", help="Dry run: do not delete zero-score files.")
    return p.parse_args()


def list_hdf5_files(repo: str) -> List[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=repo, repo_type="dataset")
    return [f for f in files if f.lower().endswith(".hdf5")]


def prune_empty_dirs(path: Path, stop_at: Path) -> None:
    """Remove empty parent dirs up to stop_at (exclusive)."""
    cur = path.parent
    while cur != stop_at and cur != cur.parent:
        try:
            cur.rmdir()
        except OSError:
            break
        cur = cur.parent


def scan_file(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        if "score" not in f:
            return {"has_score": False, "nonzero": False, "count": 0, "min": None, "max": None}
        score = np.array(f["score"])
        nonzero = np.any(score != 0)
        return {
            "has_score": True,
            "nonzero": bool(nonzero),
            "count": int(score.size),
            "min": float(score.min()) if score.size else None,
            "max": float(score.max()) if score.size else None,
        }


def main() -> None:
    args = parse_args()
    cache_root = Path(args.local_dir).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    hdf5_files = list_hdf5_files(args.repo)
    print(f"Found {len(hdf5_files)} .hdf5 files in {args.repo}")

    kept = []
    removed = []
    missing_score = []

    for fname in hdf5_files:
        if len(kept) >= args.max_keep:
            break
        local_path = hf_hub_download(
            repo_id=args.repo,
            filename=fname,
            repo_type="dataset",
            local_dir=str(cache_root),
        )
        local_path = Path(local_path)
        info = scan_file(local_path)
        if not info["has_score"]:
            missing_score.append(fname)
        if info["has_score"] and info["nonzero"]:
            kept.append({"file": fname, "local_path": str(local_path), **info})
            print(f"KEEP {fname}: non-zero score (min={info['min']}, max={info['max']})")
        else:
            removed.append({"file": fname, "reason": "score_zero_or_missing", **info})
            print(f"DROP {fname}: score all zero or missing")
            if not args.pretend:
                try:
                    local_path.unlink()
                    prune_empty_dirs(local_path, cache_root)
                except OSError as exc:
                    print(f"Could not delete {local_path}: {exc}")

    summary = {
        "repo": args.repo,
        "scanned": len(kept) + len(removed),
        "kept": kept,
        "removed": removed,
        "missing_score": missing_score,
        "max_keep": args.max_keep,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
