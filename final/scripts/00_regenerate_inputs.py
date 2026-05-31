#!/usr/bin/env python3
"""
00_regenerate_inputs.py — Regenerate frame state, draft features, and v5 scores.

This is a wrapper that calls the existing processing scripts from
ProgresoActual/ and ProgresoActual2/ with the correct arguments to produce
full-dataset outputs (no --sample-frac).

Run this whenever the raw data changes (e.g. after collecting more matches).
After this completes, re-run 01_prepare_final_dataset.py.

Estimated time: ~3-5 hours total (step 1 is the bottleneck).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

STEPS = [
    {
        "name": "Step 1/3: Extract support frame state (~195k matches, SLOW)",
        "script": str(REPO_ROOT / "ProgresoActual" / "src" / "02_data_processing"
                      / "new_02a_extract_support_frame_state.py"),
        "args": [
            "--raw-root", str(REPO_ROOT / "data" / "raw" / "raw"),
            "--region", "europe",
            "--outdir", str(REPO_ROOT / "final" / "data" / "frame_state"),
            "--out-name", "support_frame_state",
            "--write-mode", "dataset",
            "--chunk-matches", "5000",
            "--overwrite-output",
        ],
        "cwd": str(REPO_ROOT / "ProgresoActual" / "src" / "02_data_processing"),
    },
    {
        "name": "Step 2/3: Build draft features (fast)",
        "script": str(REPO_ROOT / "ProgresoActual" / "src" / "02_data_processing"
                      / "build_draft_features.py"),
        "args": [
            "--raw-root", str(REPO_ROOT / "data" / "raw" / "raw"),
            "--region", "europe",
            "--outdir", str(REPO_ROOT / "final" / "data" / "features"),
            "--out-name", "draft_features",
        ],
        "cwd": str(REPO_ROOT / "ProgresoActual" / "src" / "02_data_processing"),
    },
    {
        "name": "Step 3/3: Build support scores v5 (moderate)",
        "script": str(REPO_ROOT / "ProgresoActual2" / "scripts"
                      / "build_support_roam_score_v5_distribution.py"),
        "args": [
            "--frame-state-path", str(REPO_ROOT / "final" / "data" / "frame_state"
                                      / "support_frame_state.parquet"),
            "--config", str(REPO_ROOT / "ProgresoActual2" / "data" / "geometry"
                           / "manual_geometry_v5_config.json"),
            "--outdir", str(REPO_ROOT / "final" / "analysis" / "label_health"),
            "--export-dir", str(REPO_ROOT / "final" / "data" / "scores"),
            "--export-scores",
        ],
        "cwd": str(REPO_ROOT / "ProgresoActual2" / "scripts"),
    },
]


def run_step(step: dict) -> None:
    print("\n" + "=" * 70)
    print(f"  {step['name']}")
    print("=" * 70)
    cmd = [PYTHON, step["script"]] + step["args"]
    print(f"  CMD: {' '.join(cmd[:3])} ...")
    print(f"  CWD: {step['cwd']}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=step["cwd"])
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        sys.exit(result.returncode)
    print(f"  Done in {elapsed:.1f}s")


def main() -> None:
    print("=" * 70)
    print("  00_regenerate_inputs.py")
    print("  Regenerating all preprocessing outputs for the final dataset.")
    print("  This may take several hours (frame state extraction is slow).")
    print("=" * 70)

    total_t0 = time.time()
    for step in STEPS:
        run_step(step)

    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 70)
    print(f"  ALL DONE in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print("  Next: python final/scripts/01_prepare_final_dataset.py")
    print("=" * 70)

    # Update the input paths in 01_prepare_final_dataset.py
    print("\n  NOTE: 01_prepare_final_dataset.py reads from ProgresoActual/ by default.")
    print("  To use the new outputs in final/data/, run:")
    print(f"    python final/scripts/01_prepare_final_dataset.py \\")
    print(f"      --draft-path final/data/features/draft_features.parquet \\")
    print(f"      --scores-path final/data/scores/support_scores_v5_geometry_m12.parquet")


if __name__ == "__main__":
    main()
