#!/usr/bin/env python3
"""
test_label_audit.py — Auditoría de la etiqueta geométrica v5.

Tests incluidos:
  1. Unit tests con DataFrame sintético (incluyendo far_ratio missing).
  2. Regression check sobre datos reales train/val/test: max_abs_delta == 0.
  3. Smoke test de metadata: start_minute / max_minute respetan los argumentos.
  4. Validaciones de cobertura/consistencia del chaos filter.

Uso:
    python final/scripts/test_label_audit.py
    python final/scripts/test_label_audit.py --verbose
    python final/scripts/test_label_audit.py --skip-parquet   # solo unit tests
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = REPO_ROOT / "final" / "data" / "training"
SCORES_PATH = (
    REPO_ROOT / "final" / "data" / "scores"
    / "support_scores_v5_geometry_m12.parquet"
)
CHAOS_SUMMARY_PATH = TRAINING_DIR / "chaos_filter_summary.json"

# ── Recipe constants (must match build script defaults) ───────────────────────

W_OUTSIDE = 0.45
W_FAR = 0.35
W_XP = 0.20
WEIGHTS = np.array([W_OUTSIDE, W_FAR, W_XP])
GAMMA = 0.75
XP_RATIO_MIN = 0.60
XP_RATIO_MAX = 1.00
MIN_FRAMES_CHAOS = 3   # threshold in 16_add_chaos_filter_weights.py

# ── Test harness ─────────────────────────────────────────────────────────────

PASS = "✓ PASS"
FAIL = "✗ FAIL"
_results: List[Tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    status = PASS if ok else FAIL
    _results.append((name, ok, detail))
    marker = "  " if ok else "  "
    print(f"  [{status}] {name}" + (f"\n          {detail}" if detail and not ok else ""))


def assert_close(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(a - b) <= tol


# ── Formula helpers (pure Python, no dependency on build script) ──────────────

def compute_raw_v5(
    outside: float | None,
    far: float | None,
    xp_gap: float | None,
    weights: np.ndarray = WEIGHTS,
) -> float:
    """Replicate the renormalized weighted average of v5 components."""
    vals = np.array([
        outside if outside is not None else np.nan,
        far if far is not None else np.nan,
        xp_gap if xp_gap is not None else np.nan,
    ], dtype=float)
    valid_mask = np.isfinite(vals).astype(float)
    den = (valid_mask * weights).sum()
    if den <= 0:
        return np.nan
    return float((np.nan_to_num(vals) * weights).sum() / den)


def apply_gamma(raw: float, gamma: float = GAMMA) -> float:
    return float(np.clip(raw, 0.0, 1.0) ** gamma)


# ── Vectorized version (mirrors build script implementation exactly) ──────────

def recompute_from_components(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return (raw_recomputed, score_recomputed) arrays from component columns."""
    components = df[["outside_ratio_v5", "far_ratio_v5", "xp_gap_v5"]].astype(float)
    valid_mask = components.notna().to_numpy(dtype=float)
    weighted_values = components.fillna(0.0).to_numpy(dtype=float) * WEIGHTS
    den = (valid_mask * WEIGHTS).sum(axis=1)
    raw = np.where(den > 0, weighted_values.sum(axis=1) / den, np.nan)
    score = np.clip(raw, 0.0, 1.0) ** GAMMA
    return raw, score


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Unit tests on synthetic DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def run_unit_tests(verbose: bool) -> None:
    print("\n── SECTION 1: Unit Tests (synthetic data) ───────────────────────────────")

    # --- 1.1  All components present, values known ---
    outside, far, xp = 0.8, 0.6, 0.4
    expected_raw = (0.45 * outside + 0.35 * far + 0.20 * xp) / 1.0
    expected_score = expected_raw ** GAMMA
    raw = compute_raw_v5(outside, far, xp)
    score = apply_gamma(raw)
    ok = assert_close(raw, expected_raw) and assert_close(score, expected_score)
    record(
        "1.1  All components present",
        ok,
        f"raw={raw:.6f} expected={expected_raw:.6f}  score={score:.6f} expected={expected_score:.6f}",
    )

    # --- 1.2  far_ratio missing → renormalize over outside + xp ---
    outside, far, xp = 0.5, None, 0.3
    w_sum = W_OUTSIDE + W_XP  # 0.65
    expected_raw = (W_OUTSIDE * outside + W_XP * xp) / w_sum
    raw = compute_raw_v5(outside, far, xp)
    ok = assert_close(raw, expected_raw)
    record(
        "1.2  far_ratio missing → renormalization",
        ok,
        f"raw={raw:.6f} expected={expected_raw:.6f}",
    )

    # --- 1.3  xp_gap missing → renormalize over outside + far ---
    outside, far, xp = 0.4, 0.7, None
    w_sum = W_OUTSIDE + W_FAR  # 0.80
    expected_raw = (W_OUTSIDE * outside + W_FAR * far) / w_sum
    raw = compute_raw_v5(outside, far, xp)
    ok = assert_close(raw, expected_raw)
    record(
        "1.3  xp_gap missing → renormalization",
        ok,
        f"raw={raw:.6f} expected={expected_raw:.6f}",
    )

    # --- 1.4  All components missing → nan ---
    raw = compute_raw_v5(None, None, None)
    ok = np.isnan(raw)
    record("1.4  All components missing → NaN", ok, f"raw={raw}")

    # --- 1.5  raw=0 → score=0 ---
    raw = compute_raw_v5(0.0, 0.0, 0.0)
    score = apply_gamma(raw)
    ok = assert_close(score, 0.0)
    record("1.5  raw=0 → score=0", ok, f"score={score}")

    # --- 1.6  raw=1 → score=1 ---
    raw = compute_raw_v5(1.0, 1.0, 1.0)
    score = apply_gamma(raw)
    ok = assert_close(score, 1.0)
    record("1.6  raw=1 → score=1", ok, f"score={score}")

    # --- 1.7  gamma compression: score > raw for raw in (0,1) ---
    raw = compute_raw_v5(0.5, 0.5, 0.5)
    score = apply_gamma(raw)
    ok = score > raw  # gamma < 1 compresses upward
    record("1.7  gamma<1 → score > raw (for raw in (0,1))", ok,
           f"raw={raw:.4f} score={score:.4f}")

    # --- 1.8  Vectorized implementation on synthetic df matches scalar ---
    syn = pd.DataFrame({
        "outside_ratio_v5": [0.8, 0.5, 0.4, 0.0],
        "far_ratio_v5":     [0.6, np.nan, 0.7, np.nan],
        "xp_gap_v5":        [0.4, 0.3, np.nan, np.nan],
    })
    raw_vec, score_vec = recompute_from_components(syn)
    scalar_raws = np.array([
        compute_raw_v5(r.outside_ratio_v5,
                       None if np.isnan(r.far_ratio_v5) else r.far_ratio_v5,
                       None if np.isnan(r.xp_gap_v5) else r.xp_gap_v5)
        for _, r in syn.iterrows()
    ])
    delta = np.abs(raw_vec - scalar_raws)
    ok = bool(np.nanmax(delta) < 1e-10)
    record("1.8  Vectorized == scalar on synthetic df", ok,
           f"max_delta={np.nanmax(delta):.2e}")

    # --- 1.9  xp_gap formula: high xp_ratio → low xp_gap (scout) ---
    # xp_gap = 1 - (xp_ratio - min) / (max - min); high xp_ratio → low gap
    xp_ratio_high = 0.95
    xp_gap_high = 1.0 - (xp_ratio_high - XP_RATIO_MIN) / (XP_RATIO_MAX - XP_RATIO_MIN)
    xp_ratio_low = 0.65
    xp_gap_low = 1.0 - (xp_ratio_low - XP_RATIO_MIN) / (XP_RATIO_MAX - XP_RATIO_MIN)
    ok = xp_gap_high < xp_gap_low
    record("1.9  High xp_ratio → lower xp_gap (support closer to ADC xp)",
           ok, f"xp_gap_high={xp_gap_high:.3f} xp_gap_low={xp_gap_low:.3f}")

    # --- 1.10  Weights sum to 1.0 ---
    ok = assert_close(WEIGHTS.sum(), 1.0)
    record("1.10 Default weights sum to 1.0", ok, f"sum={WEIGHTS.sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Regression checks on real parquets
# ─────────────────────────────────────────────────────────────────────────────

def run_regression_checks(verbose: bool) -> None:
    print("\n── SECTION 2: Regression Checks (real train/val/test parquets) ──────────")

    for split in ["train", "val", "test"]:
        path = TRAINING_DIR / f"{split}.parquet"
        if not path.exists():
            record(f"2.{split}.exists", False, f"File not found: {path}")
            continue

        df = pd.read_parquet(
            path,
            columns=[
                "outside_ratio_v5", "far_ratio_v5", "xp_gap_v5",
                "raw_support_roam_score_v5_geometry", "support_roam_score",
                "valid_support_frames_v5",
            ],
        )

        raw_rec, score_rec = recompute_from_components(df)
        stored_raw = df["raw_support_roam_score_v5_geometry"].to_numpy()
        stored_score = df["support_roam_score"].to_numpy()

        delta_raw = np.abs(raw_rec - stored_raw)
        delta_score = np.abs(score_rec - stored_score)

        max_d_raw = float(np.nanmax(delta_raw))
        max_d_score = float(np.nanmax(delta_score))
        n_rows = len(df)
        n_far_null = int(df["far_ratio_v5"].isna().sum())

        # Test raw
        ok_raw = max_d_raw == 0.0
        record(
            f"2.{split}.raw_formula_exact  ({n_rows:,} rows)",
            ok_raw,
            f"max_abs_delta_raw={max_d_raw:.2e}",
        )

        # Test score (gamma transform)
        ok_score = max_d_score == 0.0
        record(
            f"2.{split}.score_gamma_exact  ({n_rows:,} rows)",
            ok_score,
            f"max_abs_delta_score={max_d_score:.2e}",
        )

        # Far_ratio nulls: renormalization should still be exact
        if n_far_null > 0:
            null_mask = df["far_ratio_v5"].isna()
            sub = df[null_mask]
            w_renorm = np.array([W_OUTSIDE, W_XP]) / (W_OUTSIDE + W_XP)
            raw_renorm = (
                sub["outside_ratio_v5"].to_numpy() * w_renorm[0]
                + sub["xp_gap_v5"].to_numpy() * w_renorm[1]
            )
            delta_renorm = np.abs(
                raw_renorm - sub["raw_support_roam_score_v5_geometry"].to_numpy()
            )
            max_d_renorm = float(np.nanmax(delta_renorm)) if len(delta_renorm) else 0.0
            ok_renorm = max_d_renorm < 1e-10
            record(
                f"2.{split}.far_null_renorm  ({n_far_null} rows with null far_ratio)",
                ok_renorm,
                f"max_delta_renorm={max_d_renorm:.2e}",
            )

        # min_frames filter: all rows must have valid_support_frames_v5 >= MIN_FRAMES_CHAOS
        n_below_min = int((df["valid_support_frames_v5"] < MIN_FRAMES_CHAOS).sum())
        ok_frames = n_below_min == 0
        record(
            f"2.{split}.min_frames_filter  (>= {MIN_FRAMES_CHAOS})",
            ok_frames,
            f"rows below min_frames={n_below_min}",
        )

        # score in [0, 1]
        n_out_of_range = int(
            (~df["support_roam_score"].between(0.0, 1.0, inclusive="both")).sum()
        )
        ok_range = n_out_of_range == 0
        record(
            f"2.{split}.score_in_0_1_range",
            ok_range,
            f"out-of-range rows={n_out_of_range}",
        )

        if verbose:
            print(f"    [{split}] rows={n_rows:,}  far_null={n_far_null}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Metadata / scores parquet smoke test
# ─────────────────────────────────────────────────────────────────────────────

def run_metadata_checks(verbose: bool) -> None:
    print("\n── SECTION 3: Metadata Checks (scores parquet + chaos summary) ──────────")

    # 3.1 Scores parquet exists
    ok = SCORES_PATH.exists()
    record("3.1  scores parquet exists", ok, str(SCORES_PATH))
    if not ok:
        return

    scores = pd.read_parquet(SCORES_PATH)

    # 3.2  start_minute / max_minute present as columns
    for col in ["start_minute", "max_minute"]:
        ok = col in scores.columns
        record(f"3.2  '{col}' column present in scores parquet", ok)

    # 3.3  start_minute == 5.0, max_minute == 12.0 (default run values)
    if "start_minute" in scores.columns:
        uniq = scores["start_minute"].dropna().unique()
        ok = len(uniq) == 1 and float(uniq[0]) == 5.0
        record("3.3  start_minute == 5.0 (default)", ok, f"values={uniq}")
    if "max_minute" in scores.columns:
        uniq = scores["max_minute"].dropna().unique()
        ok = len(uniq) == 1 and float(uniq[0]) == 12.0
        record("3.4  max_minute == 12.0 (default)", ok, f"values={uniq}")

    # 3.5  Weights in parquet match recipe
    for col, expected in [("w_outside", W_OUTSIDE), ("w_far", W_FAR), ("w_xp", W_XP), ("gamma", GAMMA)]:
        if col in scores.columns:
            val = float(scores[col].dropna().unique()[0])
            ok = assert_close(val, expected)
            record(f"3.5  scores.{col} == {expected}", ok, f"actual={val}")
        else:
            record(f"3.5  scores.{col} present", False, "column absent")

    # 3.6  Chaos filter summary exists and documents min_frames consistently
    if CHAOS_SUMMARY_PATH.exists():
        import json
        summary = json.loads(CHAOS_SUMMARY_PATH.read_text(encoding="utf-8"))
        documented_min = summary.get("min_frames", None)
        ok = documented_min == MIN_FRAMES_CHAOS
        record(
            f"3.6  chaos_filter_summary min_frames == {MIN_FRAMES_CHAOS}",
            ok,
            f"documented={documented_min}",
        )
        # 3.7  chaos_flag_rules documented
        rules = summary.get("chaos_flag_rules", [])
        ok = len(rules) >= 3
        record(f"3.7  chaos_flag_rules documented ({len(rules)} rules)", ok)
    else:
        record("3.6  chaos_filter_summary.json exists", False, str(CHAOS_SUMMARY_PATH))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Smoke test — non-default start/max minute propagation
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test_metadata(verbose: bool) -> None:
    """
    Verify that compute_scores() correctly stores the start_minute / max_minute
    arguments it receives, rather than hardcoded values.
    Uses a tiny synthetic input so no actual frame-state files are needed.
    """
    print("\n── SECTION 4: Smoke Test — start/max minute metadata propagation ────────")

    # We import compute_scores directly from the build script
    build_script = REPO_ROOT / "ProgresoActual2" / "scripts" / "build_support_roam_score_v5_distribution.py"
    ok = build_script.exists()
    record("4.1  build script exists", ok, str(build_script))
    if not ok:
        return

    # Build a minimal synthetic 'spatial / coop / xp' dataset that would be
    # produced after add_v5_frame_flags.  We replicate just enough columns to
    # feed compute_scores.
    JOIN_KEYS = ["match_id", "team_id"]
    N = 10
    rng = np.random.default_rng(0)
    syn = pd.DataFrame({
        "match_id": ["m1"] * N,
        "team_id": [1] * N,
        "frame_idx": list(range(N)),
        "side": ["blue"] * N,
        "patch": ["14.10"] * N,
        "support_champion_name": ["Lulu"] * N,
        "adc_champion_name": ["Jinx"] * N,
        "support_alive": [True] * N,
        "adc_alive": [True] * N,
        "support_x": rng.uniform(0, 15000, N),
        "support_y": rng.uniform(0, 15000, N),
        "adc_x": rng.uniform(0, 15000, N),
        "adc_y": rng.uniform(0, 15000, N),
        "dist_to_adc": rng.uniform(0, 5000, N),
        "support_xp": rng.uniform(1000, 4000, N),
        "adc_xp": rng.uniform(1000, 4000, N),
        "support_in_base_v5": [False] * N,
        "adc_in_base_v5": [False] * N,
        "support_in_bot_context_v5": rng.choice([True, False], N),
    })

    # Import compute_scores
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("build_v5", build_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        record("4.2  build script importable", False, str(exc))
        return

    record("4.2  build script importable", True)

    # Test with non-default start/max minute
    TEST_START = 3.0
    TEST_MAX = 10.0

    try:
        result = module.compute_scores(
            df=syn,
            far_adc_threshold=2500.0,
            weights=WEIGHTS,
            gamma=GAMMA,
            xp_ratio_min=XP_RATIO_MIN,
            xp_ratio_max=XP_RATIO_MAX,
            min_support_frames=1,
            start_minute=TEST_START,
            max_minute=TEST_MAX,
        )

        ok_start = (result["start_minute"] == TEST_START).all()
        ok_max = (result["max_minute"] == TEST_MAX).all()
        record(
            f"4.3  start_minute stored as {TEST_START} (not hardcoded 5.0)",
            ok_start,
            f"values={result['start_minute'].unique()}",
        )
        record(
            f"4.4  max_minute stored as {TEST_MAX} (not hardcoded 12.0)",
            ok_max,
            f"values={result['max_minute'].unique()}",
        )

        # Check that score is in [0,1]
        score_col = module.SCORE_COL
        ok_range = result[score_col].between(0.0, 1.0, inclusive="both").all()
        record("4.5  smoke score in [0, 1]", ok_range)

    except Exception as exc:
        record("4.3  compute_scores executed without error", False, traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Chaos filter consistency
# ─────────────────────────────────────────────────────────────────────────────

def run_chaos_consistency(verbose: bool) -> None:
    print("\n── SECTION 5: Chaos Filter Consistency ─────────────────────────────────")

    for split in ["train", "val", "test"]:
        path = TRAINING_DIR / f"{split}.parquet"
        if not path.exists():
            record(f"5.{split}.exists", False, str(path))
            continue

        df = pd.read_parquet(
            path,
            columns=[
                "chaos_flag", "sample_weight", "valid_support_frames_v5",
                "support_deaths_0_12", "adc_deaths_0_12",
                "support_kill_assists_out_bot_0_12",
            ],
        )

        # 5.a  No row below MIN_FRAMES (already filtered by script)
        n_below = int((df["valid_support_frames_v5"] < MIN_FRAMES_CHAOS).sum())
        record(
            f"5.{split}.no_rows_below_min_frames ({MIN_FRAMES_CHAOS})",
            n_below == 0,
            f"found {n_below} rows",
        )

        # 5.b  sample_weight is either CHAOS_WEIGHT or CLEAN_WEIGHT
        CHAOS_WEIGHT = 0.2
        CLEAN_WEIGHT = 1.0
        valid_weights = df["sample_weight"].isin([CHAOS_WEIGHT, CLEAN_WEIGHT])
        n_invalid = int((~valid_weights).sum())
        record(
            f"5.{split}.sample_weight is 0.2 or 1.0",
            n_invalid == 0,
            f"invalid weight rows={n_invalid}",
        )

        # 5.c  chaos_flag == True iff sample_weight == CHAOS_WEIGHT
        mismatch = (df["chaos_flag"] != (df["sample_weight"] == CHAOS_WEIGHT)).sum()
        record(
            f"5.{split}.chaos_flag matches sample_weight",
            mismatch == 0,
            f"mismatch rows={mismatch}",
        )

        # 5.d  Chaos rate sanity: expect between 1% and 40%
        chaos_rate = float(df["chaos_flag"].mean())
        ok = 0.01 <= chaos_rate <= 0.40
        record(
            f"5.{split}.chaos_rate in [1%, 40%]",
            ok,
            f"chaos_rate={chaos_rate:.2%}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Label audit tests for v5 geometry.")
    p.add_argument("--verbose", action="store_true", help="Print extra detail for passing tests.")
    p.add_argument("--skip-parquet", action="store_true", help="Skip sections 2/5 (no parquet access).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("  LABEL AUDIT — support_roam_score_v5_geometry")
    print("=" * 70)

    run_unit_tests(args.verbose)
    run_smoke_test_metadata(args.verbose)

    if not args.skip_parquet:
        run_regression_checks(args.verbose)
        run_metadata_checks(args.verbose)
        run_chaos_consistency(args.verbose)
    else:
        print("\n  [INFO] Skipped sections 2/3/5 (--skip-parquet)")

    # Summary
    print("\n" + "=" * 70)
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    print(f"  TOTAL: {total}  |  PASS: {passed}  |  FAIL: {failed}")
    if failed:
        print(f"\n  FAILED TESTS:")
        for name, ok, detail in _results:
            if not ok:
                print(f"    [{FAIL}] {name}")
                if detail:
                    print(f"             {detail}")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
