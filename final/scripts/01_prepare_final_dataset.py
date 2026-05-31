#!/usr/bin/env python3
"""
01_prepare_final_dataset.py — Prepare the final train/val/test splits.

Reads draft features and support scores v5, joins them, creates a 3-way split
by match_id (70/15/15), fits a QuantileTransformer on train only, and persists
everything to final/data/training/.

See final/docs/technical_spec.md (Script 01) for the full specification.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import QuantileTransformer


# ── Paths (relative to repo root) ──────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DRAFT_PATH = (
    REPO_ROOT / "ProgresoActual" / "data" / "clean" / "features"
    / "draft_features.parquet"
)
DEFAULT_SCORES_PATH = (
    REPO_ROOT / "ProgresoActual2" / "data" / "clean" / "scores"
    / "support_scores_v5_geometry_m12.parquet"
)
DEFAULT_OUT_DIR = REPO_ROOT / "final" / "data" / "training"

# ── Column names ────────────────────────────────────────────────────────────

JOIN_KEYS = ["match_id", "team_id"]
SCORE_SRC_COL = "support_roam_score_v5_geometry"
RAW_SCORE_COL = "raw_support_roam_score_v5_geometry"
TARGET_COL = "support_roam_score"                       # canonical target
QUANTILE_COL = "support_roam_score_quantile"            # quantile-transformed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build final train/val/test splits with quantile target."
    )
    p.add_argument("--draft-path", type=str, default=str(DEFAULT_DRAFT_PATH))
    p.add_argument("--scores-path", type=str, default=str(DEFAULT_SCORES_PATH))
    p.add_argument("--score-col", type=str, default=SCORE_SRC_COL)
    p.add_argument("--raw-score-col", type=str, default=RAW_SCORE_COL)
    p.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-quantiles", type=int, default=1000)
    return p.parse_args()


def load_and_join(draft_path: str, scores_path: str, score_col: str, raw_score_col: str) -> pd.DataFrame:
    """Inner join draft features with support scores on (match_id, team_id)."""

    print(f"[Load] draft features: {draft_path}")
    df_draft = pd.read_parquet(draft_path)
    print(f"       rows={len(df_draft):,}")

    print(f"[Load] support scores: {scores_path}")
    df_scores = pd.read_parquet(scores_path)
    print(f"       rows={len(df_scores):,}")

    # Keep only score columns + join keys + champion name for analysis
    score_cols_to_keep = [
        c for c in [score_col, raw_score_col, "support_champion_name"]
        if c in df_scores.columns
    ]
    df_scores_slim = df_scores[JOIN_KEYS + score_cols_to_keep].copy()

    df = df_draft.merge(df_scores_slim, on=JOIN_KEYS, how="inner")

    # Rename to canonical target column
    if score_col in df.columns:
        df = df.rename(columns={score_col: TARGET_COL})
    else:
        raise SystemExit(f"Missing score column: {score_col}")

    # Basic validation
    before = len(df)
    df = df[df[TARGET_COL].notna()].copy()
    df = df[df[TARGET_COL].between(0.0, 1.0, inclusive="both")].copy()
    dropped = before - len(df)
    if dropped:
        print(f"[Filter] dropped {dropped} rows with invalid target")

    print(f"[Join]  final rows={len(df):,}  "
          f"unique matches={df['match_id'].nunique():,}")
    return df


def three_way_split(
    df: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train/val/test by match_id: ~70 / 15 / 15.

    Step 1: separate 15% as test.
    Step 2: from the remaining 85%, separate ~17.6% as val (≈15% of total).
    """
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=0.15, random_state=seed
    )
    rest_idx, test_idx = next(gss_test.split(df, groups=df["match_id"]))
    df_rest = df.iloc[rest_idx].copy()
    df_test = df.iloc[test_idx].copy()

    # 0.15 / 0.85 ≈ 0.176 → val is ~15% of total
    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=0.176, random_state=seed
    )
    train_idx, val_idx = next(gss_val.split(df_rest, groups=df_rest["match_id"]))
    df_train = df_rest.iloc[train_idx].copy()
    df_val = df_rest.iloc[val_idx].copy()

    print(f"[Split] train={len(df_train):,}  val={len(df_val):,}  "
          f"test={len(df_test):,}")
    print(f"        train matches={df_train['match_id'].nunique():,}  "
          f"val matches={df_val['match_id'].nunique():,}  "
          f"test matches={df_test['match_id'].nunique():,}")

    # Sanity: no match leaks between splits
    train_matches = set(df_train["match_id"])
    val_matches = set(df_val["match_id"])
    test_matches = set(df_test["match_id"])
    assert train_matches.isdisjoint(val_matches), "Match leak: train ∩ val"
    assert train_matches.isdisjoint(test_matches), "Match leak: train ∩ test"
    assert val_matches.isdisjoint(test_matches), "Match leak: val ∩ test"
    print("        OK - no match leakage between splits")

    return df_train, df_val, df_test


def fit_quantile_on_train(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    n_quantiles: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Fit QuantileTransformer on train targets only. Apply to val and test.
    Zero-preserved variant: fit only on positive scores, keep score=0 as 0.
    """
    y_train = df_train[TARGET_COL].to_numpy(dtype=np.float64)

    # --- Zero-preserved quantile ---
    positive_mask_train = y_train > 0.0
    y_positive_train = y_train[positive_mask_train].reshape(-1, 1)

    n_q = min(n_quantiles, int(np.isfinite(y_positive_train).sum()))
    qt = QuantileTransformer(
        n_quantiles=max(1, n_q),
        output_distribution="uniform",
        random_state=seed,
        subsample=max(int(y_positive_train.shape[0]), 1),
    )
    qt.fit(y_positive_train)

    # Apply to each split
    for label, df_split in [("train", df_train), ("val", df_val), ("test", df_test)]:
        y = df_split[TARGET_COL].to_numpy(dtype=np.float64)
        q = np.zeros_like(y)
        pos = y > 0.0
        if pos.any():
            q[pos] = np.clip(
                qt.transform(y[pos].reshape(-1, 1)).ravel(), 0.0, 1.0
            )
        df_split[QUANTILE_COL] = q
        print(f"[Quantile] {label}: "
              f"mean={df_split[QUANTILE_COL].mean():.4f}  "
              f"median={df_split[QUANTILE_COL].median():.4f}  "
              f"zero_share={float((df_split[QUANTILE_COL] == 0).mean()):.4f}")

    transformer_info = {
        "type": "QuantileTransformer_zero_preserved",
        "n_quantiles_requested": n_quantiles,
        "n_quantiles_effective": n_q,
        "n_positive_train_samples": int(positive_mask_train.sum()),
        "seed": seed,
        "fitted_on": "train positives only",
    }
    return df_train, df_val, df_test, {"qt": qt, "info": transformer_info}


def build_summary(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    transformer_info: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Build a JSON-serializable summary of the final dataset."""

    def stats(s: pd.Series) -> Dict[str, float]:
        return {
            "n": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=0)),
            "min": float(s.min()),
            "q05": float(s.quantile(0.05)),
            "q25": float(s.quantile(0.25)),
            "median": float(s.quantile(0.50)),
            "q75": float(s.quantile(0.75)),
            "q95": float(s.quantile(0.95)),
            "max": float(s.max()),
            "share_eq_0": float((s == 0).mean()),
        }

    return {
        "draft_path": os.path.abspath(args.draft_path),
        "scores_path": os.path.abspath(args.scores_path),
        "score_col_source": args.score_col,
        "raw_score_col_source": args.raw_score_col,
        "seed": args.seed,
        "split_sizes": {
            "train": len(df_train),
            "val": len(df_val),
            "test": len(df_test),
            "total": len(df_train) + len(df_val) + len(df_test),
        },
        "split_match_counts": {
            "train": int(df_train["match_id"].nunique()),
            "val": int(df_val["match_id"].nunique()),
            "test": int(df_test["match_id"].nunique()),
        },
        "target_col": TARGET_COL,
        "quantile_col": QUANTILE_COL,
        "target_stats_train": stats(df_train[TARGET_COL]),
        "quantile_stats_train": stats(df_train[QUANTILE_COL]),
        "target_stats_val": stats(df_val[TARGET_COL]),
        "target_stats_test": stats(df_test[TARGET_COL]),
        "quantile_transformer": transformer_info,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1-2. Load and join
    df = load_and_join(args.draft_path, args.scores_path, args.score_col, args.raw_score_col)

    # 3-4. (renaming already done inside load_and_join)

    # 5. Three-way split
    df_train, df_val, df_test = three_way_split(df, args.seed)

    # 6-7. Quantile transform (fitted on train only)
    df_train, df_val, df_test, qt_artifacts = fit_quantile_on_train(
        df_train, df_val, df_test,
        n_quantiles=args.n_quantiles,
        seed=args.seed,
    )

    # 8. Save parquets
    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    test_path = out_dir / "test.parquet"

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)
    print(f"\n[Saved] {train_path}")
    print(f"[Saved] {val_path}")
    print(f"[Saved] {test_path}")

    # 9. Save transformer
    qt_path = out_dir / "quantile_transformer.joblib"
    joblib.dump(qt_artifacts["qt"], qt_path)
    print(f"[Saved] {qt_path}")

    # 10. Save summary
    summary = build_summary(
        df_train, df_val, df_test, qt_artifacts["info"], args
    )
    summary_path = out_dir / "split_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[Saved] {summary_path}")

    # Print key stats
    print("\n" + "=" * 60)
    print("DATASET FINAL - RESUMEN")
    print("=" * 60)
    s = summary["split_sizes"]
    print(f"  Total:  {s['total']:,} filas")
    print(f"  Train:  {s['train']:,}  ({100*s['train']/s['total']:.1f}%)")
    print(f"  Val:    {s['val']:,}  ({100*s['val']/s['total']:.1f}%)")
    print(f"  Test:   {s['test']:,}  ({100*s['test']/s['total']:.1f}%)")
    ts = summary["target_stats_train"]
    print(f"\n  Target (train): mean={ts['mean']:.4f}  "
          f"median={ts['median']:.4f}  std={ts['std']:.4f}")
    qs = summary["quantile_stats_train"]
    print(f"  Quantile (train): mean={qs['mean']:.4f}  "
          f"median={qs['median']:.4f}  std={qs['std']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
