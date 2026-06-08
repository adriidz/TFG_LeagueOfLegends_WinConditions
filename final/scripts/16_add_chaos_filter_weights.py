#!/usr/bin/env python3
"""
16_add_chaos_filter_weights.py — Add chaos_flag and sample_weight to splits.

Reads the existing train/val/test splits, merges with the v5 score metadata
(valid_support_frames_v5) and event context (deaths, assists), computes a
chaos_flag for noisy botlane games, assigns sample_weight, and saves updated
splits.

Changes:
  - Rows with valid_support_frames_v5 < MIN_FRAMES are dropped.
  - chaos_flag is set for games with extreme botlane collapse.
  - sample_weight is 0.2 for chaotic games, 1.0 otherwise.
  - confidence_final is computed.

See final/docs/label_quality.md for the full rationale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


# ── Paths ───────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAINING_DIR = REPO_ROOT / "final" / "data" / "training"
DEFAULT_SCORES_PATH = (
    REPO_ROOT / "final" / "data" / "scores"
    / "support_scores_v5_geometry_m12.parquet"
)
DEFAULT_EVENT_CTX_PATH = (
    REPO_ROOT / "final" / "data" / "event_context"
    / "support_event_context_m12.parquet"
)

JOIN_KEYS = ["match_id", "team_id"]

# ── Thresholds ──────────────────────────────────────────────────────────────

MIN_FRAMES = 3
CHAOS_WEIGHT = 0.2
CLEAN_WEIGHT = 1.0
CHAOS_CONFIDENCE_PENALTY = 0.3


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add chaos_flag and sample_weight to training splits."
    )
    p.add_argument(
        "--training-dir", type=str, default=str(DEFAULT_TRAINING_DIR),
        help="Directory containing train.parquet, val.parquet, test.parquet.",
    )
    p.add_argument(
        "--scores-path", type=str, default=str(DEFAULT_SCORES_PATH),
        help="Path to v5 scores parquet (for valid_support_frames_v5).",
    )
    p.add_argument(
        "--event-ctx-path", type=str, default=str(DEFAULT_EVENT_CTX_PATH),
        help="Path to event context parquet (for death counts).",
    )
    p.add_argument(
        "--min-frames", type=int, default=MIN_FRAMES,
        help="Minimum valid_support_frames_v5 to keep a row.",
    )
    p.add_argument(
        "--chaos-weight", type=float, default=CHAOS_WEIGHT,
        help="sample_weight for chaotic observations.",
    )
    p.add_argument(
        "--backup", action="store_true", default=True,
        help="Backup original splits before overwriting.",
    )
    return p.parse_args()


# ── Chaos logic ─────────────────────────────────────────────────────────────

def compute_chaos_flag(df: pd.DataFrame) -> pd.Series:
    """
    Compute chaos_flag from event context columns.

    A game is flagged as chaotic if:
      - Combined bot deaths (support + ADC) >= 6, OR
      - ADC died 5+ times, OR
      - Support died 4+ times with zero active roaming events outside bot.
    """
    supp_deaths = df["support_deaths_0_12"].fillna(0)
    adc_deaths = df["adc_deaths_0_12"].fillna(0)
    active_out = df["support_kill_assists_out_bot_0_12"].fillna(0)

    flag = (
        ((supp_deaths + adc_deaths) >= 6)
        | (adc_deaths >= 5)
        | ((supp_deaths >= 4) & (active_out == 0))
    )
    return flag.astype(bool)


def compute_confidence_final(
    valid_frames: pd.Series,
    chaos_flag: pd.Series,
    penalty: float = CHAOS_CONFIDENCE_PENALTY,
) -> pd.Series:
    """Confidence penalized by chaos."""
    base_conf = (valid_frames / 6.0).clip(upper=1.0)
    return base_conf * (1.0 - penalty * chaos_flag.astype(float))


# ── Processing ──────────────────────────────────────────────────────────────

def load_enrichment_tables(
    scores_path: str, event_ctx_path: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the two enrichment sources."""

    print(f"[Load] scores: {scores_path}")
    scores = pd.read_parquet(
        scores_path,
        columns=JOIN_KEYS + [
            "valid_support_frames_v5",
            "valid_coop_frames_v5",
            "outside_ratio_v5",
            "far_ratio_v5",
            "xp_gap_v5",
        ],
    )
    print(f"       rows={len(scores):,}")

    print(f"[Load] event context: {event_ctx_path}")
    events = pd.read_parquet(
        event_ctx_path,
        columns=JOIN_KEYS + [
            "support_deaths_0_12",
            "adc_deaths_0_12",
            "support_kill_assists_out_bot_0_12",
            "support_kill_assists_bot_0_12",
            "support_active_events_out_bot_0_12",
            "botlane_deaths_bot_0_12",
        ],
    )
    print(f"       rows={len(events):,}")

    return scores, events


def process_split(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    events: pd.DataFrame,
    min_frames: int,
    chaos_weight: float,
    split_name: str,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Enrich a single split with chaos_flag, sample_weight, and filter by
    min_frames.
    """
    n_before = len(df)

    # Merge score metadata
    df = df.merge(scores, on=JOIN_KEYS, how="left")

    # Merge event context
    df = df.merge(events, on=JOIN_KEYS, how="left")

    # Check coverage
    n_missing_frames = int(df["valid_support_frames_v5"].isna().sum())
    n_missing_events = int(df["support_deaths_0_12"].isna().sum())
    if n_missing_frames > 0:
        print(f"  [WARN] {split_name}: {n_missing_frames} rows without "
              f"valid_support_frames_v5 - dropping them")
        df = df[df["valid_support_frames_v5"].notna()].copy()
    if n_missing_events > 0:
        print(f"  [WARN] {split_name}: {n_missing_events} rows without "
              f"event context - filling deaths with 0")
        for col in ["support_deaths_0_12", "adc_deaths_0_12",
                     "support_kill_assists_out_bot_0_12"]:
            df[col] = df[col].fillna(0)

    # Filter by min_frames
    n_low_frames = int((df["valid_support_frames_v5"] < min_frames).sum())
    df = df[df["valid_support_frames_v5"] >= min_frames].copy()

    # Compute chaos_flag
    df["chaos_flag"] = compute_chaos_flag(df)

    # Compute sample_weight
    df["sample_weight"] = np.where(
        df["chaos_flag"], chaos_weight, CLEAN_WEIGHT
    )

    # Compute confidence
    df["confidence_final"] = compute_confidence_final(
        df["valid_support_frames_v5"], df["chaos_flag"]
    )

    n_after = len(df)
    n_chaotic = int(df["chaos_flag"].sum())
    n_clean = n_after - n_chaotic

    stats = {
        "split": split_name,
        "rows_before": n_before,
        "rows_after": n_after,
        "rows_dropped_low_frames": n_low_frames,
        "rows_dropped_missing": n_before - n_after - n_low_frames
                                + (n_before - n_after - n_low_frames < 0) * (n_low_frames - n_before + n_after),
        "n_chaotic": n_chaotic,
        "n_clean": n_clean,
        "chaos_rate": float(n_chaotic / max(n_after, 1)),
        "effective_weight_sum": float(df["sample_weight"].sum()),
        "effective_weight_mean": float(df["sample_weight"].mean()),
        "score_mean_chaotic": float(
            df.loc[df["chaos_flag"], "support_roam_score"].mean()
        ) if n_chaotic > 0 else None,
        "score_mean_clean": float(
            df.loc[~df["chaos_flag"], "support_roam_score"].mean()
        ) if n_clean > 0 else None,
    }

    print(f"  [{split_name}] {n_before:,} -> {n_after:,} rows "
          f"(dropped {n_low_frames} low-frame rows)")
    print(f"           chaotic={n_chaotic:,} ({100*n_chaotic/max(n_after,1):.1f}%)  "
          f"clean={n_clean:,}")
    print(f"           score_mean: chaotic={stats['score_mean_chaotic']!r}  "
          f"clean={stats['score_mean_clean']!r}")

    return df, stats


def update_split_summary_json(training_dir: Path, df_splits: Dict[str, pd.DataFrame]) -> None:
    summary_path = training_dir / "split_summary.json"
    if not summary_path.exists():
        print("  [INFO] split_summary.json not found, skipping update.")
        return

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  [WARN] Failed to read split_summary.json ({exc}), skipping update.")
        return

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

    df_train = df_splits["train"]
    df_val = df_splits["val"]
    df_test = df_splits["test"]

    target_col = "support_roam_score"
    quantile_col = "support_roam_score_quantile"

    summary["split_sizes"] = {
        "train": len(df_train),
        "val": len(df_val),
        "test": len(df_test),
        "total": len(df_train) + len(df_val) + len(df_test),
    }
    summary["split_match_counts"] = {
        "train": int(df_train["match_id"].nunique()),
        "val": int(df_val["match_id"].nunique()),
        "test": int(df_test["match_id"].nunique()),
    }
    if target_col in df_train.columns:
        summary["target_stats_train"] = stats(df_train[target_col])
    if quantile_col in df_train.columns:
        summary["quantile_stats_train"] = stats(df_train[quantile_col])
    if target_col in df_val.columns:
        summary["target_stats_val"] = stats(df_val[target_col])
    if target_col in df_test.columns:
        summary["target_stats_test"] = stats(df_test[target_col])

    try:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [Saved] Updated {summary_path}")
    except Exception as exc:
        print(f"  [WARN] Failed to write updated split_summary.json ({exc})")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    training_dir = Path(args.training_dir)

    scores, events = load_enrichment_tables(
        args.scores_path, args.event_ctx_path
    )

    all_stats = []
    df_splits = {}

    for split_name in ["train", "val", "test"]:
        split_path = training_dir / f"{split_name}.parquet"
        print(f"\n[Process] {split_path}")

        df = pd.read_parquet(split_path)

        # Backup
        if args.backup:
            backup_path = training_dir / f"{split_name}_pre_chaos_filter.parquet"
            if not backup_path.exists():
                df.to_parquet(backup_path, index=False)
                print(f"  [Backup] -> {backup_path}")

        df_out, stats = process_split(
            df, scores, events,
            min_frames=args.min_frames,
            chaos_weight=args.chaos_weight,
            split_name=split_name,
        )

        # Save
        df_out.to_parquet(split_path, index=False)
        print(f"  [Saved] {split_path}")
        all_stats.append(stats)
        df_splits[split_name] = df_out

    # Update split_summary.json if it exists
    update_split_summary_json(training_dir, df_splits)

    # Summary
    summary = {
        "script": "16_add_chaos_filter_weights.py",
        "min_frames": args.min_frames,
        "chaos_weight": args.chaos_weight,
        "clean_weight": CLEAN_WEIGHT,
        "chaos_confidence_penalty": CHAOS_CONFIDENCE_PENALTY,
        "chaos_flag_rules": [
            "support_deaths_0_12 + adc_deaths_0_12 >= 6",
            "adc_deaths_0_12 >= 5",
            "support_deaths_0_12 >= 4 AND support_kill_assists_out_bot_0_12 == 0",
        ],
        "splits": all_stats,
    }

    summary_path = training_dir / "chaos_filter_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[Saved] {summary_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("CHAOS FILTER - RESUMEN")
    print("=" * 60)
    total_before = sum(s["rows_before"] for s in all_stats)
    total_after = sum(s["rows_after"] for s in all_stats)
    total_chaotic = sum(s["n_chaotic"] for s in all_stats)
    total_clean = sum(s["n_clean"] for s in all_stats)
    print(f"  Rows total:   {total_before:,} -> {total_after:,} "
          f"(dropped {total_before - total_after:,})")
    print(f"  Chaotic:      {total_chaotic:,} "
          f"({100*total_chaotic/max(total_after,1):.1f}%)")
    print(f"  Clean:        {total_clean:,}")
    print(f"  Min frames:   {args.min_frames}")
    print(f"  Chaos weight: {args.chaos_weight}")
    print("=" * 60)


if __name__ == "__main__":
    main()
