#!/usr/bin/env python3
"""
15_build_support_roam_score_v7_event_snapshots.py

Build a support roaming label where timeline events add extra spatial samples.

This is different from v6_events:
  - v6 added aggregate event counters on top of the v5 frame score.
  - v7 keeps the v5 score recipe but augments the spatial sample set.

The selected v7 label uses:
  base samples: the minute-level frames from 5..12
  extra samples: support-related event coordinates from 5..12

For support kill/assist and support death events, the Riot event coordinate is
treated as the support's active position at that moment. Distance to the ADC is
estimated using the ADC state from the nearest minute frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V5 = REPO_ROOT / "final" / "data" / "scores" / "support_scores_v5_geometry_m12.parquet"
DEFAULT_FRAME_STATE = REPO_ROOT / "final" / "data" / "frame_state" / "support_frame_state.parquet"
DEFAULT_EVENT_POSITIONS = REPO_ROOT / "final" / "data" / "event_context" / "support_event_positions_m12.parquet"
DEFAULT_OUTDIR = REPO_ROOT / "final" / "analysis" / "label_v7_event_snapshots"
DEFAULT_EXPORT_DIR = REPO_ROOT / "final" / "data" / "scores"

JOIN_KEYS = ["match_id", "team_id"]
V5_RAW = "raw_support_roam_score_v5_geometry"
V5_SCORE = "support_roam_score_v5_geometry"
RAW_SCORE_COL = "raw_support_roam_score_v7_event_snapshots"
SCORE_COL = "support_roam_score_v7_event_snapshots"

DEFAULT_EVENT_KINDS = ["support_kill_assist", "support_death"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build v7 support roam label with event snapshots.")
    p.add_argument("--v5-scores", default=str(DEFAULT_V5))
    p.add_argument("--frame-state", default=str(DEFAULT_FRAME_STATE))
    p.add_argument("--event-positions", default=str(DEFAULT_EVENT_POSITIONS))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    p.add_argument("--selected-out-name", default="support_scores_v7_event_snapshots_m12.parquet")
    p.add_argument("--start-minute", type=float, default=5.0)
    p.add_argument("--max-minute", type=float, default=12.0)
    p.add_argument("--far-adc-threshold", type=float, default=2500.0)
    p.add_argument("--w-outside", type=float, default=0.45)
    p.add_argument("--w-far", type=float, default=0.35)
    p.add_argument("--w-xp", type=float, default=0.20)
    p.add_argument("--gamma", type=float, default=0.75)
    p.add_argument("--event-kinds", nargs="+", default=DEFAULT_EVENT_KINDS)
    p.add_argument("--event-weight", type=float, default=1.0)
    p.add_argument("--export-scores", action="store_true")
    p.add_argument("--save-event-samples", action="store_true")
    return p.parse_args()


def clip01(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def weighted_average(parts: List[np.ndarray], weights: List[float]) -> np.ndarray:
    values = np.vstack(parts).T
    w = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values)
    num = np.nan_to_num(values, nan=0.0) @ w
    den = valid.astype(float) @ w
    return np.where(den > 0, num / den, np.nan)


def numeric_summary(df: pd.DataFrame, score_col: str) -> Dict[str, Any]:
    s = df[score_col].dropna()
    return {
        "n": int(s.shape[0]),
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
        "share_eq_1": float((s == 1).mean()),
    }


def load_nearest_adc_state(frame_state_path: Path, start_minute: float, max_minute: float) -> pd.DataFrame:
    cols = [
        "match_id", "team_id", "frame_idx", "minute",
        "adc_alive", "adc_x", "adc_y", "adc_in_base",
    ]
    frames = pd.read_parquet(frame_state_path, columns=cols)
    frames = frames[frames["minute"].between(start_minute, max_minute, inclusive="both")].copy()
    frames = frames.rename(
        columns={
            "frame_idx": "nearest_frame_idx",
            "minute": "nearest_frame_minute",
            "adc_alive": "nearest_adc_alive",
            "adc_x": "nearest_adc_x",
            "adc_y": "nearest_adc_y",
            "adc_in_base": "nearest_adc_in_base",
        }
    )
    return frames


def load_event_samples(
    event_positions_path: Path,
    frame_state_path: Path,
    start_minute: float,
    max_minute: float,
    event_kinds: Iterable[str],
    event_weight: float,
    far_adc_threshold: float,
) -> pd.DataFrame:
    events = pd.read_parquet(event_positions_path)
    events = events[
        events["minute"].between(start_minute, max_minute, inclusive="both")
        & events["event_kind"].isin(set(event_kinds))
    ].copy()
    if events.empty:
        return events

    events["nearest_frame_idx"] = events["minute"].round().astype(int)
    nearest_adc = load_nearest_adc_state(frame_state_path, start_minute, max_minute)
    events = events.merge(nearest_adc, on=JOIN_KEYS + ["nearest_frame_idx"], how="left")

    events["sample_type"] = "event_snapshot"
    events["sample_weight"] = float(event_weight)
    events["support_x_sample"] = events["x"].astype(float)
    events["support_y_sample"] = events["y"].astype(float)
    events["out_bot_context_event_v7"] = events["out_bot_context_v6"].fillna(False).astype(bool)
    events["valid_event_far_sample_v7"] = (
        events["nearest_adc_alive"].fillna(False)
        & events["nearest_adc_x"].notna()
        & events["nearest_adc_y"].notna()
        & ~events["nearest_adc_in_base"].fillna(False)
    )
    dx = events["support_x_sample"].astype(float) - events["nearest_adc_x"].astype(float)
    dy = events["support_y_sample"].astype(float) - events["nearest_adc_y"].astype(float)
    events["distance_to_nearest_adc_v7"] = np.sqrt(dx * dx + dy * dy)
    events["far_from_adc_event_v7"] = (
        events["valid_event_far_sample_v7"]
        & (events["distance_to_nearest_adc_v7"] >= far_adc_threshold)
    )
    return events


def aggregate_event_samples(events: pd.DataFrame) -> pd.DataFrame:
    base_cols = JOIN_KEYS + [
        "event_snapshot_samples_v7",
        "event_snapshot_weight_v7",
        "event_out_bot_weight_v7",
        "event_far_valid_weight_v7",
        "event_far_weight_v7",
        "support_kill_assist_event_samples_v7",
        "support_death_event_samples_v7",
    ]
    if events.empty:
        return pd.DataFrame(columns=base_cols)

    rows = []
    for (match_id, team_id), g in events.groupby(JOIN_KEYS, dropna=False):
        weight = g["sample_weight"].astype(float)
        far_valid = g["valid_event_far_sample_v7"].fillna(False)
        rows.append(
            {
                "match_id": match_id,
                "team_id": team_id,
                "event_snapshot_samples_v7": int(len(g)),
                "event_snapshot_weight_v7": float(weight.sum()),
                "event_out_bot_weight_v7": float(weight[g["out_bot_context_event_v7"].fillna(False)].sum()),
                "event_far_valid_weight_v7": float(weight[far_valid].sum()),
                "event_far_weight_v7": float(weight[g["far_from_adc_event_v7"].fillna(False)].sum()),
                "support_kill_assist_event_samples_v7": int((g["event_kind"] == "support_kill_assist").sum()),
                "support_death_event_samples_v7": int((g["event_kind"] == "support_death").sum()),
            }
        )
    return pd.DataFrame(rows)


def build_scores(
    v5: pd.DataFrame,
    event_agg: pd.DataFrame,
    weights: List[float],
    gamma: float,
) -> pd.DataFrame:
    out = v5.copy()
    out = out.merge(event_agg, on=JOIN_KEYS, how="left")
    event_cols = [
        "event_snapshot_samples_v7",
        "event_snapshot_weight_v7",
        "event_out_bot_weight_v7",
        "event_far_valid_weight_v7",
        "event_far_weight_v7",
        "support_kill_assist_event_samples_v7",
        "support_death_event_samples_v7",
    ]
    for col in event_cols:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].fillna(0.0)

    out["valid_support_samples_v7"] = (
        out["valid_support_frames_v5"].astype(float) + out["event_snapshot_weight_v7"].astype(float)
    )
    out["valid_coop_samples_v7"] = (
        out["valid_coop_frames_v5"].fillna(0).astype(float)
        + out["event_far_valid_weight_v7"].astype(float)
    )
    out["samples_out_bot_context_v7"] = (
        out["frames_out_bot_context_v5"].astype(float) + out["event_out_bot_weight_v7"].astype(float)
    )
    out["samples_far_from_adc_v7"] = (
        out["frames_far_from_adc_v5"].fillna(0).astype(float) + out["event_far_weight_v7"].astype(float)
    )

    out["outside_ratio_v7_event_snapshots"] = np.where(
        out["valid_support_samples_v7"] > 0,
        out["samples_out_bot_context_v7"] / out["valid_support_samples_v7"],
        np.nan,
    )
    out["far_ratio_v7_event_snapshots"] = np.where(
        out["valid_coop_samples_v7"] > 0,
        out["samples_far_from_adc_v7"] / out["valid_coop_samples_v7"],
        np.nan,
    )
    out["xp_gap_v7_event_snapshots"] = out["xp_gap_v5"]

    raw = weighted_average(
        [
            out["outside_ratio_v7_event_snapshots"].to_numpy(dtype=np.float64),
            out["far_ratio_v7_event_snapshots"].to_numpy(dtype=np.float64),
            out["xp_gap_v7_event_snapshots"].to_numpy(dtype=np.float64),
        ],
        weights,
    )
    out[RAW_SCORE_COL] = raw
    out[SCORE_COL] = np.power(clip01(raw), gamma)
    out["support_score_confidence_v7_event_snapshots"] = np.clip(
        out["valid_support_samples_v7"] / 8.0,
        0.0,
        1.0,
    )
    out["variant_id_v7"] = "v7_event_snapshots"
    out["variant_description_v7"] = (
        "v5 spatial recipe with support kill/assist and support death event coordinates as extra samples"
    )
    out["event_snapshot_share_v7"] = np.where(
        out["valid_support_samples_v7"] > 0,
        out["event_snapshot_weight_v7"] / out["valid_support_samples_v7"],
        0.0,
    )
    return out


def save_plots(scores: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(scores[V5_SCORE].dropna(), bins=50, range=(0, 1), alpha=0.45, label="v5 minute frames", color="#999999")
    ax.hist(scores[SCORE_COL].dropna(), bins=50, range=(0, 1), alpha=0.55, label="v7 event snapshots", color="#2f80ed")
    ax.set_title("Support roam score: v5 vs v7 event snapshots")
    ax.set_xlabel("score")
    ax.set_ylabel("match-team rows")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "support_roam_score_v5_vs_v7_event_snapshots_overlay.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(scores[V5_SCORE], scores[SCORE_COL], s=2, alpha=0.08, color="#2f80ed")
    ax.plot([0, 1], [0, 1], color="black", linewidth=1)
    ax.set_title("Row-level score relation: v5 vs v7 event snapshots")
    ax.set_xlabel("v5 score")
    ax.set_ylabel("v7 event snapshot score")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "support_roam_score_v5_vs_v7_event_snapshots_scatter.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    export_dir = Path(args.export_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    v5 = pd.read_parquet(args.v5_scores)
    events = load_event_samples(
        Path(args.event_positions),
        Path(args.frame_state),
        args.start_minute,
        args.max_minute,
        args.event_kinds,
        args.event_weight,
        args.far_adc_threshold,
    )
    event_agg = aggregate_event_samples(events)
    scores = build_scores(
        v5,
        event_agg,
        [args.w_outside, args.w_far, args.w_xp],
        args.gamma,
    )

    if args.save_event_samples:
        sample_cols = [
            "match_id", "team_id", "event_kind", "sample_type", "minute",
            "nearest_frame_idx", "nearest_frame_minute", "sample_weight",
            "support_x_sample", "support_y_sample", "nearest_adc_x", "nearest_adc_y",
            "zone_v6", "out_bot_context_event_v7",
            "valid_event_far_sample_v7", "distance_to_nearest_adc_v7", "far_from_adc_event_v7",
        ]
        sample_path = outdir / "support_event_snapshot_samples_m5_12.parquet"
        events[[c for c in sample_cols if c in events.columns]].sort_values(
            JOIN_KEYS + ["minute", "event_kind"]
        ).to_parquet(sample_path, index=False)
        print(f"[Saved] {sample_path.resolve()}")

    save_plots(scores, outdir)

    metadata = {
        "score_col": SCORE_COL,
        "raw_score_col": RAW_SCORE_COL,
        "source_v5_scores": str(Path(args.v5_scores).resolve()),
        "source_event_positions": str(Path(args.event_positions).resolve()),
        "source_frame_state": str(Path(args.frame_state).resolve()),
        "recipe": {
            "start_minute": args.start_minute,
            "max_minute": args.max_minute,
            "event_kinds": args.event_kinds,
            "event_weight": args.event_weight,
            "far_adc_threshold": args.far_adc_threshold,
            "w_outside": args.w_outside,
            "w_far": args.w_far,
            "w_xp": args.w_xp,
            "gamma": args.gamma,
            "note": "Event coordinates are extra spatial samples. Distance to ADC uses the nearest minute-frame ADC position.",
        },
        "event_samples": {
            "rows": int(len(events)),
            "match_team_keys": int(events[JOIN_KEYS].drop_duplicates().shape[0]) if not events.empty else 0,
            "by_kind": {str(k): int(v) for k, v in events["event_kind"].value_counts().items()} if not events.empty else {},
        },
        "v5_summary": numeric_summary(scores, V5_SCORE),
        "v7_summary": numeric_summary(scores, SCORE_COL),
        "row_corr_v5_v7": float(scores[V5_SCORE].corr(scores[SCORE_COL])),
        "mean_delta_v7_minus_v5": float((scores[SCORE_COL] - scores[V5_SCORE]).mean()),
    }
    (outdir / "support_roam_score_v7_event_snapshots_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    keep_cols = [
        "match_id", "team_id", "side", "patch",
        "support_champion_name", "adc_champion_name",
        "valid_support_frames_v5", "valid_coop_frames_v5",
        "outside_ratio_v5", "far_ratio_v5", "xp_gap_v5",
        "event_snapshot_samples_v7", "event_snapshot_weight_v7",
        "support_kill_assist_event_samples_v7", "support_death_event_samples_v7",
        "valid_support_samples_v7", "valid_coop_samples_v7",
        "samples_out_bot_context_v7", "samples_far_from_adc_v7",
        "outside_ratio_v7_event_snapshots", "far_ratio_v7_event_snapshots", "xp_gap_v7_event_snapshots",
        "event_snapshot_share_v7",
        "support_score_confidence_v5", "support_score_confidence_v7_event_snapshots",
        V5_RAW, V5_SCORE, RAW_SCORE_COL, SCORE_COL,
        "variant_id_v7", "variant_description_v7",
    ]
    selected = scores[[c for c in keep_cols if c in scores.columns]].sort_values(JOIN_KEYS)
    analysis_path = outdir / args.selected_out_name
    selected.to_parquet(analysis_path, index=False)
    print(f"[Saved] {analysis_path.resolve()}")

    if args.export_scores:
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir / args.selected_out_name
        selected.to_parquet(export_path, index=False)
        print(f"[Exported] {export_path.resolve()}")

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
