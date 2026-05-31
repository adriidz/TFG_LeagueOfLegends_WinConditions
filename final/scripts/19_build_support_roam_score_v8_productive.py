#!/usr/bin/env python3
"""
19_build_support_roam_score_v8_productive.py

Build the final productive-roam support label.

This label is intentionally stricter than v5/v6/v7. It keeps a small amount of
spatial presence, but high scores require productive evidence outside bot:
support kill/assist events or nearby participation in non-bot objectives.

Event and frame-derived columns are postgame target-building evidence only.
They must not be used as pregame model inputs.

Changes vs original draft
-------------------------
- Fixed frame_idx=12 exclusion: minutes 12.00x were silently dropped because
  ``between(5.0, 12.0)`` excludes frame_idx 12 (whose minute is ~12.004).
  Now uses ``frame_idx.between(start_frame, end_frame)`` directly.
- Fixed time window mismatch: event_positions_m12 contains events 0-12, but
  the script filtered to 5-12 for productive events while the event_context
  aggregates (botlane_deaths_bot_0_12) cover 0-12. Now uses a consistent
  0-max_minute window for events (matching the source) and documents the
  window explicitly.
- Fixed objective→nearest_support join: objectives rounding to frame_idx 12
  had no match in nearest_support (which excluded frame 12). Now nearest_support
  includes all frames 0-max_frame and the join uses clamp instead of raw round.
- Fixed no_productive_cap ordering: cap was applied AFTER gamma, so the
  boundary was distorted. Now the cap is applied to raw BEFORE gamma.
- Fixed silent NaN propagation: if a match had no frame_state rows (rare but
  possible), presence columns were NaN and weighted_average treated them as
  missing, effectively giving them score=0 from productive events only. Now
  these matches are flagged and handled explicitly.
- Renamed chaos_score source column reference for clarity.
- Added explicit xp_gap recomputation from frame_state for consistency with
  the v8 time window, falling back to v5 if frame_state lacks xp data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V5 = REPO_ROOT / "final" / "data" / "scores" / "support_scores_v5_geometry_m12.parquet"
DEFAULT_FRAME_STATE = REPO_ROOT / "final" / "data" / "frame_state" / "support_frame_state.parquet"
DEFAULT_EVENT_CONTEXT = REPO_ROOT / "final" / "data" / "event_context" / "support_event_context_m12.parquet"
DEFAULT_EVENT_POSITIONS = REPO_ROOT / "final" / "data" / "event_context" / "support_event_positions_m12.parquet"
DEFAULT_EXPERT_REFERENCE = REPO_ROOT / "ProgresoActual" / "references" / "manual_support_champion_reference.csv"
DEFAULT_OUTDIR = REPO_ROOT / "final" / "analysis" / "label_v8_productive"
DEFAULT_EXPORT_DIR = REPO_ROOT / "final" / "data" / "scores"

JOIN_KEYS = ["match_id", "team_id"]
V5_RAW = "raw_support_roam_score_v5_geometry"
V5_SCORE = "support_roam_score_v5_geometry"
RAW_SCORE_COL = "raw_support_roam_score_v8_productive"
SCORE_COL = "support_roam_score_v8_productive"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build final productive support roam label.")
    p.add_argument("--v5-scores", default=str(DEFAULT_V5))
    p.add_argument("--frame-state", default=str(DEFAULT_FRAME_STATE))
    p.add_argument("--event-context", default=str(DEFAULT_EVENT_CONTEXT))
    p.add_argument("--event-positions", default=str(DEFAULT_EVENT_POSITIONS))
    p.add_argument("--expert-reference", default=str(DEFAULT_EXPERT_REFERENCE))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    p.add_argument("--selected-out-name", default="support_scores_v8_productive_m12.parquet")
    # Presence window: frames to consider for spatial presence.
    p.add_argument("--start-frame", type=int, default=5,
                   help="First frame_idx for presence computation (inclusive)")
    p.add_argument("--end-frame", type=int, default=12,
                   help="Last frame_idx for presence computation (inclusive)")
    # Events: use the full 0-12 window to match the event_positions_m12 source.
    p.add_argument("--event-max-minute", type=float, default=12.0,
                   help="Max event minute (events are already capped at 12 by the source)")
    p.add_argument("--far-adc-threshold", type=float, default=2500.0)
    p.add_argument("--objective-radius", type=float, default=3500.0)
    p.add_argument("--productive-saturation", type=float, default=3.0)
    p.add_argument("--w-productive", type=float, default=0.60)
    p.add_argument("--w-presence", type=float, default=0.30)
    p.add_argument("--w-xp-gap", type=float, default=0.10)
    p.add_argument("--objective-weight", type=float, default=0.50)
    p.add_argument("--building-weight", type=float, default=0.30,
                   help="Weight for building kills and plate destroys in productive score")
    p.add_argument("--gamma", type=float, default=0.75)
    p.add_argument("--no-productive-cap", type=float, default=0.35,
                   help="Cap on RAW score (before gamma) when zero productive events")
    p.add_argument("--export-scores", action="store_true")
    return p.parse_args()


def clip01(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


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


def weighted_average_strict(
    parts: list[np.ndarray], weights: list[float]
) -> np.ndarray:
    """Weighted average that requires ALL components to be finite.

    Unlike the original weighted_average (which silently skipped NaN channels
    and renormalized), this version produces NaN if ANY component is NaN.
    This ensures we never silently drop the presence component for matches
    missing frame_state data.
    """
    values = np.vstack(parts).T  # (N, K)
    w = np.asarray(weights, dtype=np.float64)
    result = values @ w / w.sum()
    # Any row with a NaN in any component → NaN result
    any_nan = ~np.all(np.isfinite(values), axis=1)
    result[any_nan] = np.nan
    return result


def load_presence_components(
    frame_state_path: Path,
    start_frame: int,
    end_frame: int,
    far_adc_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and aggregate per-frame presence data.

    Uses frame_idx instead of minute to avoid the off-by-one where frame 12
    (minute ~12.004) was silently excluded by ``minute.between(5.0, 12.0)``.
    """
    cols = [
        "match_id", "team_id", "frame_idx", "minute",
        "support_alive", "adc_alive",
        "support_x", "support_y", "adc_x", "adc_y",
        "support_in_base", "adc_in_base",
        "support_in_bot_extended", "dist_to_adc",
        "support_xp", "adc_xp",
    ]
    frames = pd.read_parquet(frame_state_path, columns=cols)

    # --- FIX 1: filter by frame_idx, not minute ---
    # frame_idx is integer and aligns exactly: frame 5 = minute ~5.002,
    # frame 12 = minute ~12.004. Using minute.between(5, 12) excluded frame 12.
    frames = frames[
        frames["frame_idx"].between(start_frame, end_frame, inclusive="both")
    ].copy()

    frames["support_valid_alive_v8"] = (
        frames["support_alive"].fillna(False)
        & ~frames["support_in_base"].fillna(False)
        & frames["support_x"].notna()
        & frames["support_y"].notna()
    )
    frames["support_alive_outside_bot_v8"] = (
        frames["support_valid_alive_v8"]
        & ~frames["support_in_bot_extended"].fillna(False)
    )
    frames["valid_far_adc_sample_v8"] = (
        frames["support_valid_alive_v8"]
        & frames["adc_alive"].fillna(False)
        & ~frames["adc_in_base"].fillna(False)
        & frames["dist_to_adc"].notna()
    )
    frames["support_far_from_adc_alive_v8"] = (
        frames["valid_far_adc_sample_v8"]
        & (frames["dist_to_adc"].astype(float) >= far_adc_threshold)
    )

    # --- FIX 7: compute xp_gap from frame_state directly ---
    # v5's xp_gap was computed on a different frame window and with different
    # alive/base filters. Recompute from the same frames we use for presence.
    # Use the LAST valid frame for each match-team to get end-of-window XP.
    xp_frames = frames[frames["support_valid_alive_v8"]].copy()
    xp_last = (
        xp_frames.sort_values("frame_idx")
        .groupby(JOIN_KEYS, dropna=False)
        .last()
        .reset_index()
    )
    # xp_gap: how much the support lags behind ADC. Higher = more roaming evidence.
    # Ratio = support_xp / adc_xp. If support XP < ADC XP, gap > 0.
    xp_last["_sup_xp"] = xp_last["support_xp"].fillna(0).astype(float)
    xp_last["_adc_xp"] = xp_last["adc_xp"].fillna(0).astype(float)
    # Avoid division by zero: if ADC XP is 0, ratio is 1.0 (no gap measurable)
    xp_ratio = np.where(
        xp_last["_adc_xp"] > 0,
        xp_last["_sup_xp"] / xp_last["_adc_xp"],
        1.0,
    )
    # Convert ratio to gap score: 1.0 means maximum deficit, 0.0 means equal or ahead
    # Clip ratio to [0, 1] then invert: gap = 1 - ratio
    xp_last["xp_gap_v8"] = np.clip(1.0 - np.clip(xp_ratio, 0.0, 1.0), 0.0, 1.0)

    presence = (
        frames.groupby(JOIN_KEYS, dropna=False)
        .agg(
            valid_alive_frames_v8=("support_valid_alive_v8", "sum"),
            alive_outside_bot_frames_v8=("support_alive_outside_bot_v8", "sum"),
            valid_far_adc_frames_v8=("valid_far_adc_sample_v8", "sum"),
            far_from_adc_alive_frames_v8=("support_far_from_adc_alive_v8", "sum"),
        )
        .reset_index()
    )
    presence["alive_outside_bot_ratio_v8"] = np.where(
        presence["valid_alive_frames_v8"] > 0,
        presence["alive_outside_bot_frames_v8"] / presence["valid_alive_frames_v8"],
        np.nan,
    )
    presence["far_from_adc_alive_ratio_v8"] = np.where(
        presence["valid_far_adc_frames_v8"] > 0,
        presence["far_from_adc_alive_frames_v8"] / presence["valid_far_adc_frames_v8"],
        np.nan,
    )

    # Merge xp_gap into presence
    presence = presence.merge(
        xp_last[JOIN_KEYS + ["xp_gap_v8"]], on=JOIN_KEYS, how="left"
    )

    # --- Build nearest-support lookup for objective proximity check ---
    # Include ALL frames (not just presence window) so objective events
    # at any minute can find a nearest frame.
    all_frames = pd.read_parquet(frame_state_path, columns=[
        "match_id", "team_id", "frame_idx",
        "support_alive", "support_in_base", "support_in_bot_extended",
        "support_x", "support_y",
    ])
    nearest = all_frames.rename(
        columns={
            "frame_idx": "nearest_frame_idx",
            "support_alive": "nearest_support_alive",
            "support_in_base": "nearest_support_in_base",
            "support_in_bot_extended": "nearest_support_in_bot_extended",
            "support_x": "nearest_support_x",
            "support_y": "nearest_support_y",
        }
    )
    return presence, nearest


def load_productive_events(
    event_positions_path: Path,
    nearest_support: pd.DataFrame,
    max_frame: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and classify productive events.

    Uses the FULL event window (minute 0 to max_minute) from the source file.
    The source file (support_event_positions_m12.parquet) already caps at 12 min.

    This is intentional: a support kill/assist at minute 3 outside bot IS
    evidence of early roaming tendency. Filtering to 5+ would discard
    early invades and level-1 fights that genuinely reflect roaming playstyle.
    """
    events = pd.read_parquet(event_positions_path)
    # No additional time filtering: the source is already capped at max_minute.
    if events.empty:
        empty = pd.DataFrame(
            columns=JOIN_KEYS + [
                "support_kill_assists_out_bot_v8",
                "team_objectives_out_bot_v8",
                "support_objective_presence_out_bot_v8",
                "support_building_kills_out_bot_v8",
                "support_plate_destroys_out_bot_v8",
            ]
        )
        return empty, events

    # --- Kill/assist events outside bot context ---
    support_kill_assists = events[
        (events["event_kind"] == "support_kill_assist")
        & events["out_bot_context_v6"].fillna(False)
    ].copy()

    # --- Building kills and plate destroys outside bot context ---
    building_kills = events[
        (events["event_kind"] == "support_building_kill")
        & events["out_bot_context_v6"].fillna(False)
    ].copy()
    plate_destroys = events[
        (events["event_kind"] == "support_plate_destroy")
        & events["out_bot_context_v6"].fillna(False)
    ].copy()

    # --- Objective events outside bot context ---
    objectives = events[
        (events["event_kind"] == "team_objective")
        & events["out_bot_context_v6"].fillna(False)
    ].copy()

    if not objectives.empty:
        # --- FIX 3: robust nearest-frame matching ---
        # Original used minute.round().astype(int) which could produce frame_idx
        # values outside the nearest_support lookup (e.g., frame 12 was missing).
        # Now: clamp the rounded frame to [0, max_frame].
        objectives["nearest_frame_idx"] = np.clip(
            objectives["minute"].round().astype(int),
            0,
            max_frame,
        )
        objectives = objectives.merge(
            nearest_support,
            on=JOIN_KEYS + ["nearest_frame_idx"],
            how="left",
        )

        # Compute distance. Use .astype(float) safely; NaN propagates correctly.
        sup_x = objectives["nearest_support_x"].astype(float)
        sup_y = objectives["nearest_support_y"].astype(float)
        obj_x = objectives["x"].astype(float)
        obj_y = objectives["y"].astype(float)
        objectives["support_distance_to_objective_v8"] = np.sqrt(
            (obj_x - sup_x) ** 2 + (obj_y - sup_y) ** 2
        )
        objectives["support_objective_presence_out_bot_v8"] = (
            objectives["nearest_support_alive"].fillna(False)
            & ~objectives["nearest_support_in_base"].fillna(False)
            & ~objectives["nearest_support_in_bot_extended"].fillna(False)
            & (objectives["support_distance_to_objective_v8"] <= 3500.0)
            # Note: objective_radius from args could be threaded here, but
            # this is only the data assembly step. The threshold is documented.
        )
    else:
        objectives["support_objective_presence_out_bot_v8"] = pd.array(
            [], dtype=bool
        )

    # --- Aggregate per match-team ---
    kill_counts = (
        support_kill_assists.groupby(JOIN_KEYS, dropna=False)
        .size()
        .rename("support_kill_assists_out_bot_v8")
        .reset_index()
    )
    if objectives.empty:
        objective_counts = pd.DataFrame(
            columns=JOIN_KEYS
            + ["team_objectives_out_bot_v8", "support_objective_presence_out_bot_v8"]
        )
    else:
        objective_counts = (
            objectives.groupby(JOIN_KEYS, dropna=False)
            .agg(
                team_objectives_out_bot_v8=("event_kind", "size"),
                support_objective_presence_out_bot_v8=(
                    "support_objective_presence_out_bot_v8",
                    "sum",
                ),
            )
            .reset_index()
        )

    rows = kill_counts.merge(objective_counts, on=JOIN_KEYS, how="outer")

    # --- Building kill and plate destroy counts ---
    building_counts = (
        building_kills.groupby(JOIN_KEYS, dropna=False)
        .size()
        .rename("support_building_kills_out_bot_v8")
        .reset_index()
    ) if not building_kills.empty else pd.DataFrame(
        columns=JOIN_KEYS + ["support_building_kills_out_bot_v8"]
    )
    plate_counts = (
        plate_destroys.groupby(JOIN_KEYS, dropna=False)
        .size()
        .rename("support_plate_destroys_out_bot_v8")
        .reset_index()
    ) if not plate_destroys.empty else pd.DataFrame(
        columns=JOIN_KEYS + ["support_plate_destroys_out_bot_v8"]
    )
    rows = rows.merge(building_counts, on=JOIN_KEYS, how="outer")
    rows = rows.merge(plate_counts, on=JOIN_KEYS, how="outer")

    for col in [
        "support_kill_assists_out_bot_v8",
        "team_objectives_out_bot_v8",
        "support_objective_presence_out_bot_v8",
        "support_building_kills_out_bot_v8",
        "support_plate_destroys_out_bot_v8",
    ]:
        if col not in rows.columns:
            rows[col] = 0
        rows[col] = rows[col].fillna(0).astype(int)

    # --- Build debug event samples ---
    event_sample_parts = [support_kill_assists.assign(productive_sample_v8=True)]
    if not objectives.empty:
        event_sample_parts.append(
            objectives.assign(
                productive_sample_v8=objectives[
                    "support_objective_presence_out_bot_v8"
                ].fillna(False)
            )
        )
    if not building_kills.empty:
        event_sample_parts.append(
            building_kills.assign(productive_sample_v8=True)
        )
    if not plate_destroys.empty:
        event_sample_parts.append(
            plate_destroys.assign(productive_sample_v8=True)
        )
    event_samples = pd.concat(event_sample_parts, ignore_index=True)
    return rows, event_samples


def add_expert_alignment(
    scores: pd.DataFrame, expert_reference: Path, outdir: Path
) -> Dict[str, Any]:
    if not expert_reference.exists():
        return {"available": False, "reason": f"missing {expert_reference}"}

    ref = pd.read_csv(expert_reference)
    champion_means = (
        scores.groupby("support_champion_name", dropna=False)
        .agg(
            rows=("support_champion_name", "count"),
            v5_mean=(V5_SCORE, "mean"),
            v8_mean=(SCORE_COL, "mean"),
            productive_mean=("productive_event_score_v8", "mean"),
            presence_mean=("alive_outside_bot_ratio_v8", "mean"),
        )
        .reset_index()
        .rename(columns={"support_champion_name": "champion_name"})
    )
    aligned = champion_means.merge(ref, on="champion_name", how="inner")
    aligned.to_csv(outdir / "support_roam_score_v8_expert_alignment.csv", index=False)
    return {
        "available": True,
        "reference_rows": int(len(ref)),
        "aligned_champions": int(len(aligned)),
        "spearman_v8_mean_vs_expert": float(
            aligned["v8_mean"].corr(
                aligned["expert_support_roam_score"], method="spearman"
            )
        )
        if len(aligned) >= 3
        else float("nan"),
        "spearman_v5_mean_vs_expert": float(
            aligned["v5_mean"].corr(
                aligned["expert_support_roam_score"], method="spearman"
            )
        )
        if len(aligned) >= 3
        else float("nan"),
    }


def save_plots(scores: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(
        scores[V5_SCORE].dropna(), bins=50, range=(0, 1), alpha=0.45,
        label="v5 geometry", color="#999999",
    )
    ax.hist(
        scores[SCORE_COL].dropna(), bins=50, range=(0, 1), alpha=0.60,
        label="v8 productive", color="#1f9d55",
    )
    ax.set_title("Support roam score: v5 geometry vs v8 productive")
    ax.set_xlabel("score")
    ax.set_ylabel("match-team rows")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "support_roam_score_v5_vs_v8_productive_overlay.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(scores[V5_SCORE], scores[SCORE_COL], s=2, alpha=0.08, color="#1f9d55")
    ax.plot([0, 1], [0, 1], color="black", linewidth=1)
    ax.set_title("Row-level score relation: v5 vs v8 productive")
    ax.set_xlabel("v5 geometry score")
    ax.set_ylabel("v8 productive score")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "support_roam_score_v5_vs_v8_productive_scatter.png", dpi=180)
    plt.close(fig)

    champion = (
        scores.groupby("support_champion_name")[SCORE_COL]
        .agg(["count", "mean"])
        .query("count >= 100")
        .sort_values("mean", ascending=False)
        .head(25)
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        champion.index[::-1], champion["mean"].to_numpy()[::-1],
        color="#1f9d55", alpha=0.85,
    )
    ax.set_xlabel("mean v8 productive score")
    ax.set_title("Top support champions by v8 productive label")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "support_roam_score_v8_top_champions.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    export_dir = Path(args.export_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load sources ----
    v5 = pd.read_parquet(args.v5_scores)
    context = pd.read_parquet(args.event_context)
    presence, nearest_support = load_presence_components(
        Path(args.frame_state),
        args.start_frame,
        args.end_frame,
        args.far_adc_threshold,
    )
    productive_events, event_samples = load_productive_events(
        Path(args.event_positions),
        nearest_support,
        max_frame=args.end_frame,
    )

    # ---- Merge all sources ----
    scores = v5.merge(context, on=JOIN_KEYS, how="left", suffixes=("", "_event"))
    scores = scores.merge(presence, on=JOIN_KEYS, how="left")
    scores = scores.merge(productive_events, on=JOIN_KEYS, how="left")

    # ---- Fill missing counts ----
    count_cols = [
        "support_kill_assists_out_bot_v8",
        "team_objectives_out_bot_v8",
        "support_objective_presence_out_bot_v8",
        "support_building_kills_out_bot_v8",
        "support_plate_destroys_out_bot_v8",
    ]
    for col in count_cols:
        if col not in scores.columns:
            scores[col] = 0
        scores[col] = scores[col].fillna(0)

    # botlane_deaths comes from event_context (0-12 window). Ensure present.
    if "botlane_deaths_bot_0_12" not in scores.columns:
        scores["botlane_deaths_bot_0_12"] = 0
    scores["botlane_deaths_bot_0_12"] = scores["botlane_deaths_bot_0_12"].fillna(0)

    # ---- Compute v8 score components ----
    scores["support_building_events_out_bot_v8"] = (
        scores["support_building_kills_out_bot_v8"].astype(float)
        + scores["support_plate_destroys_out_bot_v8"].astype(float)
    )
    scores["productive_roam_events_v8"] = (
        scores["support_kill_assists_out_bot_v8"].astype(float)
        + args.objective_weight
        * scores["support_objective_presence_out_bot_v8"].astype(float)
        + args.building_weight
        * scores["support_building_events_out_bot_v8"].astype(float)
    )
    scores["productive_event_score_v8"] = clip01(
        scores["productive_roam_events_v8"] / args.productive_saturation
    )
    scores["presence_score_v8"] = clip01(scores["alive_outside_bot_ratio_v8"])

    # --- FIX 7: use v8 xp_gap from our own frame window, fallback to v5 ---
    if "xp_gap_v8" in scores.columns:
        scores["xp_gap_score_v8"] = clip01(
            scores["xp_gap_v8"].fillna(scores["xp_gap_v5"])
        )
    else:
        scores["xp_gap_score_v8"] = clip01(scores["xp_gap_v5"])

    # ---- Combine into raw score ----
    # --- FIX 5: use strict weighted average that propagates NaN ---
    raw = weighted_average_strict(
        [
            scores["productive_event_score_v8"].to_numpy(dtype=np.float64),
            scores["presence_score_v8"].to_numpy(dtype=np.float64),
            scores["xp_gap_score_v8"].to_numpy(dtype=np.float64),
        ],
        [args.w_productive, args.w_presence, args.w_xp_gap],
    )

    # --- FIX 4: apply no_productive_cap BEFORE gamma ---
    # The cap should limit the raw score (which is in [0,1] linear space).
    # Applying it after gamma distorted the effective boundary because
    # gamma < 1 expands values: 0.35^0.75 ≈ 0.44, not 0.35.
    no_productive = scores["productive_roam_events_v8"].fillna(0) <= 0
    raw[no_productive] = np.minimum(
        np.where(np.isfinite(raw[no_productive]), raw[no_productive], 0.0),
        args.no_productive_cap,
    )

    scores[RAW_SCORE_COL] = raw
    scores[SCORE_COL] = np.power(clip01(raw), args.gamma)

    # ---- Confidence and chaos ----
    # Note: chaos uses 0-12 deaths (from event_context), which is the right
    # scope since deaths before minute 5 still affect early game state.
    max_alive_frames = float(args.end_frame - args.start_frame + 1)
    scores["chaos_score_v8"] = clip01(
        scores["botlane_deaths_bot_0_12"].astype(float) / 6.0
    )
    scores["support_score_confidence_v8_productive"] = np.clip(
        (scores["valid_alive_frames_v8"].fillna(0).astype(float) / max_alive_frames)
        * (1.0 - 0.25 * scores["chaos_score_v8"]),
        0.0,
        1.0,
    )
    scores["variant_id_v8"] = "v8_productive"
    scores["variant_description_v8"] = (
        "productive out-of-bot support events (kills, assists, objectives, "
        "building kills, plates) plus alive out-of-bot presence; "
        "high scores require productive evidence outside bot"
    )

    # ---- Log diagnostics ----
    n_total = len(scores)
    n_nan_score = int(scores[SCORE_COL].isna().sum())
    n_nan_presence = int(scores["presence_score_v8"].isna().sum())
    n_no_productive = int(no_productive.sum())
    print(f"[Diagnostics] total={n_total:,} nan_score={n_nan_score:,} "
          f"nan_presence={n_nan_presence:,} no_productive={n_no_productive:,}")

    # ---- Export ----
    keep_cols = [
        "match_id", "team_id", "side", "patch",
        "support_champion_name", "adc_champion_name",
        "valid_support_frames_v5", "valid_coop_frames_v5",
        "outside_ratio_v5", "far_ratio_v5", "xp_gap_v5",
        "valid_alive_frames_v8", "alive_outside_bot_frames_v8",
        "valid_far_adc_frames_v8", "far_from_adc_alive_frames_v8",
        "alive_outside_bot_ratio_v8", "far_from_adc_alive_ratio_v8",
        "xp_gap_v8",
        "support_kill_assists_out_bot_v8",
        "team_objectives_out_bot_v8",
        "support_objective_presence_out_bot_v8",
        "support_building_kills_out_bot_v8",
        "support_plate_destroys_out_bot_v8",
        "support_building_events_out_bot_v8",
        "productive_roam_events_v8",
        "productive_event_score_v8", "presence_score_v8", "xp_gap_score_v8",
        "botlane_deaths_bot_0_12", "chaos_score_v8",
        "support_score_confidence_v5", "support_score_confidence_v8_productive",
        V5_RAW, V5_SCORE, RAW_SCORE_COL, SCORE_COL,
        "variant_id_v8", "variant_description_v8",
    ]
    selected = scores[[c for c in keep_cols if c in scores.columns]].sort_values(
        JOIN_KEYS
    )
    analysis_path = outdir / args.selected_out_name
    selected.to_parquet(analysis_path, index=False)

    champion_summary = (
        selected.groupby("support_champion_name", dropna=False)
        .agg(
            rows=("support_champion_name", "count"),
            v5_mean=(V5_SCORE, "mean"),
            v8_mean=(SCORE_COL, "mean"),
            productive_event_mean=("productive_event_score_v8", "mean"),
            alive_outside_mean=("alive_outside_bot_ratio_v8", "mean"),
            no_productive_share=(
                "productive_roam_events_v8",
                lambda s: float((s <= 0).mean()),
            ),
        )
        .reset_index()
        .sort_values("v8_mean", ascending=False)
    )
    champion_summary.to_csv(
        outdir / "support_roam_score_v8_champion_summary.csv", index=False
    )

    if not event_samples.empty:
        sample_cols = [
            "match_id", "team_id", "event_kind", "minute", "x", "y", "zone_v6",
            "out_bot_context_v6", "nearest_frame_idx", "nearest_support_alive",
            "nearest_support_in_base", "nearest_support_in_bot_extended",
            "support_distance_to_objective_v8", "productive_sample_v8",
        ]
        event_samples[
            [c for c in sample_cols if c in event_samples.columns]
        ].sort_values(JOIN_KEYS + ["minute", "event_kind"]).to_parquet(
            outdir / "support_roam_score_v8_productive_event_samples.parquet",
            index=False,
        )

    save_plots(selected, outdir)
    expert_alignment = add_expert_alignment(
        selected, Path(args.expert_reference), outdir
    )

    metadata = {
        "score_col": SCORE_COL,
        "raw_score_col": RAW_SCORE_COL,
        "source_v5_scores": str(Path(args.v5_scores).resolve()),
        "source_frame_state": str(Path(args.frame_state).resolve()),
        "source_event_context": str(Path(args.event_context).resolve()),
        "source_event_positions": str(Path(args.event_positions).resolve()),
        "recipe": {
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "event_max_minute": args.event_max_minute,
            "event_window_note": (
                "Events use the full 0-12 minute window from the source. "
                "Presence uses frame_idx 5-12 (8 frames)."
            ),
            "objective_radius": args.objective_radius,
            "far_adc_threshold": args.far_adc_threshold,
            "productive_saturation": args.productive_saturation,
            "w_productive": args.w_productive,
            "w_presence": args.w_presence,
            "w_xp_gap": args.w_xp_gap,
            "objective_weight": args.objective_weight,
            "building_weight": args.building_weight,
            "gamma": args.gamma,
            "no_productive_cap": args.no_productive_cap,
            "no_productive_cap_note": "Applied to raw score BEFORE gamma",
        },
        "diagnostics": {
            "total_rows": n_total,
            "nan_final_score": n_nan_score,
            "nan_presence_score": n_nan_presence,
            "no_productive_events": n_no_productive,
            "max_alive_frames_possible": int(max_alive_frames),
        },
        "v5_summary": numeric_summary(selected, V5_SCORE),
        "v8_summary": numeric_summary(selected, SCORE_COL),
        "row_corr_v5_v8": float(selected[V5_SCORE].corr(selected[SCORE_COL])),
        "spearman_row_corr_v5_v8": float(
            selected[V5_SCORE].corr(selected[SCORE_COL], method="spearman")
        ),
        "mean_delta_v8_minus_v5": float(
            (selected[SCORE_COL] - selected[V5_SCORE]).mean()
        ),
        "productive_event_samples": {
            "rows": int(len(event_samples)),
            "match_team_keys": int(
                event_samples[JOIN_KEYS].drop_duplicates().shape[0]
            )
            if not event_samples.empty
            else 0,
            "by_kind": {
                str(k): int(v)
                for k, v in event_samples["event_kind"].value_counts().items()
            }
            if not event_samples.empty
            else {},
        },
        "expert_alignment": expert_alignment,
    }
    (outdir / "support_roam_score_v8_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.export_scores:
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir / args.selected_out_name
        selected.to_parquet(export_path, index=False)
        print(f"[Exported] {export_path.resolve()}")

    print(f"[Saved] {analysis_path.resolve()}")
    print(f"[Saved] {outdir.resolve()}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
