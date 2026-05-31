#!/usr/bin/env python3
"""
20_build_support_roam_score_v9_balanced.py

Build the balanced support roaming label (v9).

Design rationale
----------------
v5 measured spatial predisposition (outside_ratio + far_ratio + xp_gap) but was
contaminated by deaths, recalls, and chaos.  v8 cleaned the noise by requiring
productive evidence (kills/assists/objectives outside bot) but overshot: the
productive component dominated (Spearman 0.945 with final score) and the label
became an execution metric unpredictable from draft.

v9 recovers the spatial backbone of v5 with v8's data-quality fixes (alive, not
in base, frame_idx inclusion) and uses productive events as a **multiplicative
modulator**: they confirm or soften the spatial signal rather than replacing it.

Formula
-------
    backbone     = alive_outside_bot_ratio_v8  (frames 5-12, alive & ~base)
    xp_evidence  = xp_gap_v8

    raw_spatial = w_backbone * backbone + w_xp * xp_evidence

    productive_bonus = min(productive_roam_events / saturation, 1.0)
    modulator = {
        1.0 + boost * productive_bonus   if productive_roam_events > 0
        dampener                         otherwise
    }

    raw_v9 = clip(raw_spatial * modulator, 0, 1)
    raw_v9 *= (1.0 - chaos_weight * chaos_score)

    score_v9 = raw_v9 ^ gamma

Event and frame-derived columns are postgame target-building evidence only.
They must not be used as pregame model inputs.
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
DEFAULT_OUTDIR = REPO_ROOT / "final" / "analysis" / "label_v9_balanced"
DEFAULT_EXPORT_DIR = REPO_ROOT / "final" / "data" / "scores"

JOIN_KEYS = ["match_id", "team_id"]
V5_RAW = "raw_support_roam_score_v5_geometry"
V5_SCORE = "support_roam_score_v5_geometry"
RAW_SCORE_COL = "raw_support_roam_score_v9_balanced"
SCORE_COL = "support_roam_score_v9_balanced"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build balanced support roam label (v9).")
    # Sources
    p.add_argument("--v5-scores", default=str(DEFAULT_V5))
    p.add_argument("--frame-state", default=str(DEFAULT_FRAME_STATE))
    p.add_argument("--event-context", default=str(DEFAULT_EVENT_CONTEXT))
    p.add_argument("--event-positions", default=str(DEFAULT_EVENT_POSITIONS))
    p.add_argument("--expert-reference", default=str(DEFAULT_EXPERT_REFERENCE))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    p.add_argument("--selected-out-name", default="support_scores_v9_balanced_m12.parquet")
    # Presence window
    p.add_argument("--start-frame", type=int, default=5)
    p.add_argument("--end-frame", type=int, default=12)
    p.add_argument("--far-adc-threshold", type=float, default=2500.0)
    # Objective proximity (for productive events)
    p.add_argument("--objective-radius", type=float, default=3500.0)
    # Spatial backbone weights
    p.add_argument("--w-backbone", type=float, default=0.75,
                   help="Weight of alive_outside_bot_ratio in the spatial backbone")
    p.add_argument("--w-xp", type=float, default=0.25,
                   help="Weight of xp_gap in the spatial backbone")
    # Productive modulator
    p.add_argument("--productive-saturation", type=float, default=3.0,
                   help="Number of productive events at which the bonus saturates")
    p.add_argument("--objective-weight", type=float, default=0.50,
                   help="Weight of objective presence in productive event count")
    p.add_argument("--building-weight", type=float, default=0.30,
                   help="Weight of building kills/plates in productive event count")
    p.add_argument("--productive-boost", type=float, default=0.15,
                   help="Max multiplicative boost when productive events are present")
    p.add_argument("--no-productive-dampener", type=float, default=0.80,
                   help="Multiplicative factor when zero productive events (1.0 = no dampening)")
    # Chaos
    p.add_argument("--chaos-weight", type=float, default=0.20,
                   help="How much chaos_score dampens the raw score (multiplicative)")
    # Transform
    p.add_argument("--gamma", type=float, default=0.75)
    p.add_argument("--export-scores", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def clip01(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)


def numeric_summary(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    s = df[col].dropna()
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


# ---------------------------------------------------------------------------
# Data loading (shared with v8, duplicated for self-containment)
# ---------------------------------------------------------------------------

def load_presence_components(
    frame_state_path: Path,
    start_frame: int,
    end_frame: int,
    far_adc_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-frame presence aggregation.  Uses frame_idx (not minute) to include
    frame 12 correctly."""
    cols = [
        "match_id", "team_id", "frame_idx", "minute",
        "support_alive", "adc_alive",
        "support_x", "support_y", "adc_x", "adc_y",
        "support_in_base", "adc_in_base",
        "support_in_bot_extended", "dist_to_adc",
        "support_xp", "adc_xp",
    ]
    frames = pd.read_parquet(frame_state_path, columns=cols)
    frames = frames[
        frames["frame_idx"].between(start_frame, end_frame, inclusive="both")
    ].copy()

    frames["support_valid_alive"] = (
        frames["support_alive"].fillna(False)
        & ~frames["support_in_base"].fillna(False)
        & frames["support_x"].notna()
        & frames["support_y"].notna()
    )
    frames["support_alive_outside_bot"] = (
        frames["support_valid_alive"]
        & ~frames["support_in_bot_extended"].fillna(False)
    )
    frames["valid_far_adc_sample"] = (
        frames["support_valid_alive"]
        & frames["adc_alive"].fillna(False)
        & ~frames["adc_in_base"].fillna(False)
        & frames["dist_to_adc"].notna()
    )
    frames["support_far_from_adc_alive"] = (
        frames["valid_far_adc_sample"]
        & (frames["dist_to_adc"].astype(float) >= far_adc_threshold)
    )

    # XP gap from last valid frame in our window
    xp_frames = frames[frames["support_valid_alive"]].copy()
    xp_last = (
        xp_frames.sort_values("frame_idx")
        .groupby(JOIN_KEYS, dropna=False)
        .last()
        .reset_index()
    )
    xp_last["_sup_xp"] = xp_last["support_xp"].fillna(0).astype(float)
    xp_last["_adc_xp"] = xp_last["adc_xp"].fillna(0).astype(float)
    xp_ratio = np.where(
        xp_last["_adc_xp"] > 0,
        xp_last["_sup_xp"] / xp_last["_adc_xp"],
        1.0,
    )
    xp_last["xp_gap_v9"] = np.clip(1.0 - np.clip(xp_ratio, 0.0, 1.0), 0.0, 1.0)

    presence = (
        frames.groupby(JOIN_KEYS, dropna=False)
        .agg(
            valid_alive_frames_v9=("support_valid_alive", "sum"),
            alive_outside_bot_frames_v9=("support_alive_outside_bot", "sum"),
            valid_far_adc_frames_v9=("valid_far_adc_sample", "sum"),
            far_from_adc_alive_frames_v9=("support_far_from_adc_alive", "sum"),
        )
        .reset_index()
    )
    presence["alive_outside_bot_ratio_v9"] = np.where(
        presence["valid_alive_frames_v9"] > 0,
        presence["alive_outside_bot_frames_v9"] / presence["valid_alive_frames_v9"],
        np.nan,
    )
    presence["far_from_adc_alive_ratio_v9"] = np.where(
        presence["valid_far_adc_frames_v9"] > 0,
        presence["far_from_adc_alive_frames_v9"] / presence["valid_far_adc_frames_v9"],
        np.nan,
    )
    presence = presence.merge(
        xp_last[JOIN_KEYS + ["xp_gap_v9"]], on=JOIN_KEYS, how="left"
    )

    # Nearest-support lookup (all frames, for objective proximity)
    all_frames = pd.read_parquet(frame_state_path, columns=[
        "match_id", "team_id", "frame_idx",
        "support_alive", "support_in_base", "support_in_bot_extended",
        "support_x", "support_y",
    ])
    nearest = all_frames.rename(columns={
        "frame_idx": "nearest_frame_idx",
        "support_alive": "nearest_support_alive",
        "support_in_base": "nearest_support_in_base",
        "support_in_bot_extended": "nearest_support_in_bot_extended",
        "support_x": "nearest_support_x",
        "support_y": "nearest_support_y",
    })
    return presence, nearest


def load_productive_events(
    event_positions_path: Path,
    nearest_support: pd.DataFrame,
    max_frame: int,
    objective_radius: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load productive events from the full 0-12 min window."""
    events = pd.read_parquet(event_positions_path)
    if events.empty:
        empty = pd.DataFrame(columns=JOIN_KEYS + [
            "support_kill_assists_out_bot_v9",
            "team_objectives_out_bot_v9",
            "support_objective_presence_out_bot_v9",
            "support_building_kills_out_bot_v9",
            "support_plate_destroys_out_bot_v9",
        ])
        return empty, events

    # Kill/assist events outside bot
    support_kill_assists = events[
        (events["event_kind"] == "support_kill_assist")
        & events["out_bot_context_v6"].fillna(False)
    ].copy()

    # Building kills and plate destroys outside bot
    building_kills = events[
        (events["event_kind"] == "support_building_kill")
        & events["out_bot_context_v6"].fillna(False)
    ].copy()
    plate_destroys = events[
        (events["event_kind"] == "support_plate_destroy")
        & events["out_bot_context_v6"].fillna(False)
    ].copy()

    # Objective events outside bot (with proximity check)
    objectives = events[
        (events["event_kind"] == "team_objective")
        & events["out_bot_context_v6"].fillna(False)
    ].copy()

    if not objectives.empty:
        objectives["nearest_frame_idx"] = np.clip(
            objectives["minute"].round().astype(int), 0, max_frame,
        )
        objectives = objectives.merge(
            nearest_support, on=JOIN_KEYS + ["nearest_frame_idx"], how="left",
        )
        sup_x = objectives["nearest_support_x"].astype(float)
        sup_y = objectives["nearest_support_y"].astype(float)
        obj_x = objectives["x"].astype(float)
        obj_y = objectives["y"].astype(float)
        objectives["support_distance_to_objective"] = np.sqrt(
            (obj_x - sup_x) ** 2 + (obj_y - sup_y) ** 2
        )
        objectives["support_near_objective"] = (
            objectives["nearest_support_alive"].fillna(False)
            & ~objectives["nearest_support_in_base"].fillna(False)
            & ~objectives["nearest_support_in_bot_extended"].fillna(False)
            & (objectives["support_distance_to_objective"] <= objective_radius)
        )
    else:
        objectives["support_near_objective"] = pd.array([], dtype=bool)

    # --- Aggregate per match-team ---
    kill_counts = (
        support_kill_assists.groupby(JOIN_KEYS, dropna=False)
        .size().rename("support_kill_assists_out_bot_v9").reset_index()
    )
    obj_counts = (
        objectives.groupby(JOIN_KEYS, dropna=False).agg(
            team_objectives_out_bot_v9=("event_kind", "size"),
            support_objective_presence_out_bot_v9=("support_near_objective", "sum"),
        ).reset_index()
    ) if not objectives.empty else pd.DataFrame(
        columns=JOIN_KEYS + ["team_objectives_out_bot_v9", "support_objective_presence_out_bot_v9"]
    )
    bldg_counts = (
        building_kills.groupby(JOIN_KEYS, dropna=False)
        .size().rename("support_building_kills_out_bot_v9").reset_index()
    ) if not building_kills.empty else pd.DataFrame(
        columns=JOIN_KEYS + ["support_building_kills_out_bot_v9"]
    )
    plate_counts = (
        plate_destroys.groupby(JOIN_KEYS, dropna=False)
        .size().rename("support_plate_destroys_out_bot_v9").reset_index()
    ) if not plate_destroys.empty else pd.DataFrame(
        columns=JOIN_KEYS + ["support_plate_destroys_out_bot_v9"]
    )

    rows = kill_counts
    for right in [obj_counts, bldg_counts, plate_counts]:
        rows = rows.merge(right, on=JOIN_KEYS, how="outer")

    for col in [
        "support_kill_assists_out_bot_v9",
        "team_objectives_out_bot_v9",
        "support_objective_presence_out_bot_v9",
        "support_building_kills_out_bot_v9",
        "support_plate_destroys_out_bot_v9",
    ]:
        if col not in rows.columns:
            rows[col] = 0
        rows[col] = rows[col].fillna(0).astype(int)

    # Debug event samples
    parts = [support_kill_assists.assign(productive_sample=True)]
    if not objectives.empty:
        parts.append(objectives.assign(
            productive_sample=objectives["support_near_objective"].fillna(False)
        ))
    if not building_kills.empty:
        parts.append(building_kills.assign(productive_sample=True))
    if not plate_destroys.empty:
        parts.append(plate_destroys.assign(productive_sample=True))
    event_samples = pd.concat(parts, ignore_index=True)
    return rows, event_samples


# ---------------------------------------------------------------------------
# Expert alignment
# ---------------------------------------------------------------------------

def add_expert_alignment(
    scores: pd.DataFrame, expert_path: Path, outdir: Path,
) -> Dict[str, Any]:
    if not expert_path.exists():
        return {"available": False, "reason": f"missing {expert_path}"}
    ref = pd.read_csv(expert_path)
    champ = (
        scores.groupby("support_champion_name", dropna=False)
        .agg(
            rows=("support_champion_name", "count"),
            v5_mean=(V5_SCORE, "mean"),
            v9_mean=(SCORE_COL, "mean"),
            backbone_mean=("backbone_v9", "mean"),
            productive_mean=("productive_roam_events_v9", "mean"),
        )
        .reset_index()
        .rename(columns={"support_champion_name": "champion_name"})
    )
    aligned = champ.merge(ref, on="champion_name", how="inner")
    aligned.to_csv(outdir / "support_roam_score_v9_expert_alignment.csv", index=False)
    n = len(aligned)
    return {
        "available": True,
        "reference_rows": int(len(ref)),
        "aligned_champions": n,
        "spearman_v9_vs_expert": float(
            aligned["v9_mean"].corr(aligned["expert_support_roam_score"], method="spearman")
        ) if n >= 3 else float("nan"),
        "spearman_v5_vs_expert": float(
            aligned["v5_mean"].corr(aligned["expert_support_roam_score"], method="spearman")
        ) if n >= 3 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_plots(scores: pd.DataFrame, outdir: Path) -> None:
    # Overlay histogram
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(scores[V5_SCORE].dropna(), bins=50, range=(0, 1),
            alpha=0.40, label="v5 geometry", color="#999999")
    ax.hist(scores[SCORE_COL].dropna(), bins=50, range=(0, 1),
            alpha=0.60, label="v9 balanced", color="#2563eb")
    ax.set_title("Support roam score: v5 geometry vs v9 balanced")
    ax.set_xlabel("score"); ax.set_ylabel("match-team rows")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "v9_vs_v5_overlay.png", dpi=180)
    plt.close(fig)

    # Scatter v5 vs v9
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(scores[V5_SCORE], scores[SCORE_COL], s=2, alpha=0.08, color="#2563eb")
    ax.plot([0, 1], [0, 1], color="black", linewidth=1)
    ax.set_title("Row-level: v5 vs v9 balanced")
    ax.set_xlabel("v5 geometry"); ax.set_ylabel("v9 balanced")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "v9_vs_v5_scatter.png", dpi=180)
    plt.close(fig)

    # Component correlation bar
    comp_cols = {
        "backbone_v9": "backbone (spatial)",
        "xp_gap_v9": "xp_gap",
        "productive_event_score_v9": "productive_event_score",
        "modulator_v9": "modulator",
        V5_SCORE: "v5_geometry",
    }
    corrs = {}
    for col, label in comp_cols.items():
        if col in scores.columns:
            corrs[label] = scores[col].corr(scores[SCORE_COL], method="spearman")
    corrs = dict(sorted(corrs.items(), key=lambda x: x[1]))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(list(corrs.keys()), list(corrs.values()), color="#2563eb", alpha=0.85)
    ax.set_xlabel("Spearman with v9 score")
    ax.set_title("Component correlation with final v9 label")
    ax.set_xlim(0, 1); ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "v9_component_correlations.png", dpi=180)
    plt.close(fig)

    # Top champions
    champ = (
        scores.groupby("support_champion_name")[SCORE_COL]
        .agg(["count", "mean"])
        .query("count >= 100")
        .sort_values("mean", ascending=False)
        .head(25)
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(champ.index[::-1], champ["mean"].to_numpy()[::-1],
            color="#2563eb", alpha=0.85)
    ax.set_xlabel("mean v9 balanced score")
    ax.set_title("Top support champions by v9 balanced label")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "v9_top_champions.png", dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    export_dir = Path(args.export_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Load sources ----
    v5 = pd.read_parquet(args.v5_scores)
    context = pd.read_parquet(args.event_context)
    presence, nearest_support = load_presence_components(
        Path(args.frame_state), args.start_frame, args.end_frame,
        args.far_adc_threshold,
    )
    productive_events, event_samples = load_productive_events(
        Path(args.event_positions), nearest_support,
        max_frame=args.end_frame,
        objective_radius=args.objective_radius,
    )

    # ---- Merge ----
    scores = v5.merge(context, on=JOIN_KEYS, how="left", suffixes=("", "_event"))
    scores = scores.merge(presence, on=JOIN_KEYS, how="left")
    scores = scores.merge(productive_events, on=JOIN_KEYS, how="left")

    # ---- Fill missing counts ----
    for col in [
        "support_kill_assists_out_bot_v9",
        "team_objectives_out_bot_v9",
        "support_objective_presence_out_bot_v9",
        "support_building_kills_out_bot_v9",
        "support_plate_destroys_out_bot_v9",
    ]:
        if col not in scores.columns:
            scores[col] = 0
        scores[col] = scores[col].fillna(0)

    if "botlane_deaths_bot_0_12" not in scores.columns:
        scores["botlane_deaths_bot_0_12"] = 0
    scores["botlane_deaths_bot_0_12"] = scores["botlane_deaths_bot_0_12"].fillna(0)

    # ==================================================================
    # STEP 1: Spatial backbone  (predisposition signal)
    # ==================================================================
    scores["backbone_v9"] = clip01(scores["alive_outside_bot_ratio_v9"])
    scores["xp_gap_score_v9"] = clip01(
        scores["xp_gap_v9"].fillna(scores["xp_gap_v5"])
    )

    raw_spatial = (
        args.w_backbone * scores["backbone_v9"].to_numpy(dtype=np.float64)
        + args.w_xp * scores["xp_gap_score_v9"].to_numpy(dtype=np.float64)
    )
    # For rows where backbone is NaN (missing frame_state), raw_spatial is NaN
    backbone_nan = ~np.isfinite(scores["backbone_v9"].to_numpy(dtype=np.float64))
    raw_spatial[backbone_nan] = np.nan

    # ==================================================================
    # STEP 2: Productive modulator  (confirmatory, NOT dominant)
    # ==================================================================
    scores["support_building_events_out_bot_v9"] = (
        scores["support_building_kills_out_bot_v9"].astype(float)
        + scores["support_plate_destroys_out_bot_v9"].astype(float)
    )
    scores["productive_roam_events_v9"] = (
        scores["support_kill_assists_out_bot_v9"].astype(float)
        + args.objective_weight
        * scores["support_objective_presence_out_bot_v9"].astype(float)
        + args.building_weight
        * scores["support_building_events_out_bot_v9"].astype(float)
    )
    scores["productive_event_score_v9"] = clip01(
        scores["productive_roam_events_v9"] / args.productive_saturation
    )

    has_productive = scores["productive_roam_events_v9"].fillna(0) > 0
    productive_bonus = clip01(
        scores["productive_roam_events_v9"].fillna(0) / args.productive_saturation
    )

    modulator = np.where(
        has_productive,
        1.0 + args.productive_boost * productive_bonus,
        args.no_productive_dampener,
    )
    scores["modulator_v9"] = modulator

    # ==================================================================
    # STEP 3: Apply modulator + chaos dampening
    # ==================================================================
    raw = np.clip(raw_spatial * modulator, 0.0, 1.0)

    scores["chaos_score_v9"] = clip01(
        scores["botlane_deaths_bot_0_12"].astype(float) / 6.0
    )
    chaos_dampener = 1.0 - args.chaos_weight * scores["chaos_score_v9"].to_numpy(
        dtype=np.float64
    )
    raw = raw * chaos_dampener

    # ==================================================================
    # STEP 4: Gamma transform
    # ==================================================================
    scores[RAW_SCORE_COL] = raw
    scores[SCORE_COL] = np.power(clip01(raw), args.gamma)

    # ---- Confidence ----
    max_frames = float(args.end_frame - args.start_frame + 1)
    scores["support_score_confidence_v9"] = np.clip(
        (scores["valid_alive_frames_v9"].fillna(0).astype(float) / max_frames)
        * (1.0 - 0.25 * scores["chaos_score_v9"]),
        0.0, 1.0,
    )

    # ---- Diagnostics ----
    n_total = len(scores)
    n_nan = int(scores[SCORE_COL].isna().sum())
    n_no_prod = int((~has_productive).sum())
    print(f"[Diagnostics] total={n_total:,} nan_score={n_nan:,} "
          f"no_productive={n_no_prod:,} ({100*n_no_prod/max(n_total,1):.1f}%)")

    # Component correlations with final score
    comp_corrs = {}
    for c in ["backbone_v9", "xp_gap_score_v9", "productive_event_score_v9",
              "modulator_v9", V5_SCORE]:
        if c in scores.columns:
            comp_corrs[c] = float(scores[c].corr(scores[SCORE_COL], method="spearman"))
    print("[Component Spearman with v9]")
    for k, v in sorted(comp_corrs.items(), key=lambda x: -x[1]):
        print(f"  {k:40s} {v:.3f}")

    # ---- Export ----
    keep_cols = [
        "match_id", "team_id", "side", "patch",
        "support_champion_name", "adc_champion_name",
        # v5 reference
        "valid_support_frames_v5", "valid_coop_frames_v5",
        "outside_ratio_v5", "far_ratio_v5", "xp_gap_v5",
        # v9 presence
        "valid_alive_frames_v9", "alive_outside_bot_frames_v9",
        "valid_far_adc_frames_v9", "far_from_adc_alive_frames_v9",
        "alive_outside_bot_ratio_v9", "far_from_adc_alive_ratio_v9",
        "xp_gap_v9",
        # v9 productive
        "support_kill_assists_out_bot_v9",
        "team_objectives_out_bot_v9",
        "support_objective_presence_out_bot_v9",
        "support_building_kills_out_bot_v9",
        "support_plate_destroys_out_bot_v9",
        "support_building_events_out_bot_v9",
        "productive_roam_events_v9",
        "productive_event_score_v9",
        # v9 score components
        "backbone_v9", "xp_gap_score_v9", "modulator_v9",
        "botlane_deaths_bot_0_12", "chaos_score_v9",
        "support_score_confidence_v5", "support_score_confidence_v9",
        V5_RAW, V5_SCORE, RAW_SCORE_COL, SCORE_COL,
    ]
    selected = scores[[c for c in keep_cols if c in scores.columns]].sort_values(
        JOIN_KEYS
    )
    analysis_path = outdir / args.selected_out_name
    selected.to_parquet(analysis_path, index=False)

    # Champion summary
    champion_summary = (
        selected.groupby("support_champion_name", dropna=False)
        .agg(
            rows=("support_champion_name", "count"),
            v5_mean=(V5_SCORE, "mean"),
            v9_mean=(SCORE_COL, "mean"),
            backbone_mean=("backbone_v9", "mean"),
            productive_mean=("productive_roam_events_v9", "mean"),
            no_productive_share=(
                "productive_roam_events_v9",
                lambda s: float((s <= 0).mean()),
            ),
        )
        .reset_index()
        .sort_values("v9_mean", ascending=False)
    )
    champion_summary.to_csv(
        outdir / "support_roam_score_v9_champion_summary.csv", index=False,
    )

    # Event samples
    if not event_samples.empty:
        sample_cols = [
            "match_id", "team_id", "event_kind", "minute", "x", "y", "zone_v6",
            "out_bot_context_v6", "nearest_frame_idx", "nearest_support_alive",
            "nearest_support_in_base", "nearest_support_in_bot_extended",
            "support_distance_to_objective", "productive_sample",
        ]
        event_samples[
            [c for c in sample_cols if c in event_samples.columns]
        ].sort_values(
            JOIN_KEYS + ["minute", "event_kind"]
        ).to_parquet(
            outdir / "v9_productive_event_samples.parquet", index=False,
        )

    save_plots(selected, outdir)
    expert_alignment = add_expert_alignment(
        selected, Path(args.expert_reference), outdir,
    )

    # ---- Metadata ----
    metadata = {
        "score_col": SCORE_COL,
        "raw_score_col": RAW_SCORE_COL,
        "sources": {
            "v5_scores": str(Path(args.v5_scores).resolve()),
            "frame_state": str(Path(args.frame_state).resolve()),
            "event_context": str(Path(args.event_context).resolve()),
            "event_positions": str(Path(args.event_positions).resolve()),
        },
        "recipe": {
            "start_frame": args.start_frame,
            "end_frame": args.end_frame,
            "w_backbone": args.w_backbone,
            "w_xp": args.w_xp,
            "productive_saturation": args.productive_saturation,
            "objective_weight": args.objective_weight,
            "building_weight": args.building_weight,
            "productive_boost": args.productive_boost,
            "no_productive_dampener": args.no_productive_dampener,
            "chaos_weight": args.chaos_weight,
            "gamma": args.gamma,
            "design_note": (
                "Spatial backbone (alive_outside_bot_ratio) with productive "
                "events as multiplicative modulator. Productive events confirm "
                "or soften the spatial signal but do not dominate it."
            ),
        },
        "diagnostics": {
            "total_rows": n_total,
            "nan_final_score": n_nan,
            "no_productive_events": n_no_prod,
            "no_productive_pct": round(100 * n_no_prod / max(n_total, 1), 1),
            "max_alive_frames_possible": int(max_frames),
        },
        "component_spearman_with_v9": comp_corrs,
        "v5_summary": numeric_summary(selected, V5_SCORE),
        "v9_summary": numeric_summary(selected, SCORE_COL),
        "row_corr_v5_v9_pearson": float(selected[V5_SCORE].corr(selected[SCORE_COL])),
        "row_corr_v5_v9_spearman": float(
            selected[V5_SCORE].corr(selected[SCORE_COL], method="spearman")
        ),
        "mean_delta_v9_minus_v5": float(
            (selected[SCORE_COL] - selected[V5_SCORE]).mean()
        ),
        "event_samples": {
            "rows": int(len(event_samples)),
            "match_team_keys": int(
                event_samples[JOIN_KEYS].drop_duplicates().shape[0]
            ) if not event_samples.empty else 0,
            "by_kind": {
                str(k): int(v)
                for k, v in event_samples["event_kind"].value_counts().items()
            } if not event_samples.empty else {},
        },
        "expert_alignment": expert_alignment,
    }
    (outdir / "support_roam_score_v9_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
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
