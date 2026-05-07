#!/usr/bin/env python3
"""
Build support_roam_score distributions with manual geometry v5.

This keeps the selected v3 label recipe fixed and swaps the bot-context
geometry to manual v5:

- window: minutes 5 to 12
- raw score: 0.45 outside bot + 0.35 far from ADC + 0.20 XP gap
- far ADC threshold: 2500 map units
- calibration: gamma 0.75
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "ProgresoActual2" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_geometry_v5_frame_state_distributions import classify_chunk_absolute  # noqa: E402


DEFAULT_FRAME_STATE = REPO_ROOT / "ProgresoActual" / "data" / "clean" / "frame_state" / "support_frame_state.parquet"
DEFAULT_CONFIG = REPO_ROOT / "ProgresoActual2" / "data" / "geometry" / "manual_geometry_v5_config.json"
DEFAULT_BASELINE = REPO_ROOT / "ProgresoActual2" / "data" / "clean" / "scores" / "support_scores_v3_m12.parquet"
DEFAULT_OUTDIR = REPO_ROOT / "ProgresoActual2" / "analysis" / "support_roam_score_v5_geometry"
DEFAULT_EXPORT_DIR = REPO_ROOT / "ProgresoActual2" / "data" / "clean" / "scores"

JOIN_KEYS = ["match_id", "team_id"]
SCORE_COL = "support_roam_score_v5_geometry"
RAW_SCORE_COL = "raw_support_roam_score_v5_geometry"
BOT_CONTEXT_ZONES = {"BOT_LANE_CORE", "BOT_SIDE_NEAR", "RIVER_BOT", "DRAGON_AREA"}
BASE_ZONES = {"BLUE_BASE", "RED_BASE"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute support_roam_score with manual geometry v5.")
    p.add_argument("--frame-state-path", default=str(DEFAULT_FRAME_STATE))
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--baseline-path", default=str(DEFAULT_BASELINE))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--export-dir", default=str(DEFAULT_EXPORT_DIR))
    p.add_argument("--start-minute", type=float, default=5.0)
    p.add_argument("--max-minute", type=float, default=12.0)
    p.add_argument("--far-adc-threshold", type=float, default=2500.0)
    p.add_argument("--w-outside", type=float, default=0.45)
    p.add_argument("--w-far", type=float, default=0.35)
    p.add_argument("--w-xp", type=float, default=0.20)
    p.add_argument("--gamma", type=float, default=0.75)
    p.add_argument("--xp-ratio-min", type=float, default=0.60)
    p.add_argument("--xp-ratio-max", type=float, default=1.00)
    p.add_argument("--min-support-frames", type=int, default=2)
    p.add_argument("--chunk-size", type=int, default=750000)
    p.add_argument("--sample-match-frac", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--export-scores", action="store_true")
    p.add_argument("--selected-out-name", default="support_scores_v5_geometry_m12.parquet")
    return p.parse_args()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_name(value: object) -> str:
    return str(value).strip().lower().replace(" ", "").replace("'", "").replace(".", "")


def load_frame_state(path: str, start_minute: float, max_minute: float) -> pd.DataFrame:
    columns = [
        "match_id",
        "team_id",
        "side",
        "patch",
        "frame_idx",
        "minute",
        "support_champion_name",
        "adc_champion_name",
        "support_alive",
        "adc_alive",
        "support_x",
        "support_y",
        "adc_x",
        "adc_y",
        "dist_to_adc",
        "support_xp",
        "adc_xp",
    ]
    df = pd.read_parquet(path, columns=columns)
    return df[df["minute"].between(start_minute, max_minute, inclusive="both")].copy()


def sample_by_match_id(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    if frac <= 0.0 or frac >= 1.0:
        return df
    match_ids = pd.Series(df["match_id"].dropna().unique())
    sampled = match_ids.sample(n=max(1, int(round(len(match_ids) * frac))), random_state=seed)
    return df[df["match_id"].isin(set(sampled))].copy()


def zone_order(config: dict) -> List[str]:
    order = ["OUT_OF_MAP", "UNCLASSIFIED"] + list(config["colors"].keys())
    for zone in config["priority"]:
        if zone not in order:
            order.append(zone)
    return order


def classify_xy(x: np.ndarray, y: np.ndarray, config: dict, chunk_size: int, label: str) -> np.ndarray:
    order = zone_order(config)
    zone_to_id = {zone: idx for idx, zone in enumerate(order)}
    id_to_zone = np.asarray(order, dtype=object)
    out = np.empty(x.shape[0], dtype=np.int16)
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        out[start:end] = classify_chunk_absolute(x[start:end], y[start:end], config, zone_to_id)
        print(f"[Classify {label}] rows {end:,}/{x.shape[0]:,}")
    return id_to_zone[out]


def add_v5_frame_flags(df: pd.DataFrame, config: dict, chunk_size: int) -> pd.DataFrame:
    out = df.copy()
    support_x = out["support_x"].to_numpy(dtype=np.float64, copy=False)
    support_y = out["support_y"].to_numpy(dtype=np.float64, copy=False)
    adc_x = out["adc_x"].to_numpy(dtype=np.float64, copy=False)
    adc_y = out["adc_y"].to_numpy(dtype=np.float64, copy=False)

    out["support_zone_v5_abs"] = classify_xy(support_x, support_y, config, chunk_size, "support")
    out["adc_zone_v5_abs"] = classify_xy(adc_x, adc_y, config, chunk_size, "adc")
    out["support_in_base_v5"] = out["support_zone_v5_abs"].isin(BASE_ZONES)
    out["adc_in_base_v5"] = out["adc_zone_v5_abs"].isin(BASE_ZONES)
    out["support_in_bot_context_v5"] = out["support_zone_v5_abs"].isin(BOT_CONTEXT_ZONES)
    return out


def compute_scores(
    df: pd.DataFrame,
    far_adc_threshold: float,
    weights: np.ndarray,
    gamma: float,
    xp_ratio_min: float,
    xp_ratio_max: float,
    min_support_frames: int,
) -> pd.DataFrame:
    xp_last = (
        df.sort_values(["match_id", "team_id", "frame_idx"])
        .groupby(JOIN_KEYS, as_index=False)
        .agg(
            support_adc_xp_ratio_v5=("support_xp", "last"),
            adc_xp_last=("adc_xp", "last"),
        )
    )
    xp_last["support_adc_xp_ratio_v5"] = np.where(
        xp_last["adc_xp_last"].fillna(0) > 0,
        xp_last["support_adc_xp_ratio_v5"] / xp_last["adc_xp_last"],
        np.nan,
    )
    xp_last = xp_last.drop(columns=["adc_xp_last"])

    spatial = df[
        df["support_alive"].fillna(False)
        & df["support_x"].notna()
        & df["support_y"].notna()
        & ~df["support_in_base_v5"].fillna(False)
    ].copy()
    spatial["out_bot_context_v5"] = ~spatial["support_in_bot_context_v5"].fillna(False)

    coop = spatial[
        spatial["adc_alive"].fillna(False)
        & spatial["adc_x"].notna()
        & spatial["adc_y"].notna()
        & ~spatial["adc_in_base_v5"].fillna(False)
    ].copy()
    coop["far_from_adc_v5"] = coop["dist_to_adc"].fillna(-1.0) >= far_adc_threshold

    agg_spatial = spatial.groupby(JOIN_KEYS, as_index=False).agg(
        side=("side", "first"),
        patch=("patch", "first"),
        support_champion_name=("support_champion_name", "first"),
        adc_champion_name=("adc_champion_name", "first"),
        valid_support_frames_v5=("frame_idx", "count"),
        frames_out_bot_context_v5=("out_bot_context_v5", "sum"),
        frames_in_bot_context_v5=("support_in_bot_context_v5", "sum"),
    )
    agg_spatial["outside_ratio_v5"] = (
        agg_spatial["frames_out_bot_context_v5"] / agg_spatial["valid_support_frames_v5"]
    )

    agg_coop = coop.groupby(JOIN_KEYS, as_index=False).agg(
        valid_coop_frames_v5=("frame_idx", "count"),
        frames_far_from_adc_v5=("far_from_adc_v5", "sum"),
        mean_distance_to_adc_v5=("dist_to_adc", "mean"),
    )
    if not agg_coop.empty:
        agg_coop["far_ratio_v5"] = agg_coop["frames_far_from_adc_v5"] / agg_coop["valid_coop_frames_v5"]

    out = agg_spatial.merge(agg_coop, on=JOIN_KEYS, how="left").merge(xp_last, on=JOIN_KEYS, how="left")
    out = out[out["valid_support_frames_v5"] >= min_support_frames].copy()

    xp_ratio = out["support_adc_xp_ratio_v5"].clip(lower=xp_ratio_min, upper=xp_ratio_max)
    out["xp_gap_v5"] = 1.0 - ((xp_ratio - xp_ratio_min) / (xp_ratio_max - xp_ratio_min))
    out.loc[out["support_adc_xp_ratio_v5"].isna(), "xp_gap_v5"] = np.nan

    components = out[["outside_ratio_v5", "far_ratio_v5", "xp_gap_v5"]].astype(float)
    valid_mask = components.notna().to_numpy(dtype=float)
    weighted_values = components.fillna(0.0).to_numpy(dtype=float) * weights
    den = (valid_mask * weights).sum(axis=1)
    out[RAW_SCORE_COL] = np.where(den > 0, weighted_values.sum(axis=1) / den, np.nan)
    out[SCORE_COL] = out[RAW_SCORE_COL].clip(0.0, 1.0).pow(gamma)
    out["support_score_confidence_v5"] = np.minimum(1.0, out["valid_support_frames_v5"] / 6.0)
    out["variant_id"] = "v5_geometry_gamma075"
    out["variant_description"] = "v3 selected recipe with manual geometry v5 bot context"
    out["start_minute"] = 5.0
    out["max_minute"] = 12.0
    out["far_adc_threshold"] = far_adc_threshold
    out["w_outside"] = weights[0]
    out["w_far"] = weights[1]
    out["w_xp"] = weights[2]
    out["transform"] = "gamma"
    out["gamma"] = gamma
    return out


def numeric_summary(scores: pd.Series) -> Dict[str, float]:
    valid = pd.to_numeric(scores, errors="coerce").dropna()
    return {
        "score_n": int(valid.shape[0]),
        "score_missing": int(scores.shape[0] - valid.shape[0]),
        "score_mean": float(valid.mean()),
        "score_std": float(valid.std(ddof=0)),
        "score_min": float(valid.min()),
        "score_q01": float(valid.quantile(0.01)),
        "score_q05": float(valid.quantile(0.05)),
        "score_q25": float(valid.quantile(0.25)),
        "score_median": float(valid.quantile(0.50)),
        "score_q75": float(valid.quantile(0.75)),
        "score_q95": float(valid.quantile(0.95)),
        "score_q99": float(valid.quantile(0.99)),
        "score_max": float(valid.max()),
        "share_eq_0": float((valid == 0.0).mean()),
        "share_eq_1": float((valid == 1.0).mean()),
    }


def champion_means(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby("support_champion_name", dropna=False)[SCORE_COL]
        .agg(games="count", mean="mean", median="median", std="std", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        .reset_index()
        .sort_values("mean", ascending=False)
    )


def side_summary(scores: pd.DataFrame) -> pd.DataFrame:
    return (
        scores.groupby("side", dropna=False)[SCORE_COL]
        .agg(rows="count", mean="mean", median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        .reset_index()
    )


def compare_to_baseline(scores: pd.DataFrame, baseline_path: str) -> pd.DataFrame:
    if not os.path.exists(baseline_path):
        return pd.DataFrame()
    baseline = pd.read_parquet(baseline_path, columns=JOIN_KEYS + ["support_roam_score_v3", "outside_ratio"])
    merged = scores[JOIN_KEYS + [SCORE_COL, "outside_ratio_v5"]].merge(baseline, on=JOIN_KEYS, how="inner")
    merged["score_delta_v5_minus_v3"] = merged[SCORE_COL] - merged["support_roam_score_v3"]
    merged["outside_ratio_delta_v5_minus_v3"] = merged["outside_ratio_v5"] - merged["outside_ratio"]
    return merged


def save_plots(scores: pd.DataFrame, comparison: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.hist(scores[SCORE_COL].dropna(), bins=50, range=(0, 1), color="#2f80ed", alpha=0.78)
    ax.set_title("support_roam_score distribution with geometry v5")
    ax.set_xlabel(SCORE_COL)
    ax.set_ylabel("Match-team rows")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "support_roam_score_v5_distribution.png", dpi=180)
    plt.close(fig)

    if not comparison.empty:
        fig, ax = plt.subplots(figsize=(9, 5.8))
        ax.hist(comparison["support_roam_score_v3"].dropna(), bins=50, range=(0, 1), color="#999999", alpha=0.48, label="v3 legacy geometry")
        ax.hist(comparison[SCORE_COL].dropna(), bins=50, range=(0, 1), color="#2f80ed", alpha=0.48, label="v5 manual geometry")
        ax.set_title("support_roam_score distribution: v3 vs v5 geometry")
        ax.set_xlabel("score")
        ax.set_ylabel("Match-team rows")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / "support_roam_score_v3_vs_v5_distribution_overlay.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5.8))
        ax.hist(comparison["score_delta_v5_minus_v3"].dropna(), bins=60, color="#eb5757", alpha=0.78)
        ax.axvline(0, color="black", linewidth=1.0)
        ax.set_title("Score delta: geometry v5 minus v3 legacy geometry")
        ax.set_xlabel("delta")
        ax.set_ylabel("Match-team rows")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(outdir / "support_roam_score_v5_minus_v3_delta.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_dir(outdir)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    weights = np.asarray([args.w_outside, args.w_far, args.w_xp], dtype=float)

    print(f"[Input] frame_state={os.path.abspath(args.frame_state_path)}")
    frame_state = load_frame_state(args.frame_state_path, args.start_minute, args.max_minute)
    frame_state = sample_by_match_id(frame_state, args.sample_match_frac, args.seed)
    print(f"[Loaded] frames={len(frame_state):,} match_ids={frame_state['match_id'].nunique():,}")

    frame_state = add_v5_frame_flags(frame_state, config, args.chunk_size)
    scores = compute_scores(
        frame_state,
        far_adc_threshold=args.far_adc_threshold,
        weights=weights,
        gamma=args.gamma,
        xp_ratio_min=args.xp_ratio_min,
        xp_ratio_max=args.xp_ratio_max,
        min_support_frames=args.min_support_frames,
    )
    comparison = compare_to_baseline(scores, args.baseline_path)

    summary = {
        "rows": int(len(scores)),
        "total_match_team_keys_in_window": int(frame_state[JOIN_KEYS].drop_duplicates().shape[0]),
        "coverage": float(len(scores) / max(frame_state[JOIN_KEYS].drop_duplicates().shape[0], 1)),
        "config_version": config.get("version"),
        "source_frame_state_path": os.path.abspath(args.frame_state_path),
        "baseline_path": os.path.abspath(args.baseline_path),
    }
    summary.update(numeric_summary(scores[SCORE_COL]))
    if not comparison.empty:
        summary.update({
            "row_corr_vs_v3": float(comparison[SCORE_COL].corr(comparison["support_roam_score_v3"])),
            "mean_delta_v5_minus_v3": float(comparison["score_delta_v5_minus_v3"].mean()),
            "median_delta_v5_minus_v3": float(comparison["score_delta_v5_minus_v3"].median()),
            "q05_delta_v5_minus_v3": float(comparison["score_delta_v5_minus_v3"].quantile(0.05)),
            "q95_delta_v5_minus_v3": float(comparison["score_delta_v5_minus_v3"].quantile(0.95)),
        })

    pd.DataFrame([summary]).to_csv(outdir / "support_roam_score_v5_summary.csv", index=False)
    champion_means(scores).to_csv(outdir / "support_roam_score_v5_champion_means.csv", index=False)
    side_summary(scores).to_csv(outdir / "support_roam_score_v5_side_summary.csv", index=False)
    if not comparison.empty:
        comparison.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_csv(
            outdir / "support_roam_score_v5_vs_v3_describe.csv"
        )
    save_plots(scores, comparison, outdir)

    metadata = {
        "score_col": SCORE_COL,
        "raw_score_col": RAW_SCORE_COL,
        "recipe": {
            "start_minute": args.start_minute,
            "max_minute": args.max_minute,
            "far_adc_threshold": args.far_adc_threshold,
            "w_outside": args.w_outside,
            "w_far": args.w_far,
            "w_xp": args.w_xp,
            "gamma": args.gamma,
            "xp_ratio_min": args.xp_ratio_min,
            "xp_ratio_max": args.xp_ratio_max,
            "min_support_frames": args.min_support_frames,
            "bot_context_zones": sorted(BOT_CONTEXT_ZONES),
            "base_zones": sorted(BASE_ZONES),
        },
        "config_path": str(Path(args.config).resolve()),
        "config_version": config.get("version"),
        "summary": summary,
    }
    (outdir / "support_roam_score_v5_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.export_scores:
        ensure_dir(args.export_dir)
        export_path = Path(args.export_dir) / args.selected_out_name
        keep_cols = [
            "match_id",
            "team_id",
            "side",
            "patch",
            "support_champion_name",
            "adc_champion_name",
            "valid_support_frames_v5",
            "valid_coop_frames_v5",
            "outside_ratio_v5",
            "far_ratio_v5",
            "xp_gap_v5",
            "frames_out_bot_context_v5",
            "frames_in_bot_context_v5",
            "frames_far_from_adc_v5",
            "mean_distance_to_adc_v5",
            "support_adc_xp_ratio_v5",
            "support_score_confidence_v5",
            RAW_SCORE_COL,
            SCORE_COL,
            "variant_id",
            "variant_description",
            "start_minute",
            "max_minute",
            "far_adc_threshold",
            "w_outside",
            "w_far",
            "w_xp",
            "transform",
            "gamma",
        ]
        scores[keep_cols].sort_values(JOIN_KEYS).to_parquet(export_path, index=False)
        print(f"[Exported] {export_path.resolve()}")

    print(f"[Saved] {outdir.resolve()}")
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
