#!/usr/bin/env python3
"""
10_label_error_diagnostics.py -- Diagnose top model errors at label/frame level.

This script links the largest HistGBT test errors back to the frame-state table
used to build the support_roam_score_v5_geometry label. It is designed to answer
whether an outlier looks like:

- a legitimate off-bot / far-from-ADC support pattern,
- a label artifact caused by few valid frames, base/death handling, or
  cooperation-window dropouts,
- or simply a high-variance draft case that the pre-game model cannot predict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "ProgresoActual2" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from build_geometry_v5_frame_state_distributions import classify_chunk_absolute  # noqa: E402


DEFAULT_TOP_ERRORS = REPO_ROOT / "final" / "analysis" / "error_analysis" / "top_abs_errors.csv"
DEFAULT_FRAME_STATE = (
    REPO_ROOT / "final" / "data" / "frame_state" / "support_frame_state.parquet"
)
DEFAULT_SCORES = (
    REPO_ROOT / "final" / "data" / "scores" / "support_scores_v5_geometry_m12.parquet"
)
DEFAULT_CONFIG = REPO_ROOT / "ProgresoActual2" / "data" / "geometry" / "manual_geometry_v5_config.json"
DEFAULT_OUTDIR = REPO_ROOT / "final" / "analysis" / "label_error_diagnostics"

JOIN_KEYS = ["match_id", "team_id"]
TARGET_COL = "support_roam_score"
SCORE_COL = "support_roam_score_v5_geometry"
RAW_SCORE_COL = "raw_support_roam_score_v5_geometry"
BOT_CONTEXT_ZONES = {"BOT_LANE_CORE", "BOT_SIDE_NEAR", "RIVER_BOT", "DRAGON_AREA"}
BASE_ZONES = {"BLUE_BASE", "RED_BASE"}
FAR_ADC_THRESHOLD = 2500.0
WEIGHTS = np.asarray([0.45, 0.35, 0.20], dtype=float)
GAMMA = 0.75
XP_RATIO_MIN = 0.60
XP_RATIO_MAX = 1.00


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose top HistGBT errors via label frame timelines.")
    p.add_argument("--top-errors", default=str(DEFAULT_TOP_ERRORS))
    p.add_argument("--frame-state-path", default=str(DEFAULT_FRAME_STATE))
    p.add_argument("--scores-path", default=str(DEFAULT_SCORES))
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--start-minute", type=float, default=5.0)
    p.add_argument("--max-minute", type=float, default=12.0)
    p.add_argument("--chunk-size", type=int, default=500000)
    return p.parse_args()


def zone_order(config: Dict[str, Any]) -> List[str]:
    order = ["OUT_OF_MAP", "UNCLASSIFIED"] + list(config["colors"].keys())
    for zone in config["priority"]:
        if zone not in order:
            order.append(zone)
    return order


def classify_xy(x: np.ndarray, y: np.ndarray, config: Dict[str, Any], chunk_size: int) -> np.ndarray:
    order = zone_order(config)
    zone_to_id = {zone: idx for idx, zone in enumerate(order)}
    id_to_zone = np.asarray(order, dtype=object)
    out = np.empty(x.shape[0], dtype=np.int16)
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        out[start:end] = classify_chunk_absolute(x[start:end], y[start:end], config, zone_to_id)
    return id_to_zone[out]


def load_top_errors(path: Path, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(path).head(top_n).copy()
    if "actual" not in df.columns and TARGET_COL in df.columns:
        df["actual"] = df[TARGET_COL]
    return df


def load_case_frames(
    frame_state_path: Path,
    cases: pd.DataFrame,
    start_minute: float,
    max_minute: float,
) -> pd.DataFrame:
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
        "support_zone",
        "adc_zone",
        "support_in_base",
        "adc_in_base",
        "support_in_bot_extended",
        "dist_to_adc",
        "support_xp",
        "adc_xp",
    ]
    match_ids = set(cases["match_id"].astype(str))
    team_pairs = set(zip(cases["match_id"].astype(str), cases["team_id"].astype(int)))
    df = pd.read_parquet(frame_state_path, columns=columns)
    df = df[
        df["match_id"].astype(str).isin(match_ids)
        & df["minute"].between(start_minute, max_minute, inclusive="both")
    ].copy()
    df = df[df.apply(lambda r: (str(r["match_id"]), int(r["team_id"])) in team_pairs, axis=1)].copy()
    return df.sort_values(["match_id", "team_id", "frame_idx"]).reset_index(drop=True)


def add_v5_flags(frames: pd.DataFrame, config: Dict[str, Any], chunk_size: int) -> pd.DataFrame:
    out = frames.copy()
    out["support_zone_v5_abs"] = classify_xy(
        out["support_x"].to_numpy(dtype=np.float64),
        out["support_y"].to_numpy(dtype=np.float64),
        config,
        chunk_size,
    )
    out["adc_zone_v5_abs"] = classify_xy(
        out["adc_x"].to_numpy(dtype=np.float64),
        out["adc_y"].to_numpy(dtype=np.float64),
        config,
        chunk_size,
    )
    out["support_in_base_v5"] = out["support_zone_v5_abs"].isin(BASE_ZONES)
    out["adc_in_base_v5"] = out["adc_zone_v5_abs"].isin(BASE_ZONES)
    out["support_in_bot_context_v5"] = out["support_zone_v5_abs"].isin(BOT_CONTEXT_ZONES)
    out["valid_support_frame_v5"] = (
        out["support_alive"].fillna(False)
        & out["support_x"].notna()
        & out["support_y"].notna()
        & ~out["support_in_base_v5"].fillna(False)
    )
    out["valid_coop_frame_v5"] = (
        out["valid_support_frame_v5"]
        & out["adc_alive"].fillna(False)
        & out["adc_x"].notna()
        & out["adc_y"].notna()
        & ~out["adc_in_base_v5"].fillna(False)
    )
    out["out_bot_context_v5"] = out["valid_support_frame_v5"] & ~out["support_in_bot_context_v5"].fillna(False)
    out["far_from_adc_v5"] = out["valid_coop_frame_v5"] & (out["dist_to_adc"].fillna(-1.0) >= FAR_ADC_THRESHOLD)
    out["support_dead_or_base"] = (~out["support_alive"].fillna(False)) | out["support_in_base_v5"].fillna(False)
    out["adc_dead_or_base"] = (~out["adc_alive"].fillna(False)) | out["adc_in_base_v5"].fillna(False)
    out["xp_ratio_frame"] = np.where(
        out["adc_xp"].fillna(0) > 0,
        out["support_xp"] / out["adc_xp"],
        np.nan,
    )
    return out


def xp_gap_from_last(group: pd.DataFrame) -> Tuple[float, float]:
    ordered = group.sort_values("frame_idx")
    support_xp = float(ordered["support_xp"].iloc[-1]) if pd.notna(ordered["support_xp"].iloc[-1]) else np.nan
    adc_xp = float(ordered["adc_xp"].iloc[-1]) if pd.notna(ordered["adc_xp"].iloc[-1]) else np.nan
    if not np.isfinite(adc_xp) or adc_xp <= 0 or not np.isfinite(support_xp):
        return np.nan, np.nan
    ratio = support_xp / adc_xp
    clipped = min(max(ratio, XP_RATIO_MIN), XP_RATIO_MAX)
    gap = 1.0 - ((clipped - XP_RATIO_MIN) / (XP_RATIO_MAX - XP_RATIO_MIN))
    return ratio, gap


def reconstruct_case_components(frames: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (match_id, team_id), group in frames.groupby(JOIN_KEYS, dropna=False):
        support_valid = group[group["valid_support_frame_v5"]]
        coop_valid = group[group["valid_coop_frame_v5"]]
        ratio, xp_gap = xp_gap_from_last(group)
        outside_ratio = (
            float(support_valid["out_bot_context_v5"].mean())
            if len(support_valid) > 0
            else np.nan
        )
        far_ratio = (
            float(coop_valid["far_from_adc_v5"].mean())
            if len(coop_valid) > 0
            else np.nan
        )
        components = np.asarray([outside_ratio, far_ratio, xp_gap], dtype=float)
        valid = np.isfinite(components)
        den = float((WEIGHTS * valid.astype(float)).sum())
        raw = float((np.nan_to_num(components) * WEIGHTS).sum() / den) if den > 0 else np.nan
        score = float(np.clip(raw, 0, 1) ** GAMMA) if np.isfinite(raw) else np.nan
        rows.append(
            {
                "match_id": match_id,
                "team_id": int(team_id),
                "frames_in_window": int(len(group)),
                "valid_support_frames_reconstructed": int(len(support_valid)),
                "valid_coop_frames_reconstructed": int(len(coop_valid)),
                "invalid_support_frames": int((~group["valid_support_frame_v5"]).sum()),
                "invalid_coop_frames": int((group["valid_support_frame_v5"] & ~group["valid_coop_frame_v5"]).sum()),
                "support_dead_or_base_frames": int(group["support_dead_or_base"].sum()),
                "adc_dead_or_base_frames": int(group["adc_dead_or_base"].sum()),
                "outside_ratio_reconstructed": outside_ratio,
                "far_ratio_reconstructed": far_ratio,
                "xp_ratio_reconstructed": ratio,
                "xp_gap_reconstructed": xp_gap,
                "raw_score_reconstructed": raw,
                "score_reconstructed": score,
                "mean_distance_to_adc_reconstructed": float(coop_valid["dist_to_adc"].mean()) if len(coop_valid) else np.nan,
                "max_distance_to_adc": float(group["dist_to_adc"].max()) if group["dist_to_adc"].notna().any() else np.nan,
                "frames_out_bot_context_reconstructed": int(support_valid["out_bot_context_v5"].sum()),
                "frames_far_from_adc_reconstructed": int(coop_valid["far_from_adc_v5"].sum()) if len(coop_valid) else 0,
            }
        )
    return pd.DataFrame(rows)


def load_scores(path: Path, cases: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "match_id",
        "team_id",
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
    ]
    match_ids = set(cases["match_id"].astype(str))
    scores = pd.read_parquet(path, columns=cols)
    scores = scores[scores["match_id"].astype(str).isin(match_ids)].copy()
    pairs = set(zip(cases["match_id"].astype(str), cases["team_id"].astype(int)))
    scores = scores[scores.apply(lambda r: (str(r["match_id"]), int(r["team_id"])) in pairs, axis=1)].copy()
    return scores


def infer_label_diagnostic(row: pd.Series) -> str:
    if row["score_reconstructed_delta"] > 1e-6:
        return "reconstruction_mismatch"
    if row["valid_support_frames_v5"] <= 3:
        return "low_valid_support_frames"
    if row["valid_coop_frames_v5"] <= 3:
        return "low_valid_coop_frames"
    if row["adc_dead_or_base_frames"] >= 3 and row["far_ratio_v5"] >= 0.5:
        return "possible_adc_death_base_coop_artifact"
    if row["outside_ratio_v5"] >= 0.85 and row["far_ratio_v5"] >= 0.85:
        return "consistent_full_roam_label"
    if row["outside_ratio_v5"] >= 0.70:
        return "mostly_outside_bot_context"
    if row["far_ratio_v5"] >= 0.70:
        return "mostly_far_from_adc"
    if row["xp_gap_v5"] >= 0.70:
        return "xp_gap_driven"
    return "mixed_components"


def zone_distribution(frames: pd.DataFrame) -> pd.DataFrame:
    valid = frames[frames["valid_support_frame_v5"]].copy()
    dist = (
        valid.groupby(["match_id", "team_id", "support_zone_v5_abs"], dropna=False)
        .size()
        .reset_index(name="frames")
    )
    totals = dist.groupby(["match_id", "team_id"])["frames"].transform("sum")
    dist["share"] = dist["frames"] / totals
    return dist.sort_values(["match_id", "team_id", "frames"], ascending=[True, True, False])


def write_frame_timeline(frames: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    keep = [
        "error_rank",
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
        "support_zone",
        "adc_zone",
        "support_zone_v5_abs",
        "adc_zone_v5_abs",
        "support_in_bot_context_v5",
        "valid_support_frame_v5",
        "valid_coop_frame_v5",
        "out_bot_context_v5",
        "far_from_adc_v5",
        "support_dead_or_base",
        "adc_dead_or_base",
        "dist_to_adc",
        "support_xp",
        "adc_xp",
        "xp_ratio_frame",
    ]
    timeline = frames[[c for c in keep if c in frames.columns]].copy()
    timeline.to_csv(outdir / "label_error_case_frame_timeline.csv", index=False)
    return timeline


def plot_case_timeline(group: pd.DataFrame, summary_row: pd.Series, outdir: Path) -> str:
    group = group.sort_values("frame_idx")
    zones = list(dict.fromkeys(group["support_zone_v5_abs"].astype(str).tolist()))
    palette = plt.get_cmap("tab20")
    color_map = {zone: palette(i % 20) for i, zone in enumerate(zones)}

    fig, (ax_zone, ax_dist) = plt.subplots(
        2,
        1,
        figsize=(11, 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.5]},
    )

    for _, row in group.iterrows():
        zone = str(row["support_zone_v5_abs"])
        ax_zone.barh(
            [0],
            width=0.85,
            left=float(row["minute"]) - 0.425,
            color=color_map[zone],
            edgecolor="white",
            height=0.6,
        )
        marker = "x" if bool(row["out_bot_context_v5"]) else "."
        ax_zone.text(float(row["minute"]), 0, marker, ha="center", va="center", fontsize=9, color="black")

    ax_zone.set_yticks([])
    ax_zone.set_ylabel("support zone")
    handles = [
        plt.Line2D([0], [0], color=color_map[z], lw=6, label=z)
        for z in zones
    ]
    ax_zone.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7)

    ax_dist.plot(group["minute"], group["dist_to_adc"], marker="o", color="#2f80ed", label="distance to ADC")
    ax_dist.axhline(FAR_ADC_THRESHOLD, color="#eb5757", linestyle="--", linewidth=1.2, label="far threshold")
    far = group[group["far_from_adc_v5"]]
    if not far.empty:
        ax_dist.scatter(far["minute"], far["dist_to_adc"], color="#eb5757", zorder=3, label="far frames")
    ax_dist.set_xlabel("minute")
    ax_dist.set_ylabel("map units")
    ax_dist.grid(alpha=0.25)
    ax_dist.legend(loc="upper left", fontsize=8)

    title = (
        f"#{int(summary_row['error_rank'])} {summary_row['ally_utility_champion_name']}+"
        f"{summary_row['ally_bottom_champion_name']} | pred={summary_row['prediction']:.3f} "
        f"label={summary_row['actual']:.3f} | diag={summary_row['label_diagnostic']}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    filename = f"timeline_case_{int(summary_row['error_rank']):02d}_{summary_row['match_id']}_{int(summary_row['team_id'])}.png"
    filename = filename.replace(":", "_").replace("/", "_").replace("\\", "_")
    fig.savefig(outdir / filename, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return filename


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            display[col] = display[col].fillna("").astype(str)
    headers = list(display.columns)
    rows = display.astype(str).values.tolist()
    widths = [
        max(len(str(header)), *(len(row[i]) for row in rows))
        for i, header in enumerate(headers)
    ]
    lines = [
        "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend(
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    )
    return "\n".join(lines)


def write_markdown(summary: pd.DataFrame, outdir: Path) -> None:
    cols = [
        "error_rank",
        "ally_utility_champion_name",
        "ally_bottom_champion_name",
        "enemy_utility_champion_name",
        "enemy_bottom_champion_name",
        "prediction",
        "actual",
        "abs_error",
        "outside_ratio_v5",
        "far_ratio_v5",
        "xp_gap_v5",
        "valid_support_frames_v5",
        "valid_coop_frames_v5",
        "label_diagnostic",
        "timeline_plot",
    ]
    counts = summary["label_diagnostic"].value_counts().reset_index()
    counts.columns = ["label_diagnostic", "cases"]
    md = [
        "# Label Error Diagnostics",
        "",
        "This report diagnoses the largest HistGBT test errors using the frame-level timeline that produced the v5 support roaming label.",
        "",
        "## Diagnostic Counts",
        "",
        markdown_table(counts),
        "",
        "## Top Error Components",
        "",
        markdown_table(summary[[c for c in cols if c in summary.columns]]),
        "",
        "## Reading",
        "",
        "- `outside_ratio_v5` is the share of valid support frames outside bot context.",
        "- `far_ratio_v5` is the share of valid cooperation frames with support at least 2500 units away from ADC.",
        "- `xp_gap_v5` increases when support XP lags behind ADC XP at the end of the window.",
        "- Cases marked `consistent_full_roam_label` are likely real label extremes, not obvious scoring artifacts.",
        "- Cases marked `low_valid_*` or `possible_adc_death_base_coop_artifact` deserve manual review before using as examples.",
        "",
    ]
    (outdir / "label_error_diagnostics.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cases = load_top_errors(Path(args.top_errors), args.top_n)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    frames = load_case_frames(Path(args.frame_state_path), cases, args.start_minute, args.max_minute)
    frames = add_v5_flags(frames, config, args.chunk_size)

    cases_small_cols = [
        "error_rank",
        "match_id",
        "team_id",
        "prediction",
        "actual",
        "signed_error",
        "abs_error",
        "ally_utility_champion_name",
        "ally_bottom_champion_name",
        "enemy_utility_champion_name",
        "enemy_bottom_champion_name",
    ]
    frames = frames.merge(cases[[c for c in cases_small_cols if c in cases.columns]], on=JOIN_KEYS, how="left")
    timeline = write_frame_timeline(frames, outdir)

    reconstructed = reconstruct_case_components(frames)
    scores = load_scores(Path(args.scores_path), cases)
    summary = (
        cases[[c for c in cases_small_cols if c in cases.columns]]
        .merge(scores, on=JOIN_KEYS, how="left")
        .merge(reconstructed, on=JOIN_KEYS, how="left")
    )
    summary["score_reconstructed_delta"] = (summary[SCORE_COL] - summary["score_reconstructed"]).abs()
    summary["raw_score_reconstructed_delta"] = (
        summary[RAW_SCORE_COL] - summary["raw_score_reconstructed"]
    ).abs()
    summary["label_diagnostic"] = summary.apply(infer_label_diagnostic, axis=1)

    plots: List[str] = []
    for _, row in summary.iterrows():
        group = frames[
            (frames["match_id"] == row["match_id"])
            & (frames["team_id"].astype(int) == int(row["team_id"]))
        ]
        plots.append(plot_case_timeline(group, row, outdir))
    summary["timeline_plot"] = plots

    summary.sort_values("error_rank").to_csv(outdir / "label_error_case_summary.csv", index=False)
    zone_distribution(frames).to_csv(outdir / "zone_distribution_by_case.csv", index=False)
    write_markdown(summary.sort_values("error_rank"), outdir)

    metadata = {
        "top_errors_path": str(Path(args.top_errors).resolve()),
        "frame_state_path": str(Path(args.frame_state_path).resolve()),
        "scores_path": str(Path(args.scores_path).resolve()),
        "config_path": str(Path(args.config).resolve()),
        "outdir": str(outdir.resolve()),
        "top_n": args.top_n,
        "cases": int(len(summary)),
        "frame_rows": int(len(timeline)),
        "start_minute": args.start_minute,
        "max_minute": args.max_minute,
        "formula": {
            "outside_weight": float(WEIGHTS[0]),
            "far_weight": float(WEIGHTS[1]),
            "xp_weight": float(WEIGHTS[2]),
            "gamma": GAMMA,
            "far_adc_threshold": FAR_ADC_THRESHOLD,
            "bot_context_zones": sorted(BOT_CONTEXT_ZONES),
            "base_zones": sorted(BASE_ZONES),
        },
        "diagnostic_counts": summary["label_diagnostic"].value_counts().to_dict(),
        "max_score_reconstruction_delta": float(summary["score_reconstructed_delta"].max()),
        "max_raw_score_reconstruction_delta": float(summary["raw_score_reconstructed_delta"].max()),
    }
    (outdir / "label_error_diagnostics_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[Frames] rows={len(timeline):,} cases={len(summary):,}")
    print(f"[Reconstruction] max_score_delta={metadata['max_score_reconstruction_delta']:.12f}")
    print(f"[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
