#!/usr/bin/env python3
"""
Build full frame-state distributions for manual geometry v5.

The classifier is vectorized over chunks with NumPy. It follows the same
priority semantics as geometry_v5_manual.py, but avoids one Python function
call per frame.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAME_STATE = REPO_ROOT / "ProgresoActual" / "data" / "clean" / "frame_state" / "support_frame_state.parquet"
DEFAULT_CONFIG = REPO_ROOT / "ProgresoActual2" / "data" / "geometry" / "manual_geometry_v5_config.json"
DEFAULT_OUTDIR = REPO_ROOT / "ProgresoActual2" / "analysis" / "geometry_v5_manual" / "frame_state_distributions"

MAP_MAX = 14800.0
MAP_CENTER_SUM = MAP_MAX
BLUE_TEAM_ID = 100
RED_TEAM_ID = 200
ZONE_OUT_OF_MAP = "OUT_OF_MAP"
ZONE_UNCLASSIFIED = "UNCLASSIFIED"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classify support frame-state rows with manual geometry v5.")
    p.add_argument("--frame-state-path", default=str(DEFAULT_FRAME_STATE))
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--start-minute", type=float, default=5.0)
    p.add_argument("--max-minute", type=float, default=12.0)
    p.add_argument("--tag", default=None)
    p.add_argument("--chunk-size", type=int, default=750000)
    p.add_argument("--sample-rows", type=int, default=0, help="Optional smoke cap after filtering. 0 means full.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--export-classified", action="store_true")
    return p.parse_args()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def window_tag(start_minute: float, max_minute: float) -> str:
    return f"m{int(round(start_minute))}_{int(round(max_minute))}"


def as_points(points: Iterable[Iterable[float]]) -> np.ndarray:
    return np.asarray([[float(x), float(y)] for x, y in points], dtype=np.float64)


def points_in_polygon(x: np.ndarray, y: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    inside = np.zeros(x.shape[0], dtype=bool)
    xj, yj = polygon[-1]
    for xi, yi in polygon:
        crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) + 1e-12) + xi)
        inside ^= crosses
        xj, yj = xi, yi
    return inside


def point_to_polyline_distance(x: np.ndarray, y: np.ndarray, centerline: np.ndarray) -> np.ndarray:
    best = np.full(x.shape[0], np.inf, dtype=np.float64)
    for idx in range(len(centerline) - 1):
        ax, ay = centerline[idx]
        bx, by = centerline[idx + 1]
        abx = bx - ax
        aby = by - ay
        denom = abx * abx + aby * aby
        if denom <= 0.0:
            dist = np.hypot(x - ax, y - ay)
        else:
            t = ((x - ax) * abx + (y - ay) * aby) / denom
            t = np.clip(t, 0.0, 1.0)
            qx = ax + t * abx
            qy = ay + t * aby
            dist = np.hypot(x - qx, y - qy)
        best = np.minimum(best, dist)
    return best


def classify_chunk_absolute(x: np.ndarray, y: np.ndarray, config: dict, zone_to_id: Dict[str, int]) -> np.ndarray:
    out = np.full(x.shape[0], zone_to_id[ZONE_UNCLASSIFIED], dtype=np.int16)
    in_map = (x >= 0.0) & (y >= 0.0) & (x <= MAP_MAX) & (y <= MAP_MAX)
    out[~in_map] = zone_to_id[ZONE_OUT_OF_MAP]

    polygons = {zone: as_points(points) for zone, points in config.get("polygons", {}).items()}
    centerlines = config.get("centerline_zones", {})
    circles = config.get("circles", {})

    unresolved = in_map.copy()
    for zone in config["priority"]:
        if not unresolved.any():
            break
        idx = np.flatnonzero(unresolved)
        zx = x[idx]
        zy = y[idx]
        hit = np.zeros(idx.shape[0], dtype=bool)

        if zone in circles:
            cx, cy = circles[zone]["center"]
            radius = float(circles[zone]["radius"])
            hit |= (zx - float(cx)) ** 2 + (zy - float(cy)) ** 2 <= radius ** 2

        if zone in centerlines:
            spec = centerlines[zone]
            centerline = as_points(spec["centerline"])
            hit |= point_to_polyline_distance(zx, zy, centerline) <= float(spec["width"])
            if spec.get("classification_only", False):
                matched = idx[hit]
                out[matched] = zone_to_id[zone]
                unresolved[matched] = False
                continue

        if zone in polygons:
            hit |= points_in_polygon(zx, zy, polygons[zone])

        matched = idx[hit]
        out[matched] = zone_to_id[zone]
        unresolved[matched] = False

    # Match geometry_v5_manual fallback for any in-map holes.
    if unresolved.any():
        idx = np.flatnonzero(unresolved)
        ux = x[idx]
        uy = y[idx]
        blue_side = (ux + uy) < MAP_CENTER_SUM
        top_side = uy >= ux
        fallback = np.empty(idx.shape[0], dtype=np.int16)
        fallback[blue_side & top_side] = zone_to_id["BLUE_TOP_JUNGLE"]
        fallback[blue_side & ~top_side] = zone_to_id["BLUE_BOT_JUNGLE"]
        fallback[~blue_side & top_side] = zone_to_id["RED_TOP_JUNGLE"]
        fallback[~blue_side & ~top_side] = zone_to_id["RED_BOT_JUNGLE"]
        out[idx] = fallback

    return out


def make_relative_labels(zone_abs: np.ndarray, team_id: np.ndarray) -> np.ndarray:
    out = zone_abs.astype(object).copy()
    blue = team_id == BLUE_TEAM_ID
    red = team_id == RED_TEAM_ID

    out[(zone_abs == "BLUE_BASE") & blue] = "OWN_BASE"
    out[(zone_abs == "BLUE_BASE") & red] = "ENEMY_BASE"
    out[(zone_abs == "RED_BASE") & red] = "OWN_BASE"
    out[(zone_abs == "RED_BASE") & blue] = "ENEMY_BASE"

    jungle_specs = [
        ("BLUE_TOP_JUNGLE", "TOP", BLUE_TEAM_ID),
        ("BLUE_BOT_JUNGLE", "BOTTOM", BLUE_TEAM_ID),
        ("RED_TOP_JUNGLE", "TOP", RED_TEAM_ID),
        ("RED_BOT_JUNGLE", "BOTTOM", RED_TEAM_ID),
    ]
    for zone, side, owner_team in jungle_specs:
        mask = zone_abs == zone
        out[mask & (team_id == owner_team)] = f"OWN_{side}_JUNGLE"
        out[mask & (team_id != owner_team)] = f"ENEMY_{side}_JUNGLE"

    out[zone_abs == "TOP_LANE_CORE"] = "TOP_LANE"
    out[zone_abs == "BOT_LANE_CORE"] = "BOTTOM_LANE"
    return out


def classify_dataframe(df: pd.DataFrame, config: dict, chunk_size: int) -> pd.DataFrame:
    zone_order = [ZONE_OUT_OF_MAP, ZONE_UNCLASSIFIED] + list(config["colors"].keys())
    # Preserve priority additions even if a zone has no explicit color.
    for zone in config["priority"]:
        if zone not in zone_order:
            zone_order.append(zone)
    zone_to_id = {zone: idx for idx, zone in enumerate(zone_order)}
    id_to_zone = np.asarray(zone_order, dtype=object)

    x = df["support_x"].to_numpy(dtype=np.float64, copy=False)
    y = df["support_y"].to_numpy(dtype=np.float64, copy=False)
    abs_ids = np.empty(x.shape[0], dtype=np.int16)

    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        abs_ids[start:end] = classify_chunk_absolute(x[start:end], y[start:end], config, zone_to_id)
        print(f"[Classify] rows {end:,}/{x.shape[0]:,}")

    out = df.copy()
    out["support_zone_v5_abs"] = id_to_zone[abs_ids]
    out["support_zone_v5_rel"] = make_relative_labels(
        out["support_zone_v5_abs"].to_numpy(dtype=object),
        out["team_id"].to_numpy(dtype=np.int64, copy=False),
    )
    out["support_in_bot_context_v5"] = out["support_zone_v5_abs"].isin(
        {"BOT_LANE_CORE", "BOT_SIDE_NEAR", "RIVER_BOT", "DRAGON_AREA"}
    )
    return out


def distribution(df: pd.DataFrame, cols: List[str], name: str) -> pd.DataFrame:
    out = df.groupby(cols, dropna=False).size().reset_index(name="frames")
    out["share"] = out["frames"] / max(int(out["frames"].sum()), 1)
    out.insert(0, "distribution", name)
    return out.sort_values("frames", ascending=False)


def numeric_summary(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    rows = [{
        "tag": tag,
        "rows": int(len(df)),
        "match_ids": int(df["match_id"].nunique()),
        "match_team_keys": int(df[["match_id", "team_id"]].drop_duplicates().shape[0]),
        "start_minute": float(df["minute"].min()) if len(df) else math.nan,
        "max_minute": float(df["minute"].max()) if len(df) else math.nan,
        "support_in_bot_context_v5_share": float(df["support_in_bot_context_v5"].mean()) if len(df) else math.nan,
        "legacy_support_in_bot_extended_share": float(df["support_in_bot_extended"].mean()) if "support_in_bot_extended" in df else math.nan,
    }]
    return pd.DataFrame(rows)


def save_plots(df: pd.DataFrame, outdir: Path, tag: str) -> None:
    zone_counts = df["support_zone_v5_abs"].value_counts(dropna=False)
    zone_counts = zone_counts[zone_counts > 0]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.bar(zone_counts.index.astype(str), zone_counts.values, color="#2f80ed")
    ax.set_title(f"Support frames by geometry v5 absolute zone ({tag})")
    ax.set_ylabel("Frames")
    ax.tick_params(axis="x", rotation=55)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / f"zone_v5_abs_distribution_{tag}.png", dpi=180)
    plt.close(fig)

    if "support_zone" in df.columns:
        ct = pd.crosstab(df["support_zone"], df["support_zone_v5_rel"])
        top_rows = ct.sum(axis=1).sort_values(ascending=False).index[:12]
        top_cols = ct.sum(axis=0).sort_values(ascending=False).index[:14]
        matrix = ct.loc[top_rows, top_cols]
        fig, ax = plt.subplots(figsize=(12, 7.5))
        im = ax.imshow(np.log1p(matrix.to_numpy(dtype=float)), cmap="viridis", aspect="auto")
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns, rotation=55, ha="right")
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index)
        ax.set_title(f"Legacy support_zone vs geometry v5 relative zone ({tag}, log frames)")
        fig.colorbar(im, ax=ax, shrink=0.82)
        fig.tight_layout()
        fig.savefig(outdir / f"legacy_vs_v5_relative_heatmap_{tag}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    tag = args.tag or window_tag(args.start_minute, args.max_minute)
    outdir = Path(args.outdir) / tag
    ensure_dir(outdir)

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    columns = [
        "match_id",
        "team_id",
        "side",
        "patch",
        "minute",
        "support_champion_name",
        "support_alive",
        "support_x",
        "support_y",
        "support_zone",
        "support_in_bot_extended",
    ]
    print(f"[Input] frame_state={os.path.abspath(args.frame_state_path)}")
    df = pd.read_parquet(args.frame_state_path, columns=columns)
    df = df[
        df["minute"].between(args.start_minute, args.max_minute, inclusive="both")
        & df["support_alive"].fillna(False)
        & df["support_x"].notna()
        & df["support_y"].notna()
    ].copy()
    if args.sample_rows and len(df) > args.sample_rows:
        df = df.sample(n=args.sample_rows, random_state=args.seed).copy()
    print(f"[Loaded] rows={len(df):,} match_ids={df['match_id'].nunique():,}")

    classified = classify_dataframe(df, config, args.chunk_size)

    summary = numeric_summary(classified, tag)
    summary.to_csv(outdir / f"summary_{tag}.csv", index=False)

    distributions = [
        distribution(classified, ["support_zone_v5_abs"], "zone_v5_abs"),
        distribution(classified, ["support_zone_v5_rel"], "zone_v5_rel"),
        distribution(classified, ["support_zone"], "legacy_zone"),
        distribution(classified, ["support_in_bot_context_v5"], "bot_context_v5"),
        distribution(classified, ["support_zone", "support_zone_v5_rel"], "legacy_zone_to_v5_rel"),
        distribution(classified, ["side", "support_zone_v5_abs"], "side_to_v5_abs"),
        distribution(classified, ["patch", "support_zone_v5_abs"], "patch_to_v5_abs"),
        distribution(classified, ["support_champion_name", "support_zone_v5_abs"], "champion_to_v5_abs"),
    ]
    all_dist = pd.concat(distributions, ignore_index=True)
    all_dist.to_csv(outdir / f"distributions_{tag}.csv", index=False)

    cross = (
        classified.groupby(["support_zone", "support_zone_v5_rel"], dropna=False)
        .size()
        .reset_index(name="frames")
        .sort_values("frames", ascending=False)
    )
    cross["share_of_legacy_zone"] = cross["frames"] / cross.groupby("support_zone")["frames"].transform("sum")
    cross.to_csv(outdir / f"legacy_zone_to_v5_relative_{tag}.csv", index=False)

    save_plots(classified, outdir, tag)

    if args.export_classified:
        keep = [
            "match_id",
            "team_id",
            "side",
            "patch",
            "minute",
            "support_champion_name",
            "support_zone",
            "support_zone_v5_abs",
            "support_zone_v5_rel",
            "support_in_bot_extended",
            "support_in_bot_context_v5",
        ]
        classified[keep].to_parquet(outdir / f"classified_support_frames_v5_{tag}.parquet", index=False)

    metadata = {
        "tag": tag,
        "frame_state_path": os.path.abspath(args.frame_state_path),
        "config_path": str(config_path.resolve()),
        "config_version": config.get("version"),
        "start_minute": args.start_minute,
        "max_minute": args.max_minute,
        "rows": int(len(classified)),
        "chunk_size": args.chunk_size,
        "sample_rows": args.sample_rows,
    }
    (outdir / f"metadata_{tag}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
