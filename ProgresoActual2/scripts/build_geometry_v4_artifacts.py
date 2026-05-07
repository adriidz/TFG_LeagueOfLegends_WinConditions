#!/usr/bin/env python3
"""
Build observed walkable masks and geometry_v4 comparison artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.ndimage import binary_closing, binary_opening, gaussian_filter
except Exception:  # pragma: no cover - fallback for minimal envs
    binary_closing = binary_opening = gaussian_filter = None


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "ProgresoActual2" / "scripts"
GEOMETRY_SRC_DIR = REPO_ROOT / "ProgresoActual2" / "src" / "geometry"
SHARED_UTILS_DIR = REPO_ROOT / "ProgresoActual" / "src" / "02_data_processing"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(GEOMETRY_SRC_DIR))
sys.path.insert(0, str(SHARED_UTILS_DIR))

from plot_all_players_geometry_heatmap import collect_heatmap  # noqa: E402
from shared_utils import (  # noqa: E402
    BLUE_TEAM_ID,
    MAP_MAX,
    RED_TEAM_ID,
    get_team_relative_zone,
)
from geometry_v4 import (  # noqa: E402
    ZONE_ORDER_V4,
    ZONE_TO_ID_V4,
    classify_zone_v4,
)


DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw" / "raw"
DEFAULT_GEOMETRY_DIR = REPO_ROOT / "ProgresoActual2" / "data" / "geometry"
DEFAULT_ANALYSIS_ROOT = REPO_ROOT / "ProgresoActual2" / "analysis" / "geometry_v4"
DEFAULT_FRAME_STATE = REPO_ROOT / "ProgresoActual" / "data" / "clean" / "frame_state" / "support_frame_state.parquet"

ZONE_ORDER_V2 = [
    "OWN_BASE",
    "ENEMY_BASE",
    "TOP_LANE",
    "MID_LANE",
    "BOTTOM_LANE",
    "RIVER_TOP",
    "RIVER_BOT",
    "OWN_TOP_JUNGLE",
    "OWN_BOTTOM_JUNGLE",
    "ENEMY_TOP_JUNGLE",
    "ENEMY_BOTTOM_JUNGLE",
]
ZONE_COLORS_V4: Dict[str, str] = {
    "UNWALKABLE": "#f2f2f2",
    "OWN_BASE": "#2f80ed",
    "ENEMY_BASE": "#eb5757",
    "TOP_LANE": "#f2c94c",
    "MID_LANE": "#f2994a",
    "BOT_LANE_CORE": "#27ae60",
    "RIVER_TOP": "#56ccf2",
    "RIVER_BOT": "#2d9cdb",
    "DRAGON_AREA": "#00a676",
    "GRUBS_HERALD_AREA": "#8e44ad",
    "BARON_AREA": "#5e3370",
    "BOT_SIDE_NEAR": "#7bc96f",
    "OWN_TOP_JUNGLE": "#9b51e0",
    "OWN_BOTTOM_JUNGLE": "#bb6bd9",
    "ENEMY_TOP_JUNGLE": "#6fcf97",
    "ENEMY_BOTTOM_JUNGLE": "#219653",
}
ZONE_COLORS_V2: Dict[str, str] = {
    "OWN_BASE": "#2f80ed",
    "ENEMY_BASE": "#eb5757",
    "TOP_LANE": "#f2c94c",
    "MID_LANE": "#f2994a",
    "BOTTOM_LANE": "#27ae60",
    "RIVER_TOP": "#56ccf2",
    "RIVER_BOT": "#2d9cdb",
    "OWN_TOP_JUNGLE": "#9b51e0",
    "OWN_BOTTOM_JUNGLE": "#bb6bd9",
    "ENEMY_TOP_JUNGLE": "#6fcf97",
    "ENEMY_BOTTOM_JUNGLE": "#219653",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build geometry_v4 density, mask and comparison artifacts.")
    p.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--region", default="europe")
    p.add_argument("--start-minute", type=float, default=0.0)
    p.add_argument("--max-minute", type=float, default=14.0)
    p.add_argument("--max-matches", type=int, default=50000, help="0 means all available matches.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bins", type=int, default=260)
    p.add_argument("--smooth-sigma", type=float, default=1.15)
    p.add_argument("--threshold-quantile", type=float, default=0.10)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--geometry-dir", default=str(DEFAULT_GEOMETRY_DIR))
    p.add_argument("--analysis-root", default=str(DEFAULT_ANALYSIS_ROOT))
    p.add_argument("--frame-state-path", default=str(DEFAULT_FRAME_STATE))
    p.add_argument("--support-max-rows", type=int, default=300000)
    return p.parse_args()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def window_tag(start_minute: float, max_minute: float) -> str:
    return f"{int(round(start_minute))}_{int(round(max_minute))}"


def smooth_density(heatmap: np.ndarray, sigma: float) -> np.ndarray:
    if gaussian_filter is None:
        return heatmap.astype(float)
    return gaussian_filter(heatmap.astype(float), sigma=sigma)


def build_walkable_mask(smooth: np.ndarray, threshold_quantile: float) -> Tuple[np.ndarray, float]:
    positive = smooth[smooth > 0]
    if positive.size == 0:
        raise SystemExit("Cannot build walkable mask from empty density.")
    threshold = max(1.0, float(np.quantile(positive, threshold_quantile)))
    mask = smooth >= threshold
    if binary_closing is not None:
        mask = binary_closing(mask, iterations=1)
    return mask.astype(bool), threshold


def plot_heatmap(ax: plt.Axes, heatmap: np.ndarray, alpha: float = 0.55):
    masked = np.ma.masked_where(heatmap.T <= 0, heatmap.T)
    return ax.imshow(
        masked,
        extent=(0, MAP_MAX, 0, MAP_MAX),
        origin="lower",
        cmap="inferno",
        norm=mcolors.LogNorm(vmin=1, vmax=max(1.0, float(heatmap.max()))),
        interpolation="nearest",
        alpha=alpha,
    )


def common_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0, MAP_MAX)
    ax.set_ylim(0, MAP_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.08)


def make_v2_raster(team_id: int, grid_size: int) -> np.ndarray:
    xs = np.linspace(0, MAP_MAX, grid_size)
    ys = np.linspace(0, MAP_MAX, grid_size)
    zone_to_idx = {zone: idx for idx, zone in enumerate(ZONE_ORDER_V2)}
    raster = np.zeros((grid_size, grid_size), dtype=np.int16)
    for y_idx, y in enumerate(ys):
        for x_idx, x in enumerate(xs):
            raster[y_idx, x_idx] = zone_to_idx[get_team_relative_zone(float(x), float(y), team_id)]
    return raster


def make_v4_raster(team_id: int, grid_size: int, mask_path: str) -> np.ndarray:
    xs = np.linspace(0, MAP_MAX, grid_size)
    ys = np.linspace(0, MAP_MAX, grid_size)
    raster = np.zeros((grid_size, grid_size), dtype=np.int16)
    for y_idx, y in enumerate(ys):
        for x_idx, x in enumerate(xs):
            zone = classify_zone_v4(float(x), float(y), team_id, mask_path=mask_path)
            raster[y_idx, x_idx] = ZONE_TO_ID_V4.get(zone, 0)
    return raster


def draw_outlines(ax: plt.Axes, raster: np.ndarray, zones: List[str], colors: Dict[str, str], width: float = 2.6) -> None:
    for idx, zone in enumerate(zones):
        if zone == "UNWALKABLE":
            continue
        mask = (raster == idx).astype(float)
        if mask.max() <= 0:
            continue
        ax.contour(
            mask,
            levels=[0.5],
            extent=(0, MAP_MAX, 0, MAP_MAX),
            origin="lower",
            colors=[colors.get(zone, "#333333")],
            linewidths=width,
            alpha=1.0,
        )


def save_density_and_mask_plots(heatmap: np.ndarray, smooth: np.ndarray, mask: np.ndarray, outdir: str, tag: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    img = plot_heatmap(ax, heatmap, alpha=0.85)
    fig.colorbar(img, ax=ax, shrink=0.82, label="All-player frame density (log)")
    common_axes(ax)
    ax.set_title(f"Observed density {tag}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "observed_density.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(mask.T, extent=(0, MAP_MAX, 0, MAP_MAX), origin="lower", cmap="Greys", interpolation="nearest")
    common_axes(ax)
    ax.set_title(f"Observed walkable mask {tag}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "geometry_v4_walkable_mask.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 9))
    plot_heatmap(ax, heatmap, alpha=0.55)
    ax.contour(mask.T.astype(float), levels=[0.5], extent=(0, MAP_MAX, 0, MAP_MAX), origin="lower", colors=["#00ff9f"], linewidths=2.5)
    common_axes(ax)
    ax.set_title(f"Walkable mask over density {tag}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "geometry_v4_walkable_mask_on_heatmap.png"), dpi=180)
    plt.close(fig)

    np.save(os.path.join(outdir, "smooth_density_preview.npy"), smooth.astype(np.float32))


def save_zone_layer(raster: np.ndarray, outdir: str, team_label: str) -> None:
    colors = [ZONE_COLORS_V4[z] for z in ZONE_ORDER_V4]
    cmap = mcolors.ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(10.5, 9))
    ax.imshow(raster, extent=(0, MAP_MAX, 0, MAP_MAX), origin="lower", cmap=cmap, interpolation="nearest")
    common_axes(ax)
    handles = [patches.Patch(facecolor=ZONE_COLORS_V4[z], label=z) for z in ZONE_ORDER_V4 if z != "UNWALKABLE"]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_title(f"Geometry v4 zone layer ({team_label})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"geometry_v4_zone_layer_{team_label}.png"), dpi=180)
    plt.close(fig)


def save_outline_comparison(
    heatmap: np.ndarray,
    v2_raster: np.ndarray,
    v4_raster: np.ndarray,
    outdir: str,
    team_label: str,
    tag: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 9))
    plot_heatmap(ax, heatmap, alpha=0.52)
    draw_outlines(ax, v2_raster, ZONE_ORDER_V2, ZONE_COLORS_V2, width=2.6)
    common_axes(ax)
    ax.set_title(f"Geometry v2 outlines on heatmap ({team_label}, {tag})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"geometry_v2_outlines_on_heatmap_{team_label}.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.6, 9))
    plot_heatmap(ax, heatmap, alpha=0.52)
    draw_outlines(ax, v4_raster, ZONE_ORDER_V4, ZONE_COLORS_V4, width=2.6)
    common_axes(ax)
    ax.set_title(f"Geometry v4 outlines on heatmap ({team_label}, {tag})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"geometry_v4_outlines_on_heatmap_{team_label}.png"), dpi=180)
    plt.close(fig)


def classify_support_sample(frame_state_path: str, mask_path: str, outdir: str, max_rows: int) -> None:
    path = Path(frame_state_path)
    if not path.exists():
        return
    cols = ["match_id", "team_id", "side", "minute", "support_alive", "support_x", "support_y", "support_zone"]
    df = pd.read_parquet(path, columns=cols)
    df = df[
        df["minute"].between(5, 12, inclusive="both")
        & df["support_alive"].fillna(False)
        & df["support_x"].notna()
        & df["support_y"].notna()
    ].copy()
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).copy()
    zones = [
        classify_zone_v4(row.support_x, row.support_y, int(row.team_id), mask_path=mask_path)
        for row in df.itertuples(index=False)
    ]
    df["zone_v4"] = zones
    summary = (
        df.groupby(["support_zone", "zone_v4"], dropna=False)
        .size()
        .reset_index(name="frames")
        .sort_values("frames", ascending=False)
    )
    summary.to_csv(os.path.join(outdir, "support_zone_v2_to_v4_sample.csv"), index=False)
    coverage = pd.DataFrame([{
        "sample_rows": int(len(df)),
        "v4_classified_share": float((df["zone_v4"] != "UNWALKABLE").mean()),
        "unwalkable_rows": int((df["zone_v4"] == "UNWALKABLE").sum()),
    }])
    coverage.to_csv(os.path.join(outdir, "support_zone_v4_sample_coverage.csv"), index=False)


def write_metadata(outdir: str, payload: dict) -> None:
    with open(os.path.join(outdir, "geometry_v4_build_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    tag = window_tag(args.start_minute, args.max_minute)
    geometry_dir = Path(args.geometry_dir)
    outdir = Path(args.analysis_root) / f"m{tag}"
    ensure_dir(geometry_dir)
    ensure_dir(outdir)

    heatmap, n_positions, n_matches = collect_heatmap(
        raw_root=args.raw_root,
        region=args.region,
        start_minute=args.start_minute,
        max_minute=args.max_minute,
        max_matches=args.max_matches,
        shuffle=True,
        seed=args.seed,
        workers=args.workers,
        include_dead=False,
        bins=args.bins,
    )
    smooth = smooth_density(heatmap, args.smooth_sigma)
    mask, threshold = build_walkable_mask(smooth, args.threshold_quantile)

    density_path = geometry_dir / f"observed_player_density_{tag}.npz"
    mask_path = geometry_dir / f"observed_walkable_mask_{tag}.npz"
    np.savez_compressed(
        density_path,
        heatmap=heatmap.astype(np.float32),
        smooth_density=smooth.astype(np.float32),
        bins=np.asarray(args.bins),
        map_max=np.asarray(MAP_MAX),
        start_minute=np.asarray(args.start_minute),
        max_minute=np.asarray(args.max_minute),
        max_matches=np.asarray(args.max_matches),
        positions=np.asarray(n_positions),
    )
    np.savez_compressed(
        mask_path,
        walkable_mask=mask,
        smooth_density=smooth.astype(np.float32),
        bins=np.asarray(args.bins),
        map_max=np.asarray(MAP_MAX),
        threshold=np.asarray(threshold),
        threshold_quantile=np.asarray(args.threshold_quantile),
        smooth_sigma=np.asarray(args.smooth_sigma),
        start_minute=np.asarray(args.start_minute),
        max_minute=np.asarray(args.max_minute),
    )

    save_density_and_mask_plots(heatmap, smooth, mask, str(outdir), tag)
    for team_id, team_label in [(BLUE_TEAM_ID, "blue"), (RED_TEAM_ID, "red")]:
        v2_raster = make_v2_raster(team_id, args.bins)
        v4_raster = make_v4_raster(team_id, args.bins, str(mask_path))
        save_zone_layer(v4_raster, str(outdir), team_label)
        save_outline_comparison(heatmap, v2_raster, v4_raster, str(outdir), team_label, tag)

    classify_support_sample(args.frame_state_path, str(mask_path), str(outdir), args.support_max_rows)
    write_metadata(str(outdir), {
        "density_path": str(density_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "start_minute": args.start_minute,
        "max_minute": args.max_minute,
        "max_matches": args.max_matches,
        "n_matches": n_matches,
        "n_positions": n_positions,
        "bins": args.bins,
        "smooth_sigma": args.smooth_sigma,
        "threshold_quantile": args.threshold_quantile,
        "threshold": threshold,
        "walkable_share": float(mask.mean()),
    })
    print(f"[Saved] {density_path.resolve()}")
    print(f"[Saved] {mask_path.resolve()}")
    print(f"[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
