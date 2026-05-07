#!/usr/bin/env python3
"""
Build an observed map-shape heatmap from all player positions in raw timelines
and overlay the current geometry as opaque outlines.

This intentionally does not use any external/internet map image. The heatmap is
derived from Riot timeline coordinates observed in the local raw dataset.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import random
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_UTILS_DIR = REPO_ROOT / "ProgresoActual" / "src" / "02_data_processing"
sys.path.insert(0, str(SHARED_UTILS_DIR))

from shared_utils import (  # noqa: E402
    BLUE_BASE_POLYGON,
    BLUE_TEAM_ID,
    BOT_LANE_CENTERLINE,
    MAP_MAX,
    MID_LANE_CENTERLINE,
    RED_BASE_POLYGON,
    RED_TEAM_ID,
    RIVER_CENTERLINE,
    TOP_LANE_CENTERLINE,
    extract_position,
    get_participant_frame,
    get_team_relative_zone,
    get_timeline_frames,
    list_match_dirs,
    load_json,
    participant_is_alive,
)


DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw" / "raw"
DEFAULT_OUTDIR = REPO_ROOT / "ProgresoActual2" / "analysis" / "geometry_heatmap_all_players"
ZONE_ORDER = [
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
ZONE_COLORS = {
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
    p = argparse.ArgumentParser(description="Plot all-player observed heatmap with geometry outlines.")
    p.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--region", default="europe")
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--start-minute", type=float, default=0.0)
    p.add_argument("--max-minute", type=float, default=14.0)
    p.add_argument("--max-matches", type=int, default=50000,
                   help="0 means all available matches. Default uses more than the first 10k while staying practical.")
    p.add_argument("--shuffle", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=1, help="1 disables multiprocessing; use >1 only when allowed.")
    p.add_argument("--bins", type=int, default=260)
    p.add_argument("--geometry-grid", type=int, default=420)
    p.add_argument("--heatmap-alpha", type=float, default=0.58)
    p.add_argument("--outline-width", type=float, default=3.2)
    p.add_argument("--centerline-width", type=float, default=3.8)
    p.add_argument("--include-dead", action="store_true",
                   help="Include participant positions even when the frame marks them dead.")
    return p.parse_args()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
    })


def process_match(args_tuple: Tuple[str, float, float, bool]) -> Tuple[List[float], List[float]]:
    match_dir, start_minute, max_minute, include_dead = args_tuple
    timeline_path = os.path.join(match_dir, "timeline.json")
    if not os.path.exists(timeline_path):
        return [], []
    try:
        timeline = load_json(timeline_path)
    except Exception:
        return [], []
    xs: List[float] = []
    ys: List[float] = []
    for frame in get_timeline_frames(timeline):
        timestamp = frame.get("timestamp")
        minute = float(timestamp) / 60000.0 if timestamp is not None else None
        if minute is None or minute < start_minute or minute > max_minute:
            continue
        for participant_id in range(1, 11):
            pf = get_participant_frame(frame, participant_id)
            if not include_dead and not participant_is_alive(pf):
                continue
            pos = extract_position(pf)
            if not pos:
                continue
            x, y = pos
            if 0.0 <= float(x) <= MAP_MAX and 0.0 <= float(y) <= MAP_MAX:
                xs.append(float(x))
                ys.append(float(y))
    return xs, ys


def collect_heatmap(
    raw_root: str,
    region: str,
    start_minute: float,
    max_minute: float,
    max_matches: int,
    shuffle: bool,
    seed: int,
    workers: int,
    include_dead: bool,
    bins: int,
) -> Tuple[np.ndarray, int, int]:
    raw_base = os.path.join(raw_root, region)
    match_dirs = list_match_dirs(raw_base)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(match_dirs)
    if max_matches and max_matches > 0:
        match_dirs = match_dirs[:max_matches]
    tasks = [(mdir, start_minute, max_minute, include_dead) for mdir in match_dirs]
    heatmap = np.zeros((bins, bins), dtype=np.float64)
    total_positions = 0
    if workers == 1:
        iterator = map(process_match, tasks)
    else:
        max_workers = workers if workers and workers > 0 else min(8, os.cpu_count() or 1)
        pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
        iterator = pool.map(process_match, tasks, chunksize=50)
    try:
        for idx, (xs, ys) in enumerate(iterator, start=1):
            if xs and ys:
                hist, _, _ = np.histogram2d(
                    np.asarray(xs, dtype=np.float32),
                    np.asarray(ys, dtype=np.float32),
                    bins=bins,
                    range=[[0, MAP_MAX], [0, MAP_MAX]],
                )
                heatmap += hist
                total_positions += len(xs)
            if idx % 1000 == 0:
                print(f"[Collect] matches={idx}/{len(tasks)} positions={total_positions:,}")
    finally:
        if "pool" in locals():
            pool.shutdown()
    return heatmap, total_positions, len(match_dirs)


def make_zone_raster(team_id: int, grid_size: int) -> np.ndarray:
    xs = np.linspace(0, MAP_MAX, grid_size)
    ys = np.linspace(0, MAP_MAX, grid_size)
    zone_to_idx = {zone: idx for idx, zone in enumerate(ZONE_ORDER)}
    raster = np.zeros((grid_size, grid_size), dtype=np.int16)
    for y_idx, y in enumerate(ys):
        for x_idx, x in enumerate(xs):
            raster[y_idx, x_idx] = zone_to_idx[get_team_relative_zone(float(x), float(y), team_id)]
    return raster


def draw_polyline(ax: plt.Axes, points: Sequence[Tuple[float, float]], color: str, label: str, width: float) -> None:
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=width, linestyle="--", label=label, zorder=6)


def draw_reference_lines(ax: plt.Axes, centerline_width: float) -> None:
    ax.plot([0, MAP_MAX], [0, MAP_MAX], color="#ffffff", lw=1.2, alpha=0.55, zorder=5)
    ax.plot([0, MAP_MAX], [MAP_MAX, 0], color="#ffffff", lw=1.2, alpha=0.55, zorder=5)
    draw_polyline(ax, TOP_LANE_CENTERLINE, "#f2c94c", "Top lane centerline", centerline_width)
    draw_polyline(ax, MID_LANE_CENTERLINE, "#f2994a", "Mid lane centerline", centerline_width)
    draw_polyline(ax, BOT_LANE_CENTERLINE, "#27ae60", "Bot lane centerline", centerline_width)
    draw_polyline(ax, RIVER_CENTERLINE, "#56ccf2", "River centerline", centerline_width)


def draw_base_outlines(ax: plt.Axes, outline_width: float) -> None:
    ax.add_patch(plt.Polygon(BLUE_BASE_POLYGON, fill=False, edgecolor="#2f80ed", lw=outline_width, zorder=7, label="Base polygons"))
    ax.add_patch(plt.Polygon(RED_BASE_POLYGON, fill=False, edgecolor="#eb5757", lw=outline_width, zorder=7))


def draw_zone_outlines(ax: plt.Axes, raster: np.ndarray, outline_width: float, centerline_width: float) -> None:
    for idx, zone in enumerate(ZONE_ORDER):
        mask = (raster == idx).astype(float)
        ax.contour(
            mask,
            levels=[0.5],
            extent=(0, MAP_MAX, 0, MAP_MAX),
            origin="lower",
            colors=[ZONE_COLORS[zone]],
            linewidths=outline_width,
            alpha=1.0,
            zorder=5,
        )
    draw_base_outlines(ax, outline_width)
    draw_reference_lines(ax, centerline_width)


def draw_common_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0, MAP_MAX)
    ax.set_ylim(0, MAP_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.08)


def add_line_legend(ax: plt.Axes) -> None:
    handles = [
        plt.Line2D([0], [0], color=ZONE_COLORS[z], lw=2.0, label=z)
        for z in ["TOP_LANE", "MID_LANE", "BOTTOM_LANE", "RIVER_TOP", "OWN_BOTTOM_JUNGLE", "ENEMY_BOTTOM_JUNGLE"]
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.88)


def plot_heatmap_array(ax: plt.Axes, heatmap: np.ndarray, heatmap_alpha: float):
    masked = np.ma.masked_where(heatmap.T <= 0, heatmap.T)
    return ax.imshow(
        masked,
        extent=(0, MAP_MAX, 0, MAP_MAX),
        origin="lower",
        cmap="inferno",
        norm=mcolors.LogNorm(vmin=1, vmax=max(1.0, float(heatmap.max()))),
        alpha=heatmap_alpha,
        interpolation="nearest",
        zorder=1,
    )


def save_heatmap(heatmap: np.ndarray, outdir: str, start_minute: float, max_minute: float, heatmap_alpha: float) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 9))
    img = plot_heatmap_array(ax, heatmap, heatmap_alpha)
    cb = fig.colorbar(img, ax=ax, shrink=0.82)
    cb.set_label("All-player frame density (log)")
    draw_common_axes(ax)
    ax.set_title(f"Observed all-player map heatmap ({start_minute:g}-{max_minute:g} min)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "all_players_heatmap.png"), dpi=190)
    plt.close(fig)


def save_overlay(
    heatmap: np.ndarray,
    raster: np.ndarray,
    outdir: str,
    team_label: str,
    start_minute: float,
    max_minute: float,
    heatmap_alpha: float,
    outline_width: float,
    centerline_width: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 9))
    img = plot_heatmap_array(ax, heatmap, heatmap_alpha)
    draw_zone_outlines(ax, raster, outline_width, centerline_width)
    cb = fig.colorbar(img, ax=ax, shrink=0.82)
    cb.set_label("All-player frame density (log)")
    draw_common_axes(ax)
    add_line_legend(ax)
    ax.set_title(f"Observed all-player heatmap + geometry outlines ({team_label}, {start_minute:g}-{max_minute:g} min)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"all_players_heatmap_geometry_outlines_{team_label}.png"), dpi=190)
    plt.close(fig)


def save_outlines_only(raster: np.ndarray, outdir: str, team_label: str, outline_width: float, centerline_width: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    draw_zone_outlines(ax, raster, outline_width, centerline_width)
    draw_common_axes(ax)
    add_line_legend(ax)
    ax.set_title(f"Current geometry outlines ({team_label} perspective)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"geometry_outlines_{team_label}.png"), dpi=190)
    plt.close(fig)


def main() -> None:
    configure_plot_style()
    args = parse_args()
    ensure_dir(args.outdir)
    heatmap, n_positions, n_matches = collect_heatmap(
        raw_root=args.raw_root,
        region=args.region,
        start_minute=args.start_minute,
        max_minute=args.max_minute,
        max_matches=args.max_matches,
        shuffle=args.shuffle,
        seed=args.seed,
        workers=args.workers,
        include_dead=args.include_dead,
        bins=args.bins,
    )
    if n_positions == 0:
        raise SystemExit("No positions collected from raw timelines.")
    print(f"[Loaded] matches={n_matches:,} positions={n_positions:,}")
    print(f"[Output] {os.path.abspath(args.outdir)}")
    save_heatmap(heatmap, args.outdir, args.start_minute, args.max_minute, args.heatmap_alpha)
    for team_id, team_label in [(BLUE_TEAM_ID, "blue"), (RED_TEAM_ID, "red")]:
        raster = make_zone_raster(team_id, args.geometry_grid)
        save_outlines_only(raster, args.outdir, team_label, args.outline_width, args.centerline_width)
        save_overlay(
            heatmap,
            raster,
            args.outdir,
            team_label,
            args.start_minute,
            args.max_minute,
            args.heatmap_alpha,
            args.outline_width,
            args.centerline_width,
        )


if __name__ == "__main__":
    main()
