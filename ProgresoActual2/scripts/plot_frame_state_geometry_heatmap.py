#!/usr/bin/env python3
"""
Plot support position heatmaps from the frozen frame-state with the current
geometry overlaid semi-transparently.

The geometry is imported from ProgresoActual/src/02_data_processing/shared_utils.py
so the plot reflects the exact zones used by the current label pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARED_UTILS_DIR = REPO_ROOT / "ProgresoActual" / "src" / "02_data_processing"
sys.path.insert(0, str(SHARED_UTILS_DIR))

from shared_utils import (  # noqa: E402
    BLUE_BASE_POLYGON,
    BLUE_TEAM_ID,
    BOT_LANE_CENTERLINE,
    BOTTOM_LANE_WIDTH,
    MAP_MAX,
    MID_LANE_CENTERLINE,
    MID_LANE_WIDTH,
    RED_BASE_POLYGON,
    RED_TEAM_ID,
    RIVER_CENTERLINE,
    RIVER_WIDTH,
    TOP_LANE_CENTERLINE,
    TOP_LANE_WIDTH,
    get_team_relative_zone,
)


DEFAULT_FRAME_STATE = REPO_ROOT / "ProgresoActual" / "data" / "clean" / "frame_state" / "support_frame_state.parquet"
DEFAULT_OUTDIR = REPO_ROOT / "ProgresoActual2" / "analysis" / "geometry_heatmap"
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
ZONE_COLORS: Dict[str, str] = {
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
    p = argparse.ArgumentParser(description="Plot frame-state heatmap with current geometry overlay.")
    p.add_argument("--frame-state-path", default=str(DEFAULT_FRAME_STATE))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--start-minute", type=float, default=0.0)
    p.add_argument("--max-minute", type=float, default=14.0)
    p.add_argument("--sample-rows", type=int, default=0, help="Optional cap for faster smoke plots.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bins", type=int, default=220)
    p.add_argument("--geometry-grid", type=int, default=360)
    p.add_argument("--overlay-alpha", type=float, default=0.34)
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
        "figure.titlesize": 14,
    })


def load_support_positions(path: str, start_minute: float, max_minute: float, sample_rows: int, seed: int) -> pd.DataFrame:
    columns = [
        "match_id",
        "team_id",
        "side",
        "minute",
        "support_alive",
        "support_x",
        "support_y",
        "support_in_base",
        "support_zone",
    ]
    if not os.path.exists(path):
        raise SystemExit(f"Missing frame-state parquet: {path}")
    df = pd.read_parquet(path, columns=columns)
    df = df[
        df["minute"].between(start_minute, max_minute, inclusive="both")
        & df["support_alive"].fillna(False)
        & df["support_x"].notna()
        & df["support_y"].notna()
        & ~df["support_in_base"].fillna(False)
    ].copy()
    if sample_rows and sample_rows > 0 and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=seed).copy()
    return df


def make_zone_raster(team_id: int, grid_size: int) -> np.ndarray:
    xs = np.linspace(0, MAP_MAX, grid_size)
    ys = np.linspace(0, MAP_MAX, grid_size)
    zone_to_idx = {zone: idx for idx, zone in enumerate(ZONE_ORDER)}
    raster = np.zeros((grid_size, grid_size), dtype=np.int16)
    for y_idx, y in enumerate(ys):
        for x_idx, x in enumerate(xs):
            zone = get_team_relative_zone(float(x), float(y), team_id)
            raster[y_idx, x_idx] = zone_to_idx.get(zone, 0)
    return raster


def zone_cmap() -> mcolors.ListedColormap:
    return mcolors.ListedColormap([ZONE_COLORS[z] for z in ZONE_ORDER])


def draw_polyline(ax: plt.Axes, points: Iterable[Tuple[float, float]], color: str, label: str, width: float) -> None:
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=2.2, linestyle="--", label=label)
    if width > 0:
        ax.plot(xs, ys, color=color, lw=max(1.0, width / 95.0), alpha=0.12, solid_capstyle="round")


def draw_geometry_lines(ax: plt.Axes) -> None:
    ax.add_patch(patches.Polygon(BLUE_BASE_POLYGON, fill=False, edgecolor="#2f80ed", lw=2.0, label="Base polygons"))
    ax.add_patch(patches.Polygon(RED_BASE_POLYGON, fill=False, edgecolor="#eb5757", lw=2.0))
    draw_polyline(ax, TOP_LANE_CENTERLINE, "#f2c94c", "Top lane centerline", TOP_LANE_WIDTH)
    draw_polyline(ax, MID_LANE_CENTERLINE, "#f2994a", "Mid lane centerline", MID_LANE_WIDTH)
    draw_polyline(ax, BOT_LANE_CENTERLINE, "#27ae60", "Bot lane centerline", BOTTOM_LANE_WIDTH)
    draw_polyline(ax, RIVER_CENTERLINE, "#56ccf2", "River centerline", RIVER_WIDTH)


def draw_common_axes(ax: plt.Axes) -> None:
    ax.set_xlim(0, MAP_MAX)
    ax.set_ylim(0, MAP_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.08)


def add_zone_legend(ax: plt.Axes) -> None:
    handles = [patches.Patch(facecolor=ZONE_COLORS[z], edgecolor="none", alpha=0.75, label=z) for z in ZONE_ORDER]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)


def save_geometry_layer(outdir: str, team_id: int, team_label: str, grid_size: int) -> np.ndarray:
    raster = make_zone_raster(team_id, grid_size)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(raster, extent=(0, MAP_MAX, 0, MAP_MAX), origin="lower", cmap=zone_cmap(), interpolation="nearest", alpha=0.78)
    draw_geometry_lines(ax)
    draw_common_axes(ax)
    ax.set_title(f"Current geometry zones ({team_label} perspective)")
    add_zone_legend(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"geometry_layer_{team_label}.png"), dpi=180)
    plt.close(fig)
    return raster


def save_heatmap(df: pd.DataFrame, outdir: str, bins: int, start_minute: float, max_minute: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    hist = ax.hist2d(
        df["support_x"].to_numpy(dtype=float),
        df["support_y"].to_numpy(dtype=float),
        bins=bins,
        range=[[0, MAP_MAX], [0, MAP_MAX]],
        cmap="inferno",
        norm=mcolors.LogNorm(),
    )
    cb = fig.colorbar(hist[3], ax=ax, shrink=0.82)
    cb.set_label("Support frame density (log)")
    draw_common_axes(ax)
    ax.set_title(f"Support heatmap from frame-state ({start_minute:g}-{max_minute:g} min)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "support_heatmap.png"), dpi=180)
    plt.close(fig)


def save_overlay(
    df: pd.DataFrame,
    raster: np.ndarray,
    outdir: str,
    team_label: str,
    bins: int,
    overlay_alpha: float,
    start_minute: float,
    max_minute: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 9))
    hist = ax.hist2d(
        df["support_x"].to_numpy(dtype=float),
        df["support_y"].to_numpy(dtype=float),
        bins=bins,
        range=[[0, MAP_MAX], [0, MAP_MAX]],
        cmap="inferno",
        norm=mcolors.LogNorm(),
        zorder=1,
    )
    ax.imshow(
        raster,
        extent=(0, MAP_MAX, 0, MAP_MAX),
        origin="lower",
        cmap=zone_cmap(),
        interpolation="nearest",
        alpha=overlay_alpha,
        zorder=2,
    )
    draw_geometry_lines(ax)
    draw_common_axes(ax)
    cb = fig.colorbar(hist[3], ax=ax, shrink=0.82)
    cb.set_label("Support frame density (log)")
    ax.set_title(f"Support heatmap + current geometry ({team_label}, {start_minute:g}-{max_minute:g} min)")
    ax.text(
        0.01,
        0.99,
        "Geometry colors match geometry_layer image",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 3},
    )
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"support_heatmap_geometry_overlay_{team_label}.png"), dpi=180)
    plt.close(fig)


def save_bot_extended_overlay(df: pd.DataFrame, outdir: str, bins: int, start_minute: float, max_minute: float) -> None:
    """Plot current support_in_bot_extended labels as point overlay for quick sanity checking."""
    sample = df.copy()
    if len(sample) > 200000:
        sample = sample.sample(n=200000, random_state=42)
    fig, ax = plt.subplots(figsize=(9.8, 9))
    ax.hist2d(
        df["support_x"].to_numpy(dtype=float),
        df["support_y"].to_numpy(dtype=float),
        bins=bins,
        range=[[0, MAP_MAX], [0, MAP_MAX]],
        cmap="Greys",
        norm=mcolors.LogNorm(),
        alpha=0.75,
    )
    in_bot = sample["support_zone"].isin(["BOTTOM_LANE", "OWN_BOTTOM_JUNGLE", "RIVER_BOT"])
    ax.scatter(sample.loc[in_bot, "support_x"], sample.loc[in_bot, "support_y"], s=1.2, alpha=0.08, color="#27ae60", label="Current in-bot-extended")
    ax.scatter(sample.loc[~in_bot, "support_x"], sample.loc[~in_bot, "support_y"], s=1.2, alpha=0.05, color="#eb5757", label="Current out-of-bot")
    draw_geometry_lines(ax)
    draw_common_axes(ax)
    ax.set_title(f"Current support_in_bot_extended sanity overlay ({start_minute:g}-{max_minute:g} min)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "support_in_bot_extended_point_overlay.png"), dpi=180)
    plt.close(fig)


def main() -> None:
    configure_plot_style()
    args = parse_args()
    ensure_dir(args.outdir)
    df = load_support_positions(args.frame_state_path, args.start_minute, args.max_minute, args.sample_rows, args.seed)
    if df.empty:
        raise SystemExit("No support positions available after filters.")
    print(f"[Loaded] support frames: {len(df):,}")
    print(f"[Output] {os.path.abspath(args.outdir)}")

    save_heatmap(df, args.outdir, args.bins, args.start_minute, args.max_minute)
    blue_raster = save_geometry_layer(args.outdir, BLUE_TEAM_ID, "blue", args.geometry_grid)
    red_raster = save_geometry_layer(args.outdir, RED_TEAM_ID, "red", args.geometry_grid)
    save_overlay(df, blue_raster, args.outdir, "blue", args.bins, args.overlay_alpha, args.start_minute, args.max_minute)
    save_overlay(df, red_raster, args.outdir, "red", args.bins, args.overlay_alpha, args.start_minute, args.max_minute)
    save_bot_extended_overlay(df, args.outdir, args.bins, args.start_minute, args.max_minute)


if __name__ == "__main__":
    main()
