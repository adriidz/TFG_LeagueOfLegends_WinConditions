#!/usr/bin/env python3
"""
Plot support frames that are invalid for the v5 bot-context label geometry.

Here "invalid" means the frame contributes as outside bot context in the v5
spatial label: support alive, support has a position, support is not in base,
and the v5 absolute zone is not one of the bot-context zones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROGRESO2_SCRIPTS = REPO_ROOT / "ProgresoActual2" / "scripts"
sys.path.insert(0, str(PROGRESO2_SCRIPTS))

from build_geometry_v5_frame_state_distributions import classify_chunk_absolute  # noqa: E402


DEFAULT_FRAME_STATE = REPO_ROOT / "ProgresoActual" / "data" / "clean" / "frame_state" / "support_frame_state.parquet"
DEFAULT_SCORES = REPO_ROOT / "ProgresoActual2" / "data" / "clean" / "scores" / "support_scores_v5_geometry_m12.parquet"
DEFAULT_CONFIG = REPO_ROOT / "ProgresoActual2" / "data" / "geometry" / "manual_geometry_v5_config.json"
DEFAULT_OUTDIR = REPO_ROOT / "final" / "analysis" / "invalid_frame_positions"

MAP_MAX = 14800.0
BOT_CONTEXT_ZONES = {"BOT_LANE_CORE", "BOT_SIDE_NEAR", "RIVER_BOT", "DRAGON_AREA"}
BASE_ZONES = {"BLUE_BASE", "RED_BASE"}

ZONE_COLORS = {
    "BLUE_BASE": "#4c78a8",
    "RED_BASE": "#9c755f",
    "TOP_LANE_CORE": "#d7b85b",
    "BOT_LANE_CORE": "#2f9e44",
    "TOP_SIDE_NEAR": "#c49a6c",
    "BOT_SIDE_NEAR": "#6cc070",
    "RIVER_TOP": "#76a6b2",
    "RIVER_BOT": "#2d9cdb",
    "BLUE_TOP_JUNGLE": "#7da0c4",
    "BLUE_BOT_JUNGLE": "#5b8db8",
    "RED_TOP_JUNGLE": "#b78396",
    "RED_BOT_JUNGLE": "#b66b6b",
    "MID_LANE": "#8f83b8",
    "BARON_GRUBS_HERALD_AREA": "#6c9a73",
    "DRAGON_AREA": "#1f8a70",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot v5 invalid support frame positions.")
    p.add_argument("--frame-state-path", default=str(DEFAULT_FRAME_STATE))
    p.add_argument("--scores-path", default=str(DEFAULT_SCORES))
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--start-minute", type=float, default=5.0)
    p.add_argument("--max-minute", type=float, default=12.0)
    p.add_argument("--bins", type=int, default=260)
    p.add_argument("--chunk-size", type=int, default=750_000)
    p.add_argument("--sample-points", type=int, default=180_000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def zone_order(config: dict) -> list[str]:
    order = ["OUT_OF_MAP", "UNCLASSIFIED"] + list(config.get("colors", {}).keys())
    for zone in config["priority"]:
        if zone not in order:
            order.append(zone)
    return order


def classify_xy(x: np.ndarray, y: np.ndarray, config: dict, chunk_size: int) -> np.ndarray:
    order = zone_order(config)
    zone_to_id = {zone: idx for idx, zone in enumerate(order)}
    id_to_zone = np.asarray(order, dtype=object)
    out = np.empty(x.shape[0], dtype=np.int16)
    for start in range(0, x.shape[0], chunk_size):
        end = min(start + chunk_size, x.shape[0])
        out[start:end] = classify_chunk_absolute(x[start:end], y[start:end], config, zone_to_id)
        print(f"[Classify support] rows {end:,}/{x.shape[0]:,}")
    return id_to_zone[out]


def load_frames(args: argparse.Namespace, config: dict) -> pd.DataFrame:
    cols = [
        "match_id",
        "team_id",
        "side",
        "patch",
        "frame_idx",
        "minute",
        "support_champion_name",
        "support_alive",
        "support_x",
        "support_y",
    ]
    df = pd.read_parquet(args.frame_state_path, columns=cols)
    df = df[
        df["minute"].between(args.start_minute, args.max_minute, inclusive="both")
        & df["support_alive"].fillna(False).astype(bool)
        & df["support_x"].notna()
        & df["support_y"].notna()
    ].copy()
    print(f"[Loaded] alive support frames with xy: {len(df):,}")

    df["support_zone_v5_abs"] = classify_xy(
        df["support_x"].to_numpy(dtype=np.float64, copy=False),
        df["support_y"].to_numpy(dtype=np.float64, copy=False),
        config,
        args.chunk_size,
    )
    df["in_map_v5"] = df["support_zone_v5_abs"] != "OUT_OF_MAP"
    df["in_base_v5"] = df["support_zone_v5_abs"].isin(BASE_ZONES)
    df["in_bot_context_v5"] = df["support_zone_v5_abs"].isin(BOT_CONTEXT_ZONES)
    df["invalid_for_label_v5"] = df["in_map_v5"] & ~df["in_base_v5"] & ~df["in_bot_context_v5"]
    df["valid_spatial_universe_v5"] = df["in_map_v5"] & ~df["in_base_v5"]
    return df


def add_map(ax: plt.Axes, alpha: float = 0.82) -> None:
    map_path = REPO_ROOT / "images" / "minimapa.png"
    if map_path.exists():
        image = mpimg.imread(map_path)
        ax.imshow(image, extent=(0, MAP_MAX, 0, MAP_MAX), origin="upper", alpha=alpha, zorder=0)
    ax.set_xlim(0, MAP_MAX)
    ax.set_ylim(0, MAP_MAX)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_geometry(ax: plt.Axes, config: dict) -> None:
    bot_context = BOT_CONTEXT_ZONES
    for zone in config["priority"]:
        color = ZONE_COLORS.get(zone, "#777777")
        is_bot_context = zone in bot_context
        alpha = 0.30 if is_bot_context else 0.07
        linewidth = 1.35 if is_bot_context else 0.65
        if zone in config.get("polygons", {}):
            pts = [(float(x), float(y)) for x, y in config["polygons"][zone]]
            ax.add_patch(
                patches.Polygon(
                    pts,
                    closed=True,
                    facecolor=color,
                    edgecolor=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=5 if is_bot_context else 2,
                )
            )
        if zone in config.get("circles", {}):
            circle = config["circles"][zone]
            ax.add_patch(
                patches.Circle(
                    tuple(circle["center"]),
                    float(circle["radius"]),
                    facecolor=color,
                    edgecolor=color,
                    linewidth=linewidth,
                    alpha=alpha,
                    zorder=5 if is_bot_context else 2,
                )
            )
    for zone, spec in config.get("centerline_zones", {}).items():
        line = np.asarray(spec["centerline"], dtype=float)
        color = ZONE_COLORS.get(zone, "#777777")
        is_bot_context = zone in bot_context
        ax.plot(
            line[:, 0],
            line[:, 1],
            color=color,
            linewidth=2.1 if is_bot_context else 0.8,
            alpha=0.92 if is_bot_context else 0.35,
            zorder=6 if is_bot_context else 3,
        )


def hist2d(df: pd.DataFrame, bins: int) -> np.ndarray:
    hist, _, _ = np.histogram2d(
        df["support_x"].to_numpy(dtype=float),
        df["support_y"].to_numpy(dtype=float),
        bins=bins,
        range=[[0, MAP_MAX], [0, MAP_MAX]],
    )
    return hist


def plot_invalid_dashboard(df: pd.DataFrame, config: dict, scores_n: int, args: argparse.Namespace, outdir: Path) -> Path:
    universe = df[df["valid_spatial_universe_v5"]].copy()
    invalid = df[df["invalid_for_label_v5"]].copy()
    valid_bot = universe[universe["in_bot_context_v5"]].copy()
    out_of_map = df[df["support_zone_v5_abs"] == "OUT_OF_MAP"].copy()

    hist_invalid = hist2d(invalid, args.bins)
    hist_valid = hist2d(valid_bot, args.bins)
    invalid_masked = np.ma.masked_where(hist_invalid.T <= 0, hist_invalid.T)
    valid_masked = np.ma.masked_where(hist_valid.T <= 0, hist_valid.T)
    invalid_vmax = float(np.nanpercentile(hist_invalid[hist_invalid > 0], 99.7)) if np.any(hist_invalid > 0) else 2.0
    valid_vmax = float(np.nanpercentile(hist_valid[hist_valid > 0], 99.5)) if np.any(hist_valid > 0) else 2.0

    zone_counts = invalid["support_zone_v5_abs"].value_counts().sort_values(ascending=True)
    top_zone_counts = zone_counts.tail(11)

    fig = plt.figure(figsize=(14.4, 8.2), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.18, 1.18, 0.92], height_ratios=[1.0, 0.42])

    ax_map = fig.add_subplot(gs[:, :2])
    add_map(ax_map, alpha=0.76)
    ax_map.imshow(
        valid_masked,
        extent=(0, MAP_MAX, 0, MAP_MAX),
        origin="lower",
        cmap="Greys",
        alpha=0.35,
        norm=LogNorm(vmin=1, vmax=max(valid_vmax, 2.0)),
        zorder=1,
    )
    im = ax_map.imshow(
        invalid_masked,
        extent=(0, MAP_MAX, 0, MAP_MAX),
        origin="lower",
        cmap="magma",
        alpha=0.82,
        norm=LogNorm(vmin=1, vmax=max(invalid_vmax, 2.0)),
        zorder=4,
    )
    draw_geometry(ax_map, config)
    ax_map.add_patch(
        patches.Rectangle((0, 0), MAP_MAX, MAP_MAX, fill=False, edgecolor="#111111", linewidth=1.25, zorder=10)
    )
    ax_map.set_title("Frames fuera del contexto bot v5 usados por la etiqueta", fontsize=15, pad=10)
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.032, pad=0.018)
    cbar.set_label("Frames invalidos (log)")

    if args.sample_points > 0 and len(invalid) > 0:
        sample = invalid.sample(n=min(args.sample_points, len(invalid)), random_state=args.seed)
        ax_map.scatter(
            sample["support_x"],
            sample["support_y"],
            s=1.0,
            color="#ff2d2d",
            alpha=0.045,
            linewidths=0,
            zorder=7,
        )

    ax_bar = fig.add_subplot(gs[0, 2])
    colors = [ZONE_COLORS.get(z, "#777777") for z in top_zone_counts.index]
    ax_bar.barh(top_zone_counts.index.astype(str), top_zone_counts.values, color=colors)
    ax_bar.set_title("Zonas v5 de los frames invalidos", fontsize=12.5)
    ax_bar.set_xlabel("frames")
    ax_bar.grid(axis="x", alpha=0.22)
    ax_bar.tick_params(axis="y", labelsize=8.8)
    ax_bar.tick_params(axis="x", labelsize=8.8)

    ax_text = fig.add_subplot(gs[1, 2])
    ax_text.axis("off")
    invalid_share = len(invalid) / max(len(universe), 1)
    valid_share = len(valid_bot) / max(len(universe), 1)
    bbox = {"facecolor": "#f7f7f7", "edgecolor": "#dddddd", "boxstyle": "round,pad=0.55"}
    text = (
        f"Observaciones v5: {scores_n:,}\n"
        f"Frames vivos con xy: {len(df):,}\n"
        f"Universo espacial no-base: {len(universe):,}\n"
        f"Invalidos etiqueta v5: {len(invalid):,} ({invalid_share:.2%})\n"
        f"En contexto bot v5: {len(valid_bot):,} ({valid_share:.2%})\n"
        f"Fuera de mapa: {len(out_of_map):,}\n"
        f"Ventana: min {args.start_minute:g}-{args.max_minute:g}"
    )
    ax_text.text(0.02, 0.98, text, ha="left", va="top", fontsize=11.0, bbox=bbox)

    handles = [
        patches.Patch(facecolor="#2f9e44", alpha=0.38, label="Contexto bot v5"),
        patches.Patch(facecolor="#b2182b", alpha=0.78, label="Fuera contexto bot"),
        patches.Patch(facecolor="#777777", alpha=0.22, label="Frames en contexto bot (fondo)"),
    ]
    ax_map.legend(handles=handles, loc="lower left", frameon=True, facecolor="white", edgecolor="#dddddd", framealpha=0.9)

    fig.tight_layout()
    outpath = outdir / f"invalid_support_frame_positions_v5_m{int(args.start_minute)}_{int(args.max_minute)}.png"
    fig.savefig(outpath, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return outpath


def save_outputs(df: pd.DataFrame, scores_n: int, args: argparse.Namespace, outdir: Path, image_path: Path) -> None:
    universe = df[df["valid_spatial_universe_v5"]]
    invalid = df[df["invalid_for_label_v5"]]
    summary = {
        "frame_state_path": str(Path(args.frame_state_path).resolve()),
        "scores_path": str(Path(args.scores_path).resolve()),
        "config_path": str(Path(args.config).resolve()),
        "start_minute": args.start_minute,
        "max_minute": args.max_minute,
        "score_observations": int(scores_n),
        "alive_support_frames_with_xy": int(len(df)),
        "valid_spatial_non_base_frames": int(len(universe)),
        "invalid_for_label_v5_frames": int(len(invalid)),
        "invalid_for_label_v5_share": float(len(invalid) / max(len(universe), 1)),
        "out_of_map_frames": int((df["support_zone_v5_abs"] == "OUT_OF_MAP").sum()),
        "image_path": str(image_path.resolve()),
        "definition": "support alive, xy present, in map, not base, and outside BOT_LANE_CORE/BOT_SIDE_NEAR/RIVER_BOT/DRAGON_AREA",
    }
    (outdir / "invalid_support_frame_positions_v5_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    zone_summary = (
        df[df["invalid_for_label_v5"]]
        .groupby("support_zone_v5_abs", dropna=False)
        .size()
        .reset_index(name="frames")
        .sort_values("frames", ascending=False)
    )
    zone_summary["share_of_invalid"] = zone_summary["frames"] / max(int(zone_summary["frames"].sum()), 1)
    zone_summary.to_csv(outdir / "invalid_support_frame_positions_v5_by_zone.csv", index=False)

    examples = df[df["invalid_for_label_v5"]][
        ["match_id", "team_id", "side", "patch", "frame_idx", "minute", "support_champion_name", "support_x", "support_y", "support_zone_v5_abs"]
    ].head(5000)
    examples.to_csv(outdir / "invalid_support_frame_positions_v5_examples.csv", index=False)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    scores_n = int(pd.read_parquet(args.scores_path, columns=["match_id"]).shape[0])
    df = load_frames(args, config)
    image_path = plot_invalid_dashboard(df, config, scores_n, args, outdir)
    save_outputs(df, scores_n, args, outdir, image_path)
    print(f"[Saved] {image_path.resolve()}")
    print(f"[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
