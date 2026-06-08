#!/usr/bin/env python3
"""
Plot positions for frames excluded by the v5 label calculation filters.

This answers a different question than "outside bot context": it shows frames
that cannot be used in the per-frame denominator because the support or ADC
state is invalid for the relevant component.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PROGRESO2_SCRIPTS = REPO_ROOT / "ProgresoActual2" / "scripts"
sys.path.insert(0, str(PROGRESO2_SCRIPTS))

from build_geometry_v5_frame_state_distributions import classify_chunk_absolute  # noqa: E402


DEFAULT_FRAME_STATE = REPO_ROOT / "ProgresoActual" / "data" / "clean" / "frame_state" / "support_frame_state.parquet"
DEFAULT_CONFIG = REPO_ROOT / "ProgresoActual2" / "data" / "geometry" / "manual_geometry_v5_config.json"
DEFAULT_OUTDIR = REPO_ROOT / "final" / "analysis" / "invalid_frame_positions"

MAP_MAX = 14800.0
BASE_ZONES = {"BLUE_BASE", "RED_BASE"}

SUPPORT_REASON_ORDER = [
    "support_dead_with_xy",
    "support_in_base",
    "support_out_of_map",
    "support_missing_xy",
]
ADC_REASON_ORDER = [
    "adc_dead_with_xy",
    "adc_in_base",
    "adc_out_of_map",
    "adc_missing_xy",
]
REASON_COLORS = {
    "support_dead_with_xy": "#d62728",
    "support_in_base": "#1f77b4",
    "support_out_of_map": "#111111",
    "support_missing_xy": "#8c8c8c",
    "adc_dead_with_xy": "#d62728",
    "adc_in_base": "#9467bd",
    "adc_out_of_map": "#111111",
    "adc_missing_xy": "#8c8c8c",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot v5 frame calculation exclusions.")
    p.add_argument("--frame-state-path", default=str(DEFAULT_FRAME_STATE))
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--start-minute", type=float, default=5.0)
    p.add_argument("--max-minute", type=float, default=12.0)
    p.add_argument("--bins", type=int, default=240)
    p.add_argument("--chunk-size", type=int, default=750_000)
    return p.parse_args()


def zone_order(config: dict) -> list[str]:
    order = ["OUT_OF_MAP", "UNCLASSIFIED"] + list(config.get("colors", {}).keys())
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


def load_and_flag(args: argparse.Namespace, config: dict) -> pd.DataFrame:
    cols = [
        "match_id",
        "team_id",
        "frame_idx",
        "minute",
        "support_alive",
        "adc_alive",
        "support_x",
        "support_y",
        "adc_x",
        "adc_y",
    ]
    df = pd.read_parquet(args.frame_state_path, columns=cols)
    df = df[df["minute"].between(args.start_minute, args.max_minute, inclusive="both")].copy()
    print(f"[Loaded] frames in label window: {len(df):,}")

    df["support_has_xy"] = df[["support_x", "support_y"]].notna().all(axis=1)
    df["adc_has_xy"] = df[["adc_x", "adc_y"]].notna().all(axis=1)
    df["support_alive_bool"] = df["support_alive"].fillna(False).astype(bool)
    df["adc_alive_bool"] = df["adc_alive"].fillna(False).astype(bool)

    df["support_zone_v5_abs"] = classify_xy(
        df["support_x"].to_numpy(dtype=np.float64, copy=False),
        df["support_y"].to_numpy(dtype=np.float64, copy=False),
        config,
        args.chunk_size,
        "support",
    )
    df["adc_zone_v5_abs"] = classify_xy(
        df["adc_x"].to_numpy(dtype=np.float64, copy=False),
        df["adc_y"].to_numpy(dtype=np.float64, copy=False),
        config,
        args.chunk_size,
        "adc",
    )

    df["support_in_base_v5"] = df["support_zone_v5_abs"].isin(BASE_ZONES)
    df["adc_in_base_v5"] = df["adc_zone_v5_abs"].isin(BASE_ZONES)
    df["support_out_of_map_v5"] = df["support_zone_v5_abs"] == "OUT_OF_MAP"
    df["adc_out_of_map_v5"] = df["adc_zone_v5_abs"] == "OUT_OF_MAP"

    df["support_spatial_valid_v5"] = (
        df["support_alive_bool"]
        & df["support_has_xy"]
        & ~df["support_in_base_v5"]
        & ~df["support_out_of_map_v5"]
    )
    df["adc_coop_valid_v5"] = (
        df["adc_alive_bool"]
        & df["adc_has_xy"]
        & ~df["adc_in_base_v5"]
        & ~df["adc_out_of_map_v5"]
    )
    df["distance_component_valid_v5"] = df["support_spatial_valid_v5"] & df["adc_coop_valid_v5"]

    df["support_exclusion_reason"] = "support_valid"
    df.loc[~df["support_has_xy"], "support_exclusion_reason"] = "support_missing_xy"
    df.loc[df["support_has_xy"] & ~df["support_alive_bool"], "support_exclusion_reason"] = "support_dead_with_xy"
    df.loc[df["support_alive_bool"] & df["support_has_xy"] & df["support_out_of_map_v5"], "support_exclusion_reason"] = "support_out_of_map"
    df.loc[
        df["support_alive_bool"] & df["support_has_xy"] & ~df["support_out_of_map_v5"] & df["support_in_base_v5"],
        "support_exclusion_reason",
    ] = "support_in_base"

    df["adc_exclusion_reason"] = "adc_valid"
    df.loc[~df["adc_has_xy"], "adc_exclusion_reason"] = "adc_missing_xy"
    df.loc[df["adc_has_xy"] & ~df["adc_alive_bool"], "adc_exclusion_reason"] = "adc_dead_with_xy"
    df.loc[df["adc_alive_bool"] & df["adc_has_xy"] & df["adc_out_of_map_v5"], "adc_exclusion_reason"] = "adc_out_of_map"
    df.loc[
        df["adc_alive_bool"] & df["adc_has_xy"] & ~df["adc_out_of_map_v5"] & df["adc_in_base_v5"],
        "adc_exclusion_reason",
    ] = "adc_in_base"
    return df


def add_map(ax: plt.Axes) -> None:
    map_path = REPO_ROOT / "images" / "minimapa.png"
    if map_path.exists():
        image = mpimg.imread(map_path)
        ax.imshow(image, extent=(0, MAP_MAX, 0, MAP_MAX), origin="upper", alpha=0.78, zorder=0)
    ax.set_xlim(0, MAP_MAX)
    ax.set_ylim(0, MAP_MAX)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_reason_heatmap(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, title: str, bins: int) -> None:
    add_map(ax)
    plotted_any = False
    for reason in list(REASON_COLORS):
        part = df[df["reason"] == reason]
        part = part[part[[x_col, y_col]].notna().all(axis=1)]
        if part.empty:
            continue
        hist, _, _ = np.histogram2d(
            part[x_col].astype(float),
            part[y_col].astype(float),
            bins=bins,
            range=[[0, MAP_MAX], [0, MAP_MAX]],
        )
        if not np.any(hist > 0):
            continue
        masked = np.ma.masked_where(hist.T <= 0, hist.T)
        vmax = float(np.nanpercentile(hist[hist > 0], 99.5))
        ax.imshow(
            masked,
            extent=(0, MAP_MAX, 0, MAP_MAX),
            origin="lower",
            cmap="Reds" if "dead" in reason else "Blues" if "base" in reason else "Greys",
            alpha=0.62,
            norm=LogNorm(vmin=1, vmax=max(vmax, 2.0)),
            zorder=2,
        )
        plotted_any = True
    ax.set_title(title, fontsize=13.0)
    if not plotted_any:
        ax.text(0.5, 0.5, "No plottable excluded positions", ha="center", va="center", transform=ax.transAxes)


def plot_exclusions(df: pd.DataFrame, args: argparse.Namespace, outdir: Path) -> Path:
    support_excluded = df[~df["support_spatial_valid_v5"]].copy()
    support_excluded["reason"] = support_excluded["support_exclusion_reason"]

    adc_excluded = df[df["support_spatial_valid_v5"] & ~df["adc_coop_valid_v5"]].copy()
    adc_excluded["reason"] = adc_excluded["adc_exclusion_reason"]

    fig = plt.figure(figsize=(15.2, 8.4), facecolor="white")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.82], height_ratios=[1.0, 1.0])

    ax_support = fig.add_subplot(gs[:, 0])
    plot_reason_heatmap(
        ax_support,
        support_excluded,
        "support_x",
        "support_y",
        "Support: posiciones excluidas del outside_ratio",
        args.bins,
    )

    ax_adc = fig.add_subplot(gs[:, 1])
    plot_reason_heatmap(
        ax_adc,
        adc_excluded,
        "adc_x",
        "adc_y",
        "ADC: posiciones que excluyen dist_to_adc",
        args.bins,
    )

    ax_support_bar = fig.add_subplot(gs[0, 2])
    support_counts = support_excluded["support_exclusion_reason"].value_counts().reindex(SUPPORT_REASON_ORDER, fill_value=0)
    ax_support_bar.barh(support_counts.index, support_counts.values, color=[REASON_COLORS[r] for r in support_counts.index])
    ax_support_bar.invert_yaxis()
    ax_support_bar.set_title("Exclusion espacial support", fontsize=12.0)
    ax_support_bar.set_xlabel("frames")
    ax_support_bar.grid(axis="x", alpha=0.22)
    ax_support_bar.tick_params(axis="y", labelsize=8.5)

    ax_adc_bar = fig.add_subplot(gs[1, 2])
    adc_counts = adc_excluded["adc_exclusion_reason"].value_counts().reindex(ADC_REASON_ORDER, fill_value=0)
    ax_adc_bar.barh(adc_counts.index, adc_counts.values, color=[REASON_COLORS[r] for r in adc_counts.index])
    ax_adc_bar.invert_yaxis()
    ax_adc_bar.set_title("Exclusion dist_to_adc", fontsize=12.0)
    ax_adc_bar.set_xlabel("frames")
    ax_adc_bar.grid(axis="x", alpha=0.22)
    ax_adc_bar.tick_params(axis="y", labelsize=8.5)

    support_valid = int(df["support_spatial_valid_v5"].sum())
    distance_valid = int(df["distance_component_valid_v5"].sum())
    fig.suptitle(
        (
            f"Frames que no entran al calculo por filtros v5 "
            f"(min {args.start_minute:g}-{args.max_minute:g}) | "
            f"support validos: {support_valid:,}/{len(df):,} | "
            f"dist_to_adc validos: {distance_valid:,}/{support_valid:,}"
        ),
        fontsize=15.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    outpath = outdir / f"frame_calculation_exclusions_v5_m{int(args.start_minute)}_{int(args.max_minute)}.png"
    fig.savefig(outpath, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return outpath


def save_tables(df: pd.DataFrame, args: argparse.Namespace, outdir: Path, image_path: Path) -> None:
    support_excluded = df[~df["support_spatial_valid_v5"]].copy()
    adc_excluded = df[df["support_spatial_valid_v5"] & ~df["adc_coop_valid_v5"]].copy()
    summary = {
        "frame_state_path": str(Path(args.frame_state_path).resolve()),
        "config_path": str(Path(args.config).resolve()),
        "start_minute": args.start_minute,
        "max_minute": args.max_minute,
        "window_frames": int(len(df)),
        "support_spatial_valid_frames": int(df["support_spatial_valid_v5"].sum()),
        "support_spatial_excluded_frames": int(len(support_excluded)),
        "distance_component_valid_frames": int(df["distance_component_valid_v5"].sum()),
        "distance_component_excluded_frames_given_valid_support": int(len(adc_excluded)),
        "support_exclusion_counts": support_excluded["support_exclusion_reason"].value_counts().to_dict(),
        "adc_exclusion_counts_given_valid_support": adc_excluded["adc_exclusion_reason"].value_counts().to_dict(),
        "image_path": str(image_path.resolve()),
    }
    (outdir / "frame_calculation_exclusions_v5_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    support_excluded[
        ["match_id", "team_id", "frame_idx", "minute", "support_x", "support_y", "support_zone_v5_abs", "support_exclusion_reason"]
    ].head(5000).to_csv(outdir / "support_spatial_excluded_examples_v5.csv", index=False)
    adc_excluded[
        ["match_id", "team_id", "frame_idx", "minute", "adc_x", "adc_y", "adc_zone_v5_abs", "adc_exclusion_reason"]
    ].head(5000).to_csv(outdir / "adc_distance_excluded_examples_v5.csv", index=False)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    df = load_and_flag(args, config)
    image_path = plot_exclusions(df, args, outdir)
    save_tables(df, args, outdir, image_path)
    print(f"[Saved] {image_path.resolve()}")
    print(f"[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
