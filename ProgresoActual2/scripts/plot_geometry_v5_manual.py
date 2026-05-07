#!/usr/bin/env python3
"""
Render manual geometry v5 over the observed density heatmap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_SRC_DIR = REPO_ROOT / "ProgresoActual2" / "src" / "geometry"
sys.path.insert(0, str(GEOMETRY_SRC_DIR))

from geometry_v5_manual import MAP_MAX, ZONE_ORDER_V5, ZONE_TO_ID_V5, classify_zone_v5  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "ProgresoActual2" / "data" / "geometry" / "manual_geometry_v5_config.json"
DEFAULT_DENSITY = REPO_ROOT / "ProgresoActual2" / "data" / "geometry" / "observed_player_density_0_14.npz"
DEFAULT_OUTDIR = REPO_ROOT / "ProgresoActual2" / "analysis" / "geometry_v5_manual"
DEFAULT_ANNOTATION = REPO_ROOT / "ProgresoActual2" / "mapa_editado.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot manual geometry v5 diagnostics.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--density-path", default=str(DEFAULT_DENSITY))
    p.add_argument("--annotation-path", default=str(DEFAULT_ANNOTATION))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--tag", default="m0_14")
    p.add_argument("--grid-size", type=int, default=360)
    return p.parse_args()


def load_density(path: str | Path) -> np.ndarray:
    data = np.load(path)
    if "smooth_density" in data.files:
        return data["smooth_density"].astype(float)
    return data["heatmap"].astype(float)


def plot_heatmap(ax: plt.Axes, density: np.ndarray, alpha: float = 0.62) -> None:
    log_density = np.log1p(density)
    vmax = float(np.quantile(log_density[log_density > 0], 0.995)) if np.any(log_density > 0) else 1.0
    ax.imshow(
        np.clip(log_density.T / max(vmax, 1e-9), 0.0, 1.0),
        extent=(0, MAP_MAX, 0, MAP_MAX),
        origin="lower",
        cmap="Greys",
        interpolation="nearest",
        alpha=alpha,
    )


def common_axes(ax: plt.Axes, title: str) -> None:
    ax.set_xlim(0, MAP_MAX)
    ax.set_ylim(0, MAP_MAX)
    ax.set_aspect("equal")
    ax.set_xlabel("x map coordinate")
    ax.set_ylabel("y map coordinate")
    ax.set_xticks(np.arange(0, MAP_MAX + 1, 1000))
    ax.set_yticks(np.arange(0, MAP_MAX + 1, 1000))
    ax.set_xticks(np.arange(0, MAP_MAX + 1, 500), minor=True)
    ax.set_yticks(np.arange(0, MAP_MAX + 1, 500), minor=True)
    ax.grid(which="major", color="#111111", alpha=0.18, linewidth=0.55)
    ax.grid(which="minor", color="#111111", alpha=0.07, linewidth=0.25)
    ax.tick_params(labelsize=7)
    ax.set_title(title)


def _points(raw_points: List[List[float]]) -> List[Tuple[float, float]]:
    return [(float(x), float(y)) for x, y in raw_points]


def draw_geometry(ax: plt.Axes, config: dict, linewidth: float = 3.4) -> None:
    colors: Dict[str, str] = config["colors"]
    polygons = config.get("polygons", {})
    circles = config.get("circles", {})

    for zone in config["priority"]:
        color = colors.get(zone, "#333333")
        if zone in polygons:
            pts = _points(polygons[zone])
            patch = patches.Polygon(
                pts,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                joinstyle="round",
                capstyle="round",
                zorder=5,
            )
            ax.add_patch(patch)
        if zone in circles:
            cx, cy = circles[zone]["center"]
            radius = float(circles[zone]["radius"])
            ax.add_patch(
                patches.Circle(
                    (float(cx), float(cy)),
                    radius=radius,
                    fill=False,
                    edgecolor=color,
                    linewidth=linewidth,
                    zorder=6,
                )
            )


def save_overlay(density: np.ndarray, config: dict, outdir: Path, tag: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 9.6))
    plot_heatmap(ax, density)
    draw_geometry(ax, config)
    common_axes(ax, f"Manual geometry v5 outlines on observed heatmap ({tag})")
    handles = [
        patches.Patch(facecolor=config["colors"][z], label=z)
        for z in config["priority"]
        if z in config["colors"]
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    outpath = outdir / f"geometry_v5_manual_outlines_on_heatmap_{tag}.png"
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return outpath


def save_zone_layer(config_path: str, config: dict, outdir: Path, tag: str, grid_size: int) -> Path:
    colors = ["#f6f6f6", "#dddddd"] + [config["colors"].get(z, "#888888") for z in ZONE_ORDER_V5[2:]]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1.0), cmap.N)
    xs = np.linspace(0, MAP_MAX, grid_size)
    ys = np.linspace(0, MAP_MAX, grid_size)
    raster = np.zeros((grid_size, grid_size), dtype=np.int16)
    for y_idx, y in enumerate(ys):
        for x_idx, x in enumerate(xs):
            zone = classify_zone_v5(float(x), float(y), config_path=config_path)
            raster[y_idx, x_idx] = ZONE_TO_ID_V5.get(zone, ZONE_TO_ID_V5["UNCLASSIFIED"])

    fig, ax = plt.subplots(figsize=(10, 9.6))
    ax.imshow(
        raster,
        extent=(0, MAP_MAX, 0, MAP_MAX),
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        alpha=0.82,
    )
    draw_geometry(ax, config, linewidth=2.6)
    common_axes(ax, f"Manual geometry v5 classified layer ({tag})")
    handles = [
        patches.Patch(facecolor=config["colors"].get(z, "#dddddd"), label=z)
        for z in ZONE_ORDER_V5
        if z not in {"OUT_OF_MAP", "UNCLASSIFIED"}
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
    fig.tight_layout()
    outpath = outdir / f"geometry_v5_manual_zone_layer_{tag}.png"
    fig.savefig(outpath, dpi=180)
    plt.close(fig)
    return outpath


def save_side_by_side(annotation_path: str, overlay_path: Path, outdir: Path, tag: str) -> Path | None:
    path = Path(annotation_path)
    if not path.exists():
        return None
    left = Image.open(path).convert("RGB")
    right = Image.open(overlay_path).convert("RGB")
    target_h = min(left.height, right.height)
    left_w = int(left.width * target_h / left.height)
    right_w = int(right.width * target_h / right.height)
    left = left.resize((left_w, target_h), Image.Resampling.LANCZOS)
    right = right.resize((right_w, target_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (left_w + right_w, target_h), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left_w, 0))
    outpath = outdir / f"geometry_v5_manual_annotation_comparison_{tag}.png"
    canvas.save(outpath)
    return outpath


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    density = load_density(args.density_path)
    overlay_path = save_overlay(density, config, outdir, args.tag)
    zone_layer_path = save_zone_layer(args.config, config, outdir, args.tag, args.grid_size)
    comparison_path = save_side_by_side(args.annotation_path, overlay_path, outdir, args.tag)

    print(f"Saved overlay: {overlay_path}")
    print(f"Saved zone layer: {zone_layer_path}")
    if comparison_path:
        print(f"Saved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
