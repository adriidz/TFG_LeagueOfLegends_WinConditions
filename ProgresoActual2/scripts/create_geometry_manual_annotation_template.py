#!/usr/bin/env python3
"""
Create manual annotation templates for the next geometry iteration.

The template uses observed all-player density as a drawing guide, but it is
not meant to define walkability automatically. The user can draw semantic
boundaries on top of the exported image and we can later convert those strokes
back to map coordinates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DENSITY = REPO_ROOT / "ProgresoActual2" / "data" / "geometry" / "observed_player_density_5_12.npz"
DEFAULT_OUTDIR = REPO_ROOT / "ProgresoActual2" / "analysis" / "geometry_manual_annotation"

MAP_MAX = 14800.0

REFERENCE_POINTS = {
    "Blue base": (1200, 1200),
    "Red base": (13600, 13600),
    "Dragon": (10450, 4400),
    "Baron/Grubs": (4400, 10450),
    "Mid": (7400, 7400),
}

DRAWING_LEGEND = [
    ("BOT_LANE_CORE", "#ffd23f"),
    ("TOP_LANE_CORE", "#c8e600"),
    ("BOT_SIDE_NEAR", "#ff8c42"),
    ("TOP_SIDE_NEAR", "#f4a261"),
    ("RIVER_BOT / RIVER_TOP", "#2ec4b6"),
    ("BLUE_BOT_JUNGLE", "#2f80ed"),
    ("BLUE_TOP_JUNGLE", "#56ccf2"),
    ("RED_BOT_JUNGLE", "#eb5757"),
    ("RED_TOP_JUNGLE", "#b83280"),
    ("MID_LANE", "#9b5de5"),
    ("OBJECTIVE/PIT", "#00a676"),
    ("BASE", "#111111"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a manual geometry annotation template.")
    p.add_argument("--density-path", default=str(DEFAULT_DENSITY))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--tag", default="m5_12")
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def load_density(path: str | Path) -> tuple[np.ndarray, float, int, int]:
    z = np.load(path)
    density = z["smooth_density"] if "smooth_density" in z.files else z["heatmap"]
    map_max = float(z["map_max"]) if "map_max" in z.files else MAP_MAX
    max_matches = int(z["max_matches"]) if "max_matches" in z.files else 0
    positions = int(z["positions"]) if "positions" in z.files else int(density.sum())
    return density.astype(float), map_max, max_matches, positions


def normalized_density_image(density: np.ndarray) -> np.ndarray:
    log_density = np.log1p(density)
    vmax = float(np.quantile(log_density[log_density > 0], 0.995)) if np.any(log_density > 0) else 1.0
    norm = np.clip(log_density / max(vmax, 1e-9), 0.0, 1.0)
    return norm


def draw_reference_points(ax: plt.Axes) -> None:
    for label, (x, y) in REFERENCE_POINTS.items():
        ax.scatter([x], [y], s=34, c="#111111", edgecolors="#ffffff", linewidths=0.8, zorder=6)
        ax.text(
            x + 130,
            y + 130,
            label,
            color="#111111",
            fontsize=7.5,
            weight="bold",
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.68},
            zorder=7,
        )


def add_coordinate_grid(ax: plt.Axes, map_max: float) -> None:
    major = np.arange(0, map_max + 1, 1000)
    minor = np.arange(0, map_max + 1, 500)
    ax.set_xticks(major)
    ax.set_yticks(major)
    ax.set_xticks(minor, minor=True)
    ax.set_yticks(minor, minor=True)
    ax.grid(which="major", color="#111111", linewidth=0.55, alpha=0.27)
    ax.grid(which="minor", color="#111111", linewidth=0.25, alpha=0.11)
    ax.tick_params(axis="both", labelsize=6.2, length=2)
    ax.set_xlim(0, map_max)
    ax.set_ylim(0, map_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x map coordinate", fontsize=8)
    ax.set_ylabel("y map coordinate", fontsize=8)


def draw_palette(ax: plt.Axes) -> None:
    ax.axis("off")
    ax.text(0.0, 0.99, "Suggested stroke colors", fontsize=9, weight="bold", va="top")
    ax.text(
        0.0,
        0.925,
        "Draw thick opaque outlines.\nClosed regions are easier\nto digitize later.",
        fontsize=7.5,
        va="top",
        linespacing=1.25,
    )
    y = 0.78
    for label, color in DRAWING_LEGEND:
        ax.add_patch(patches.Rectangle((0.0, y - 0.018), 0.11, 0.035, facecolor=color, edgecolor="#333333", lw=0.4))
        ax.text(0.14, y, label, fontsize=7.4, va="center")
        y -= 0.064
    ax.text(
        0.0,
        y - 0.02,
        "Important: avoid filling areas.\nUse outlines/boundaries.",
        fontsize=7.4,
        va="top",
        weight="bold",
    )


def save_human_template(density: np.ndarray, map_max: float, outdir: Path, tag: str, dpi: int, max_matches: int, positions: int) -> Path:
    norm = normalized_density_image(density)
    fig = plt.figure(figsize=(11.6, 10.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.23], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[0, 1])

    ax.imshow(
        norm.T,
        extent=(0, map_max, 0, map_max),
        origin="lower",
        cmap="Greys",
        interpolation="nearest",
        alpha=0.68,
    )
    add_coordinate_grid(ax, map_max)
    draw_reference_points(ax)
    ax.set_title(
        f"Manual geometry annotation template {tag} | {max_matches:,} matches | {positions:,} observed positions",
        fontsize=10,
        pad=8,
    )
    draw_palette(legend_ax)

    outpath = outdir / f"geometry_v5_annotation_template_{tag}.png"
    fig.savefig(outpath, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return outpath


def save_parse_canvas(density: np.ndarray, map_max: float, outdir: Path, tag: str, dpi: int) -> Path:
    norm = normalized_density_image(density)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        norm.T,
        extent=(0, map_max, 0, map_max),
        origin="lower",
        cmap="Greys",
        interpolation="nearest",
        alpha=0.64,
    )
    ax.set_xlim(0, map_max)
    ax.set_ylim(0, map_max)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    outpath = outdir / f"geometry_v5_annotation_canvas_{tag}.png"
    fig.savefig(outpath, dpi=dpi, facecolor="white", pad_inches=0)
    plt.close(fig)
    return outpath


def save_metadata(outdir: Path, tag: str, map_max: float, density_path: str, canvas_path: Path, template_path: Path) -> Path:
    metadata = {
        "tag": tag,
        "density_source": str(Path(density_path).resolve()),
        "template": str(template_path.resolve()),
        "parse_canvas": str(canvas_path.resolve()),
        "map_coordinate_system": {
            "origin": "bottom-left",
            "x_min": 0.0,
            "y_min": 0.0,
            "x_max": map_max,
            "y_max": map_max,
        },
        "parse_canvas_mapping": {
            "pixel_x_to_map_x": "x_map = pixel_x / (width_px - 1) * map_max",
            "pixel_y_to_map_y": "y_map = (height_px - 1 - pixel_y) / (height_px - 1) * map_max",
            "note": "The parse canvas has no axes or margins; draw on it if automatic color extraction is desired.",
        },
        "recommended_workflow": [
            "Use the template image if you want coordinate labels while drawing.",
            "Use the canvas image if you want me to convert colors to polygons more directly.",
            "Draw thick opaque outlines or closed boundaries, not translucent fills.",
        ],
        "suggested_colors": [{"zone": zone, "hex": color} for zone, color in DRAWING_LEGEND],
    }
    outpath = outdir / f"geometry_v5_annotation_metadata_{tag}.json"
    outpath.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return outpath


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    density, map_max, max_matches, positions = load_density(args.density_path)
    template_path = save_human_template(density, map_max, outdir, args.tag, args.dpi, max_matches, positions)
    canvas_path = save_parse_canvas(density, map_max, outdir, args.tag, args.dpi)
    metadata_path = save_metadata(outdir, args.tag, map_max, args.density_path, canvas_path, template_path)

    print(f"Saved template: {template_path}")
    print(f"Saved parse canvas: {canvas_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
