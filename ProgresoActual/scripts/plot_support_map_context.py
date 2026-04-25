#!/usr/bin/env python3
"""
Create a schematic map-context figure for explaining support roam asymmetry.

The figure is intentionally schematic and does not use Riot map assets. It marks
bot lane, dragon and void grubs/baron-side objective context to support the
progress report narrative.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

DEFAULT_OUTDIR = os.path.join("ProgresoActual", "analysis", "map_context")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot schematic LoL map context for support roam asymmetry.")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--out-name", default="support_objective_context.png")
    return p.parse_args()


def add_label(ax, x: float, y: float, text: str, color: str = "black", size: int = 10) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=color,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": color, "alpha": 0.9},
    )


def main() -> None:
    args = parse_args()
    ensure_dir(args.outdir)
    out_path = os.path.join(args.outdir, args.out_name)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(0, 15000)
    ax.set_ylim(0, 15000)
    ax.set_aspect("equal")
    ax.set_facecolor("#eef2e6")

    # Map frame and bases.
    ax.add_patch(Rectangle((250, 250), 14500, 14500, fill=False, edgecolor="#415a3f", linewidth=2))
    ax.add_patch(Rectangle((350, 350), 1800, 1800, facecolor="#d8e8ff", edgecolor="#276fbf", alpha=0.75))
    ax.add_patch(Rectangle((12850, 12850), 1800, 1800, facecolor="#ffd9d6", edgecolor="#c44536", alpha=0.75))

    # Lanes and river.
    ax.plot([1200, 13800], [1700, 1700], color="#8a6f3d", linewidth=10, alpha=0.35)
    ax.plot([13800, 13800], [1700, 13200], color="#8a6f3d", linewidth=10, alpha=0.35)
    ax.plot([1500, 13500], [1500, 13500], color="#8a6f3d", linewidth=8, alpha=0.28)
    ax.plot([1700, 1700], [1200, 13800], color="#8a6f3d", linewidth=10, alpha=0.35)
    ax.plot([1700, 13200], [13800, 13800], color="#8a6f3d", linewidth=10, alpha=0.35)
    ax.plot([2400, 12600], [12600, 2400], color="#3f88c5", linewidth=16, alpha=0.26)

    # Objective areas.
    dragon = (10450, 4400)
    grubs = (4400, 10450)
    botlane = (11200, 2300)
    ax.scatter(*dragon, s=520, color="#f28e2b", edgecolor="black", linewidth=1.2, zorder=5)
    ax.scatter(*grubs, s=520, color="#8b5cf6", edgecolor="black", linewidth=1.2, zorder=5)
    ax.scatter(*botlane, s=420, color="#2a9d8f", edgecolor="black", linewidth=1.2, zorder=5)

    add_label(ax, 1200, 1200, "Blue base", "#276fbf")
    add_label(ax, 13800, 13800, "Red base", "#c44536")
    add_label(ax, botlane[0], botlane[1] - 750, "Botlane\nsupport + ADC", "#2a9d8f")
    add_label(ax, dragon[0] + 1700, dragon[1] - 250, "Dragon pit\nRED first: 60.77%", "#f28e2b")
    add_label(ax, grubs[0] - 1550, grubs[1] + 300, "Void grubs / Baron side\nBLUE first: 59.38%", "#6d3fc2")

    # Conceptual movement from bot lane to objectives.
    ax.add_patch(FancyArrowPatch(botlane, dragon, arrowstyle="->", mutation_scale=18, linewidth=2.3, color="#f28e2b"))
    ax.add_patch(FancyArrowPatch(botlane, grubs, arrowstyle="->", mutation_scale=18, linewidth=2.3, color="#6d3fc2"))
    ax.text(
        7600,
        7600,
        "Support roaming from bot to top-side objectives\nrequires a longer map movement than dragon-side rotations.",
        ha="center",
        va="center",
        fontsize=10,
        color="#263238",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#263238", "alpha": 0.92},
    )

    ax.set_title("Schematic map context for support roam asymmetry", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved map context figure: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
