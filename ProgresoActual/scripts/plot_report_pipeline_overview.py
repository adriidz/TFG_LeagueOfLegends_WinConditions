#!/usr/bin/env python3
"""
Create a report-ready conceptual figure for the current support pipeline.

The figure highlights the key methodological separation:
- draft/pregame data are model inputs;
- timeline data are only used afterwards to build the observed label.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DEFAULT_OUTDIR = os.path.join("ProgresoActual", "analysis", "report_figures")
DEFAULT_FILENAME = "fig2_pipeline_draft_timeline_v2.png"


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot conceptual draft/timeline pipeline figure.")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--filename", default=DEFAULT_FILENAME)
    return p.parse_args()


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.titlesize": 19,
        "figure.titlesize": 20,
    })


def add_box(
    ax: plt.Axes,
    xy: Tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=1.7,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.66,
        title,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#1f2933",
    )
    ax.text(
        x + width / 2,
        y + height * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=12,
        color="#1f2933",
        linespacing=1.18,
    )


def add_arrow(
    ax: plt.Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: str = "#44546a",
    linestyle: str = "-",
    rad: float = 0.0,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.4,
        color=color,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def main() -> None:
    args = parse_args()
    configure_style()
    ensure_dir(args.outdir)
    out_path = os.path.join(args.outdir, args.filename)

    fig, ax = plt.subplots(figsize=(13.2, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    pregame_fill = "#e8f3ff"
    pregame_edge = "#2f6fab"
    observed_fill = "#fff1df"
    observed_edge = "#b66a00"
    model_fill = "#e9f7ef"
    model_edge = "#2e7d4f"
    eval_fill = "#f2eefb"
    eval_edge = "#6a4c93"

    ax.text(
        0.50,
        0.94,
        "Construccion del entrenamiento: input pre-partida vs etiqueta observada",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#1f2933",
    )

    add_box(ax, (0.06, 0.60), 0.20, 0.18, "Draft", "datos disponibles\nantes de jugar", pregame_fill, pregame_edge)
    add_box(ax, (0.34, 0.60), 0.20, 0.18, "Input del modelo", "campeones, roles,\nhechizos, lado", pregame_fill, pregame_edge)
    add_box(ax, (0.62, 0.60), 0.16, 0.18, "MLP", "predice un\nscore continuo", model_fill, model_edge)
    add_box(ax, (0.82, 0.60), 0.14, 0.18, "Prediccion", "score\nestimado", model_fill, model_edge)

    add_box(ax, (0.06, 0.17), 0.20, 0.18, "Timeline", "lo que ocurre\ndurante la partida", observed_fill, observed_edge)
    add_box(ax, (0.34, 0.17), 0.20, 0.18, "Frame-state", "posicion y estado\npor minuto", observed_fill, observed_edge)
    add_box(ax, (0.62, 0.17), 0.16, 0.18, "Etiqueta real", "roaming observado\nmin. 5-12", observed_fill, observed_edge)
    add_box(ax, (0.82, 0.31), 0.14, 0.18, "Comparacion", "loss MSE\nmetricas", eval_fill, eval_edge)

    add_arrow(ax, (0.26, 0.69), (0.34, 0.69), pregame_edge)
    add_arrow(ax, (0.54, 0.69), (0.62, 0.69), pregame_edge)
    add_arrow(ax, (0.78, 0.69), (0.82, 0.69), model_edge)

    add_arrow(ax, (0.26, 0.26), (0.34, 0.26), observed_edge)
    add_arrow(ax, (0.54, 0.26), (0.62, 0.26), observed_edge)
    add_arrow(ax, (0.78, 0.26), (0.82, 0.39), eval_edge, rad=0.0)
    add_arrow(ax, (0.89, 0.60), (0.89, 0.49), eval_edge, rad=0.0)

    ax.text(
        0.18,
        0.83,
        "Rama predictiva",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=pregame_edge,
    )
    ax.text(
        0.18,
        0.40,
        "Rama observada",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color=observed_edge,
    )
    ax.text(
        0.50,
        0.50,
        "La timeline no entra como input del modelo.\nSolo sirve para calcular el valor objetivo que se compara con la prediccion.",
        ha="center",
        va="center",
        fontsize=15,
        color="#4b5563",
        bbox={"facecolor": "#ffffff", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.50", "alpha": 0.98},
    )

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(os.path.abspath(out_path))


if __name__ == "__main__":
    main()
