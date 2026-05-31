#!/usr/bin/env python3
"""
Build the five figures requested for Informe de Progreso II.

The outputs are intentionally generated in a separate folder from the older
report_style figures so the final document can choose the new visual set
without losing the previous artifacts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "final" / "docs" / "figures" / "informe_ii_final"
HIGHRES_OUT_DIR = REPO_ROOT / "final" / "docs" / "figures" / "informe_ii_final_highres"
MAP_MAX = 14800.0

BLUE = "#2f78bf"
BLUE_DARK = "#1f4e79"
BLUE_LIGHT = "#8fb9df"
SUPPORT_POINT = "#006dff"
BOTLANER_POINT = "#f28e2b"
GRAY = "#8c8c8c"
GRAY_DARK = "#333333"
GRID = "#d9d9d9"
GREEN = "#4f8f5b"
ORANGE = "#c07a2c"
RED = "#b45a56"
PURPLE = "#7a6f9b"

ZONE_COLORS = {
    "BLUE_BASE": "#4c78a8",
    "RED_BASE": "#9c755f",
    "TOP_LANE_CORE": "#d7b85b",
    "BOT_LANE_CORE": "#d7b85b",
    "TOP_SIDE_NEAR": "#c49a6c",
    "BOT_SIDE_NEAR": "#c49a6c",
    "RIVER_TOP": "#76a6b2",
    "RIVER_BOT": "#76a6b2",
    "BLUE_TOP_JUNGLE": "#7da0c4",
    "BLUE_BOT_JUNGLE": "#5b8db8",
    "RED_TOP_JUNGLE": "#b78396",
    "RED_BOT_JUNGLE": "#b66b6b",
    "MID_LANE": "#8f83b8",
    "BARON_GRUBS_HERALD_AREA": "#6c9a73",
    "DRAGON_AREA": "#6c9a73",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.5,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 1.0,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.55,
        }
    )


def finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    highres_path = HIGHRES_OUT_DIR / path.name
    highres_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(highres_path, dpi=850)
    plt.close(fig)
    print(f"saved {path.relative_to(REPO_ROOT)}")
    print(f"saved {highres_path.relative_to(REPO_ROOT)}")


def load_map(ax: plt.Axes, alpha: float = 0.9) -> None:
    image = mpimg.imread(REPO_ROOT / "images" / "minimapa.png")
    ax.imshow(image, extent=(0, MAP_MAX, 0, MAP_MAX), origin="upper", alpha=alpha)
    ax.set_xlim(0, MAP_MAX)
    ax.set_ylim(0, MAP_MAX)
    ax.set_aspect("equal")
    ax.grid(False)


def style_map_axes(ax: plt.Axes, show_ticks: bool = False) -> None:
    if show_ticks:
        ax.set_xlabel("Coordenada x")
        ax.set_ylabel("Coordenada y")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")


def figure_manual_geometry() -> None:
    config_path = REPO_ROOT / "ProgresoActual2" / "data" / "geometry" / "manual_geometry_v5_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(5.6, 5.75))
    load_map(ax, alpha=0.34)

    for zone in config["priority"]:
        color = ZONE_COLORS.get(zone, "#777777")
        if zone in config.get("polygons", {}):
            pts = [(float(x), float(y)) for x, y in config["polygons"][zone]]
            ax.add_patch(
                patches.Polygon(
                    pts,
                    closed=True,
                    facecolor=color,
                    edgecolor=GRAY_DARK,
                    linewidth=1.05,
                    alpha=0.62,
                    zorder=3,
                )
            )
        if zone in config.get("circles", {}):
            circle = config["circles"][zone]
            ax.add_patch(
                patches.Circle(
                    tuple(circle["center"]),
                    float(circle["radius"]),
                    facecolor=color,
                    edgecolor=GRAY_DARK,
                    linewidth=1.05,
                    alpha=0.68,
                    zorder=4,
                )
            )

    for zone, zone_cfg in config.get("centerline_zones", {}).items():
        line = np.asarray(zone_cfg["centerline"], dtype=float)
        ax.plot(
            line[:, 0],
            line[:, 1],
            color=ZONE_COLORS.get(zone, ORANGE),
            linewidth=1.8,
            alpha=0.95,
            zorder=5,
        )

    ax.set_title("Geometria manual v5")
    style_map_axes(ax, show_ticks=False)

    legend_items = [
        ("Bot / top lane", "BOT_LANE_CORE"),
        ("Contexto bot", "BOT_SIDE_NEAR"),
        ("Rio", "RIVER_BOT"),
        ("Mid lane", "MID_LANE"),
        ("Jungla", "BLUE_BOT_JUNGLE"),
        ("Objetivos", "DRAGON_AREA"),
    ]
    handles = [
        patches.Patch(facecolor=ZONE_COLORS[z], edgecolor=GRAY_DARK, alpha=0.45, label=label)
        for label, z in legend_items
    ]
    legend = ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3, frameon=True)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(GRID)
    legend.get_frame().set_alpha(0.86)
    finish(fig, OUT_DIR / "fig01_geometria_manual.png")


def load_scores() -> pd.Series:
    score_path = REPO_ROOT / "final" / "data" / "scores" / "support_scores_v5_geometry_m12.parquet"
    scores = pd.read_parquet(score_path, columns=["support_roam_score_v5_geometry"])
    return scores["support_roam_score_v5_geometry"].dropna().clip(0, 1)


def figure_label_distribution() -> None:
    scores = load_scores()
    mean = scores.mean()
    median = scores.median()

    fig, ax = plt.subplots(figsize=(5.25, 5.25))
    ax.hist(scores, bins=38, range=(0, 1), color=BLUE, edgecolor="white", linewidth=0.55)
    ax.axvline(mean, color=ORANGE, linewidth=1.8)
    ax.axvline(median, color=BLUE_DARK, linewidth=1.6, linestyle="--")
    ax.set_title("Distribucion final de la etiqueta")
    ax.set_xlabel("Support roam score v5")
    ax.set_ylabel("Observaciones")
    ax.set_xlim(0, 1)
    ax.grid(axis="y")
    ax.text(
        0.98,
        0.94,
        f"n = {len(scores):,}\nMedia = {mean:.3f}\nMediana = {median:.3f}".replace(",", "."),
        ha="right",
        va="top",
        transform=ax.transAxes,
        fontsize=11.2,
        bbox={"facecolor": "white", "edgecolor": GRID, "alpha": 0.9, "pad": 4},
    )
    finish(fig, OUT_DIR / "fig02_distribucion_etiqueta.png")


def case_row(case_index: pd.DataFrame, group: str, rank: int = 1) -> pd.Series:
    row = case_index[(case_index["case_group"] == group) & (case_index["case_rank"] == rank)]
    if row.empty:
        raise ValueError(f"Missing case {group} rank {rank}")
    return row.iloc[0]


def plot_case(ax: plt.Axes, frames: pd.DataFrame, row: pd.Series, title: str) -> None:
    case = frames[frames["case_id"] == row["case_id"]].copy()
    if case.empty:
        raise ValueError(f"No frame timeline for {row['case_id']}")

    load_map(ax, alpha=0.87)
    ax.scatter(
        case["adc_x"],
        case["adc_y"],
        color=BOTLANER_POINT,
        edgecolors="white",
        linewidths=0.95,
        s=74,
        marker="s",
        label="Botlaner",
        zorder=5,
    )
    ax.scatter(
        case["support_x"],
        case["support_y"],
        color=SUPPORT_POINT,
        edgecolors="white",
        linewidths=0.95,
        s=42,
        marker="o",
        label="Support",
        zorder=6,
    )

    ax.set_title(title, pad=8)
    ax.text(
        0.03,
        0.97,
        (
            f"{row['ally_utility_champion_name']} + {row['ally_bottom_champion_name']}\n"
            f"Pred {row['prediction']:.3f} | Real {row['actual']:.3f} | Error {row['abs_error']:.3f}"
        ),
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=11.0,
        bbox={"facecolor": "white", "edgecolor": GRID, "alpha": 0.9, "pad": 4},
        zorder=8,
    )
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor=GRID, framealpha=0.86)
    style_map_axes(ax)


def figure_error_case(row: pd.Series, frames: pd.DataFrame, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(5.45, 5.45))
    plot_case(ax, frames, row, title)
    finish(fig, OUT_DIR / filename)


def figure_error_cases() -> None:
    frames = pd.read_csv(REPO_ROOT / "final" / "analysis" / "qualitative_case_audit" / "case_frame_timeline.csv")
    index = pd.read_csv(REPO_ROOT / "final" / "analysis" / "qualitative_case_audit" / "case_index.csv")
    high = case_row(index, "top_error", 1)
    low = case_row(index, "bottom_error", 1)

    figure_error_case(high, frames, "Caso de error alto", "fig03a_caso_error_alto.png")
    figure_error_case(low, frames, "Caso de error bajo", "fig03b_caso_error_bajo.png")


def aggregate_support_heatmap() -> tuple[np.ndarray, int, int]:
    path = REPO_ROOT / "final" / "data" / "frame_state" / "support_frame_state.parquet"
    columns = ["match_id", "minute", "support_alive", "support_in_base", "support_x", "support_y"]
    part_paths = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    bins = 220
    hist = np.zeros((bins, bins), dtype=np.float64)
    filtered_frames = 0
    matches: set[str] = set()

    for part_path in part_paths:
        parquet = pq.ParquetFile(part_path)
        for batch in parquet.iter_batches(columns=columns, batch_size=500_000):
            df = batch.to_pandas()
            mask = (
                df["minute"].between(2, 14, inclusive="both")
                & df["support_alive"].fillna(False).astype(bool)
                & ~df["support_in_base"].fillna(False).astype(bool)
                & df["support_x"].between(0, MAP_MAX, inclusive="both")
                & df["support_y"].between(0, MAP_MAX, inclusive="both")
            )
            keep = df.loc[mask, ["match_id", "support_x", "support_y"]]
            if keep.empty:
                continue
            h, _, _ = np.histogram2d(
                keep["support_x"].astype(float),
                keep["support_y"].astype(float),
                bins=bins,
                range=[[0, MAP_MAX], [0, MAP_MAX]],
            )
            hist += h
            filtered_frames += len(keep)
            matches.update(keep["match_id"].dropna().astype(str).unique())

    return hist, filtered_frames, len(matches)


def figure_support_heatmap() -> None:
    hist, filtered_frames, n_matches = aggregate_support_heatmap()
    masked = np.ma.masked_where(hist.T <= 0, hist.T)
    vmax = float(np.nanpercentile(hist[hist > 0], 99.7)) if np.any(hist > 0) else 1.0

    fig, ax = plt.subplots(figsize=(5.55, 5.45))
    load_map(ax, alpha=0.76)
    im = ax.imshow(
        masked,
        extent=(0, MAP_MAX, 0, MAP_MAX),
        origin="lower",
        cmap="magma",
        alpha=0.70,
        norm=LogNorm(vmin=1, vmax=max(vmax, 2.0)),
        zorder=4,
    )
    ax.set_title("Mapa de calor del movimiento de support")
    style_map_axes(ax)
    ax.text(
        0.03,
        0.97,
        f"Min 2-14, vivo y fuera de base\nFrames: {filtered_frames:,} | Partidas: {n_matches:,}".replace(",", "."),
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=10.4,
        bbox={"facecolor": "white", "edgecolor": GRID, "alpha": 0.88, "pad": 4},
        zorder=7,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.02)
    cbar.set_label("Frames (log)")
    cbar.ax.tick_params(labelsize=10.0)
    finish(fig, OUT_DIR / "fig04_heatmap_movimiento_support.png")


def figure_expert_reference_scatter() -> None:
    reference = pd.read_csv(REPO_ROOT / "ProgresoActual" / "references" / "manual_support_champion_reference.csv")
    means = pd.read_csv(REPO_ROOT / "final" / "analysis" / "label_health" / "support_roam_score_v5_champion_means.csv")
    normalize_champion = lambda name: re.sub(r"[^A-Za-z0-9]", "", str(name)).lower()
    reference = reference.assign(champion_key=reference["champion_name"].map(normalize_champion))
    means = means.assign(champion_key=means["support_champion_name"].map(normalize_champion))
    merged = reference.merge(means, on="champion_key", how="inner")
    merged = merged.dropna(subset=["expert_support_roam_score", "expert_confidence", "mean", "games"]).copy()
    spearman = merged["expert_support_roam_score"].rank().corr(merged["mean"].rank())
    pearson = merged["expert_support_roam_score"].corr(merged["mean"])

    fig, ax = plt.subplots(figsize=(5.45, 5.45))
    sc = ax.scatter(
        merged["expert_support_roam_score"],
        merged["mean"],
        c=merged["expert_confidence"],
        s=58,
        cmap="viridis",
        vmin=0.65,
        vmax=1.0,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.8,
    )

    z = np.polyfit(merged["expert_support_roam_score"], merged["mean"], deg=1)
    xs = np.linspace(merged["expert_support_roam_score"].min() - 0.03, merged["expert_support_roam_score"].max() + 0.03, 120)
    ax.plot(xs, np.polyval(z, xs), color=ORANGE, linewidth=1.7)

    label_names = {"Bard", "Pyke", "Alistar", "Yuumi", "Sona", "Senna"}
    for _, r in merged[merged["champion_name"].isin(label_names)].iterrows():
        x_offset = -0.018 if r["expert_support_roam_score"] > 0.86 else 0.012
        ha = "right" if r["expert_support_roam_score"] > 0.86 else "left"
        ax.text(
            r["expert_support_roam_score"] + x_offset,
            r["mean"] + 0.002,
            r["champion_name"],
            ha=ha,
            fontsize=9.8,
            color=GRAY_DARK,
        )

    ax.set_title("Comparativa frente a referencia experta")
    ax.set_xlabel("Score experto esperado")
    ax.set_ylabel("Media empirica de la etiqueta")
    ax.grid(True)
    ax.set_xlim(0.02, 1.0)
    ax.set_ylim(max(0.10, merged["mean"].min() - 0.04), min(0.62, merged["mean"].max() + 0.05))
    ax.text(
        0.04,
        0.96,
        f"Spearman = {spearman:.3f}\nPearson = {pearson:.3f}\nn = {len(merged)} campeones",
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": GRID, "alpha": 0.9, "pad": 4},
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.044, pad=0.02)
    cbar.set_label("Confianza")
    cbar.ax.tick_params(labelsize=10.0)
    finish(fig, OUT_DIR / "fig05_scatter_referencia_experta.png")


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HIGHRES_OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_manual_geometry()
    figure_label_distribution()
    figure_error_cases()
    figure_support_heatmap()
    figure_expert_reference_scatter()


if __name__ == "__main__":
    main()
