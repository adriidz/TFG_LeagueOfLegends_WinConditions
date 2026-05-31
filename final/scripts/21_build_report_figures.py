#!/usr/bin/env python3
"""
Build report-style figures for Informe de Progreso II.

This script intentionally redraws the figures used in the report from the
analysis artifacts, using one restrained Matplotlib style inspired by the
figures in Informe de Progreso I: white background, blue primary color,
soft grid, direct titles, and minimal decorative color.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "final" / "docs" / "figures" / "report_style"

MAP_MAX = 14800.0

BLUE = "#2f78bf"
BLUE_DARK = "#1f4e79"
BLUE_LIGHT = "#8fb9df"
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
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.edgecolor": "#222222",
        "axes.linewidth": 1.1,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.55,
    })


def finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.relative_to(REPO_ROOT)}")


def clean_label(text: str) -> str:
    return (
        str(text)
        .replace("_", " ")
        .replace("champion id", "champion")
        .replace("utility", "support")
        .replace("bottom", "ADC")
        .replace("middle", "mid")
        .title()
    )


def load_training_target() -> pd.Series:
    frames = []
    for split in ["train", "val", "test"]:
        path = REPO_ROOT / "final" / "data" / "training" / f"{split}.parquet"
        frames.append(pd.read_parquet(path, columns=["support_roam_score"]))
    return pd.concat(frames, ignore_index=True)["support_roam_score"].dropna()


def fig_geometry() -> None:
    config_path = REPO_ROOT / "ProgresoActual2" / "data" / "geometry" / "manual_geometry_v5_config.json"
    map_path = REPO_ROOT / "images" / "minimapa.png"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(8.2, 7.8))
    img = mpimg.imread(map_path)
    ax.imshow(img, extent=(0, MAP_MAX, 0, MAP_MAX), origin="upper", alpha=0.92)

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
                    linewidth=1.0,
                    alpha=0.28,
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
                    linewidth=1.0,
                    alpha=0.32,
                    zorder=4,
                )
            )

    ax.set_title("Geometria manual v5 del mapa")
    ax.set_xlabel("Coordenada x")
    ax.set_ylabel("Coordenada y")
    ax.set_xlim(0, MAP_MAX)
    ax.set_ylim(0, MAP_MAX)
    ax.set_aspect("equal")
    ax.grid(False)

    legend_items = [
        ("Bot / top lane", "BOT_LANE_CORE"),
        ("Bot context", "BOT_SIDE_NEAR"),
        ("Rio", "RIVER_BOT"),
        ("Mid lane", "MID_LANE"),
        ("Jungla", "BLUE_BOT_JUNGLE"),
        ("Objetivos", "DRAGON_AREA"),
    ]
    handles = [
        patches.Patch(facecolor=ZONE_COLORS[z], edgecolor=GRAY_DARK, alpha=0.40, label=label)
        for label, z in legend_items
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    finish(fig, OUT_DIR / "fig01_geometry_v5_manual.png")


def fig_label_distribution() -> None:
    scores = load_training_target()
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.hist(scores, bins=42, range=(0, 1), color=BLUE, edgecolor="white", linewidth=0.7)
    ax.set_title("Distribucion del support roam score v5")
    ax.set_xlabel("Support roam score")
    ax.set_ylabel("Observaciones match-team")
    ax.grid(axis="y")
    ax.set_xlim(0, 1)
    ax.axvline(scores.mean(), color=ORANGE, linewidth=2.0, label=f"Media = {scores.mean():.3f}")
    ax.legend(frameon=False)
    finish(fig, OUT_DIR / "fig02_label_distribution.png")


def fig_label_sweep() -> None:
    path = REPO_ROOT / "final" / "analysis" / "label_variant_sweep" / "sweep_metrics.csv"
    df = pd.read_csv(path)
    df = df[(df["target"] == "raw") & (df["feature_set"] == "all")].copy()
    df = df.sort_values("spearman_corr", ascending=True)
    labels = [fill(x.replace("_", " "), width=26) for x in df["variant_id"]]

    fig_h = max(5.8, 0.36 * len(df))
    fig, ax = plt.subplots(figsize=(9.2, fig_h))
    colors = [BLUE if v != "v5_geometry" else ORANGE for v in df["variant_id"]]
    ax.barh(labels, df["spearman_corr"], color=colors, edgecolor="white", linewidth=0.6)
    ax.set_title("Comparacion de variantes de etiqueta")
    ax.set_xlabel("Spearman en validation")
    ax.set_ylabel("")
    ax.grid(axis="x")
    ax.set_xlim(max(0.0, df["spearman_corr"].min() - 0.01), df["spearman_corr"].max() + 0.006)
    for y, val in enumerate(df["spearman_corr"]):
        ax.text(val + 0.001, y, f"{val:.3f}", va="center", fontsize=10)
    finish(fig, OUT_DIR / "fig03_label_variant_sweep.png")


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.axis("off")

    def box(x: float, y: float, text: str, color: str) -> None:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                1.85,
                0.62,
                boxstyle="round,pad=0.04,rounding_size=0.02",
                facecolor=color,
                edgecolor=GRAY_DARK,
                linewidth=1.0,
                alpha=0.22,
            )
        )
        ax.text(x + 0.925, y + 0.31, text, ha="center", va="center", fontsize=12)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.4, "color": GRAY_DARK})

    box(0.1, 2.4, "Draft\npregame", BLUE)
    box(2.5, 2.4, "Features\nde draft", BLUE)
    box(4.9, 2.4, "Modelos\npredictivos", BLUE)
    box(7.3, 2.4, "Score\npredicho", BLUE)

    box(0.1, 0.85, "Timeline\nobservada", GREEN)
    box(2.5, 0.85, "Frame\nstate", GREEN)
    box(4.9, 0.85, "Etiqueta\nv5", GREEN)
    box(7.3, 0.85, "Score\nreal", GREEN)

    for x in [1.95, 4.35, 6.75]:
        arrow(x, 2.71, x + 0.45, 2.71)
        arrow(x, 1.16, x + 0.45, 1.16)
    arrow(8.2, 2.35, 8.2, 1.55)
    ax.text(8.65, 1.90, "Comparacion\ntrain/val/test", va="center", fontsize=12)

    ax.text(0.1, 3.35, "Entrada disponible antes de la partida", fontsize=13, color=BLUE_DARK)
    ax.text(0.1, 1.80, "Comportamiento observado para construir la etiqueta", fontsize=13, color="#386641")
    ax.set_xlim(0, 10)
    ax.set_ylim(0.35, 3.85)
    finish(fig, OUT_DIR / "fig04_pipeline.png")


def fig_training_curves() -> None:
    df = pd.read_csv(REPO_ROOT / "final" / "models" / "mlp_onehot" / "history.csv")
    df = df[df["target"] == "raw"].copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.plot(df["epoch"], df["train_loss"], color=BLUE, linewidth=2.0, label="Train")
    ax.plot(df["epoch"], df["val_loss"], color=ORANGE, linewidth=2.0, label="Validation")
    best = df.loc[df["val_loss"].idxmin()]
    ax.scatter([best["epoch"]], [best["val_loss"]], color=ORANGE, s=55, zorder=5)
    ax.set_title("Curvas de entrenamiento de la MLP OneHot")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.grid(True)
    ax.legend(frameon=False)
    finish(fig, OUT_DIR / "fig05_mlp_training_curves.png")


def fig_model_comparison() -> None:
    df = pd.read_csv(REPO_ROOT / "final" / "analysis" / "model_comparison" / "comparison_table_raw.csv")
    keep = [
        "Global Mean",
        "Champion Mean",
        "MLP Embed",
        "MLP Per-Role + Interactions",
        "MLP OneHot",
        "HistGBT",
        "HistGBT + Archetypes",
        "HistGBT + Pair TE",
    ]
    df = df[(df["trained_target"] == "raw") & (df["model"].isin(keep))].copy()
    order = {name: i for i, name in enumerate(keep)}
    df["order"] = df["model"].map(order)
    df = df.sort_values("order")

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    colors = [GRAY if "Mean" in m else BLUE for m in df["model"]]
    colors[-1] = ORANGE
    ax.barh(df["model"], df["spearman_corr"].fillna(0), color=colors, edgecolor="white", linewidth=0.6)
    ax.set_title("Comparacion de modelos en test")
    ax.set_xlabel("Spearman")
    ax.set_ylabel("")
    ax.grid(axis="x")
    ax.set_xlim(0, 0.43)
    for y, val in enumerate(df["spearman_corr"].fillna(0)):
        ax.text(val + 0.006, y, f"{val:.3f}" if val > 0 else "-", va="center", fontsize=10)
    finish(fig, OUT_DIR / "fig06_model_comparison_spearman.png")


def fig_ceiling() -> None:
    df = pd.read_csv(REPO_ROOT / "final" / "analysis" / "ceiling" / "ceiling_summary.csv")
    keep = ["support_champion", "botlane_champions", "botlane_champions+side", "support_archetype"]
    labels = {
        "support_champion": "Support",
        "botlane_champions": "Botlane",
        "botlane_champions+side": "Botlane + side",
        "support_archetype": "Arquetipo support",
    }
    df = df[df["grouping"].isin(keep)].copy()
    df["label"] = df["grouping"].map(labels)
    df = df.set_index("grouping").loc[keep].reset_index()

    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.barh(y - 0.18, df["icc"], height=0.32, color=BLUE, label="ICC")
    ax.barh(y + 0.18, df["r2_group_mean"], height=0.32, color=GREEN, label="R2 media grupo")
    ax.axvline(0.161, color=ORANGE, linestyle="--", linewidth=1.8, label="HistGBT")
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_title("Techo empirico por agrupaciones de draft")
    ax.set_xlabel("Varianza explicada")
    ax.grid(axis="x")
    ax.legend(frameon=False)
    ax.set_xlim(0, 0.20)
    finish(fig, OUT_DIR / "fig07_empirical_ceiling.png")


def fig_feature_importance() -> None:
    df = pd.read_csv(REPO_ROOT / "final" / "analysis" / "feature_importance" / "permutation_importance_groups.csv")
    df = df.sort_values("total_importance", ascending=True)
    labels = [clean_label(x) for x in df["feature_type"]]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.barh(labels, df["total_importance"], color=BLUE, edgecolor="white", linewidth=0.6)
    ax.set_title("Importancia por grupos de variables")
    ax.set_xlabel("Importancia por permutacion")
    ax.set_ylabel("")
    ax.grid(axis="x")
    finish(fig, OUT_DIR / "fig08_feature_importance_groups.png")


def fig_shap() -> None:
    df = pd.read_csv(REPO_ROOT / "final" / "analysis" / "shap" / "shap_global_importance.csv").head(8)
    df = df.sort_values("mean_abs_shap", ascending=True)
    labels = [clean_label(x) for x in df["feature"]]

    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    ax.barh(labels, df["mean_abs_shap"], color=BLUE, edgecolor="white", linewidth=0.6)
    ax.set_title("Resumen SHAP de variables principales")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_ylabel("")
    ax.grid(axis="x")
    finish(fig, OUT_DIR / "fig09_shap_summary.png")


def fig_qualitative_case() -> None:
    case_id = "top_error_01_EUW1_7831489390_200"
    frames = pd.read_csv(REPO_ROOT / "final" / "analysis" / "qualitative_case_audit" / "case_frame_timeline.csv")
    case = frames[frames["case_id"] == case_id].copy()
    info = pd.read_csv(REPO_ROOT / "final" / "analysis" / "qualitative_case_audit" / "case_index.csv")
    row = info[info["case_id"] == case_id].iloc[0]

    fig, ax = plt.subplots(figsize=(7.6, 7.4))
    img = mpimg.imread(REPO_ROOT / "images" / "minimapa.png")
    ax.imshow(img, extent=(0, MAP_MAX, 0, MAP_MAX), origin="upper", alpha=0.88)
    ax.plot(case["adc_x"], case["adc_y"], color=GRAY_DARK, linewidth=1.8, marker="o", markersize=4, label=f"ADC ({row['ally_bottom_champion_name']})")
    ax.plot(case["support_x"], case["support_y"], color=BLUE, linewidth=2.2, marker="o", markersize=4.5, label=f"Support ({row['ally_utility_champion_name']})")

    for _, r in case.iterrows():
        if pd.notna(r["support_x"]) and pd.notna(r["support_y"]):
            ax.text(r["support_x"] + 130, r["support_y"] + 130, f"{int(round(r['minute']))}", fontsize=8, color=BLUE_DARK)

    ax.set_title("Caso auditado: error alto")
    subtitle = f"Pred={row['prediction']:.3f}  Real={row['actual']:.3f}  Abs error={row['abs_error']:.3f}"
    ax.text(
        0.03,
        0.96,
        subtitle,
        ha="left",
        va="top",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": GRID, "alpha": 0.88, "pad": 4},
    )
    ax.set_xlim(0, MAP_MAX)
    ax.set_ylim(0, MAP_MAX)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
    ax.grid(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    finish(fig, OUT_DIR / "fig10_qualitative_case_map.png")


def fig_embedding_distance() -> None:
    df = pd.read_csv(REPO_ROOT / "final" / "analysis" / "embedding_analysis" / "embedding_distance_vs_roam_pairs.csv")
    sample = df.sample(n=min(len(df), 30000), random_state=42)
    x = sample["cosine_distance"].to_numpy()
    y = sample["roam_score_abs_diff"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.scatter(x, y, s=5, alpha=0.12, color=BLUE, linewidths=0)
    if len(sample) > 2:
        z = np.polyfit(x, y, deg=1)
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        ax.plot(xs, np.polyval(z, xs), color=ORANGE, linewidth=2.0, label="Tendencia lineal")
    ax.set_title("Distancia entre embeddings y diferencia de roaming")
    ax.set_xlabel("Distancia coseno entre campeones")
    ax.set_ylabel("Diferencia absoluta de score medio")
    ax.grid(True)
    ax.legend(frameon=False)
    finish(fig, OUT_DIR / "fig11_embedding_distance.png")


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_geometry()
    fig_label_distribution()
    fig_label_sweep()
    fig_pipeline()
    fig_training_curves()
    fig_model_comparison()
    fig_ceiling()
    fig_feature_importance()
    fig_shap()
    fig_qualitative_case()
    fig_embedding_distance()


if __name__ == "__main__":
    main()
