#!/usr/bin/env python3
"""
plot_progress_phase1.py

Genera las figuras del estudio de ventanas para el Informe de Progreso I:
- Figura 3: rendimiento del modelo por ventana temporal y tarea.
- Figura 4: distribución de etiquetas por ventana temporal.
- Figura 5: acuerdo de labels entre ventanas.
- Figura 6: flips entre extremos ignorando ambiguous.

Entradas esperadas:
- training_root: directorio con runs entrenadas (busca recursivamente model_config.json)
- labels_root: directorio con labels parquet por ventana
- stability_root: salida de 02c_analyze_label_stability.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASK_INFO = {
    "jungle_presence_label": {
        "task": "jungle",
        "title": "Jungla",
        "label_col": "jungle_presence_label",
        "class_order": ["farm_oriented", "ambiguous", "map_presence"],
    },
    "support_roam_label": {
        "task": "support",
        "title": "Support",
        "label_col": "support_roam_label",
        "class_order": ["lane_anchored", "ambiguous", "roamer"],
    },
    "team_tendency_label": {
        "task": "team",
        "title": "Team tendency",
        "label_col": "team_tendency_label",
        "class_order": ["botside_oriented", "ambiguous", "topside_oriented"],
    },
}
TASK_ORDER = ["jungle", "support", "team"]
TASK_TITLE = {"jungle": "Jungla", "support": "Support", "team": "Team tendency"}
WINDOW_RE = re.compile(r"_m(\d{2})_")

PAIRWISE_PARQUET = "all_tasks_pairwise_agreement.parquet"
PAIRWISE_CSV = "all_tasks_pairwise_agreement.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera Figuras 3-6 del estudio de ventanas.")
    p.add_argument("--labels-root", required=True, help="Directorio con labels parquet por ventana.")
    p.add_argument("--stability-root", required=True, help="Directorio de salida de 02c_analyze_label_stability.py")
    p.add_argument("--training-root", required=True, help="Directorio con runs entrenadas del estudio de ventanas.")
    p.add_argument("--windows", nargs="+", type=int, required=True)
    p.add_argument("--reference-window", type=int, default=10)
    p.add_argument("--metric", choices=["accuracy", "balanced_accuracy", "f1_macro"], default="f1_macro")
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pairwise_table(stability_root: Path) -> pd.DataFrame:
    parquet_path = stability_root / PAIRWISE_PARQUET
    csv_path = stability_root / PAIRWISE_CSV
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No encuentro {PAIRWISE_PARQUET} ni {PAIRWISE_CSV} en {stability_root}")


def collect_training_metrics(training_root: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for cfg_path in training_root.rglob("model_config.json"):
        cfg = load_json(cfg_path)
        target_schema = cfg.get("target_schema")
        label_max_minute = cfg.get("label_max_minute")
        quantile_tag = cfg.get("quantile_or_threshold_tag")
        feature_groups = ",".join(cfg.get("feature_groups_active", []))
        run_dir = str(cfg_path.parent)

        best_metrics = cfg.get("best_metrics", {})
        for target_col, met in best_metrics.items():
            task = TASK_INFO.get(target_col, {}).get("task")
            if not task:
                continue
            rows.append({
                "run_dir": run_dir,
                "target_schema": target_schema,
                "window": int(round(float(label_max_minute))) if label_max_minute is not None else None,
                "quantile_tag": quantile_tag,
                "feature_groups": feature_groups,
                "task": task,
                "accuracy": float(met.get("accuracy", float("nan"))),
                "balanced_accuracy": float(met.get("balanced_accuracy", float("nan"))),
                "f1_macro": float(met.get("f1_macro", float("nan"))),
                "valid_samples": int(met.get("valid_samples", 0)),
            })
    if not rows:
        raise FileNotFoundError(f"No encontré model_config.json bajo {training_root}")
    df = pd.DataFrame(rows)
    df = df[df["window"].notna()].copy()
    df["window"] = df["window"].astype(int)
    return df.sort_values(["task", "window", "run_dir"]).reset_index(drop=True)


def parse_window_from_name(path: Path) -> Optional[int]:
    m = WINDOW_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1))


def collect_label_distributions(labels_root: Path, windows: Sequence[int]) -> pd.DataFrame:
    rows: List[dict] = []
    allowed_windows = set(int(w) for w in windows)

    for pq in labels_root.glob("*.parquet"):
        name = pq.name
        if not any(name.startswith(prefix) for prefix in ("jungle_labels", "support_labels", "team_tendency_labels")):
            continue
        window = parse_window_from_name(pq)
        if window is None or window not in allowed_windows:
            continue

        if name.startswith("jungle_labels"):
            target_col = "jungle_presence_label"
        elif name.startswith("support_labels"):
            target_col = "support_roam_label"
        else:
            target_col = "team_tendency_label"

        task = TASK_INFO[target_col]["task"]
        class_order = TASK_INFO[target_col]["class_order"]
        df = pd.read_parquet(pq, columns=[target_col])
        counts = df[target_col].value_counts(dropna=False)

        total = len(df)
        for cls in class_order:
            n = int(counts.get(cls, 0))
            rows.append({
                "task": task,
                "window": window,
                "label": cls,
                "n": n,
                "pct": (n / total) if total > 0 else float("nan"),
            })
        nan_n = int(df[target_col].isna().sum())
        if nan_n > 0:
            rows.append({
                "task": task,
                "window": window,
                "label": "NaN",
                "n": nan_n,
                "pct": (nan_n / total) if total > 0 else float("nan"),
            })

    if not rows:
        raise FileNotFoundError(f"No encontré labels parquet válidas en {labels_root}")
    out = pd.DataFrame(rows)
    return out.sort_values(["task", "window", "label"]).reset_index(drop=True)


def plot_fig03_window_performance(df_metrics: pd.DataFrame, windows: Sequence[int], metric: str, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, task in zip(axes, TASK_ORDER):
        sub = df_metrics[df_metrics["task"] == task].copy()
        sub = sub.groupby("window", as_index=False)[metric].mean()
        sub = sub[sub["window"].isin(windows)].sort_values("window")
        ax.plot(sub["window"], sub[metric], marker="o")
        ax.set_title(TASK_TITLE[task])
        ax.set_xlabel("Ventana (min)")
        ax.set_xticks(list(windows))
        ax.grid(alpha=0.3)
        if not sub.empty:
            best_idx = sub[metric].idxmax()
            best_x = int(sub.loc[best_idx, "window"])
            best_y = float(sub.loc[best_idx, metric])
            ax.scatter([best_x], [best_y], marker="*", s=120)
            ax.annotate(f"mejor={best_x}", (best_x, best_y), textcoords="offset points", xytext=(0, 8), ha="center")
    axes[0].set_ylabel(metric)
    fig.suptitle(f"Figura 3. Rendimiento por ventana temporal y tarea ({metric})", y=1.02)
    save_fig(outdir / "fig03_window_performance.png")


def plot_fig04_label_distribution(df_dist: pd.DataFrame, windows: Sequence[int], outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, task in zip(axes, TASK_ORDER):
        target_key = next(k for k, v in TASK_INFO.items() if v["task"] == task)
        class_order = TASK_INFO[target_key]["class_order"]
        pivot = (
            df_dist[df_dist["task"] == task]
            .pivot_table(index="window", columns="label", values="pct", aggfunc="sum", fill_value=0.0)
            .reindex(index=list(windows), fill_value=0.0)
        )
        bottoms = np.zeros(len(pivot), dtype=float)
        for label in class_order:
            vals = pivot[label].values if label in pivot.columns else np.zeros(len(pivot))
            ax.bar(pivot.index.astype(str), vals, bottom=bottoms, label=label)
            bottoms += vals
        if "NaN" in pivot.columns:
            vals = pivot["NaN"].values
            ax.bar(pivot.index.astype(str), vals, bottom=bottoms, label="NaN")
        ax.set_title(TASK_TITLE[task])
        ax.set_xlabel("Ventana (min)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Proporción")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle("Figura 4. Distribución de etiquetas por ventana temporal", y=1.06)
    save_fig(outdir / "fig04_label_distribution_by_window.png")


def _pair_label(row: pd.Series) -> str:
    return f"{int(row['window_a'])}→{int(row['window_b'])}"


def plot_fig05_agreement(df_pairwise: pd.DataFrame, outdir: Path, reference_window: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, task in zip(axes, TASK_ORDER):
        sub = df_pairwise[df_pairwise["task"] == task].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        cons = sub[sub["relation"] == "consecutive"].copy().sort_values(["window_a", "window_b"])
        ref = sub[sub["relation"] == "vs_reference"].copy().sort_values(["window_a", "window_b"])

        x_cons = np.arange(len(cons))
        x_ref = np.arange(len(cons), len(cons) + len(ref))

        if not cons.empty:
            ax.bar(x_cons, cons["agreement_non_ambiguous"], label="Consecutivas")
        if not ref.empty:
            ax.bar(x_ref, ref["agreement_non_ambiguous"], label=f"Vs ref={reference_window}")

        labels = [_pair_label(r) for _, r in cons.iterrows()] + [_pair_label(r) for _, r in ref.iterrows()]
        ax.set_xticks(list(x_cons) + list(x_ref))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(TASK_TITLE[task])
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Agreement no-ambiguous")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Figura 5. Acuerdo de labels entre ventanas", y=1.05)
    save_fig(outdir / "fig05_label_agreement_between_windows.png")


def plot_fig06_extreme_flips(df_pairwise: pd.DataFrame, outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, task in zip(axes, TASK_ORDER):
        sub = df_pairwise[(df_pairwise["task"] == task) & (df_pairwise["relation"] == "consecutive")].copy()
        sub = sub.sort_values(["window_a", "window_b"])
        labels = [_pair_label(r) for _, r in sub.iterrows()]
        vals = sub["extreme_flip_rate"].fillna(0.0).values
        ax.bar(labels, vals)
        ax.set_title(TASK_TITLE[task])
        ax.set_xlabel("Par de ventanas")
        ax.set_ylim(0, max(0.05, float(np.nanmax(vals)) * 1.25 if len(vals) else 0.05))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Extreme flip rate")
    fig.suptitle("Figura 6. Flips entre extremos ignorando ambiguous", y=1.02)
    save_fig(outdir / "fig06_extreme_flips_ignore_ambiguous.png")


def main() -> None:
    args = parse_args()

    labels_root = Path(args.labels_root)
    stability_root = Path(args.stability_root)
    training_root = Path(args.training_root)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    windows = [int(w) for w in args.windows]

    metrics_df = collect_training_metrics(training_root)
    label_dist_df = collect_label_distributions(labels_root, windows)
    pairwise_df = load_pairwise_table(stability_root)

    # Exportes auxiliares por si luego quieres revisar datos ya preparados.
    metrics_df.to_csv(outdir / "fig03_window_performance_data.csv", index=False)
    label_dist_df.to_csv(outdir / "fig04_label_distribution_data.csv", index=False)
    pairwise_df.to_csv(outdir / "fig05_fig06_pairwise_data.csv", index=False)

    plot_fig03_window_performance(metrics_df, windows, args.metric, outdir)
    plot_fig04_label_distribution(label_dist_df, windows, outdir)
    plot_fig05_agreement(pairwise_df, outdir, args.reference_window)
    plot_fig06_extreme_flips(pairwise_df, outdir)

    print("Hecho.")
    print(f"- Figura 3: {outdir / 'fig03_window_performance.png'}")
    print(f"- Figura 4: {outdir / 'fig04_label_distribution_by_window.png'}")
    print(f"- Figura 5: {outdir / 'fig05_label_agreement_between_windows.png'}")
    print(f"- Figura 6: {outdir / 'fig06_extreme_flips_ignore_ambiguous.png'}")


if __name__ == "__main__":
    main()
