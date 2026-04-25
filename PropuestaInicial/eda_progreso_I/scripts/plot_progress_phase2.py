#!/usr/bin/env python3
"""
plot_progress_phase2.py

Genera las figuras del estudio de reformulación binaria:
- Figura 7: comparación entre ternary, binary_clean y binary_full.
- Figura 8: comparación entre q20_80, q30_70 y q40_60 dentro de binary_clean.

Busca recursivamente model_config.json bajo --study-root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASK_MAP = {
    "jungle_presence_label": "jungle",
    "support_roam_label": "support",
    "team_tendency_label": "team",
}
TASK_ORDER = ["jungle", "support", "team"]
TASK_TITLE = {"jungle": "Jungla", "support": "Support", "team": "Team tendency"}
SCHEMA_ORDER = ["ternary", "binary_clean", "binary_full"]
QUANTILE_ORDER = ["q20_80", "q30_70", "q40_60"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera Figuras 7 y 8.")
    p.add_argument("--mode", choices=["schema", "quantile"], required=True)
    p.add_argument("--study-root", required=True)
    p.add_argument("--metric", choices=["accuracy", "balanced_accuracy", "f1_macro"], default="f1_macro")
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs(study_root: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for cfg_path in study_root.rglob("model_config.json"):
        cfg = load_json(cfg_path)
        best_metrics = cfg.get("best_metrics", {})
        for target_col, met in best_metrics.items():
            task = TASK_MAP.get(target_col)
            if not task:
                continue
            rows.append({
                "run_dir": str(cfg_path.parent),
                "task": task,
                "schema": cfg.get("target_schema"),
                "quantile_tag": cfg.get("quantile_or_threshold_tag"),
                "window": cfg.get("label_max_minute"),
                "feature_groups": ",".join(cfg.get("feature_groups_active", [])),
                "accuracy": float(met.get("accuracy", float("nan"))),
                "balanced_accuracy": float(met.get("balanced_accuracy", float("nan"))),
                "f1_macro": float(met.get("f1_macro", float("nan"))),
                "valid_samples": int(met.get("valid_samples", 0)),
            })
    if not rows:
        raise FileNotFoundError(f"No encontré model_config.json bajo {study_root}")
    df = pd.DataFrame(rows)
    df["window"] = pd.to_numeric(df["window"], errors="coerce")
    return df.sort_values(["task", "schema", "quantile_tag", "run_dir"]).reset_index(drop=True)


def plot_grouped_bars(df: pd.DataFrame, group_col: str, group_order: List[str], metric: str, title: str, outpath: Path) -> None:
    pivot = (
        df.groupby(["task", group_col], as_index=False)[metric]
        .mean()
        .pivot(index="task", columns=group_col, values=metric)
        .reindex(index=TASK_ORDER)
    )

    x = np.arange(len(TASK_ORDER))
    n_groups = len(group_order)
    width = 0.22 if n_groups >= 3 else 0.3

    plt.figure(figsize=(8.8, 5.2))
    for i, group in enumerate(group_order):
        vals = pivot[group].values if group in pivot.columns else np.full(len(TASK_ORDER), np.nan)
        offset = (i - (n_groups - 1) / 2.0) * width
        plt.bar(x + offset, vals, width=width, label=group)

    plt.xticks(x, [TASK_TITLE[t] for t in TASK_ORDER])
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    study_root = Path(args.study_root)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    df = collect_runs(study_root)
    df.to_csv(outdir / f"phase2_{args.mode}_raw_metrics.csv", index=False)

    if args.mode == "schema":
        plot_grouped_bars(
            df=df[df["schema"].isin(SCHEMA_ORDER)].copy(),
            group_col="schema",
            group_order=SCHEMA_ORDER,
            metric=args.metric,
            title=f"Figura 7. Comparación entre esquemas ({args.metric})",
            outpath=outdir / "fig07_schema_comparison.png",
        )
        print(f"Hecho. Figura 7: {outdir / 'fig07_schema_comparison.png'}")
    else:
        plot_grouped_bars(
            df=df[df["quantile_tag"].isin(QUANTILE_ORDER)].copy(),
            group_col="quantile_tag",
            group_order=QUANTILE_ORDER,
            metric=args.metric,
            title=f"Figura 8. Comparación entre cuantiles en binary_clean ({args.metric})",
            outpath=outdir / "fig08_quantile_comparison.png",
        )
        print(f"Hecho. Figura 8: {outdir / 'fig08_quantile_comparison.png'}")


if __name__ == "__main__":
    main()
