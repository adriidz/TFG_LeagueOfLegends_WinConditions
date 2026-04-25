#!/usr/bin/env python3
"""
plot_progress_final.py

Genera:
- Figura 9: curvas de train loss y validation loss del experimento final.
- Figura 10: métricas finales por tarea en la configuración seleccionada.

Entrada esperada:
- --run-dir: directorio de una run final, con history.csv y model_config.json.
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
    "jungle_presence_label": "Jungla",
    "support_roam_label": "Support",
    "team_tendency_label": "Team tendency",
}
METRIC_ORDER = ["accuracy", "balanced_accuracy", "f1_macro"]
METRIC_TITLE = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "f1_macro": "F1 macro",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genera Figuras 9 y 10 de la run final.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_fig09(history_df: pd.DataFrame, best_epoch: int, outpath: Path) -> None:
    plt.figure(figsize=(8.2, 5.2))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="Validation loss")
    if best_epoch is not None and best_epoch in set(history_df["epoch"].tolist()):
        best_val = float(history_df.loc[history_df["epoch"] == best_epoch, "val_loss"].iloc[0])
        plt.scatter([best_epoch], [best_val], marker="*", s=120, label=f"Best epoch = {best_epoch}")
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Figura 9. Curvas de train loss y validation loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close()


def plot_fig10(best_metrics: Dict[str, dict], outpath: Path) -> None:
    tasks = [k for k in TASK_MAP.keys() if k in best_metrics]
    x = np.arange(len(tasks))
    width = 0.22

    plt.figure(figsize=(9, 5.2))
    for i, metric in enumerate(METRIC_ORDER):
        vals = [float(best_metrics[t].get(metric, float("nan"))) for t in tasks]
        offset = (i - 1) * width
        plt.bar(x + offset, vals, width=width, label=METRIC_TITLE[metric])

    plt.xticks(x, [TASK_MAP[t] for t in tasks])
    plt.ylabel("Valor")
    plt.ylim(0, 1.0)
    plt.title("Figura 10. Métricas finales por tarea en la configuración seleccionada")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    history_path = run_dir / "history.csv"
    config_path = run_dir / "model_config.json"
    if not history_path.exists():
        raise FileNotFoundError(f"No encuentro history.csv en {run_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"No encuentro model_config.json en {run_dir}")

    history_df = pd.read_csv(history_path)
    cfg = load_json(config_path)
    best_epoch = int(cfg.get("best_epoch")) if cfg.get("best_epoch") is not None else None
    best_metrics = cfg.get("best_metrics", {})

    history_df.to_csv(outdir / "fig09_history_data.csv", index=False)
    pd.DataFrame(best_metrics).T.to_csv(outdir / "fig10_best_metrics_data.csv")

    plot_fig09(history_df, best_epoch, outdir / "fig09_train_val_loss_curves.png")
    plot_fig10(best_metrics, outdir / "fig10_final_metrics_by_task.png")

    print("Hecho.")
    print(f"- Figura 9: {outdir / 'fig09_train_val_loss_curves.png'}")
    print(f"- Figura 10: {outdir / 'fig10_final_metrics_by_task.png'}")


if __name__ == "__main__":
    main()
