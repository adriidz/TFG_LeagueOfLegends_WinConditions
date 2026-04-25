#!/usr/bin/env python3
"""
Aggregate support OAT tuning results after cluster runs are copied back locally.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate support OAT tuning metrics.")
    p.add_argument("--experiment-name", default="support_oat_sample5_m12")
    p.add_argument("--manifest", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--objective", default="val_mse")
    return p.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metric_value(metrics: Dict[str, Any], objective: str) -> float:
    if objective in metrics:
        return float(metrics[objective])
    if objective == "val_mse" and "mse" in metrics:
        return float(metrics["mse"])
    if objective == "val_mse_loss" and "best_val_mse_loss" in metrics:
        return float(metrics["best_val_mse_loss"])
    return float("nan")


def save_ranking_plot(df: pd.DataFrame, objective: str, out_path: str) -> None:
    valid = df[df[objective].notna()].sort_values(objective).head(25)
    if valid.empty:
        return
    plt.figure(figsize=(10, max(5, len(valid) * 0.35)))
    plt.barh(valid["experiment_id"], valid[objective], color="#276fbf")
    plt.gca().invert_yaxis()
    plt.xlabel(objective)
    plt.title("OAT tuning ranking")
    plt.tight_layout()
    plt.savefig(out_path, dpi=170)
    plt.close()


def save_phase_parameter_plot(df: pd.DataFrame, objective: str, out_path: str) -> None:
    valid = df[df[objective].notna()].copy()
    if valid.empty:
        return
    phases = list(valid["phase"].dropna().unique())
    fig, axes = plt.subplots(len(phases), 1, figsize=(11, max(4, 3.5 * len(phases))))
    if len(phases) == 1:
        axes = [axes]
    for ax, phase in zip(axes, phases):
        work = valid[valid["phase"] == phase].sort_values(objective)
        labels = work["changed_parameter"].astype(str) + "=" + work["changed_value"].astype(str)
        ax.bar(labels, work[objective], color="#2a9d8f")
        ax.set_title(str(phase))
        ax.set_ylabel(objective)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_markdown(df: pd.DataFrame, objective: str, out_path: str) -> None:
    def markdown_table(table_df: pd.DataFrame) -> str:
        if table_df.empty:
            return "No rows."
        text_df = table_df.fillna("").astype(str)
        headers = list(text_df.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in text_df.itertuples(index=False):
            lines.append("| " + " | ".join(str(v) for v in row) + " |")
        return "\n".join(lines)

    cols = ["phase", "experiment_id", "changed_parameter", "changed_value", objective, "mae", "r2", "pearson_corr", "spearman_corr"]
    cols = [c for c in cols if c in df.columns]
    top = df.sort_values(objective, na_position="last")[cols].head(20)
    lines = ["# Support OAT tuning summary", "", f"Objective: `{objective}`", "", "## Top runs", ""]
    lines.append(markdown_table(top))
    lines.append("")
    lines.append("## Best by phase")
    lines.append("")
    best = df[df[objective].notna()].sort_values(objective).groupby("phase", as_index=False).head(1)
    lines.append(markdown_table(best[cols]) if not best.empty else "No completed runs.")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = args.manifest or os.path.join(
        "ProgresoActual", "experiments", "support_oat", args.experiment_name, "runs_manifest.csv"
    )
    outdir = args.outdir or os.path.join("ProgresoActual", "analysis", "oat_tuning", args.experiment_name)
    if not os.path.exists(manifest):
        raise SystemExit(f"Missing manifest: {manifest}")
    ensure_dir(outdir)

    runs = pd.read_csv(manifest)
    rows: List[Dict[str, Any]] = []
    for row in runs.to_dict(orient="records"):
        out_dir = str(row.get("train_outdir", ""))
        metrics_path = os.path.join(out_dir, "metrics.json")
        merged = dict(row)
        if os.path.exists(metrics_path):
            metrics = load_json(metrics_path)
            merged.update(metrics)
            merged[args.objective] = metric_value(metrics, args.objective)
            merged["status"] = "completed"
            merged["metrics_path"] = metrics_path
        else:
            merged[args.objective] = float("nan")
            merged["status"] = "missing_metrics"
            merged["metrics_path"] = metrics_path
        rows.append(merged)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(args.objective, na_position="last")
    summary_path = os.path.join(outdir, "experiments_summary.csv")
    summary.to_csv(summary_path, index=False)

    completed = summary[summary[args.objective].notna()].copy()
    best_by_phase = {}
    if not completed.empty:
        for phase, group in completed.groupby("phase"):
            best_by_phase[str(phase)] = group.sort_values(args.objective).iloc[0].to_dict()
    with open(os.path.join(outdir, "best_by_phase.json"), "w", encoding="utf-8") as f:
        json.dump(best_by_phase, f, indent=2, ensure_ascii=False, default=str)

    save_ranking_plot(summary, args.objective, os.path.join(outdir, "val_mse_ranking.png"))
    save_phase_parameter_plot(summary, args.objective, os.path.join(outdir, "metric_vs_parameter_plots.png"))
    write_markdown(summary, args.objective, os.path.join(outdir, "experiments_summary.md"))

    print(f"Saved OAT summary: {os.path.abspath(outdir)}")
    print(f"Completed runs: {int((summary['status'] == 'completed').sum())}/{len(summary)}")


if __name__ == "__main__":
    main()
