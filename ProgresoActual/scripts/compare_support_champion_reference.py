#!/usr/bin/env python3
"""
Compare generated support scores by champion against an expert/reference table.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DEFAULT_SUPPORT_SCORES = os.path.join("ProgresoActual", "data", "clean", "scores", "support_scores_sample5_m12.parquet")
DEFAULT_REFERENCE = os.path.join("ProgresoActual", "references", "champion_support_reference.csv")
DEFAULT_OUTDIR = os.path.join("ProgresoActual", "analysis", "champion_reference")


def configure_plot_style() -> None:
    """Use larger typography so figures remain readable in two-column reports."""
    plt.rcParams.update({
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 17,
        "figure.titlesize": 23,
        "lines.linewidth": 3,
        "axes.linewidth": 1.4,
    })


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_name(value: object) -> str:
    return str(value).strip().lower().replace(" ", "").replace("'", "").replace(".", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Champion-level comparison for support roam scores.")
    p.add_argument("--support-scores-path", default=DEFAULT_SUPPORT_SCORES)
    p.add_argument("--reference-path", default=DEFAULT_REFERENCE)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--score-col", default="support_roam_score_v2")
    p.add_argument("--champion-col", default="support_champion_name")
    p.add_argument("--min-count", type=int, default=20)
    return p.parse_args()


def select_scatter_label_rows(df: pd.DataFrame, max_labels: int = 8) -> pd.DataFrame:
    """Label only extreme points so the scatter remains readable."""
    candidates = []
    for col in ["expert_support_roam_score", "generated_mean"]:
        candidates.append(df.nsmallest(2, col))
        candidates.append(df.nlargest(2, col))
    candidates.append(df.reindex(df["delta_mean_vs_expert"].abs().sort_values(ascending=False).index).head(3))
    labels = pd.concat(candidates, ignore_index=False)
    labels = labels[~labels.index.duplicated(keep="first")].copy()
    labels["_label_priority"] = labels["delta_mean_vs_expert"].abs()
    labels = labels.sort_values("_label_priority", ascending=False).head(max_labels)
    return labels.drop(columns=["_label_priority"])


def save_scatter(df: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.8))
    y_max = float(df["generated_mean"].max())
    y_limit = min(1.0, max(0.45, y_max + 0.02))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, y_limit)
    reference_y_end = y_limit
    ax.plot(
        [0.0, 1.0],
        [0.0, reference_y_end],
        color="#4a4a4a",
        linestyle=(0, (6, 5)),
        linewidth=2.0,
        alpha=0.55,
        zorder=1,
    )
    confidence = pd.to_numeric(df.get("expert_confidence"), errors="coerce")
    scatter_kwargs = {
        "alpha": 0.82,
        "s": 78,
        "edgecolor": "black",
        "linewidth": 0.35,
        "zorder": 3,
    }
    if confidence.notna().any():
        points = ax.scatter(
            df["expert_support_roam_score"],
            df["generated_mean"],
            c=confidence,
            cmap="RdYlGn",
            vmin=0.0,
            vmax=1.0,
            **scatter_kwargs,
        )
        cbar = fig.colorbar(points, ax=ax)
        cbar.set_label("Expert label confidence")
    else:
        ax.scatter(
            df["expert_support_roam_score"],
            df["generated_mean"],
            color="#276fbf",
            **scatter_kwargs,
        )

    for row in select_scatter_label_rows(df).itertuples(index=False):
        x = float(row.expert_support_roam_score)
        y = float(row.generated_mean)
        x_offset = -8 if x > 0.72 else 8
        y_offset = -8 if y > y_limit * 0.72 else 8
        ax.annotate(
            str(row.champion_name),
            xy=(x, y),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=13,
            ha="right" if x_offset < 0 else "left",
            va="top" if y_offset < 0 else "bottom",
            bbox={"facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.82, "pad": 1.5},
            zorder=5,
        )
    ax.set_xlabel("Expert support roam score")
    ax.set_ylabel("Generated mean support roam score")
    ax.set_title("Support score: generated vs expert reference")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_top_delta(df: pd.DataFrame, out_path: str) -> None:
    work = df.reindex(df["delta_mean_vs_expert"].abs().sort_values(ascending=False).index).head(20)
    plt.figure(figsize=(13, 8))
    colors = np.where(work["delta_mean_vs_expert"] >= 0, "#276fbf", "#c44536")
    plt.bar(work["champion_name"].astype(str), work["delta_mean_vs_expert"], color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Generated mean - expert score")
    plt.title("Largest champion-level deviations")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_expert_histogram(reference: pd.DataFrame, out_path: str) -> None:
    expert = pd.to_numeric(reference["expert_support_roam_score"], errors="coerce").dropna()
    if expert.empty:
        return
    plt.figure(figsize=(8.2, 6.0))
    plt.hist(expert, bins=12, range=(0.0, 1.0), color="#5b8c5a", alpha=0.85, edgecolor="white")
    plt.xlabel("Expert support roam score")
    plt.ylabel("Champions")
    plt.title("Expert support roam score distribution")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_expert_observed_distribution(df: pd.DataFrame, out_path: str) -> None:
    compared = df.dropna(subset=["expert_support_roam_score", "generated_mean"]).copy()
    if compared.empty:
        return
    plt.figure(figsize=(8.2, 6.0))
    plt.hist(
        compared["expert_support_roam_score"],
        bins=12,
        range=(0.0, 1.0),
        alpha=0.55,
        label="Expert prior",
        edgecolor="white",
    )
    plt.hist(
        compared["generated_mean"],
        bins=12,
        range=(0.0, 1.0),
        alpha=0.55,
        label="Observed champion mean",
        edgecolor="white",
    )
    plt.xlabel("Support roam score")
    plt.ylabel("Champions")
    plt.title("Expert prior vs observed champion means")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def compute_correlations(df: pd.DataFrame) -> Dict[str, float]:
    valid = df[["expert_support_roam_score", "generated_mean"]].dropna()
    if len(valid) < 2:
        return {"pearson_corr": float("nan"), "spearman_corr": float("nan"), "n_compared": int(len(valid))}
    pearson = float(np.corrcoef(valid["expert_support_roam_score"], valid["generated_mean"])[0, 1])
    spear = spearmanr(valid["expert_support_roam_score"], valid["generated_mean"]).correlation
    return {
        "pearson_corr": pearson,
        "spearman_corr": float(spear) if spear is not None else float("nan"),
        "n_compared": int(len(valid)),
    }


def main() -> None:
    configure_plot_style()
    args = parse_args()
    if not os.path.exists(args.support_scores_path):
        raise SystemExit(f"Missing support scores parquet: {args.support_scores_path}")
    if not os.path.exists(args.reference_path):
        raise SystemExit(f"Missing champion reference CSV: {args.reference_path}")

    ensure_dir(args.outdir)
    scores = pd.read_parquet(args.support_scores_path)
    reference = pd.read_csv(args.reference_path)
    for col in (args.champion_col, args.score_col):
        if col not in scores.columns:
            raise SystemExit(f"Missing score table column: {col}")
    if "champion_name" not in reference.columns or "expert_support_roam_score" not in reference.columns:
        raise SystemExit("Reference must include champion_name and expert_support_roam_score.")

    scores[args.score_col] = pd.to_numeric(scores[args.score_col], errors="coerce")
    by_champ = (
        scores.dropna(subset=[args.score_col])
        .groupby(args.champion_col, dropna=False)[args.score_col]
        .agg(generated_count="count", generated_mean="mean", generated_median="median", generated_std="std")
        .reset_index()
        .rename(columns={args.champion_col: "champion_name"})
    )
    by_champ = by_champ[by_champ["generated_count"] >= args.min_count].copy()
    by_champ["_join_name"] = by_champ["champion_name"].map(normalize_name)
    reference["_join_name"] = reference["champion_name"].map(normalize_name)

    merged = by_champ.merge(
        reference.drop(columns=["champion_name"]),
        on="_join_name",
        how="left",
        validate="one_to_one",
    ).drop(columns=["_join_name"])
    merged["delta_mean_vs_expert"] = merged["generated_mean"] - merged["expert_support_roam_score"]
    merged = merged.sort_values("generated_mean", ascending=False)

    comparison_path = os.path.join(args.outdir, "support_champion_reference_comparison.csv")
    merged.to_csv(comparison_path, index=False)
    head_cols = [
        "champion_name",
        "expert_archetype",
        "expert_support_roam_score",
        "expert_confidence",
        "notes",
    ]
    reference_head = reference.dropna(subset=["expert_support_roam_score"]).copy()
    reference_head = reference_head[[c for c in head_cols if c in reference_head.columns]].head(3)
    reference_head.to_csv(os.path.join(args.outdir, "expert_reference_head3.csv"), index=False)

    metrics = compute_correlations(merged)
    metrics.update({
        "support_scores_path": os.path.abspath(args.support_scores_path),
        "reference_path": os.path.abspath(args.reference_path),
        "score_col": args.score_col,
        "min_count": args.min_count,
        "champions_in_score_table": int(by_champ.shape[0]),
        "champions_with_expert_reference": int(merged["expert_support_roam_score"].notna().sum()),
    })
    with open(os.path.join(args.outdir, "support_champion_reference_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    compared = merged.dropna(subset=["expert_support_roam_score", "generated_mean"])
    save_expert_histogram(reference, os.path.join(args.outdir, "expert_support_score_histogram.png"))
    if not compared.empty:
        save_scatter(compared, os.path.join(args.outdir, "generated_vs_expert_scatter.png"))
        save_top_delta(compared, os.path.join(args.outdir, "largest_deviations.png"))
        save_expert_observed_distribution(compared, os.path.join(args.outdir, "expert_vs_observed_distribution.png"))

    print(f"Saved comparison: {os.path.abspath(comparison_path)}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
