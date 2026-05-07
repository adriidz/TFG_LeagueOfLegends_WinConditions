#!/usr/bin/env python3
"""
Plot and summarize the generated continuous support label distribution.

Input
-----
Canonical support_scores parquet produced by new_02b_grid_support_scores.py.

Outputs
-------
CSV/JSON summaries and PNG plots under ProgresoActual/analysis by default.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SUPPORT_SCORES = os.path.join(
    "ProgresoActual", "data", "clean", "scores", "support_scores_sample5_m12.parquet"
)
DEFAULT_OUTDIR = os.path.join("ProgresoActual", "analysis", "support_label_distribution")


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot distribution diagnostics for support roam labels.")
    p.add_argument("--support-scores-path", default=DEFAULT_SUPPORT_SCORES)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--score-col", default="support_roam_score_v2")
    p.add_argument("--champion-col", default="support_champion_name")
    p.add_argument("--bins", type=int, default=40)
    p.add_argument("--top-champions", type=int, default=25)
    p.add_argument("--min-champion-count", type=int, default=20)
    p.add_argument(
        "--boxplot-min-count",
        type=int,
        default=None,
        help=(
            "Optional minimum count used only for the champion boxplot. "
            "If omitted, --min-champion-count is used."
        ),
    )
    return p.parse_args()


def save_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def numeric_summary(scores: pd.Series) -> dict:
    valid = pd.to_numeric(scores, errors="coerce").dropna()
    if valid.empty:
        return {"n": 0}
    return {
        "n": int(valid.shape[0]),
        "missing": int(scores.shape[0] - valid.shape[0]),
        "mean": float(valid.mean()),
        "std": float(valid.std(ddof=0)),
        "min": float(valid.min()),
        "q01": float(valid.quantile(0.01)),
        "q05": float(valid.quantile(0.05)),
        "q25": float(valid.quantile(0.25)),
        "median": float(valid.median()),
        "q75": float(valid.quantile(0.75)),
        "q95": float(valid.quantile(0.95)),
        "q99": float(valid.quantile(0.99)),
        "max": float(valid.max()),
        "share_lt_0": float((valid < 0.0).mean()),
        "share_gt_1": float((valid > 1.0).mean()),
        "share_eq_0": float((valid == 0.0).mean()),
        "share_eq_1": float((valid == 1.0).mean()),
    }


def save_histogram(scores: pd.Series, out_path: str, bins: int, title_suffix: str) -> None:
    plt.figure(figsize=(8.2, 6.0))
    plt.hist(scores, bins=bins, range=(0.0, 1.0), color="#276fbf", alpha=0.85, edgecolor="white")
    plt.xlabel("Support roam score")
    plt.ylabel("Match-team rows")
    plt.title(f"Support label distribution{title_suffix}")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_cdf(scores: pd.Series, out_path: str, title_suffix: str) -> None:
    sorted_scores = np.sort(scores.to_numpy(dtype=float))
    y = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    plt.figure(figsize=(8.2, 6.0))
    plt.plot(sorted_scores, y, color="#2a9d8f", linewidth=2)
    plt.xlabel("Support roam score")
    plt.ylabel("Cumulative share")
    plt.title(f"Support label empirical CDF{title_suffix}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_component_scatter(df: pd.DataFrame, score_col: str, out_path: str) -> None:
    component_cols = [c for c in ["outside_ratio", "far_ratio", "xp_gap"] if c in df.columns]
    if not component_cols:
        return
    fig, axes = plt.subplots(1, len(component_cols), figsize=(6.2 * len(component_cols), 5.8), sharey=True)
    if len(component_cols) == 1:
        axes = [axes]
    sample = df.dropna(subset=[score_col] + component_cols).copy()
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)
    for ax, col in zip(axes, component_cols):
        ax.scatter(sample[col], sample[score_col], s=18, alpha=0.25, color="#276fbf")
        ax.set_xlabel(col)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(score_col)
    fig.suptitle("Support label vs heuristic components")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_side_histograms(df: pd.DataFrame, score_col: str, out_path: str, bins: int) -> None:
    if "side" not in df.columns:
        return
    work = df.dropna(subset=[score_col, "side"]).copy()
    if work.empty:
        return
    plt.figure(figsize=(8.2, 6.0))
    for side, group in work.groupby("side"):
        plt.hist(
            group[score_col],
            bins=bins,
            range=(0.0, 1.0),
            alpha=0.45,
            label=str(side),
            edgecolor="white",
        )
    plt.xlabel("Support roam score")
    plt.ylabel("Match-team rows")
    plt.title("Support label distribution by side")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def champion_summary(df: pd.DataFrame, score_col: str, champion_col: str, min_count: int) -> Optional[pd.DataFrame]:
    if champion_col not in df.columns:
        return None
    summary = (
        df.dropna(subset=[score_col])
        .groupby(champion_col, dropna=False)[score_col]
        .agg(count="count", mean="mean", median="median", std="std", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        .reset_index()
        .rename(columns={champion_col: "champion_name"})
    )
    summary = summary[summary["count"] >= min_count].sort_values(["mean", "count"], ascending=[False, False])
    return summary


def save_champion_boxplot(
    df: pd.DataFrame,
    score_col: str,
    champion_col: str,
    summary: pd.DataFrame,
    out_path: str,
    top_champions: int,
) -> None:
    if summary is None or summary.empty or champion_col not in df.columns:
        return
    champs = summary.head(top_champions)["champion_name"].astype(str).tolist()
    work = df[df[champion_col].astype(str).isin(champs)].dropna(subset=[score_col, champion_col]).copy()
    if work.empty:
        return
    data = [work.loc[work[champion_col].astype(str) == champ, score_col].to_numpy(dtype=float) for champ in champs]
    plt.figure(figsize=(max(13, top_champions * 0.68), 8.0))
    plt.boxplot(data, labels=champs, showfliers=False)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Support roam score")
    plt.title(f"Top {len(champs)} champions by generated mean support roam score")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    configure_plot_style()
    args = parse_args()
    if not os.path.exists(args.support_scores_path):
        raise SystemExit(f"Missing support scores parquet: {args.support_scores_path}")

    ensure_dir(args.outdir)
    df = pd.read_parquet(args.support_scores_path)
    if args.score_col not in df.columns:
        raise SystemExit(f"Missing score column: {args.score_col}")

    df[args.score_col] = pd.to_numeric(df[args.score_col], errors="coerce")
    valid_scores = df[args.score_col].dropna()
    if valid_scores.empty:
        raise SystemExit(f"No valid numeric values in {args.score_col}")

    summary = numeric_summary(df[args.score_col])
    summary.update({
        "support_scores_path": os.path.abspath(args.support_scores_path),
        "score_col": args.score_col,
        "rows": int(len(df)),
        "unique_matches": int(df["match_id"].nunique()) if "match_id" in df.columns else None,
        "unique_match_team_keys": int(df[["match_id", "team_id"]].drop_duplicates().shape[0])
        if {"match_id", "team_id"}.issubset(df.columns)
        else None,
        "config_id": str(df["config_id"].dropna().iloc[0]) if "config_id" in df.columns and df["config_id"].notna().any() else None,
        "window_tag": str(df["window_tag"].dropna().iloc[0]) if "window_tag" in df.columns and df["window_tag"].notna().any() else None,
    })

    save_json(summary, os.path.join(args.outdir, "support_label_distribution_summary.json"))
    pd.DataFrame([summary]).to_csv(os.path.join(args.outdir, "support_label_distribution_summary.csv"), index=False)

    title_suffix = f" ({summary['window_tag']})" if summary.get("window_tag") else ""
    save_histogram(valid_scores, os.path.join(args.outdir, "support_label_histogram.png"), args.bins, title_suffix)
    save_cdf(valid_scores, os.path.join(args.outdir, "support_label_cdf.png"), title_suffix)
    save_component_scatter(df, args.score_col, os.path.join(args.outdir, "support_label_component_scatter.png"))
    save_side_histograms(df, args.score_col, os.path.join(args.outdir, "support_label_by_side_histogram.png"), args.bins)

    if "side" in df.columns:
        side_summary = (
            df.groupby("side", dropna=False)[args.score_col]
            .agg(count="count", mean="mean", median="median", std="std")
            .reset_index()
        )
        side_summary.to_csv(os.path.join(args.outdir, "support_label_by_side_summary.csv"), index=False)

    champ_summary = champion_summary(df, args.score_col, args.champion_col, args.min_champion_count)
    if champ_summary is not None:
        champ_summary.to_csv(os.path.join(args.outdir, "support_label_by_champion_summary.csv"), index=False)
        boxplot_min_count = args.boxplot_min_count or args.min_champion_count
        boxplot_summary = champion_summary(df, args.score_col, args.champion_col, boxplot_min_count)
        save_champion_boxplot(
            df=df,
            score_col=args.score_col,
            champion_col=args.champion_col,
            summary=boxplot_summary,
            out_path=os.path.join(args.outdir, "support_label_top_champion_boxplot.png"),
            top_champions=args.top_champions,
        )
        if boxplot_min_count != args.min_champion_count and boxplot_summary is not None:
            save_champion_boxplot(
                df=df,
                score_col=args.score_col,
                champion_col=args.champion_col,
                summary=boxplot_summary,
                out_path=os.path.join(
                    args.outdir,
                    f"support_label_top_champion_boxplot_min{boxplot_min_count}.png",
                ),
                top_champions=args.top_champions,
            )

    print(f"Saved support label distribution analysis: {os.path.abspath(args.outdir)}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
