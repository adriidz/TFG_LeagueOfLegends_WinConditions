#!/usr/bin/env python3
"""
Create comparison tables and plots from support score grid outputs.

Reads:
- support_score_grid_long*.parquet
- support_score_grid_summary*.csv

Produces:
- coverage vs mean plots
- top config histograms
- config heatmaps (if 2D grids are present)
- champion ranking snippets for selected configs (if champion summary exists)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_GRID_DIR = os.path.join("data_new", "analysis", "support_grid")
DEFAULT_OUT_DIR = os.path.join("data_new", "analysis", "support_grid_compare")


def format_sample_suffix(sample_frac: Optional[float]) -> str:
    if sample_frac is None or sample_frac <= 0.0 or sample_frac >= 1.0:
        return ""
    return f"_sample{int(round(sample_frac * 100))}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare support score grid configurations.")
    p.add_argument("--grid-dir", default=DEFAULT_GRID_DIR)
    p.add_argument("--outdir", default=DEFAULT_OUT_DIR)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--top-k", type=int, default=6)
    return p.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    suffix = format_sample_suffix(args.sample_frac)
    summary_path = os.path.join(args.grid_dir, f"support_score_grid_summary{suffix}.csv")
    long_path = os.path.join(args.grid_dir, f"support_score_grid_long{suffix}.parquet")
    champ_path = os.path.join(args.grid_dir, f"support_score_grid_champion_summary{suffix}.csv")

    if not os.path.exists(summary_path):
        raise SystemExit(f"No existe el summary CSV: {summary_path}")
    summary = pd.read_csv(summary_path)
    long_df = pd.read_parquet(long_path) if os.path.exists(long_path) else None
    champ_df = pd.read_csv(champ_path) if os.path.exists(champ_path) else None

    ensure_dir(args.outdir)
    print(f"[Summary] {os.path.abspath(summary_path)}")
    print(f"[Outdir]  {os.path.abspath(args.outdir)}")

    # Ranking heuristic: prefer good coverage and healthy spread
    work = summary.copy()
    work["rank_score"] = (
        work["coverage"].fillna(0.0) * 0.50
        + work["score_std"].fillna(0.0) * 0.30
        + work["score_mean"].fillna(0.0) * 0.20
    )
    work = work.sort_values(["rank_score", "coverage", "score_std"], ascending=[False, False, False])
    work.to_csv(os.path.join(args.outdir, "ranked_configs.csv"), index=False)

    top = work.head(args.top_k).copy()

    plt.figure(figsize=(8, 5))
    plt.scatter(work["coverage"], work["score_mean"], alpha=0.7)
    for _, r in top.iterrows():
        plt.annotate(r["config_id"], (r["coverage"], r["score_mean"]), fontsize=8)
    plt.xlabel("Coverage")
    plt.ylabel("Mean score")
    plt.title("Support config comparison: coverage vs mean score")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "coverage_vs_mean.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.scatter(work["coverage"], work["score_std"], alpha=0.7)
    for _, r in top.iterrows():
        plt.annotate(r["config_id"], (r["coverage"], r["score_std"]), fontsize=8)
    plt.xlabel("Coverage")
    plt.ylabel("Score std")
    plt.title("Support config comparison: coverage vs score std")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "coverage_vs_std.png"), dpi=150)
    plt.close()

    # Histograms for top-k configs
    if long_df is not None and not long_df.empty:
        hist_dir = os.path.join(args.outdir, "top_config_hists")
        ensure_dir(hist_dir)
        for _, r in top.iterrows():
            cfg = r["config_id"]
            part = long_df[long_df["config_id"] == cfg]
            if part.empty:
                continue
            plt.figure(figsize=(8, 5))
            plt.hist(part["support_roam_score_v2"].dropna(), bins=40)
            plt.title(f"Support score distribution - {cfg}")
            plt.xlabel("support_roam_score_v2")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(hist_dir, f"{cfg}_hist.png"), dpi=150)
            plt.close()

    # Simple heatmap if exactly one weight-triplet and one far-threshold per start/max grid
    try:
        subset = work[["start_minute", "max_minute", "coverage"]].drop_duplicates()
        if subset.shape[0] == work.shape[0]:
            pivot = work.pivot(index="start_minute", columns="max_minute", values="coverage")
            plt.figure(figsize=(7, 5))
            plt.imshow(pivot.values, aspect="auto")
            plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
            plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
            plt.colorbar(label="Coverage")
            plt.xlabel("max_minute")
            plt.ylabel("start_minute")
            plt.title("Coverage heatmap")
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, "coverage_heatmap.png"), dpi=150)
            plt.close()
    except Exception:
        pass

    # Champion snapshots for top configs
    if champ_df is not None and not champ_df.empty:
        champ_dir = os.path.join(args.outdir, "top_config_champions")
        ensure_dir(champ_dir)
        for _, r in top.iterrows():
            cfg = r["config_id"]
            part = champ_df[champ_df["config_id"] == cfg].sort_values(["count", "mean"], ascending=[False, False]).head(20)
            if not part.empty:
                part.to_csv(os.path.join(champ_dir, f"{cfg}_top20_champions.csv"), index=False)

    print("\nTop configs:")
    print(top[["config_id", "coverage", "score_mean", "score_std", "rank_score"]].to_string(index=False))
    print(f"\nOutputs saved to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
