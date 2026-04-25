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


def save_scatter(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(df["expert_support_roam_score"], df["generated_mean"], alpha=0.75)
    for row in df.itertuples(index=False):
        if abs(float(row.delta_mean_vs_expert)) >= 0.25:
            plt.text(row.expert_support_roam_score, row.generated_mean, str(row.champion_name), fontsize=8)
    plt.xlabel("Expert support roam score")
    plt.ylabel("Generated mean support roam score")
    plt.title("Support score: generated vs expert reference")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_top_delta(df: pd.DataFrame, out_path: str) -> None:
    work = df.reindex(df["delta_mean_vs_expert"].abs().sort_values(ascending=False).index).head(20)
    plt.figure(figsize=(10, 6))
    colors = np.where(work["delta_mean_vs_expert"] >= 0, "#276fbf", "#c44536")
    plt.bar(work["champion_name"].astype(str), work["delta_mean_vs_expert"], color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Generated mean - expert score")
    plt.title("Largest champion-level deviations")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
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
    if not compared.empty:
        save_scatter(compared, os.path.join(args.outdir, "generated_vs_expert_scatter.png"))
        save_top_delta(compared, os.path.join(args.outdir, "largest_deviations.png"))

    print(f"Saved comparison: {os.path.abspath(comparison_path)}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
