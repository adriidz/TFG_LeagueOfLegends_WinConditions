#!/usr/bin/env python3
"""
Check outputs from new_02a_build_labels_and_draft_features_supportv2.py and
plot score distributions, with special focus on support v1 vs v2.

Main features
-------------
- Locates parquet outputs under data_new/clean/scores, labels and features.
- Prints basic validation info: existence, row counts, duplicate keys, missing scores.
- Summarizes numeric score columns.
- For support outputs, compares v1 (support_roam_score) vs v2 (support_roam_score_v2).
- Saves plots and CSV summaries to an analysis directory.

Example usage
-------------
python check_new_02a_outputs_and_plot.py --max-minute 8
python check_new_02a_outputs_and_plot.py --scores-dir data_new/clean/scores --labels-dir data_new/clean/labels --features-dir data_new/clean/features
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SCORES_DIR = os.path.join("data_new", "clean", "scores")
DEFAULT_LABELS_DIR = os.path.join("data_new", "clean", "labels")
DEFAULT_FEATURES_DIR = os.path.join("data_new", "clean", "features")
DEFAULT_OUTDIR = os.path.join("data_new", "analysis", "check_new_02a")

JOIN_KEYS = ["match_id", "team_id"]


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check new_02a outputs and plot distributions.")
    p.add_argument("--scores-dir", default=DEFAULT_SCORES_DIR, help="Directory with score parquet files.")
    p.add_argument("--labels-dir", default=DEFAULT_LABELS_DIR, help="Directory with label parquet files.")
    p.add_argument("--features-dir", default=DEFAULT_FEATURES_DIR, help="Directory with features parquet files.")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Directory to store analysis outputs.")
    p.add_argument("--max-minute", type=float, default=None,
                   help="If set, prefer files suffixed with _mXX (e.g. 8 -> _m08).")
    p.add_argument("--bins", type=int, default=40, help="Histogram bins.")
    p.add_argument("--sample", type=int, default=None,
                   help="Optional row sample size for very large parquets.")
    return p.parse_args()


# -----------------------------
# File resolution helpers
# -----------------------------
def minute_suffix(max_minute: Optional[float]) -> str:
    if max_minute is None:
        return ""
    mm = int(round(max_minute))
    return f"_m{mm:02d}"


def resolve_parquet(base_dir: str, stem: str, max_minute: Optional[float]) -> Optional[str]:
    suff = minute_suffix(max_minute)

    # 1) exacto sin sample
    preferred = os.path.join(base_dir, f"{stem}{suff}.parquet")
    if os.path.exists(preferred):
        return preferred

    # 2) con cualquier sufijo intermedio, pero respetando la ventana
    if suff:
        window_matches = sorted(glob.glob(os.path.join(base_dir, f"{stem}*{suff}*.parquet")))
        if window_matches:
            return window_matches[0]

    # 3) sin ventana, exacto
    unsuffixed = os.path.join(base_dir, f"{stem}.parquet")
    if os.path.exists(unsuffixed):
        return unsuffixed

    # 4) fallback general
    wildcard = sorted(glob.glob(os.path.join(base_dir, f"{stem}*.parquet")))
    if wildcard:
        return wildcard[0]

    return None


# -----------------------------
# IO / validation helpers
# -----------------------------
def load_parquet(path: Optional[str], sample: Optional[int] = None) -> Optional[pd.DataFrame]:
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if sample is not None and len(df) > sample:
        df = df.sample(sample, random_state=42)
    return df


def duplicate_count(df: pd.DataFrame, keys: Sequence[str] = JOIN_KEYS) -> int:
    existing = [k for k in keys if k in df.columns]
    if len(existing) != len(keys):
        return -1
    return int(df.duplicated(existing).sum())


def numeric_summary(series: pd.Series, name: str) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    out = {"metric": name, "n": int(s.size)}
    if s.empty:
        out.update({"mean": np.nan, "std": np.nan, "min": np.nan, "p25": np.nan,
                    "median": np.nan, "p75": np.nan, "max": np.nan})
        return out
    out.update({
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
    })
    return out


# -----------------------------
# Plot helpers
# -----------------------------
def save_hist(series: pd.Series, title: str, xlabel: str, out_path: str, bins: int = 40) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(s, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



def save_overlay_hist(s1: pd.Series, s2: pd.Series, labels: tuple[str, str], title: str,
                      xlabel: str, out_path: str, bins: int = 40) -> None:
    a = pd.to_numeric(s1, errors="coerce").dropna()
    b = pd.to_numeric(s2, errors="coerce").dropna()
    if a.empty and b.empty:
        return
    plt.figure(figsize=(8, 5))
    if not a.empty:
        plt.hist(a, bins=bins, alpha=0.55, label=labels[0], density=False)
    if not b.empty:
        plt.hist(b, bins=bins, alpha=0.55, label=labels[1], density=False)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



def save_scatter(x: pd.Series, y: pd.Series, xlabel: str, ylabel: str, title: str, out_path: str) -> None:
    df = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if df.empty:
        return
    plt.figure(figsize=(6, 6))
    plt.scatter(df["x"], df["y"], alpha=0.4, s=10)
    mn = float(min(df["x"].min(), df["y"].min()))
    mx = float(max(df["x"].max(), df["y"].max()))
    plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------
# Main analysis
# -----------------------------
def check_dataset(name: str, path: Optional[str], df: Optional[pd.DataFrame]) -> dict:
    row = {"dataset": name, "path": path, "exists": path is not None and os.path.exists(path)}
    if df is None:
        row.update({"rows": 0, "cols": 0, "duplicate_keys": np.nan})
        return row
    row.update({
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "duplicate_keys": duplicate_count(df),
    })
    return row


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve expected files
    paths = {
        "jungle_scores": resolve_parquet(args.scores_dir, "jungle_scores", args.max_minute),
        "support_scores": resolve_parquet(args.scores_dir, "support_scores", args.max_minute),
        "team_scores": resolve_parquet(args.scores_dir, "team_tendency_scores", args.max_minute),
        "jungle_labels": resolve_parquet(args.labels_dir, "jungle_labels", args.max_minute),
        "support_labels": resolve_parquet(args.labels_dir, "support_labels", args.max_minute),
        "team_labels": resolve_parquet(args.labels_dir, "team_tendency_labels", args.max_minute),
        "draft_features": resolve_parquet(args.features_dir, "draft_features", args.max_minute),
    }

    dfs = {name: load_parquet(path, sample=args.sample) for name, path in paths.items()}

    print("=" * 78)
    print("CHECK new_02a OUTPUTS")
    print("=" * 78)
    print(f"scores-dir:   {os.path.abspath(args.scores_dir)}")
    print(f"labels-dir:   {os.path.abspath(args.labels_dir)}")
    print(f"features-dir: {os.path.abspath(args.features_dir)}")
    print(f"outdir:       {os.path.abspath(str(outdir))}")
    if args.max_minute is not None:
        print(f"preferred window suffix: {minute_suffix(args.max_minute)}")
    print()

    # Dataset-level report
    dataset_rows = []
    for name in ["jungle_scores", "support_scores", "team_scores", "jungle_labels", "support_labels", "team_labels", "draft_features"]:
        row = check_dataset(name, paths[name], dfs[name])
        dataset_rows.append(row)
    dataset_df = pd.DataFrame(dataset_rows)
    print(dataset_df.to_string(index=False))
    dataset_df.to_csv(outdir / "dataset_check.csv", index=False)

    # Numeric summaries for score tables
    summary_rows = []
    score_columns_map = {
        "jungle_scores": ["jungle_presence_score", "jungle_presence_score_v2"],
        "support_scores": ["support_roam_score", "support_roam_score_v2", "mean_distance_to_adc_v2",
                           "support_adc_xp_ratio_v2", "support_score_confidence_v2"],
        "team_scores": ["team_side_focus_score", "team_side_focus_score_v2"],
    }

    for dataset_name, cols in score_columns_map.items():
        df = dfs.get(dataset_name)
        if df is None:
            continue
        for col in cols:
            if col in df.columns:
                row = numeric_summary(df[col], col)
                row["dataset"] = dataset_name
                summary_rows.append(row)
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print("\n[Numeric summary]")
        print(summary_df.to_string(index=False))
        summary_df.to_csv(outdir / "numeric_summary.csv", index=False)

    # Support-specific analysis (the most important one now)
    sp = dfs.get("support_scores")
    if sp is not None:
        support_out = outdir / "support"
        support_out.mkdir(parents=True, exist_ok=True)

        print("\n" + "-" * 78)
        print("SUPPORT OUTPUT CHECK")
        print("-" * 78)
        print(f"support_scores path: {paths['support_scores']}")
        print(f"rows: {len(sp)} | cols: {sp.shape[1]}")
        if all(k in sp.columns for k in JOIN_KEYS):
            print(f"duplicate (match_id, team_id): {duplicate_count(sp)}")

        for c in ["support_roam_score", "support_roam_score_v2", "support_champion_name",
                  "support_score_confidence_v2", "support_adc_xp_ratio_v2"]:
            print(f"column '{c}': {'YES' if c in sp.columns else 'NO'}")

        # Missingness report
        miss_rows = []
        for c in ["support_roam_score", "support_roam_score_v2", "mean_distance_to_adc_v2",
                  "support_adc_xp_ratio_v2", "support_score_confidence_v2"]:
            if c in sp.columns:
                miss_rows.append({
                    "column": c,
                    "missing": int(sp[c].isna().sum()),
                    "missing_pct": float(sp[c].isna().mean()),
                })
        if miss_rows:
            miss_df = pd.DataFrame(miss_rows)
            print("\n[Support missingness]")
            print(miss_df.to_string(index=False))
            miss_df.to_csv(support_out / "support_missingness.csv", index=False)

        # Plots: v1, v2, overlay
        if "support_roam_score" in sp.columns:
            save_hist(sp["support_roam_score"],
                      "Support score v1 distribution",
                      "support_roam_score",
                      str(support_out / "support_roam_score_v1_hist.png"),
                      bins=args.bins)

        if "support_roam_score_v2" in sp.columns:
            save_hist(sp["support_roam_score_v2"],
                      "Support score v2 distribution",
                      "support_roam_score_v2",
                      str(support_out / "support_roam_score_v2_hist.png"),
                      bins=args.bins)

        if "support_roam_score" in sp.columns and "support_roam_score_v2" in sp.columns:
            save_overlay_hist(sp["support_roam_score"], sp["support_roam_score_v2"],
                              ("v1", "v2"),
                              "Support score distribution: v1 vs v2",
                              "score",
                              str(support_out / "support_v1_v2_overlay_hist.png"),
                              bins=args.bins)
            save_scatter(sp["support_roam_score"], sp["support_roam_score_v2"],
                         "support_roam_score (v1)", "support_roam_score_v2",
                         "Support v1 vs v2 scatter",
                         str(support_out / "support_v1_v2_scatter.png"))

            pair = sp[["support_roam_score", "support_roam_score_v2"]].apply(pd.to_numeric, errors="coerce").dropna()
            if not pair.empty:
                corr = pair.corr(numeric_only=True).iloc[0, 1]
                with open(support_out / "support_v1_v2_correlation.txt", "w", encoding="utf-8") as f:
                    f.write(f"pearson_corr_v1_v2={corr:.6f}\n")
                print(f"\nSupport v1-v2 Pearson corr: {corr:.6f}")

        # Champion breakdown for v2
        if "support_champion_name" in sp.columns and "support_roam_score_v2" in sp.columns:
            champ_df = (
                sp.groupby("support_champion_name", dropna=False)["support_roam_score_v2"]
                .agg(["count", "mean", "median", "std"])
                .reset_index()
                .sort_values(["count", "mean"], ascending=[False, False])
            )
            champ_df.to_csv(support_out / "support_v2_by_champion.csv", index=False)
            print("\n[Top champions by count - support v2]")
            print(champ_df.head(15).to_string(index=False))

            # Plot top-N champions by count
            top_n = champ_df.head(15).copy()
            if not top_n.empty:
                plt.figure(figsize=(10, 6))
                plt.bar(top_n["support_champion_name"].astype(str), top_n["mean"])
                plt.xticks(rotation=45, ha="right")
                plt.title("Mean support_roam_score_v2 by champion (top 15 by count)")
                plt.xlabel("Champion")
                plt.ylabel("Mean support_roam_score_v2")
                plt.tight_layout()
                plt.savefig(support_out / "support_v2_top15_champions.png", dpi=150)
                plt.close()

    print("\nDone. Analysis outputs saved to:")
    print(os.path.abspath(str(outdir)))


if __name__ == "__main__":
    main()
