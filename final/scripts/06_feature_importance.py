#!/usr/bin/env python3
"""
06_feature_importance.py — Permutation importance from the trained GBT model.

Uses sklearn.inspection.permutation_importance on the validation set to measure
the real impact of each feature on prediction quality.

See final/docs/technical_spec.md (Script 06) for the full specification.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VAL = str(REPO_ROOT / "final" / "data" / "training" / "val.parquet")
DEFAULT_MODEL_DIR = str(REPO_ROOT / "final" / "models" / "gbt")
DEFAULT_OUTDIR = str(REPO_ROOT / "final" / "analysis" / "feature_importance")

TARGET_COL = "support_roam_score"

ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")

# Feature group classification for aggregation
FEATURE_TYPE_MAP: Dict[str, str] = {}
for s in SIDES:
    for r in ROLE_KEYS:
        FEATURE_TYPE_MAP[f"{s}_{r}_champion_id"] = f"{s}_champions"
        for i in (1, 2):
            FEATURE_TYPE_MAP[f"{s}_{r}_summoner{i}_id"] = f"{s}_spells"
        FEATURE_TYPE_MAP[f"{s}_{r}_keystone_id"] = f"{s}_keystones"
        FEATURE_TYPE_MAP[f"{s}_{r}_primary_style_id"] = f"{s}_rune_styles"
        FEATURE_TYPE_MAP[f"{s}_{r}_sub_style_id"] = f"{s}_rune_styles"
    for i in range(1, 6):
        FEATURE_TYPE_MAP[f"{s}_ban_{i}_champion_id"] = f"{s}_bans"
FEATURE_TYPE_MAP["side"] = "context"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature importance via permutation.")
    p.add_argument("--val", default=DEFAULT_VAL)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--n-repeats", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-n", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    # Load model and preprocessor
    model = joblib.load(model_dir / "gbt_model_raw.joblib")
    preprocess = joblib.load(model_dir / "preprocess.joblib")
    encoder = preprocess["encoder"]
    feature_cols: List[str] = preprocess["feature_columns"]

    # Load val data and encode
    df_val = pd.read_parquet(args.val)
    X_val_raw = df_val[feature_cols].copy()
    for col in feature_cols:
        X_val_raw[col] = X_val_raw[col].fillna("__MISSING__").astype(str)
    X_val = encoder.transform(X_val_raw)
    y_val = df_val[TARGET_COL].to_numpy(dtype=np.float32)

    print(f"[Data] val={len(df_val):,}  features={len(feature_cols)}")
    print(f"[Permutation] n_repeats={args.n_repeats}...")

    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=args.n_repeats,
        random_state=args.seed,
        scoring="r2",
        n_jobs=-1,
    )

    # Per-feature table
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
        "feature_type": [FEATURE_TYPE_MAP.get(f, "other") for f in feature_cols],
    }).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(outdir / "permutation_importance_features.csv", index=False)

    # Group-level aggregation
    group_imp = (
        importance_df.groupby("feature_type")["importance_mean"]
        .agg(total_importance="sum", n_features="count", mean_importance="mean")
        .sort_values("total_importance", ascending=False)
        .reset_index()
    )
    group_imp.to_csv(outdir / "permutation_importance_groups.csv", index=False)

    # Print top features
    print(f"\n  Top-{args.top_n} features by permutation importance (R2 drop):")
    for _, row in importance_df.head(args.top_n).iterrows():
        print(f"    {row['feature']:45s}  {row['importance_mean']:+.6f}  "
              f"(+/- {row['importance_std']:.6f})  [{row['feature_type']}]")

    print(f"\n  Group-level importance:")
    for _, row in group_imp.iterrows():
        print(f"    {row['feature_type']:25s}  total={row['total_importance']:+.6f}  "
              f"n={int(row['n_features'])}")

    # Plot: top features
    top = importance_df.head(args.top_n)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {"ally_champions": "#2f80ed", "enemy_champions": "#eb5757",
              "ally_spells": "#6fcf97", "enemy_spells": "#f2994a",
              "context": "#9b51e0"}
    bar_colors = [colors.get(t, "#bbbbbb") for t in top["feature_type"]]
    ax.barh(range(len(top)), top["importance_mean"].values, xerr=top["importance_std"].values,
            color=bar_colors, alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance (R2 drop)")
    ax.set_title(f"Top-{args.top_n} Feature Importance (GBT)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "permutation_importance_top_features.png", dpi=180)
    plt.close(fig)

    # Plot: group-level
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(group_imp)), group_imp["total_importance"].values,
            color="#2f80ed", alpha=0.8)
    ax.set_yticks(range(len(group_imp)))
    ax.set_yticklabels(group_imp["feature_type"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Total Permutation Importance (R2 drop)")
    ax.set_title("Feature Group Importance (GBT)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "permutation_importance_groups.png", dpi=180)
    plt.close(fig)

    # Save metadata
    meta = {
        "model_path": str(model_dir / "gbt_model_raw.joblib"),
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "scoring": "r2",
        "n_features": len(feature_cols),
        "n_val": len(df_val),
    }
    (outdir / "feature_importance_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"\n[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
