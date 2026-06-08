#!/usr/bin/env python3
"""
02_baseline_champion_mean.py — Trivial baseline: predict mean score by champion.

This is the most important baseline: if the MLP barely beats a lookup table
by champion, it isn't learning meaningful draft interactions.

See final/docs/technical_spec.md (Script 02) for the full specification.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = str(REPO_ROOT / "final" / "data" / "training" / "train.parquet")
DEFAULT_VAL = str(REPO_ROOT / "final" / "data" / "training" / "val.parquet")
DEFAULT_OUTDIR = str(REPO_ROOT / "final" / "baselines")

TARGET_COL = "support_roam_score"
QUANTILE_COL = "support_roam_score_quantile"
CHAMPION_COL = "ally_utility_champion_id"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Champion-mean baseline.")
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--val", default=DEFAULT_VAL)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    return p.parse_args()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> Dict:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else float("nan")
    sp = spearmanr(y_true, y_pred, nan_policy="omit")
    return {
        "model": f"champion_mean_{label}",
        "target": label,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "r2": r2,
        "pearson_corr": pearson,
        "spearman_corr": float(sp.correlation) if sp.correlation is not None else float("nan"),
        "pred_std": float(np.std(y_pred)),
        "target_std": float(np.std(y_true)),
        "compression_ratio": float(np.std(y_pred) / np.std(y_true)) if np.std(y_true) > 0 else float("nan"),
        "n_train": None,  # filled later
        "n_eval": int(len(y_true)),
        "eval_split": "val",
    }


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w_sum = float(weights.sum())
    if w_sum <= 1e-8:
        return float(values.mean())
    return float((values * weights).sum() / w_sum)


def run_baseline(df_train: pd.DataFrame, df_val: pd.DataFrame,
                 target_col: str, label: str) -> tuple[Dict, pd.DataFrame]:
    """Compute sample-weighted mean per champion on train, predict on val."""
    if "sample_weight" not in df_train.columns:
        raise SystemExit("[Weights] Missing required sample_weight column for champion mean baseline.")

    weights = df_train["sample_weight"].astype(np.float64)
    means = df_train.groupby(CHAMPION_COL).apply(
        lambda g: weighted_mean(g[target_col].astype(np.float64), g["sample_weight"].astype(np.float64))
    )
    global_mean = weighted_mean(df_train[target_col].astype(np.float64), weights)

    y_pred = df_val[CHAMPION_COL].map(means).fillna(global_mean).to_numpy()
    y_true = df_val[target_col].to_numpy()

    metrics = compute_metrics(y_true, y_pred, label)
    metrics["n_train"] = int(len(df_train))
    metrics["n_champions_in_train"] = int(len(means))
    metrics["global_mean"] = global_mean
    metrics["sample_weight_column"] = "sample_weight"
    metrics["used_sample_weight"] = True
    metrics["n_unseen_champions_in_val"] = int((~df_val[CHAMPION_COL].isin(means.index)).sum())

    # Per-champion table
    champ_table = means.reset_index()
    champ_table.columns = [CHAMPION_COL, f"mean_{target_col}"]
    champ_table = champ_table.sort_values(f"mean_{target_col}", ascending=False)

    return metrics, champ_table


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(args.train)
    df_val = pd.read_parquet(args.val)
    print(f"[Data] train={len(df_train):,}  val={len(df_val):,}")

    results = []

    # Raw target
    m_raw, champ_raw = run_baseline(df_train, df_val, TARGET_COL, "raw")
    results.append(m_raw)
    print(f"\n[Raw]     R2={m_raw['r2']:.4f}  Spearman={m_raw['spearman_corr']:.4f}  "
          f"pred_std={m_raw['pred_std']:.4f}")

    # Quantile target
    if QUANTILE_COL in df_train.columns:
        m_q, champ_q = run_baseline(df_train, df_val, QUANTILE_COL, "quantile")
        results.append(m_q)
        print(f"[Quantile] R2={m_q['r2']:.4f}  Spearman={m_q['spearman_corr']:.4f}  "
              f"pred_std={m_q['pred_std']:.4f}")

    # Also compute a "predict global mean" baseline for reference
    for target_col, label in [(TARGET_COL, "raw"), (QUANTILE_COL, "quantile")]:
        if target_col not in df_val.columns:
            continue
        y_true = df_val[target_col].to_numpy()
        if "sample_weight" not in df_train.columns:
            raise SystemExit("[Weights] Missing required sample_weight column for global mean baseline.")
        global_mean = weighted_mean(
            df_train[target_col].astype(np.float64),
            df_train["sample_weight"].astype(np.float64),
        )
        y_pred = np.full_like(y_true, global_mean)
        m = compute_metrics(y_true, y_pred, label)
        m["model"] = f"global_mean_{label}"
        m["n_train"] = int(len(df_train))
        m["global_mean"] = global_mean
        m["sample_weight_column"] = "sample_weight"
        m["used_sample_weight"] = True
        results.append(m)
        print(f"[Global mean {label}] R2={m['r2']:.4f}  (this should be ~0 by definition)")

    # Save
    metrics_path = outdir / "champion_mean_metrics.json"
    metrics_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    champ_raw.to_csv(outdir / "champion_mean_table_raw.csv", index=False)
    print(f"\n[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
