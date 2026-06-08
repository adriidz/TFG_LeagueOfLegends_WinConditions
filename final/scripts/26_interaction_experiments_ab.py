#!/usr/bin/env python3
"""
26_interaction_experiments_ab.py -- Experiment script for:
  - Experiment A: Excluding raw botlane categoricals to reduce multicollinearity.
  - Experiment B: Sweeping smoothing parameters to find the optimal balance.

It runs a grid of configurations on the train/val splits and outputs a comparative table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import OrdinalEncoder

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = str(REPO_ROOT / "final" / "data" / "training" / "train.parquet")
DEFAULT_VAL = str(REPO_ROOT / "final" / "data" / "training" / "val.parquet")
DEFAULT_OUTDIR = str(REPO_ROOT / "final" / "models" / "experiments_ab")

TARGET_COL = "support_roam_score"
WEIGHT_COL = "sample_weight"
SUPPORT_COL = "ally_utility_champion_id"
ADC_COL = "ally_bottom_champion_id"

ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")
CHAMPION_COLS = [f"{side}_{role}_champion_id" for side in SIDES for role in ROLE_KEYS]

INTERACTION_SPECS: Dict[str, List[str]] = {
    "support_adc_synergy": [SUPPORT_COL, ADC_COL],
    "support_enemy_support_matchup": [SUPPORT_COL, "enemy_utility_champion_id"],
    "support_jungle_setup": [SUPPORT_COL, "ally_jungle_champion_id"],
    "support_mid_payoff": [SUPPORT_COL, "ally_middle_champion_id"],
    "support_adc_enemy_support": [
        SUPPORT_COL, ADC_COL, "enemy_utility_champion_id"
    ],
    "botlane_2v2_matchup": [
        SUPPORT_COL, ADC_COL, "enemy_utility_champion_id", "enemy_bottom_champion_id"
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Experiment A and B grid sweep on HistGBT.")
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--val", default=DEFAULT_VAL)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Limit train/val rows for a fast smoke test.",
    )
    return p.parse_args()


def make_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    parts = [df[col].fillna(-1).astype(int).astype(str) for col in cols]
    return pd.Series(["|".join(values) for values in zip(*parts)], index=df.index)


def fit_encoding_map(
    keys: pd.Series,
    y: np.ndarray,
    weights: np.ndarray,
    global_mean: float,
    smoothing: float,
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, float]]:
    tmp = pd.DataFrame(
        {
            "key": keys.to_numpy(),
            "target": np.asarray(y, dtype=np.float64),
            "weight": weights,
        }
    )
    tmp["weighted_target"] = tmp["target"] * tmp["weight"]
    grouped = tmp.groupby("key").agg(
        weighted_sum=("weighted_target", "sum"),
        weight_sum=("weight", "sum"),
        row_count=("target", "size"),
    )
    values = (grouped["weighted_sum"] + smoothing * global_mean) / (
        grouped["weight_sum"] + smoothing
    )
    return (
        values.astype(float).to_dict(),
        grouped["row_count"].astype(int).to_dict(),
        grouped["weight_sum"].astype(float).to_dict(),
    )


def apply_encoding(
    keys: pd.Series,
    values: Dict[str, float],
    counts: Dict[str, int],
    global_mean: float,
) -> Tuple[np.ndarray, np.ndarray]:
    encoded = keys.map(values).fillna(global_mean).to_numpy(dtype=np.float32)
    count_feature = np.log1p(keys.map(counts).fillna(0).to_numpy(dtype=np.float32))
    return encoded, count_feature


def make_oof_splits(df: pd.DataFrame, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if "match_id" in df.columns:
        n_splits = min(n_folds, df["match_id"].nunique())
        if n_splits >= 2:
            splitter = GroupKFold(n_splits=n_splits)
            return list(splitter.split(df, groups=df["match_id"]))
    splitter = KFold(n_splits=max(2, min(n_folds, len(df))), shuffle=True, random_state=seed)
    return list(splitter.split(df))


def build_interaction_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    y_train: np.ndarray,
    weights: np.ndarray,
    specs: Dict[str, List[str]],
    smoothing: float,
    n_folds: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    global_mean = float(np.sum(y_train * weights) / np.sum(weights))
    train_out = pd.DataFrame(index=df_train.index)
    val_out = pd.DataFrame(index=df_val.index)
    splits = make_oof_splits(df_train, n_folds, seed)

    for name, cols in specs.items():
        train_key = make_key(df_train, cols)
        val_key = make_key(df_val, cols)
        mean_col = f"te_{name}"
        count_col = f"te_{name}_log_count"

        oof_mean = np.full(len(df_train), global_mean, dtype=np.float32)
        oof_count = np.zeros(len(df_train), dtype=np.float32)
        for fit_idx, hold_idx in splits:
            fold_weights = weights[fit_idx]
            fold_mean = float(np.sum(y_train[fit_idx] * fold_weights) / np.sum(fold_weights))
            values, counts, _ = fit_encoding_map(
                train_key.iloc[fit_idx],
                y_train[fit_idx],
                fold_weights,
                fold_mean,
                smoothing,
            )
            enc, cnt = apply_encoding(train_key.iloc[hold_idx], values, counts, fold_mean)
            oof_mean[hold_idx] = enc
            oof_count[hold_idx] = cnt

        full_values, full_counts, _ = fit_encoding_map(
            train_key,
            y_train,
            weights,
            global_mean,
            smoothing,
        )
        val_mean, val_count = apply_encoding(val_key, full_values, full_counts, global_mean)

        train_out[mean_col] = oof_mean
        train_out[count_col] = oof_count
        val_out[mean_col] = val_mean
        val_out[count_col] = val_count

    return train_out, val_out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    pred_std = float(np.std(y_pred))
    target_std = float(np.std(y_true))
    
    if pred_std > 1e-12 and target_std > 1e-12:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        sp = spearmanr(y_true, y_pred, nan_policy="omit")
        spearman = float(sp.correlation) if sp.correlation is not None else float("nan")
    else:
        pearson = float("nan")
        spearman = float("nan")

    return {
        "r2": r2,
        "spearman": spearman,
        "mae": mae,
        "rmse": math.sqrt(mse),
        "pred_std": pred_std,
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[Data] Loading Parquet splits...")
    df_train = pd.read_parquet(args.train)
    df_val = pd.read_parquet(args.val)

    if args.limit_rows is not None:
        print(f"[Smoke Test] Limiting to first {args.limit_rows} rows.")
        df_train = df_train.head(args.limit_rows).copy()
        df_val = df_val.head(args.limit_rows).copy()

    # Pre-extract weights and targets
    y_train = df_train[TARGET_COL].to_numpy(dtype=np.float32)
    y_val = df_val[TARGET_COL].to_numpy(dtype=np.float32)
    
    if WEIGHT_COL in df_train.columns:
        w_train = df_train[WEIGHT_COL].to_numpy(dtype=np.float32)
    else:
        w_train = np.ones(len(df_train), dtype=np.float32)

    print(f"[Data] train={len(df_train):,}  val={len(df_val):,}")

    # Base features
    base_categorical_cols = [c for c in CHAMPION_COLS if c in df_train.columns] + ["side"]

    # Define experimental grid
    # Experimento A (Exclusión de variables crudas de botlane)
    # Experimento B (Sweeps de smoothing: 10.0, 25.0, 50.0)
    experiments = [
        {"name": "1. Baseline (No interactions)", "exclude_botlane": False, "use_interactions": False, "smoothing": 0.0},
        {"name": "2. Interactions (S=50, Keep Botlane)", "exclude_botlane": False, "use_interactions": True, "smoothing": 50.0},
        {"name": "3. Interactions (S=50, Exclude Botlane)", "exclude_botlane": True, "use_interactions": True, "smoothing": 50.0},
        {"name": "4. Interactions (S=25, Keep Botlane)", "exclude_botlane": False, "use_interactions": True, "smoothing": 25.0},
        {"name": "5. Interactions (S=25, Exclude Botlane)", "exclude_botlane": True, "use_interactions": True, "smoothing": 25.0},
        {"name": "6. Interactions (S=15, Keep Botlane)", "exclude_botlane": False, "use_interactions": True, "smoothing": 15.0},
        {"name": "7. Interactions (S=15, Exclude Botlane)", "exclude_botlane": True, "use_interactions": True, "smoothing": 15.0},
        {"name": "8. Interactions (S=5, Exclude Botlane)", "exclude_botlane": True, "use_interactions": True, "smoothing": 5.0},
    ]

    results_table = []

    for idx, exp in enumerate(experiments, start=1):
        print(f"\n--- Running Experiment {idx}/{len(experiments)}: {exp['name']} ---")
        
        # Prepare categorical features
        cat_cols = list(base_categorical_cols)
        if exp["exclude_botlane"]:
            cat_cols = [c for c in cat_cols if c not in [SUPPORT_COL, ADC_COL]]
        
        # Build interaction target encodings if needed
        if exp["use_interactions"]:
            print(f"  Building smoothed target encodings (smoothing={exp['smoothing']})...")
            num_train, num_val = build_interaction_features(
                df_train,
                df_val,
                y_train,
                w_train,
                specs=INTERACTION_SPECS,
                smoothing=exp["smoothing"],
                n_folds=args.n_folds,
                seed=args.seed,
            )
            num_cols = list(num_train.columns)
        else:
            num_train = pd.DataFrame(index=df_train.index)
            num_val = pd.DataFrame(index=df_val.index)
            num_cols = []

        # Prepare X matrices
        X_train_cat = df_train[cat_cols].fillna("__MISSING__").astype(str)
        X_val_cat = df_val[cat_cols].fillna("__MISSING__").astype(str)
        
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
        X_train_cat_arr = encoder.fit_transform(X_train_cat)
        X_val_cat_arr = encoder.transform(X_val_cat)

        if num_cols:
            X_train = np.hstack([X_train_cat_arr, num_train[num_cols].to_numpy(dtype=np.float32)])
            X_val = np.hstack([X_val_cat_arr, num_val[num_cols].to_numpy(dtype=np.float32)])
        else:
            X_train = X_train_cat_arr
            X_val = X_val_cat_arr

        categorical_mask = [True] * len(cat_cols) + [False] * len(num_cols)
        
        # Train HistGBT
        print(f"  Training HistGBT on {X_train.shape[0]:,} rows with {X_train.shape[1]} features...")
        t0 = time.time()
        model = HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=50,
            max_leaf_nodes=31,
            categorical_features=categorical_mask,
            random_state=args.seed,
        )
        model.fit(X_train, y_train, sample_weight=w_train)
        elapsed = time.time() - t0
        
        # Evaluate
        y_pred = model.predict(X_val)
        metrics = compute_metrics(y_val, y_pred)
        metrics["time"] = elapsed
        metrics["name"] = exp["name"]
        metrics["exclude_botlane"] = exp["exclude_botlane"]
        metrics["smoothing"] = exp["smoothing"]
        metrics["features_cat"] = len(cat_cols)
        metrics["features_num"] = len(num_cols)
        
        results_table.append(metrics)
        print(f"  Result: R2={metrics['r2']:.5f} | Spearman={metrics['spearman']:.5f} | MAE={metrics['mae']:.5f} | Pred_Std={metrics['pred_std']:.5f} | Time={elapsed:.1f}s")

    # Save output summary
    out_path = outdir / "experiments_ab_results.json"
    out_path.write_text(json.dumps(results_table, indent=2), encoding="utf-8")
    
    # Print Markdown Summary
    print("\n" + "="*80)
    print(" EXPERIMENTS SUMMARY (VAL SET EVALUATION)")
    print("="*80)
    print(f"| {'Experiment Configuration':<40} | {'R2':^7} | {'Spearman':^8} | {'MAE':^7} | {'Pred Std':^8} | {'Features (C/N)':^14} |")
    print(f"|{'-'*42}|{'-'*9}|{'-'*10}|{'-'*9}|{'-'*10}|{'-'*16}|")
    for r in results_table:
        feat_str = f"{r['features_cat']} cat / {r['features_num']} num"
        print(f"| {r['name']:<40} | {r['r2']:>7.5f} | {r['spearman']:>8.5f} | {r['mae']:>7.5f} | {r['pred_std']:>8.5f} | {feat_str:^14} |")
    print("="*80)
    print(f"Results saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
