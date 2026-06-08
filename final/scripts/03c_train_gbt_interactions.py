#!/usr/bin/env python3
"""
03c_train_gbt_interactions.py -- HistGBT with smoothed draft-interaction features.

This experiment is deliberately different from champion archetypes. Archetypes
are f(champion_id); interaction encodings estimate f(champion_a, champion_b)
from train data only:

  - support + ADC synergy
  - support vs enemy support matchup
  - support + jungler gank setup proxy
  - support + mid roam payoff proxy
  - botlane 2v2 matchup proxy

To avoid target leakage, train rows receive out-of-fold target encodings grouped
by match_id. Val/test rows are encoded with mappings fitted on the full train
split only. Script 07 evaluates the saved models on test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
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
DEFAULT_OUTDIR = str(REPO_ROOT / "final" / "models" / "gbt_interactions")

TARGET_COL = "support_roam_score"
QUANTILE_COL = "support_roam_score_quantile"
WEIGHT_COL = "sample_weight"

ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")
FEATURE_PROTOCOL_ID = "draft_10_champions_side_plus_smoothed_interactions"

BASE_FEATURE_GROUPS: Dict[str, List[str]] = {
    "champions": [f"{s}_{r}_champion_id" for s in SIDES for r in ROLE_KEYS],
    "summoner_spells": [
        f"{s}_{r}_summoner{i}_id" for s in SIDES for r in ROLE_KEYS for i in (1, 2)
    ],
    "context": ["side"],
}
FEATURE_SET_GROUPS: Dict[str, List[str]] = {
    "main": ["champions", "context"],
    "all": ["champions", "summoner_spells", "context"],
}

INTERACTION_SPECS: Dict[str, List[str]] = {
    "support_adc_synergy": ["ally_utility_champion_id", "ally_bottom_champion_id"],
    "support_enemy_support_matchup": ["ally_utility_champion_id", "enemy_utility_champion_id"],
    "support_jungle_setup": ["ally_utility_champion_id", "ally_jungle_champion_id"],
    "support_mid_payoff": ["ally_utility_champion_id", "ally_middle_champion_id"],
    "support_adc_enemy_support": [
        "ally_utility_champion_id", "ally_bottom_champion_id", "enemy_utility_champion_id",
    ],
    "botlane_2v2_matchup": [
        "ally_utility_champion_id", "ally_bottom_champion_id",
        "enemy_utility_champion_id", "enemy_bottom_champion_id",
    ],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HistGBT with OOF interaction target encodings.")
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--val", default=DEFAULT_VAL)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--smoothing", type=float, default=50.0)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--min-samples-leaf", type=int, default=50)
    p.add_argument("--max-leaf-nodes", type=int, default=31)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SET_GROUPS),
        default="main",
        help="main = 10 champion IDs + side; all = legacy champions + summoner spells + side.",
    )
    p.add_argument(
        "--allow-missing-sample-weight",
        action="store_true",
        help="Allow unweighted training if sample_weight is absent.",
    )
    return p.parse_args()


def available_base_features(df: pd.DataFrame, feature_set: str = "main") -> List[str]:
    cols: List[str] = []
    for group_name in FEATURE_SET_GROUPS[feature_set]:
        group_cols = BASE_FEATURE_GROUPS[group_name]
        cols.extend([c for c in group_cols if c in df.columns])
    return list(dict.fromkeys(cols))


def available_interactions(df: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        name: cols
        for name, cols in INTERACTION_SPECS.items()
        if all(c in df.columns for c in cols)
    }


def make_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    parts = []
    for col in cols:
        parts.append(df[col].fillna(-1).astype(int).astype(str))
    return pd.Series(["|".join(values) for values in zip(*parts)], index=df.index)


def fit_encoding_map(
    keys: pd.Series,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    global_mean: float,
    smoothing: float,
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, float]]:
    if sample_weight is None:
        weights = np.ones(len(y), dtype=np.float64)
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)

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
    values = (
        grouped["weighted_sum"] + smoothing * global_mean
    ) / (grouped["weight_sum"] + smoothing)
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


def dump_joblib_atomic(obj: Any, path: Path) -> None:
    tmp_dir = Path(tempfile.gettempdir())
    tmp_file = tempfile.NamedTemporaryFile(
        prefix=f"{path.stem}_",
        suffix=path.suffix,
        dir=tmp_dir,
        delete=False,
    )
    tmp_path = Path(tmp_file.name)
    tmp_file.close()
    try:
        joblib.dump(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def make_oof_splits(df: pd.DataFrame, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    n_splits = min(n_folds, df["match_id"].nunique() if "match_id" in df.columns else len(df))
    if "match_id" in df.columns and n_splits >= 2:
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(df, groups=df["match_id"]))
    splitter = KFold(n_splits=max(2, min(n_folds, len(df))), shuffle=True, random_state=seed)
    return list(splitter.split(df))


def build_interaction_features(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    y_train: np.ndarray,
    sample_weight: Optional[np.ndarray],
    specs: Dict[str, List[str]],
    smoothing: float,
    n_folds: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if sample_weight is None:
        global_mean = float(np.mean(y_train))
    else:
        weights = np.asarray(sample_weight, dtype=np.float64)
        global_mean = float(np.sum(np.asarray(y_train, dtype=np.float64) * weights) / np.sum(weights))
    train_out = pd.DataFrame(index=df_train.index)
    eval_out = pd.DataFrame(index=df_eval.index)
    mappings: Dict[str, Any] = {}
    splits = make_oof_splits(df_train, n_folds, seed)

    for name, cols in specs.items():
        train_key = make_key(df_train, cols)
        eval_key = make_key(df_eval, cols)
        mean_col = f"te_{name}"
        count_col = f"te_{name}_log_count"

        oof_mean = np.full(len(df_train), global_mean, dtype=np.float32)
        oof_count = np.zeros(len(df_train), dtype=np.float32)
        for fit_idx, hold_idx in splits:
            fold_weights = None if sample_weight is None else sample_weight[fit_idx]
            if fold_weights is None:
                fold_mean = float(np.mean(y_train[fit_idx]))
            else:
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

        full_values, full_counts, full_weight_sums = fit_encoding_map(
            train_key,
            y_train,
            sample_weight,
            global_mean,
            smoothing,
        )
        eval_mean, eval_count = apply_encoding(eval_key, full_values, full_counts, global_mean)

        train_out[mean_col] = oof_mean
        train_out[count_col] = oof_count
        eval_out[mean_col] = eval_mean
        eval_out[count_col] = eval_count
        mappings[name] = {
            "columns": cols,
            "mean_column": mean_col,
            "count_column": count_col,
            "global_mean": global_mean,
            "smoothing": smoothing,
            "values": full_values,
            "counts": full_counts,
            "weight_sums": full_weight_sums,
        }
        print(f"    [{name}] groups={len(full_values):,}  cols={cols}")

    return train_out, eval_out, mappings


def prepare_mixed_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    categorical_cols: List[str],
    numeric_train: pd.DataFrame,
    numeric_val: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, OrdinalEncoder, List[bool], List[str]]:
    X_train_cat = df_train[categorical_cols].copy()
    X_val_cat = df_val[categorical_cols].copy()
    for col in categorical_cols:
        X_train_cat[col] = X_train_cat[col].fillna("__MISSING__").astype(str)
        X_val_cat[col] = X_val_cat[col].fillna("__MISSING__").astype(str)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    X_train_cat_arr = encoder.fit_transform(X_train_cat)
    X_val_cat_arr = encoder.transform(X_val_cat)

    numeric_cols = list(numeric_train.columns)
    X_train = np.hstack([X_train_cat_arr, numeric_train[numeric_cols].to_numpy(dtype=np.float32)])
    X_val = np.hstack([X_val_cat_arr, numeric_val[numeric_cols].to_numpy(dtype=np.float32)])
    categorical_mask = [True] * len(categorical_cols) + [False] * len(numeric_cols)
    feature_cols = categorical_cols + numeric_cols
    return X_train, X_val, encoder, categorical_mask, feature_cols


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    label: str,
    n_train: int,
    elapsed: float,
) -> Dict[str, Any]:
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
        "model": model_name,
        "target": label,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "r2": r2,
        "pearson_corr": pearson,
        "spearman_corr": spearman,
        "pred_std": pred_std,
        "target_std": target_std,
        "compression_ratio": pred_std / target_std if target_std > 0 else float("nan"),
        "n_train": int(n_train),
        "n_eval": int(len(y_true)),
        "eval_split": "val",
        "training_seconds": float(elapsed),
    }


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    categorical_mask: List[bool],
    args: argparse.Namespace,
    target_label: str,
    outdir: Path,
    sample_weight: Optional[np.ndarray],
) -> Dict[str, Any]:
    weight_info = "with sample_weight" if sample_weight is not None else "no weights"
    print(f"\n  Training interaction GBT ({target_label}, {weight_info})...")
    model = HistGradientBoostingRegressor(
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_samples_leaf=args.min_samples_leaf,
        max_leaf_nodes=args.max_leaf_nodes,
        categorical_features=categorical_mask,
        random_state=args.seed,
        verbose=1,
    )
    t0 = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weight)
    elapsed = time.time() - t0

    y_pred = model.predict(X_val)
    metrics = compute_metrics(
        y_val,
        y_pred,
        f"gbt_interactions_{target_label}",
        target_label,
        n_train=len(y_train),
        elapsed=elapsed,
    )
    metrics["used_sample_weight"] = sample_weight is not None
    metrics["sample_weight_column"] = WEIGHT_COL if sample_weight is not None else None
    dump_joblib_atomic(model, outdir / f"gbt_model_{target_label}.joblib")
    print(
        f"  R2={metrics['r2']:.4f}  Spearman={metrics['spearman_corr']:.4f}  "
        f"pred_std={metrics['pred_std']:.4f}  time={elapsed:.1f}s"
    )
    return metrics


def run_target(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    categorical_cols: List[str],
    specs: Dict[str, List[str]],
    sample_weight: Optional[np.ndarray],
    target_col: str,
    target_label: str,
    args: argparse.Namespace,
    outdir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], List[bool], OrdinalEncoder]:
    print(f"\n[Encoding] target={target_label}")
    y_train = df_train[target_col].to_numpy(dtype=np.float32)
    y_val = df_val[target_col].to_numpy(dtype=np.float32)
    numeric_train, numeric_val, mappings = build_interaction_features(
        df_train,
        df_val,
        y_train,
        sample_weight,
        specs,
        smoothing=args.smoothing,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    X_train, X_val, encoder, cat_mask, feature_cols = prepare_mixed_features(
        df_train,
        df_val,
        categorical_cols,
        numeric_train,
        numeric_val,
    )
    print(f"[Features {target_label}] shape={X_train.shape}  numeric={numeric_train.shape[1]}")
    metrics = train_and_evaluate(
        X_train,
        y_train,
        X_val,
        y_val,
        cat_mask,
        args,
        target_label,
        outdir,
        sample_weight,
    )
    preprocess = {
        "encoder": encoder,
        "categorical_columns": categorical_cols,
        "numeric_columns": list(numeric_train.columns),
        "feature_columns": feature_cols,
        "categorical_mask": cat_mask,
        "interaction_specs": specs,
        "interaction_mappings": mappings,
        "feature_protocol_id": FEATURE_PROTOCOL_ID,
        "sample_weight_column": WEIGHT_COL,
        "used_sample_weight": sample_weight is not None,
    }
    return metrics, preprocess, feature_cols, cat_mask, encoder


def sample_weight_from_train(
    df_train: pd.DataFrame,
    allow_missing: bool,
) -> Optional[np.ndarray]:
    if WEIGHT_COL not in df_train.columns:
        if allow_missing:
            print("[Weights] No sample_weight column found - training without weights")
            return None
        raise SystemExit(
            "[Weights] Missing required sample_weight column. "
            "Use --allow-missing-sample-weight only for legacy/debug runs."
        )

    sample_weight = df_train[WEIGHT_COL].to_numpy(dtype=np.float32)
    print(
        f"[Weights] Using sample_weight: mean={sample_weight.mean():.3f}  "
        f"min={sample_weight.min():.3f}  max={sample_weight.max():.3f}"
    )
    return sample_weight


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(args.train)
    df_val = pd.read_parquet(args.val)
    categorical_cols = available_base_features(df_train, args.feature_set)
    specs = available_interactions(df_train)
    sample_weight = sample_weight_from_train(df_train, args.allow_missing_sample_weight)
    print(
        f"[Data] train={len(df_train):,}  val={len(df_val):,}  "
        f"feature_set={args.feature_set}  categorical={len(categorical_cols)}  "
        f"interactions={len(specs)}"
    )
    print("[Categorical] " + ", ".join(categorical_cols))

    results: List[Dict[str, Any]] = []
    preprocess_by_target: Dict[str, Any] = {}

    metrics_raw, preprocess_raw, _, _, _ = run_target(
        df_train, df_val, categorical_cols, specs, sample_weight, TARGET_COL, "raw", args, outdir
    )
    results.append(metrics_raw)
    preprocess_by_target["raw"] = preprocess_raw

    if QUANTILE_COL in df_train.columns and QUANTILE_COL in df_val.columns:
        metrics_q, preprocess_q, _, _, _ = run_target(
            df_train,
            df_val,
            categorical_cols,
            specs,
            sample_weight,
            QUANTILE_COL,
            "quantile",
            args,
            outdir,
        )
        results.append(metrics_q)
        preprocess_by_target["quantile"] = preprocess_q

    config = {
        "model_type": "hist_gbt_smoothed_interactions",
        "feature_set": args.feature_set,
        "feature_protocol_id": FEATURE_PROTOCOL_ID,
        "input_feature_columns": categorical_cols,
        "feature_columns": (
            preprocess_by_target["raw"]["feature_columns"]
            if "raw" in preprocess_by_target
            else categorical_cols
        ),
        "included_feature_groups": FEATURE_SET_GROUPS[args.feature_set],
        "excluded_feature_groups": [
            name for name in BASE_FEATURE_GROUPS if name not in FEATURE_SET_GROUPS[args.feature_set]
        ],
        "categorical_columns": categorical_cols,
        "interaction_specs": specs,
        "n_folds": args.n_folds,
        "smoothing": args.smoothing,
        "sample_weight_column": WEIGHT_COL,
        "used_sample_weight": sample_weight is not None,
        "max_iter": args.max_iter,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "min_samples_leaf": args.min_samples_leaf,
        "max_leaf_nodes": args.max_leaf_nodes,
        "seed": args.seed,
    }
    dump_joblib_atomic(preprocess_by_target, outdir / "preprocess.joblib")
    (outdir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (outdir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
