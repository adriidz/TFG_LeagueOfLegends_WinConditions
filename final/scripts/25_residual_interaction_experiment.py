#!/usr/bin/env python3
"""
25_residual_interaction_experiment.py -- Support baseline + contextual residual model.

This secondary experiment separates two questions:

  1. What is the average roaming tendency of the allied support champion?
  2. Which draft interactions move that match above or below that support mean?

The final prediction is:

    support_mean(ally_support) + residual_gbt(context, smoothed pair interactions)

The residual model does not receive ally_utility_champion_id as a direct
categorical feature. It can only use it through explicit, smoothed interaction
features such as support+ADC and support-vs-enemy-support. Train encodings are
out-of-fold to avoid target leakage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
DEFAULT_OUTDIR = str(REPO_ROOT / "final" / "models" / "residual_interactions")
DEFAULT_CHAMPION_CLASSES = str(REPO_ROOT / "final" / "data" / "champion_classes.json")

TARGET_COL = "support_roam_score"
WEIGHT_COL = "sample_weight"
SUPPORT_COL = "ally_utility_champion_id"
ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")
CHAMPION_COLS = [f"{side}_{role}_champion_id" for side in SIDES for role in ROLE_KEYS]
CANONICAL_FEATURE_COLUMNS = CHAMPION_COLS + ["side"]
RESIDUAL_FEATURE_PROTOCOL_ID = "support_mean_plus_contextual_residual_interactions"

CONTEXT_CATEGORICAL_COLUMNS = [
    col for col in CANONICAL_FEATURE_COLUMNS if col != SUPPORT_COL
]

INTERACTION_SPECS: Dict[str, List[str]] = {
    "support_adc_synergy": [SUPPORT_COL, "ally_bottom_champion_id"],
    "support_enemy_support_matchup": [SUPPORT_COL, "enemy_utility_champion_id"],
    "support_jungle_setup": [SUPPORT_COL, "ally_jungle_champion_id"],
    "support_mid_payoff": [SUPPORT_COL, "ally_middle_champion_id"],
    "support_adc_enemy_support": [
        SUPPORT_COL,
        "ally_bottom_champion_id",
        "enemy_utility_champion_id",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a support-mean + residual interaction HistGBT experiment."
    )
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--val", default=DEFAULT_VAL)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--champion-classes", default=DEFAULT_CHAMPION_CLASSES)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--support-smoothing", type=float, default=20.0)
    parser.add_argument("--interaction-smoothing", type=float, default=50.0)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-samples-leaf", type=int, default=50)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit-train-rows",
        type=int,
        default=None,
        help="Optional smoke/debug limit. Do not use for reported metrics.",
    )
    parser.add_argument(
        "--limit-val-rows",
        type=int,
        default=None,
        help="Optional smoke/debug limit. Do not use for reported metrics.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: List[str], split_name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise SystemExit(f"[{split_name}] Missing required columns: {missing}")


def load_champion_names(path: Path) -> Dict[int, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    names: Dict[int, str] = {}
    if not isinstance(raw, dict):
        return names
    for key, value in raw.items():
        try:
            champion_id = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict) and value.get("name"):
            names[champion_id] = str(value["name"])
    return names


def format_interaction_key(key: str, cols: List[str], champion_names: Dict[int, str]) -> str:
    labels: List[str] = []
    for col, value in zip(cols, str(key).split("|")):
        try:
            champion_id = int(value)
        except (TypeError, ValueError):
            name = str(value)
        else:
            name = champion_names.get(champion_id, str(champion_id))
        role = col.replace("_champion_id", "")
        labels.append(f"{role}={name}")
    return " | ".join(labels)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = values.to_numpy(dtype=np.float64)
    w = weights.to_numpy(dtype=np.float64)
    w_sum = float(w.sum())
    if w_sum <= 1e-8:
        return float(np.mean(v))
    return float(np.sum(v * w) / w_sum)


def safe_rank_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    corr = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return float(corr) if corr is not None else float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    n_train: int,
    elapsed: float,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    pred_std = float(np.std(y_pred))
    target_std = float(np.std(y_true))
    pearson = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if pred_std > 1e-12 and target_std > 1e-12
        else float("nan")
    )
    return {
        "model": model_name,
        "target": TARGET_COL,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "pearson_corr": pearson,
        "spearman_corr": safe_rank_corr(y_true, y_pred),
        "pred_std": pred_std,
        "target_std": target_std,
        "compression_ratio": pred_std / target_std if target_std > 0 else float("nan"),
        "n_train": int(n_train),
        "n_eval": int(len(y_true)),
        "eval_split": "val",
        "training_seconds": float(elapsed),
    }


def make_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    parts = [df[col].fillna(-1).astype(int).astype(str) for col in cols]
    return pd.Series(["|".join(values) for values in zip(*parts)], index=df.index)


def fit_smoothed_mean_map(
    keys: pd.Series,
    values: np.ndarray,
    weights: np.ndarray,
    prior: float,
    smoothing: float,
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, float]]:
    tmp = pd.DataFrame(
        {
            "key": keys.to_numpy(),
            "value": np.asarray(values, dtype=np.float64),
            "weight": np.asarray(weights, dtype=np.float64),
        }
    )
    tmp["weighted_value"] = tmp["value"] * tmp["weight"]
    grouped = tmp.groupby("key").agg(
        weighted_sum=("weighted_value", "sum"),
        weight_sum=("weight", "sum"),
        row_count=("value", "size"),
    )
    means = (grouped["weighted_sum"] + smoothing * prior) / (
        grouped["weight_sum"] + smoothing
    )
    return (
        means.astype(float).to_dict(),
        grouped["row_count"].astype(int).to_dict(),
        grouped["weight_sum"].astype(float).to_dict(),
    )


def apply_smoothed_mean_map(
    keys: pd.Series,
    means: Dict[str, float],
    counts: Dict[str, int],
    prior: float,
) -> Tuple[np.ndarray, np.ndarray]:
    encoded = keys.map(means).fillna(prior).to_numpy(dtype=np.float32)
    log_count = np.log1p(keys.map(counts).fillna(0).to_numpy(dtype=np.float32))
    return encoded, log_count


def make_oof_splits(df: pd.DataFrame, n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if len(df) < 2:
        raise ValueError("At least two rows are required for OOF encodings.")
    if "match_id" in df.columns:
        n_groups = int(df["match_id"].nunique())
        if n_groups >= 2:
            splitter = GroupKFold(n_splits=max(2, min(n_folds, n_groups)))
            return list(splitter.split(df, groups=df["match_id"]))
    splitter = KFold(n_splits=max(2, min(n_folds, len(df))), shuffle=True, random_state=seed)
    return list(splitter.split(df))


def build_support_baseline(
    df_train: pd.DataFrame,
    target_col: str,
    support_smoothing: float,
) -> Dict[str, Any]:
    y = df_train[target_col].to_numpy(dtype=np.float64)
    w = df_train[WEIGHT_COL].to_numpy(dtype=np.float64)
    global_mean = float(np.sum(y * w) / np.sum(w))
    keys = df_train[SUPPORT_COL].fillna(-1).astype(int).astype(str)
    means, counts, weight_sums = fit_smoothed_mean_map(
        keys,
        y,
        w,
        prior=global_mean,
        smoothing=support_smoothing,
    )
    return {
        "global_mean": global_mean,
        "support_means": means,
        "support_counts": counts,
        "support_weight_sums": weight_sums,
        "support_smoothing": support_smoothing,
    }


def predict_support_baseline(df: pd.DataFrame, baseline: Dict[str, Any]) -> np.ndarray:
    keys = df[SUPPORT_COL].fillna(-1).astype(int).astype(str)
    values, _ = apply_smoothed_mean_map(
        keys,
        baseline["support_means"],
        baseline["support_counts"],
        baseline["global_mean"],
    )
    return values.astype(np.float32)


def build_oof_support_baseline(
    df_train: pd.DataFrame,
    target_col: str,
    n_folds: int,
    support_smoothing: float,
    seed: int,
) -> np.ndarray:
    y = df_train[target_col].to_numpy(dtype=np.float64)
    w = df_train[WEIGHT_COL].to_numpy(dtype=np.float64)
    global_mean = float(np.sum(y * w) / np.sum(w))
    keys = df_train[SUPPORT_COL].fillna(-1).astype(int).astype(str)
    oof = np.full(len(df_train), global_mean, dtype=np.float32)
    for fit_idx, hold_idx in make_oof_splits(df_train, n_folds, seed):
        fold_prior = weighted_mean(
            df_train.iloc[fit_idx][target_col],
            df_train.iloc[fit_idx][WEIGHT_COL],
        )
        means, counts, _ = fit_smoothed_mean_map(
            keys.iloc[fit_idx],
            y[fit_idx],
            w[fit_idx],
            prior=fold_prior,
            smoothing=support_smoothing,
        )
        encoded, _ = apply_smoothed_mean_map(keys.iloc[hold_idx], means, counts, fold_prior)
        oof[hold_idx] = encoded
    return oof


def available_interactions(df: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        name: cols
        for name, cols in INTERACTION_SPECS.items()
        if all(col in df.columns for col in cols)
    }


def build_residual_interaction_features(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    residual_train: np.ndarray,
    specs: Dict[str, List[str]],
    smoothing: float,
    n_folds: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    weights = df_train[WEIGHT_COL].to_numpy(dtype=np.float64)
    prior = float(np.sum(residual_train * weights) / np.sum(weights))
    train_out = pd.DataFrame(index=df_train.index)
    eval_out = pd.DataFrame(index=df_eval.index)
    mappings: Dict[str, Any] = {}
    splits = make_oof_splits(df_train, n_folds, seed)

    for name, cols in specs.items():
        train_key = make_key(df_train, cols)
        eval_key = make_key(df_eval, cols)
        mean_col = f"resid_te_{name}"
        count_col = f"resid_te_{name}_log_count"

        oof_mean = np.full(len(df_train), prior, dtype=np.float32)
        oof_count = np.zeros(len(df_train), dtype=np.float32)
        for fit_idx, hold_idx in splits:
            fold_weights = weights[fit_idx]
            fold_prior = float(np.sum(residual_train[fit_idx] * fold_weights) / np.sum(fold_weights))
            means, counts, weight_sums = fit_smoothed_mean_map(
                train_key.iloc[fit_idx],
                residual_train[fit_idx],
                fold_weights,
                prior=fold_prior,
                smoothing=smoothing,
            )
            enc, cnt = apply_smoothed_mean_map(train_key.iloc[hold_idx], means, counts, fold_prior)
            oof_mean[hold_idx] = enc
            oof_count[hold_idx] = cnt

        means, counts, weight_sums = fit_smoothed_mean_map(
            train_key,
            residual_train,
            weights,
            prior=prior,
            smoothing=smoothing,
        )
        eval_mean, eval_count = apply_smoothed_mean_map(eval_key, means, counts, prior)
        train_out[mean_col] = oof_mean
        train_out[count_col] = oof_count
        eval_out[mean_col] = eval_mean
        eval_out[count_col] = eval_count
        mappings[name] = {
            "columns": cols,
            "mean_column": mean_col,
            "count_column": count_col,
            "prior": prior,
            "smoothing": smoothing,
            "means": means,
            "counts": counts,
            "weight_sums": weight_sums,
        }

    return train_out, eval_out, mappings


def prepare_model_matrix(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    numeric_train: pd.DataFrame,
    numeric_val: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, OrdinalEncoder, List[bool], List[str]]:
    categorical_cols = [col for col in CONTEXT_CATEGORICAL_COLUMNS if col in df_train.columns]
    if SUPPORT_COL in categorical_cols:
        raise ValueError("Residual context features must not include ally support directly.")

    train_cat = df_train[categorical_cols].copy()
    val_cat = df_val[categorical_cols].copy()
    for col in categorical_cols:
        train_cat[col] = train_cat[col].fillna("__MISSING__").astype(str)
        val_cat[col] = val_cat[col].fillna("__MISSING__").astype(str)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
    x_train_cat = encoder.fit_transform(train_cat)
    x_val_cat = encoder.transform(val_cat)
    numeric_cols = list(numeric_train.columns)
    x_train = np.hstack([x_train_cat, numeric_train[numeric_cols].to_numpy(dtype=np.float32)])
    x_val = np.hstack([x_val_cat, numeric_val[numeric_cols].to_numpy(dtype=np.float32)])
    categorical_mask = [True] * len(categorical_cols) + [False] * len(numeric_cols)
    feature_columns = categorical_cols + numeric_cols
    return x_train, x_val, encoder, categorical_mask, feature_columns


def summarize_interactions(
    df_val: pd.DataFrame,
    y_val: np.ndarray,
    baseline_val: np.ndarray,
    residual_pred: np.ndarray,
    specs: Dict[str, List[str]],
    champion_names: Optional[Dict[int, str]] = None,
    min_count: int = 5,
    top_n: int = 25,
) -> pd.DataFrame:
    champion_names = champion_names or {}
    rows: List[Dict[str, Any]] = []
    actual_residual = y_val - baseline_val
    for name, cols in specs.items():
        tmp = pd.DataFrame(
            {
                "key": make_key(df_val, cols),
                "actual_residual": actual_residual,
                "predicted_residual": residual_pred,
            }
        )
        grouped = tmp.groupby("key").agg(
            n=("actual_residual", "size"),
            actual_residual_mean=("actual_residual", "mean"),
            predicted_residual_mean=("predicted_residual", "mean"),
        )
        grouped = grouped[grouped["n"] >= min_count].copy()
        if grouped.empty:
            continue
        grouped["interaction"] = name
        grouped["columns"] = ",".join(cols)
        grouped["key_label"] = [
            format_interaction_key(key, cols, champion_names) for key in grouped.index
        ]
        grouped["abs_predicted_residual_mean"] = grouped["predicted_residual_mean"].abs()
        rows.extend(grouped.reset_index().to_dict(orient="records"))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values("abs_predicted_residual_mean", ascending=False).head(top_n)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    champion_names = load_champion_names(Path(args.champion_classes))

    df_train = pd.read_parquet(args.train)
    df_val = pd.read_parquet(args.val)
    if args.limit_train_rows is not None:
        df_train = df_train.head(args.limit_train_rows).copy()
    if args.limit_val_rows is not None:
        df_val = df_val.head(args.limit_val_rows).copy()

    required = [TARGET_COL, WEIGHT_COL, SUPPORT_COL, *CONTEXT_CATEGORICAL_COLUMNS]
    require_columns(df_train, required, "train")
    require_columns(df_val, [TARGET_COL, SUPPORT_COL, *CONTEXT_CATEGORICAL_COLUMNS], "val")
    print(f"[Data] train={len(df_train):,}  val={len(df_val):,}")

    support_baseline = build_support_baseline(df_train, TARGET_COL, args.support_smoothing)
    oof_baseline_train = build_oof_support_baseline(
        df_train,
        TARGET_COL,
        args.n_folds,
        args.support_smoothing,
        args.seed,
    )
    y_train = df_train[TARGET_COL].to_numpy(dtype=np.float32)
    y_val = df_val[TARGET_COL].to_numpy(dtype=np.float32)
    residual_train = y_train - oof_baseline_train
    baseline_val = predict_support_baseline(df_val, support_baseline)

    baseline_metrics = compute_metrics(
        y_val,
        baseline_val,
        "Support Mean Baseline",
        len(df_train),
        elapsed=0.0,
    )
    print(
        f"[Baseline] R2={baseline_metrics['r2']:.4f}  "
        f"Spearman={baseline_metrics['spearman_corr']:.4f}"
    )

    specs = available_interactions(df_train)
    numeric_train, numeric_val, mappings = build_residual_interaction_features(
        df_train,
        df_val,
        residual_train,
        specs,
        smoothing=args.interaction_smoothing,
        n_folds=args.n_folds,
        seed=args.seed,
    )
    x_train, x_val, encoder, categorical_mask, feature_columns = prepare_model_matrix(
        df_train,
        df_val,
        numeric_train,
        numeric_val,
    )
    print(
        f"[Features] shape={x_train.shape}  categorical={sum(categorical_mask)}  "
        f"numeric={len(categorical_mask) - sum(categorical_mask)}"
    )

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
    model.fit(
        x_train,
        residual_train,
        sample_weight=df_train[WEIGHT_COL].to_numpy(dtype=np.float32),
    )
    elapsed = time.time() - t0

    residual_pred = model.predict(x_val).astype(np.float32)
    final_pred = np.clip(baseline_val + residual_pred, 0.0, 1.0)
    final_metrics = compute_metrics(
        y_val,
        final_pred,
        "Support Mean + Residual Interaction GBT",
        len(df_train),
        elapsed,
    )
    residual_metrics = compute_metrics(
        y_val - baseline_val,
        residual_pred,
        "Residual Interaction GBT",
        len(df_train),
        elapsed,
    )
    final_metrics["baseline_r2"] = baseline_metrics["r2"]
    final_metrics["r2_lift_over_support_mean"] = final_metrics["r2"] - baseline_metrics["r2"]
    final_metrics["feature_protocol_id"] = RESIDUAL_FEATURE_PROTOCOL_ID
    final_metrics["used_sample_weight"] = True
    final_metrics["sample_weight_column"] = WEIGHT_COL

    print(
        f"[Residual final] R2={final_metrics['r2']:.4f}  "
        f"Spearman={final_metrics['spearman_corr']:.4f}  "
        f"lift_R2={final_metrics['r2_lift_over_support_mean']:+.4f}"
    )
    print(
        f"[Residual only] R2={residual_metrics['r2']:.4f}  "
        f"Spearman={residual_metrics['spearman_corr']:.4f}"
    )

    interaction_summary = summarize_interactions(
        df_val,
        y_val,
        baseline_val,
        residual_pred,
        specs,
        champion_names=champion_names,
    )
    if not interaction_summary.empty:
        interaction_summary.to_csv(outdir / "top_residual_interactions_val.csv", index=False)

    preprocess = {
        "encoder": encoder,
        "categorical_columns": [col for col in CONTEXT_CATEGORICAL_COLUMNS if col in df_train.columns],
        "numeric_columns": list(numeric_train.columns),
        "feature_columns": feature_columns,
        "categorical_mask": categorical_mask,
        "support_baseline": support_baseline,
        "interaction_mappings": mappings,
    }
    config = {
        "model_type": "support_mean_plus_residual_interaction_gbt",
        "feature_protocol_id": RESIDUAL_FEATURE_PROTOCOL_ID,
        "base_prediction": "sample_weighted_smoothed_ally_support_mean",
        "residual_target": "support_roam_score - OOF_support_mean",
        "excluded_direct_features": [SUPPORT_COL],
        "categorical_columns": preprocess["categorical_columns"],
        "numeric_columns": preprocess["numeric_columns"],
        "interaction_specs": specs,
        "n_folds": args.n_folds,
        "support_smoothing": args.support_smoothing,
        "interaction_smoothing": args.interaction_smoothing,
        "max_iter": args.max_iter,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "min_samples_leaf": args.min_samples_leaf,
        "max_leaf_nodes": args.max_leaf_nodes,
        "seed": args.seed,
        "sample_weight_column": WEIGHT_COL,
        "used_sample_weight": True,
        "debug_limited_rows": {
            "train": args.limit_train_rows,
            "val": args.limit_val_rows,
        },
    }

    joblib.dump(model, outdir / "residual_gbt_model.joblib")
    joblib.dump(preprocess, outdir / "preprocess.joblib")
    (outdir / "model_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (outdir / "metrics.json").write_text(
        json.dumps(
            [baseline_metrics, residual_metrics, final_metrics],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "y_true": y_val,
            "support_mean_pred": baseline_val,
            "residual_pred": residual_pred,
            "final_pred": final_pred,
        }
    ).to_csv(outdir / "val_predictions.csv", index=False)
    print(f"[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
