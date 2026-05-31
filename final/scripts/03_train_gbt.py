#!/usr/bin/env python3
"""
03_train_gbt.py — HistGradientBoostingRegressor with proper categorical handling.

Uses OrdinalEncoder (fitted on train only) + categorical_features to avoid
imposing artificial ordering on champion/spell/rune IDs.

See final/docs/technical_spec.md (Script 03) for the full specification.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = str(REPO_ROOT / "final" / "data" / "training" / "train.parquet")
DEFAULT_VAL = str(REPO_ROOT / "final" / "data" / "training" / "val.parquet")
DEFAULT_OUTDIR = str(REPO_ROOT / "final" / "models" / "gbt")

TARGET_COL = "support_roam_score"
QUANTILE_COL = "support_roam_score_quantile"

ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")

FEATURE_GROUPS: Dict[str, List[str]] = {
    "champions": [f"{s}_{r}_champion_id" for s in SIDES for r in ROLE_KEYS],
    "summoner_spells": [
        f"{s}_{r}_summoner{i}_id" for s in SIDES for r in ROLE_KEYS for i in (1, 2)
    ],
    "context": ["side"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train HistGBT regressor.")
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--val", default=DEFAULT_VAL)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--min-samples-leaf", type=int, default=50)
    p.add_argument("--max-leaf-nodes", type=int, default=31)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for group_cols in FEATURE_GROUPS.values():
        cols.extend([c for c in group_cols if c in df.columns])
    return list(dict.fromkeys(cols))


def prepare_features(
    df_train: pd.DataFrame, df_val: pd.DataFrame, feature_cols: List[str]
) -> tuple[np.ndarray, np.ndarray, OrdinalEncoder, List[bool]]:
    """OrdinalEncoder on train, transform val. Returns categorical mask."""
    X_train_raw = df_train[feature_cols].copy()
    X_val_raw = df_val[feature_cols].copy()

    # Convert all to string for uniform encoding
    for col in feature_cols:
        X_train_raw[col] = X_train_raw[col].fillna("__MISSING__").astype(str)
        X_val_raw[col] = X_val_raw[col].fillna("__MISSING__").astype(str)

    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float32,
    )
    X_train = enc.fit_transform(X_train_raw)
    X_val = enc.transform(X_val_raw)

    # All columns are categorical
    categorical_mask = [True] * len(feature_cols)

    return X_train, X_val, enc, categorical_mask


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, label: str,
    n_train: int, elapsed: float
) -> Dict[str, Any]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else float("nan")
    sp = spearmanr(y_true, y_pred, nan_policy="omit")
    return {
        "model": model_name,
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
        "n_train": n_train,
        "n_eval": int(len(y_true)),
        "eval_split": "val",
        "training_seconds": elapsed,
    }


def train_and_evaluate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    categorical_mask: List[bool],
    args: argparse.Namespace,
    target_label: str,
    outdir: Path,
    sample_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Train a single GBT model and evaluate."""
    weight_info = "with sample_weight" if sample_weight is not None else "no weights"
    print(f"\n  Training GBT ({target_label}, {weight_info})...")
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
        y_val, y_pred, f"gbt_{target_label}", target_label,
        n_train=len(y_train), elapsed=elapsed,
    )
    metrics["used_sample_weight"] = sample_weight is not None

    # Save model
    suffix = f"_{target_label}"
    model_path = outdir / f"gbt_model{suffix}.joblib"
    joblib.dump(model, model_path)

    print(f"  R2={metrics['r2']:.4f}  Spearman={metrics['spearman_corr']:.4f}  "
          f"pred_std={metrics['pred_std']:.4f}  time={elapsed:.1f}s")

    return metrics


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(args.train)
    df_val = pd.read_parquet(args.val)
    feature_cols = get_feature_columns(df_train)
    print(f"[Data] train={len(df_train):,}  val={len(df_val):,}  features={len(feature_cols)}")

    X_train, X_val, encoder, cat_mask = prepare_features(df_train, df_val, feature_cols)
    print(f"[Encoding] OrdinalEncoder fitted. Shape: {X_train.shape}")

    # Extract sample_weight if available
    sw_col = "sample_weight"
    if sw_col in df_train.columns:
        sample_weight = df_train[sw_col].to_numpy(dtype=np.float32)
        print(f"[Weights] Using sample_weight: mean={sample_weight.mean():.3f}  "
              f"min={sample_weight.min():.3f}  max={sample_weight.max():.3f}")
    else:
        sample_weight = None
        print("[Weights] No sample_weight column found - training without weights")

    results = []

    # Raw target
    y_train_raw = df_train[TARGET_COL].to_numpy(dtype=np.float32)
    y_val_raw = df_val[TARGET_COL].to_numpy(dtype=np.float32)
    m_raw = train_and_evaluate(
        X_train, y_train_raw, X_val, y_val_raw,
        cat_mask, args, "raw", outdir, sample_weight=sample_weight,
    )
    results.append(m_raw)

    # Quantile target
    if QUANTILE_COL in df_train.columns:
        y_train_q = df_train[QUANTILE_COL].to_numpy(dtype=np.float32)
        y_val_q = df_val[QUANTILE_COL].to_numpy(dtype=np.float32)
        m_q = train_and_evaluate(
            X_train, y_train_q, X_val, y_val_q,
            cat_mask, args, "quantile", outdir, sample_weight=sample_weight,
        )
        results.append(m_q)

    # Save encoder and config
    joblib.dump({"encoder": encoder, "feature_columns": feature_cols},
                outdir / "preprocess.joblib")

    config = {
        "feature_columns": feature_cols,
        "feature_groups": {k: v for k, v in FEATURE_GROUPS.items()},
        "max_iter": args.max_iter,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "min_samples_leaf": args.min_samples_leaf,
        "max_leaf_nodes": args.max_leaf_nodes,
        "seed": args.seed,
    }
    (outdir / "model_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (outdir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
