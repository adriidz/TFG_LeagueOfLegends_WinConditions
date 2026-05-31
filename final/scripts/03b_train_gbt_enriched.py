#!/usr/bin/env python3
"""
03b_train_gbt_enriched.py -- HistGBT with champion IDs + role-aware archetypes.

This experiment tests whether explicit domain features from
final/data/champion_archetypes.json add signal on top of exact champion IDs.
It keeps the same validation protocol as 03_train_gbt.py:

  train split -> fit encoder/model
  val split   -> model selection metric
  test split  -> only used by 07_model_comparison.py
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = str(REPO_ROOT / "final" / "data" / "training" / "train.parquet")
DEFAULT_VAL = str(REPO_ROOT / "final" / "data" / "training" / "val.parquet")
DEFAULT_OUTDIR = str(REPO_ROOT / "final" / "models" / "gbt_enriched")
DEFAULT_CLASSES = str(REPO_ROOT / "final" / "data" / "champion_classes.json")
DEFAULT_ARCHETYPES = str(REPO_ROOT / "final" / "data" / "champion_archetypes.json")

TARGET_COL = "support_roam_score"
QUANTILE_COL = "support_roam_score_quantile"

ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")
ROLE_TO_ARCH_KEY = {
    "top": "top",
    "jungle": "jungle",
    "middle": "mid",
    "bottom": "bottom",
    "utility": "support",
}

BASE_FEATURE_GROUPS: Dict[str, List[str]] = {
    "champions": [f"{s}_{r}_champion_id" for s in SIDES for r in ROLE_KEYS],
    "summoner_spells": [
        f"{s}_{r}_summoner{i}_id" for s in SIDES for r in ROLE_KEYS for i in (1, 2)
    ],
    "context": ["side"],
}
ENRICHED_FEATURE_GROUPS: Dict[str, List[str]] = {
    "archetypes": [f"{s}_{r}_archetype" for s in SIDES for r in ROLE_KEYS],
    "riot_classes": [f"{s}_{r}_class" for s in SIDES for r in ROLE_KEYS],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train enriched HistGBT regressor.")
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--val", default=DEFAULT_VAL)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--champion-classes", default=DEFAULT_CLASSES)
    p.add_argument("--champion-archetypes", default=DEFAULT_ARCHETYPES)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--min-samples-leaf", type=int, default=50)
    p.add_argument("--max-leaf-nodes", type=int, default=31)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_class_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): str(v["primary_class"]) for k, v in raw.items()}


def load_archetypes(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw.get("champions", {})


def add_class_columns(df: pd.DataFrame, class_map: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for side in SIDES:
        for role in ROLE_KEYS:
            id_col = f"{side}_{role}_champion_id"
            class_col = f"{side}_{role}_class"
            if id_col in out.columns:
                out[class_col] = out[id_col].astype("Int64").astype(str).map(class_map).fillna("unknown")
    return out


def add_archetype_columns(
    df: pd.DataFrame,
    archetypes: Dict[str, Dict[str, str]],
    class_map: Dict[str, str],
) -> pd.DataFrame:
    out = df.copy()
    for side in SIDES:
        for role in ROLE_KEYS:
            id_col = f"{side}_{role}_champion_id"
            arch_col = f"{side}_{role}_archetype"
            if id_col not in out.columns:
                continue
            role_key = ROLE_TO_ARCH_KEY[role]

            def lookup(cid: Any, role_key: str = role_key) -> str:
                if pd.isna(cid):
                    return "unknown"
                cid_str = str(int(cid))
                entry = archetypes.get(cid_str, {})
                if role_key in entry:
                    return str(entry[role_key])
                if "generic" in entry:
                    return str(entry["generic"])
                if cid_str in class_map:
                    return class_map[cid_str].lower()
                return "other"

            out[arch_col] = out[id_col].apply(lookup)
    return out


def enrich_dataframe(
    df: pd.DataFrame,
    archetypes: Dict[str, Dict[str, str]],
    class_map: Dict[str, str],
) -> pd.DataFrame:
    out = add_class_columns(df, class_map)
    out = add_archetype_columns(out, archetypes, class_map)
    return out


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], Dict[str, List[str]]]:
    groups: Dict[str, List[str]] = {}
    for name, cols in {**BASE_FEATURE_GROUPS, **ENRICHED_FEATURE_GROUPS}.items():
        groups[name] = [c for c in cols if c in df.columns]

    feature_cols: List[str] = []
    for cols in groups.values():
        feature_cols.extend(cols)
    return list(dict.fromkeys(feature_cols)), groups


def prepare_features(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, OrdinalEncoder, List[bool]]:
    X_train_raw = df_train[feature_cols].copy()
    X_val_raw = df_val[feature_cols].copy()

    for col in feature_cols:
        X_train_raw[col] = X_train_raw[col].fillna("__MISSING__").astype(str)
        X_val_raw[col] = X_val_raw[col].fillna("__MISSING__").astype(str)

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float32,
    )
    X_train = encoder.fit_transform(X_train_raw)
    X_val = encoder.transform(X_val_raw)
    categorical_mask = [True] * len(feature_cols)
    return X_train, X_val, encoder, categorical_mask


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
) -> Dict[str, Any]:
    print(f"\n  Training enriched GBT ({target_label})...")
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
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    y_pred = model.predict(X_val)
    metrics = compute_metrics(
        y_val,
        y_pred,
        f"gbt_enriched_{target_label}",
        target_label,
        n_train=len(y_train),
        elapsed=elapsed,
    )
    joblib.dump(model, outdir / f"gbt_model_{target_label}.joblib")
    print(
        f"  R2={metrics['r2']:.4f}  Spearman={metrics['spearman_corr']:.4f}  "
        f"pred_std={metrics['pred_std']:.4f}  time={elapsed:.1f}s"
    )
    return metrics


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    class_map = load_class_map(Path(args.champion_classes))
    archetypes = load_archetypes(Path(args.champion_archetypes))
    if not archetypes:
        raise SystemExit(f"Missing or empty archetype mapping: {args.champion_archetypes}")

    df_train = enrich_dataframe(pd.read_parquet(args.train), archetypes, class_map)
    df_val = enrich_dataframe(pd.read_parquet(args.val), archetypes, class_map)
    feature_cols, feature_groups = get_feature_columns(df_train)

    print(
        f"[Data] train={len(df_train):,}  val={len(df_val):,}  "
        f"features={len(feature_cols)}"
    )
    print(
        "[Features] "
        + ", ".join(f"{name}={len(cols)}" for name, cols in feature_groups.items())
    )

    X_train, X_val, encoder, cat_mask = prepare_features(df_train, df_val, feature_cols)
    print(f"[Encoding] OrdinalEncoder fitted. Shape: {X_train.shape}")

    results: List[Dict[str, Any]] = []
    y_train_raw = df_train[TARGET_COL].to_numpy(dtype=np.float32)
    y_val_raw = df_val[TARGET_COL].to_numpy(dtype=np.float32)
    results.append(
        train_and_evaluate(X_train, y_train_raw, X_val, y_val_raw, cat_mask, args, "raw", outdir)
    )

    if QUANTILE_COL in df_train.columns and QUANTILE_COL in df_val.columns:
        y_train_q = df_train[QUANTILE_COL].to_numpy(dtype=np.float32)
        y_val_q = df_val[QUANTILE_COL].to_numpy(dtype=np.float32)
        results.append(
            train_and_evaluate(X_train, y_train_q, X_val, y_val_q, cat_mask, args, "quantile", outdir)
        )

    joblib.dump(
        {
            "encoder": encoder,
            "feature_columns": feature_cols,
            "feature_groups": feature_groups,
            "champion_classes_path": str(Path(args.champion_classes).resolve()),
            "champion_archetypes_path": str(Path(args.champion_archetypes).resolve()),
            "class_map": class_map,
            "archetypes": archetypes,
        },
        outdir / "preprocess.joblib",
    )

    config = {
        "feature_columns": feature_cols,
        "feature_groups": feature_groups,
        "max_iter": args.max_iter,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "min_samples_leaf": args.min_samples_leaf,
        "max_leaf_nodes": args.max_leaf_nodes,
        "seed": args.seed,
    }
    (outdir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (outdir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
