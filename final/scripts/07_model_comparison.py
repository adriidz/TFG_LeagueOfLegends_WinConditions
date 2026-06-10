#!/usr/bin/env python3
"""
07_model_comparison.py -- Final comparison tables on the held-out TEST split.

This script does not trust validation metrics saved during development. It
reloads every available artifact and evaluates it on test. Quantile-trained
models are reported twice:
  - in quantile space, against support_roam_score_quantile
  - inverse-transformed to raw space, against support_roam_score
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = str(REPO_ROOT / "final" / "data" / "training" / "train.parquet")
DEFAULT_TEST = str(REPO_ROOT / "final" / "data" / "training" / "test.parquet")
DEFAULT_TRANSFORMER = str(REPO_ROOT / "final" / "data" / "training" / "quantile_transformer.joblib")
DEFAULT_OUTDIR = str(REPO_ROOT / "final" / "analysis" / "model_comparison")

TARGET_COL = "support_roam_score"
QUANTILE_COL = "support_roam_score_quantile"

ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")
CHAMPION_COLS = [f"{s}_{r}_champion_id" for s in SIDES for r in ROLE_KEYS]
CANONICAL_MAIN_FEATURES = CHAMPION_COLS + ["side"]
MAIN_FEATURE_PROTOCOL_ID = "draft_10_champions_side"
PRIMARY_MODEL_KEYS = ["baselines", "gbt", "mlp_onehot", "mlp_embed", "mlp_per_role"]
SECONDARY_MODEL_KEYS = [
    "gbt_enriched",
    "gbt_interactions",
    "residual_interactions",
    "mlp_per_role_tuned",
    "ceiling",
]
MAIN_LEARNED_MODELS = {
    "HistGBT",
    "MLP OneHot",
    "MLP Embed Shared",
    "MLP Per-Role + Interactions",
}
MAIN_BASELINE_MODELS = {"Global Mean", "Champion Mean"}
FINAL_MAIN_COLUMNS = [
    "model",
    "r2",
    "spearman_corr",
    "pearson_corr",
    "mae",
    "rmse",
    "pred_std",
    "n_eval",
    "within_010",
    "within_020",
]
SIDE_MAPPING = {"blue": 0.0, "red": 1.0}
ROLE_TO_ARCH_KEY = {
    "top": "top",
    "jungle": "jungle",
    "middle": "mid",
    "bottom": "bottom",
    "utility": "support",
}

PRACTICAL_CONTEXT: Dict[str, Any] = {}
DECILE_ROWS: List[Dict[str, Any]] = []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final model comparison on test set.")
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--test", default=DEFAULT_TEST)
    p.add_argument("--quantile-transformer", default=DEFAULT_TRANSFORMER)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument(
        "--models",
        nargs="+",
        choices=PRIMARY_MODEL_KEYS + SECONDARY_MODEL_KEYS,
        default=None,
        help="Models to include. Defaults to the fair primary comparison only.",
    )
    p.add_argument(
        "--include-secondary",
        action="store_true",
        help="Also include enriched/Pair-TE/HP-best/reference rows as secondary analyses.",
    )
    return p.parse_args()


def selected_model_keys(args: argparse.Namespace) -> List[str]:
    if args.models:
        keys = list(args.models)
        if args.include_secondary:
            keys.extend(SECONDARY_MODEL_KEYS)
        return list(dict.fromkeys(keys))
    keys = list(PRIMARY_MODEL_KEYS)
    if args.include_secondary:
        keys.extend(SECONDARY_MODEL_KEYS)
    return keys


def find_model_run_dirs(base_dir: Path, target_names: List[str] = ["model_config.json"]) -> List[Path]:
    if not base_dir.exists():
        return []
    seed_subdirs = sorted(list(base_dir.glob("seed*")))
    valid_seed_dirs = []
    for d in seed_subdirs:
        if d.is_dir() and all((d / name).exists() for name in target_names):
            valid_seed_dirs.append(d)
    
    if valid_seed_dirs:
        return valid_seed_dirs
    if all((base_dir / name).exists() for name in target_names):
        return [base_dir]
    return []


def init_practical_context(train_y_raw: np.ndarray) -> None:
    """Store raw-scale bin edges derived from train only."""
    train_y = np.asarray(train_y_raw, dtype=np.float64)
    quantiles = np.quantile(train_y, [0.25, 0.50, 0.75])
    PRACTICAL_CONTEXT.clear()
    PRACTICAL_CONTEXT.update(
        {
            "fixed_edges": np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float64),
            "train_quantile_edges": np.array(
                [0.0, quantiles[0], quantiles[1], quantiles[2], 1.0],
                dtype=np.float64,
            ),
            "train_quantiles": {
                "p25": float(quantiles[0]),
                "p50": float(quantiles[1]),
                "p75": float(quantiles[2]),
            },
        }
    )
    DECILE_ROWS.clear()


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.to_numpy(dtype=np.float64)
    v = values.to_numpy(dtype=np.float64)
    w_sum = float(w.sum())
    if w_sum <= 1e-8:
        return float(np.mean(v))
    return float(np.sum(v * w) / w_sum)


def bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), edges[0], edges[-1])
    return np.digitize(clipped, edges[1:-1], right=False).astype(np.int64)


def safe_rank_corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        return float("nan")
    if method == "spearman":
        corr = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    elif method == "kendall":
        corr = kendalltau(y_true, y_pred, nan_policy="omit").correlation
    else:
        raise ValueError(f"Unknown rank correlation method: {method}")
    return float(corr) if corr is not None else float("nan")


def quadratic_weighted_kappa(y_true_bins: np.ndarray, y_pred_bins: np.ndarray) -> float:
    if len(np.unique(y_true_bins)) < 2 or len(np.unique(y_pred_bins)) < 2:
        return float("nan")
    return float(cohen_kappa_score(y_true_bins, y_pred_bins, weights="quadratic"))


def practical_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Raw-scale tolerant metrics for strategic usefulness."""
    if not PRACTICAL_CONTEXT:
        return {}

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    abs_error = np.abs(y_true - y_pred)

    fixed_true = bin_indices(y_true, PRACTICAL_CONTEXT["fixed_edges"])
    fixed_pred = bin_indices(y_pred, PRACTICAL_CONTEXT["fixed_edges"])
    quant_true = bin_indices(y_true, PRACTICAL_CONTEXT["train_quantile_edges"])
    quant_pred = bin_indices(y_pred, PRACTICAL_CONTEXT["train_quantile_edges"])

    return {
        "within_005": float(np.mean(abs_error <= 0.05)),
        "within_010": float(np.mean(abs_error <= 0.10)),
        "within_015": float(np.mean(abs_error <= 0.15)),
        "within_020": float(np.mean(abs_error <= 0.20)),
        "fixed_bin_acc": float(np.mean(fixed_true == fixed_pred)),
        "fixed_bin_adjacent_acc": float(np.mean(np.abs(fixed_true - fixed_pred) <= 1)),
        "train_quantile_bin_acc": float(np.mean(quant_true == quant_pred)),
        "train_quantile_adjacent_acc": float(np.mean(np.abs(quant_true - quant_pred) <= 1)),
        "fixed_bin_spearman": safe_rank_corr(fixed_true, fixed_pred, "spearman"),
        "train_quantile_bin_spearman": safe_rank_corr(quant_true, quant_pred, "spearman"),
        "fixed_bin_kendall_tau": safe_rank_corr(fixed_true, fixed_pred, "kendall"),
        "train_quantile_bin_kendall_tau": safe_rank_corr(quant_true, quant_pred, "kendall"),
        "fixed_bin_qwk": quadratic_weighted_kappa(fixed_true, fixed_pred),
        "train_quantile_bin_qwk": quadratic_weighted_kappa(quant_true, quant_pred),
    }


def record_deciles(model: str, trained_target: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Record raw-scale decile summaries ordered by prediction."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    order = np.argsort(y_pred, kind="mergesort")
    for decile, idx in enumerate(np.array_split(order, 10), start=1):
        if len(idx) == 0:
            continue
        true_slice = y_true[idx]
        pred_slice = y_pred[idx]
        DECILE_ROWS.append(
            {
                "model": model,
                "trained_target": trained_target,
                "decile": decile,
                "n": int(len(idx)),
                "pred_mean": float(np.mean(pred_slice)),
                "true_mean": float(np.mean(true_slice)),
                "true_std": float(np.std(true_slice)),
                "abs_error_mean": float(np.mean(np.abs(true_slice - pred_slice))),
                "true_q25": float(np.quantile(true_slice, 0.25)),
                "true_q75": float(np.quantile(true_slice, 0.75)),
            }
        )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    target_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred))
    if target_std > 1e-12 and pred_std > 1e-12:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        sp = spearmanr(y_true, y_pred, nan_policy="omit")
        spearman = float(sp.correlation) if sp.correlation is not None else float("nan")
    else:
        pearson = float("nan")
        spearman = float("nan")
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "r2": float(r2),
        "pearson_corr": pearson,
        "spearman_corr": spearman,
        "pred_std": pred_std,
        "target_std": target_std,
        "compression_ratio": pred_std / target_std if target_std > 0 else float("nan"),
    }


def make_row(
    model: str,
    trained_target: str,
    evaluation_scale: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_train: int,
    notes: str = "",
) -> Dict[str, Any]:
    row = {
        "model": model,
        "trained_target": trained_target,
        "evaluation_scale": evaluation_scale,
        "eval_split": "test",
        "n_train": int(n_train),
        "n_eval": int(len(y_true)),
        "notes": notes,
        **regression_metrics(y_true, y_pred),
    }
    if evaluation_scale == "raw":
        row.update(practical_metrics(y_true, y_pred))
        record_deciles(model, trained_target, y_true, y_pred)
    return row


def inverse_quantile_predictions(q_pred: np.ndarray, transformer: Optional[Any]) -> Optional[np.ndarray]:
    if transformer is None:
        return None
    q = np.asarray(q_pred, dtype=np.float64)
    q = np.clip(q, 0.0, 1.0)
    raw = np.zeros_like(q, dtype=np.float64)
    positive = q > 0.0
    if positive.any():
        raw[positive] = transformer.inverse_transform(q[positive].reshape(-1, 1)).reshape(-1)
    return np.clip(raw, 0.0, 1.0)


def add_quantile_rows(
    rows: List[Dict[str, Any]],
    model: str,
    q_pred: np.ndarray,
    df_test: pd.DataFrame,
    n_train: int,
    transformer: Optional[Any],
) -> None:
    rows.append(
        make_row(
            model=model,
            trained_target="quantile",
            evaluation_scale="quantile",
            y_true=df_test[QUANTILE_COL].to_numpy(),
            y_pred=q_pred,
            n_train=n_train,
        )
    )
    raw_pred = inverse_quantile_predictions(q_pred, transformer)
    if raw_pred is not None:
        rows.append(
            make_row(
                model=f"{model} (quantile->raw)",
                trained_target="quantile",
                evaluation_scale="raw",
                y_true=df_test[TARGET_COL].to_numpy(),
                y_pred=raw_pred,
                n_train=n_train,
                notes="Quantile predictions inverse-transformed to raw scale.",
            )
        )


def eval_mean_baselines(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    transformer: Optional[Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n_train = len(df_train)
    if "sample_weight" not in df_train.columns:
        raise SystemExit("[Weights] Missing required sample_weight column for main baselines.")
    weights = df_train["sample_weight"].astype(np.float64)

    for label, target_col in [("raw", TARGET_COL), ("quantile", QUANTILE_COL)]:
        if target_col not in df_train.columns or target_col not in df_test.columns:
            continue

        global_mean = weighted_mean(df_train[target_col], weights)
        y_pred_global = np.full(len(df_test), global_mean, dtype=np.float64)
        if label == "raw":
            row = make_row(
                "Global Mean",
                "raw",
                "raw",
                df_test[TARGET_COL].to_numpy(),
                y_pred_global,
                n_train,
                notes="Sample-weighted train mean.",
            )
            row["used_sample_weight"] = True
            row["seed"] = None
            row["feature_protocol_id"] = "baseline_no_features"
            rows.append(row)
        else:
            add_quantile_rows(rows, "Global Mean", y_pred_global, df_test, n_train, transformer)
            for row in rows[-2:]:
                if row["model"].startswith("Global Mean"):
                    row["notes"] = (
                        f"{row['notes']} Sample-weighted train mean."
                        if row["notes"]
                        else "Sample-weighted train mean."
                    )
                    row["used_sample_weight"] = True

        champ_col = "ally_utility_champion_id"
        means = df_train.groupby(champ_col).apply(
            lambda g: weighted_mean(g[target_col], g["sample_weight"])
        )
        fallback = global_mean
        y_pred_champ = df_test[champ_col].map(means).fillna(fallback).to_numpy(dtype=np.float64)
        unseen = int((~df_test[champ_col].isin(means.index)).sum())
        notes = f"Sample-weighted support-champion means. {unseen} unseen support champions in test."
        if label == "raw":
            row = make_row(
                "Champion Mean",
                "raw",
                "raw",
                df_test[TARGET_COL].to_numpy(),
                y_pred_champ,
                n_train,
                notes=notes,
            )
            row["used_sample_weight"] = True
            row["seed"] = None
            row["feature_protocol_id"] = "baseline_support_champion_only"
            rows.append(row)
        else:
            before = len(rows)
            add_quantile_rows(rows, "Champion Mean", y_pred_champ, df_test, n_train, transformer)
            for row in rows[before:]:
                row["notes"] = notes if not row["notes"] else f"{row['notes']} {notes}"
                row["used_sample_weight"] = True

    return rows


def add_gbt_enrichment_columns(df: pd.DataFrame, preprocess: Dict[str, Any]) -> pd.DataFrame:
    class_map = preprocess.get("class_map") or {}
    archetypes = preprocess.get("archetypes") or {}
    if not class_map and not archetypes:
        return df

    out = df.copy()
    for side in SIDES:
        for role in ROLE_KEYS:
            id_col = f"{side}_{role}_champion_id"
            if id_col not in out.columns:
                continue

            class_col = f"{side}_{role}_class"
            if class_map:
                out[class_col] = out[id_col].astype("Int64").astype(str).map(class_map).fillna("unknown")

            arch_col = f"{side}_{role}_archetype"
            if archetypes:
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
                        return str(class_map[cid_str]).lower()
                    return "other"

                out[arch_col] = out[id_col].apply(lookup)
    return out


def prepare_gbt_features(df: pd.DataFrame, feature_cols: List[str], encoder: Any) -> np.ndarray:
    X_raw = df[feature_cols].copy()
    for col in feature_cols:
        X_raw[col] = X_raw[col].fillna("__MISSING__").astype(str)
    return encoder.transform(X_raw)


def eval_gbt(
    df_test: pd.DataFrame,
    n_train: int,
    transformer: Optional[Any],
) -> List[Dict[str, Any]]:
    return eval_gbt_family(
        df_test=df_test,
        n_train=n_train,
        transformer=transformer,
        model_dir=REPO_ROOT / "final" / "models" / "gbt",
        model_name="HistGBT",
    )


def eval_gbt_family(
    df_test: pd.DataFrame,
    n_train: int,
    transformer: Optional[Any],
    model_dir: Path,
    model_name: str,
) -> List[Dict[str, Any]]:
    run_dirs = find_model_run_dirs(model_dir, ["model_config.json", "preprocess.joblib"])
    if not run_dirs:
        print(f"[{model_name}] no valid run directories found in {model_dir}, skipping")
        return []

    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        preprocess_path = run_dir / "preprocess.joblib"
        preprocess = joblib.load(preprocess_path)
        encoder = preprocess["encoder"]
        feature_cols = preprocess["feature_columns"]
        config = load_json_if_exists(run_dir / "model_config.json")
        df_eval = add_gbt_enrichment_columns(df_test, preprocess)
        X_test = prepare_gbt_features(df_eval, feature_cols, encoder)

        for label, target_col in [("raw", TARGET_COL), ("quantile", QUANTILE_COL)]:
            model_path = run_dir / f"gbt_model_{label}.joblib"
            if not model_path.exists():
                continue
            model = joblib.load(model_path)
            y_pred = model.predict(X_test)
            if label == "raw":
                rows.append(
                    {
                        **make_row(
                            model_name,
                            "raw",
                            "raw",
                            df_test[TARGET_COL].to_numpy(),
                            y_pred,
                            n_train,
                        ),
                        "seed": config.get("seed"),
                        "feature_protocol_id": config.get("feature_protocol_id"),
                    }
                )
            elif target_col in df_test.columns:
                add_quantile_rows(rows, model_name, y_pred, df_test, n_train, transformer)
    return rows


def make_interaction_key(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    parts = []
    for col in cols:
        parts.append(df[col].fillna(-1).astype(int).astype(str))
    return pd.Series(["|".join(values) for values in zip(*parts)], index=df.index)


def build_interaction_eval_matrix(df_test: pd.DataFrame, preprocess: Dict[str, Any]) -> np.ndarray:
    cat_cols = preprocess["categorical_columns"]
    encoder = preprocess["encoder"]
    X_cat = df_test[cat_cols].copy()
    for col in cat_cols:
        X_cat[col] = X_cat[col].fillna("__MISSING__").astype(str)
    X_cat_arr = encoder.transform(X_cat)

    numeric_cols = preprocess["numeric_columns"]
    numeric = pd.DataFrame(index=df_test.index)
    mappings = preprocess["interaction_mappings"]
    for spec_name, info in mappings.items():
        key = make_interaction_key(df_test, info["columns"])
        mean_col = info["mean_column"]
        count_col = info["count_column"]
        values = info["values"]
        counts = info["counts"]
        global_mean = float(info["global_mean"])
        numeric[mean_col] = key.map(values).fillna(global_mean).astype(np.float32)
        numeric[count_col] = np.log1p(key.map(counts).fillna(0).to_numpy(dtype=np.float32))

    return np.hstack([X_cat_arr, numeric[numeric_cols].to_numpy(dtype=np.float32)])


def eval_gbt_interactions(
    df_test: pd.DataFrame,
    n_train: int,
    transformer: Optional[Any],
) -> List[Dict[str, Any]]:
    model_dir = REPO_ROOT / "final" / "models" / "gbt_interactions"
    preprocess_path = model_dir / "preprocess.joblib"
    if not preprocess_path.exists():
        print("[HistGBT + Pair TE] preprocess.joblib not found, skipping")
        return []

    preprocess_by_target = joblib.load(preprocess_path)
    rows: List[Dict[str, Any]] = []
    for label, target_col in [("raw", TARGET_COL), ("quantile", QUANTILE_COL)]:
        model_path = model_dir / f"gbt_model_{label}.joblib"
        if not model_path.exists() or label not in preprocess_by_target:
            print(f"[HistGBT + Pair TE] {model_path.name} not found, skipping")
            continue

        X_test = build_interaction_eval_matrix(df_test, preprocess_by_target[label])
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)

        if label == "raw":
            rows.append(
                make_row(
                    "HistGBT + Pair TE",
                    "raw",
                    "raw",
                    df_test[TARGET_COL].to_numpy(),
                    y_pred,
                    n_train,
                    notes="OOF-smoothed pair target encodings fitted on train only.",
                )
            )
        elif target_col in df_test.columns:
            before = len(rows)
            add_quantile_rows(rows, "HistGBT + Pair TE", y_pred, df_test, n_train, transformer)
            for row in rows[before:]:
                row["notes"] = (
                    "OOF-smoothed pair target encodings fitted on train only."
                    if not row["notes"]
                    else f"{row['notes']} OOF-smoothed pair target encodings fitted on train only."
                )
    return rows


def predict_residual_support_effect(
    df: pd.DataFrame,
    support_baseline: Dict[str, Any],
) -> np.ndarray:
    keys = df["ally_utility_champion_id"].fillna(-1).astype(int).astype(str)
    return (
        keys.map(support_baseline["support_means"])
        .fillna(float(support_baseline["global_mean"]))
        .to_numpy(dtype=np.float32)
    )


def build_residual_eval_matrix(df_test: pd.DataFrame, preprocess: Dict[str, Any]) -> np.ndarray:
    cat_cols = preprocess["categorical_columns"]
    encoder = preprocess["encoder"]
    x_cat = df_test[cat_cols].copy()
    for col in cat_cols:
        x_cat[col] = x_cat[col].fillna("__MISSING__").astype(str)
    x_cat_arr = encoder.transform(x_cat)

    numeric_cols = preprocess["numeric_columns"]
    numeric = pd.DataFrame(index=df_test.index)
    for _, info in preprocess["interaction_mappings"].items():
        key = make_interaction_key(df_test, info["columns"])
        mean_col = info["mean_column"]
        count_col = info["count_column"]
        means = info["means"]
        counts = info["counts"]
        prior = float(info["prior"])
        numeric[mean_col] = key.map(means).fillna(prior).astype(np.float32)
        numeric[count_col] = np.log1p(key.map(counts).fillna(0).to_numpy(dtype=np.float32))

    return np.hstack([x_cat_arr, numeric[numeric_cols].to_numpy(dtype=np.float32)])


def make_residual_context_rows(
    df_test: pd.DataFrame,
    support_pred: np.ndarray,
    residual_pred: np.ndarray,
    n_train: int,
) -> List[Dict[str, Any]]:
    def diagnostic_row(
        model: str,
        trained_target: str,
        evaluation_scale: str,
        y_row_true: np.ndarray,
        y_row_pred: np.ndarray,
        notes: str,
    ) -> Dict[str, Any]:
        return {
            "model": model,
            "trained_target": trained_target,
            "evaluation_scale": evaluation_scale,
            "eval_split": "test",
            "n_train": int(n_train),
            "n_eval": int(len(y_row_true)),
            "notes": notes,
            **regression_metrics(y_row_true, y_row_pred),
        }

    y_true = df_test[TARGET_COL].to_numpy(dtype=np.float64)
    support_pred = np.asarray(support_pred, dtype=np.float64)
    residual_pred = np.asarray(residual_pred, dtype=np.float64)
    final_pred = np.clip(support_pred + residual_pred, 0.0, 1.0)
    residual_true = y_true - support_pred

    support_row = diagnostic_row(
        "Smoothed Support Mean",
        "raw",
        "raw",
        y_true,
        support_pred,
        "Sample-weighted smoothed ally-support mean fitted on train only.",
    )
    residual_row = diagnostic_row(
        "Residual Context GBT",
        "residual",
        "residual",
        residual_true,
        residual_pred,
        (
            "Residual target y - support_effect; direct ally support excluded "
            "from residual categorical features."
        ),
    )
    final_row = diagnostic_row(
        "Smoothed Support Mean + Residual Context GBT",
        "raw",
        "raw",
        y_true,
        final_pred,
        (
            "Additive diagnostic: smoothed support effect plus residual context "
            "GBT. Direct ally support is excluded from residual categorical "
            "features and used only through explicit smoothed interactions."
        ),
    )
    final_row["support_effect_r2"] = support_row["r2"]
    final_row["support_effect_spearman_corr"] = support_row["spearman_corr"]
    final_row["r2_lift_over_support_effect"] = final_row["r2"] - support_row["r2"]
    final_row["spearman_lift_over_support_effect"] = (
        final_row["spearman_corr"] - support_row["spearman_corr"]
    )
    final_row["residual_r2"] = residual_row["r2"]
    final_row["residual_spearman_corr"] = residual_row["spearman_corr"]
    for row in (support_row, residual_row, final_row):
        row["diagnostic_family"] = "support_residual"
    return [support_row, residual_row, final_row]


def eval_residual_interactions(
    df_test: pd.DataFrame,
    n_train: int,
) -> List[Dict[str, Any]]:
    model_dir = REPO_ROOT / "final" / "models" / "residual_interactions"
    preprocess_path = model_dir / "preprocess.joblib"
    model_path = model_dir / "residual_gbt_model.joblib"
    if not preprocess_path.exists() or not model_path.exists():
        print("[Residual Context] residual artifacts not found, skipping")
        return []

    preprocess = joblib.load(preprocess_path)
    support_pred = predict_residual_support_effect(df_test, preprocess["support_baseline"])
    x_test = build_residual_eval_matrix(df_test, preprocess)
    model = joblib.load(model_path)
    residual_pred = model.predict(x_test)
    return make_residual_context_rows(df_test, support_pred, residual_pred, n_train)


def iter_batches(n: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, n, batch_size):
        yield slice(start, min(start + batch_size, n))


def encode_champion_ids(df: pd.DataFrame, vocab: Dict[int, int]) -> np.ndarray:
    ids = np.zeros((len(df), len(CHAMPION_COLS)), dtype=np.int64)
    for i, col in enumerate(CHAMPION_COLS):
        if col in df.columns:
            ids[:, i] = (
                df[col]
                .fillna(-1)
                .astype(int)
                .map(lambda x: vocab.get(x, 0))
                .to_numpy(dtype=np.int64)
            )
    return ids


def encode_side(df: pd.DataFrame) -> np.ndarray:
    return (
        df["side"]
        .map(SIDE_MAPPING)
        .fillna(0.5)
        .to_numpy(dtype=np.float32)
        .reshape(-1, 1)
    )


def torch_load_state(path: Path, device: Any) -> Dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def eval_mlp(
    df_test: pd.DataFrame,
    model_type: str,
    n_train: int,
    transformer: Optional[Any],
    batch_size: int,
) -> List[Dict[str, Any]]:
    model_dir = REPO_ROOT / "final" / "models" / f"mlp_{model_type}"
    run_dirs = find_model_run_dirs(model_dir, ["model_config.json", "vocab.json"])
    if not run_dirs:
        print(f"[MLP {model_type}] no valid run directories found, skipping")
        return []

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ModuleNotFoundError as exc:
        print(f"[MLP {model_type}] torch unavailable ({exc}), skipping")
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class MLPOneHotEval(nn.Module):
        def __init__(self, vocab_size: int, n_slots: int, hidden_dims: List[int]):
            super().__init__()
            self.vocab_size = vocab_size
            layers: List[nn.Module] = []
            prev = n_slots * vocab_size + 1
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.0)])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, champion_ids: Any, side: Any) -> Any:
            onehot = F.one_hot(champion_ids, num_classes=self.vocab_size).to(torch.float32)
            x = torch.cat([onehot.flatten(start_dim=1), side], dim=1)
            return self.net(x).squeeze(-1)

    class MLPEmbedEval(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int, n_slots: int, hidden_dims: List[int]):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            layers: List[nn.Module] = []
            prev = n_slots * embed_dim + 1
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.0)])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.head = nn.Sequential(*layers)

        def forward(self, champion_ids: Any, side: Any) -> Any:
            emb = self.embed(champion_ids).view(champion_ids.size(0), -1)
            return self.head(torch.cat([emb, side], dim=1)).squeeze(-1)

    class MLPPerRoleEval(nn.Module):
        def __init__(self, vocab_size: int, embed_dim: int, n_slots: int, hidden_dims: List[int]):
            super().__init__()
            self.slot_embeddings = nn.ModuleList(
                [nn.Embedding(vocab_size, embed_dim, padding_idx=0) for _ in range(n_slots)]
            )
            layers: List[nn.Module] = []
            prev = n_slots * embed_dim + 1 + 2
            for h in hidden_dims:
                layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.0)])
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.head = nn.Sequential(*layers)

        def forward(self, champion_ids: Any, side: Any) -> Any:
            emb = torch.stack(
                [slot(champion_ids[:, i]) for i, slot in enumerate(self.slot_embeddings)],
                dim=1,
            )
            ally_utility_idx = CHAMPION_COLS.index("ally_utility_champion_id")
            enemy_utility_idx = CHAMPION_COLS.index("enemy_utility_champion_id")
            ally_bottom_idx = CHAMPION_COLS.index("ally_bottom_champion_id")
            support_vs_support = (emb[:, ally_utility_idx] * emb[:, enemy_utility_idx]).sum(
                dim=1, keepdim=True
            )
            support_adc = (emb[:, ally_utility_idx] * emb[:, ally_bottom_idx]).sum(
                dim=1, keepdim=True
            )
            x = torch.cat(
                [emb.flatten(start_dim=1), side, support_vs_support, support_adc],
                dim=1,
            )
            return self.head(x).squeeze(-1)

    rows: List[Dict[str, Any]] = []
    pretty_names = {
        "onehot": "MLP OneHot",
        "embed": "MLP Embed Shared",
        "per_role": "MLP Per-Role + Interactions",
        "per_role_tuned": "MLP Per-Role + Interactions HP Best",
    }
    pretty_name = pretty_names.get(model_type, f"MLP {model_type}")

    for run_dir in run_dirs:
        config_path = run_dir / "model_config.json"
        vocab_path = run_dir / "vocab.json"
        
        weight_paths = {
            label: run_dir / f"mlp_{model_type}_{label}.pt"
            for label in ("raw", "quantile")
        }
        weight_paths = {k: v for k, v in weight_paths.items() if v.exists()}
        if not weight_paths:
            print(f"[MLP {model_type}] weights not found in {run_dir}, skipping")
            continue

        config = json.loads(config_path.read_text(encoding="utf-8"))
        vocab_raw = json.loads(vocab_path.read_text(encoding="utf-8"))
        vocab = {int(k): int(v) for k, v in vocab_raw.items()}
        champ_ids_np = encode_champion_ids(df_test, vocab)
        side_np = encode_side(df_test)
        n_slots = int(config.get("n_champion_slots", len(CHAMPION_COLS)))

        for label, path in weight_paths.items():
            if model_type == "onehot":
                model = MLPOneHotEval(
                    vocab_size=int(config["vocab_size"]),
                    n_slots=n_slots,
                    hidden_dims=list(config["hidden_dims"]),
                )
            elif model_type in {"per_role", "per_role_tuned"}:
                model = MLPPerRoleEval(
                    vocab_size=int(config["vocab_size"]),
                    embed_dim=int(config["embed_dim"]),
                    n_slots=n_slots,
                    hidden_dims=list(config["hidden_dims"]),
                )
            else:
                model = MLPEmbedEval(
                    vocab_size=int(config["vocab_size"]),
                    embed_dim=int(config["embed_dim"]),
                    n_slots=n_slots,
                    hidden_dims=list(config["hidden_dims"]),
                )
            model.load_state_dict(torch_load_state(path, device))
            model.to(device)
            model.eval()

            preds: List[np.ndarray] = []
            with torch.no_grad():
                for sl in iter_batches(len(df_test), batch_size):
                    champion_ids = torch.from_numpy(champ_ids_np[sl]).to(device)
                    side = torch.from_numpy(side_np[sl]).to(device)
                    preds.append(model(champion_ids, side).cpu().numpy())
            y_pred = np.concatenate(preds)

            if label == "raw":
                rows.append(
                    {
                        **make_row(
                            pretty_name,
                            "raw",
                            "raw",
                            df_test[TARGET_COL].to_numpy(),
                            y_pred,
                            n_train,
                        ),
                        "seed": config.get("seed"),
                        "feature_protocol_id": config.get("feature_protocol_id"),
                    }
                )
            elif QUANTILE_COL in df_test.columns:
                add_quantile_rows(rows, pretty_name, y_pred, df_test, n_train, transformer)

    return rows


def add_ceiling_reference(rows: List[Dict[str, Any]], n_eval: int) -> None:
    ceiling_path = REPO_ROOT / "final" / "analysis" / "ceiling" / "ceiling_oos_summary.csv"
    if not ceiling_path.exists():
        return
    data = pd.read_csv(ceiling_path)
    ref_df = data[data["grouping"] == "botlane_champions+side"]
    if ref_df.empty:
        return
    ref = ref_df.iloc[0]
    rows.append(
        {
            "model": "OOS Group Mean Reference (botlane+side)",
            "trained_target": "reference",
            "evaluation_scale": "raw",
            "eval_split": "test",
            "n_train": int(ref.get("n_train_groups", 0)),
            "n_eval": int(n_eval),
            "notes": (
                "Train-only botlane+side group means applied to test; unseen groups "
                "fall back to train global mean. ICC is descriptive and not used as model R2."
            ),
            "mse": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float(ref.get("r2_group_mean_oos", float("nan"))),
            "pearson_corr": float("nan"),
            "spearman_corr": float("nan"),
            "pred_std": float("nan"),
            "target_std": float("nan"),
            "compression_ratio": float("nan"),
        }
    )


def sorted_table(rows: List[Dict[str, Any]], evaluation_scale: str) -> pd.DataFrame:
    df = pd.DataFrame([r for r in rows if r["evaluation_scale"] == evaluation_scale])
    if df.empty:
        return df
    df = df.sort_values(["spearman_corr", "r2"], ascending=[False, False], na_position="last")
    cols = [
        "model",
        "trained_target",
        "evaluation_scale",
        "r2",
        "spearman_corr",
        "pearson_corr",
        "rmse",
        "mae",
        "compression_ratio",
        "pred_std",
        "target_std",
        "n_eval",
        "notes",
    ]
    return df[[c for c in cols if c in df.columns]]


def practical_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            r for r in rows
            if r.get("evaluation_scale") == "raw" and "fixed_bin_adjacent_acc" in r
        ]
    )
    if df.empty:
        return df

    df = df.sort_values(
        ["fixed_bin_qwk", "fixed_bin_spearman", "within_010"],
        ascending=[False, False, False],
        na_position="last",
    )
    cols = [
        "model",
        "trained_target",
        "r2",
        "spearman_corr",
        "mae",
        "within_005",
        "within_010",
        "within_015",
        "within_020",
        "fixed_bin_acc",
        "fixed_bin_adjacent_acc",
        "fixed_bin_spearman",
        "fixed_bin_kendall_tau",
        "fixed_bin_qwk",
        "train_quantile_bin_acc",
        "train_quantile_adjacent_acc",
        "train_quantile_bin_spearman",
        "train_quantile_bin_kendall_tau",
        "train_quantile_bin_qwk",
        "n_eval",
        "notes",
    ]
    return df[[c for c in cols if c in df.columns]]


def decile_table() -> pd.DataFrame:
    if not DECILE_ROWS:
        return pd.DataFrame()
    return pd.DataFrame(DECILE_ROWS).sort_values(["model", "trained_target", "decile"])


def format_markdown_table(df: pd.DataFrame, title: str) -> str:
    if df.empty:
        return f"## {title}\n\n_No rows available._\n"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "-" if pd.isna(x) else f"{x:.4f}")
        else:
            display[col] = display[col].fillna("").astype(str)

    headers = list(display.columns)
    rows = display.astype(str).values.tolist()
    widths = [
        max(len(str(header)), *(len(row[i]) for row in rows))
        for i, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return f"## {title}\n\n" + "\n".join([header_line, sep_line, *row_lines]) + "\n"


def save_tolerance_plot(practical_df: pd.DataFrame, outdir: Path) -> None:
    if practical_df.empty:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[Plot] matplotlib unavailable, skipping tolerance plot")
        return

    thresholds = ["within_005", "within_010", "within_015", "within_020"]
    x = np.array([0.05, 0.10, 0.15, 0.20])
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in practical_df.iterrows():
        label = f"{row['model']} [{row['trained_target']}]"
        y = [row[t] for t in thresholds]
        ax.plot(x, y, marker="o", linewidth=1.7, label=label)
    ax.set_xlabel("Absolute error tolerance")
    ax.set_ylabel("Share within tolerance")
    ax.set_title("Practical raw-scale tolerance metrics")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "comparison_tolerance_plot.png", dpi=160)
    plt.close(fig)


def save_ordinal_report(practical_df: pd.DataFrame, outdir: Path) -> None:
    if practical_df.empty:
        (outdir / "comparison_ordinal_metrics.md").write_text(
            "# Ordinal Metrics\n\n_No raw-scale practical rows available._\n",
            encoding="utf-8",
        )
        return

    cols = [
        "model",
        "trained_target",
        "spearman_corr",
        "within_010",
        "within_020",
        "fixed_bin_acc",
        "fixed_bin_spearman",
        "fixed_bin_kendall_tau",
        "fixed_bin_qwk",
        "train_quantile_bin_spearman",
        "train_quantile_bin_kendall_tau",
        "train_quantile_bin_qwk",
    ]
    ordinal_df = practical_df[[c for c in cols if c in practical_df.columns]].copy()

    best = ordinal_df.iloc[0]
    md = [
        "# Ordinal Metrics for Strategic Utility",
        "",
        "These metrics complement exact regression metrics. They evaluate whether the model places a draft in a sensible strategic roaming zone rather than requiring an exact decimal score.",
        "",
        "- **Continuous Spearman** (`spearman_corr`) ranks exact raw scores and predictions.",
        "- **Bin Spearman** ranks ordinal bins, ignoring small differences inside the same bin.",
        "- **Bin Kendall tau** measures pairwise ordinal agreement between true and predicted bins.",
        "- **Quadratic Weighted Kappa (QWK)** measures ordinal agreement while penalizing distant bin errors more than adjacent bin errors.",
        "",
        "All ordinal metrics are computed on raw-scale bins. Quantile-trained models are inverse-transformed to raw before evaluation.",
        "",
        format_markdown_table(ordinal_df, "Ordinal Metric Comparison"),
        "",
        "## Reading",
        "",
        (
            f"The top row by fixed-bin QWK is **{best['model']}** "
            f"({best['trained_target']}). Its fixed-bin QWK is "
            f"{best['fixed_bin_qwk']:.4f}, compared with continuous Spearman "
            f"{best['spearman_corr']:.4f}. This should be read as strategic "
            "zone agreement, not exact score recovery."
        ),
        "",
        "If QWK or bin correlations are meaningfully higher than exact-score metrics, the model is more useful as an ordinal coach signal than as a precise regressor. If Champion Mean remains close to GBT, most of the ordinal signal is already carried by support champion identity.",
        "",
    ]
    (outdir / "comparison_ordinal_metrics.md").write_text("\n".join(md), encoding="utf-8")


def save_practical_outputs(rows: List[Dict[str, Any]], outdir: Path) -> None:
    practical_df = practical_table(rows)
    deciles_df = decile_table()

    practical_df.to_csv(outdir / "comparison_table_practical_raw.csv", index=False)
    deciles_df.to_csv(outdir / "comparison_deciles_raw.csv", index=False)

    (outdir / "comparison_table_practical_raw.md").write_text(
        "# Practical Raw-Scale Model Comparison\n\n"
        + format_markdown_table(practical_df, "Tolerance and Ordinal Metrics"),
        encoding="utf-8",
    )
    (outdir / "comparison_deciles_raw.md").write_text(
        "# Raw-Scale Prediction Deciles\n\n"
        + format_markdown_table(deciles_df, "Prediction Decile Diagnostics"),
        encoding="utf-8",
    )

    practical_payload = {
        "methodology": {
            "scale": "raw support_roam_score",
            "fixed_edges": PRACTICAL_CONTEXT.get("fixed_edges", np.array([])).tolist(),
            "train_quantile_edges": PRACTICAL_CONTEXT.get(
                "train_quantile_edges", np.array([])
            ).tolist(),
            "train_quantiles": PRACTICAL_CONTEXT.get("train_quantiles", {}),
            "note": (
                "Practical metrics are computed only on raw-scale rows. "
                "Quantile-trained models are inverse-transformed to raw first."
            ),
        },
        "practical_rows": practical_df.to_dict(orient="records"),
        "decile_rows": deciles_df.to_dict(orient="records"),
    }
    (outdir / "comparison_practical_results.json").write_text(
        json.dumps(practical_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_tolerance_plot(practical_df, outdir)
    save_ordinal_report(practical_df, outdir)


def save_plot(raw_df: pd.DataFrame, quantile_df: pd.DataFrame, outdir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("[Plot] matplotlib unavailable, skipping plot")
        return

    plot_df = pd.concat(
        [
            raw_df.assign(table="raw"),
            quantile_df.assign(table="quantile"),
        ],
        ignore_index=True,
    )
    if "spearman_corr" not in plot_df.columns:
        return
    plot_df = plot_df.dropna(subset=["spearman_corr"])
    if plot_df.empty:
        return

    labels = plot_df["model"] + " [" + plot_df["trained_target"] + "/" + plot_df["table"] + "]"
    fig_h = max(4.0, 0.38 * len(plot_df))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(labels, plot_df["spearman_corr"])
    ax.invert_yaxis()
    ax.set_xlabel("Spearman correlation on test")
    ax.set_title("Final model comparison")
    fig.tight_layout()
    fig.savefig(outdir / "comparison_spearman.png", dpi=160)
    plt.close(fig)


def attach_test_metadata(rows: List[Dict[str, Any]], test_path: str) -> None:
    resolved = str(Path(test_path).resolve())
    for row in rows:
        row["test_dataset"] = resolved
        row["metrics_source"] = "recomputed_from_predictions"


def format_metric_mean_std(values: pd.Series) -> str:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return ""
    if len(vals) == 1:
        return f"{float(vals.iloc[0]):.4f}"
    return f"{float(vals.mean()):.4f} +/- {float(vals.std(ddof=0)):.4f}"


def build_final_main_table(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    raw_rows = [
        row
        for row in rows
        if row.get("evaluation_scale") == "raw"
        and row.get("trained_target") == "raw"
        and row.get("model") in MAIN_BASELINE_MODELS.union(MAIN_LEARNED_MODELS)
    ]
    df = pd.DataFrame(raw_rows)
    if df.empty:
        return df

    required = FINAL_MAIN_COLUMNS + ["pearson_corr", "test_dataset", "metrics_source"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"[Final table] Missing required columns: {missing}")

    n_eval_values = sorted(df["n_eval"].dropna().unique().tolist())
    if len(n_eval_values) != 1:
        raise SystemExit(f"[Final table] n_eval mismatch across main rows: {n_eval_values}")

    test_values = sorted(df["test_dataset"].dropna().unique().tolist())
    if len(test_values) != 1:
        raise SystemExit(f"[Final table] test dataset mismatch across main rows: {test_values}")

    source_values = sorted(df["metrics_source"].dropna().unique().tolist())
    if source_values != ["recomputed_from_predictions"]:
        raise SystemExit(f"[Final table] Metrics were not all recomputed from predictions: {source_values}")

    order = {
        "Global Mean": 0,
        "Champion Mean": 1,
        "HistGBT": 2,
        "MLP OneHot": 3,
        "MLP Embed Shared": 4,
        "MLP Per-Role + Interactions": 5,
    }
    metric_cols = [
        "r2",
        "spearman_corr",
        "pearson_corr",
        "mae",
        "rmse",
        "pred_std",
        "within_010",
        "within_020",
    ]
    grouped_rows: List[Dict[str, Any]] = []
    for model, sub in df.groupby("model", sort=False):
        row: Dict[str, Any] = {
            "model": model,
            "n_eval": int(sub["n_eval"].iloc[0]),
            "n_seeds": int(sub["seed"].dropna().nunique()) if "seed" in sub.columns else 0,
        }
        for metric in metric_cols:
            row[metric] = format_metric_mean_std(sub[metric])
        row["_order"] = order.get(model, 99)
        grouped_rows.append(row)

    out = pd.DataFrame(grouped_rows).sort_values("_order").drop(columns=["_order"])
    return out[
        [
            "model",
            "r2",
            "spearman_corr",
            "pearson_corr",
            "mae",
            "rmse",
            "pred_std",
            "within_010",
            "within_020",
            "n_eval",
            "n_seeds",
        ]
    ]


def validate_main_rows_have_manifests(
    rows: List[Dict[str, Any]],
    audit_rows: List[Dict[str, Any]],
) -> None:
    audit_by_model = {row["model"]: row for row in audit_rows}
    for row in rows:
        model = row.get("model")
        if model not in MAIN_LEARNED_MODELS:
            continue
        audit = audit_by_model.get(model)
        if audit is None:
            raise SystemExit(f"[Final table] Missing protocol manifest audit for {model}.")
        if audit.get("manifest_feature_protocol_id") != MAIN_FEATURE_PROTOCOL_ID:
            raise SystemExit(f"[Final table] {model} lacks explicit main protocol manifest.")
        if audit.get("matches_main_feature_protocol") is not True:
            raise SystemExit(f"[Final table] {model} does not match main feature protocol.")


def load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def learned_manifest_columns(cfg: Dict[str, Any]) -> List[str]:
    if "input_feature_columns" in cfg:
        return list(cfg["input_feature_columns"])
    if "feature_columns" in cfg:
        return list(cfg["feature_columns"])
    return []


def audit_feature_protocol(model_keys: List[str], outdir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add_row(
        model_key: str,
        model_name: str,
        feature_columns: List[str],
        role: str,
        used_sample_weight: Optional[bool],
        manifest_feature_protocol_id: str = "",
        notes: str = "",
    ) -> None:
        feature_columns = list(feature_columns)
        matches_main = feature_columns == CANONICAL_MAIN_FEATURES
        has_manifest = (
            role not in {"main_learned_model", "secondary_hp_tuned"}
            or bool(feature_columns)
        )
        rows.append(
            {
                "model_key": model_key,
                "model": model_name,
                "comparison_role": role,
                "feature_protocol_id": (
                    MAIN_FEATURE_PROTOCOL_ID
                    if matches_main
                    else manifest_feature_protocol_id or role
                ),
                "manifest_feature_protocol_id": manifest_feature_protocol_id,
                "feature_count": len(feature_columns),
                "input_feature_columns": feature_columns,
                "matches_main_feature_protocol": matches_main,
                "sample_weight_column": "sample_weight",
                "used_sample_weight": used_sample_weight,
                "has_manifest": has_manifest,
                "notes": notes,
            }
        )

    if "baselines" in model_keys:
        add_row(
            "global_mean",
            "Global Mean",
            [],
            "baseline_no_features",
            True,
            manifest_feature_protocol_id="baseline_no_features",
        )
        add_row(
            "champion_mean",
            "Champion Mean",
            ["ally_utility_champion_id"],
            "baseline_support_champion_only",
            True,
            manifest_feature_protocol_id="baseline_support_champion_only",
        )

    if "gbt" in model_keys:
        cfg = load_json_if_exists(REPO_ROOT / "final" / "models" / "gbt" / "model_config.json")
        add_row(
            "gbt",
            "HistGBT",
            learned_manifest_columns(cfg),
            "main_learned_model",
            cfg.get("used_sample_weight"),
            manifest_feature_protocol_id=str(cfg.get("feature_protocol_id", "missing_manifest"))
            if cfg
            else "missing_manifest",
            notes="" if cfg else "model_config.json not found; retrain GBT before final comparison.",
        )

    for model_key, model_name, model_dir in [
        ("mlp_onehot", "MLP OneHot", "mlp_onehot"),
        ("mlp_embed", "MLP Embed Shared", "mlp_embed"),
        ("mlp_per_role", "MLP Per-Role + Interactions", "mlp_per_role"),
        ("mlp_per_role_tuned", "MLP Per-Role + Interactions HP Best", "mlp_per_role_tuned"),
    ]:
        if model_key not in model_keys:
            continue
        cfg = load_json_if_exists(REPO_ROOT / "final" / "models" / model_dir / "model_config.json")
        add_row(
            model_key,
            model_name,
            learned_manifest_columns(cfg),
            "main_learned_model" if model_key != "mlp_per_role_tuned" else "secondary_hp_tuned",
            cfg.get("used_sample_weight") if cfg else None,
            manifest_feature_protocol_id=str(cfg.get("feature_protocol_id", "missing_manifest"))
            if cfg
            else "missing_manifest",
            notes="" if cfg else "model_config.json not found.",
        )

    if "gbt_enriched" in model_keys:
        cfg = load_json_if_exists(REPO_ROOT / "final" / "models" / "gbt_enriched" / "model_config.json")
        add_row(
            "gbt_enriched",
            "HistGBT + Archetypes",
            learned_manifest_columns(cfg),
            "secondary_enriched_features",
            cfg.get("used_sample_weight"),
            manifest_feature_protocol_id=str(
                cfg.get("feature_protocol_id", "secondary_enriched_features")
            )
            if cfg
            else "missing_manifest",
            notes="Excluded from main table because it adds archetype/class features.",
        )

    if "gbt_interactions" in model_keys:
        cfg = load_json_if_exists(REPO_ROOT / "final" / "models" / "gbt_interactions" / "model_config.json")
        add_row(
            "gbt_interactions",
            "HistGBT + Pair TE",
            learned_manifest_columns(cfg),
            "secondary_target_encoded_features",
            cfg.get("used_sample_weight"),
            manifest_feature_protocol_id=str(
                cfg.get("feature_protocol_id", "secondary_target_encoded_features")
            )
            if cfg
            else "missing_manifest",
            notes="Excluded from main table because it adds target-encoded interaction features.",
        )

    if "residual_interactions" in model_keys:
        cfg = load_json_if_exists(
            REPO_ROOT / "final" / "models" / "residual_interactions" / "model_config.json"
        )
        add_row(
            "residual_interactions",
            "Smoothed Support Mean + Residual Context GBT",
            list(cfg.get("categorical_columns", [])) + list(cfg.get("numeric_columns", [])),
            "secondary_residual_diagnostic",
            cfg.get("used_sample_weight") if cfg else None,
            manifest_feature_protocol_id=str(
                cfg.get("feature_protocol_id", "secondary_residual_diagnostic")
            )
            if cfg
            else "missing_manifest",
            notes=(
                "Excluded from main table because it decomposes prediction into a "
                "support mean plus a residual context model."
            ),
        )

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "feature_protocol_audit.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                **row,
                "input_feature_columns": ", ".join(row["input_feature_columns"]),
            }
            for row in rows
        ]
    ).to_csv(outdir / "feature_protocol_audit.csv", index=False)

    print("\n--- Feature Protocol Audit ---")
    for row in rows:
        print(
            f"[Features] {row['model']}: count={row['feature_count']} "
            f"role={row['comparison_role']} used_sample_weight={row['used_sample_weight']}"
        )
        if row["input_feature_columns"]:
            print("           " + ", ".join(row["input_feature_columns"]))
        if row["notes"]:
            print("           " + row["notes"])

    bad = [
        row
        for row in rows
        if row["comparison_role"] == "main_learned_model"
        and not row["matches_main_feature_protocol"]
    ]
    missing_manifest = [
        row
        for row in rows
        if row["comparison_role"] == "main_learned_model"
        and (
            row["manifest_feature_protocol_id"] != MAIN_FEATURE_PROTOCOL_ID
            or not row["has_manifest"]
        )
    ]
    missing_weights = [
        row
        for row in rows
        if row["comparison_role"] in {"main_learned_model", "baseline_no_features", "baseline_support_champion_only"}
        and row["used_sample_weight"] is not True
    ]
    if missing_manifest:
        names = ", ".join(row["model"] for row in missing_manifest)
        raise SystemExit(
            "[Feature audit] Main comparison has models without an explicit "
            f"{MAIN_FEATURE_PROTOCOL_ID} manifest: {names}. Retrain or regenerate those artifacts."
        )
    if bad:
        names = ", ".join(row["model"] for row in bad)
        raise SystemExit(
            "[Feature audit] Main learned comparison would mix feature protocols: "
            f"{names}. Retrain these models with 10 champion IDs + side."
        )
    if missing_weights:
        names = ", ".join(row["model"] for row in missing_weights)
        raise SystemExit(
            "[Feature audit] Main comparison has models without confirmed sample_weight: "
            f"{names}."
        )
    return rows


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_train = pd.read_parquet(args.train)
    df_test = pd.read_parquet(args.test)
    transformer = joblib.load(args.quantile_transformer) if Path(args.quantile_transformer).exists() else None
    init_practical_context(df_train[TARGET_COL].to_numpy())

    print(f"[Data] train={len(df_train):,}  test={len(df_test):,}")
    print(f"[Target] test raw std={df_test[TARGET_COL].std():.4f}")
    if QUANTILE_COL in df_test.columns:
        print(f"[Target] test quantile std={df_test[QUANTILE_COL].std():.4f}")

    model_keys = selected_model_keys(args)
    audit_rows = audit_feature_protocol(model_keys, outdir)

    rows: List[Dict[str, Any]] = []

    if "baselines" in model_keys:
        print("\n--- Baselines ---")
        rows.extend(eval_mean_baselines(df_train, df_test, transformer))

    if "gbt" in model_keys:
        print("\n--- HistGBT ---")
        rows.extend(eval_gbt(df_test, len(df_train), transformer))

    secondary_rows: List[Dict[str, Any]] = []
    if "gbt_enriched" in model_keys:
        print("\n--- HistGBT + Archetypes ---")
        secondary_rows.extend(
            eval_gbt_family(
                df_test=df_test,
                n_train=len(df_train),
                transformer=transformer,
                model_dir=REPO_ROOT / "final" / "models" / "gbt_enriched",
                model_name="HistGBT + Archetypes",
            )
        )

    if "gbt_interactions" in model_keys:
        print("\n--- HistGBT + Pair Target Encodings ---")
        secondary_rows.extend(eval_gbt_interactions(df_test, len(df_train), transformer))

    if "residual_interactions" in model_keys:
        print("\n--- Smoothed Support Mean + Residual Context GBT ---")
        secondary_rows.extend(eval_residual_interactions(df_test, len(df_train)))

    if "mlp_onehot" in model_keys:
        print("\n--- MLP OneHot ---")
        rows.extend(eval_mlp(df_test, "onehot", len(df_train), transformer, args.batch_size))

    if "mlp_embed" in model_keys:
        print("\n--- MLP Embed ---")
        rows.extend(eval_mlp(df_test, "embed", len(df_train), transformer, args.batch_size))

    if "mlp_per_role" in model_keys:
        print("\n--- MLP Per-Role + Interactions ---")
        rows.extend(eval_mlp(df_test, "per_role", len(df_train), transformer, args.batch_size))

    if "mlp_per_role_tuned" in model_keys:
        print("\n--- MLP Per-Role + Interactions HP Best ---")
        secondary_rows.extend(
            eval_mlp(df_test, "per_role_tuned", len(df_train), transformer, args.batch_size)
        )

    if "ceiling" in model_keys:
        add_ceiling_reference(secondary_rows, len(df_test))

    attach_test_metadata(rows, args.test)
    attach_test_metadata(secondary_rows, args.test)
    validate_main_rows_have_manifests(rows, audit_rows)
    final_main_df = build_final_main_table(rows)

    all_df = pd.DataFrame(rows)
    raw_df = sorted_table(rows, "raw")
    quantile_df = sorted_table(rows, "quantile")
    secondary_df = pd.DataFrame(secondary_rows)
    secondary_raw_df = sorted_table(secondary_rows, "raw")
    secondary_quantile_df = sorted_table(secondary_rows, "quantile")
    secondary_residual_df = sorted_table(secondary_rows, "residual")
    residual_diagnostics_df = pd.DataFrame(
        [row for row in secondary_rows if row.get("diagnostic_family") == "support_residual"]
    )
    if not residual_diagnostics_df.empty:
        residual_cols = [
            "model",
            "trained_target",
            "evaluation_scale",
            "r2",
            "spearman_corr",
            "mae",
            "pred_std",
            "target_std",
            "compression_ratio",
            "support_effect_r2",
            "support_effect_spearman_corr",
            "r2_lift_over_support_effect",
            "spearman_lift_over_support_effect",
            "residual_r2",
            "residual_spearman_corr",
            "n_eval",
            "notes",
        ]
        residual_diagnostics_df = residual_diagnostics_df[
            [col for col in residual_cols if col in residual_diagnostics_df.columns]
        ]

    all_df.to_csv(outdir / "comparison_all_rows.csv", index=False)
    final_main_df.to_csv(outdir / "final_main_table_raw.csv", index=False)
    raw_df.to_csv(outdir / "comparison_table_raw.csv", index=False)
    quantile_df.to_csv(outdir / "comparison_table_quantile.csv", index=False)
    secondary_df.to_csv(outdir / "comparison_secondary_all_rows.csv", index=False)
    secondary_raw_df.to_csv(outdir / "comparison_secondary_table_raw.csv", index=False)
    secondary_quantile_df.to_csv(outdir / "comparison_secondary_table_quantile.csv", index=False)
    secondary_residual_df.to_csv(outdir / "comparison_secondary_table_residual.csv", index=False)
    residual_diagnostics_df.to_csv(outdir / "residual_context_diagnostics.csv", index=False)
    (outdir / "comparison_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (outdir / "comparison_results_with_secondary.json").write_text(
        json.dumps(
            {
                "main_rows": rows,
                "secondary_rows": secondary_rows,
                "excluded_from_main": [
                    {
                        "model": "HistGBT + Archetypes",
                        "reason": "Adds champion archetype/class features beyond 10 champion IDs + side.",
                    },
                    {
                        "model": "HistGBT + Pair TE",
                        "reason": "Adds target-encoded pair interaction features.",
                    },
                    {
                        "model": "Smoothed Support Mean + Residual Context GBT",
                        "reason": (
                            "Diagnostic additive decomposition into support mean "
                            "and residual context signal."
                        ),
                    },
                    {
                        "model": "MLP Per-Role + Interactions HP Best",
                        "reason": "Hyperparameter-search result; not retrained by the default master run.",
                    },
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    raw_md = format_markdown_table(raw_df, "Table A - Raw Scale (Test)")
    final_main_md = format_markdown_table(final_main_df, "Main Table - Common Protocol Raw Models")
    quantile_md = format_markdown_table(quantile_df, "Table B - Quantile Scale (Test)")
    secondary_raw_md = format_markdown_table(secondary_raw_df, "Secondary A - Raw Scale (Test)")
    secondary_quantile_md = format_markdown_table(
        secondary_quantile_df, "Secondary B - Quantile Scale (Test)"
    )
    secondary_residual_md = format_markdown_table(
        secondary_residual_df,
        "Secondary C - Residual-Scale Diagnostics (Test)",
    )
    residual_diagnostics_md = format_markdown_table(
        residual_diagnostics_df,
        "Support Residual Diagnostic Rows",
    )
    md = (
        "# Final Model Comparison (Test Set)\n\n"
        "Main tables use the fair feature protocol for learned models: "
        "10 champion IDs + side. Global Mean and Champion Mean are retained as "
        "lower-information baselines and are labelled in the feature audit.\n\n"
        + final_main_md
        + "\n"
        + raw_md
        + "\n"
        + quantile_md
        + "\n"
        + secondary_raw_md
        + "\n"
        + secondary_quantile_md
        + "\n"
        + secondary_residual_md
        + "\n"
        + residual_diagnostics_md
    )
    (outdir / "comparison_tables.md").write_text(md, encoding="utf-8")
    (outdir / "final_main_table_raw.md").write_text(
        "# Final Main Table - Common Protocol\n\n"
        "Metrics are recomputed from predictions on the same held-out test split. "
        "Rows are restricted to raw-target models trained under the common input "
        "protocol for learned models: 10 champion IDs + side. Practical columns "
        "`within_010` and `within_020` are the share of predictions within +/-0.10 "
        "and +/-0.20 absolute error.\n\n"
        + final_main_md,
        encoding="utf-8",
    )
    save_plot(raw_df, quantile_df, outdir)
    save_practical_outputs(rows, outdir)

    print("\n" + "=" * 80)
    print("FINAL COMPARISON - RAW SCALE (TEST)")
    print("=" * 80)
    print(raw_df.to_string(index=False) if not raw_df.empty else "No raw rows.")
    print("\n" + "=" * 80)
    print("FINAL COMPARISON - QUANTILE SCALE (TEST)")
    print("=" * 80)
    print(quantile_df.to_string(index=False) if not quantile_df.empty else "No quantile rows.")
    print(f"\n[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
