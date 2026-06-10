#!/usr/bin/env python3
"""
28_chaos_weight_sweep.py -- Sweep sample_weight values for chaotic games.

This experiment answers a narrow methodological question: whether the 0.2
weight assigned to chaos_flag rows is an arbitrary heuristic or a value that is
supported by validation performance.

Protocol:
  - Keep the final main feature protocol: 10 champion IDs + side.
  - For each candidate chaos weight, rebuild sample_weight in memory:
        clean rows -> 1.0
        chaotic rows -> candidate weight
  - Train HistGradientBoostingRegressor on train.
  - Select the best weight using validation metrics only.
  - Report test metrics for all weights as a post-selection diagnostic.

The script does not overwrite the training parquet files or model artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = REPO_ROOT / "final" / "data" / "training" / "train.parquet"
DEFAULT_VAL = REPO_ROOT / "final" / "data" / "training" / "val.parquet"
DEFAULT_TEST = REPO_ROOT / "final" / "data" / "training" / "test.parquet"
DEFAULT_OUTDIR = REPO_ROOT / "logs"

TARGET_COL = "support_roam_score"
CHAOS_COL = "chaos_flag"
ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")
FEATURE_COLS = [f"{side}_{role}_champion_id" for side in SIDES for role in ROLE_KEYS] + ["side"]
DEFAULT_WEIGHTS = [round(x, 1) for x in np.arange(0.0, 1.0 + 0.0001, 0.1)]
DEFAULT_SEEDS = [42, 123, 456]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep chaos sample_weight values for the final HistGBT protocol."
    )
    p.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--val", type=Path, default=DEFAULT_VAL)
    p.add_argument("--test", type=Path, default=DEFAULT_TEST)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=DEFAULT_WEIGHTS,
        help="Candidate weights for chaos_flag rows. Clean rows always use 1.0.",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds. Metrics are aggregated across seeds.",
    )
    p.add_argument(
        "--selection-metric",
        choices=["spearman_corr", "r2", "mae", "rmse"],
        default="spearman_corr",
        help="Validation metric used to select the best chaos weight.",
    )
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--min-samples-leaf", type=int, default=50)
    p.add_argument("--max-leaf-nodes", type=int, default=31)
    p.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Optional smoke-test limit. Do not use for final results.",
    )
    p.add_argument(
        "--limit-val",
        type=int,
        default=None,
        help="Optional smoke-test limit. Do not use for final results.",
    )
    p.add_argument(
        "--limit-test",
        type=int,
        default=None,
        help="Optional smoke-test limit. Do not use for final results.",
    )
    return p.parse_args()


def validate_inputs(df: pd.DataFrame, split_name: str) -> None:
    missing = [col for col in [TARGET_COL, CHAOS_COL, *FEATURE_COLS] if col not in df.columns]
    if missing:
        raise SystemExit(f"[{split_name}] Missing required columns: {missing}")


def load_split(path: Path, split_name: str, limit: int | None) -> pd.DataFrame:
    print(f"[Load] {split_name}: {path}")
    df = pd.read_parquet(path)
    if limit is not None:
        df = df.head(limit).copy()
        print(f"       smoke-test limit applied: {len(df):,} rows")
    validate_inputs(df, split_name)
    print(
        f"       rows={len(df):,}  chaos_rate={100.0 * df[CHAOS_COL].mean():.2f}%  "
        f"target_mean={df[TARGET_COL].mean():.4f}"
    )
    return df


def prepare_features(
    df_train: pd.DataFrame,
    eval_frames: Sequence[pd.DataFrame],
) -> tuple[np.ndarray, List[np.ndarray], OrdinalEncoder, List[bool]]:
    x_train_raw = df_train[FEATURE_COLS].copy()
    x_eval_raw = [df[FEATURE_COLS].copy() for df in eval_frames]

    for col in FEATURE_COLS:
        x_train_raw[col] = x_train_raw[col].fillna("__MISSING__").astype(str)
        for x_raw in x_eval_raw:
            x_raw[col] = x_raw[col].fillna("__MISSING__").astype(str)

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float32,
    )
    x_train = encoder.fit_transform(x_train_raw)
    x_eval = [encoder.transform(x_raw) for x_raw in x_eval_raw]
    categorical_mask = [True] * len(FEATURE_COLS)
    return x_train, x_eval, encoder, categorical_mask


def sample_weight_for(df: pd.DataFrame, chaos_weight: float) -> np.ndarray:
    if chaos_weight < 0:
        raise ValueError("chaos_weight must be non-negative")
    return np.where(df[CHAOS_COL].to_numpy(dtype=bool), chaos_weight, 1.0).astype(np.float32)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pred_std = float(np.std(y_pred))
    target_std = float(np.std(y_true))
    if target_std > 1e-12 and pred_std > 1e-12:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        sp = spearmanr(y_true, y_pred, nan_policy="omit")
        spearman = float(sp.correlation) if sp.correlation is not None else float("nan")
    else:
        pearson = float("nan")
        spearman = float("nan")
    abs_error = np.abs(y_true - y_pred)
    return {
        "r2": float(r2),
        "spearman_corr": spearman,
        "pearson_corr": pearson,
        "mae": mae,
        "rmse": math.sqrt(mse),
        "pred_std": pred_std,
        "target_std": target_std,
        "within_010": float(np.mean(abs_error <= 0.10)),
        "within_020": float(np.mean(abs_error <= 0.20)),
    }


def add_eval_rows(
    rows: List[Dict[str, Any]],
    split_name: str,
    df_eval: pd.DataFrame,
    y_pred: np.ndarray,
    chaos_weight: float,
    seed: int,
    training_seconds: float,
) -> None:
    y_true = df_eval[TARGET_COL].to_numpy(dtype=np.float64)
    masks = {
        "all": np.ones(len(df_eval), dtype=bool),
        "clean": ~df_eval[CHAOS_COL].to_numpy(dtype=bool),
        "chaotic": df_eval[CHAOS_COL].to_numpy(dtype=bool),
    }
    for subset, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        rows.append(
            {
                "chaos_weight": float(chaos_weight),
                "seed": int(seed),
                "eval_split": split_name,
                "subset": subset,
                "n_eval": int(mask.sum()),
                "training_seconds": float(training_seconds),
                **regression_metrics(y_true[mask], y_pred[mask]),
            }
        )


def train_one(
    x_train: np.ndarray,
    y_train: np.ndarray,
    categorical_mask: List[bool],
    sample_weight: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[HistGradientBoostingRegressor, float]:
    model = HistGradientBoostingRegressor(
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_samples_leaf=args.min_samples_leaf,
        max_leaf_nodes=args.max_leaf_nodes,
        categorical_features=categorical_mask,
        random_state=seed,
        verbose=0,
    )
    t0 = time.time()
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model, time.time() - t0


def aggregate_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    metric_cols = [
        "r2",
        "spearman_corr",
        "pearson_corr",
        "mae",
        "rmse",
        "pred_std",
        "target_std",
        "within_010",
        "within_020",
        "training_seconds",
    ]
    grouped = (
        df.groupby(["chaos_weight", "eval_split", "subset"], as_index=False)
        .agg(
            n_eval=("n_eval", "first"),
            n_seeds=("seed", "nunique"),
            **{f"{col}_mean": (col, "mean") for col in metric_cols},
            **{f"{col}_std": (col, "std") for col in metric_cols},
        )
        .sort_values(["eval_split", "subset", "chaos_weight"])
    )
    return grouped


def select_best(agg: pd.DataFrame, metric: str) -> pd.Series:
    col = f"{metric}_mean"
    candidates = agg[(agg["eval_split"] == "val") & (agg["subset"] == "all")].copy()
    if candidates.empty:
        raise SystemExit("[Selection] No validation rows found.")
    if col not in candidates.columns:
        raise SystemExit(f"[Selection] Unknown metric column: {col}")

    ascending = metric in {"mae", "rmse"}
    candidates = candidates.sort_values(
        [col, "chaos_weight"],
        ascending=[ascending, True],
        kind="mergesort",
    )
    return candidates.iloc[0]


def fmt_mean_std(row: pd.Series, metric: str, decimals: int = 4) -> str:
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def write_markdown_summary(
    outdir: Path,
    agg: pd.DataFrame,
    best: pd.Series,
    args: argparse.Namespace,
) -> None:
    val_all = agg[(agg["eval_split"] == "val") & (agg["subset"] == "all")].copy()
    test_all = agg[(agg["eval_split"] == "test") & (agg["subset"] == "all")].copy()
    val_all = val_all.sort_values(
        [f"{args.selection_metric}_mean", "chaos_weight"],
        ascending=[args.selection_metric in {"mae", "rmse"}, True],
        kind="mergesort",
    )
    test_all = test_all.sort_values("chaos_weight")

    def table(df: pd.DataFrame) -> str:
        lines = [
            "| chaos_weight | R² | Spearman | MAE | within ±0.10 | within ±0.20 |",
            "| :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for _, row in df.iterrows():
            lines.append(
                "| "
                f"{row['chaos_weight']:.2f} | "
                f"{fmt_mean_std(row, 'r2')} | "
                f"{fmt_mean_std(row, 'spearman_corr')} | "
                f"{fmt_mean_std(row, 'mae')} | "
                f"{fmt_mean_std(row, 'within_010')} | "
                f"{fmt_mean_std(row, 'within_020')} |"
            )
        return "\n".join(lines)

    payload = (
        "# Chaos Weight Sweep\n\n"
        "This experiment sweeps the sample weight assigned to `chaos_flag` rows. "
        "Clean rows always keep weight 1.0. The selected value is chosen from "
        "validation metrics only; test metrics are reported as a diagnostic.\n\n"
        f"- Selection metric: `{args.selection_metric}` on validation/all rows\n"
        f"- Best chaos weight: **{best['chaos_weight']:.2f}**\n"
        f"- Validation {args.selection_metric}: "
        f"**{best[f'{args.selection_metric}_mean']:.4f}**\n"
        f"- Seeds: {', '.join(str(s) for s in args.seeds)}\n"
        f"- Feature protocol: 10 champion IDs + side\n\n"
        "## Validation Results\n\n"
        + table(val_all)
        + "\n\n"
        "## Test Diagnostics\n\n"
        + table(test_all)
        + "\n"
    )
    (outdir / "chaos_weight_sweep_summary.md").write_text(payload, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df_train = load_split(args.train, "train", args.limit_train)
    df_val = load_split(args.val, "val", args.limit_val)
    df_test = load_split(args.test, "test", args.limit_test)

    x_train, (x_val, x_test), encoder, categorical_mask = prepare_features(
        df_train, [df_val, df_test]
    )
    y_train = df_train[TARGET_COL].to_numpy(dtype=np.float32)

    rows: List[Dict[str, Any]] = []
    for chaos_weight in args.weights:
        sample_weight = sample_weight_for(df_train, chaos_weight)
        print(
            f"\n[Sweep] chaos_weight={chaos_weight:.3f}  "
            f"effective_train_weight_mean={sample_weight.mean():.4f}"
        )
        for seed in args.seeds:
            model, elapsed = train_one(
                x_train=x_train,
                y_train=y_train,
                categorical_mask=categorical_mask,
                sample_weight=sample_weight,
                args=args,
                seed=seed,
            )
            val_pred = model.predict(x_val)
            test_pred = model.predict(x_test)
            val_all_metrics = regression_metrics(
                df_val[TARGET_COL].to_numpy(dtype=np.float64), val_pred
            )
            test_all_metrics = regression_metrics(
                df_test[TARGET_COL].to_numpy(dtype=np.float64), test_pred
            )
            add_eval_rows(rows, "val", df_val, val_pred, chaos_weight, seed, elapsed)
            add_eval_rows(rows, "test", df_test, test_pred, chaos_weight, seed, elapsed)
            print(
                f"  seed={seed}  train_time={elapsed:.1f}s  "
                f"val_spearman={val_all_metrics['spearman_corr']:.4f}  "
                f"test_spearman={test_all_metrics['spearman_corr']:.4f}"
            )

    per_seed_df = pd.DataFrame(rows)
    agg_df = aggregate_rows(rows)
    best = select_best(agg_df, args.selection_metric)

    per_seed_df.to_csv(args.outdir / "chaos_weight_sweep_per_seed.csv", index=False)
    agg_df.to_csv(args.outdir / "chaos_weight_sweep_summary.csv", index=False)
    joblib.dump(
        {
            "encoder": encoder,
            "feature_columns": FEATURE_COLS,
            "feature_protocol_id": "draft_10_champions_side",
        },
        args.outdir / "preprocess.joblib",
    )
    metadata = {
        "script": "28_chaos_weight_sweep.py",
        "selection_metric": args.selection_metric,
        "best_chaos_weight": float(best["chaos_weight"]),
        "best_validation_row": best.to_dict(),
        "weights": [float(w) for w in args.weights],
        "seeds": [int(s) for s in args.seeds],
        "model_params": {
            "max_iter": args.max_iter,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "min_samples_leaf": args.min_samples_leaf,
            "max_leaf_nodes": args.max_leaf_nodes,
        },
        "feature_columns": FEATURE_COLS,
        "target_col": TARGET_COL,
        "chaos_col": CHAOS_COL,
        "limits": {
            "limit_train": args.limit_train,
            "limit_val": args.limit_val,
            "limit_test": args.limit_test,
        },
    }
    (args.outdir / "chaos_weight_sweep_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown_summary(args.outdir, agg_df, best, args)

    print("\n" + "=" * 80)
    print("CHAOS WEIGHT SWEEP - VALIDATION SELECTION")
    print("=" * 80)
    print(
        f"Best weight by val/{args.selection_metric}: "
        f"{best['chaos_weight']:.2f} "
        f"({best[f'{args.selection_metric}_mean']:.4f})"
    )
    print(f"[Saved] {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
