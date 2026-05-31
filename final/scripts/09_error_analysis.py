#!/usr/bin/env python3
"""
09_error_analysis.py -- Qualitative error-analysis tables for base HistGBT.

This script exports the largest held-out test errors with draft context, but it
does not infer in-game causes. The qualitative note column is intentionally left
blank so selected cases can be annotated after manual review.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEST = str(REPO_ROOT / "final" / "data" / "training" / "test.parquet")
DEFAULT_MODEL_DIR = str(REPO_ROOT / "final" / "models" / "gbt")
DEFAULT_COMPARISON = str(
    REPO_ROOT / "final" / "analysis" / "model_comparison" / "comparison_table_raw.csv"
)
DEFAULT_OUTDIR = str(REPO_ROOT / "final" / "analysis" / "error_analysis")

TARGET_COL = "support_roam_score"
MISSING_TOKEN = "__MISSING__"
SCORE_BINS = [0.0, 0.25, 0.50, 0.75, 1.0]
SCORE_LABELS = ["very_low", "low_mid", "high_mid", "very_high"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export test error diagnostics for base HistGBT.")
    p.add_argument("--test", default=DEFAULT_TEST)
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--comparison-table", default=DEFAULT_COMPARISON)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--top-n", type=int, default=20)
    return p.parse_args()


def encode_features(df: pd.DataFrame, feature_cols: List[str], encoder: Any) -> np.ndarray:
    raw = df[feature_cols].copy()
    for col in feature_cols:
        raw[col] = raw[col].fillna(MISSING_TOKEN).astype(str)
    return encoder.transform(raw)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
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
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "pearson_corr": pearson,
        "spearman_corr": spearman,
        "pred_std": pred_std,
        "target_std": target_std,
        "compression_ratio": pred_std / target_std if target_std > 0 else float("nan"),
    }


def add_score_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["actual_bin"] = pd.cut(
        out["actual"].clip(SCORE_BINS[0], SCORE_BINS[-1]),
        bins=SCORE_BINS,
        labels=SCORE_LABELS,
        include_lowest=True,
    )
    out["prediction_bin"] = pd.cut(
        out["prediction"].clip(SCORE_BINS[0], SCORE_BINS[-1]),
        bins=SCORE_BINS,
        labels=SCORE_LABELS,
        include_lowest=True,
    )
    return out


def ordered_columns(df: pd.DataFrame) -> List[str]:
    preferred = [
        "error_rank",
        "match_id",
        "team_id",
        "side",
        "patch",
        "game_version",
        "ally_utility_champion_name",
        "ally_bottom_champion_name",
        "enemy_utility_champion_name",
        "enemy_bottom_champion_name",
        "prediction",
        "actual",
        "signed_error",
        "abs_error",
        "prediction_bin",
        "actual_bin",
        "qualitative_note",
    ]
    draft_names = [
        c for c in df.columns
        if c.endswith("_champion_name") and c not in preferred
    ]
    rest = [c for c in df.columns if c not in preferred and c not in draft_names]
    return [c for c in preferred if c in df.columns] + draft_names + rest


def summarize_errors(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("abs_error", "size"),
            mean_abs_error=("abs_error", "mean"),
            median_abs_error=("abs_error", "median"),
            p90_abs_error=("abs_error", lambda s: float(np.quantile(s, 0.90))),
            mean_signed_error=("signed_error", "mean"),
            mean_prediction=("prediction", "mean"),
            mean_actual=("actual", "mean"),
        )
        .reset_index()
        .sort_values(["mean_abs_error", "n"], ascending=[False, False])
    )


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            display[col] = display[col].fillna("").astype(str)
    headers = list(display.columns)
    rows = display.astype(str).values.tolist()
    widths = [
        max(len(str(header)), *(len(row[i]) for row in rows))
        for i, header in enumerate(headers)
    ]
    header = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def load_reference_metrics(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    row = table[
        (table["model"] == "HistGBT")
        & (table["trained_target"] == "raw")
        & (table["evaluation_scale"] == "raw")
    ]
    if row.empty:
        return {}
    keys = ["mse", "rmse", "mae", "r2", "pearson_corr", "spearman_corr"]
    return {key: float(row.iloc[0][key]) for key in keys if key in row.columns}


def metric_deltas(metrics: Dict[str, float], reference: Dict[str, float]) -> Dict[str, float]:
    return {
        key: float(metrics[key] - reference[key])
        for key in reference
        if key in metrics and math.isfinite(metrics[key]) and math.isfinite(reference[key])
    }


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    model = joblib.load(model_dir / "gbt_model_raw.joblib")
    preprocess = joblib.load(model_dir / "preprocess.joblib")
    encoder = preprocess["encoder"]
    feature_cols: List[str] = preprocess["feature_columns"]

    df_test = pd.read_parquet(args.test)
    X_test = encode_features(df_test, feature_cols, encoder)
    predictions = model.predict(X_test)
    actual = df_test[TARGET_COL].to_numpy(dtype=np.float64)

    result = df_test.copy()
    result["prediction"] = predictions
    result["actual"] = actual
    result["signed_error"] = result["prediction"] - result["actual"]
    result["abs_error"] = result["signed_error"].abs()
    result = add_score_bins(result)
    result["error_rank"] = result["abs_error"].rank(method="first", ascending=False).astype(int)
    result["qualitative_note"] = ""
    result = result.sort_values("error_rank")

    output_cols = ordered_columns(result)
    result[output_cols].to_csv(outdir / "test_predictions_with_errors.csv", index=False)

    top = result.head(args.top_n).copy()
    top[output_cols].to_csv(outdir / "top_abs_errors.csv", index=False)

    summarize_errors(result, ["ally_utility_champion_name"]).to_csv(
        outdir / "error_summary_by_support.csv", index=False
    )
    summarize_errors(result, ["ally_bottom_champion_name"]).to_csv(
        outdir / "error_summary_by_adc.csv", index=False
    )
    summarize_errors(result, ["ally_utility_champion_name", "ally_bottom_champion_name"]).to_csv(
        outdir / "error_summary_by_support_adc.csv", index=False
    )

    md_cols = [
        "match_id",
        "team_id",
        "side",
        "patch",
        "ally_utility_champion_name",
        "ally_bottom_champion_name",
        "enemy_utility_champion_name",
        "enemy_bottom_champion_name",
        "prediction",
        "actual",
        "abs_error",
        "qualitative_note",
    ]
    md = [
        "# Top Absolute Test Errors - HistGBT Base",
        "",
        "These cases are intentionally exported as draft-context examples only. Add manual qualitative notes after inspecting the match timeline or replay context.",
        "",
        markdown_table(top[[c for c in md_cols if c in top.columns]]),
        "",
        "## Suggested Reading",
        "",
        "Use 2-3 cases to illustrate that draft creates a measurable predisposition, while early-game execution, lane state, recalls, deaths, and jungle pressure add variance not visible in pre-game features.",
        "",
    ]
    (outdir / "top_error_cases.md").write_text("\n".join(md), encoding="utf-8")

    metrics = regression_metrics(actual, predictions)
    reference = load_reference_metrics(Path(args.comparison_table))
    deltas = metric_deltas(metrics, reference)
    sorted_abs = top["abs_error"].to_numpy()
    meta: Dict[str, Any] = {
        "model_path": str((model_dir / "gbt_model_raw.joblib").resolve()),
        "preprocess_path": str((model_dir / "preprocess.joblib").resolve()),
        "test_path": str(Path(args.test).resolve()),
        "comparison_table_path": str(Path(args.comparison_table).resolve()),
        "outdir": str(outdir.resolve()),
        "target": TARGET_COL,
        "top_n": args.top_n,
        "n_eval": int(len(result)),
        "n_top_errors": int(len(top)),
        "metrics": metrics,
        "reference_metrics": reference,
        "metric_deltas_vs_reference": deltas,
        "metrics_match_reference": all(abs(v) <= 1e-10 for v in deltas.values()) if deltas else None,
        "top_abs_errors_descending": bool(np.all(sorted_abs[:-1] >= sorted_abs[1:])),
        "score_bins": SCORE_BINS,
        "score_bin_labels": SCORE_LABELS,
        "note": (
            "qualitative_note is intentionally blank. This output supports manual "
            "case analysis; it does not infer in-game causes from draft features."
        ),
    }
    (outdir / "error_analysis_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"[Metrics] R2={metrics['r2']:.6f} Spearman={metrics['spearman_corr']:.6f} "
        f"MAE={metrics['mae']:.6f}"
    )
    if deltas:
        max_delta = max(abs(v) for v in deltas.values())
        print(f"[Reference check] max_delta={max_delta:.12f}")
    print(f"[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
