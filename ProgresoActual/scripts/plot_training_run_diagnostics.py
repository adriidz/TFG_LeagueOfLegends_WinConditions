#!/usr/bin/env python3
"""
Create diagnostics for one support MLP training run.

Expected inputs in a run directory:
- history.csv
- metrics.json
- validation_predictions.parquet
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot diagnostics for a support MLP training run.")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--outdir", default=None, help="Defaults to <run-dir>/diagnostics.")
    p.add_argument("--target-col", default="support_roam_score")
    p.add_argument("--bins", type=int, default=20)
    return p.parse_args()


def find_prediction_columns(df: pd.DataFrame, target_col: str) -> Tuple[str, str]:
    preferred_true = f"true_{target_col}"
    preferred_pred = f"pred_{target_col}"
    if preferred_true in df.columns and preferred_pred in df.columns:
        return preferred_true, preferred_pred
    true_cols = [c for c in df.columns if c.startswith("true_")]
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    if not true_cols or not pred_cols:
        raise SystemExit("No true_/pred_ columns found in validation_predictions.parquet")
    return true_cols[0], pred_cols[0]


def save_loss_curve(history: pd.DataFrame, out_path: str) -> None:
    if history.empty or "epoch" not in history.columns:
        return
    plt.figure(figsize=(8, 5))
    if "train_mse_loss" in history.columns:
        plt.plot(history["epoch"], history["train_mse_loss"], label="train MSE", linewidth=2)
    if "val_mse_loss" in history.columns:
        plt.plot(history["epoch"], history["val_mse_loss"], label="val MSE", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training and validation loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_true_vs_pred(df: pd.DataFrame, true_col: str, pred_col: str, out_path: str) -> None:
    work = df[[true_col, pred_col]].dropna()
    if work.empty:
        return
    plt.figure(figsize=(6, 6))
    plt.scatter(work[true_col], work[pred_col], s=12, alpha=0.35, color="#276fbf")
    lo = float(min(work[true_col].min(), work[pred_col].min(), 0.0))
    hi = float(max(work[true_col].max(), work[pred_col].max(), 1.0))
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.xlabel("True support roam score")
    plt.ylabel("Predicted support roam score")
    plt.title("Validation: true vs predicted")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_residual_histogram(df: pd.DataFrame, true_col: str, pred_col: str, out_path: str, bins: int) -> None:
    residual = (df[pred_col] - df[true_col]).dropna()
    if residual.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(residual, bins=bins, color="#c44536", alpha=0.85, edgecolor="white")
    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlabel("Prediction residual (pred - true)")
    plt.ylabel("Rows")
    plt.title("Validation residuals")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_abs_error_by_score_bin(df: pd.DataFrame, true_col: str, pred_col: str, out_path: str) -> pd.DataFrame:
    work = df[[true_col, pred_col]].dropna().copy()
    if work.empty:
        return pd.DataFrame()
    work["abs_error"] = (work[pred_col] - work[true_col]).abs()
    work["score_bin"] = pd.cut(work[true_col], bins=np.linspace(0.0, 1.0, 11), include_lowest=True)
    summary = (
        work.groupby("score_bin", dropna=False)["abs_error"]
        .agg(count="count", mean="mean", median="median", std="std")
        .reset_index()
    )
    summary["score_bin"] = summary["score_bin"].astype(str)
    if not summary.empty:
        plt.figure(figsize=(9, 5))
        plt.bar(summary["score_bin"], summary["mean"], color="#2a9d8f", alpha=0.85)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Mean absolute error")
        plt.xlabel("True score bin")
        plt.title("Validation absolute error by score bin")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
    return summary


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    outdir = Path(args.outdir) if args.outdir else run_dir / "diagnostics"
    ensure_dir(str(outdir))

    history_path = run_dir / "history.csv"
    predictions_path = run_dir / "validation_predictions.parquet"
    metrics_path = run_dir / "metrics.json"

    metrics = load_json(str(metrics_path))
    if metrics:
        pd.DataFrame([metrics]).to_csv(outdir / "metrics_summary.csv", index=False)

    if history_path.exists():
        history = pd.read_csv(history_path)
        save_loss_curve(history, str(outdir / "loss_curve.png"))

    if predictions_path.exists():
        preds = pd.read_parquet(predictions_path)
        true_col, pred_col = find_prediction_columns(preds, args.target_col)
        save_true_vs_pred(preds, true_col, pred_col, str(outdir / "true_vs_pred_scatter.png"))
        save_residual_histogram(preds, true_col, pred_col, str(outdir / "residual_histogram.png"), args.bins)
        error_summary = save_abs_error_by_score_bin(
            preds,
            true_col,
            pred_col,
            str(outdir / "abs_error_by_score_bin.png"),
        )
        if not error_summary.empty:
            error_summary.to_csv(outdir / "abs_error_by_score_bin.csv", index=False)

    print(f"Saved training diagnostics: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()
