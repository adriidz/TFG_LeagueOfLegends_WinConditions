#!/usr/bin/env python3
"""
run_all_training.py -- Master script: retrain all models + generate comparison.

Designed to run on a GPU cluster. Executes in order:
  1. HistGBT (base, with sample_weight)
  2. MLP OneHot (with stronger regularization to reduce overfitting)
  3. MLP Embeddings (same)
  4. Model comparison table
  5. Clean vs Chaotic analysis
  6. Training curve plots

Usage:
  python run_all_training.py                    # full run
  python run_all_training.py --skip-gbt         # skip GBT (already trained)
  python run_all_training.py --mlp-only         # only MLPs + comparison

The MLP hyperparameters are tuned to reduce the train/val gap:
  - Higher dropout (0.35 vs 0.20)
  - Stronger weight decay (5e-4 vs 1e-4)
  - Smaller hidden layers ([192, 96] vs [256, 128])
  - More patience (20 vs 15) to let LR decay work
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
SCRIPTS_DIR = REPO_ROOT / "final" / "scripts"
TRAINING_DIR = REPO_ROOT / "final" / "data" / "training"
RESULTS_DIR = REPO_ROOT / "final" / "analysis" / "model_comparison"


def run_script(name: str, args: List[str], label: str) -> float:
    """Run a script and return elapsed time."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    cmd = [PYTHON, str(SCRIPTS_DIR / name)] + args
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}) after {elapsed:.1f}s")
        sys.exit(result.returncode)
    print(f"  Done in {elapsed:.1f}s")
    return elapsed


def clean_vs_chaotic_analysis() -> None:
    """Evaluate best model (GBT) separately on clean vs chaotic test rows."""
    print(f"\n{'='*60}")
    print("  Clean vs Chaotic Analysis")
    print(f"{'='*60}")

    import joblib
    from scipy.stats import spearmanr

    model_dir = REPO_ROOT / "final" / "models" / "gbt"
    model = joblib.load(model_dir / "gbt_model_raw.joblib")
    preprocess = joblib.load(model_dir / "preprocess.joblib")
    encoder = preprocess["encoder"]
    feature_cols = preprocess["feature_columns"]

    df_test = pd.read_parquet(TRAINING_DIR / "test.parquet")

    # Encode
    X_raw = df_test[feature_cols].copy()
    for col in feature_cols:
        X_raw[col] = X_raw[col].fillna("__MISSING__").astype(str)
    X = encoder.transform(X_raw)
    y = df_test["support_roam_score"].to_numpy(dtype=np.float64)
    pred = model.predict(X)

    def metrics_for(mask, label):
        yt, yp = y[mask], pred[mask]
        if len(yt) < 10:
            return None
        mse = float(np.mean((yt - yp) ** 2))
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        sp = spearmanr(yt, yp, nan_policy="omit")
        mae = float(np.mean(np.abs(yt - yp)))
        return {
            "subset": label,
            "n": int(len(yt)),
            "r2": r2,
            "spearman": float(sp.correlation) if sp.correlation is not None else float("nan"),
            "mae": mae,
            "rmse": math.sqrt(mse),
            "target_mean": float(yt.mean()),
            "target_std": float(yt.std()),
            "pred_std": float(yp.std()),
        }

    if "chaos_flag" not in df_test.columns:
        print("  [SKIP] No chaos_flag column in test set")
        return

    chaos = df_test["chaos_flag"].to_numpy(dtype=bool)
    results = [
        metrics_for(np.ones(len(y), dtype=bool), "all"),
        metrics_for(~chaos, "clean"),
        metrics_for(chaos, "chaotic"),
    ]
    results = [r for r in results if r is not None]

    outdir = REPO_ROOT / "final" / "analysis" / "clean_vs_chaotic"
    outdir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(results)
    df_out.to_csv(outdir / "clean_vs_chaotic_gbt.csv", index=False)

    md = ["# Clean vs Chaotic - HistGBT (Test Set)", ""]
    md.append("| subset | n | R2 | Spearman | MAE | RMSE | target_mean | pred_std |")
    md.append("|--------|---|----| ---------|-----|------|-------------|----------|")
    for r in results:
        md.append(
            f"| {r['subset']} | {r['n']:,} | {r['r2']:.4f} | {r['spearman']:.4f} "
            f"| {r['mae']:.4f} | {r['rmse']:.4f} | {r['target_mean']:.4f} | {r['pred_std']:.4f} |"
        )
    md.append("")
    (outdir / "clean_vs_chaotic.md").write_text("\n".join(md), encoding="utf-8")

    print(f"  All:     R2={results[0]['r2']:.4f}  Spearman={results[0]['spearman']:.4f}  n={results[0]['n']:,}")
    if len(results) > 1:
        print(f"  Clean:   R2={results[1]['r2']:.4f}  Spearman={results[1]['spearman']:.4f}  n={results[1]['n']:,}")
    if len(results) > 2:
        print(f"  Chaotic: R2={results[2]['r2']:.4f}  Spearman={results[2]['spearman']:.4f}  n={results[2]['n']:,}")
    print(f"  [Saved] {outdir}")


def plot_training_curves() -> None:
    """Generate training curve plots from history CSVs."""
    print(f"\n{'='*60}")
    print("  Training Curve Plots")
    print(f"{'='*60}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not installed")
        return

    outdir = REPO_ROOT / "final" / "analysis" / "training_curves"
    outdir.mkdir(parents=True, exist_ok=True)

    for model_name, model_dir_name in [
        ("MLP OneHot", "mlp_onehot"),
        ("MLP Embed", "mlp_embed"),
        ("MLP Per-Role + Interactions", "mlp_per_role"),
    ]:
        history_path = REPO_ROOT / "final" / "models" / model_dir_name / "history.csv"
        if not history_path.exists():
            print(f"  [SKIP] {history_path} not found")
            continue

        df = pd.read_csv(history_path)
        targets = df["target"].unique()

        for target in targets:
            sub = df[df["target"] == target]
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(sub["epoch"], sub["train_loss"], label="Train", linewidth=1.5)
            ax1.plot(sub["epoch"], sub["val_loss"], label="Val", linewidth=1.5)
            best = sub[sub["is_best"]]
            if not best.empty:
                ax1.axvline(best.iloc[-1]["epoch"], color="gray", linestyle="--",
                           alpha=0.5, label=f"Best (epoch {int(best.iloc[-1]['epoch'])})")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("MSE Loss")
            ax1.set_title(f"{model_name} ({target}) - Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(sub["epoch"], sub["lr"], color="tab:orange", linewidth=1.5)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Learning Rate")
            ax2.set_title(f"{model_name} ({target}) - LR Schedule")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            fname = f"{model_dir_name}_{target}_curves.png"
            fig.savefig(outdir / fname, dpi=150)
            plt.close(fig)
            print(f"  [Saved] {fname}")

    print(f"  [Saved] {outdir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Master training script.")
    p.add_argument("--skip-gbt", action="store_true",
                   help="Skip GBT training (already done)")
    p.add_argument("--mlp-only", action="store_true",
                   help="Only train MLPs + comparison")
    p.add_argument("--skip-comparison", action="store_true",
                   help="Skip model comparison table generation")
    # MLP regularization overrides
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[192, 96],
                   help="MLP hidden layer dimensions (default: 192 96)")
    p.add_argument("--dropout", type=float, default=0.35,
                   help="MLP dropout rate (default: 0.35)")
    p.add_argument("--weight-decay", type=float, default=5e-4,
                   help="MLP weight decay (default: 5e-4)")
    p.add_argument("--patience", type=int, default=20,
                   help="MLP early stopping patience (default: 20)")
    p.add_argument("--epochs", type=int, default=150,
                   help="MLP max epochs (default: 150)")
    p.add_argument("--lr", type=float, default=5e-4,
                   help="MLP learning rate (default: 5e-4)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t_total = time.time()
    timings: Dict[str, float] = {}

    train_path = str(TRAINING_DIR / "train.parquet")
    val_path = str(TRAINING_DIR / "val.parquet")

    # MLP args (stronger regularization)
    mlp_args = [
        "--train", train_path,
        "--val", val_path,
        "--hidden-dims", *[str(d) for d in args.hidden_dims],
        "--dropout", str(args.dropout),
        "--weight-decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
    ]

    # 1. GBT
    if not args.skip_gbt and not args.mlp_only:
        timings["gbt"] = run_script(
            "03_train_gbt.py",
            ["--train", train_path, "--val", val_path],
            "Step 1/6: HistGBT (with sample_weight)"
        )

    # 2. MLP OneHot
    timings["mlp_onehot"] = run_script(
        "04a_train_mlp_onehot.py",
        mlp_args,
        "Step 2/6: MLP OneHot (regularized)"
    )

    # 3. MLP Embeddings
    timings["mlp_embed"] = run_script(
        "04b_train_mlp_embed.py",
        mlp_args,
        "Step 3/6: MLP Embeddings (regularized)"
    )

    # 4. MLP Per-Role + Interactions
    timings["mlp_per_role"] = run_script(
        "04c_train_mlp_per_role.py",
        mlp_args,
        "Step 4/6: MLP Per-Role + Interactions"
    )

    # 5. Comparison table
    if not args.skip_comparison:
        timings["comparison"] = run_script(
            "07_model_comparison.py", [],
            "Step 5/6: Model Comparison Table"
        )

    # 6. Clean vs Chaotic + Training curves
    clean_vs_chaotic_analysis()
    plot_training_curves()

    total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  ALL DONE in {total:.1f}s ({total/60:.1f} min)")
    print(f"{'='*60}")
    for step, t in timings.items():
        print(f"  {step}: {t:.1f}s")
    print(f"\nMLP config: hidden={args.hidden_dims} dropout={args.dropout} "
          f"wd={args.weight_decay} lr={args.lr} patience={args.patience}")
    print(f"\nOutputs:")
    print(f"  models:     final/models/{{gbt,mlp_onehot,mlp_embed,mlp_per_role}}/")
    print(f"  comparison: final/analysis/model_comparison/")
    print(f"  curves:     final/analysis/training_curves/")
    print(f"  clean/chaos: final/analysis/clean_vs_chaotic/")


if __name__ == "__main__":
    main()
