#!/usr/bin/env python3
"""
run_all_training.py -- Master script: retrain all models + generate comparison.

Designed to run on a GPU cluster. Executes in order:
  1. HistGBT (base, 10 champion IDs + side, with sample_weight)
  2. MLP OneHot (with stronger regularization to reduce overfitting)
  3. MLP Embeddings (same)
  4. Fair main model comparison table
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
from typing import Any, Dict, List, Optional

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


def clean_vs_chaotic_analysis(expected_seeds: Optional[List[int]] = None) -> None:
    """Evaluate final HistGBT predictions separately on clean vs chaotic test rows."""
    print(f"\n{'='*60}")
    print("  Clean vs Chaotic Analysis")
    print(f"{'='*60}")

    import joblib
    from scipy.stats import spearmanr

    df_test = pd.read_parquet(TRAINING_DIR / "test.parquet")

    if "chaos_flag" not in df_test.columns:
        raise SystemExit("[Clean/chaos] No chaos_flag column in test set")
    if df_test["chaos_flag"].isna().any():
        n_nan = int(df_test["chaos_flag"].isna().sum())
        raise SystemExit(f"[Clean/chaos] chaos_flag has {n_nan} NaN rows")

    y = df_test["support_roam_score"].to_numpy(dtype=np.float64)

    model_root = REPO_ROOT / "final" / "models" / "gbt"
    seed_dirs = sorted(
        path for path in model_root.glob("seed*")
        if path.is_dir() and (path / "gbt_model_raw.joblib").exists()
    )
    if expected_seeds:
        expected_names = {f"seed{seed}" for seed in expected_seeds}
        seed_dirs = [path for path in seed_dirs if path.name in expected_names]
        found_names = {path.name for path in seed_dirs}
        missing = sorted(expected_names - found_names)
        if missing:
            raise SystemExit(f"[Clean/chaos] Missing expected final HistGBT seed dirs: {missing}")
    if not seed_dirs:
        seed_dirs = [model_root]

    pred_by_seed: List[np.ndarray] = []
    run_manifests: List[Dict[str, Any]] = []
    reference_feature_cols: Optional[List[str]] = None
    reference_protocol: Optional[str] = None

    for run_dir in seed_dirs:
        config_path = run_dir / "model_config.json"
        preprocess_path = run_dir / "preprocess.joblib"
        model_path = run_dir / "gbt_model_raw.joblib"
        if not config_path.exists() or not preprocess_path.exists() or not model_path.exists():
            raise SystemExit(f"[Clean/chaos] Incomplete HistGBT artifact at {run_dir}")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        preprocess = joblib.load(preprocess_path)
        feature_cols = list(preprocess["feature_columns"])
        protocol = str(config.get("feature_protocol_id", ""))

        if protocol != "draft_10_champions_side":
            raise SystemExit(f"[Clean/chaos] {run_dir} is not the final main feature protocol: {protocol}")
        if config.get("used_sample_weight") is not True:
            raise SystemExit(f"[Clean/chaos] {run_dir} was not trained with sample_weight")
        if reference_feature_cols is None:
            reference_feature_cols = feature_cols
            reference_protocol = protocol
        elif feature_cols != reference_feature_cols:
            raise SystemExit(f"[Clean/chaos] Feature columns differ across HistGBT seed artifacts")

        X_raw = df_test[feature_cols].copy()
        for col in feature_cols:
            X_raw[col] = X_raw[col].fillna("__MISSING__").astype(str)
        X = preprocess["encoder"].transform(X_raw)
        model = joblib.load(model_path)
        pred_by_seed.append(model.predict(X).astype(np.float64))
        run_manifests.append(
            {
                "run_dir": str(run_dir.resolve()),
                "model_path": str(model_path.resolve()),
                "preprocess_path": str(preprocess_path.resolve()),
                "config_path": str(config_path.resolve()),
                "seed": config.get("seed"),
                "feature_protocol_id": protocol,
                "feature_columns": feature_cols,
                "used_sample_weight": bool(config.get("used_sample_weight")),
                "model_mtime": model_path.stat().st_mtime,
            }
        )

    pred_matrix = np.vstack(pred_by_seed)
    pred = pred_matrix.mean(axis=0)
    pred_seed_std_mean = pred_matrix.std(axis=0).mean() if len(pred_by_seed) > 1 else 0.0

    def metrics_for(mask, label):
        yt, yp = y[mask], pred[mask]
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
            "pred_std": float(yp.std()),
            "n_model_seeds": int(len(pred_by_seed)),
            "mean_prediction_seed_std": float(pred_matrix[:, mask].std(axis=0).mean())
            if len(pred_by_seed) > 1 else 0.0,
        }

    chaos = df_test["chaos_flag"].to_numpy(dtype=bool)
    results = [
        metrics_for(np.ones(len(y), dtype=bool), "all"),
        metrics_for(~chaos, "clean"),
        metrics_for(chaos, "chaotic"),
    ]
    n_all = results[0]["n"]
    n_clean = results[1]["n"]
    n_chaotic = results[2]["n"]
    if n_clean + n_chaotic != n_all:
        raise SystemExit(
            f"[Clean/chaos] clean + chaotic != all ({n_clean} + {n_chaotic} != {n_all})"
        )

    outdir = REPO_ROOT / "final" / "analysis" / "clean_vs_chaotic"
    outdir.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(results)
    df_out.to_csv(outdir / "clean_vs_chaotic_histgbt_final.csv", index=False)
    df_out.to_csv(outdir / "clean_vs_chaotic_gbt.csv", index=False)

    checks = {
        "clean_plus_chaotic_equals_all": True,
        "n_all": n_all,
        "n_clean": n_clean,
        "n_chaotic": n_chaotic,
        "chaos_flag_nan_count": int(df_test["chaos_flag"].isna().sum()),
        "chaos_flag_has_nan": False,
        "prediction_rows_match_test": int(len(pred)) == int(len(df_test)),
        "model_source": "HistGBT final seed ensemble" if len(seed_dirs) > 1 else "HistGBT final single artifact",
        "n_model_seeds": int(len(pred_by_seed)),
        "mean_prediction_seed_std": float(pred_seed_std_mean),
        "feature_protocol_id": reference_protocol,
        "feature_columns": reference_feature_cols,
        "used_sample_weight": all(row["used_sample_weight"] for row in run_manifests),
        "predictions_match_final_retrained_artifacts": True,
        "artifact_runs": run_manifests,
        "chaos_flag_definition": [
            "support_deaths_0_12 + adc_deaths_0_12 >= 6",
            "adc_deaths_0_12 >= 5",
            "support_deaths_0_12 >= 4 AND support_kill_assists_out_bot_0_12 == 0",
        ],
    }
    (outdir / "clean_vs_chaotic_checks.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    md = ["# Clean vs Chaotic - HistGBT Final Model (Test Set)", ""]
    md.append(
        "Model source: HistGBT final retrained seed artifacts, averaged at prediction time."
        if len(seed_dirs) > 1
        else "Model source: HistGBT final retrained artifact."
    )
    md.append("")
    md.append("Chaos flag definition:")
    md.append("")
    md.append("- `support_deaths_0_12 + adc_deaths_0_12 >= 6`")
    md.append("- `adc_deaths_0_12 >= 5`")
    md.append("- `support_deaths_0_12 >= 4 AND support_kill_assists_out_bot_0_12 == 0`")
    md.append("")
    md.append("Validation checks:")
    md.append("")
    md.append(f"- `clean + chaotic == all`: {n_clean:,} + {n_chaotic:,} = {n_all:,}")
    md.append("- `chaos_flag` NaN count: 0")
    md.append(f"- Feature protocol: `{reference_protocol}`")
    md.append(f"- Model seeds used: {', '.join(str(row['seed']) for row in run_manifests)}")
    md.append("")
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
    print(f"  Clean:   R2={results[1]['r2']:.4f}  Spearman={results[1]['spearman']:.4f}  n={results[1]['n']:,}")
    print(f"  Chaotic: R2={results[2]['r2']:.4f}  Spearman={results[2]['spearman']:.4f}  n={results[2]['n']:,}")
    print(f"  Checks:  clean+chaotic={n_clean + n_chaotic:,} all={n_all:,}; chaos_flag NaN=0")
    print(f"  Model:   final HistGBT seeds={', '.join(str(row['seed']) for row in run_manifests)}")
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
    p.add_argument("--include-secondary-comparison", action="store_true",
                   help="Also report enriched/Pair-TE/HP-best secondary rows.")
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
    p.add_argument("--use-wandb", action="store_true",
                   help="Use Weights & Biases to track training metrics.")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                   help="Seeds to train on (default: 42 123 456)")
    return p.parse_args()


def copy_seed_to_parent(model_dir: Path, seed: int) -> None:
    seed_dir = model_dir / f"seed{seed}"
    if not seed_dir.exists():
        return
    print(f"  [Copy] Promoting seed {seed} outputs from {seed_dir} to {model_dir}")
    import shutil
    for path in seed_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, model_dir / path.name)


def main() -> None:
    args = parse_args()
    t_total = time.time()
    timings: Dict[str, float] = {}
    comparison_models = ["baselines"]

    train_path = str(TRAINING_DIR / "train.parquet")
    val_path = str(TRAINING_DIR / "val.parquet")

    # Base MLP args
    mlp_base_args = [
        "--train", train_path,
        "--val", val_path,
        "--hidden-dims", *[str(d) for d in args.hidden_dims],
        "--dropout", str(args.dropout),
        "--weight-decay", str(args.weight_decay),
        "--patience", str(args.patience),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
    ]

    gbt_base_args = ["--train", train_path, "--val", val_path, "--feature-set", "main"]
    if args.use_wandb:
        gbt_base_args.append("--use-wandb")
        mlp_base_args.append("--use-wandb")

    # Loop over all seeds to train models
    for seed in args.seeds:
        print(f"\n============================================================")
        print(f"  STARTING TRAINING FOR SEED {seed}")
        print(f"============================================================")

        # 1. GBT
        if not args.skip_gbt and not args.mlp_only:
            gbt_outdir = REPO_ROOT / "final" / "models" / "gbt" / f"seed{seed}"
            gbt_args = gbt_base_args + ["--seed", str(seed), "--outdir", str(gbt_outdir)]
            elapsed = run_script(
                "03_train_gbt.py",
                gbt_args,
                f"Step 1/4: HistGBT (seed {seed}) - 10 champion IDs + side, with sample_weight"
            )
            timings[f"gbt_seed{seed}"] = elapsed
            if "gbt" not in comparison_models:
                comparison_models.append("gbt")

        # 2. MLP OneHot
        mlp_oh_outdir = REPO_ROOT / "final" / "models" / "mlp_onehot" / f"seed{seed}"
        mlp_oh_args = mlp_base_args + ["--seed", str(seed), "--outdir", str(mlp_oh_outdir)]
        elapsed = run_script(
            "04a_train_mlp_onehot.py",
            mlp_oh_args,
            f"Step 2/4: MLP OneHot (seed {seed}, regularized)"
        )
        timings[f"mlp_onehot_seed{seed}"] = elapsed
        if "mlp_onehot" not in comparison_models:
            comparison_models.append("mlp_onehot")

        # 3. MLP Embeddings
        mlp_emb_outdir = REPO_ROOT / "final" / "models" / "mlp_embed" / f"seed{seed}"
        mlp_emb_args = mlp_base_args + ["--seed", str(seed), "--outdir", str(mlp_emb_outdir)]
        elapsed = run_script(
            "04b_train_mlp_embed.py",
            mlp_emb_args,
            f"Step 3/4: MLP Embeddings (seed {seed}, regularized)"
        )
        timings[f"mlp_embed_seed{seed}"] = elapsed
        if "mlp_embed" not in comparison_models:
            comparison_models.append("mlp_embed")

        # 4. MLP Per-Role + Interactions
        mlp_pr_outdir = REPO_ROOT / "final" / "models" / "mlp_per_role" / f"seed{seed}"
        mlp_pr_args = mlp_base_args + ["--seed", str(seed), "--outdir", str(mlp_pr_outdir)]
        elapsed = run_script(
            "04c_train_mlp_per_role.py",
            mlp_pr_args,
            f"Step 4/4: MLP Per-Role + Interactions (seed {seed})"
        )
        timings[f"mlp_per_role_seed{seed}"] = elapsed
        if "mlp_per_role" not in comparison_models:
            comparison_models.append("mlp_per_role")

    # Promote the first seed (e.g. 42) outputs to parent directory for downstream analyses
    if args.seeds:
        first_seed = args.seeds[0]
        if "gbt" in comparison_models:
            copy_seed_to_parent(REPO_ROOT / "final" / "models" / "gbt", first_seed)
        copy_seed_to_parent(REPO_ROOT / "final" / "models" / "mlp_onehot", first_seed)
        copy_seed_to_parent(REPO_ROOT / "final" / "models" / "mlp_embed", first_seed)
        copy_seed_to_parent(REPO_ROOT / "final" / "models" / "mlp_per_role", first_seed)

    # 5. Comparison table
    if not args.skip_comparison:
        comparison_args = ["--models", *comparison_models]
        if args.include_secondary_comparison:
            comparison_args.append("--include-secondary")
        timings["comparison"] = run_script(
            "07_model_comparison.py", comparison_args,
            "Step 5/6: Fair Model Comparison Table"
        )

    # 6. Clean vs Chaotic + Training curves
    if "gbt" in comparison_models:
        clean_vs_chaotic_analysis(expected_seeds=args.seeds)
    else:
        print("\n[SKIP] Clean vs Chaotic Analysis requires a GBT retrained in this run")
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
