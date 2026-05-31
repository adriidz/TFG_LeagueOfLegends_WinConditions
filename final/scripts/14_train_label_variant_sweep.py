#!/usr/bin/env python3
"""
14_train_label_variant_sweep.py

Train models across support-roam label variants.

The script keeps the existing match-level train/val/test split from
final/data/training, swaps in each requested label variant as the canonical
support_roam_score target, rebuilds the quantile target from train only, and
then trains one or more model families.

HistGBT is trained inside this script so feature ablations can be applied for
every label variant. MLP models reuse the existing 04a/04b scripts after this
script writes variant-specific split files.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_DIR = REPO_ROOT / "final" / "data" / "training"
DEFAULT_LABELS = REPO_ROOT / "final" / "data" / "scores" / "support_scores_v6_event_variants_m12.parquet"
DEFAULT_OUTDIR = REPO_ROOT / "final" / "analysis" / "label_variant_sweep"
DEFAULT_MODEL_DIR = REPO_ROOT / "final" / "models" / "label_variant_sweep"

TARGET_COL = "support_roam_score"
QUANTILE_COL = "support_roam_score_quantile"
JOIN_KEYS = ["match_id", "team_id"]
ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")

FEATURE_GROUPS: Dict[str, List[str]] = {
    "champions": [f"{s}_{r}_champion_id" for s in SIDES for r in ROLE_KEYS],
    "summoner_spells": [
        f"{s}_{r}_summoner{i}_id" for s in SIDES for r in ROLE_KEYS for i in (1, 2)
    ],
    "context": ["side"],
}

HISTGBR_FEATURE_SETS: Dict[str, List[str]] = {
    "all": ["champions", "summoner_spells", "context"],
    "no_spells": ["champions", "context"],
    "no_side": ["champions", "summoner_spells"],
    "champions_only": ["champions"],
    "spells_only": ["summoner_spells", "context"],
    "side_only": ["context"],
    "no_champions": ["summoner_spells", "context"],
}

MLP_FEATURE_SETS = ("champions_side", "champions_only")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train models across label variants and ablations.")
    p.add_argument("--training-dir", default=str(DEFAULT_TRAINING_DIR))
    p.add_argument("--labels", default=str(DEFAULT_LABELS))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument(
        "--models",
        nargs="+",
        default=["histgbr"],
        choices=["histgbr", "mlp_onehot", "mlp_embed"],
        help="Model families to train.",
    )
    p.add_argument(
        "--label-variants",
        nargs="+",
        default=["all"],
        help="Variant ids or full score columns. Use 'all' to train every discovered variant.",
    )
    p.add_argument(
        "--feature-ablations",
        nargs="+",
        default=["all", "no_spells", "no_side", "champions_only"],
        help="HistGBT feature sets. Use 'all_sets' for every built-in set.",
    )
    p.add_argument(
        "--mlp-feature-ablations",
        nargs="+",
        default=["champions_side"],
        help="MLP feature sets. Available: champions_side, champions_only, all_sets.",
    )
    p.add_argument("--target-mode", choices=["raw", "quantile", "both"], default="both")
    p.add_argument("--n-quantiles", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit-variants", type=int, default=0)
    p.add_argument("--limit-train-rows", type=int, default=0)
    p.add_argument("--limit-val-rows", type=int, default=0)

    p.add_argument("--hist-max-iter", type=int, default=300)
    p.add_argument("--hist-max-depth", type=int, default=6)
    p.add_argument("--hist-learning-rate", type=float, default=0.05)
    p.add_argument("--hist-min-samples-leaf", type=int, default=50)
    p.add_argument("--hist-max-leaf-nodes", type=int, default=31)

    p.add_argument("--mlp-epochs", type=int, default=100)
    p.add_argument("--mlp-patience", type=int, default=15)
    p.add_argument("--mlp-batch-size", type=int, default=512)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-embed-dim", type=int, default=16)
    return p.parse_args()


def safe_name(value: str) -> str:
    return (
        value.replace("support_roam_score_", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def discover_label_columns(labels: pd.DataFrame, requested: List[str]) -> List[Tuple[str, str]]:
    prefix = "support_roam_score_"
    candidates = [
        c for c in labels.columns
        if c.startswith(prefix) and not c.startswith("raw_")
    ]
    # support_roam_score_v6_events is the selected-label alias. Prefer the more
    # explicit support_roam_score_events_selected_75_15_10 when present.
    explicit_selected = f"{prefix}events_selected_75_15_10"
    if explicit_selected in candidates and f"{prefix}v6_events" in candidates:
        candidates.remove(f"{prefix}v6_events")

    by_id = {safe_name(c): c for c in candidates}
    by_col = {c: c for c in candidates}
    if requested == ["all"]:
        return [(safe_name(c), c) for c in candidates]

    selected: List[Tuple[str, str]] = []
    for item in requested:
        col = by_col.get(item) or by_id.get(item)
        if col is None:
            raise SystemExit(
                f"Unknown label variant '{item}'. Available: {', '.join(sorted(by_id))}"
            )
        selected.append((safe_name(col), col))
    return selected


def resolve_feature_sets(requested: List[str]) -> List[str]:
    if requested == ["all_sets"]:
        return list(HISTGBR_FEATURE_SETS)
    unknown = [x for x in requested if x not in HISTGBR_FEATURE_SETS]
    if unknown:
        raise SystemExit(
            f"Unknown feature ablation(s): {unknown}. Available: {', '.join(HISTGBR_FEATURE_SETS)}"
        )
    return requested


def resolve_mlp_feature_sets(requested: List[str]) -> List[str]:
    if requested == ["all_sets"]:
        return list(MLP_FEATURE_SETS)
    unknown = [x for x in requested if x not in MLP_FEATURE_SETS]
    if unknown:
        raise SystemExit(
            f"Unknown MLP feature ablation(s): {unknown}. Available: {', '.join(MLP_FEATURE_SETS)}"
        )
    return requested


def load_base_splits(training_dir: Path) -> Dict[str, pd.DataFrame]:
    splits = {}
    for split in ["train", "val", "test"]:
        path = training_dir / f"{split}.parquet"
        if not path.exists():
            raise SystemExit(f"Missing split file: {path}")
        splits[split] = pd.read_parquet(path)
    return splits


def attach_label(df: pd.DataFrame, labels: pd.DataFrame, score_col: str) -> pd.DataFrame:
    drop_cols = [c for c in [TARGET_COL, QUANTILE_COL] if c in df.columns]
    out = df.drop(columns=drop_cols).merge(
        labels[JOIN_KEYS + [score_col]],
        on=JOIN_KEYS,
        how="inner",
    )
    out = out.rename(columns={score_col: TARGET_COL})
    out = out[out[TARGET_COL].notna()].copy()
    out = out[out[TARGET_COL].between(0.0, 1.0, inclusive="both")].copy()
    return out


def add_quantile_target(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    n_quantiles: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, QuantileTransformer]:
    y_train = train[TARGET_COL].to_numpy(dtype=np.float64)
    pos_train = y_train > 0.0
    positive = y_train[pos_train].reshape(-1, 1)
    n_q = min(n_quantiles, int(np.isfinite(positive).sum()))
    qt = QuantileTransformer(
        n_quantiles=max(1, n_q),
        output_distribution="uniform",
        random_state=seed,
        subsample=max(int(positive.shape[0]), 1),
    )
    qt.fit(positive)

    for split_df in [train, val, test]:
        y = split_df[TARGET_COL].to_numpy(dtype=np.float64)
        q = np.zeros_like(y)
        pos = y > 0.0
        if pos.any():
            q[pos] = np.clip(qt.transform(y[pos].reshape(-1, 1)).ravel(), 0.0, 1.0)
        split_df[QUANTILE_COL] = q.astype(np.float32)
    return train, val, test, qt


def prepare_variant_splits(
    base_splits: Dict[str, pd.DataFrame],
    labels: pd.DataFrame,
    variant_id: str,
    score_col: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, pd.DataFrame], Path]:
    train = attach_label(base_splits["train"], labels, score_col)
    val = attach_label(base_splits["val"], labels, score_col)
    test = attach_label(base_splits["test"], labels, score_col)

    if args.limit_train_rows > 0:
        train = train.head(args.limit_train_rows).copy()
    if args.limit_val_rows > 0:
        val = val.head(args.limit_val_rows).copy()

    train, val, test, qt = add_quantile_target(train, val, test, args.n_quantiles, args.seed)

    split_dir = Path(args.outdir) / "splits" / variant_id
    split_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(split_dir / "train.parquet", index=False)
    val.to_parquet(split_dir / "val.parquet", index=False)
    test.to_parquet(split_dir / "test.parquet", index=False)
    joblib.dump(qt, split_dir / "quantile_transformer.joblib")

    split_summary = {
        "variant_id": variant_id,
        "score_col": score_col,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "matches": {
            "train": int(train["match_id"].nunique()),
            "val": int(val["match_id"].nunique()),
            "test": int(test["match_id"].nunique()),
        },
        "target_train_mean": float(train[TARGET_COL].mean()),
        "target_train_std": float(train[TARGET_COL].std(ddof=0)),
    }
    (split_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"train": train, "val": val, "test": test}, split_dir


def prepare_mlp_split_dir(split_dir: Path, variant_id: str, feature_set: str) -> Path:
    if feature_set == "champions_side":
        return split_dir
    if feature_set != "champions_only":
        raise ValueError(f"Unsupported MLP feature_set={feature_set}")

    outdir = split_dir.parent / f"{variant_id}__mlp_{feature_set}"
    outdir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        df = pd.read_parquet(split_dir / f"{split}.parquet")
        # Existing MLP scripts always read side. Setting it to an unknown value
        # makes encode_side map it to the neutral 0.5 value for every row.
        df["side"] = "__NEUTRAL__"
        df.to_parquet(outdir / f"{split}.parquet", index=False)
    qt_path = split_dir / "quantile_transformer.joblib"
    if qt_path.exists():
        joblib.dump(joblib.load(qt_path), outdir / "quantile_transformer.joblib")
    return outdir


def feature_columns(df: pd.DataFrame, feature_set: str) -> List[str]:
    cols: List[str] = []
    for group in HISTGBR_FEATURE_SETS[feature_set]:
        cols.extend([c for c in FEATURE_GROUPS[group] if c in df.columns])
    return list(dict.fromkeys(cols))


def encode_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, OrdinalEncoder, List[bool]]:
    x_train = train[cols].copy()
    x_val = val[cols].copy()
    for col in cols:
        x_train[col] = x_train[col].fillna("__MISSING__").astype(str)
        x_val[col] = x_val[col].fillna("__MISSING__").astype(str)
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float32,
    )
    return encoder.fit_transform(x_train), encoder.transform(x_val), encoder, [True] * len(cols)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    target_label: str,
    variant_id: str,
    feature_set: str,
    n_train: int,
    elapsed: float,
) -> Dict[str, Any]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    pred_std = float(np.std(y_pred))
    target_std = float(np.std(y_true))
    spearman = spearmanr(y_true, y_pred, nan_policy="omit")
    return {
        "model": model_name,
        "variant_id": variant_id,
        "feature_set": feature_set,
        "target": target_label,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "pearson_corr": float(np.corrcoef(y_true, y_pred)[0, 1])
        if pred_std > 1e-12 and target_std > 1e-12 else float("nan"),
        "spearman_corr": float(spearman.correlation)
        if spearman.correlation is not None else float("nan"),
        "pred_std": pred_std,
        "target_std": target_std,
        "compression_ratio": pred_std / target_std if target_std > 0 else float("nan"),
        "n_train": int(n_train),
        "n_eval": int(len(y_true)),
        "eval_split": "val",
        "training_seconds": float(elapsed),
    }


def train_histgbr(
    splits: Dict[str, pd.DataFrame],
    variant_id: str,
    feature_set: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    cols = feature_columns(splits["train"], feature_set)
    if not cols:
        raise RuntimeError(f"No feature columns found for feature_set={feature_set}")

    outdir = Path(args.model_dir) / "histgbr" / variant_id / feature_set
    outdir.mkdir(parents=True, exist_ok=True)
    metrics_path = outdir / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    x_train, x_val, encoder, cat_mask = encode_features(splits["train"], splits["val"], cols)
    targets = [("raw", TARGET_COL)]
    if args.target_mode in {"quantile", "both"}:
        targets.append(("quantile", QUANTILE_COL))
    if args.target_mode == "quantile":
        targets = [("quantile", QUANTILE_COL)]

    results = []
    for target_label, target_col in targets:
        y_train = splits["train"][target_col].to_numpy(dtype=np.float32)
        y_val = splits["val"][target_col].to_numpy(dtype=np.float32)
        print(f"[HistGBT] variant={variant_id} features={feature_set} target={target_label}")
        model = HistGradientBoostingRegressor(
            max_iter=args.hist_max_iter,
            max_depth=args.hist_max_depth,
            learning_rate=args.hist_learning_rate,
            min_samples_leaf=args.hist_min_samples_leaf,
            max_leaf_nodes=args.hist_max_leaf_nodes,
            categorical_features=cat_mask,
            random_state=args.seed,
            verbose=0,
        )
        t0 = time.time()
        model.fit(x_train, y_train)
        elapsed = time.time() - t0
        pred = model.predict(x_val)
        metrics = compute_metrics(
            y_val, pred, "histgbr", target_label, variant_id, feature_set, len(y_train), elapsed
        )
        results.append(metrics)
        joblib.dump(model, outdir / f"histgbr_{target_label}.joblib")
        print(
            f"  R2={metrics['r2']:.4f} Spearman={metrics['spearman_corr']:.4f} "
            f"RMSE={metrics['rmse']:.4f} time={elapsed:.1f}s"
        )

    joblib.dump({"encoder": encoder, "feature_columns": cols}, outdir / "preprocess.joblib")
    config = {
        "variant_id": variant_id,
        "feature_set": feature_set,
        "feature_columns": cols,
        "hist_max_iter": args.hist_max_iter,
        "hist_max_depth": args.hist_max_depth,
        "hist_learning_rate": args.hist_learning_rate,
        "hist_min_samples_leaf": args.hist_min_samples_leaf,
        "hist_max_leaf_nodes": args.hist_max_leaf_nodes,
        "seed": args.seed,
    }
    (outdir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def run_mlp_script(
    model_name: str,
    split_dir: Path,
    variant_id: str,
    feature_set: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    script = {
        "mlp_onehot": REPO_ROOT / "final" / "scripts" / "04a_train_mlp_onehot.py",
        "mlp_embed": REPO_ROOT / "final" / "scripts" / "04b_train_mlp_embed.py",
    }[model_name]
    outdir = Path(args.model_dir) / model_name / variant_id / feature_set
    metrics_path = outdir / "metrics.json"
    if args.skip_existing and metrics_path.exists():
        rows = json.loads(metrics_path.read_text(encoding="utf-8"))
        for row in rows:
            row["variant_id"] = variant_id
            row["feature_set"] = feature_set
        return rows

    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "--train", str(split_dir / "train.parquet"),
        "--val", str(split_dir / "val.parquet"),
        "--outdir", str(outdir),
        "--epochs", str(args.mlp_epochs),
        "--patience", str(args.mlp_patience),
        "--batch-size", str(args.mlp_batch_size),
        "--lr", str(args.mlp_lr),
        "--seed", str(args.seed),
    ]
    if model_name == "mlp_embed":
        cmd.extend(["--embed-dim", str(args.mlp_embed_dim)])

    print(f"[{model_name}] variant={variant_id} features={feature_set}")
    subprocess.run(cmd, check=True)
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    for row in rows:
        row["variant_id"] = variant_id
        row["feature_set"] = feature_set
    metrics_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def write_summary(outdir: Path, rows: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "sweep_metrics.csv", index=False)
    (outdir / "sweep_metrics.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (outdir / "sweep_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not df.empty:
        sort_cols = [c for c in ["model", "target", "feature_set", "spearman_corr"] if c in df.columns]
        if "spearman_corr" in sort_cols:
            table = df.sort_values("spearman_corr", ascending=False)
            table.head(40).to_csv(outdir / "sweep_top40_by_spearman.csv", index=False)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    model_dir = Path(args.model_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_parquet(args.labels)
    variants = discover_label_columns(labels, args.label_variants)
    if args.limit_variants > 0:
        variants = variants[:args.limit_variants]
    feature_sets = resolve_feature_sets(args.feature_ablations)
    mlp_feature_sets = resolve_mlp_feature_sets(args.mlp_feature_ablations)
    base_splits = load_base_splits(Path(args.training_dir))

    config = {
        "labels": str(Path(args.labels).resolve()),
        "training_dir": str(Path(args.training_dir).resolve()),
        "outdir": str(outdir.resolve()),
        "model_dir": str(model_dir.resolve()),
        "models": args.models,
        "variants": [{"variant_id": v, "score_col": c} for v, c in variants],
        "feature_ablations": feature_sets,
        "mlp_feature_ablations": mlp_feature_sets,
        "target_mode": args.target_mode,
    }
    print(json.dumps(config, indent=2, ensure_ascii=False))
    if args.dry_run:
        write_summary(outdir, [], config)
        print("[Dry run] No models trained.")
        return

    all_metrics: List[Dict[str, Any]] = []
    for variant_id, score_col in variants:
        print(f"\n[Variant] {variant_id} <- {score_col}")
        splits, split_dir = prepare_variant_splits(base_splits, labels, variant_id, score_col, args)

        if "histgbr" in args.models:
            for feature_set in feature_sets:
                all_metrics.extend(train_histgbr(splits, variant_id, feature_set, args))
                write_summary(outdir, all_metrics, config)

        for model_name in ["mlp_onehot", "mlp_embed"]:
            if model_name in args.models:
                for feature_set in mlp_feature_sets:
                    mlp_split_dir = prepare_mlp_split_dir(split_dir, variant_id, feature_set)
                    all_metrics.extend(
                        run_mlp_script(model_name, mlp_split_dir, variant_id, feature_set, args)
                    )
                    write_summary(outdir, all_metrics, config)

    write_summary(outdir, all_metrics, config)
    print(f"\n[Saved] {outdir.resolve()}")
    print(f"[Models] {model_dir.resolve()}")


if __name__ == "__main__":
    main()
