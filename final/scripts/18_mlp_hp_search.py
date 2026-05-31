#!/usr/bin/env python3
"""
18_mlp_hp_search.py -- Small hyperparameter search for the best MLP family.

The roadmap asks for a compact grid over the strongest MLP architecture. After
Tarea 1B that architecture is `04c_train_mlp_per_role.py`: per-slot champion
embeddings plus explicit support matchup / ADC synergy dot products.

This script trains each config on train, selects by validation Spearman, then
evaluates the best config on test exactly once.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = REPO_ROOT / "final" / "data" / "training" / "train.parquet"
DEFAULT_VAL = REPO_ROOT / "final" / "data" / "training" / "val.parquet"
DEFAULT_TEST = REPO_ROOT / "final" / "data" / "training" / "test.parquet"
DEFAULT_TRANSFORMER = REPO_ROOT / "final" / "data" / "training" / "quantile_transformer.joblib"
DEFAULT_OUTDIR = REPO_ROOT / "final" / "analysis" / "hp_search"
DEFAULT_DEFAULT_METRICS = REPO_ROOT / "final" / "models" / "mlp_per_role" / "metrics.json"
PER_ROLE_SCRIPT = REPO_ROOT / "final" / "scripts" / "04c_train_mlp_per_role.py"

TARGET_COL = "support_roam_score"
QUANTILE_COL = "support_roam_score_quantile"

HIDDEN_GRID = [[128, 64], [192, 96], [256, 128], [256, 128, 64]]
DROPOUT_GRID = [0.2, 0.3, 0.4]
LR_GRID = [1e-3, 5e-4, 2e-4]
WEIGHT_DECAY_GRID = [1e-4, 5e-4, 1e-3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hyperparameter search for MLP Per-Role + Interactions.")
    p.add_argument("--train", default=str(DEFAULT_TRAIN))
    p.add_argument("--val", default=str(DEFAULT_VAL))
    p.add_argument("--test", default=str(DEFAULT_TEST))
    p.add_argument("--quantile-transformer", default=str(DEFAULT_TRANSFORMER))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--target", choices=["raw", "quantile"], default="raw")
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-configs", type=int, default=0, help="Limit configs for smoke tests; 0 means full grid.")
    p.add_argument("--resume", action="store_true", help="Skip configs with existing metrics.json.")
    p.add_argument("--dry-run", action="store_true", help="Write the planned grid but do not train.")
    p.add_argument("--default-metrics", default=str(DEFAULT_DEFAULT_METRICS))
    p.add_argument("--improvement-threshold", type=float, default=0.005)
    p.add_argument(
        "--promote-if-improved",
        action="store_true",
        help="Copy the best run to final/models/mlp_per_role_tuned if val Spearman improves enough.",
    )
    return p.parse_args()


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("mlp_per_role_train", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def iter_grid() -> Iterable[Dict[str, Any]]:
    for hidden_dims, dropout, lr, weight_decay in itertools.product(
        HIDDEN_GRID, DROPOUT_GRID, LR_GRID, WEIGHT_DECAY_GRID
    ):
        yield {
            "hidden_dims": list(hidden_dims),
            "dropout": float(dropout),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
        }


def config_id(index: int, cfg: Dict[str, Any]) -> str:
    hidden = "-".join(str(v) for v in cfg["hidden_dims"])
    dropout = str(cfg["dropout"]).replace(".", "p")
    lr = f"{cfg['lr']:.0e}".replace("-", "m")
    wd = f"{cfg['weight_decay']:.0e}".replace("-", "m")
    return f"{index:03d}_h{hidden}_d{dropout}_lr{lr}_wd{wd}"


def target_column(target: str) -> str:
    return TARGET_COL if target == "raw" else QUANTILE_COL


def load_default_spearman(path: Path, target: str) -> Optional[float]:
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        if row.get("target") == target:
            return float(row["spearman_corr"])
    return None


def train_one_config(
    module: Any,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    cfg: Dict[str, Any],
    run_id: str,
    run_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if args.resume and metrics_path.exists():
        rows = json.loads(metrics_path.read_text(encoding="utf-8"))
        if rows:
            row = rows[0]
            row["run_id"] = run_id
            row["outdir"] = str(run_dir)
            return row

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    vocab = module.build_champion_vocab(df_train)
    vocab_size = len(vocab) + 1
    n_slots = len(module.CHAMPION_COLS)
    ds_train = module.make_dataset(df_train, vocab, target_column(args.target))
    ds_val = module.make_dataset(df_val, vocab, target_column(args.target))

    train_args = argparse.Namespace(
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
    )
    model = module.MLPPerRoleInteractions(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        n_champion_slots=n_slots,
        hidden_dims=list(cfg["hidden_dims"]),
        dropout=cfg["dropout"],
    )
    started = time.time()
    metrics, _trained_model, history = module.train_model(
        model=model,
        ds_train=ds_train,
        ds_val=ds_val,
        args=train_args,
        target_label=args.target,
        outdir=run_dir,
    )

    cfg_payload = {
        "model_type": "mlp_per_role_interactions",
        "run_id": run_id,
        "target": args.target,
        "champion_cols": list(module.CHAMPION_COLS),
        "slot_names": list(module.SLOT_NAMES),
        "vocab_size": vocab_size,
        "embed_dim": args.embed_dim,
        "n_champion_slots": n_slots,
        "input_dim": n_slots * args.embed_dim + 1 + 2,
        "hidden_dims": list(cfg["hidden_dims"]),
        "dropout": cfg["dropout"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
    }
    (run_dir / "model_config.json").write_text(json.dumps(cfg_payload, indent=2), encoding="utf-8")
    (run_dir / "vocab.json").write_text(
        json.dumps({str(k): int(v) for k, v in vocab.items()}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    joblib.dump(
        {
            "vocab": vocab,
            "champion_cols": list(module.CHAMPION_COLS),
            "slot_names": list(module.SLOT_NAMES),
            "side_mapping": {"blue": 0.0, "red": 1.0},
        },
        run_dir / "preprocess.joblib",
    )
    metrics.update(cfg)
    metrics["run_id"] = run_id
    metrics["outdir"] = str(run_dir)
    metrics["wall_seconds"] = float(time.time() - started)
    metrics_path.write_text(json.dumps([metrics], indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else float("nan")
    sp = spearmanr(y_true, y_pred, nan_policy="omit")
    pred_std = float(np.std(y_pred))
    target_std = float(np.std(y_true))
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "pearson_corr": pearson,
        "spearman_corr": float(sp.correlation) if sp.correlation is not None else float("nan"),
        "pred_std": pred_std,
        "target_std": target_std,
        "compression_ratio": pred_std / target_std if target_std > 0 else float("nan"),
    }


def inverse_quantile_predictions(q_pred: np.ndarray, transformer: Optional[Any]) -> Optional[np.ndarray]:
    if transformer is None:
        return None
    q = np.clip(np.asarray(q_pred, dtype=np.float64), 0.0, 1.0)
    raw = np.zeros_like(q, dtype=np.float64)
    positive = q > 0.0
    if positive.any():
        raw[positive] = transformer.inverse_transform(q[positive].reshape(-1, 1)).reshape(-1)
    return np.clip(raw, 0.0, 1.0)


def load_state(path: Path, device: Any) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def evaluate_best_on_test(
    module: Any,
    best: Dict[str, Any],
    df_test: pd.DataFrame,
    n_train: int,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    run_dir = Path(str(best["outdir"]))
    config = json.loads((run_dir / "model_config.json").read_text(encoding="utf-8"))
    vocab_raw = json.loads((run_dir / "vocab.json").read_text(encoding="utf-8"))
    vocab = {int(k): int(v) for k, v in vocab_raw.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = module.MLPPerRoleInteractions(
        vocab_size=int(config["vocab_size"]),
        embed_dim=int(config["embed_dim"]),
        n_champion_slots=int(config["n_champion_slots"]),
        hidden_dims=list(config["hidden_dims"]),
        dropout=0.0,
    )
    weight_path = run_dir / f"mlp_per_role_{args.target}.pt"
    model.load_state_dict(load_state(weight_path, device))
    model.to(device)
    model.eval()

    ds_test = module.make_dataset(df_test, vocab, target_column(args.target))
    loader = DataLoader(ds_test, batch_size=args.batch_size * 4, shuffle=False)
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for champion_ids, side, _y, _w in loader:
            preds.append(model(champion_ids.to(device), side.to(device)).cpu().numpy())
    y_pred = np.concatenate(preds)

    rows: List[Dict[str, Any]] = []
    model_name = "MLP Per-Role + Interactions HP Best"
    if args.target == "raw":
        rows.append(
            {
                "model": model_name,
                "trained_target": "raw",
                "evaluation_scale": "raw",
                "eval_split": "test",
                "n_train": int(n_train),
                "n_eval": int(len(df_test)),
                "run_id": best["run_id"],
                **regression_metrics(df_test[TARGET_COL].to_numpy(), y_pred),
            }
        )
    else:
        rows.append(
            {
                "model": model_name,
                "trained_target": "quantile",
                "evaluation_scale": "quantile",
                "eval_split": "test",
                "n_train": int(n_train),
                "n_eval": int(len(df_test)),
                "run_id": best["run_id"],
                **regression_metrics(df_test[QUANTILE_COL].to_numpy(), y_pred),
            }
        )
        transformer_path = Path(args.quantile_transformer)
        transformer = joblib.load(transformer_path) if transformer_path.exists() else None
        raw_pred = inverse_quantile_predictions(y_pred, transformer)
        if raw_pred is not None:
            rows.append(
                {
                    "model": f"{model_name} (quantile->raw)",
                    "trained_target": "quantile",
                    "evaluation_scale": "raw",
                    "eval_split": "test",
                    "n_train": int(n_train),
                    "n_eval": int(len(df_test)),
                    "run_id": best["run_id"],
                    **regression_metrics(df_test[TARGET_COL].to_numpy(), raw_pred),
                }
            )
    return rows


def promote_best(best: Dict[str, Any], outdir: Path) -> None:
    src = Path(str(best["outdir"]))
    dest = REPO_ROOT / "final" / "models" / "mlp_per_role_tuned"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    for target in ["raw", "quantile"]:
        old = dest / f"mlp_per_role_{target}.pt"
        if old.exists():
            old.rename(dest / f"mlp_per_role_tuned_{target}.pt")
    (outdir / "promoted_model_dir.txt").write_text(str(dest.resolve()), encoding="utf-8")


def write_outputs(
    outdir: Path,
    rows: List[Dict[str, Any]],
    best: Optional[Dict[str, Any]],
    test_rows: List[Dict[str, Any]],
    config: Dict[str, Any],
    default_spearman: Optional[float],
    threshold: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["spearman_corr", "r2"], ascending=[False, False])
    df.to_csv(outdir / "hp_search_results.csv", index=False)
    (outdir / "hp_search_results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (outdir / "hp_search_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if test_rows:
        pd.DataFrame(test_rows).to_csv(outdir / "hp_search_best_test.csv", index=False)
        (outdir / "hp_search_best_test.json").write_text(
            json.dumps(test_rows, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    lines = ["# MLP HP Search Summary", ""]
    lines.append(f"- Configurations evaluated: {len(rows)}")
    if default_spearman is not None:
        lines.append(f"- Default validation Spearman: {default_spearman:.6f}")
    if best is not None:
        best_sp = float(best["spearman_corr"])
        delta = best_sp - default_spearman if default_spearman is not None else float("nan")
        lines.extend(
            [
                f"- Best run: `{best['run_id']}`",
                f"- Best validation Spearman: {best_sp:.6f}",
                f"- Best validation R2: {float(best['r2']):.6f}",
                f"- Hidden dims: {best['hidden_dims']}",
                f"- Dropout: {best['dropout']}",
                f"- LR: {best['lr']}",
                f"- Weight decay: {best['weight_decay']}",
            ]
        )
        if default_spearman is not None:
            lines.append(f"- Delta vs default: {delta:.6f}")
            if delta <= threshold:
                lines.append("")
                lines.append(
                    "Decision: la MLP es robusta a la configuracion; se conserva el default "
                    "porque la mejora no supera el umbral de 0.005 Spearman."
                )
            else:
                lines.append("")
                lines.append("Decision: la mejor configuracion supera el umbral y puede promoverse.")
    if test_rows:
        lines.append("")
        lines.append("## Test Evaluation Of Selected Run")
        for row in test_rows:
            lines.append(
                f"- {row['model']} [{row['evaluation_scale']}]: "
                f"R2={row['r2']:.6f}, Spearman={row['spearman_corr']:.6f}"
            )
    (outdir / "hp_search_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    runs_dir = outdir / "runs"
    outdir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    module = load_module(PER_ROLE_SCRIPT)
    grid = list(iter_grid())
    if args.max_configs > 0:
        grid = grid[: args.max_configs]

    planned_rows = [
        {"run_id": config_id(i + 1, cfg), **cfg}
        for i, cfg in enumerate(grid)
    ]
    pd.DataFrame(planned_rows).to_csv(outdir / "hp_search_grid.csv", index=False)
    if args.dry_run:
        print(f"[Dry run] planned {len(planned_rows)} configs -> {outdir / 'hp_search_grid.csv'}")
        return

    df_train = pd.read_parquet(args.train)
    df_val = pd.read_parquet(args.val)
    df_test = pd.read_parquet(args.test)
    default_spearman = load_default_spearman(Path(args.default_metrics), args.target)

    config_payload = {
        "script": str(Path(__file__).resolve()),
        "architecture": "mlp_per_role_interactions",
        "target": args.target,
        "train": str(Path(args.train).resolve()),
        "val": str(Path(args.val).resolve()),
        "test": str(Path(args.test).resolve()),
        "grid_size": len(grid),
        "grid": {
            "hidden_dims": HIDDEN_GRID,
            "dropout": DROPOUT_GRID,
            "lr": LR_GRID,
            "weight_decay": WEIGHT_DECAY_GRID,
        },
        "embed_dim": args.embed_dim,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "default_validation_spearman": default_spearman,
        "improvement_threshold": args.improvement_threshold,
    }

    rows: List[Dict[str, Any]] = []
    for i, cfg in enumerate(grid, start=1):
        run_id = config_id(i, cfg)
        print(f"\n[Run {i}/{len(grid)}] {run_id}")
        row = train_one_config(module, df_train, df_val, cfg, run_id, runs_dir / run_id, args)
        rows.append(row)
        write_outputs(outdir, rows, None, [], config_payload, default_spearman, args.improvement_threshold)

    best = max(rows, key=lambda r: (float(r["spearman_corr"]), float(r["r2"]))) if rows else None
    test_rows: List[Dict[str, Any]] = []
    if best is not None:
        print(f"\n[Test] Evaluating selected run once: {best['run_id']}")
        test_rows = evaluate_best_on_test(module, best, df_test, len(df_train), args)
        if default_spearman is not None and args.promote_if_improved:
            delta = float(best["spearman_corr"]) - default_spearman
            if delta > args.improvement_threshold:
                promote_best(best, outdir)

    write_outputs(
        outdir=outdir,
        rows=rows,
        best=best,
        test_rows=test_rows,
        config=config_payload,
        default_spearman=default_spearman,
        threshold=args.improvement_threshold,
    )

    if best is not None:
        print(
            f"[Best] {best['run_id']}  val_spearman={float(best['spearman_corr']):.4f}  "
            f"val_r2={float(best['r2']):.4f}"
        )
    print(f"[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
