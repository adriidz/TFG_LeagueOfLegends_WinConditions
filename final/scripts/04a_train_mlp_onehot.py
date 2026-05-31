#!/usr/bin/env python3
"""
04a_train_mlp_onehot.py -- MLP baseline with one-hot champion IDs.

This script trains the support-roaming MLP twice:
  1. target raw:      support_roam_score
  2. target quantile: support_roam_score_quantile

Validation is used only for model selection / early stopping. The definitive
test-set comparison is generated later by 07_model_comparison.py.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = str(REPO_ROOT / "final" / "data" / "training" / "train.parquet")
DEFAULT_VAL = str(REPO_ROOT / "final" / "data" / "training" / "val.parquet")
DEFAULT_OUTDIR = str(REPO_ROOT / "final" / "models" / "mlp_onehot")

TARGET_COL = "support_roam_score"
QUANTILE_COL = "support_roam_score_quantile"

ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")
CHAMPION_COLS = [f"{s}_{r}_champion_id" for s in SIDES for r in ROLE_KEYS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MLP with one-hot champion encoding.")
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--val", default=DEFAULT_VAL)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_champion_vocab(df_train: pd.DataFrame) -> Dict[int, int]:
    """Map raw champion_id -> contiguous index. Index 0 is reserved for unknown."""
    all_ids: Set[int] = set()
    for col in CHAMPION_COLS:
        if col in df_train.columns:
            all_ids.update(df_train[col].dropna().astype(int).unique())
    return {cid: idx + 1 for idx, cid in enumerate(sorted(all_ids))}


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
        .map({"blue": 0.0, "red": 1.0})
        .fillna(0.5)
        .to_numpy(dtype=np.float32)
        .reshape(-1, 1)
    )


def make_dataset(
    df: pd.DataFrame,
    vocab: Dict[int, int],
    target_col: str,
) -> TensorDataset:
    champ_ids = encode_champion_ids(df, vocab)
    side = encode_side(df)
    y = df[target_col].to_numpy(dtype=np.float32)
    # Include sample_weight if available
    if "sample_weight" in df.columns:
        w = df["sample_weight"].to_numpy(dtype=np.float32)
    else:
        w = np.ones(len(df), dtype=np.float32)
    return TensorDataset(
        torch.from_numpy(champ_ids),
        torch.from_numpy(side),
        torch.from_numpy(y),
        torch.from_numpy(w),
    )


class MLPOneHot(nn.Module):
    """MLP over per-slot champion one-hot vectors plus side."""

    def __init__(
        self,
        vocab_size: int,
        n_champion_slots: int,
        hidden_dims: List[int],
        dropout: float,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_champion_slots = n_champion_slots
        input_dim = n_champion_slots * vocab_size + 1

        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.ReLU(),
                    nn.BatchNorm1d(h),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, champion_ids: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(champion_ids, num_classes=self.vocab_size).to(torch.float32)
        x = torch.cat([onehot.flatten(start_dim=1), side], dim=1)
        return self.net(x).squeeze(-1)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    label: str,
    n_train: int,
    elapsed: float,
    epoch: int,
) -> Dict[str, Any]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else float("nan")
    sp = spearmanr(y_true, y_pred, nan_policy="omit")
    target_std = float(np.std(y_true))
    pred_std = float(np.std(y_pred))
    return {
        "model": model_name,
        "target": label,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "r2": r2,
        "pearson_corr": pearson,
        "spearman_corr": float(sp.correlation) if sp.correlation is not None else float("nan"),
        "pred_std": pred_std,
        "target_std": target_std,
        "compression_ratio": pred_std / target_std if target_std > 0 else float("nan"),
        "n_train": int(n_train),
        "n_eval": int(len(y_true)),
        "eval_split": "val",
        "training_seconds": float(elapsed),
        "best_epoch": int(epoch),
    }


def train_model(
    model: MLPOneHot,
    ds_train: TensorDataset,
    ds_val: TensorDataset,
    args: argparse.Namespace,
    target_label: str,
    outdir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size * 2, shuffle=False)

    # Check if we have non-trivial weights
    has_weights = len(ds_train.tensors) >= 4
    if has_weights:
        w_mean = ds_train.tensors[3].mean().item()
        print(f"  [{target_label}] Using sample_weight (mean={w_mean:.3f})")
    else:
        print(f"  [{target_label}] No sample_weight")

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, Any]] = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for batch in loader_train:
            champion_ids = batch[0].to(device)
            side = batch[1].to(device)
            y = batch[2].to(device)
            w = batch[3].to(device) if has_weights else None

            pred = model(champion_ids, side)
            if w is not None:
                # Weighted MSE
                loss = (w * (pred - y) ** 2).mean()
            else:
                loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(y)
        train_loss = train_loss_sum / len(ds_train)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in loader_val:
                champion_ids = batch[0].to(device)
                side = batch[1].to(device)
                y = batch[2].to(device)
                pred = model(champion_ids, side)
                val_loss_sum += criterion(pred, y).item() * len(y)
        val_loss = val_loss_sum / len(ds_val)
        scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "target": target_label,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "is_best": is_best,
            }
        )

        if epoch == 1 or epoch % 5 == 0 or is_best:
            print(
                f"  [{target_label}] epoch {epoch:3d}  train={train_loss:.5f}  "
                f"val={val_loss:.5f}  lr={lr:.1e}"
            )

        if epoch - best_epoch >= args.patience:
            print(f"  [{target_label}] Early stopping at epoch {epoch} (best={best_epoch})")
            break

    if best_state is None:
        raise RuntimeError(f"No best state recorded for target={target_label}")

    elapsed = time.time() - t0
    model.load_state_dict(best_state)
    model.eval()

    all_preds: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader_val:
            champion_ids = batch[0].to(device)
            side = batch[1].to(device)
            all_preds.append(model(champion_ids, side).cpu().numpy())
            all_true.append(batch[2].numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    metrics = compute_metrics(
        y_true,
        y_pred,
        f"mlp_onehot_{target_label}",
        target_label,
        n_train=len(ds_train),
        elapsed=elapsed,
        epoch=best_epoch,
    )
    torch.save(best_state, outdir / f"mlp_onehot_{target_label}.pt")

    print(
        f"  [{target_label}] R2={metrics['r2']:.4f}  "
        f"Spearman={metrics['spearman_corr']:.4f}  "
        f"pred_std={metrics['pred_std']:.4f}  best_epoch={best_epoch}"
    )
    return metrics, history


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df_train = pd.read_parquet(args.train)
    df_val = pd.read_parquet(args.val)
    print(f"[Data] train={len(df_train):,}  val={len(df_val):,}")

    vocab = build_champion_vocab(df_train)
    vocab_size = len(vocab) + 1
    n_slots = len(CHAMPION_COLS)
    input_dim = n_slots * vocab_size + 1
    print(f"[Vocab] {len(vocab)} champions  input_dim={input_dim}  hidden={args.hidden_dims}")

    results: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []

    targets = [("raw", TARGET_COL)]
    if QUANTILE_COL in df_train.columns and QUANTILE_COL in df_val.columns:
        targets.append(("quantile", QUANTILE_COL))

    for target_label, target_col in targets:
        print(f"\n[Train] MLP OneHot target={target_label}")
        ds_train = make_dataset(df_train, vocab, target_col)
        ds_val = make_dataset(df_val, vocab, target_col)
        model = MLPOneHot(vocab_size, n_slots, args.hidden_dims, args.dropout)
        metrics, history = train_model(model, ds_train, ds_val, args, target_label, outdir)
        results.append(metrics)
        history_rows.extend(history)

    config = {
        "champion_cols": CHAMPION_COLS,
        "vocab_size": vocab_size,
        "n_champion_slots": n_slots,
        "input_dim": input_dim,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
    }
    (outdir / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (outdir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (outdir / "vocab.json").write_text(
        json.dumps({str(k): v for k, v in vocab.items()}, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history_rows).to_csv(outdir / "history.csv", index=False)
    joblib.dump(
        {
            "vocab": vocab,
            "champion_cols": CHAMPION_COLS,
            "side_mapping": {"blue": 0.0, "red": 1.0},
        },
        outdir / "preprocess.joblib",
    )

    print(f"\n[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
