#!/usr/bin/env python3
"""
27_mlp_interaction_experiments.py -- PyTorch MLP experiments grid to:
  - Experiment A': Exclude raw allied support embeddings from dense input, forcing interactions.
  - Experiment B': Use robust loss functions (L1/Huber) to combat caothic match noise.
  - Experiment C': Add extra dot product interactions (jungle, middle).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = str(REPO_ROOT / "final" / "data" / "training" / "train.parquet")
DEFAULT_VAL = str(REPO_ROOT / "final" / "data" / "training" / "val.parquet")
DEFAULT_OUTDIR = str(REPO_ROOT / "final" / "models" / "mlp_experiments")

TARGET_COL = "support_roam_score"
WEIGHT_COL = "sample_weight"

ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")
CHAMPION_COLS = [f"{s}_{r}_champion_id" for s in SIDES for r in ROLE_KEYS]
SLOT_NAMES = [col[: -len("_champion_id")] for col in CHAMPION_COLS]

ALLY_UTILITY_SLOT = SLOT_NAMES.index("ally_utility")
ENEMY_UTILITY_SLOT = SLOT_NAMES.index("enemy_utility")
ALLY_BOTTOM_SLOT = SLOT_NAMES.index("ally_bottom")
ALLY_JUNGLE_SLOT = SLOT_NAMES.index("ally_jungle")
ALLY_MID_SLOT = SLOT_NAMES.index("ally_middle")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PyTorch MLP interaction experiments.")
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--val", default=DEFAULT_VAL)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Limit train/val rows for a fast smoke test.",
    )
    return p.parse_args()


def build_champion_vocab(df_train: pd.DataFrame) -> Dict[int, int]:
    all_ids = set()
    for col in CHAMPION_COLS:
        if col in df_train.columns:
            all_ids.update(df_train[col].dropna().astype(int).unique())
    return {int(cid): idx + 1 for idx, cid in enumerate(sorted(all_ids))}


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
    y = df[target_col].to_numpy(dtype=np.float32)
    if WEIGHT_COL in df.columns:
        w = df[WEIGHT_COL].to_numpy(dtype=np.float32)
    else:
        w = np.ones(len(df), dtype=np.float32)

    return TensorDataset(
        torch.from_numpy(encode_champion_ids(df, vocab)),
        torch.from_numpy(encode_side(df)),
        torch.from_numpy(y),
        torch.from_numpy(w),
    )


def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (weights * (pred - target) ** 2).mean()


class ExperimentalMLP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_champion_slots: int,
        hidden_dims: List[int],
        dropout: float,
        exclude_support: bool = False,
        use_extra_dots: bool = False,
    ):
        super().__init__()
        self.exclude_support = exclude_support
        self.use_extra_dots = use_extra_dots
        self.n_champion_slots = n_champion_slots
        self.slot_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, embed_dim, padding_idx=0) for _ in range(n_champion_slots)]
        )

        n_slots_in_linear = n_champion_slots - 1 if exclude_support else n_champion_slots
        extra_dots_dim = 4 if use_extra_dots else 2
        input_dim = n_slots_in_linear * embed_dim + 1 + extra_dots_dim

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
        self.head = nn.Sequential(*layers)

    def forward(self, champion_ids: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        embeddings = torch.stack(
            [emb(champion_ids[:, i]) for i, emb in enumerate(self.slot_embeddings)],
            dim=1,
        )
        support_vs_support = (
            embeddings[:, ALLY_UTILITY_SLOT] * embeddings[:, ENEMY_UTILITY_SLOT]
        ).sum(dim=1, keepdim=True)
        support_adc = (
            embeddings[:, ALLY_UTILITY_SLOT] * embeddings[:, ALLY_BOTTOM_SLOT]
        ).sum(dim=1, keepdim=True)

        extra_dots = []
        if self.use_extra_dots:
            support_jungle = (
                embeddings[:, ALLY_UTILITY_SLOT] * embeddings[:, ALLY_JUNGLE_SLOT]
            ).sum(dim=1, keepdim=True)
            support_mid = (
                embeddings[:, ALLY_UTILITY_SLOT] * embeddings[:, ALLY_MID_SLOT]
            ).sum(dim=1, keepdim=True)
            extra_dots = [support_jungle, support_mid]

        if self.exclude_support:
            indices = [i for i in range(self.n_champion_slots) if i != ALLY_UTILITY_SLOT]
            flat_embeddings = embeddings[:, indices].flatten(start_dim=1)
        else:
            flat_embeddings = embeddings.flatten(start_dim=1)

        x = torch.cat(
            [
                flat_embeddings,
                side,
                support_vs_support,
                support_adc,
            ] + extra_dots,
            dim=1,
        )
        return self.head(x).squeeze(-1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    pred_std = float(np.std(y_pred))
    target_std = float(np.std(y_true))
    
    if pred_std > 1e-12 and target_std > 1e-12:
        sp = spearmanr(y_true, y_pred, nan_policy="omit")
        spearman = float(sp.correlation) if sp.correlation is not None else float("nan")
    else:
        spearman = float("nan")

    return {
        "r2": r2,
        "spearman": spearman,
        "mae": mae,
        "rmse": math.sqrt(mse),
        "pred_std": pred_std,
    }


def train_and_eval_exp(
    exp_name: str,
    vocab_size: int,
    ds_train: TensorDataset,
    ds_val: TensorDataset,
    args: argparse.Namespace,
    exclude_support: bool,
    use_extra_dots: bool,
    loss_type: str,
) -> Dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExperimentalMLP(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        n_champion_slots=len(CHAMPION_COLS),
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        exclude_support=exclude_support,
        use_extra_dots=use_extra_dots,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Setup loss functions
    if loss_type == "l1":
        def train_criterion(p, t, w):
            return (w * torch.abs(p - t)).mean()
        val_criterion = nn.L1Loss()
    elif loss_type == "huber":
        huber_loss = nn.HuberLoss(reduction="none", delta=0.1)
        def train_criterion(p, t, w):
            return (w * huber_loss(p, t)).mean()
        val_criterion = nn.HuberLoss(delta=0.1)
    else:
        train_criterion = weighted_mse_loss
        val_criterion = nn.MSELoss()

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size * 2, shuffle=False)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for champ_ids, side, y, w in loader_train:
            champ_ids = champ_ids.to(device)
            side = side.to(device)
            y = y.to(device)
            w = w.to(device)

            pred = model(champ_ids, side)
            loss = train_criterion(pred, y, w)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation (unweighted by pipeline protocol)
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for champ_ids, side, y, _ in loader_val:
                champ_ids = champ_ids.to(device)
                side = side.to(device)
                y = y.to(device)
                pred = model(champ_ids, side)
                val_loss_sum += val_criterion(pred, y).item() * len(y)
        val_loss = val_loss_sum / len(ds_val)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch - best_epoch >= args.patience:
            break

    elapsed = time.time() - t0
    
    # Load best weights for final eval
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for champ_ids, side, y, _ in loader_val:
            champ_ids = champ_ids.to(device)
            side = side.to(device)
            all_preds.append(model(champ_ids, side).cpu().numpy())
            all_true.append(y.numpy())
            
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)
    metrics = compute_metrics(y_true, y_pred)
    
    metrics["name"] = exp_name
    metrics["time"] = elapsed
    metrics["best_epoch"] = best_epoch
    metrics["loss_type"] = loss_type
    metrics["exclude_support"] = exclude_support
    metrics["use_extra_dots"] = use_extra_dots

    return metrics


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[Data] Loading Parquet splits...")
    df_train = pd.read_parquet(args.train)
    df_val = pd.read_parquet(args.val)

    if args.limit_rows is not None:
        print(f"[Smoke Test] Limiting to first {args.limit_rows} rows.")
        df_train = df_train.head(args.limit_rows).copy()
        df_val = df_val.head(args.limit_rows).copy()

    vocab = build_champion_vocab(df_train)
    vocab_size = len(vocab) + 1

    ds_train = make_dataset(df_train, vocab, TARGET_COL)
    ds_val = make_dataset(df_val, vocab, TARGET_COL)

    print(f"[Data] train={len(df_train):,}  val={len(df_val):,}  vocab_size={vocab_size}")

    experiments = [
        {
            "name": "1. Baseline MLP (MSE Loss, keep support)",
            "exclude_support": False, "use_extra_dots": False, "loss_type": "mse"
        },
        {
            "name": "2. MLP (MSE Loss, Exclude Support - Exp A')",
            "exclude_support": True, "use_extra_dots": False, "loss_type": "mse"
        },
        {
            "name": "3. MLP (L1 Loss / MAE, keep support - Exp B')",
            "exclude_support": False, "use_extra_dots": False, "loss_type": "l1"
        },
        {
            "name": "4. MLP (Huber Loss, keep support - Exp B')",
            "exclude_support": False, "use_extra_dots": False, "loss_type": "huber"
        },
        {
            "name": "5. MLP (Huber Loss, Exclude Support - Exp A'+B')",
            "exclude_support": True, "use_extra_dots": False, "loss_type": "huber"
        },
        {
            "name": "6. MLP (MSE Loss + extra dots - Exp C')",
            "exclude_support": False, "use_extra_dots": True, "loss_type": "mse"
        },
        {
            "name": "7. MLP (Huber Loss + extra dots - Exp B'+C')",
            "exclude_support": False, "use_extra_dots": True, "loss_type": "huber"
        },
        {
            "name": "8. MLP Combinado (Huber + Exclude + extra dots)",
            "exclude_support": True, "use_extra_dots": True, "loss_type": "huber"
        },
    ]

    results_table = []

    for idx, exp in enumerate(experiments, start=1):
        print(f"\n--- Running MLP Experiment {idx}/{len(experiments)}: {exp['name']} ---")
        try:
            metrics = train_and_eval_exp(
                exp_name=exp["name"],
                vocab_size=vocab_size,
                ds_train=ds_train,
                ds_val=ds_val,
                args=args,
                exclude_support=exp["exclude_support"],
                use_extra_dots=exp["use_extra_dots"],
                loss_type=exp["loss_type"],
            )
            results_table.append(metrics)
            print(f"  Result: R2={metrics['r2']:.5f} | Spearman={metrics['spearman']:.5f} | MAE={metrics['mae']:.5f} | Pred_Std={metrics['pred_std']:.5f} | Best Epoch={metrics['best_epoch']} | Time={metrics['time']:.1f}s")
        except Exception as e:
            print(f"  [Error] Experiment failed: {e}")

    # Save output summary
    out_path = outdir / "mlp_experiments_results.json"
    out_path.write_text(json.dumps(results_table, indent=2), encoding="utf-8")

    # Print Markdown Summary
    print("\n" + "="*80)
    print(" MLP EXPERIMENTS SUMMARY (VAL SET EVALUATION)")
    print("="*80)
    print(f"| {'Experiment Configuration':<40} | {'R2':^7} | {'Spearman':^8} | {'MAE':^7} | {'Pred Std':^8} | {'Best Ep':^7} |")
    print(f"|{'-'*42}|{'-'*9}|{'-'*10}|{'-'*9}|{'-'*10}|{'-'*9}|")
    for r in results_table:
        print(f"| {r['name']:<40} | {r['r2']:>7.5f} | {r['spearman']:>8.5f} | {r['mae']:>7.5f} | {r['pred_std']:>8.5f} | {r['best_epoch']:^7} |")
    print("="*80)
    print(f"Results saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
