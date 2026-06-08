#!/usr/bin/env python3
"""
04d_train_mlp_per_role_huber.py -- MLP with per-role champion embeddings trained with Huber Loss.

This script is identical to 04c but uses Huber Loss (delta=0.1) instead of MSE
to combat chaotic match noise. Evaluation metrics (R2, MAE, Spearman) remain 
comparable with standard MSE models.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = str(REPO_ROOT / "final" / "data" / "training" / "train.parquet")
DEFAULT_VAL = str(REPO_ROOT / "final" / "data" / "training" / "val.parquet")
DEFAULT_OUTDIR = str(REPO_ROOT / "final" / "models" / "mlp_per_role_huber")

TARGET_COL = "support_roam_score"
QUANTILE_COL = "support_roam_score_quantile"

ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")
CHAMPION_COLS = [f"{s}_{r}_champion_id" for s in SIDES for r in ROLE_KEYS]
SLOT_NAMES = [col[: -len("_champion_id")] for col in CHAMPION_COLS]
INPUT_FEATURE_COLUMNS = CHAMPION_COLS + ["side"]
FEATURE_PROTOCOL_ID = "draft_10_champions_side"

ALLY_UTILITY_SLOT = SLOT_NAMES.index("ally_utility")
ENEMY_UTILITY_SLOT = SLOT_NAMES.index("enemy_utility")
ALLY_BOTTOM_SLOT = SLOT_NAMES.index("ally_bottom")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MLP with per-role champion embeddings using Huber Loss.")
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--val", default=DEFAULT_VAL)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--allow-missing-sample-weight",
        action="store_true",
        help="Allow unweighted training if sample_weight is absent.",
    )
    p.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases to track training metrics.",
    )
    return p.parse_args()


def build_champion_vocab(df_train: pd.DataFrame) -> Dict[int, int]:
    """Map raw champion_id -> contiguous index. Index 0 is reserved for unknown."""
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
    if "sample_weight" in df.columns:
        w = df["sample_weight"].to_numpy(dtype=np.float32)
    else:
        w = np.ones(len(df), dtype=np.float32)

    return TensorDataset(
        torch.from_numpy(encode_champion_ids(df, vocab)),
        torch.from_numpy(encode_side(df)),
        torch.from_numpy(y),
        torch.from_numpy(w),
    )


def require_sample_weight(df: pd.DataFrame, allow_missing: bool) -> None:
    if "sample_weight" in df.columns:
        return
    if allow_missing:
        print("[Weights] No sample_weight column found - using unit weights")
        return
    raise SystemExit(
        "[Weights] Missing required sample_weight column. "
        "Use --allow-missing-sample-weight only for legacy/debug runs."
    )


class MLPPerRoleInteractions(nn.Module):
    """MLP over ten slot-specific embeddings, side, and two dot interactions."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_champion_slots: int,
        hidden_dims: List[int],
        dropout: float,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_champion_slots = n_champion_slots
        self.slot_embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size, embed_dim, padding_idx=0) for _ in range(n_champion_slots)]
        )

        input_dim = n_champion_slots * embed_dim + 1 + 2
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

        x = torch.cat(
            [
                embeddings.flatten(start_dim=1),
                side,
                support_vs_support,
                support_adc,
            ],
            dim=1,
        )
        return self.head(x).squeeze(-1)

    def get_slot_embeddings(self, slot_idx: int) -> np.ndarray:
        return self.slot_embeddings[slot_idx].weight.detach().cpu().numpy()


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
    model: MLPPerRoleInteractions,
    ds_train: TensorDataset,
    ds_val: TensorDataset,
    args: argparse.Namespace,
    target_label: str,
    outdir: Path,
) -> Tuple[Dict[str, Any], MLPPerRoleInteractions, List[Dict[str, Any]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Use Huber Loss (delta=0.1) as the robust optimization criterion
    huber = nn.HuberLoss(reduction="none", delta=0.1)
    def weighted_huber_loss(pred_t, target_t, weights_t):
        return (weights_t * huber(pred_t, target_t)).mean()

    criterion_val = nn.HuberLoss(delta=0.1)

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size * 2, shuffle=False)

    w_mean = ds_train.tensors[3].mean().item()
    print(f"  [{target_label}] Using sample_weight (mean={w_mean:.3f}) with Huber Loss")

    wandb_run = None
    if getattr(args, "use_wandb", False):
        try:
            import wandb
            wandb_run = wandb.init(
                project="tfg-support-roaming",
                name=f"MLP_Per_Role_Huber_{target_label}",
                config={
                    "model": "MLP_Per_Role_Huber",
                    "target": target_label,
                    "embed_dim": args.embed_dim,
                    "hidden_dims": args.hidden_dims,
                    "dropout": args.dropout,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "seed": args.seed,
                    "w_mean": w_mean,
                    "huber_delta": 0.1,
                },
                reinit=True
            )
        except ImportError:
            print("[Warning] wandb is not installed. Running without wandb.")

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, Any]] = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        for champ_ids, side, y, w in loader_train:
            champ_ids = champ_ids.to(device)
            side = side.to(device)
            y = y.to(device)
            w = w.to(device)

            pred = model(champ_ids, side)
            loss = weighted_huber_loss(pred, y, w)
            
            # Record unweighted Huber loss for history consistency
            with torch.no_grad():
                train_loss_sum += (w * huber(pred, y)).sum().item()
            train_weight_sum += w.sum().item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss = train_loss_sum / max(train_weight_sum, 1e-8)

        # Validation (unweighted Huber loss)
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for champ_ids, side, y, _w in loader_val:
                champ_ids = champ_ids.to(device)
                side = side.to(device)
                y = y.to(device)
                pred = model(champ_ids, side)
                val_loss_sum += criterion_val(pred, y).item() * len(y)
        val_loss = val_loss_sum / len(ds_val)
        scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

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

        if wandb_run:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr": lr,
            })

        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(
                f"  [{target_label}] epoch {epoch:3d}  train_huber={train_loss:.5f}  "
                f"val_huber={val_loss:.5f}  lr={lr:.1e}"
            )

        if epoch - best_epoch >= args.patience:
            print(f"  [{target_label}] Early stopping at epoch {epoch} (best={best_epoch})")
            break

    elapsed = time.time() - t0
    if best_state is None:
        raise RuntimeError(f"No best state recorded for target={target_label}")
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_true = [], []
    with torch.no_grad():
        for champ_ids, side, y, _w in loader_val:
            champ_ids = champ_ids.to(device)
            side = side.to(device)
            all_preds.append(model(champ_ids, side).cpu().numpy())
            all_true.append(y.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    # Standard metrics are computed over MSE, MAE, Spearman to maintain global comparison
    metrics = compute_metrics(
        y_true,
        y_pred,
        f"mlp_per_role_huber_{target_label}",
        target_label,
        n_train=len(ds_train),
        elapsed=elapsed,
        epoch=best_epoch,
    )
    print(
        f"  [{target_label}] R2={metrics['r2']:.4f}  "
        f"Spearman={metrics['spearman_corr']:.4f}  "
        f"pred_std={metrics['pred_std']:.4f}  best_epoch={best_epoch}"
    )

    torch.save(best_state, outdir / f"mlp_per_role_huber_{target_label}.pt")

    if wandb_run:
        for k, v in metrics.items():
            if k not in ["model", "target", "eval_split"]:
                wandb_run.summary[f"val/{k}"] = v
        wandb_run.finish()

    return metrics, model, history


def load_archetype_metadata(archetypes_path: Path) -> Dict[int, Dict[str, str]]:
    if not archetypes_path.exists():
        return {}
    raw = json.loads(archetypes_path.read_text(encoding="utf-8"))
    out: Dict[int, Dict[str, str]] = {}
    for cid_str, info in raw.get("champions", {}).items():
        cid = int(cid_str)
        out[cid] = {
            "name": str(info.get("name", f"champ_{cid}")),
            "archetype": str(info.get("support", info.get("generic", "unknown"))),
        }
    return out


def save_slot_embeddings(
    model: MLPPerRoleInteractions,
    vocab: Dict[int, int],
    outdir: Path,
    target_label: str,
    archetypes_path: Path,
) -> None:
    idx_to_cid = {v: k for k, v in vocab.items()}
    metadata = load_archetype_metadata(archetypes_path)

    for slot_idx, slot_name in enumerate(SLOT_NAMES):
        embeddings = model.get_slot_embeddings(slot_idx)
        rows = []
        for idx in range(1, len(embeddings)):
            cid = int(idx_to_cid.get(idx, -1))
            info = metadata.get(cid, {"name": f"champ_{cid}", "archetype": "unknown"})
            rows.append(
                {
                    "champion_id": cid,
                    "name": info["name"],
                    "archetype": info["archetype"],
                    "slot": slot_name,
                    "vocab_index": idx,
                    "embedding": [float(v) for v in embeddings[idx]],
                }
            )
        path = outdir / f"embeddings_{target_label}_{slot_name}.json"
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

        if slot_name == "ally_utility":
            (outdir / f"embeddings_{target_label}.json").write_text(
                json.dumps(rows, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    print(f"  [{target_label}] Saved per-slot embeddings ({len(SLOT_NAMES)} slots)")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df_train = pd.read_parquet(args.train)
    df_val = pd.read_parquet(args.val)
    require_sample_weight(df_train, args.allow_missing_sample_weight)
    print(f"[Data] train={len(df_train):,}  val={len(df_val):,}")

    vocab = build_champion_vocab(df_train)
    vocab_size = len(vocab) + 1
    n_slots = len(CHAMPION_COLS)
    input_dim = n_slots * args.embed_dim + 1 + 2
    print(
        f"[Vocab] {len(vocab)} champions  embed_dim={args.embed_dim}  "
        f"input_dim={input_dim}  hidden={args.hidden_dims}"
    )

    archetypes_path = REPO_ROOT / "final" / "data" / "champion_archetypes.json"
    results: List[Dict[str, Any]] = []
    history_rows: List[Dict[str, Any]] = []

    ds_train = make_dataset(df_train, vocab, TARGET_COL)
    ds_val = make_dataset(df_val, vocab, TARGET_COL)
    model = MLPPerRoleInteractions(vocab_size, args.embed_dim, n_slots, args.hidden_dims, args.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {n_params:,} parameters")
    metrics, trained_model, history = train_model(model, ds_train, ds_val, args, "raw", outdir)
    results.append(metrics)
    history_rows.extend(history)
    save_slot_embeddings(trained_model, vocab, outdir, "raw", archetypes_path)

    if QUANTILE_COL in df_train.columns:
        ds_train_q = make_dataset(df_train, vocab, QUANTILE_COL)
        ds_val_q = make_dataset(df_val, vocab, QUANTILE_COL)
        model_q = MLPPerRoleInteractions(vocab_size, args.embed_dim, n_slots, args.hidden_dims, args.dropout)
        metrics_q, trained_model_q, history_q = train_model(
            model_q, ds_train_q, ds_val_q, args, "quantile", outdir
        )
        results.append(metrics_q)
        history_rows.extend(history_q)
        save_slot_embeddings(trained_model_q, vocab, outdir, "quantile", archetypes_path)

    config = {
        "model_type": "mlp_per_role_huber_interactions",
        "feature_set": "main",
        "feature_protocol_id": FEATURE_PROTOCOL_ID,
        "champion_cols": CHAMPION_COLS,
        "input_feature_columns": INPUT_FEATURE_COLUMNS,
        "slot_names": SLOT_NAMES,
        "interaction_features": [
            "dot(ally_utility, enemy_utility)",
            "dot(ally_utility, ally_bottom)",
        ],
        "vocab_size": vocab_size,
        "embed_dim": args.embed_dim,
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
        "sample_weight_column": "sample_weight",
        "used_sample_weight": "sample_weight" in df_train.columns,
        "loss_type": "huber",
        "huber_delta": 0.1,
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
            "input_feature_columns": INPUT_FEATURE_COLUMNS,
            "feature_protocol_id": FEATURE_PROTOCOL_ID,
            "slot_names": SLOT_NAMES,
            "side_mapping": {"blue": 0.0, "red": 1.0},
            "interaction_features": config["interaction_features"],
            "sample_weight_column": "sample_weight",
            "used_sample_weight": "sample_weight" in df_train.columns,
        },
        outdir / "preprocess.joblib",
    )

    print(f"\n[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
