#!/usr/bin/env python3
"""
04b_train_mlp_embed.py -- MLP with learned champion embeddings.

Instead of one-hot, each champion ID is mapped to a dense vector of
dimension `embed_dim`. The 10 champion embeddings are concatenated with
the side feature and passed through a standard MLP.

Key differences vs 04a (one-hot):
  - Input dim: 10 * embed_dim + 1   (vs 10 * vocab_size + 1  for one-hot)
  - The embedding layer learns champion similarities automatically
  - After training, we can inspect the embedding space to see if the model
    discovered the same archetype clusters as domain experts

Trains on both raw and quantile targets for comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

from mlp_losses import weighted_mse_loss


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN = str(REPO_ROOT / "final" / "data" / "training" / "train.parquet")
DEFAULT_VAL = str(REPO_ROOT / "final" / "data" / "training" / "val.parquet")
DEFAULT_OUTDIR = str(REPO_ROOT / "final" / "models" / "mlp_embed")

TARGET_COL = "support_roam_score"
QUANTILE_COL = "support_roam_score_quantile"

ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")
CHAMPION_COLS = [f"{s}_{r}_champion_id" for s in SIDES for r in ROLE_KEYS]
INPUT_FEATURE_COLUMNS = CHAMPION_COLS + ["side"]
FEATURE_PROTOCOL_ID = "draft_10_champions_side"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MLP with champion embeddings.")
    p.add_argument("--train", default=DEFAULT_TRAIN)
    p.add_argument("--val", default=DEFAULT_VAL)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--embed-dim", type=int, default=16,
                    help="Embedding dimension per champion (default: 16)")
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


# ──────────────────────── Data ────────────────────────


def build_champion_vocab(df_train: pd.DataFrame) -> Dict[int, int]:
    """Map raw champion_id -> contiguous index. Index 0 is reserved for unknown."""
    all_ids = set()
    for col in CHAMPION_COLS:
        if col in df_train.columns:
            all_ids.update(df_train[col].dropna().astype(int).unique())
    vocab = {int(cid): idx + 1 for idx, cid in enumerate(sorted(all_ids))}
    return vocab


def encode_ids(df: pd.DataFrame, vocab: Dict[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      champion_indices: (n, 10) int64 array of vocab indices
      side:             (n, 1)  float32 array
    """
    n = len(df)
    champ_idx = np.zeros((n, len(CHAMPION_COLS)), dtype=np.int64)
    for i, col in enumerate(CHAMPION_COLS):
        if col in df.columns:
            champ_idx[:, i] = (
                df[col].fillna(-1).astype(int)
                .map(lambda x: vocab.get(x, 0))
                .to_numpy()
            )
    side = (df["side"].map({"blue": 0.0, "red": 1.0}).fillna(0.5)
            .to_numpy(dtype=np.float32).reshape(-1, 1))
    return champ_idx, side


def make_datasets(
    df_train: pd.DataFrame, df_val: pd.DataFrame, vocab: Dict[int, int],
    target_col: str
) -> Tuple[TensorDataset, TensorDataset]:
    idx_tr, side_tr = encode_ids(df_train, vocab)
    idx_va, side_va = encode_ids(df_val, vocab)
    y_tr = df_train[target_col].to_numpy(dtype=np.float32)
    y_va = df_val[target_col].to_numpy(dtype=np.float32)

    # Include sample_weight if available
    if "sample_weight" in df_train.columns:
        w_tr = df_train["sample_weight"].to_numpy(dtype=np.float32)
    else:
        w_tr = np.ones(len(df_train), dtype=np.float32)
    if "sample_weight" in df_val.columns:
        w_va = df_val["sample_weight"].to_numpy(dtype=np.float32)
    else:
        w_va = np.ones(len(df_val), dtype=np.float32)

    ds_train = TensorDataset(
        torch.from_numpy(idx_tr),
        torch.from_numpy(side_tr),
        torch.from_numpy(y_tr),
        torch.from_numpy(w_tr),
    )
    ds_val = TensorDataset(
        torch.from_numpy(idx_va),
        torch.from_numpy(side_va),
        torch.from_numpy(y_va),
        torch.from_numpy(w_va),
    )
    return ds_train, ds_val


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


# ──────────────────────── Model ────────────────────────


class MLPEmbed(nn.Module):
    """
    Architecture:
      champion_ids (10,) -> Embedding(vocab_size, embed_dim) -> (10 * embed_dim,)
      concat with side (1,)
      -> Linear layers with BN + ReLU + Dropout
      -> Output (1,)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        n_champion_slots: int,
        hidden_dims: List[int],
        dropout: float,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.n_slots = n_champion_slots

        input_dim = n_champion_slots * embed_dim + 1  # +1 for side

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, champ_ids: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        # champ_ids: (batch, 10)  side: (batch, 1)
        emb = self.embed(champ_ids)          # (batch, 10, embed_dim)
        emb_flat = emb.view(emb.size(0), -1) # (batch, 10 * embed_dim)
        x = torch.cat([emb_flat, side], dim=1)
        return self.head(x).squeeze(-1)

    def get_embeddings(self) -> np.ndarray:
        """Extract the embedding matrix as numpy array."""
        return self.embed.weight.detach().cpu().numpy()


# ──────────────────────── Training ────────────────────────


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str, label: str,
    n_train: int, elapsed: float, epoch: int
) -> Dict[str, Any]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else float("nan")
    sp = spearmanr(y_true, y_pred, nan_policy="omit")
    return {
        "model": model_name,
        "target": label,
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "r2": r2,
        "pearson_corr": pearson,
        "spearman_corr": float(sp.correlation) if sp.correlation is not None else float("nan"),
        "pred_std": float(np.std(y_pred)),
        "target_std": float(np.std(y_true)),
        "compression_ratio": float(np.std(y_pred) / np.std(y_true)) if np.std(y_true) > 0 else float("nan"),
        "n_train": n_train,
        "n_eval": int(len(y_true)),
        "eval_split": "val",
        "training_seconds": elapsed,
        "best_epoch": epoch,
    }


def train_model(
    model: nn.Module, ds_train: TensorDataset, ds_val: TensorDataset,
    args: argparse.Namespace, target_label: str, outdir: Path
) -> Tuple[Dict[str, Any], nn.Module, List[Dict[str, Any]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size * 2)

    # Check if we have non-trivial weights
    has_weights = len(ds_train.tensors) >= 4
    if has_weights:
        w_mean = ds_train.tensors[3].mean().item()
        print(f"  [{target_label}] Using sample_weight (mean={w_mean:.3f})")
    else:
        print(f"  [{target_label}] No sample_weight")

    wandb_run = None
    if getattr(args, "use_wandb", False):
        try:
            import wandb
            wandb_run = wandb.init(
                project="tfg-support-roaming",
                name=f"MLP_Embed_{target_label}",
                config={
                    "model": "MLP_Embed",
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
                    "has_weights": has_weights,
                },
                reinit=True
            )
        except ImportError:
            print("[Warning] wandb is not installed. Running without wandb.")

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    history: List[Dict[str, Any]] = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        for batch in loader_train:
            champ_ids = batch[0].to(device)
            side = batch[1].to(device)
            y = batch[2].to(device)
            w = batch[3].to(device) if has_weights else None

            pred = model(champ_ids, side)
            if w is not None:
                loss = weighted_mse_loss(pred, y, w)
                train_loss_sum += (w * (pred.detach() - y) ** 2).sum().item()
                train_weight_sum += w.sum().item()
            else:
                loss = criterion(pred, y)
                train_loss_sum += ((pred.detach() - y) ** 2).sum().item()
                train_weight_sum += float(len(y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = train_loss_sum / max(train_weight_sum, 1e-8)

        # Validation and checkpoint selection stay unweighted by design.
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in loader_val:
                champ_ids = batch[0].to(device)
                side = batch[1].to(device)
                y = batch[2].to(device)
                pred = model(champ_ids, side)
                val_loss_sum += criterion(pred, y).item() * len(y)
        val_loss = val_loss_sum / len(ds_val)
        scheduler.step(val_loss)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        lr = float(optimizer.param_groups[0]["lr"])
        history.append({
            "target": target_label,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "is_best": is_best,
        })

        if wandb_run:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr": lr,
            })

        if epoch % 5 == 0 or epoch == 1 or is_best:
            print(f"  [{target_label}] epoch {epoch:3d}  train={train_loss:.5f}  "
                  f"val={val_loss:.5f}  lr={lr:.1e}")

        if epoch - best_epoch >= args.patience:
            print(f"  [{target_label}] Early stopping at epoch {epoch} (best={best_epoch})")
            break

    elapsed = time.time() - t0
    if best_state is None:
        raise RuntimeError(f"No best state recorded for target={target_label}")
    model.load_state_dict(best_state)
    model.eval()

    # Final predictions
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in loader_val:
            champ_ids = batch[0].to(device)
            side = batch[1].to(device)
            all_preds.append(model(champ_ids, side).cpu().numpy())
            all_true.append(batch[2].numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    metrics = compute_metrics(
        y_true, y_pred, f"mlp_embed_{target_label}", target_label,
        n_train=len(ds_train), elapsed=elapsed, epoch=best_epoch,
    )
    print(f"  [{target_label}] R2={metrics['r2']:.4f}  Spearman={metrics['spearman_corr']:.4f}  "
          f"pred_std={metrics['pred_std']:.4f}  best_epoch={best_epoch}")

    # Save model
    torch.save(best_state, outdir / f"mlp_embed_{target_label}.pt")

    if wandb_run:
        for k, v in metrics.items():
            if k not in ["model", "target", "eval_split"]:
                wandb_run.summary[f"val/{k}"] = v
        wandb_run.finish()

    return metrics, model, history


# ──────────────────────── Embedding Analysis ────────────────────────


def save_embedding_analysis(
    model: MLPEmbed, vocab: Dict[int, int], outdir: Path,
    target_label: str, archetypes_path: Path
) -> None:
    """Save embedding matrix and optional archetype-colored t-SNE data."""
    embeddings = model.get_embeddings()  # (vocab_size, embed_dim)

    # Build reverse vocab: index -> champion_id
    idx_to_cid = {v: k for k, v in vocab.items()}

    # Load archetypes if available
    arch_map = {}
    if archetypes_path.exists():
        raw = json.loads(archetypes_path.read_text(encoding="utf-8"))
        champs = raw.get("champions", {})
        for cid_str, info in champs.items():
            arch_map[int(cid_str)] = {
                "name": info.get("name", f"champ_{cid_str}"),
                "archetype": info.get("support", info.get("generic", "unknown")),
            }

    # Save per-champion embedding data
    embed_data = []
    for idx in range(1, len(embeddings)):  # skip padding at 0
        cid = int(idx_to_cid.get(idx, -1))
        info = arch_map.get(cid, {"name": f"champ_{cid}", "archetype": "unknown"})
        embed_data.append({
            "champion_id": cid,
            "name": info["name"],
            "archetype": info["archetype"],
            "vocab_index": idx,
            "embedding": [float(v) for v in embeddings[idx]],
        })

    (outdir / f"embeddings_{target_label}.json").write_text(
        json.dumps(embed_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  [{target_label}] Saved {len(embed_data)} champion embeddings")


# ──────────────────────── Main ────────────────────────


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
    input_dim = n_slots * args.embed_dim + 1
    print(f"[Vocab] {len(vocab)} champions  embed_dim={args.embed_dim}  "
          f"input_dim={input_dim}  hidden={args.hidden_dims}")

    archetypes_path = REPO_ROOT / "final" / "data" / "champion_archetypes.json"
    results = []
    history_rows: List[Dict[str, Any]] = []

    # --- Raw target ---
    ds_train, ds_val = make_datasets(df_train, df_val, vocab, TARGET_COL)
    model = MLPEmbed(vocab_size, args.embed_dim, n_slots, args.hidden_dims, args.dropout)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] {n_params:,} parameters")
    m, trained_model, history = train_model(model, ds_train, ds_val, args, "raw", outdir)
    results.append(m)
    history_rows.extend(history)
    save_embedding_analysis(trained_model, vocab, outdir, "raw", archetypes_path)

    # --- Quantile target ---
    if QUANTILE_COL in df_train.columns:
        ds_train_q, ds_val_q = make_datasets(df_train, df_val, vocab, QUANTILE_COL)
        model_q = MLPEmbed(vocab_size, args.embed_dim, n_slots, args.hidden_dims, args.dropout)
        m_q, trained_q, history_q = train_model(model_q, ds_train_q, ds_val_q, args, "quantile", outdir)
        results.append(m_q)
        history_rows.extend(history_q)
        save_embedding_analysis(trained_q, vocab, outdir, "quantile", archetypes_path)

    # Save config + metrics
    config = {
        "model_type": "mlp_embed_shared",
        "feature_set": "main",
        "feature_protocol_id": FEATURE_PROTOCOL_ID,
        "champion_cols": CHAMPION_COLS,
        "input_feature_columns": INPUT_FEATURE_COLUMNS,
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
            "side_mapping": {"blue": 0.0, "red": 1.0},
            "sample_weight_column": "sample_weight",
            "used_sample_weight": "sample_weight" in df_train.columns,
        },
        outdir / "preprocess.joblib",
    )

    print(f"\n[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
