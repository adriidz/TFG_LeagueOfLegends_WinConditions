#!/usr/bin/env python3
"""
Train a support-only continuous regression model.

Input
-----
`model_input_support_regression*.parquet`, produced by the support-only model
input builder. The trainer uses draft categorical features, applies One-Hot
Encoding, and predicts one continuous target: `support_roam_score`.

The model is intentionally simple and easy to defend in the report:
OneHotEncoder -> MLP -> scalar output, optimized with MSELoss.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DEFAULT_INPUT_PATH = os.path.join("ProgresoActual", "data", "training", "model_input_support_regression.parquet")
DEFAULT_OUTPUT_DIR = os.path.join("ProgresoActual", "models", "support_mlp_regression")
DEFAULT_GROUP_COL = "match_id"
DEFAULT_TARGET_COL = "support_roam_score"

ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")

FEATURE_GROUP_DEFS: Dict[str, Dict[str, Any]] = {
    "champions": {
        "columns": [f"{s}_{r}_champion_id" for s in SIDES for r in ROLE_KEYS],
        "description": "Champion picks de ambos equipos.",
    },
    "summoner_spells": {
        "columns": [f"{s}_{r}_summoner{i}_id" for s in SIDES for r in ROLE_KEYS for i in (1, 2)],
        "description": "Hechizos de invocador.",
    },
    "keystones": {
        "columns": [f"{s}_{r}_keystone_id" for s in SIDES for r in ROLE_KEYS],
        "description": "Runas keystone.",
    },
    "rune_styles": {
        "columns": [
            f"{s}_{r}_{style}_style_id"
            for s in SIDES for r in ROLE_KEYS for style in ("primary", "sub")
        ],
        "description": "Arboles de runas primario/secundario.",
    },
    "bans": {
        "columns": [f"{s}_ban_{i}_champion_id" for s in SIDES for i in range(1, 6)],
        "description": "Bans de ambos equipos.",
    },
    "context": {
        "columns": ["side"],
        "description": "Side blue/red.",
    },
}

ALL_GROUP_NAMES = list(FEATURE_GROUP_DEFS.keys())
ABLATION_PRESETS = {
    "minimal": ["champions", "context"],
    "standard": ["champions", "summoner_spells", "context"],
    "draft_plus_runes": ["champions", "summoner_spells", "keystones", "rune_styles", "context"],
    "all": ALL_GROUP_NAMES,
}


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_window_tag(max_minute: float) -> str:
    return f"m{int(round(float(max_minute))):02d}"


def apply_window_suffix(path: str, max_minute: Optional[float]) -> str:
    if max_minute is None:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_{format_window_tag(max_minute)}{ext}"


def apply_sample_suffix(path: str, frac: Optional[float]) -> str:
    if frac is None or frac <= 0.0 or frac >= 1.0:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_sample{int(round(frac * 100))}{ext}"


def get_target_frac(args_frac: Optional[float]) -> Optional[float]:
    if args_frac is not None:
        return args_frac
    env_frac = os.getenv("TFG_SAMPLE_FRAC")
    if env_frac:
        try:
            return float(env_frac)
        except ValueError:
            return None
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Support-only MLP regression with One-Hot + MSELoss.")
    p.add_argument("--input", default=DEFAULT_INPUT_PATH)
    p.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--score-max-minute", type=float, default=None)
    p.add_argument("--feature-groups", nargs="*", default=["standard"],
                   help=f"Feature groups or one preset. Groups={ALL_GROUP_NAMES}; presets={list(ABLATION_PRESETS)}")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden1", type=int, default=256)
    p.add_argument("--hidden2", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.20)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rows", type=int, default=0, help="Optional smoke-test row cap after loading.")
    p.add_argument("--support-config-json", default=None,
                   help="Optional selected_support_score_config.json to record the label heuristic.")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="tfg-support-regression")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    p.add_argument("--run-name", default=None)
    p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("--skip-diagnostics-plots", action="store_true")
    return p.parse_args()


def select_active_groups(feature_groups_arg: Optional[List[str]]) -> List[str]:
    if not feature_groups_arg:
        return list(ABLATION_PRESETS["standard"])
    if len(feature_groups_arg) == 1 and feature_groups_arg[0] in ABLATION_PRESETS:
        return list(ABLATION_PRESETS[feature_groups_arg[0]])
    active = [g for g in feature_groups_arg if g in FEATURE_GROUP_DEFS]
    if not active:
        raise SystemExit(f"No valid feature groups selected: {feature_groups_arg}")
    return active


def resolve_paths(args: argparse.Namespace) -> Tuple[str, str]:
    frac = get_target_frac(args.sample_frac)
    in_path = args.input
    out_dir = args.outdir
    if frac is not None and 0.0 < frac < 1.0:
        in_path = apply_sample_suffix(in_path, frac)
        out_dir = apply_sample_suffix(out_dir, frac)
    if args.score_max_minute is not None:
        in_path = apply_window_suffix(in_path, args.score_max_minute)
        out_dir = apply_window_suffix(out_dir, args.score_max_minute)
    return in_path, out_dir


def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def prepare_features(df: pd.DataFrame, active_groups: List[str]) -> List[str]:
    feature_columns: List[str] = []
    for group_name in active_groups:
        feature_columns.extend([c for c in FEATURE_GROUP_DEFS[group_name]["columns"] if c in df.columns])
    feature_columns = list(dict.fromkeys(feature_columns))
    if not feature_columns:
        raise SystemExit("No feature columns found for selected groups.")
    return feature_columns


def categorical_frame(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df[columns].copy()
    for col in columns:
        out[col] = out[col].where(out[col].notna(), "__MISSING__").astype(str)
    return out


class SupportMLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden1),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden2),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
) -> float:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_rows = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if train:
            optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * len(yb)
        total_rows += len(yb)
    return total_loss / max(total_rows, 1)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(torch.from_numpy(X.astype(np.float32)), batch_size=batch_size, shuffle=False)
    model.eval()
    parts: List[np.ndarray] = []
    for xb in loader:
        xb = xb.to(device)
        parts.append(model(xb).detach().cpu().numpy())
    return np.concatenate(parts) if parts else np.asarray([], dtype=np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = math.sqrt(mse)
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 0 and np.std(y_pred) > 0 else float("nan")
    spear = spearmanr(y_true, y_pred, nan_policy="omit").correlation
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson_corr": pearson,
        "spearman_corr": float(spear) if spear is not None else float("nan"),
    }


def init_wandb(args: argparse.Namespace, config: Dict[str, Any]):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is not installed. Install requirements or run without --wandb.") from exc
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=args.run_name,
        tags=args.tags,
        config=config,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    active_groups = select_active_groups(args.feature_groups)
    in_path, out_dir = resolve_paths(args)

    print(f"[Input] {os.path.abspath(in_path)}")
    if not os.path.exists(in_path):
        raise SystemExit(f"Input parquet not found: {in_path}")
    df = pd.read_parquet(in_path)
    if args.max_rows and args.max_rows > 0:
        df = df.sample(n=min(args.max_rows, len(df)), random_state=args.seed).reset_index(drop=True)
        print(f"[Smoke] Using max_rows={len(df)}")

    if args.target_col not in df.columns:
        raise SystemExit(f"Missing target column: {args.target_col}")
    if DEFAULT_GROUP_COL not in df.columns:
        raise SystemExit(f"Missing group column: {DEFAULT_GROUP_COL}")

    before = len(df)
    df = df[df[args.target_col].notna()].copy()
    df[args.target_col] = pd.to_numeric(df[args.target_col], errors="coerce")
    df = df[df[args.target_col].between(0.0, 1.0, inclusive="both")].copy()
    print(f"[Filter] valid target rows: {len(df)} (removed {before - len(df)})")
    if len(df) < 10:
        raise SystemExit("Not enough rows to train.")

    feature_columns = prepare_features(df, active_groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(df, groups=df[DEFAULT_GROUP_COL]))
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()

    ohe = make_onehot_encoder()
    X_train = ohe.fit_transform(categorical_frame(df_train, feature_columns))
    X_val = ohe.transform(categorical_frame(df_val, feature_columns))
    y_train = df_train[args.target_col].to_numpy(dtype=np.float32)
    y_val = df_val[args.target_col].to_numpy(dtype=np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SupportMLP(
        input_dim=int(X_train.shape[1]),
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, args.batch_size, shuffle=False)

    support_config: Dict[str, Any] = {}
    if args.support_config_json and os.path.exists(args.support_config_json):
        with open(args.support_config_json, "r", encoding="utf-8") as f:
            support_config = json.load(f)

    run_config = {
        "input_path": os.path.abspath(in_path),
        "out_dir": os.path.abspath(out_dir),
        "target_col": args.target_col,
        "feature_groups": active_groups,
        "feature_columns": feature_columns,
        "onehot_dim": int(X_train.shape[1]),
        "n_train": int(len(df_train)),
        "n_val": int(len(df_val)),
        "val_size": args.val_size,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden1": args.hidden1,
        "hidden2": args.hidden2,
        "dropout": args.dropout,
        "score_max_minute": args.score_max_minute,
        "sample_frac": get_target_frac(args.sample_frac),
        "support_score_config": support_config,
    }
    wb_run = init_wandb(args, run_config)

    ensure_dir(out_dir)
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history: List[Dict[str, Any]] = []
    best_model_path = os.path.join(out_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, device, criterion, optimizer=None)
        row = {"epoch": epoch, "train_mse_loss": train_loss, "val_mse_loss": val_loss}
        history.append(row)
        if wb_run is not None:
            wb_run.log(row, step=epoch)
        print(f"epoch={epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"[Early stop] patience reached at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    y_pred = predict(model, X_val, args.batch_size, device)
    y_pred = np.clip(y_pred, 0.0, 1.0)
    metrics = compute_metrics(y_val, y_pred)
    metrics["best_epoch"] = int(best_epoch)
    metrics["best_val_mse_loss"] = float(best_val)

    predictions = df_val[["match_id", "team_id"] + [c for c in ("side", "patch") if c in df_val.columns]].copy()
    predictions[f"true_{args.target_col}"] = y_val
    predictions[f"pred_{args.target_col}"] = y_pred
    predictions[f"abs_error_{args.target_col}"] = np.abs(y_val - y_pred)

    pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
    predictions.to_parquet(os.path.join(out_dir, "validation_predictions.parquet"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)
    joblib.dump({"onehot": ohe, "feature_columns": feature_columns}, os.path.join(out_dir, "preprocess.joblib"))

    if not args.skip_diagnostics_plots:
        diagnostics_script = Path(__file__).with_name("plot_training_run_diagnostics.py")
        if diagnostics_script.exists():
            result = subprocess.run(
                [
                    sys.executable,
                    str(diagnostics_script),
                    "--run-dir",
                    out_dir,
                    "--target-col",
                    args.target_col,
                ],
                check=False,
            )
            if result.returncode != 0:
                print("[WARN] Training diagnostics plot script failed.")

    print("\n[Validation metrics]")
    for key, value in metrics.items():
        print(f"- {key}: {value:.6f}" if isinstance(value, float) else f"- {key}: {value}")
    print(f"\n[Saved] {os.path.abspath(out_dir)}")

    if wb_run is not None:
        wb_run.summary.update(metrics)
        wb_run.finish()


if __name__ == "__main__":
    main()
