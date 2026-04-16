#!/usr/bin/env python3
"""
03_train_multioutput.py

Entrena un modelo Multi-Output (MLP con Embeddings Categóricos) usando PyTorch.
Compatible con esquemas ternarios y binarios, con sufijos por ventana, schema
y quantiles/thresholds para trazabilidad de experimentos.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── CONFIGURACIÓN ────────────────────────────────────────────────────────────
DEFAULT_INPUT_PATH = os.path.join("data", "training", "model_input_multioutput.parquet")
DEFAULT_OUTPUT_DIR = os.path.join("Models", "multioutput_nn")
DEFAULT_GROUP_COL = "match_id"
MISSING_LABEL = -100


def format_window_tag(max_minute: float) -> str:
    rounded = int(round(float(max_minute)))
    return f"m{rounded:02d}"


def apply_window_suffix(path: str, max_minute: Optional[float]) -> str:
    if max_minute is None:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_{format_window_tag(max_minute)}{ext}"


def apply_sample_suffix(path: str, frac: Optional[float]) -> str:
    if frac is None or frac >= 1.0 or frac <= 0.0:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_sample{int(frac * 100)}{ext}"


def format_quantile_or_threshold_tag(
    lower_q: Optional[float],
    upper_q: Optional[float],
    lower_thr: Optional[float],
    upper_thr: Optional[float],
) -> str:
    if lower_thr is not None and upper_thr is not None:
        def _fmt_thr(x: float) -> str:
            s = f"{float(x):.4f}".rstrip("0").rstrip(".")
            return s.replace("-", "m").replace(".", "p")
        return f"thr{_fmt_thr(lower_thr)}_{_fmt_thr(upper_thr)}"

    if lower_q is None or upper_q is None:
        return "qNA_NA"

    l = int(round(float(lower_q) * 100))
    u = int(round(float(upper_q) * 100))
    return f"q{l:02d}_{u:02d}"


def apply_quantile_or_threshold_suffix(
    path: str,
    lower_q: Optional[float],
    upper_q: Optional[float],
    lower_thr: Optional[float],
    upper_thr: Optional[float],
) -> str:
    tag = format_quantile_or_threshold_tag(lower_q, upper_q, lower_thr, upper_thr)
    base, ext = os.path.splitext(path)
    return f"{base}_{tag}{ext}"


def apply_schema_suffix(path: str, target_schema: Optional[str]) -> str:
    if not target_schema:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}_{target_schema}{ext}"


TASK_DEFINITIONS_BY_SCHEMA: Dict[str, List[Tuple[str, List[str]]]] = {
    "ternary": [
        ("jungle_presence_label", ["farm_oriented", "ambiguous", "map_presence"]),
        ("support_roam_label",    ["lane_anchored", "ambiguous", "roamer"]),
        ("team_tendency_label",   ["botside_oriented", "ambiguous", "topside_oriented"]),
    ],
    "binary_clean": [
        ("jungle_presence_label", ["farm_oriented", "map_presence"]),
        ("support_roam_label",    ["lane_anchored", "roamer"]),
        ("team_tendency_label",   ["botside_oriented", "topside_oriented"]),
    ],
    "binary_full": [
        ("jungle_presence_label", ["farm_oriented", "map_presence"]),
        ("support_roam_label",    ["lane_anchored", "roamer"]),
        ("team_tendency_label",   ["botside_oriented", "topside_oriented"]),
    ],
}

ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")

FEATURE_GROUP_DEFS: Dict[str, Dict[str, Any]] = {
    "champions": {
        "columns": [f"{s}_{r}_champion_id" for s in SIDES for r in ROLE_KEYS],
        "embed_dim": 8,
        "description": "Champion picks de ambos equipos (10 posiciones).",
    },
    "summoner_spells": {
        "columns": [f"{s}_{r}_summoner{i}_id" for s in SIDES for r in ROLE_KEYS for i in (1, 2)],
        "embed_dim": 2,
        "description": "Hechizos de invocador (Flash, Teleport, Ignite...). Definen prioridad de línea y estilo temprano.",
    },
    "keystones": {
        "columns": [f"{s}_{r}_keystone_id" for s in SIDES for r in ROLE_KEYS],
        "embed_dim": 4,
        "description": "Runa keystone por jugador.",
    },
    "rune_styles": {
        "columns": [
            f"{s}_{r}_{style}_style_id"
            for s in SIDES for r in ROLE_KEYS for style in ("primary", "sub")
        ],
        "embed_dim": 2,
        "description": "Árboles de runas primario y secundario.",
    },
    "bans": {
        "columns": [f"{s}_ban_{i}_champion_id" for s in SIDES for i in range(1, 6)],
        "embed_dim": 4,
        "description": "Bans de ambos equipos.",
    },
    "context": {
        "columns": ["side"],
        "embed_dim": 2,
        "description": "Side (blue/red).",
    },
}

ALL_GROUP_NAMES = list(FEATURE_GROUP_DEFS.keys())
ABLATION_PRESETS = {
    "minimal": ["champions", "context"],
    "standard": ["champions", "summoner_spells", "context"],
    "full_runes": ["champions", "summoner_spells", "keystones", "context"],
    "all": ALL_GROUP_NAMES,
}


# ── UTILIDADES ───────────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_target_frac(args_frac: Optional[float]) -> Optional[float]:
    if args_frac is not None:
        return args_frac
    env_frac = os.getenv("TFG_SAMPLE_FRAC")
    if env_frac:
        try:
            return float(env_frac)
        except ValueError:
            pass
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-output classifier con embeddings y ablación.")
    p.add_argument("--input", default=DEFAULT_INPUT_PATH)
    p.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden1", type=int, default=128)
    p.add_argument("--hidden2", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--drop-ambiguous", action="store_true",
                   help="Solo válido con target-schema ternary.")
    p.add_argument(
        "--feature-groups", nargs="*", default=None,
        help=f"Grupos de features a usar. Opciones: {ALL_GROUP_NAMES}. Presets: {list(ABLATION_PRESETS.keys())}."
    )
    p.add_argument("--label-max-minute", type=float, default=None)
    p.add_argument(
        "--target-schema",
        choices=["ternary", "binary_clean", "binary_full"],
        default="ternary",
    )
    p.add_argument("--lower-quantile", type=float, default=0.20)
    p.add_argument("--upper-quantile", type=float, default=0.80)
    p.add_argument("--lower-threshold", type=float, default=None)
    p.add_argument("--upper-threshold", type=float, default=None)
    return p.parse_args()


# ── LABEL ENCODING ───────────────────────────────────────────────────────────

def build_vocab(series: pd.Series) -> Dict[Any, int]:
    unique = sorted(set(v for v in series.dropna().unique() if v is not None))
    return {v: i + 1 for i, v in enumerate(unique)}


def encode_column(series: pd.Series, vocab: Dict[Any, int]) -> np.ndarray:
    return np.array([vocab.get(v, 0) if pd.notna(v) else 0 for v in series], dtype=np.int64)


def encode_targets(
    df: pd.DataFrame, task_defs: List[Tuple[str, List[str]]],
) -> Tuple[np.ndarray, Dict[str, Dict[str, int]]]:
    mappings: Dict[str, Dict[str, int]] = {}
    cols: List[np.ndarray] = []
    for col, classes in task_defs:
        col_map = {cls: i for i, cls in enumerate(classes)}
        mappings[col] = col_map
        if col not in df.columns:
            cols.append(np.full(len(df), MISSING_LABEL, dtype=np.int64))
        else:
            def _map(val, m=col_map):
                return m.get(str(val), MISSING_LABEL) if pd.notna(val) else MISSING_LABEL
            cols.append(df[col].apply(_map).values.astype(np.int64))
    return np.column_stack(cols), mappings


# ── DATASET ──────────────────────────────────────────────────────────────────

class EmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ── MODELO ───────────────────────────────────────────────────────────────────

class MultiOutputEmbeddingMLP(nn.Module):
    def __init__(
        self,
        embedding_specs: List[Tuple[int, int]],
        hidden1: int,
        hidden2: int,
        num_classes_per_task: List[int],
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_cat + 1, emb_dim, padding_idx=0)
            for num_cat, emb_dim in embedding_specs
        ])
        total_emb_dim = sum(emb_dim for _, emb_dim in embedding_specs)

        self.shared = nn.Sequential(
            nn.Linear(total_emb_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(hidden2),
                nn.Dropout(dropout / 2),
                nn.Linear(hidden2, n_cls)
            ) for n_cls in num_classes_per_task
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        emb_list = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        concat = torch.cat(emb_list, dim=1)
        shared_out = self.shared(concat)
        return [head(shared_out) for head in self.heads]


# ── LOSS + TRAIN + EVAL ─────────────────────────────────────────────────────

def masked_multiclass_loss(logits_list: List[torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
    total_loss = torch.tensor(0.0, device=logits_list[0].device)
    n_valid = 0

    for task_idx, logits in enumerate(logits_list):
        tt = targets[:, task_idx]
        num_classes = logits.shape[1]

        if num_classes == 3:
            class_weights = torch.tensor([2.0, 1.0, 2.0], device=logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=MISSING_LABEL)
        elif num_classes == 2:
            loss_fn = nn.CrossEntropyLoss(ignore_index=MISSING_LABEL)
        else:
            raise ValueError(f"Número de clases no soportado en la loss: {num_classes}")

        task_loss = loss_fn(logits, tt)

        if (tt != MISSING_LABEL).any():
            total_loss = total_loss + task_loss
            n_valid += 1

    if n_valid == 0:
        return torch.tensor(0.0, device=logits_list[0].device, requires_grad=True)

    return total_loss / n_valid


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = total_n = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = masked_multiclass_loss(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_b.size(0)
        total_n += X_b.size(0)
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device,
    task_defs: List[Tuple[str, List[str]]],
) -> Tuple[float, Dict[str, Dict[str, Any]]]:
    model.eval()
    total_loss = total_n = 0
    all_t: List[List[int]] = [[] for _ in task_defs]
    all_p: List[List[int]] = [[] for _ in task_defs]

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        logits_list = model(X_b)
        loss = masked_multiclass_loss(logits_list, y_b)
        total_loss += loss.item() * X_b.size(0)
        total_n += X_b.size(0)
        for ti, logits in enumerate(logits_list):
            tt = y_b[:, ti].cpu().tolist()
            pp = logits.argmax(dim=1).cpu().tolist()
            for t, p in zip(tt, pp):
                if t != MISSING_LABEL:
                    all_t[ti].append(t)
                    all_p[ti].append(p)

    avg_loss = total_loss / max(total_n, 1)
    metrics: Dict[str, Dict[str, Any]] = {}
    for ti, (col, classes) in enumerate(task_defs):
        yt = np.array(all_t[ti])
        yp = np.array(all_p[ti])
        if len(yt) == 0:
            metrics[col] = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "f1_macro": 0.0,
                "valid_samples": 0,
                "class_names": classes,
            }
            continue
        pred_dist = {c: int((yp == ci).sum()) for ci, c in enumerate(classes)}
        metrics[col] = {
            "accuracy": round(float(accuracy_score(yt, yp)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(yt, yp)), 4),
            "f1_macro": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
            "valid_samples": int(len(yt)),
            "class_names": classes,
            "prediction_distribution": pred_dist,
        }
    return avg_loss, metrics


# ── VALIDACIONES ─────────────────────────────────────────────────────────────

def validate_targets(df: pd.DataFrame, task_defs: List[Tuple[str, List[str]]]) -> None:
    print("\n[Validación] Comprobando columnas target...")
    for col, expected_classes in task_defs:
        if col not in df.columns:
            print(f"  ✗ {col}: NO ENCONTRADA.")
            sys.exit(1)
        non_null = df[col].dropna()
        if len(non_null) == 0:
            print(f"  ✗ {col}: Todo NaN.")
            sys.exit(1)
        unexpected = set(non_null.unique()) - set(expected_classes)
        if unexpected:
            print(f"  ✗ {col}: Clases inesperadas: {unexpected}")
            sys.exit(1)
        for cls in expected_classes:
            n = int((non_null == cls).sum())
            pct = n / len(non_null) * 100 if len(non_null) else 0.0
            print(f"      {'✓' if n > 0 else '⚠'} {cls}: {n} ({pct:.1f}%)")
        print(f"      NaN: {int(df[col].isna().sum())} ({df[col].isna().mean()*100:.1f}%)")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.feature_groups is None:
        active_groups = list(ALL_GROUP_NAMES)
    elif len(args.feature_groups) == 1 and args.feature_groups[0] in ABLATION_PRESETS:
        active_groups = ABLATION_PRESETS[args.feature_groups[0]]
    else:
        active_groups = [g for g in args.feature_groups if g in FEATURE_GROUP_DEFS]
    if not active_groups:
        raise SystemExit(f"No hay feature groups activos. Opciones: {ALL_GROUP_NAMES}")

    quantile_tag = format_quantile_or_threshold_tag(
        args.lower_quantile, args.upper_quantile, args.lower_threshold, args.upper_threshold
    )

    print(f"\n{'='*60}")
    print(" Multi-Output Embedding Classifier (PyTorch)")
    print(f" Dispositivo: {device.type.upper()}")
    print(f" Feature groups: {active_groups}")
    if args.label_max_minute is not None:
        print(f" Label window: {format_window_tag(args.label_max_minute)}")
    print(f" Target schema: {args.target_schema}")
    print(f" Quantile/threshold tag: {quantile_tag}")
    print(f"{'='*60}")

    target_frac = get_target_frac(args.sample_frac)
    in_path = args.input
    out_dir = args.outdir

    if target_frac is not None and 0.0 < target_frac < 1.0:
        in_path = apply_sample_suffix(in_path, target_frac)
        out_dir = apply_sample_suffix(out_dir, target_frac)
    if args.label_max_minute is not None:
        in_path = apply_window_suffix(in_path, args.label_max_minute)
        out_dir = apply_window_suffix(out_dir, args.label_max_minute)
    in_path = apply_quantile_or_threshold_suffix(
        in_path,
        args.lower_quantile,
        args.upper_quantile,
        args.lower_threshold,
        args.upper_threshold,
    )
    out_dir = apply_quantile_or_threshold_suffix(
        out_dir,
        args.lower_quantile,
        args.upper_quantile,
        args.lower_threshold,
        args.upper_threshold,
    )
    in_path = apply_schema_suffix(in_path, args.target_schema)
    out_dir = apply_schema_suffix(out_dir, args.target_schema)

    print(f"\n[Datos] Cargando: {os.path.abspath(in_path)}")
    if not os.path.exists(in_path):
        print(f"[Error] Archivo no encontrado: {in_path}")
        sys.exit(1)
    df = pd.read_parquet(in_path)
    print(f"        Filas: {len(df)}")

    ensure_dir(out_dir)

    task_defs = list(TASK_DEFINITIONS_BY_SCHEMA[args.target_schema])
    if args.target_schema != "ternary" and args.drop_ambiguous:
        raise SystemExit("--drop-ambiguous solo debe usarse con --target-schema ternary.")
    if args.drop_ambiguous:
        print("\n[Filtro] --drop-ambiguous: eliminando filas ambiguas...")
        for col, _classes in TASK_DEFINITIONS_BY_SCHEMA["ternary"]:
            if col in df.columns:
                before = len(df)
                df = df[df[col].isna() | (df[col] != "ambiguous")].copy()
                dropped = before - len(df)
                if dropped > 0:
                    print(f"         {col}: {dropped} eliminadas")
        task_defs = list(TASK_DEFINITIONS_BY_SCHEMA["binary_clean"])
        print(f"         Filas restantes: {len(df)}")

    validate_targets(df, task_defs)

    feature_columns: List[str] = []
    for gname in active_groups:
        gdef = FEATURE_GROUP_DEFS[gname]
        for col in gdef["columns"]:
            if col in df.columns:
                feature_columns.append(col)
            else:
                print(f"[WARN] Feature {col} ({gname}) no encontrada en parquet — se omite.")

    if not feature_columns:
        raise SystemExit("No hay features disponibles en el parquet.")
    print(f"\n[Features] {len(feature_columns)} columnas de {len(active_groups)} grupos")

    print(f"\n[Split] GroupShuffleSplit por {DEFAULT_GROUP_COL}...")
    groups = df[DEFAULT_GROUP_COL]
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(df, groups=groups))
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()
    print(f"        Train: {len(df_train)} | Val: {len(df_val)}")

    print("\n[Preprocesamiento] Construyendo vocabularios desde train...")
    vocabs: Dict[str, Dict[Any, int]] = {}
    for col in feature_columns:
        vocabs[col] = build_vocab(df_train[col])

    X_train_cols = [encode_column(df_train[col], vocabs[col]) for col in feature_columns]
    X_val_cols = [encode_column(df_val[col], vocabs[col]) for col in feature_columns]
    X_train = np.column_stack(X_train_cols)
    X_val = np.column_stack(X_val_cols)

    embedding_specs: List[Tuple[int, int]] = []
    embed_dim_by_group = {g: FEATURE_GROUP_DEFS[g]["embed_dim"] for g in active_groups}
    col_to_group = {}
    for gname in active_groups:
        for col in FEATURE_GROUP_DEFS[gname]["columns"]:
            if col in feature_columns:
                col_to_group[col] = gname

    for col in feature_columns:
        num_cat = len(vocabs[col])
        group = col_to_group.get(col, "context")
        emb_dim = embed_dim_by_group.get(group, 4)
        embedding_specs.append((num_cat, emb_dim))

    total_emb_dim = sum(ed for _, ed in embedding_specs)
    print(f"        {len(feature_columns)} embeddings → {total_emb_dim} dimensiones totales")
    print(f"        (vs ~{sum(len(v) for v in vocabs.values())}+ con OneHot)")

    y_train, target_mappings = encode_targets(df_train, task_defs)
    y_val, _ = encode_targets(df_val, task_defs)

    train_loader = DataLoader(EmbeddingDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(EmbeddingDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)

    num_classes = [len(cl) for _, cl in task_defs]
    model = MultiOutputEmbeddingMLP(
        embedding_specs=embedding_specs,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        num_classes_per_task=num_classes,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n        Parámetros del modelo: {n_params:,}")

    print(f"\n{'='*60}")
    print(f" ENTRENAMIENTO ({args.epochs} épocas max, patience={args.patience})")
    print(f"{'='*60}\n")

    best_val_loss = float("inf")
    best_metrics: Dict[str, Dict[str, Any]] = {}
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: List[Dict[str, Any]] = []
    epoch_metric_rows: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_loader, device, task_defs)

        task_str = "  ".join(
            f"{col.split('_')[0][:3]}:acc={m['accuracy']:.3f}"
            for col, m in val_metrics.items()
        )
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
            marker = " ★"
        else:
            epochs_without_improvement += 1

        epoch_row: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "is_best": int(val_loss <= best_val_loss),
        }
        for task_name, met in val_metrics.items():
            prefix = task_name.replace("_label", "")
            epoch_row[f"{prefix}_accuracy"] = float(met.get("accuracy", 0.0))
            epoch_row[f"{prefix}_balanced_accuracy"] = float(met.get("balanced_accuracy", 0.0))
            epoch_row[f"{prefix}_f1_macro"] = float(met.get("f1_macro", 0.0))
            epoch_row[f"{prefix}_valid_samples"] = int(met.get("valid_samples", 0))
            epoch_metric_rows.append({
                "epoch": epoch,
                "task": task_name,
                "accuracy": float(met.get("accuracy", 0.0)),
                "balanced_accuracy": float(met.get("balanced_accuracy", 0.0)),
                "f1_macro": float(met.get("f1_macro", 0.0)),
                "valid_samples": int(met.get("valid_samples", 0)),
            })
        history_rows.append(epoch_row)

        print(
            f"Época [{epoch:02d}/{args.epochs:02d}] "
            f"TrLoss: {train_loss:.4f} | VlLoss: {val_loss:.4f} | {task_str}{marker}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"\n⏹ Early stopping: {args.patience} épocas sin mejora.")
            break

    print(f"\n{'='*60}")
    print(" ENTRENAMIENTO FINALIZADO")
    print(f" Mejor época: {best_epoch} (Val Loss: {best_val_loss:.4f})")
    print(f"{'='*60}")

    print("\n Métricas de validación (mejor checkpoint):\n")
    for task_name, met in best_metrics.items():
        print(f"  ● {task_name}:")
        print(f"      Accuracy:          {met['accuracy']:.4f}")
        print(f"      Balanced Accuracy: {met['balanced_accuracy']:.4f}")
        print(f"      F1 Macro:          {met['f1_macro']:.4f}")
        print(f"      Muestras válidas:  {met['valid_samples']}")
        if "prediction_distribution" in met:
            print(f"      Predicciones: {met['prediction_distribution']}")

    print(f"\n[Exportación] Guardando en: {os.path.abspath(out_dir)}")

    vocabs_serializable = {col: {str(k): v for k, v in vocab.items()} for col, vocab in vocabs.items()}
    joblib.dump(vocabs_serializable, os.path.join(out_dir, "vocabs.joblib"))

    class_distributions = {}
    for col, classes in task_defs:
        td, vd = {}, {}
        if col in df_train.columns:
            vc = df_train[col].value_counts(dropna=False)
            for cls in classes:
                td[cls] = int(vc.get(cls, 0))
            td["NaN"] = int(df_train[col].isna().sum())
        if col in df_val.columns:
            vc = df_val[col].value_counts(dropna=False)
            for cls in classes:
                vd[cls] = int(vc.get(cls, 0))
            vd["NaN"] = int(df_val[col].isna().sum())
        class_distributions[col] = {"train": td, "val": vd}

    config = {
        "feature_groups_active": active_groups,
        "feature_groups_available": ALL_GROUP_NAMES,
        "feature_columns": feature_columns,
        "embedding_specs": [
            {"column": col, "num_categories": ns, "embed_dim": ed}
            for col, (ns, ed) in zip(feature_columns, embedding_specs)
        ],
        "total_embedding_dim": total_emb_dim,
        "tasks": [
            {"column": col, "classes": classes, "num_classes": len(classes)}
            for col, classes in task_defs
        ],
        "target_mappings": target_mappings,
        "class_distributions": class_distributions,
        "hyperparameters": {
            "hidden1": args.hidden1,
            "hidden2": args.hidden2,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs_max": args.epochs,
            "patience": args.patience,
            "epochs_trained": best_epoch,
            "seed": args.seed,
            "val_size": args.val_size,
            "drop_ambiguous": args.drop_ambiguous,
            "num_parameters": n_params,
        },
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "best_metrics": best_metrics,
        "sample_frac": target_frac,
        "target_schema": args.target_schema,
        "label_max_minute": args.label_max_minute,
        "label_window_tag": format_window_tag(args.label_max_minute) if args.label_max_minute is not None else None,
        "resolved_input_path": os.path.abspath(in_path),
        "resolved_output_dir": os.path.abspath(out_dir),
        "quantile_or_threshold_tag": quantile_tag,
        "lower_quantile": args.lower_quantile,
        "upper_quantile": args.upper_quantile,
        "lower_threshold": args.lower_threshold,
        "upper_threshold": args.upper_threshold,
        "device_used": device.type,
        "ablation_presets": ABLATION_PRESETS,
        "feature_group_descriptions": {
            g: FEATURE_GROUP_DEFS[g]["description"] for g in ALL_GROUP_NAMES
        },
    }
    with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("  ✓ best_model.pt          (State Dict PyTorch)")
    history_df = pd.DataFrame(history_rows)
    epoch_metrics_df = pd.DataFrame(epoch_metric_rows)
    history_csv_path = os.path.join(out_dir, "history.csv")
    epoch_metrics_csv_path = os.path.join(out_dir, "epoch_metrics.csv")
    history_df.to_csv(history_csv_path, index=False)
    epoch_metrics_df.to_csv(epoch_metrics_csv_path, index=False)

    history_plot_path = os.path.join(out_dir, "history_loss.png")
    if not history_df.empty:
        plt.figure(figsize=(8, 5))
        plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
        plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training history")
        plt.legend()
        plt.tight_layout()
        plt.savefig(history_plot_path, dpi=160)
        plt.close()

    print("  ✓ history.csv           (Train/Val loss por época)")
    print("  ✓ epoch_metrics.csv     (Métricas de validación por época y tarea)")
    print("  ✓ history_loss.png      (Curva Train/Val loss)")
    print("  ✓ vocabs.joblib         (Vocabularios cat → idx)")
    print("  ✓ model_config.json     (Config, métricas, ablación)")
    print("\nHecho. Yippie! Kaggle.")


if __name__ == "__main__":
    main()
