#!/usr/bin/env python3
"""
CLI prototype for support-roaming inference from draft data.

The client uses the final report protocol: 10 champion IDs + side. It supports
single-draft, JSON, and batch modes, and can select any raw-scale model from
the final main comparison table.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_ROOT = REPO_ROOT / "final" / "models"
DEFAULT_TRAIN_PATH = REPO_ROOT / "final" / "data" / "training" / "train.parquet"
DEFAULT_METRICS_TABLE = REPO_ROOT / "final" / "analysis" / "model_comparison" / "final_main_table_raw.csv"
DEFAULT_CHAMPION_CLASSES = REPO_ROOT / "final" / "data" / "champion_classes.json"
DEFAULT_CHAMPION_ARCHETYPES = REPO_ROOT / "final" / "data" / "champion_archetypes.json"

MODEL_KEYS = ("histgbt", "mlp_onehot", "mlp_embed", "mlp_per_role", "champion_mean", "global_mean")
REPORT_MODEL_NAMES = {
    "histgbt": "HistGBT",
    "mlp_onehot": "MLP OneHot",
    "mlp_embed": "MLP Embed Shared",
    "mlp_per_role": "MLP Per-Role + Interactions",
    "champion_mean": "Champion Mean",
    "global_mean": "Global Mean",
}
MODEL_DIRS = {
    "histgbt": "gbt",
    "mlp_onehot": "mlp_onehot",
    "mlp_embed": "mlp_embed",
    "mlp_per_role": "mlp_per_role",
}
MODEL_FILES = {
    "histgbt": "gbt_model_raw.joblib",
    "mlp_onehot": "mlp_onehot_raw.pt",
    "mlp_embed": "mlp_embed_raw.pt",
    "mlp_per_role": "mlp_per_role_raw.pt",
}
CANONICAL_FEATURE_COLUMNS = [
    "ally_top_champion_id",
    "ally_jungle_champion_id",
    "ally_middle_champion_id",
    "ally_bottom_champion_id",
    "ally_utility_champion_id",
    "enemy_top_champion_id",
    "enemy_jungle_champion_id",
    "enemy_middle_champion_id",
    "enemy_bottom_champion_id",
    "enemy_utility_champion_id",
    "side",
]
CHAMPION_COLS = [c for c in CANONICAL_FEATURE_COLUMNS if c.endswith("_champion_id")]
ROLES: Tuple[Tuple[str, str], ...] = (
    ("top", "Top"),
    ("jungle", "Jungle"),
    ("middle", "Mid"),
    ("bottom", "ADC"),
    ("utility", "Support"),
)
SIDES: Tuple[str, str] = ("ally", "enemy")
PROFILE_BANDS: Tuple[Tuple[float, str], ...] = (
    (0.25, "Perfil de laning - el support tiende a quedarse en botlane"),
    (0.40, "Perfil mixto - roaming ocasional pero anclado a bot"),
    (0.55, "Perfil de roaming moderado - rotaciones frecuentes esperadas"),
    (math.inf, "Perfil de roaming intenso - el support abandona bot activamente"),
)


@dataclass(frozen=True)
class ChampionInfo:
    champion_id: int
    name: str
    primary_class: str = "Unknown"
    support_archetype: str = ""


@dataclass
class PredictionResult:
    score: float
    profile: str
    confidence_label: str
    confidence_score: float
    percentile: Optional[float]
    unknown_features: List[str]
    top_features: List[Dict[str, Any]]
    explanation_method: str
    assumptions: List[str]


@dataclass
class MatchupResult:
    ally: PredictionResult
    enemy: PredictionResult
    delta: float
    label: str
    reading: str


class DraftPredictor(Protocol):
    key: str
    model_label: str
    feature_columns: List[str]
    no_shap: bool

    def predict_score(self, row: pd.DataFrame) -> float: ...
    def unknown_features(self, row: pd.DataFrame) -> List[str]: ...
    def target_percentile(self, score: float) -> Optional[float]: ...
    def confidence(self, score: float, unknown_count: int) -> Tuple[str, float]: ...
    def explain_local(self, row: pd.DataFrame, score: float, top_n: int) -> Tuple[List[Dict[str, Any]], str]: ...
    def display_metrics(self) -> Dict[str, str]: ...


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prompt_value(label: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or (default or "")


def score_profile(score: float) -> str:
    for upper, label in PROFILE_BANDS:
        if score < upper:
            return label
    return PROFILE_BANDS[-1][1]


def matchup_profile(delta: float) -> Tuple[str, str]:
    if delta >= 0.10:
        return (
            "ventaja clara de roaming aliado",
            "Tu support proyecta bastante mas movilidad que el support enemigo. Si la partida acompana, puede iniciar o contestar rotaciones antes.",
        )
    if delta >= 0.05:
        return (
            "ligera ventaja de roaming aliado",
            "Tu support apunta a moverse algo mas que el enemigo, pero el margen no es enorme; el matchup deberia depender mucho del estado de linea.",
        )
    if delta <= -0.10:
        return (
            "ventaja clara de roaming enemigo",
            "El support enemigo proyecta bastante mas roaming. Tu botlane probablemente necesite vigilar river, mid y resets para no quedarse por detras en tempo.",
        )
    if delta <= -0.05:
        return (
            "ligera ventaja de roaming enemigo",
            "El support enemigo apunta a moverse algo mas. Tu support puede matchearlo, pero no parte con ventaja clara de tendencia.",
        )
    return (
        "matchup de roaming equilibrado",
        "Ambos supports tienen una tendencia de roaming parecida. No hay senal fuerte de ventaja por estilo desde el draft.",
    )


def confidence_label(value: float) -> str:
    if value >= 0.78:
        return "alta"
    if value >= 0.55:
        return "media"
    return "baja"


def load_champion_info(classes_path: Path, archetypes_path: Path) -> Dict[int, ChampionInfo]:
    classes_raw = read_json(classes_path) if classes_path.exists() else {}
    archetypes_raw = read_json(archetypes_path) if archetypes_path.exists() else {}
    archetype_champs = archetypes_raw.get("champions", {}) if isinstance(archetypes_raw, dict) else {}

    ids: set[int] = set()
    for source in (classes_raw, archetype_champs):
        for key in source.keys():
            try:
                ids.add(int(key))
            except (TypeError, ValueError):
                pass

    info: Dict[int, ChampionInfo] = {}
    for cid in sorted(ids):
        class_entry = classes_raw.get(str(cid), {}) if isinstance(classes_raw, dict) else {}
        archetype_entry = archetype_champs.get(str(cid), {}) if isinstance(archetype_champs, dict) else {}
        name = class_entry.get("name") or archetype_entry.get("name") or f"champ_{cid}"
        info[cid] = ChampionInfo(
            champion_id=cid,
            name=str(name),
            primary_class=str(class_entry.get("primary_class", "Unknown")),
            support_archetype=str(archetype_entry.get("support", "")),
        )
    return info


def build_champion_lookup(info: Mapping[int, ChampionInfo]) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for cid, champion in info.items():
        lookup.setdefault(normalize_key(champion.name), cid)

    aliases = {
        "aurelionsol": "Aurelion Sol",
        "belveth": "Bel'Veth",
        "chogath": "Cho'Gath",
        "drmundo": "Dr. Mundo",
        "jarvaniv": "Jarvan IV",
        "kaisa": "Kai'Sa",
        "khazix": "Kha'Zix",
        "kogmaw": "Kog'Maw",
        "ksante": "K'Sante",
        "leesin": "Lee Sin",
        "masteryi": "Master Yi",
        "missfortune": "Miss Fortune",
        "nunu": "Nunu & Willump",
        "nunuandwillump": "Nunu & Willump",
        "reksai": "Rek'Sai",
        "renata": "Renata Glasc",
        "tahmkench": "Tahm Kench",
        "twistedfate": "Twisted Fate",
        "velkoz": "Vel'Koz",
        "xinzhao": "Xin Zhao",
        "wukong": "Wukong",
    }
    canonical_by_key = {normalize_key(champion.name): cid for cid, champion in info.items()}
    for alias, canonical in aliases.items():
        cid = canonical_by_key.get(normalize_key(canonical))
        if cid is not None:
            lookup.setdefault(alias, cid)
    return lookup


def resolve_champion(raw: Any, lookup: Mapping[str, int], info: Mapping[int, ChampionInfo]) -> int:
    if isinstance(raw, (int, np.integer)):
        cid = int(raw)
        if cid not in info:
            raise ValueError(f"Unknown champion id: {cid}")
        return cid
    if isinstance(raw, (float, np.floating)) and not math.isnan(float(raw)):
        cid = int(raw)
        if cid not in info:
            raise ValueError(f"Unknown champion id: {cid}")
        return cid
    value = str(raw).strip()
    if not value:
        raise ValueError("Champion name or id is required.")
    if value.isdigit() or re.fullmatch(r"\d+\.0+", value):
        cid = int(float(value))
        if cid not in info:
            raise ValueError(f"Unknown champion id: {cid}")
        return cid
    key = normalize_key(value)
    if key in lookup:
        return lookup[key]
    suggestions = difflib.get_close_matches(key, sorted(lookup), n=5, cutoff=0.62)
    pretty = [info[lookup[s]].name for s in suggestions if s in lookup]
    hint = f" Suggestions: {', '.join(dict.fromkeys(pretty))}." if pretty else ""
    raise ValueError(f"Unknown champion: {raw}.{hint}")


def first_present(values: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in values and pd.notna(values[key]) and str(values[key]).strip() != "":
            return values[key]
    return None


def categorical_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.loc[:, list(columns)].copy()
    for col in columns:
        out[col] = out[col].where(out[col].notna(), "__MISSING__").astype(str)
    return out


def find_run_dirs(model_dir: Path, required_file: str) -> List[Path]:
    seed_dirs = [
        model_dir / f"seed{seed}"
        for seed in (42, 123, 456)
        if (model_dir / f"seed{seed}" / required_file).exists()
    ]
    if seed_dirs:
        return seed_dirs
    if (model_dir / required_file).exists():
        return [model_dir]
    raise FileNotFoundError(f"No model artifact '{required_file}' found under {model_dir}")


def load_report_metrics(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    return {str(row["model"]): row.to_dict() for _, row in df.iterrows()}


def parse_metric_mean(value: str) -> Optional[float]:
    value = str(value or "").strip()
    if not value:
        return None
    try:
        return float(value.split("+/-")[0].strip())
    except ValueError:
        return None


class BasePredictor:
    key: str
    model_label: str
    feature_columns: List[str]

    def __init__(
        self,
        key: str,
        model_label: str,
        train_path: Path,
        champion_info: Mapping[int, ChampionInfo],
        report_metrics: Mapping[str, Mapping[str, str]],
        background_size: int,
        no_shap: bool,
    ) -> None:
        self.key = key
        self.model_label = model_label
        self.train_path = train_path
        self.champion_info = champion_info
        self.report_metrics = dict(report_metrics.get(model_label, {}))
        self.background_size = background_size
        self.no_shap = no_shap
        self.feature_columns = list(CANONICAL_FEATURE_COLUMNS)
        self._target_values: Optional[np.ndarray] = None

    def display_metrics(self) -> Dict[str, str]:
        return {
            "r2": str(self.report_metrics.get("r2", "")),
            "spearman_corr": str(self.report_metrics.get("spearman_corr", "")),
            "mae": str(self.report_metrics.get("mae", "")),
            "n_eval": str(self.report_metrics.get("n_eval", "")),
            "n_seeds": str(self.report_metrics.get("n_seeds", "")),
        }

    def target_percentile(self, score: float) -> Optional[float]:
        if self._target_values is None:
            if not self.train_path.exists():
                return None
            try:
                vals = pd.read_parquet(self.train_path, columns=["support_roam_score"])
                self._target_values = vals["support_roam_score"].dropna().to_numpy(dtype=float)
            except Exception:
                return None
        if self._target_values.size == 0:
            return None
        return float((self._target_values <= score).mean() * 100.0)

    def confidence(self, score: float, unknown_count: int) -> Tuple[str, float]:
        coverage = 1.0 - (unknown_count / max(len(self.feature_columns), 1))
        boundaries = np.asarray([0.25, 0.40, 0.55], dtype=float)
        distance = float(np.min(np.abs(boundaries - score)))
        stability = min(distance / 0.08, 1.0)
        spearman = parse_metric_mean(self.report_metrics.get("spearman_corr", "")) or 0.38
        model_quality = min(max(spearman / 0.45, 0.0), 1.0)
        value = 0.45 * coverage + 0.35 * stability + 0.20 * model_quality
        return confidence_label(value), float(value)

    def unknown_features(self, row: pd.DataFrame) -> List[str]:
        return []

    def explain_local(self, row: pd.DataFrame, score: float, top_n: int) -> Tuple[List[Dict[str, Any]], str]:
        return [], "disabled"

    def display_value(self, col: str, value: Any) -> str:
        if col.endswith("_champion_id"):
            try:
                cid = int(float(value))
                return self.champion_info.get(cid, ChampionInfo(cid, str(cid))).name
            except (TypeError, ValueError):
                return str(value)
        return str(value)


class GbtEnsemblePredictor(BasePredictor):
    def __init__(self, model_dir: Path, *args: Any, **kwargs: Any) -> None:
        super().__init__("histgbt", REPORT_MODEL_NAMES["histgbt"], *args, **kwargs)
        self.run_dirs = find_run_dirs(model_dir, MODEL_FILES["histgbt"])
        self.models = []
        self.encoders = []
        self.category_sets: List[set[str]] = []
        self._background: Optional[np.ndarray] = None
        self._background_mode: Optional[np.ndarray] = None
        for run_dir in self.run_dirs:
            self.models.append(joblib.load(run_dir / "gbt_model_raw.joblib"))
            preprocess = joblib.load(run_dir / "preprocess.joblib")
            feature_columns = list(preprocess["feature_columns"])
            if feature_columns != CANONICAL_FEATURE_COLUMNS:
                raise ValueError(f"{run_dir} does not use final feature protocol.")
            self.encoders.append(preprocess["encoder"])
        self.feature_columns = list(CANONICAL_FEATURE_COLUMNS)
        self.category_sets = [set(map(str, cats)) for cats in self.encoders[0].categories_]

    def encode(self, row: pd.DataFrame, encoder: Any) -> np.ndarray:
        return encoder.transform(categorical_frame(row, self.feature_columns))

    def predict_score(self, row: pd.DataFrame) -> float:
        preds = [
            float(model.predict(self.encode(row, encoder))[0])
            for model, encoder in zip(self.models, self.encoders)
        ]
        return float(np.clip(np.mean(preds), 0.0, 1.0))

    def unknown_features(self, row: pd.DataFrame) -> List[str]:
        unknown: List[str] = []
        for idx, col in enumerate(self.feature_columns):
            value = str(row.iloc[0].get(col, "__MISSING__"))
            if value not in self.category_sets[idx]:
                unknown.append(col)
        return unknown

    def load_background(self) -> Optional[np.ndarray]:
        if self._background is not None:
            return self._background
        if not self.train_path.exists() or self.background_size <= 0:
            return None
        try:
            df = pd.read_parquet(self.train_path, columns=self.feature_columns)
            if len(df) > self.background_size:
                df = df.sample(n=self.background_size, random_state=42)
            self._background = self.encoders[0].transform(categorical_frame(df, self.feature_columns))
            self._background_mode = np.asarray(
                [pd.Series(self._background[:, i]).mode(dropna=False).iloc[0] for i in range(self._background.shape[1])],
                dtype=np.float32,
            )
        except Exception:
            self._background = None
            self._background_mode = None
        return self._background

    def explain_local(self, row: pd.DataFrame, score: float, top_n: int) -> Tuple[List[Dict[str, Any]], str]:
        if self.no_shap:
            return [], "disabled"
        encoded = self.encode(row, self.encoders[0])
        background = self.load_background()
        if background is None or len(background) == 0:
            return self._fallback_deltas(row, encoded, score, top_n), "fallback_delta"
        try:
            import shap

            masker = shap.maskers.Independent(background)
            explainer = shap.PermutationExplainer(self.models[0].predict, masker, seed=42)
            values = explainer(encoded, max_evals=2 * encoded.shape[1] + 1)
            contribs = np.asarray(values.values[0], dtype=float)
            return self._format_contributions(row, contribs, top_n), "shap_permutation"
        except Exception:
            return self._fallback_deltas(row, encoded, score, top_n), "fallback_delta"

    def _fallback_deltas(self, row: pd.DataFrame, encoded: np.ndarray, score: float, top_n: int) -> List[Dict[str, Any]]:
        background = self.load_background()
        if background is None:
            return []
        if self._background_mode is None:
            self._background_mode = np.asarray(
                [pd.Series(background[:, i]).mode(dropna=False).iloc[0] for i in range(background.shape[1])],
                dtype=np.float32,
            )
        deltas = []
        for i in range(encoded.shape[1]):
            replaced = encoded.copy()
            replaced[0, i] = self._background_mode[i]
            deltas.append(score - float(self.models[0].predict(replaced)[0]))
        return self._format_contributions(row, np.asarray(deltas, dtype=float), top_n)

    def _format_contributions(self, row: pd.DataFrame, contribs: np.ndarray, top_n: int) -> List[Dict[str, Any]]:
        order = np.argsort(np.abs(contribs))[::-1][:top_n]
        out: List[Dict[str, Any]] = []
        for idx in order:
            col = self.feature_columns[int(idx)]
            out.append(
                {
                    "feature": col,
                    "value": self.display_value(col, row.iloc[0].get(col)),
                    "contribution": float(contribs[int(idx)]),
                    "direction": "sube" if contribs[int(idx)] >= 0 else "baja",
                }
            )
        return out


def torch_load_state(path: Path, device: Any) -> Any:
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class MLPEnsemblePredictor(BasePredictor):
    def __init__(self, key: str, model_dir: Path, *args: Any, **kwargs: Any) -> None:
        super().__init__(key, REPORT_MODEL_NAMES[key], *args, **kwargs)
        self.run_dirs = find_run_dirs(model_dir, MODEL_FILES[key])
        self.models = []
        self.vocabs = []
        self.device = None
        self._torch = None
        for run_dir in self.run_dirs:
            model, vocab = self._load_one_model(key, run_dir)
            self.models.append(model)
            self.vocabs.append(vocab)

    def _load_one_model(self, key: str, run_dir: Path) -> Tuple[Any, Dict[int, int]]:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        self._torch = torch
        self.device = torch.device("cpu")
        config = read_json(run_dir / "model_config.json")
        vocab_raw = read_json(run_dir / "vocab.json")
        vocab = {int(k): int(v) for k, v in vocab_raw.items()}
        hidden_dims = [int(v) for v in config["hidden_dims"]]
        vocab_size = int(config["vocab_size"])
        n_slots = int(config["n_champion_slots"])
        embed_dim = int(config.get("embed_dim", 16))
        dropout = float(config.get("dropout", 0.0))

        class MLPOneHotEval(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.vocab_size = vocab_size
                layers: List[nn.Module] = []
                prev = n_slots * vocab_size + 1
                for dim in hidden_dims:
                    layers.extend([nn.Linear(prev, dim), nn.ReLU(), nn.BatchNorm1d(dim), nn.Dropout(dropout)])
                    prev = dim
                layers.append(nn.Linear(prev, 1))
                self.net = nn.Sequential(*layers)

            def forward(self, champion_ids: Any, side: Any) -> Any:
                onehot = F.one_hot(champion_ids, num_classes=self.vocab_size).to(torch.float32)
                x = torch.cat([onehot.reshape(onehot.shape[0], -1), side.reshape(-1, 1)], dim=1)
                return self.net(x).squeeze(-1)

        class MLPEmbedEval(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                layers: List[nn.Module] = []
                prev = n_slots * embed_dim + 1
                for dim in hidden_dims:
                    layers.extend([nn.Linear(prev, dim), nn.ReLU(), nn.BatchNorm1d(dim), nn.Dropout(dropout)])
                    prev = dim
                layers.append(nn.Linear(prev, 1))
                self.head = nn.Sequential(*layers)

            def forward(self, champion_ids: Any, side: Any) -> Any:
                emb = self.embed(champion_ids).reshape(champion_ids.shape[0], -1)
                x = torch.cat([emb, side.reshape(-1, 1)], dim=1)
                return self.head(x).squeeze(-1)

        class MLPPerRoleEval(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.slot_embeddings = nn.ModuleList([nn.Embedding(vocab_size, embed_dim, padding_idx=0) for _ in range(n_slots)])
                layers: List[nn.Module] = []
                prev = n_slots * embed_dim + 1 + 2
                for dim in hidden_dims:
                    layers.extend([nn.Linear(prev, dim), nn.ReLU(), nn.BatchNorm1d(dim), nn.Dropout(dropout)])
                    prev = dim
                layers.append(nn.Linear(prev, 1))
                self.head = nn.Sequential(*layers)

            def forward(self, champion_ids: Any, side: Any) -> Any:
                embs = [emb(champion_ids[:, i]) for i, emb in enumerate(self.slot_embeddings)]
                flat = torch.cat(embs, dim=1)
                support_vs_support = (embs[4] * embs[9]).sum(dim=1, keepdim=True)
                support_adc = (embs[4] * embs[3]).sum(dim=1, keepdim=True)
                x = torch.cat([flat, side.reshape(-1, 1), support_vs_support, support_adc], dim=1)
                return self.head(x).squeeze(-1)

        if key == "mlp_onehot":
            model = MLPOneHotEval()
        elif key == "mlp_embed":
            model = MLPEmbedEval()
        elif key == "mlp_per_role":
            model = MLPPerRoleEval()
        else:
            raise ValueError(f"Unsupported MLP key: {key}")
        model.load_state_dict(torch_load_state(run_dir / MODEL_FILES[key], self.device))
        model.eval()
        return model, vocab

    @staticmethod
    def encode_champion_ids(row: pd.DataFrame, vocab: Mapping[int, int]) -> np.ndarray:
        ids = np.zeros((len(row), len(CHAMPION_COLS)), dtype=np.int64)
        for i, col in enumerate(CHAMPION_COLS):
            ids[:, i] = (
                pd.to_numeric(row[col], errors="coerce")
                .fillna(-1)
                .astype(int)
                .map(lambda x: vocab.get(x, 0))
                .to_numpy(dtype=np.int64)
            )
        return ids

    @staticmethod
    def encode_side(row: pd.DataFrame) -> np.ndarray:
        return (row["side"].astype(str).str.lower() == "red").astype(np.float32).to_numpy()

    def predict_score(self, row: pd.DataFrame) -> float:
        torch = self._torch
        assert torch is not None
        preds: List[float] = []
        with torch.no_grad():
            for model, vocab in zip(self.models, self.vocabs):
                champ = torch.from_numpy(self.encode_champion_ids(row, vocab)).to(self.device)
                side = torch.from_numpy(self.encode_side(row)).to(self.device)
                pred = model(champ, side).cpu().numpy()
                preds.append(float(pred[0]))
        return float(np.clip(np.mean(preds), 0.0, 1.0))

    def unknown_features(self, row: pd.DataFrame) -> List[str]:
        vocab = self.vocabs[0]
        unknown = []
        for col in CHAMPION_COLS:
            cid = int(row.iloc[0][col])
            if cid not in vocab:
                unknown.append(col)
        return unknown

    def explain_local(self, row: pd.DataFrame, score: float, top_n: int) -> Tuple[List[Dict[str, Any]], str]:
        if self.no_shap:
            return [], "disabled"
        return fallback_deltas_for_predictor(self, row, score, top_n), "fallback_delta"


class GlobalMeanPredictor(BasePredictor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("global_mean", REPORT_MODEL_NAMES["global_mean"], *args, **kwargs)
        train = pd.read_parquet(self.train_path, columns=["support_roam_score", "sample_weight"])
        self.mean = weighted_mean(train["support_roam_score"], train["sample_weight"])

    def predict_score(self, row: pd.DataFrame) -> float:
        return float(np.clip(self.mean, 0.0, 1.0))


class ChampionMeanPredictor(BasePredictor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__("champion_mean", REPORT_MODEL_NAMES["champion_mean"], *args, **kwargs)
        train = pd.read_parquet(
            self.train_path,
            columns=["ally_utility_champion_id", "support_roam_score", "sample_weight"],
        )
        self.global_mean = weighted_mean(train["support_roam_score"], train["sample_weight"])
        self.means = (
            train.groupby("ally_utility_champion_id")
            .apply(lambda g: weighted_mean(g["support_roam_score"], g["sample_weight"]))
            .to_dict()
        )

    def predict_score(self, row: pd.DataFrame) -> float:
        cid = int(row.iloc[0]["ally_utility_champion_id"])
        return float(np.clip(self.means.get(cid, self.global_mean), 0.0, 1.0))

    def unknown_features(self, row: pd.DataFrame) -> List[str]:
        cid = int(row.iloc[0]["ally_utility_champion_id"])
        return [] if cid in self.means else ["ally_utility_champion_id"]

    def explain_local(self, row: pd.DataFrame, score: float, top_n: int) -> Tuple[List[Dict[str, Any]], str]:
        if self.no_shap:
            return [], "disabled"
        return [
            {
                "feature": "ally_utility_champion_id",
                "value": self.display_value("ally_utility_champion_id", row.iloc[0]["ally_utility_champion_id"]),
                "contribution": float(score - self.global_mean),
                "direction": "sube" if score >= self.global_mean else "baja",
            }
        ], "baseline_delta"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = values.astype(float).to_numpy()
    w = weights.astype(float).to_numpy()
    return float((v * w).sum() / max(w.sum(), 1e-12))


def fallback_deltas_for_predictor(predictor: DraftPredictor, row: pd.DataFrame, score: float, top_n: int) -> List[Dict[str, Any]]:
    deltas: List[Dict[str, Any]] = []
    for col in predictor.feature_columns:
        if col == "side":
            continue
        replaced = row.copy()
        if col.endswith("_champion_id"):
            replaced.at[replaced.index[0], col] = row.iloc[0].get("ally_utility_champion_id")
        else:
            continue
        new_score = predictor.predict_score(replaced)
        delta = score - new_score
        deltas.append(
            {
                "feature": col,
                "value": predictor.display_value(col, row.iloc[0].get(col)) if isinstance(predictor, BasePredictor) else str(row.iloc[0].get(col)),
                "contribution": float(delta),
                "direction": "sube" if delta >= 0 else "baja",
            }
        )
    deltas.sort(key=lambda item: abs(float(item["contribution"])), reverse=True)
    return deltas[:top_n]


def build_predictor(
    key: str,
    models_root: Path,
    train_path: Path,
    champion_info: Mapping[int, ChampionInfo],
    report_metrics: Mapping[str, Mapping[str, str]],
    background_size: int,
    no_shap: bool,
) -> DraftPredictor:
    common = (train_path, champion_info, report_metrics, background_size, no_shap)
    if key == "histgbt":
        return GbtEnsemblePredictor(models_root / MODEL_DIRS[key], *common)
    if key in {"mlp_onehot", "mlp_embed", "mlp_per_role"}:
        return MLPEnsemblePredictor(key, models_root / MODEL_DIRS[key], *common)
    if key == "champion_mean":
        return ChampionMeanPredictor(*common)
    if key == "global_mean":
        return GlobalMeanPredictor(*common)
    raise ValueError(f"Unknown model key: {key}")


def build_predictors(
    selected: str,
    models_root: Path,
    train_path: Path,
    champion_info: Mapping[int, ChampionInfo],
    report_metrics: Mapping[str, Mapping[str, str]],
    background_size: int,
    no_shap: bool,
) -> Dict[str, DraftPredictor]:
    keys = MODEL_KEYS if selected == "all" else (selected,)
    return {
        key: build_predictor(key, models_root, train_path, champion_info, report_metrics, background_size, no_shap)
        for key in keys
    }


def build_row_from_values(
    values: Mapping[str, Any],
    feature_columns: Sequence[str],
    lookup: Mapping[str, int],
    champion_info: Mapping[int, ChampionInfo],
    interactive: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    row: Dict[str, Any] = {}
    side = values.get("side")
    if interactive and not side:
        side = prompt_value("Side del equipo aliado (blue/red)", "blue")
    side = str(side or "blue").lower()
    if side not in {"blue", "red"}:
        raise ValueError("side must be 'blue' or 'red'.")
    row["side"] = side

    for prefix in SIDES:
        side_label = "aliado" if prefix == "ally" else "enemigo"
        for role, role_label in ROLES:
            champ_col = f"{prefix}_{role}_champion_id"
            if champ_col in feature_columns:
                raw_champ = first_present(
                    values,
                    [champ_col, f"{prefix}_{role}_champion_name", f"{prefix}_{role}", f"{prefix}_{role}_champion"],
                )
                if interactive and (raw_champ is None or str(raw_champ).strip() == ""):
                    raw_champ = prompt_value(f"Campeon {side_label} {role_label}")
                row[champ_col] = resolve_champion(raw_champ, lookup, champion_info)

    for col in feature_columns:
        row.setdefault(col, "__MISSING__")
    return pd.DataFrame([row], columns=list(feature_columns)), []


def values_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    values: Dict[str, Any] = {"side": args.side}
    for prefix in SIDES:
        for role, _ in ROLES:
            values[f"{prefix}_{role}"] = getattr(args, f"{prefix}_{role}")
    return values


def row_champion_names(row: pd.DataFrame, champion_info: Mapping[int, ChampionInfo]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {"ally": {}, "enemy": {}}
    raw = row.iloc[0]
    for prefix in SIDES:
        for role, _ in ROLES:
            col = f"{prefix}_{role}_champion_id"
            if col in row.columns:
                cid = int(raw[col])
                out[prefix][role] = champion_info.get(cid, ChampionInfo(cid, str(cid))).name
    return out


def invert_side(side: Any) -> str:
    return "red" if str(side).lower() == "blue" else "blue"


def swap_team_perspective(row: pd.DataFrame) -> pd.DataFrame:
    swapped = row.copy()
    original = row.iloc[0].to_dict()
    for col in row.columns:
        if col.startswith("ally_"):
            other = "enemy_" + col[len("ally_"):]
            if other in row.columns:
                swapped.at[swapped.index[0], col] = original.get(other)
                swapped.at[swapped.index[0], other] = original.get(col)
    if "side" in swapped.columns:
        swapped.at[swapped.index[0], "side"] = invert_side(original.get("side", "blue"))
    return swapped


def predict_full(predictor: DraftPredictor, row: pd.DataFrame, assumptions: List[str], top_n: int) -> PredictionResult:
    score = predictor.predict_score(row)
    unknown = predictor.unknown_features(row)
    conf_label, conf_score = predictor.confidence(score, len(unknown))
    top_features, method = predictor.explain_local(row, score, top_n=top_n)
    return PredictionResult(
        score=score,
        profile=score_profile(score),
        confidence_label=conf_label,
        confidence_score=conf_score,
        percentile=predictor.target_percentile(score),
        unknown_features=unknown,
        top_features=top_features,
        explanation_method=method,
        assumptions=assumptions,
    )


def predict_matchup(
    predictor: DraftPredictor,
    row: pd.DataFrame,
    assumptions: List[str],
    top_n: int,
    explain_enemy: bool,
) -> MatchupResult:
    ally = predict_full(predictor, row, assumptions, top_n=top_n)
    original_no_shap = predictor.no_shap
    if not explain_enemy:
        predictor.no_shap = True
    enemy = predict_full(predictor, swap_team_perspective(row), assumptions, top_n=top_n)
    predictor.no_shap = original_no_shap
    delta = ally.score - enemy.score
    label, reading = matchup_profile(delta)
    return MatchupResult(ally=ally, enemy=enemy, delta=delta, label=label, reading=reading)


def predict_all_models(
    predictors: Mapping[str, DraftPredictor],
    row: pd.DataFrame,
    assumptions: List[str],
    top_n: int,
    explain_enemy: bool,
) -> Dict[str, MatchupResult]:
    return {
        key: predict_matchup(predictor, row, assumptions, top_n=top_n, explain_enemy=explain_enemy)
        for key, predictor in predictors.items()
    }


def print_prediction(
    row: pd.DataFrame,
    matchups: Mapping[str, MatchupResult],
    champion_info: Mapping[int, ChampionInfo],
    primary_key: str,
    predictors: Mapping[str, DraftPredictor],
) -> None:
    names = row_champion_names(row, champion_info)
    support_id = int(row.iloc[0]["ally_utility_champion_id"])
    support = champion_info.get(support_id, ChampionInfo(support_id, str(support_id)))
    enemy_support_id = int(row.iloc[0]["enemy_utility_champion_id"])
    enemy_support = champion_info.get(enemy_support_id, ChampionInfo(enemy_support_id, str(enemy_support_id)))
    matchup = matchups[primary_key]
    result = matchup.ally
    predictor = predictors[primary_key]

    print("\n=== Draft aliado ===")
    print("Top: {top} | Jungle: {jungle} | Mid: {middle} | ADC: {bottom} | Support: {utility}".format(**names["ally"]))
    print("\n=== Draft enemigo ===")
    print("Top: {top} | Jungle: {jungle} | Mid: {middle} | ADC: {bottom} | Support: {utility}".format(**names["enemy"]))

    print("\n=== Prediccion principal ===")
    print(f"Modelo: {predictor.model_label}")
    print(f"Support aliado: {support.name}")
    if support.support_archetype:
        print(f"Arquetipo de referencia: {support.support_archetype}")
    print(f"Score estimado: {result.score:.4f}")
    if result.percentile is not None:
        print(f"Percentil vs distribucion de train: {result.percentile:.1f}")
    print(f"Lectura: {result.profile}")
    print(f"Confianza: {result.confidence_label} ({result.confidence_score:.2f}; no es una probabilidad calibrada)")
    metrics = predictor.display_metrics()
    if metrics.get("r2"):
        print(
            f"Metricas informe: R2={metrics.get('r2')} | "
            f"Spearman={metrics.get('spearman_corr')} | MAE={metrics.get('mae')} | n={metrics.get('n_eval')}"
        )

    print("\n=== Matchup support vs support ===")
    print(f"Support aliado: {support.name} -> {matchup.ally.score:.4f} ({matchup.ally.profile})")
    print(f"Support enemigo: {enemy_support.name} -> {matchup.enemy.score:.4f} ({matchup.enemy.profile})")
    print(f"Diferencial aliado - enemigo: {matchup.delta:+.4f}")
    print(f"Lectura matchup: {matchup.label}")
    print(matchup.reading)
    print("Cuidado: esto compara tendencia de roaming, no poder de 2v2 ni probabilidad de ganar linea.")

    if len(matchups) > 1:
        print("\n=== Comparacion de modelos finales ===")
        for key, item in sorted(matchups.items(), key=lambda kv: kv[1].ally.score, reverse=True):
            print(f"- {predictors[key].model_label}: ally={item.ally.score:.4f} enemy={item.enemy.score:.4f} delta={item.delta:+.4f}")

    if result.top_features:
        method_label = "SHAP local" if result.explanation_method == "shap_permutation" else "delta local"
        print(f"\nTop features que influyen ({method_label}):")
        for item in result.top_features:
            sign = "+" if item["contribution"] >= 0 else ""
            print(f"- {item['feature']}={item['value']}: {sign}{item['contribution']:.4f} ({item['direction']} el score)")
    elif result.explanation_method == "disabled":
        print("\nExplicacion local: desactivada por --no-shap.")

    if result.unknown_features:
        print("\nAviso: features no vistas en entrenamiento: " + ", ".join(result.unknown_features))
    print("\nNota: el modelo solo observa draft pregame; no observa eventos reales de la partida.")


def prediction_to_dict(
    row: pd.DataFrame,
    result: PredictionResult,
    champion_info: Mapping[int, ChampionInfo],
    prefix: str,
) -> Dict[str, Any]:
    support_id = int(row.iloc[0][f"{prefix}_utility_champion_id"])
    support = champion_info.get(support_id, ChampionInfo(support_id, str(support_id)))
    return {
        "support_id": support_id,
        "support_name": support.name,
        "score": result.score,
        "profile": result.profile,
        "confidence_label": result.confidence_label,
        "confidence_score": result.confidence_score,
        "percentile_train": result.percentile,
        "top_features": result.top_features,
        "explanation_method": result.explanation_method,
        "unknown_features": result.unknown_features,
    }


def matchup_to_dict(
    row: pd.DataFrame,
    matchup: MatchupResult,
    champion_info: Mapping[int, ChampionInfo],
    predictor: DraftPredictor,
) -> Dict[str, Any]:
    support_id = int(row.iloc[0]["ally_utility_champion_id"])
    support = champion_info.get(support_id, ChampionInfo(support_id, str(support_id)))
    enemy_support_id = int(row.iloc[0]["enemy_utility_champion_id"])
    enemy_support = champion_info.get(enemy_support_id, ChampionInfo(enemy_support_id, str(enemy_support_id)))
    return {
        "model_key": predictor.key,
        "model_label": predictor.model_label,
        "model_metrics": predictor.display_metrics(),
        "ally_support_id": support_id,
        "ally_support_name": support.name,
        "ally_score": matchup.ally.score,
        "ally_profile": matchup.ally.profile,
        "ally_confidence_label": matchup.ally.confidence_label,
        "ally_confidence_score": matchup.ally.confidence_score,
        "ally_percentile_train": matchup.ally.percentile,
        "enemy_support_id": enemy_support_id,
        "enemy_support_name": enemy_support.name,
        "enemy_score": matchup.enemy.score,
        "enemy_profile": matchup.enemy.profile,
        "enemy_confidence_label": matchup.enemy.confidence_label,
        "enemy_confidence_score": matchup.enemy.confidence_score,
        "enemy_percentile_train": matchup.enemy.percentile,
        "matchup_delta": matchup.delta,
        "matchup_label": matchup.label,
        "matchup_reading": matchup.reading,
        "ally_top_features": matchup.ally.top_features,
        "enemy_top_features": matchup.enemy.top_features,
        "ally_explanation_method": matchup.ally.explanation_method,
        "enemy_explanation_method": matchup.enemy.explanation_method,
        "ally_unknown_features": matchup.ally.unknown_features,
        "enemy_unknown_features": matchup.enemy.unknown_features,
        "assumptions": matchup.ally.assumptions,
        "ally": prediction_to_dict(row, matchup.ally, champion_info, "ally"),
        "enemy": prediction_to_dict(row, matchup.enemy, champion_info, "enemy"),
        "matchup": {"delta": matchup.delta, "label": matchup.label, "reading": matchup.reading},
    }


def result_to_dict(
    row: pd.DataFrame,
    matchups: Mapping[str, MatchupResult],
    champion_info: Mapping[int, ChampionInfo],
    primary_key: str,
    predictors: Mapping[str, DraftPredictor],
) -> Dict[str, Any]:
    primary = matchup_to_dict(row, matchups[primary_key], champion_info, predictors[primary_key])
    if len(matchups) > 1:
        primary["model_predictions"] = {
            key: matchup_to_dict(row, matchup, champion_info, predictors[key])
            for key, matchup in matchups.items()
        }
    return primary


def read_batch(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        return pd.read_json(path)
    raise ValueError("Batch input must be .csv, .json or .jsonl")


def run_batch(
    args: argparse.Namespace,
    predictors: Mapping[str, DraftPredictor],
    lookup: Mapping[str, int],
    champion_info: Mapping[int, ChampionInfo],
    primary_key: str,
) -> None:
    df = read_batch(args.batch)
    records: List[Dict[str, Any]] = []
    original_no_shap = {key: predictor.no_shap for key, predictor in predictors.items()}
    if not args.explain_batch:
        for predictor in predictors.values():
            predictor.no_shap = True
    for idx, raw in df.iterrows():
        try:
            row, assumptions = build_row_from_values(
                raw.to_dict(),
                CANONICAL_FEATURE_COLUMNS,
                lookup,
                champion_info,
                interactive=False,
            )
            matchups = predict_all_models(
                predictors,
                row,
                assumptions,
                top_n=args.top_n,
                explain_enemy=args.explain_enemy and args.explain_batch,
            )
            payload = result_to_dict(row, matchups, champion_info, primary_key, predictors)
            if len(matchups) > 1:
                for key, matchup in matchups.items():
                    payload[f"score_{key}"] = matchup.ally.score
                    payload[f"profile_{key}"] = matchup.ally.profile
            payload["row_index"] = int(idx)
            payload["status"] = "ok"
        except Exception as exc:
            payload = {"row_index": int(idx), "status": "error", "error": str(exc)}
        records.append(payload)
    for key, value in original_no_shap.items():
        predictors[key].no_shap = value

    out = pd.DataFrame(records)
    json_cols = [
        "model_metrics",
        "ally_top_features",
        "enemy_top_features",
        "ally_unknown_features",
        "enemy_unknown_features",
        "assumptions",
        "ally",
        "enemy",
        "matchup",
        "model_predictions",
    ]
    for col in json_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)

    if args.output:
        out.to_csv(args.output, index=False)
        print(f"[Saved] {args.output}")
    else:
        print(out.to_csv(index=False))


def add_draft_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--side", choices=["blue", "red"], default=None, help="Ally team side.")
    for prefix in SIDES:
        for role, _ in ROLES:
            parser.add_argument(f"--{prefix}-{role}", dest=f"{prefix}_{role}", default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict support roaming score from 10 champions + side using the final report models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=list(MODEL_KEYS) + ["all"], default="histgbt")
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--metrics-table", type=Path, default=DEFAULT_METRICS_TABLE)
    parser.add_argument("--champion-classes", type=Path, default=DEFAULT_CHAMPION_CLASSES)
    parser.add_argument("--champion-archetypes", type=Path, default=DEFAULT_CHAMPION_ARCHETYPES)
    parser.add_argument("--batch", type=Path, default=None, help="CSV/JSON/JSONL file with drafts.")
    parser.add_argument("--output", type=Path, default=None, help="CSV output for batch mode.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON for single-draft mode.")
    parser.add_argument("--no-interactive", action="store_true", help="Do not prompt for missing champions.")
    parser.add_argument("--no-shap", action="store_true", help="Disable local explanation.")
    parser.add_argument("--explain-batch", action="store_true", help="Compute local explanations in batch mode.")
    parser.add_argument("--explain-enemy", action="store_true", help="Also compute local explanation for enemy support.")
    parser.add_argument("--background-size", type=int, default=80, help="Background rows for SHAP/fallback.")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--list-champions", nargs="?", const="", default=None)
    parser.add_argument("--example-batch", action="store_true", help="Print a minimal batch CSV template.")
    add_draft_args(parser)
    return parser.parse_args()


def print_example_batch() -> None:
    columns = ["side"] + [f"{prefix}_{role}" for prefix in SIDES for role, _ in ROLES]
    row = [
        "blue",
        "Ornn",
        "Lee Sin",
        "Ahri",
        "Jinx",
        "Nautilus",
        "Gwen",
        "Viego",
        "Orianna",
        "Kai'Sa",
        "Lulu",
    ]
    print(",".join(columns))
    print(",".join(row))


def main() -> None:
    args = parse_args()

    champion_info = load_champion_info(args.champion_classes, args.champion_archetypes)
    lookup = build_champion_lookup(champion_info)

    if args.list_champions is not None:
        needle = normalize_key(args.list_champions)
        for champion in sorted(champion_info.values(), key=lambda c: c.name):
            if not needle or needle in normalize_key(champion.name):
                print(f"{champion.name} ({champion.champion_id})")
        return

    if args.example_batch:
        print_example_batch()
        return

    report_metrics = load_report_metrics(args.metrics_table)
    predictors = build_predictors(
        args.model,
        args.models_root,
        args.train_path,
        champion_info,
        report_metrics,
        args.background_size,
        args.no_shap,
    )
    primary_key = "histgbt" if args.model == "all" else args.model

    if args.batch:
        run_batch(args, predictors, lookup, champion_info, primary_key)
        return

    try:
        row, assumptions = build_row_from_values(
            values_from_args(args),
            CANONICAL_FEATURE_COLUMNS,
            lookup,
            champion_info,
            interactive=not args.no_interactive,
        )
        matchups = predict_all_models(
            predictors,
            row,
            assumptions,
            top_n=args.top_n,
            explain_enemy=args.explain_enemy,
        )
    except (ValueError, TypeError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.json:
        print(json.dumps(result_to_dict(row, matchups, champion_info, primary_key, predictors), indent=2, ensure_ascii=False))
    else:
        print_prediction(row, matchups, champion_info, primary_key, predictors)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
