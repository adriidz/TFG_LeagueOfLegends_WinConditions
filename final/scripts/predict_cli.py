#!/usr/bin/env python3
"""
Polished CLI prototype for support roaming inference from draft data.

The CLI loads the final HistGradientBoosting model and predicts the expected
support_roam_score from champion select features. It supports:

  - interactive single-draft mode,
  - non-interactive flags,
  - batch CSV/JSON drafts,
  - nearest champions from learned MLP embeddings,
  - local SHAP explanations when available.

The final GBT was trained with champion ids, summoner spells and side. The
prototype asks primarily for the 10 champions + side; if summoner spells are
not supplied it fills standard role defaults and reports that assumption.
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Avoid a noisy loky warning on Windows when joblib cannot inspect physical cores.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "final" / "models" / "gbt"
DEFAULT_TRAIN_PATH = REPO_ROOT / "final" / "data" / "training" / "train.parquet"
DEFAULT_CHAMPION_CLASSES = REPO_ROOT / "final" / "data" / "champion_classes.json"
DEFAULT_CHAMPION_ARCHETYPES = REPO_ROOT / "final" / "data" / "champion_archetypes.json"
DEFAULT_EMBEDDINGS = REPO_ROOT / "final" / "models" / "mlp_per_role" / "embeddings_raw_ally_utility.json"

ROLES: Tuple[Tuple[str, str], ...] = (
    ("top", "Top"),
    ("jungle", "Jungle"),
    ("middle", "Mid"),
    ("bottom", "ADC"),
    ("utility", "Support"),
)
SIDES: Tuple[str, str] = ("ally", "enemy")

SPELL_IDS: Dict[str, int] = {
    "cleanse": 1,
    "exhaust": 3,
    "flash": 4,
    "ghost": 6,
    "heal": 7,
    "smite": 11,
    "teleport": 12,
    "ignite": 14,
    "barrier": 21,
}
SPELL_NAMES_BY_ID: Dict[int, str] = {spell_id: name for name, spell_id in SPELL_IDS.items()}
DEFAULT_ROLE_SPELLS: Dict[str, Tuple[int, int]] = {
    "top": (4, 12),
    "jungle": (4, 11),
    "middle": (4, 12),
    "bottom": (4, 21),
    "utility": (4, 14),
}

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
    similar_champions: List[Dict[str, Any]]
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


def parse_spell_pair(raw: Optional[str], role: str) -> Tuple[Tuple[int, int], bool]:
    if raw is None or not str(raw).strip():
        return DEFAULT_ROLE_SPELLS[role], True
    tokens = [t for t in re.split(r"[,/|+ ]+", str(raw).strip()) if t]
    if len(tokens) != 2:
        raise ValueError(f"Expected two summoner spells for {role}, got: {raw}")
    values: List[int] = []
    for token in tokens:
        if token.isdigit():
            values.append(int(token))
            continue
        key = normalize_key(token)
        if key not in SPELL_IDS:
            valid = ", ".join(sorted(SPELL_IDS))
            raise ValueError(f"Unknown summoner spell '{token}'. Valid names: {valid}")
        values.append(SPELL_IDS[key])
    return (values[0], values[1]), False


def format_spell_pair(spells: Tuple[int, int]) -> str:
    return ",".join(SPELL_NAMES_BY_ID.get(spell_id, str(spell_id)) for spell_id in spells)


def load_champion_info(classes_path: Path, archetypes_path: Path) -> Dict[int, ChampionInfo]:
    classes_raw = read_json(classes_path) if classes_path.exists() else {}
    archetypes_raw = read_json(archetypes_path) if archetypes_path.exists() else {}
    archetype_champs = archetypes_raw.get("champions", {}) if isinstance(archetypes_raw, dict) else {}

    ids: set[int] = set()
    for key in classes_raw.keys():
        try:
            ids.add(int(key))
        except (TypeError, ValueError):
            pass
    for key in archetype_champs.keys():
        try:
            ids.add(int(key))
        except (TypeError, ValueError):
            pass

    info: Dict[int, ChampionInfo] = {}
    for cid in sorted(ids):
        class_entry = classes_raw.get(str(cid), {}) if isinstance(classes_raw, dict) else {}
        archetype_entry = archetype_champs.get(str(cid), {}) if isinstance(archetype_champs, dict) else {}
        name = (
            class_entry.get("name")
            or archetype_entry.get("name")
            or f"champ_{cid}"
        )
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


def categorical_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.loc[:, list(columns)].copy()
    for col in columns:
        out[col] = out[col].where(out[col].notna(), "__MISSING__").astype(str)
    return out


class GbtDraftPredictor:
    def __init__(
        self,
        model_dir: Path,
        train_path: Path,
        champion_info: Mapping[int, ChampionInfo],
        embeddings_path: Path,
        background_size: int,
        no_shap: bool,
    ) -> None:
        self.model_dir = model_dir
        self.train_path = train_path
        self.champion_info = champion_info
        self.embeddings_path = embeddings_path
        self.background_size = background_size
        self.no_shap = no_shap

        self.model = joblib.load(model_dir / "gbt_model_raw.joblib")
        preprocess = joblib.load(model_dir / "preprocess.joblib")
        self.encoder = preprocess["encoder"]
        self.feature_columns = list(preprocess["feature_columns"])
        self.metrics = self._load_metrics(model_dir / "metrics.json")
        self._category_sets = [set(map(str, cats)) for cats in self.encoder.categories_]
        self._target_values: Optional[np.ndarray] = None
        self._background: Optional[np.ndarray] = None
        self._background_mode: Optional[np.ndarray] = None
        self._embedding_frame: Optional[pd.DataFrame] = None
        self._embedding_matrix: Optional[np.ndarray] = None

    @staticmethod
    def _load_metrics(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        rows = read_json(path)
        if isinstance(rows, list):
            for row in rows:
                if row.get("target") == "raw":
                    return dict(row)
        return rows if isinstance(rows, dict) else {}

    def encode(self, row: pd.DataFrame) -> np.ndarray:
        for col in self.feature_columns:
            if col not in row.columns:
                row[col] = "__MISSING__"
        return self.encoder.transform(categorical_frame(row, self.feature_columns))

    def predict_score(self, row: pd.DataFrame) -> float:
        encoded = self.encode(row)
        return float(np.clip(self.model.predict(encoded)[0], 0.0, 1.0))

    def unknown_features(self, row: pd.DataFrame) -> List[str]:
        unknown: List[str] = []
        for idx, col in enumerate(self.feature_columns):
            value = str(row.iloc[0].get(col, "__MISSING__"))
            if value not in self._category_sets[idx]:
                unknown.append(col)
        return unknown

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
        spearman = float(self.metrics.get("spearman_corr", 0.38) or 0.38)
        model_quality = min(max(spearman / 0.45, 0.0), 1.0)
        value = 0.45 * coverage + 0.35 * stability + 0.20 * model_quality
        return confidence_label(value), float(value)

    def load_background(self) -> Optional[np.ndarray]:
        if self._background is not None:
            return self._background
        if not self.train_path.exists() or self.background_size <= 0:
            return None
        try:
            df = pd.read_parquet(self.train_path, columns=self.feature_columns)
            if len(df) > self.background_size:
                df = df.sample(n=self.background_size, random_state=42)
            self._background = self.encoder.transform(categorical_frame(df, self.feature_columns))
            self._background_mode = np.asarray(
                [
                    pd.Series(self._background[:, i]).mode(dropna=False).iloc[0]
                    for i in range(self._background.shape[1])
                ],
                dtype=np.float32,
            )
        except Exception:
            self._background = None
            self._background_mode = None
        return self._background

    def explain_local(
        self,
        row: pd.DataFrame,
        score: float,
        top_n: int,
    ) -> Tuple[List[Dict[str, Any]], str]:
        if self.no_shap:
            return [], "disabled"

        encoded = self.encode(row)
        background = self.load_background()
        if background is None or len(background) == 0:
            return self._fallback_deltas(row, encoded, score, top_n), "fallback_delta"

        try:
            import shap

            masker = shap.maskers.Independent(background)
            explainer = shap.PermutationExplainer(self.model.predict, masker, seed=42)
            values = explainer(encoded, max_evals=2 * encoded.shape[1] + 1)
            contribs = np.asarray(values.values[0], dtype=float)
            base_value = float(np.asarray(values.base_values).reshape(-1)[0])
            return self._format_contributions(row, contribs, top_n, base_value), "shap_permutation"
        except Exception:
            return self._fallback_deltas(row, encoded, score, top_n), "fallback_delta"

    def _fallback_deltas(
        self,
        row: pd.DataFrame,
        encoded: np.ndarray,
        score: float,
        top_n: int,
    ) -> List[Dict[str, Any]]:
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
            deltas.append(score - float(self.model.predict(replaced)[0]))
        return self._format_contributions(row, np.asarray(deltas, dtype=float), top_n, None)

    def _format_contributions(
        self,
        row: pd.DataFrame,
        contribs: np.ndarray,
        top_n: int,
        base_value: Optional[float],
    ) -> List[Dict[str, Any]]:
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
                    "base_value": base_value,
                }
            )
        return out

    def display_value(self, col: str, value: Any) -> str:
        if col.endswith("_champion_id"):
            try:
                cid = int(float(value))
                return self.champion_info.get(cid, ChampionInfo(cid, str(cid))).name
            except (TypeError, ValueError):
                return str(value)
        if col.endswith("_summoner1_id") or col.endswith("_summoner2_id"):
            try:
                return SPELL_NAMES_BY_ID.get(int(float(value)), str(value))
            except (TypeError, ValueError):
                return str(value)
        return str(value)

    def similar_champions(self, champion_id: int, top_n: int) -> List[Dict[str, Any]]:
        if self._embedding_frame is None or self._embedding_matrix is None:
            self._load_embeddings()
        if self._embedding_frame is None or self._embedding_matrix is None:
            return []

        frame = self._embedding_frame
        matches = frame.index[frame["champion_id"] == int(champion_id)].tolist()
        if not matches:
            return []
        idx = matches[0]
        sims = self._embedding_matrix @ self._embedding_matrix[idx]
        order = np.argsort(sims)[::-1]
        out: List[Dict[str, Any]] = []
        for neighbor_idx in order:
            if int(neighbor_idx) == int(idx):
                continue
            row = frame.iloc[int(neighbor_idx)]
            out.append(
                {
                    "champion_id": int(row["champion_id"]),
                    "name": str(row["name"]),
                    "similarity": float(sims[int(neighbor_idx)]),
                    "support_archetype": str(row.get("support_archetype", "")),
                }
            )
            if len(out) >= top_n:
                break
        return out

    def _load_embeddings(self) -> None:
        if not self.embeddings_path.exists():
            return
        try:
            rows = read_json(self.embeddings_path)
        except Exception:
            return
        records: List[Dict[str, Any]] = []
        vectors: List[List[float]] = []
        for row in rows:
            try:
                cid = int(row["champion_id"])
                vector = [float(v) for v in row["embedding"]]
            except (KeyError, TypeError, ValueError):
                continue
            champion = self.champion_info.get(cid)
            records.append(
                {
                    "champion_id": cid,
                    "name": champion.name if champion else str(row.get("name", cid)),
                    "support_archetype": champion.support_archetype if champion else str(row.get("archetype", "")),
                }
            )
            vectors.append(vector)
        if not records:
            return
        self._embedding_frame = pd.DataFrame.from_records(records)
        self._embedding_matrix = normalize(np.asarray(vectors, dtype=np.float32), norm="l2")

    def predict_full(self, row: pd.DataFrame, assumptions: List[str], top_n: int) -> PredictionResult:
        score = self.predict_score(row)
        unknown = self.unknown_features(row)
        conf_label, conf_score = self.confidence(score, len(unknown))
        support_id = int(row.iloc[0]["ally_utility_champion_id"])
        similar = self.similar_champions(support_id, top_n=top_n)
        top_features, method = self.explain_local(row, score, top_n=top_n)
        return PredictionResult(
            score=score,
            profile=score_profile(score),
            confidence_label=conf_label,
            confidence_score=conf_score,
            percentile=self.target_percentile(score),
            unknown_features=unknown,
            similar_champions=similar,
            top_features=top_features,
            explanation_method=method,
            assumptions=assumptions,
        )


def default_spell_assumption(prefix: str, role: str, spells: Tuple[int, int]) -> str:
    return f"{prefix}_{role}_spells={format_spell_pair(spells)}"


def build_row_from_values(
    values: Mapping[str, Any],
    feature_columns: Sequence[str],
    lookup: Mapping[str, int],
    champion_info: Mapping[int, ChampionInfo],
    interactive: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    row: Dict[str, Any] = {}
    assumptions: List[str] = []

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
                    [
                        champ_col,
                        f"{prefix}_{role}_champion_name",
                        f"{prefix}_{role}",
                        f"{prefix}_{role}_champion",
                    ],
                )
                if interactive and (raw_champ is None or str(raw_champ).strip() == ""):
                    raw_champ = prompt_value(f"Campeon {side_label} {role_label}")
                row[champ_col] = resolve_champion(raw_champ, lookup, champion_info)

            spell1_col = f"{prefix}_{role}_summoner1_id"
            spell2_col = f"{prefix}_{role}_summoner2_id"
            if spell1_col in feature_columns or spell2_col in feature_columns:
                raw_spell_pair = first_present(
                    values,
                    [
                        f"{prefix}_{role}_spells",
                        f"{prefix}_{role}_summoners",
                    ],
                )
                spell1 = values.get(spell1_col)
                spell2 = values.get(spell2_col)
                if spell1 is not None and spell2 is not None and str(spell1) != "" and str(spell2) != "":
                    spells = (int(float(spell1)), int(float(spell2)))
                    used_default = False
                else:
                    spells, used_default = parse_spell_pair(raw_spell_pair, role)
                row[spell1_col] = spells[0]
                row[spell2_col] = spells[1]
                if used_default:
                    assumptions.append(default_spell_assumption(prefix, role, spells))

    for col in feature_columns:
        row.setdefault(col, "__MISSING__")
    return pd.DataFrame([row], columns=list(feature_columns)), assumptions


def first_present(values: Mapping[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in values and pd.notna(values[key]) and str(values[key]).strip() != "":
            return values[key]
    return None


def values_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    values: Dict[str, Any] = {"side": args.side}
    for prefix in SIDES:
        for role, _ in ROLES:
            values[f"{prefix}_{role}"] = getattr(args, f"{prefix}_{role}")
            values[f"{prefix}_{role}_spells"] = getattr(args, f"{prefix}_{role}_spells")
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


def predict_matchup(
    predictor: GbtDraftPredictor,
    row: pd.DataFrame,
    assumptions: List[str],
    top_n: int,
    explain_enemy: bool,
) -> MatchupResult:
    ally = predictor.predict_full(row, assumptions, top_n=top_n)
    original_no_shap = predictor.no_shap
    if not explain_enemy:
        predictor.no_shap = True
    enemy = predictor.predict_full(swap_team_perspective(row), assumptions, top_n=top_n)
    predictor.no_shap = original_no_shap
    delta = ally.score - enemy.score
    label, reading = matchup_profile(delta)
    return MatchupResult(ally=ally, enemy=enemy, delta=delta, label=label, reading=reading)


def print_prediction(
    row: pd.DataFrame,
    matchup: MatchupResult,
    champion_info: Mapping[int, ChampionInfo],
    predictor: GbtDraftPredictor,
) -> None:
    names = row_champion_names(row, champion_info)
    support_id = int(row.iloc[0]["ally_utility_champion_id"])
    support = champion_info.get(support_id, ChampionInfo(support_id, str(support_id)))
    enemy_support_id = int(row.iloc[0]["enemy_utility_champion_id"])
    enemy_support = champion_info.get(enemy_support_id, ChampionInfo(enemy_support_id, str(enemy_support_id)))
    result = matchup.ally

    print("\n=== Draft aliado ===")
    print(
        "Top: {top} | Jungle: {jungle} | Mid: {middle} | ADC: {bottom} | Support: {utility}".format(
            **names["ally"]
        )
    )
    print("\n=== Draft enemigo ===")
    print(
        "Top: {top} | Jungle: {jungle} | Mid: {middle} | ADC: {bottom} | Support: {utility}".format(
            **names["enemy"]
        )
    )

    print("\n=== Prediccion ===")
    print(f"Support aliado: {support.name}")
    if support.support_archetype:
        print(f"Arquetipo de referencia: {support.support_archetype}")
    print(f"Score estimado: {result.score:.4f}")
    if result.percentile is not None:
        print(f"Percentil vs distribucion de train: {result.percentile:.1f}")
    print(f"Lectura: {result.profile}")
    print(
        f"Confianza: {result.confidence_label} "
        f"({result.confidence_score:.2f}; no es una probabilidad calibrada)"
    )
    spearman = predictor.metrics.get("spearman_corr")
    r2 = predictor.metrics.get("r2")
    if spearman is not None and r2 is not None:
        print(f"Modelo: HistGBT raw, val R2={float(r2):.3f}, Spearman={float(spearman):.3f}")

    print("\n=== Matchup support vs support ===")
    print(f"Support aliado: {support.name} -> {matchup.ally.score:.4f} ({matchup.ally.profile})")
    print(f"Support enemigo: {enemy_support.name} -> {matchup.enemy.score:.4f} ({matchup.enemy.profile})")
    print(f"Diferencial aliado - enemigo: {matchup.delta:+.4f}")
    print(f"Lectura matchup: {matchup.label}")
    print(matchup.reading)
    print("Cuidado: esto compara tendencia de roaming, no poder de 2v2 ni probabilidad de ganar linea.")

    if result.similar_champions:
        print("\nCampeones similares por embeddings:")
        for item in result.similar_champions:
            arch = f", {item['support_archetype']}" if item.get("support_archetype") else ""
            print(f"- {item['name']} (sim={item['similarity']:.3f}{arch})")

    if result.top_features:
        method_label = "SHAP local" if result.explanation_method == "shap_permutation" else "delta local"
        print(f"\nTop features que influyen ({method_label}):")
        for item in result.top_features:
            sign = "+" if item["contribution"] >= 0 else ""
            print(
                f"- {item['feature']}={item['value']}: "
                f"{sign}{item['contribution']:.4f} ({item['direction']} el score)"
            )
    elif result.explanation_method == "disabled":
        print("\nExplicacion local: desactivada por --no-shap.")

    if result.unknown_features:
        print("\nAviso: features no vistas en entrenamiento: " + ", ".join(result.unknown_features))
    if result.assumptions:
        print("\nSuposiciones usadas:")
        for assumption in result.assumptions:
            print(f"- {assumption}")
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
        "similar_champions": result.similar_champions,
        "top_features": result.top_features,
        "explanation_method": result.explanation_method,
        "unknown_features": result.unknown_features,
    }


def result_to_dict(row: pd.DataFrame, matchup: MatchupResult, champion_info: Mapping[int, ChampionInfo]) -> Dict[str, Any]:
    support_id = int(row.iloc[0]["ally_utility_champion_id"])
    support = champion_info.get(support_id, ChampionInfo(support_id, str(support_id)))
    enemy_support_id = int(row.iloc[0]["enemy_utility_champion_id"])
    enemy_support = champion_info.get(enemy_support_id, ChampionInfo(enemy_support_id, str(enemy_support_id)))
    return {
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
        "ally_similar_champions": matchup.ally.similar_champions,
        "enemy_similar_champions": matchup.enemy.similar_champions,
        "ally_top_features": matchup.ally.top_features,
        "enemy_top_features": matchup.enemy.top_features,
        "ally_explanation_method": matchup.ally.explanation_method,
        "enemy_explanation_method": matchup.enemy.explanation_method,
        "ally_unknown_features": matchup.ally.unknown_features,
        "enemy_unknown_features": matchup.enemy.unknown_features,
        "assumptions": matchup.ally.assumptions,
        "ally": prediction_to_dict(row, matchup.ally, champion_info, "ally"),
        "enemy": prediction_to_dict(row, matchup.enemy, champion_info, "enemy"),
        "matchup": {
            "delta": matchup.delta,
            "label": matchup.label,
            "reading": matchup.reading,
        },
    }


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
    predictor: GbtDraftPredictor,
    lookup: Mapping[str, int],
    champion_info: Mapping[int, ChampionInfo],
) -> None:
    df = read_batch(args.batch)
    records: List[Dict[str, Any]] = []
    original_no_shap = predictor.no_shap
    if not args.explain_batch:
        predictor.no_shap = True
    for idx, raw in df.iterrows():
        try:
            row, assumptions = build_row_from_values(
                raw.to_dict(),
                predictor.feature_columns,
                lookup,
                champion_info,
                interactive=False,
            )
            matchup = predict_matchup(
                predictor,
                row,
                assumptions,
                top_n=args.top_n,
                explain_enemy=args.explain_enemy and args.explain_batch,
            )
            payload = result_to_dict(row, matchup, champion_info)
            payload["row_index"] = int(idx)
            payload["status"] = "ok"
        except Exception as exc:
            payload = {"row_index": int(idx), "status": "error", "error": str(exc)}
        records.append(payload)
    predictor.no_shap = original_no_shap

    out = pd.DataFrame(records)
    json_cols = [
        "ally_similar_champions",
        "enemy_similar_champions",
        "ally_top_features",
        "enemy_top_features",
        "ally_unknown_features",
        "enemy_unknown_features",
        "assumptions",
        "ally",
        "enemy",
        "matchup",
    ]
    for col in json_cols:
        if col in out.columns:
            out[col] = out[col].map(
                lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
            )

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
            parser.add_argument(
                f"--{prefix}-{role}-spells",
                dest=f"{prefix}_{role}_spells",
                default=None,
                help="Two spells by name or id, e.g. flash,ignite or 4,14.",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict support roaming score from 10 champions + side using the final GBT model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--champion-classes", type=Path, default=DEFAULT_CHAMPION_CLASSES)
    parser.add_argument("--champion-archetypes", type=Path, default=DEFAULT_CHAMPION_ARCHETYPES)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--batch", type=Path, default=None, help="CSV/JSON/JSONL file with drafts.")
    parser.add_argument("--output", type=Path, default=None, help="CSV output for batch mode.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON for single-draft mode.")
    parser.add_argument("--no-interactive", action="store_true", help="Do not prompt for missing champions.")
    parser.add_argument("--no-shap", action="store_true", help="Disable local SHAP explanation.")
    parser.add_argument("--explain-batch", action="store_true", help="Compute local explanations in batch mode.")
    parser.add_argument("--explain-enemy", action="store_true", help="Also compute local explanation for enemy support.")
    parser.add_argument("--background-size", type=int, default=80, help="Background rows for SHAP.")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--list-champions", nargs="?", const="", default=None)
    parser.add_argument("--list-spells", action="store_true")
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

    if args.list_spells:
        for name, spell_id in sorted(SPELL_IDS.items(), key=lambda item: item[1]):
            print(f"{name}: {spell_id}")
        return

    if args.list_champions is not None:
        needle = normalize_key(args.list_champions)
        for champion in sorted(champion_info.values(), key=lambda c: c.name):
            if not needle or needle in normalize_key(champion.name):
                print(f"{champion.name} ({champion.champion_id})")
        return

    if args.example_batch:
        print_example_batch()
        return

    required = [
        args.model_dir / "gbt_model_raw.joblib",
        args.model_dir / "preprocess.joblib",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing model artifacts:\n" + "\n".join(f"- {path}" for path in missing))

    predictor = GbtDraftPredictor(
        model_dir=args.model_dir,
        train_path=args.train_path,
        champion_info=champion_info,
        embeddings_path=args.embeddings,
        background_size=args.background_size,
        no_shap=args.no_shap,
    )

    if args.batch:
        run_batch(args, predictor, lookup, champion_info)
        return

    try:
        row, assumptions = build_row_from_values(
            values_from_args(args),
            predictor.feature_columns,
            lookup,
            champion_info,
            interactive=not args.no_interactive,
        )
        matchup = predict_matchup(
            predictor,
            row,
            assumptions,
            top_n=args.top_n,
            explain_enemy=args.explain_enemy,
        )
    except (ValueError, TypeError) as exc:
        raise SystemExit(str(exc)) from exc

    if args.json:
        print(json.dumps(result_to_dict(row, matchup, champion_info), indent=2, ensure_ascii=False))
    else:
        print_prediction(row, matchup, champion_info, predictor)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
