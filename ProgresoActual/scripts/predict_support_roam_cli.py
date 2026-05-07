#!/usr/bin/env python3
"""
Terminal prototype for support roaming inference from champion select data.

The prototype loads an already trained support-only MLP run:

    model_config.json + preprocess.joblib + best_model.pt

It then builds a single-row draft feature table, applies the exact
OneHotEncoder used during training, and returns interpretable support roam
predictions for both teams by running the model from each team's perspective.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / "ProgresoActual" / "models" / "support_mlp_full_m12"
DEFAULT_DRAFT_PATH = REPO_ROOT / "ProgresoActual" / "data" / "clean" / "features" / "draft_features.parquet"
DEFAULT_MODEL_INPUT_PATH = REPO_ROOT / "ProgresoActual" / "data" / "training" / "model_input_support_regression_m12.parquet"
DEFAULT_REFERENCE_PATH = REPO_ROOT / "ProgresoActual" / "references" / "champion_support_reference.csv"

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
ROLE_SPELL_HINTS: Dict[str, str] = {
    role: ",".join(SPELL_NAMES_BY_ID[spell_id] for spell_id in spell_ids)
    for role, spell_ids in DEFAULT_ROLE_SPELLS.items()
}


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


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def categorical_frame(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df[columns].copy()
    for col in columns:
        out[col] = out[col].where(out[col].notna(), "__MISSING__").astype(str)
    return out


def load_model(model_dir: Path) -> Tuple[SupportMLP, Dict[str, Any], Dict[str, Any]]:
    config_path = model_dir / "model_config.json"
    preprocess_path = model_dir / "preprocess.joblib"
    model_path = model_dir / "best_model.pt"
    missing = [p for p in (config_path, preprocess_path, model_path) if not p.exists()]
    if missing:
        raise SystemExit("Missing model artifacts:\n" + "\n".join(f"- {p}" for p in missing))

    config = read_json(config_path)
    preprocess = joblib.load(preprocess_path)
    input_dim = int(config.get("onehot_dim") or len(preprocess["onehot"].get_feature_names_out()))
    model = SupportMLP(
        input_dim=input_dim,
        hidden1=int(config.get("hidden1", 256)),
        hidden2=int(config.get("hidden2", 128)),
        dropout=float(config.get("dropout", 0.2)),
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, config, preprocess


def iter_champion_pairs(df: pd.DataFrame) -> Iterable[Tuple[str, int]]:
    for name_col in [c for c in df.columns if c.endswith("_champion_name")]:
        id_col = name_col.replace("_champion_name", "_champion_id")
        if id_col not in df.columns:
            continue
        pairs = df[[name_col, id_col]].dropna().drop_duplicates()
        for name, champion_id in pairs.itertuples(index=False):
            try:
                yield str(name), int(champion_id)
            except (TypeError, ValueError):
                continue


def build_champion_lookup(draft_path: Path, reference_path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    if not draft_path.exists():
        raise SystemExit(f"Draft feature table not found: {draft_path}")
    df = pd.read_parquet(draft_path)

    lookup: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}
    for name, champion_id in iter_champion_pairs(df):
        key = normalize_key(name)
        lookup.setdefault(key, champion_id)
        id_to_name.setdefault(champion_id, name)

    if 62 in id_to_name:
        lookup.setdefault("wukong", 62)
    alias_pairs = {
        "belveth": "belveth",
        "chogath": "chogath",
        "drmundo": "drmundo",
        "jarvaniv": "jarvaniv",
        "khazix": "khazix",
        "kogmaw": "kogmaw",
        "ksante": "ksante",
        "leesin": "leesin",
        "masteryi": "masteryi",
        "missfortune": "missfortune",
        "reksai": "reksai",
        "tahmkench": "tahmkench",
        "twistedfate": "twistedfate",
        "velkoz": "velkoz",
        "xinzhao": "xinzhao",
    }
    for alias, canonical in alias_pairs.items():
        if canonical in lookup:
            lookup.setdefault(alias, lookup[canonical])

    if reference_path.exists():
        ref = pd.read_csv(reference_path)
        for name in ref.get("champion_name", pd.Series(dtype=str)).dropna().astype(str):
            key = normalize_key(name)
            if key in lookup:
                id_to_name.setdefault(lookup[key], name)
    return lookup, id_to_name


def resolve_champion(raw: str, lookup: Dict[str, int]) -> int:
    value = raw.strip()
    if not value:
        raise ValueError("Champion name is required.")
    if value.isdigit():
        return int(value)
    key = normalize_key(value)
    if key in lookup:
        return lookup[key]
    suggestions = difflib.get_close_matches(key, sorted(lookup), n=5, cutoff=0.65)
    hint = f" Suggestions: {', '.join(suggestions)}." if suggestions else ""
    raise ValueError(f"Unknown champion: {raw}.{hint}")


def parse_spell_pair(raw: Optional[str], role: str) -> Tuple[Tuple[int, int], bool]:
    if raw is None or not raw.strip():
        return DEFAULT_ROLE_SPELLS[role], True
    tokens = [t for t in re.split(r"[,/|+ ]+", raw.strip()) if t]
    if len(tokens) != 2:
        raise ValueError(f"Expected two summoner spells for {role}, got: {raw}")
    spell_ids: List[int] = []
    for token in tokens:
        if token.isdigit():
            spell_ids.append(int(token))
            continue
        key = normalize_key(token)
        if key not in SPELL_IDS:
            valid = ", ".join(sorted(SPELL_IDS))
            raise ValueError(f"Unknown summoner spell '{token}'. Valid names: {valid}")
        spell_ids.append(SPELL_IDS[key])
    return (spell_ids[0], spell_ids[1]), False


def format_spell_pair(spell_ids: Tuple[int, int]) -> str:
    return ",".join(SPELL_NAMES_BY_ID.get(spell_id, str(spell_id)) for spell_id in spell_ids)


def prompt_value(label: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or (default or "")


def build_manual_row(
    args: argparse.Namespace,
    feature_columns: List[str],
    champion_lookup: Dict[str, int],
    id_to_name: Dict[int, str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    interactive = not args.no_interactive
    row: Dict[str, Any] = {"side": args.side}
    assumptions: List[str] = []
    display: Dict[str, Any] = {"champions": {}, "spells": {}}

    if interactive and not row["side"]:
        row["side"] = prompt_value("Side del equipo aliado (blue/red)", "blue").lower()
    row["side"] = (row["side"] or "blue").lower()
    if row["side"] not in {"blue", "red"}:
        raise SystemExit("--side must be 'blue' or 'red'.")

    for prefix in SIDES:
        display["champions"][prefix] = {}
        display["spells"][prefix] = {}
        side_label = "aliado" if prefix == "ally" else "enemigo"
        for role, role_label in ROLES:
            champ_col = f"{prefix}_{role}_champion_id"
            champ_arg = getattr(args, f"{prefix}_{role}")
            if interactive and champ_col in feature_columns and not champ_arg:
                champ_arg = prompt_value(f"Campeon {side_label} {role_label}")
            if champ_col in feature_columns:
                try:
                    champion_id = resolve_champion(champ_arg or "", champion_lookup)
                except ValueError as exc:
                    raise SystemExit(str(exc)) from exc
                row[champ_col] = champion_id
                display_name = id_to_name.get(champion_id, str(champion_id))
                display["champions"][prefix][role] = display_name

            spell_arg = getattr(args, f"{prefix}_{role}_spells")
            spell1_col = f"{prefix}_{role}_summoner1_id"
            spell2_col = f"{prefix}_{role}_summoner2_id"
            if spell1_col in feature_columns or spell2_col in feature_columns:
                if interactive and spell_arg is None:
                    default = ROLE_SPELL_HINTS[role]
                    spell_arg = prompt_value(f"Hechizos {side_label} {role_label}", default)
                try:
                    (spell1, spell2), used_default = parse_spell_pair(spell_arg, role)
                except ValueError as exc:
                    raise SystemExit(str(exc)) from exc
                row[spell1_col] = spell1
                row[spell2_col] = spell2
                display["spells"][prefix][role] = (spell1, spell2)
                if used_default:
                    assumptions.append(f"{prefix}_{role}_spells={format_spell_pair((spell1, spell2))}")

    for col in feature_columns:
        row.setdefault(col, "__MISSING__")
    return pd.DataFrame([row]), assumptions, display


def load_row_from_match(model_input_path: Path, match_id: str, team_id: Optional[int]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if not model_input_path.exists():
        raise SystemExit(f"Model input table not found: {model_input_path}")
    df = pd.read_parquet(model_input_path)
    subset = df[df["match_id"].astype(str) == str(match_id)].copy()
    if team_id is not None:
        subset = subset[subset["team_id"].astype(int) == int(team_id)].copy()
    if subset.empty:
        raise SystemExit(f"No rows found for match_id={match_id}, team_id={team_id}.")
    if len(subset) > 1:
        teams = ", ".join(str(v) for v in subset["team_id"].tolist())
        raise SystemExit(f"Match has multiple teams ({teams}); pass --team-id 100 or --team-id 200.")
    row = subset.iloc[[0]].copy()
    truth = {}
    if "support_roam_score" in row.columns:
        truth["support_roam_score"] = float(row["support_roam_score"].iloc[0])
    return row, truth


def prediction_percentile(model_dir: Path, score: float) -> Optional[float]:
    path = model_dir / "validation_predictions.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["pred_support_roam_score"])
    vals = df["pred_support_roam_score"].dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return None
    return float((vals <= score).mean() * 100.0)


def percentile_label(percentile: Optional[float], score: float) -> str:
    if percentile is None:
        if score >= 0.50:
            return "alta"
        if score >= 0.35:
            return "moderada-alta"
        if score >= 0.22:
            return "moderada"
        return "baja"
    if percentile >= 90:
        return "muy alta dentro de la distribucion del modelo"
    if percentile >= 75:
        return "alta dentro de la distribucion del modelo"
    if percentile >= 50:
        return "moderada"
    if percentile >= 25:
        return "moderada-baja"
    return "baja"


def load_reference(reference_path: Path) -> Dict[str, Dict[str, Any]]:
    if not reference_path.exists():
        return {}
    ref = pd.read_csv(reference_path)
    out: Dict[str, Dict[str, Any]] = {}
    for _, row in ref.iterrows():
        name = str(row.get("champion_name", ""))
        if name:
            out[normalize_key(name)] = row.to_dict()
    return out


def predict_row(model: SupportMLP, preprocess: Dict[str, Any], row: pd.DataFrame) -> float:
    feature_columns = list(preprocess["feature_columns"])
    for col in feature_columns:
        if col not in row.columns:
            row[col] = "__MISSING__"
    X = preprocess["onehot"].transform(categorical_frame(row, feature_columns))
    with torch.no_grad():
        pred = model(torch.from_numpy(X.astype(np.float32))).detach().cpu().numpy()
    return float(np.clip(pred[0], 0.0, 1.0))


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
    if "team_id" in swapped.columns:
        team_id = str(original.get("team_id"))
        if team_id == "100":
            swapped.at[swapped.index[0], "team_id"] = 200
        elif team_id == "200":
            swapped.at[swapped.index[0], "team_id"] = 100
    return swapped


def swap_display_perspective(display: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "champions": {
            "ally": dict(display.get("champions", {}).get("enemy", {})),
            "enemy": dict(display.get("champions", {}).get("ally", {})),
        },
        "spells": {
            "ally": dict(display.get("spells", {}).get("enemy", {})),
            "enemy": dict(display.get("spells", {}).get("ally", {})),
        },
    }


def make_prediction(
    model: SupportMLP,
    preprocess: Dict[str, Any],
    model_dir: Path,
    row: pd.DataFrame,
) -> Dict[str, Any]:
    score = predict_row(model, preprocess, row)
    percentile = prediction_percentile(model_dir, score)
    return {
        "score": score,
        "validation_percentile": percentile,
        "label": percentile_label(percentile, score),
    }


def print_prediction_block(
    title: str,
    result: Dict[str, Any],
    config: Dict[str, Any],
    display: Dict[str, Any],
    reference: Dict[str, Dict[str, Any]],
    truth: Dict[str, Any],
    support_label: str,
) -> None:
    score = float(result["score"])
    percentile = result.get("validation_percentile")
    label = str(result["label"])

    print(f"\n=== {title} ===")
    print(f"Score estimado: {score:.4f}")
    if percentile is not None:
        print(f"Percentil vs validacion del modelo: {float(percentile):.1f}")
    print(f"Lectura: tendencia {label}.")

    support_name = (
        display.get("champions", {})
        .get("ally", {})
        .get("utility")
    )
    if support_name:
        ref = reference.get(normalize_key(support_name))
        if ref:
            expert = float(ref.get("expert_support_roam_score", math.nan))
            confidence = float(ref.get("expert_confidence", math.nan))
            archetype = ref.get("expert_archetype", "unknown")
            print(
                f"Prior experto del {support_label}: "
                f"{support_name} -> {archetype}, score={expert:.2f}, confianza={confidence:.2f}."
            )
        else:
            print(f"{support_label.capitalize()}: {support_name}. No hay prior experto manual para este campeon.")

    support_cfg = config.get("support_score_config") or {}
    if support_cfg:
        start = support_cfg.get("start_minute", 5)
        end = support_cfg.get("max_minute", 12)
        print(f"Target aprendido: roaming observado entre minuto {start:g} y {end:g}.")
    print("Nota: el modelo actual usa draft pregame; no observa eventos de la partida.")

    if truth:
        true_score = float(truth["support_roam_score"])
        print(f"Etiqueta real de la fila: {true_score:.4f} (error abs={abs(true_score - score):.4f}).")


def print_human_output(
    ally_result: Dict[str, Any],
    enemy_result: Dict[str, Any],
    config: Dict[str, Any],
    display: Dict[str, Any],
    assumptions: List[str],
    reference: Dict[str, Dict[str, Any]],
    truth: Dict[str, Any],
) -> None:
    print_prediction_block(
        title="Prediccion support aliado",
        result=ally_result,
        config=config,
        display=display,
        reference=reference,
        truth=truth,
        support_label="support aliado",
    )
    print_prediction_block(
        title="Prediccion support enemigo",
        result=enemy_result,
        config=config,
        display=swap_display_perspective(display),
        reference=reference,
        truth={},
        support_label="support enemigo",
    )
    if assumptions:
        print("\nSuposiciones usadas: " + "; ".join(assumptions))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict support roaming from champion select data.")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--draft-path", type=Path, default=DEFAULT_DRAFT_PATH)
    p.add_argument("--model-input-path", type=Path, default=DEFAULT_MODEL_INPUT_PATH)
    p.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    p.add_argument("--from-match-id", default=None, help="Use an existing model_input row as input.")
    p.add_argument("--team-id", type=int, choices=[100, 200], default=None)
    p.add_argument("--side", choices=["blue", "red"], default=None)
    p.add_argument("--no-interactive", action="store_true", help="Do not prompt; require args or defaults.")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    p.add_argument("--list-champions", nargs="?", const="", default=None,
                   help="List known champion names, optionally filtered by text.")
    p.add_argument("--list-spells", action="store_true", help="List supported summoner spells and Riot IDs.")

    for prefix in SIDES:
        for role, _ in ROLES:
            p.add_argument(f"--{prefix}-{role}", dest=f"{prefix}_{role}", default=None)
            p.add_argument(f"--{prefix}-{role}-spells", dest=f"{prefix}_{role}_spells", default=None,
                           help="Two spells by name or id, e.g. flash,ignite or 4,14.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model, config, preprocess = load_model(args.model_dir)
    feature_columns = list(preprocess["feature_columns"])
    champion_lookup, id_to_name = build_champion_lookup(args.draft_path, args.reference_path)

    if args.list_spells:
        print("Summoner spells disponibles:")
        for name, spell_id in sorted(SPELL_IDS.items(), key=lambda item: item[1]):
            print(f"- {name}: {spell_id}")
        print("\nPuedes introducirlos por nombre, por id o mezclados. Ejemplos: flash,ignite | 4,14")
        return

    if args.list_champions is not None:
        needle = normalize_key(args.list_champions)
        names = sorted(set(id_to_name.values()))
        for name in names:
            if not needle or needle in normalize_key(name):
                print(name)
        return

    truth: Dict[str, Any] = {}
    assumptions: List[str] = []
    display: Dict[str, Any] = {"champions": {}, "spells": {}}
    if args.from_match_id:
        row, truth = load_row_from_match(args.model_input_path, args.from_match_id, args.team_id)
        for prefix in SIDES:
            display["champions"][prefix] = {}
            for role, _ in ROLES:
                name_col = f"{prefix}_{role}_champion_name"
                if name_col in row.columns:
                    display["champions"][prefix][role] = str(row[name_col].iloc[0])
    else:
        row, assumptions, display = build_manual_row(args, feature_columns, champion_lookup, id_to_name)

    ally_result = make_prediction(model, preprocess, args.model_dir, row)
    enemy_result = make_prediction(model, preprocess, args.model_dir, swap_team_perspective(row))
    reference = load_reference(args.reference_path)

    if args.json:
        payload = {
            "ally_support": ally_result,
            "enemy_support": enemy_result,
            "assumptions": assumptions,
            "truth": truth,
            "model_dir": str(args.model_dir),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print_human_output(ally_result, enemy_result, config, display, assumptions, reference, truth)


if __name__ == "__main__":
    main()
