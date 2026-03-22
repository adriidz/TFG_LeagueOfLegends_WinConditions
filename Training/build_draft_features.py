
#!/usr/bin/env python3
"""
build_draft_features.py

Construye features de draft a nivel (match_id, team_id) a partir de match.json.

Salida:
- Un parquet con una fila por equipo en partida.
- Columnas orientadas a join con etiquetas por:
    (match_id, team_id)

Objetivo:
- Dejar listo un dataset tabular para modelado posterior, por ejemplo
  para predecir jungle_presence_label desde picks, side y patch.

Supuestos v1:
- Se usan solo datos prepartida o estáticos del match.json.
- Los roles se infieren desde info.participants[*].teamPosition.
- Solo se conservan partidas en las que cada equipo tiene exactamente
  un participante por rol canónico:
      TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY
- Los bans se extraen si están disponibles en info.teams[*].bans
  y se guardan como columnas opcionales.

Uso típico:
python build_draft_features.py \
  --raw-root Data_raw/raw \
  --region europe \
  --out-path Data_clean/features/draft_features.parquet
"""

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


CANONICAL_ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
BLUE_TEAM_ID = 100
RED_TEAM_ID = 200
EMPTY_ROLE_VALUES = {None, "", "INVALID", "NONE"}


DRAFT_FEATURE_COLUMN_ORDER = [
    "match_id",
    "team_id",
    "side",
    "patch",
    "game_version",
    "game_start_timestamp",
    "platform_id",
    "queue_id",
    "game_duration_seconds",
    "ally_top_participant_id",
    "ally_jungle_participant_id",
    "ally_middle_participant_id",
    "ally_bottom_participant_id",
    "ally_utility_participant_id",
    "enemy_top_participant_id",
    "enemy_jungle_participant_id",
    "enemy_middle_participant_id",
    "enemy_bottom_participant_id",
    "enemy_utility_participant_id",
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
    "ally_top_champion_name",
    "ally_jungle_champion_name",
    "ally_middle_champion_name",
    "ally_bottom_champion_name",
    "ally_utility_champion_name",
    "enemy_top_champion_name",
    "enemy_jungle_champion_name",
    "enemy_middle_champion_name",
    "enemy_bottom_champion_name",
    "enemy_utility_champion_name",
    "ally_ban_1_champion_id",
    "ally_ban_2_champion_id",
    "ally_ban_3_champion_id",
    "ally_ban_4_champion_id",
    "ally_ban_5_champion_id",
    "enemy_ban_1_champion_id",
    "enemy_ban_2_champion_id",
    "enemy_ban_3_champion_id",
    "enemy_ban_4_champion_id",
    "enemy_ban_5_champion_id",
]


@dataclass
class TeamDraftRow:
    match_id: str
    team_id: int
    side: str
    patch: Optional[str]
    game_version: Optional[str]
    game_start_timestamp: Optional[int]
    platform_id: Optional[str]
    queue_id: Optional[int]
    game_duration_seconds: Optional[float]

    ally_top_participant_id: int
    ally_jungle_participant_id: int
    ally_middle_participant_id: int
    ally_bottom_participant_id: int
    ally_utility_participant_id: int

    enemy_top_participant_id: int
    enemy_jungle_participant_id: int
    enemy_middle_participant_id: int
    enemy_bottom_participant_id: int
    enemy_utility_participant_id: int

    ally_top_champion_id: Optional[int]
    ally_jungle_champion_id: Optional[int]
    ally_middle_champion_id: Optional[int]
    ally_bottom_champion_id: Optional[int]
    ally_utility_champion_id: Optional[int]

    enemy_top_champion_id: Optional[int]
    enemy_jungle_champion_id: Optional[int]
    enemy_middle_champion_id: Optional[int]
    enemy_bottom_champion_id: Optional[int]
    enemy_utility_champion_id: Optional[int]

    ally_top_champion_name: Optional[str]
    ally_jungle_champion_name: Optional[str]
    ally_middle_champion_name: Optional[str]
    ally_bottom_champion_name: Optional[str]
    ally_utility_champion_name: Optional[str]

    enemy_top_champion_name: Optional[str]
    enemy_jungle_champion_name: Optional[str]
    enemy_middle_champion_name: Optional[str]
    enemy_bottom_champion_name: Optional[str]
    enemy_utility_champion_name: Optional[str]

    ally_ban_1_champion_id: Optional[int]
    ally_ban_2_champion_id: Optional[int]
    ally_ban_3_champion_id: Optional[int]
    ally_ban_4_champion_id: Optional[int]
    ally_ban_5_champion_id: Optional[int]

    enemy_ban_1_champion_id: Optional[int]
    enemy_ban_2_champion_id: Optional[int]
    enemy_ban_3_champion_id: Optional[int]
    enemy_ban_4_champion_id: Optional[int]
    enemy_ban_5_champion_id: Optional[int]

    def to_dict(self) -> dict:
        return {col: getattr(self, col) for col in DRAFT_FEATURE_COLUMN_ORDER}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye features de draft por (match_id, team_id)."
    )
    parser.add_argument(
        "--raw-root",
        default=os.path.join("Data_raw", "raw"),
        help="Directorio base que contiene subdirectorios de partida.",
    )
    parser.add_argument(
        "--region",
        default="europe",
        help=(
            "Subdirectorio opcional dentro de --raw-root. "
            "Ejemplo: europe -> Data_raw/raw/europe"
        ),
    )
    parser.add_argument(
        "--out-path",
        default=os.path.join("Data_clean", "features", "draft_features.parquet"),
        help="Ruta de salida .parquet",
    )
    parser.add_argument(
        "--summary-dir",
        default=None,
        help="Directorio opcional para guardar resúmenes CSV/Parquet.",
    )
    parser.add_argument(
        "--min-duration-minutes",
        type=float,
        default=15.0,
        help="Filtra partidas demasiado cortas (default: 15).",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=0,
        help="Límite opcional de partidas para iteración rápida.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Cada cuántas partidas imprimir progreso (default: 1000).",
    )
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def list_match_dirs(base: str) -> List[str]:
    if not os.path.isdir(base):
        raise SystemExit(f"No existe el directorio RAW: {base}")
    out = []
    for name in os.listdir(base):
        mdir = os.path.join(base, name)
        if os.path.isdir(mdir):
            out.append(mdir)
    out.sort()
    return out


def load_match_json(match_dir: str) -> dict:
    path = os.path.join(match_dir, "match.json")
    with open(path, "r", encoding="utf-8") as f:
        import json
        return json.load(f)


def normalize_role(role: Optional[str]) -> Optional[str]:
    if role in EMPTY_ROLE_VALUES:
        return None
    role = str(role).strip().upper()

    if role in {"TOP"}:
        return "TOP"
    if role in {"JUNGLE", "JGL"}:
        return "JUNGLE"
    if role in {"MIDDLE", "MID"}:
        return "MIDDLE"
    if role in {"BOTTOM", "ADC", "BOT"}:
        return "BOTTOM"
    if role in {"UTILITY", "SUPPORT", "SUP"}:
        return "UTILITY"
    return None


def infer_patch(game_version: Optional[str]) -> Optional[str]:
    if not game_version:
        return None
    parts = str(game_version).split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return str(game_version)


def side_from_team_id(team_id: int) -> str:
    if team_id == BLUE_TEAM_ID:
        return "blue"
    if team_id == RED_TEAM_ID:
        return "red"
    return "unknown"


def build_team_role_map(info: dict) -> Dict[int, Dict[str, int]]:
    participants = list(info.get("participants") or [])
    team_role_map: Dict[int, Dict[str, int]] = {}

    for team_id in (BLUE_TEAM_ID, RED_TEAM_ID):
        team_participants = [p for p in participants if p.get("teamId") == team_id]
        role_to_pid: Dict[str, List[int]] = {role: [] for role in CANONICAL_ROLES}

        for p in team_participants:
            role = normalize_role(p.get("teamPosition"))
            pid = p.get("participantId")
            if role in role_to_pid and isinstance(pid, int):
                role_to_pid[role].append(pid)

        if all(len(role_to_pid[role]) == 1 for role in CANONICAL_ROLES):
            team_role_map[team_id] = {
                role: role_to_pid[role][0]
                for role in CANONICAL_ROLES
            }

    return team_role_map


def build_participant_lookup(info: dict) -> Dict[int, dict]:
    participants = list(info.get("participants") or [])
    lookup = {}
    for p in participants:
        pid = p.get("participantId")
        if isinstance(pid, int):
            lookup[pid] = p
    return lookup


def extract_team_bans(info: dict) -> Dict[int, List[Optional[int]]]:
    teams = list(info.get("teams") or [])
    out: Dict[int, List[Optional[int]]] = {
        BLUE_TEAM_ID: [None] * 5,
        RED_TEAM_ID: [None] * 5,
    }

    for team in teams:
        team_id = team.get("teamId")
        if team_id not in out:
            continue

        bans_raw = list(team.get("bans") or [])
        bans: List[Optional[int]] = []
        for b in bans_raw[:5]:
            champ_id = b.get("championId")
            bans.append(champ_id if isinstance(champ_id, int) and champ_id > 0 else None)

        while len(bans) < 5:
            bans.append(None)

        out[team_id] = bans

    return out


def safe_game_duration_seconds(info: dict) -> Optional[float]:
    # Riot puede devolver gameDuration en segundos o milisegundos según endpoint/época.
    gd = info.get("gameDuration")
    if gd is None:
        return None
    try:
        value = float(gd)
    except Exception:
        return None

    # Heurística simple: si es muy grande, probablemente esté en ms.
    if value > 100000:
        return value / 1000.0
    return value


def extract_team_row(
    info: dict,
    match_id: str,
    team_id: int,
    team_role_map: Dict[int, Dict[str, int]],
    participant_lookup: Dict[int, dict],
    bans_by_team: Dict[int, List[Optional[int]]],
) -> TeamDraftRow:
    enemy_team_id = RED_TEAM_ID if team_id == BLUE_TEAM_ID else BLUE_TEAM_ID

    own_roles = team_role_map[team_id]
    enemy_roles = team_role_map[enemy_team_id]

    def champ_id(pid: int) -> Optional[int]:
        p = participant_lookup.get(pid, {})
        value = p.get("championId")
        return value if isinstance(value, int) else None

    def champ_name(pid: int) -> Optional[str]:
        p = participant_lookup.get(pid, {})
        value = p.get("championName")
        if value is None:
            return None
        return str(value)

    game_version = info.get("gameVersion")
    patch = infer_patch(game_version)
    game_start_timestamp = info.get("gameStartTimestamp")
    platform_id = info.get("platformId")
    queue_id = info.get("queueId")
    game_duration_seconds = safe_game_duration_seconds(info)

    own_bans = bans_by_team.get(team_id, [None] * 5)
    enemy_bans = bans_by_team.get(enemy_team_id, [None] * 5)

    return TeamDraftRow(
        match_id=match_id,
        team_id=team_id,
        side=side_from_team_id(team_id),
        patch=patch,
        game_version=game_version,
        game_start_timestamp=game_start_timestamp if isinstance(game_start_timestamp, int) else None,
        platform_id=str(platform_id) if platform_id is not None else None,
        queue_id=queue_id if isinstance(queue_id, int) else None,
        game_duration_seconds=game_duration_seconds,

        ally_top_participant_id=own_roles["TOP"],
        ally_jungle_participant_id=own_roles["JUNGLE"],
        ally_middle_participant_id=own_roles["MIDDLE"],
        ally_bottom_participant_id=own_roles["BOTTOM"],
        ally_utility_participant_id=own_roles["UTILITY"],

        enemy_top_participant_id=enemy_roles["TOP"],
        enemy_jungle_participant_id=enemy_roles["JUNGLE"],
        enemy_middle_participant_id=enemy_roles["MIDDLE"],
        enemy_bottom_participant_id=enemy_roles["BOTTOM"],
        enemy_utility_participant_id=enemy_roles["UTILITY"],

        ally_top_champion_id=champ_id(own_roles["TOP"]),
        ally_jungle_champion_id=champ_id(own_roles["JUNGLE"]),
        ally_middle_champion_id=champ_id(own_roles["MIDDLE"]),
        ally_bottom_champion_id=champ_id(own_roles["BOTTOM"]),
        ally_utility_champion_id=champ_id(own_roles["UTILITY"]),

        enemy_top_champion_id=champ_id(enemy_roles["TOP"]),
        enemy_jungle_champion_id=champ_id(enemy_roles["JUNGLE"]),
        enemy_middle_champion_id=champ_id(enemy_roles["MIDDLE"]),
        enemy_bottom_champion_id=champ_id(enemy_roles["BOTTOM"]),
        enemy_utility_champion_id=champ_id(enemy_roles["UTILITY"]),

        ally_top_champion_name=champ_name(own_roles["TOP"]),
        ally_jungle_champion_name=champ_name(own_roles["JUNGLE"]),
        ally_middle_champion_name=champ_name(own_roles["MIDDLE"]),
        ally_bottom_champion_name=champ_name(own_roles["BOTTOM"]),
        ally_utility_champion_name=champ_name(own_roles["UTILITY"]),

        enemy_top_champion_name=champ_name(enemy_roles["TOP"]),
        enemy_jungle_champion_name=champ_name(enemy_roles["JUNGLE"]),
        enemy_middle_champion_name=champ_name(enemy_roles["MIDDLE"]),
        enemy_bottom_champion_name=champ_name(enemy_roles["BOTTOM"]),
        enemy_utility_champion_name=champ_name(enemy_roles["UTILITY"]),

        ally_ban_1_champion_id=own_bans[0],
        ally_ban_2_champion_id=own_bans[1],
        ally_ban_3_champion_id=own_bans[2],
        ally_ban_4_champion_id=own_bans[3],
        ally_ban_5_champion_id=own_bans[4],

        enemy_ban_1_champion_id=enemy_bans[0],
        enemy_ban_2_champion_id=enemy_bans[1],
        enemy_ban_3_champion_id=enemy_bans[2],
        enemy_ban_4_champion_id=enemy_bans[3],
        enemy_ban_5_champion_id=enemy_bans[4],
    )


def build_summary_tables(df: pd.DataFrame, summary_dir: str) -> None:
    ensure_dir(summary_dir)

    overall = pd.DataFrame(
        [
            {
                "rows": len(df),
                "matches": int(df["match_id"].nunique()) if "match_id" in df.columns else None,
                "teams": int(df["team_id"].nunique()) if "team_id" in df.columns else None,
                "patches": int(df["patch"].nunique(dropna=True)) if "patch" in df.columns else None,
                "queue_ids": int(df["queue_id"].nunique(dropna=True)) if "queue_id" in df.columns else None,
            }
        ]
    )

    overall.to_parquet(Path(summary_dir) / "overall_summary.parquet", index=False)
    overall.to_csv(Path(summary_dir) / "overall_summary.csv", index=False)

    if "patch" in df.columns:
        patch_counts = (
            df.groupby("patch", dropna=False)
            .size()
            .reset_index(name="n_rows")
            .sort_values("n_rows", ascending=False)
        )
        patch_counts.to_parquet(Path(summary_dir) / "patch_counts.parquet", index=False)
        patch_counts.to_csv(Path(summary_dir) / "patch_counts.csv", index=False)

    if "side" in df.columns:
        side_counts = (
            df.groupby("side", dropna=False)
            .size()
            .reset_index(name="n_rows")
            .sort_values("n_rows", ascending=False)
        )
        side_counts.to_parquet(Path(summary_dir) / "side_counts.parquet", index=False)
        side_counts.to_csv(Path(summary_dir) / "side_counts.csv", index=False)

    role_cols = [c for c in df.columns if c.endswith("_champion_name")]
    champion_long = []
    for col in role_cols:
        role_name = col.replace("_champion_name", "")
        tmp = (
            df[[col]]
            .rename(columns={col: "champion_name"})
            .assign(role_slot=role_name)
        )
        champion_long.append(tmp)

    if champion_long:
        champion_df = pd.concat(champion_long, ignore_index=True)
        champion_counts = (
            champion_df.groupby(["role_slot", "champion_name"], dropna=False)
            .size()
            .reset_index(name="n")
            .sort_values(["role_slot", "n"], ascending=[True, False])
        )
        champion_counts.to_parquet(Path(summary_dir) / "champion_counts_by_role.parquet", index=False)
        champion_counts.to_csv(Path(summary_dir) / "champion_counts_by_role.csv", index=False)


def main() -> None:
    args = parse_args()

    raw_base = args.raw_root
    if args.region:
        raw_base = os.path.join(raw_base, args.region)

    out_path = Path(args.out_path)
    ensure_dir(str(out_path.parent))

    summary_dir = args.summary_dir
    if summary_dir is None:
        summary_dir = str(out_path.with_suffix("")) + "_analysis"

    print(f"Leyendo partidas desde: {raw_base}")
    match_dirs = list_match_dirs(raw_base)
    print(f"Directorios de partida detectados: {len(match_dirs)}")
    if args.max_matches and args.max_matches > 0:
        print(f"Directorios de partida a procesar: {len(match_dirs)}")
        match_dirs = match_dirs[: args.max_matches]

    
    rows: List[dict] = []

    total_seen = 0
    total_kept_matches = 0
    bad_match_json = 0
    filtered_short = 0
    filtered_roles = 0
    build_errors = 0

    t0 = time.time()
    last_log = t0

    for mdir in match_dirs:
        total_seen += 1

        if total_seen % args.progress_every == 0 or (time.time() - last_log) > 15:
            now = time.time()
            elapsed = now - t0
            rate = total_seen / elapsed if elapsed > 0 else 0.0
            pct = 100.0 * total_seen / len(match_dirs) if match_dirs else 0.0
            print(
                f"[{total_seen}/{len(match_dirs)} | {pct:.1f}%] "
                f"kept_matches={total_kept_matches} "
                f"rows={len(rows)} "
                f"bad_match={bad_match_json} "
                f"short={filtered_short} "
                f"roles={filtered_roles} "
                f"errors={build_errors} "
                f"rate={rate:.1f} matches/s"
            )
            last_log = now

        try:
            match_data = load_match_json(mdir)
        except Exception:
            bad_match_json += 1
            continue

        metadata = match_data.get("metadata") or {}
        info = match_data.get("info") or {}
        match_id = metadata.get("matchId") or os.path.basename(mdir)

        duration_seconds = safe_game_duration_seconds(info)
        if duration_seconds is not None and duration_seconds < args.min_duration_minutes * 60.0:
            filtered_short += 1
            continue

        team_role_map = build_team_role_map(info)
        if BLUE_TEAM_ID not in team_role_map or RED_TEAM_ID not in team_role_map:
            filtered_roles += 1
            continue

        participant_lookup = build_participant_lookup(info)
        bans_by_team = extract_team_bans(info)

        try:
            for team_id in (BLUE_TEAM_ID, RED_TEAM_ID):
                row = extract_team_row(
                    info=info,
                    match_id=str(match_id),
                    team_id=team_id,
                    team_role_map=team_role_map,
                    participant_lookup=participant_lookup,
                    bans_by_team=bans_by_team,
                )
                rows.append(row.to_dict())
            total_kept_matches += 1
        except Exception:
            build_errors += 1
            continue

    if not rows:
        raise SystemExit("No se generó ninguna fila. Revisa filtros y estructura de datos.")

    df = pd.DataFrame(rows)

    # Garantiza orden estable de columnas.
    for col in DRAFT_FEATURE_COLUMN_ORDER:
        if col not in df.columns:
            df[col] = None
    df = df[DRAFT_FEATURE_COLUMN_ORDER].copy()

    print(f"\nFilas finales: {len(df)}")
    print(f"Partidas válidas: {total_kept_matches}")
    print(f"Guardando parquet en: {out_path}")
    df.to_parquet(out_path, index=False)

    print(f"Guardando resúmenes en: {summary_dir}")
    build_summary_tables(df, summary_dir)

    print("\nHecho.")
    print(f"- parquet principal: {out_path}")
    print(f"- analysis dir: {summary_dir}")


if __name__ == "__main__":
    main()
