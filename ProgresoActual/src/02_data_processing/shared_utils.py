#!/usr/bin/env python3
"""
shared_utils.py

Módulo común para los scripts de procesamiento de datos del TFG.
Contiene: geometría canónica del mapa, parsing de JSON de Riot,
funciones de participantes/roles, clasificación de zonas,
y utilidades de I/O.

Geometría v2: calibrada contra posiciones reales de torres (Riot API)
y datos empíricos de 500 partidas (spawn ranges, coordinate bounds).
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from matplotlib.path import Path as MplPath

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════════════

CANONICAL_ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
ROLE_KEYS_LOWER = ("top", "jungle", "middle", "bottom", "utility")
BLUE_TEAM_ID = 100
RED_TEAM_ID = 200
MAP_MAX = 14800.0           # Validado empíricamente: datos reales van de ~60 a ~14700
MAP_CENTER_SUM = MAP_MAX    # Anti-diagonal: x+y < MAP_CENTER_SUM → blue-side
DEFAULT_MIN_DURATION_MINUTES = 15.0
DEFAULT_MAX_MINUTE = 14.0
EMPTY_ROLE_VALUES = {None, "", "INVALID", "NONE"}

# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRÍA CANÓNICA DEL MAPA (perspectiva blue-side)
#
# Calibración basada en:
#   - Posiciones de torres T1/T2/T3 (Match-v5 API)
#   - Spawns reales (empírico, 500 partidas):
#       Blue spawn: x=[130,662], y=[135,675]
#       Red spawn:  x=[14055,14589], y=[14170,14673]
#   - Coordenadas de objetivos conocidos (Baron, Dragon)
#
# Sistema: (0,0) = esquina inferior-izquierda (blue fountain)
#          (14800,14800) = esquina superior-derecha (red fountain)
#          Diagonal principal: y = x
#          Anti-diagonal: x + y = 14800
# ═══════════════════════════════════════════════════════════════════════════════

# ── Torres de referencia ────────────────────────────────────────────────────
# (se usan para calibrar los centerlines de las lanes)
# Blue side
_BLUE_TOP_T3 = (1169, 4287)
_BLUE_TOP_T2 = (1512, 6699)
_BLUE_TOP_T1 = (981, 10441)
_BLUE_MID_T3 = (3651, 3696)
_BLUE_MID_T2 = (5048, 4812)
_BLUE_MID_T1 = (5846, 6396)
_BLUE_BOT_T3 = (4281, 1253)
_BLUE_BOT_T2 = (6919, 1483)
_BLUE_BOT_T1 = (10504, 1029)
# Red side
_RED_TOP_T1 = (4318, 13875)
_RED_TOP_T2 = (7943, 13411)
_RED_TOP_T3 = (10481, 13650)
_RED_MID_T1 = (8955, 8510)
_RED_MID_T2 = (9767, 10113)
_RED_MID_T3 = (11134, 11207)
_RED_BOT_T1 = (13866, 4505)
_RED_BOT_T2 = (13327, 8226)
_RED_BOT_T3 = (13624, 10572)

# ── Polígonos de base ──────────────────────────────────────────────────────
# Envuelven spawn + nexus + inhibidores + torres T3
# Calibrados contra spawns reales y posiciones de T3/nexus turrets
BLUE_BASE_POLYGON = [
    (0, 0), (0, 4500), (1300, 4500),
    (3800, 3800), (4500, 1300), (4500, 0),
]
RED_BASE_POLYGON = [
    (MAP_MAX, MAP_MAX), (MAP_MAX, MAP_MAX - 4500),
    (MAP_MAX - 1300, MAP_MAX - 4500), (MAP_MAX - 3800, MAP_MAX - 3800),
    (MAP_MAX - 4500, MAP_MAX - 1300), (MAP_MAX - 4500, MAP_MAX),
]

# ── Centerlines de las lanes (calibrados por torres) ───────────────────────
# Top lane: sube por la pared izquierda, luego cruza por arriba
TOP_LANE_CENTERLINE = [
    (1200, 4500),    # salida de blue base (cerca T3 top)
    (1000, 6700),    # T2 top blue
    (980, 10450),    # T1 top blue
    (1500, 12500),   # curva de la esquina
    (2500, 13600),   # esquina top-left
    (4300, 13900),   # T1 top red
    (7950, 13400),   # T2 top red
    (10500, 13650),  # T3 top red
]

# Bot lane: cruza por abajo, luego sube por la pared derecha
BOT_LANE_CENTERLINE = [
    (4500, 1200),    # salida de blue base (cerca T3 bot)
    (6700, 1000),    # T2 bot blue
    (10450, 980),    # T1 bot blue
    (12500, 1500),   # curva de la esquina
    (13600, 2500),   # esquina bot-right
    (13900, 4300),   # T1 bot red
    (13400, 7950),   # T2 bot red (sí, la x decrece ligeramente)
    (13650, 10500),  # T3 bot red
]

# Mid lane: diagonal central recta. Aún más corta para evitar
# cualquier contacto con las salidas de base.
MID_LANE_CENTERLINE = [
    (5800, 5800),
    (7400, 7400),
    (9000, 9000),
]

# ── Río (anti-diagonal) ───────────────────────────────────────────────────
RIVER_CENTERLINE = [
    (2500, 12000),   # entrada top-left
    (3500, 11000),   # acercándose a Baron
    (4400, 10450),   # Baron pit center
    (6000, 8800),    # entre Baron y centro
    (7400, 7400),    # centro del mapa (cruce con mid)
    (8800, 6000),    # entre centro y Dragon
    (10450, 4400),   # Dragon pit center
    (11000, 3500),   # acercándose a bot-right
    (12000, 2500),   # entrada bot-right
]

BARON_PIT_CENTER = (4400.0, 10450.0)
DRAGON_PIT_CENTER = (10450.0, 4400.0)

# ── Anchos de zonas (distancia máxima al centerline para pertenecer) ──────
TOP_LANE_WIDTH = 850.0
BOTTOM_LANE_WIDTH = 850.0
MID_LANE_WIDTH = 750.0  # Reducido ligeramente
RIVER_WIDTH = 950.0
PIT_RADIUS = 0.0  # Eliminamos las 'bolas' que invadían la jungla

# ── Conjuntos de zonas (agrupaciones semánticas) ─────────────────────────
LANE_ZONES = {"TOP_LANE", "MID_LANE", "BOTTOM_LANE"}
OWN_JUNGLE_ZONES = {"OWN_TOP_JUNGLE", "OWN_BOTTOM_JUNGLE"}
ENEMY_JUNGLE_ZONES = {"ENEMY_TOP_JUNGLE", "ENEMY_BOTTOM_JUNGLE"}
BASE_ZONES = {"OWN_BASE", "ENEMY_BASE"}
RIVER_ZONES = {"RIVER_TOP", "RIVER_BOT"}

# Bot-side para support: lane bot + jungla inferior propia
# NO incluye OWN_BASE (jugador inactivo) ni RIVER_BOT (rotación)
BOT_SIDE_ZONES = {"BOTTOM_LANE", "OWN_BOTTOM_JUNGLE"}


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE GEOMETRÍA
# ═══════════════════════════════════════════════════════════════════════════════

def point_in_polygon(x: float, y: float, polygon: Sequence[Tuple[float, float]]) -> bool:
    return MplPath(polygon).contains_point((x, y), radius=1e-9)


def is_near_point(x: float, y: float, center: Tuple[float, float], radius: float) -> bool:
    cx, cy = center
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2


def point_to_segment_distance(px: float, py: float,
                              ax: float, ay: float,
                              bx: float, by: float) -> float:
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom <= 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
    return math.hypot(px - (ax + t * abx), py - (ay + t * aby))


def point_to_polyline_distance(x: float, y: float,
                               polyline: Sequence[Tuple[float, float]]) -> float:
    best = float("inf")
    for (ax, ay), (bx, by) in zip(polyline[:-1], polyline[1:]):
        best = min(best, point_to_segment_distance(x, y, ax, ay, bx, by))
    return best


def is_in_top_lane_canonical(x: float, y: float) -> bool:
    # Top lane está en la mitad superior del mapa (y > x) o pegada al borde izquierdo/superior
    # Filtro rápido: excluir puntos claramente fuera
    if y < 3500.0 and x > 3500.0:
        return False
    return point_to_polyline_distance(x, y, TOP_LANE_CENTERLINE) <= TOP_LANE_WIDTH


def is_in_bottom_lane_canonical(x: float, y: float) -> bool:
    # Bot lane está en la mitad inferior del mapa (x > y) o pegada al borde derecho/inferior
    if x < 3500.0 and y > 3500.0:
        return False
    return point_to_polyline_distance(x, y, BOT_LANE_CENTERLINE) <= BOTTOM_LANE_WIDTH


def is_in_mid_lane_canonical(x: float, y: float) -> bool:
    # Mid lane es la diagonal central; filtro: solo en la franja central del mapa
    if x < 2800.0 or y < 2800.0 or x > 12000.0 or y > 12000.0:
        return False
    return point_to_polyline_distance(x, y, MID_LANE_CENTERLINE) <= MID_LANE_WIDTH


def is_in_river_canonical(x: float, y: float) -> bool:
    """Comprueba si un punto está en el río (incluyendo fosos de objetivos)."""
    if is_near_point(x, y, BARON_PIT_CENTER, PIT_RADIUS):
        return True
    if is_near_point(x, y, DRAGON_PIT_CENTER, PIT_RADIUS):
        return True
    return point_to_polyline_distance(x, y, RIVER_CENTERLINE) <= RIVER_WIDTH


# ═══════════════════════════════════════════════════════════════════════════════
# CLASIFICACIÓN DE ZONAS
#
# Prioridad de evaluación:
#   1. Bases (polígonos cerrados)
#   2. Lanes (distancia a centerline < width)
#   3. Río (distancia a centerline o cerca de objetivo)
#   4. Jungla (fallback: cuadrante por diagonales)
# ═══════════════════════════════════════════════════════════════════════════════

def is_in_base_canonical(x: float, y: float) -> Optional[str]:
    if point_in_polygon(x, y, BLUE_BASE_POLYGON):
        return "OWN_BASE"
    if point_in_polygon(x, y, RED_BASE_POLYGON):
        return "ENEMY_BASE"
    return None


def is_in_top_lane_canonical(x: float, y: float) -> bool:
    # Top lane está en la mitad superior del mapa (y > x) o pegada al borde izquierdo/superior
    # Filtro rápido: excluir puntos claramente fuera
    if y < 3500.0 and x > 3500.0:
        return False
    return point_to_polyline_distance(x, y, TOP_LANE_CENTERLINE) <= TOP_LANE_WIDTH


def is_in_bottom_lane_canonical(x: float, y: float) -> bool:
    # Bot lane está en la mitad inferior del mapa (x > y) o pegada al borde derecho/inferior
    if x < 3500.0 and y > 3500.0:
        return False
    return point_to_polyline_distance(x, y, BOT_LANE_CENTERLINE) <= BOTTOM_LANE_WIDTH


def is_in_mid_lane_canonical(x: float, y: float) -> bool:
    # Mid lane es la diagonal central; filtro: solo en la franja central del mapa
    if x < 2800.0 or y < 2800.0 or x > 12000.0 or y > 12000.0:
        return False
    return point_to_polyline_distance(x, y, MID_LANE_CENTERLINE) <= MID_LANE_WIDTH


def is_in_river_canonical(x: float, y: float) -> bool:
    """Comprueba si un punto está en el río (incluyendo fosos de objetivos)."""
    if is_near_point(x, y, BARON_PIT_CENTER, PIT_RADIUS):
        return True
    if is_near_point(x, y, DRAGON_PIT_CENTER, PIT_RADIUS):
        return True
    return point_to_polyline_distance(x, y, RIVER_CENTERLINE) <= RIVER_WIDTH


def get_team_relative_zone(x: float, y: float, team_id: int) -> str:
    """Clasifica un punto directamente en una zona relativa al equipo.
    
    Sin rotaciones de coordenadas, para preservar que TOP siempre es top-left
    y BOT siempre es bot-right.

    Retorna una de:
        OWN_BASE, ENEMY_BASE,
        TOP_LANE, MID_LANE, BOTTOM_LANE,
        RIVER_TOP, RIVER_BOT,
        OWN_TOP_JUNGLE, OWN_BOTTOM_JUNGLE,
        ENEMY_TOP_JUNGLE, ENEMY_BOTTOM_JUNGLE
    """
    # 1. Bases
    if point_in_polygon(x, y, BLUE_BASE_POLYGON):
        return "OWN_BASE" if team_id == BLUE_TEAM_ID else "ENEMY_BASE"
    if point_in_polygon(x, y, RED_BASE_POLYGON):
        return "OWN_BASE" if team_id == RED_TEAM_ID else "ENEMY_BASE"

    # 2. Lanes
    if is_in_top_lane_canonical(x, y):
        return "TOP_LANE"
    if is_in_bottom_lane_canonical(x, y):
        return "BOTTOM_LANE"
    if is_in_mid_lane_canonical(x, y):
        return "MID_LANE"

    # 3. Río
    if is_in_river_canonical(x, y):
        return "RIVER_TOP" if y >= x else "RIVER_BOT"

    # 4. Jungla (fallback: cuadrante por diagonales)
    blue_side = (x + y) < MAP_CENTER_SUM
    own_or_enemy = "OWN" if (team_id == BLUE_TEAM_ID) == blue_side else "ENEMY"
    top_or_bottom = "TOP" if y >= x else "BOTTOM"
    
    return f"{own_or_enemy}_{top_or_bottom}_JUNGLE"


def classify_map_zone(x: float, y: float, team_id: int) -> str:
    """Alias para la clasificación de zonas."""
    return get_team_relative_zone(x, y, team_id)


def classify_team_side(x: float, y: float, team_id: int) -> str:
    """Clasifica si una posición es TOP/BOT/NONE para la métrica de team tendency.

    El río se asigna al lado correspondiente:
        RIVER_TOP → TOP, RIVER_BOT → BOT
    MID_LANE y bases → NONE (no aportan señal lateral).
    """
    zone = get_team_relative_zone(x, y, team_id)
    if zone in {"TOP_LANE", "OWN_TOP_JUNGLE", "ENEMY_TOP_JUNGLE", "RIVER_TOP"}:
        return "TOP"
    if zone in {"BOTTOM_LANE", "OWN_BOTTOM_JUNGLE", "ENEMY_BOTTOM_JUNGLE", "RIVER_BOT"}:
        return "BOT"
    return "NONE"


# ═══════════════════════════════════════════════════════════════════════════════
# PARSING DE JSON Y MATCH DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_match_info(match: dict) -> dict:
    return (match or {}).get("info") or {}


def get_match_id(match: dict, match_dir: str) -> str:
    metadata = (match or {}).get("metadata") or {}
    if metadata.get("matchId"):
        return str(metadata["matchId"])
    info = get_match_info(match)
    if info.get("gameId") is not None:
        return str(info["gameId"])
    return os.path.basename(match_dir)


def get_timeline_frames(timeline: dict) -> List[dict]:
    info = (timeline or {}).get("info")
    if isinstance(info, dict) and isinstance(info.get("frames"), list):
        return info["frames"]
    frames = (timeline or {}).get("frames")
    return frames if isinstance(frames, list) else []


def game_duration_minutes(info: dict) -> Optional[float]:
    raw = info.get("gameDuration")
    if raw is not None:
        try:
            value = float(raw)
            if value > 100000:
                value /= 1000.0
            return value / 60.0
        except Exception:
            pass
    game_creation = info.get("gameCreation")
    game_end = info.get("gameEndTimestamp")
    try:
        if game_creation is not None and game_end is not None:
            return max(0.0, (float(game_end) - float(game_creation)) / 1000.0 / 60.0)
    except Exception:
        pass
    return None


def safe_game_duration_seconds(info: dict) -> Optional[float]:
    gd = info.get("gameDuration")
    if gd is None:
        return None
    try:
        value = float(gd)
    except Exception:
        return None
    return value / 1000.0 if value > 100000 else value


def infer_patch(game_version: Optional[str]) -> Optional[str]:
    if not game_version:
        return None
    parts = str(game_version).split(".")
    return f"{parts[0]}.{parts[1]}" if len(parts) >= 2 else str(game_version)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTICIPANTES Y ROLES
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_role(value: object) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().upper()
    if s in EMPTY_ROLE_VALUES:
        return None
    ALIASES = {
        "TOP": "TOP", "JUNGLE": "JUNGLE", "JGL": "JUNGLE",
        "MIDDLE": "MIDDLE", "MID": "MIDDLE",
        "BOTTOM": "BOTTOM", "ADC": "BOTTOM", "BOT": "BOTTOM",
        "UTILITY": "UTILITY", "SUPPORT": "UTILITY", "SUP": "UTILITY",
    }
    return ALIASES.get(s, s if s in CANONICAL_ROLES else None)


def extract_team_role_map(info: dict) -> Dict[int, Dict[str, int]]:
    """Extrae {team_id: {role: participant_id}} para ambos equipos.

    Solo retorna un equipo si todos los 5 roles están asignados sin ambigüedad.
    """
    participants = list(info.get("participants") or [])
    team_role_map: Dict[int, Dict[str, int]] = {}
    for team_id in (BLUE_TEAM_ID, RED_TEAM_ID):
        team_p = [p for p in participants if p.get("teamId") == team_id]
        role_to_pid: Dict[str, List[int]] = {role: [] for role in CANONICAL_ROLES}
        for p in team_p:
            role = normalize_role(p.get("teamPosition"))
            pid = p.get("participantId")
            if role in role_to_pid and isinstance(pid, int):
                role_to_pid[role].append(pid)
        if all(len(role_to_pid[r]) == 1 for r in CANONICAL_ROLES):
            team_role_map[team_id] = {r: role_to_pid[r][0] for r in CANONICAL_ROLES}
    return team_role_map


def participant_lookup(info: dict) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for p in info.get("participants") or []:
        pid = p.get("participantId")
        if isinstance(pid, int):
            out[pid] = p
    return out


def get_participant_frame(frame: dict, participant_id: int) -> Optional[dict]:
    pf = frame.get("participantFrames") or {}
    if isinstance(pf, dict):
        return pf.get(str(participant_id)) or pf.get(participant_id)
    return None


def extract_position(pf: Optional[dict]) -> Optional[Tuple[float, float]]:
    if not isinstance(pf, dict):
        return None
    pos = pf.get("position") or {}
    x, y = pos.get("x"), pos.get("y")
    try:
        return None if x is None or y is None else (float(x), float(y))
    except Exception:
        return None


def participant_is_alive(pf: Optional[dict]) -> bool:
    if not isinstance(pf, dict):
        return False
    for key in ("isAlive", "alive"):
        if key in pf:
            try:
                return bool(pf[key])
            except Exception:
                pass
    stats = pf.get("championStats") or {}
    for hp_key in ("currentHealth", "health"):
        if hp_key in stats:
            try:
                return float(stats[hp_key]) > 0.0
            except Exception:
                pass
    pos = pf.get("position") or {}
    try:
        return not (float(pos.get("x")) <= 1.0 and float(pos.get("y")) <= 1.0)
    except Exception:
        return False


def extract_summoner_spells(participant: dict) -> Tuple[Optional[int], Optional[int]]:
    s1 = participant.get("summoner1Id")
    s2 = participant.get("summoner2Id")
    return (
        s1 if isinstance(s1, int) else None,
        s2 if isinstance(s2, int) else None,
    )


def extract_runes(participant: dict) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Extrae (keystone_id, primary_style_id, sub_style_id) de un participante."""
    perks = participant.get("perks") or {}
    styles = perks.get("styles") or []
    keystone, primary_style, sub_style = None, None, None
    if len(styles) >= 1:
        primary_style = styles[0].get("style")
        selections = styles[0].get("selections") or []
        if selections:
            keystone = selections[0].get("perk")
    if len(styles) >= 2:
        sub_style = styles[1].get("style")
    return (
        keystone if isinstance(keystone, int) else None,
        primary_style if isinstance(primary_style, int) else None,
        sub_style if isinstance(sub_style, int) else None,
    )


def extract_team_bans(info: dict) -> Dict[int, List[Optional[int]]]:
    teams = list(info.get("teams") or [])
    out: Dict[int, List[Optional[int]]] = {BLUE_TEAM_ID: [None] * 5, RED_TEAM_ID: [None] * 5}
    for team in teams:
        tid = team.get("teamId")
        if tid not in out:
            continue
        bans_raw = list(team.get("bans") or [])[:5]
        bans = []
        for b in bans_raw:
            cid = b.get("championId")
            bans.append(cid if isinstance(cid, int) and cid > 0 else None)
        while len(bans) < 5:
            bans.append(None)
        out[tid] = bans
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESAMIENTO DE FRAMES
# ═══════════════════════════════════════════════════════════════════════════════

def frames_in_window(frames: Iterable[dict], max_minute: float) -> List[dict]:
    """Filtra frames dentro de la ventana de análisis.

    Excluye el frame 0 (timestamp=0, pre-minion spawn) ya que los jugadores
    están en base/fountain y no aportan información posicional útil.
    """
    max_ts = max_minute * 60.0 * 1000.0
    kept: List[dict] = []
    for frame in frames:
        ts = frame.get("timestamp")
        try:
            ts_value = float(ts) if ts is not None else None
        except Exception:
            ts_value = None
        if ts_value is not None and 0.0 < ts_value <= max_ts:
            kept.append(frame)
        elif ts_value is None:
            kept.append(frame)
    return kept


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE I/O Y MUESTREO
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def apply_sample_suffix(path: Optional[str], frac: Optional[float]) -> Optional[str]:
    if path is None or frac is None or frac >= 1.0 or frac <= 0.0:
        return path
    base, ext = os.path.splitext(path)
    if ext == "":
        return f"{base}_sample{int(frac * 100)}"
    return f"{base}_sample{int(frac * 100)}{ext}"


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


def list_match_dirs(base: str) -> List[str]:
    if not os.path.isdir(base):
        raise SystemExit(f"No existe el directorio RAW: {base}")
    out = [os.path.join(base, n) for n in os.listdir(base) if os.path.isdir(os.path.join(base, n))]
    out.sort()
    return out


def validate_no_duplicate_keys(df: pd.DataFrame) -> None:
    if df.empty:
        return
    dup = df.duplicated(subset=["match_id", "team_id"], keep=False)
    if dup.any():
        preview = df.loc[dup, ["match_id", "team_id"]].head(10)
        raise SystemExit(
            "Claves duplicadas en (match_id, team_id).\n" + preview.to_string(index=False)
        )


def save_dataframe(df: pd.DataFrame, path_no_ext: str) -> None:
    ensure_parent_dir(path_no_ext + ".csv")
    df.to_csv(path_no_ext + ".csv", index=False)
    try:
        df.to_parquet(path_no_ext + ".parquet", index=False)
    except Exception:
        pass


def side_from_team_id(team_id: int) -> str:
    if team_id == BLUE_TEAM_ID:
        return "blue"
    if team_id == RED_TEAM_ID:
        return "red"
    return "unknown"
