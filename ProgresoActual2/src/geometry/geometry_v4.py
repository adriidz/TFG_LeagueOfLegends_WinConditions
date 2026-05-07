#!/usr/bin/env python3
"""
Geometry v4 for support roaming analysis.

This module is intentionally isolated under ProgresoActual2. It reuses the
current v2 constants as a baseline, but clips classification with an observed
walkable mask derived from all-player timeline heatmaps.
"""

from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_UTILS_DIR = REPO_ROOT / "ProgresoActual" / "src" / "02_data_processing"
sys.path.insert(0, str(SHARED_UTILS_DIR))

from shared_utils import (  # noqa: E402
    BLUE_BASE_POLYGON,
    BLUE_TEAM_ID,
    BOT_LANE_CENTERLINE,
    MAP_CENTER_SUM,
    MAP_MAX,
    MID_LANE_CENTERLINE,
    RED_BASE_POLYGON,
    RED_TEAM_ID,
    RIVER_CENTERLINE,
    TOP_LANE_CENTERLINE,
    point_in_polygon,
    point_to_polyline_distance,
)


DEFAULT_MASK_PATH = REPO_ROOT / "ProgresoActual2" / "data" / "geometry" / "observed_walkable_mask_0_14.npz"

ZONE_UNWALKABLE = "UNWALKABLE"
ZONE_OWN_BASE = "OWN_BASE"
ZONE_ENEMY_BASE = "ENEMY_BASE"
ZONE_TOP_LANE = "TOP_LANE"
ZONE_MID_LANE = "MID_LANE"
ZONE_BOT_LANE_CORE = "BOT_LANE_CORE"
ZONE_RIVER_TOP = "RIVER_TOP"
ZONE_RIVER_BOT = "RIVER_BOT"
ZONE_DRAGON_AREA = "DRAGON_AREA"
ZONE_GRUBS_HERALD_AREA = "GRUBS_HERALD_AREA"
ZONE_BARON_AREA = "BARON_AREA"
ZONE_BOT_SIDE_NEAR = "BOT_SIDE_NEAR"
ZONE_OWN_TOP_JUNGLE = "OWN_TOP_JUNGLE"
ZONE_OWN_BOTTOM_JUNGLE = "OWN_BOTTOM_JUNGLE"
ZONE_ENEMY_TOP_JUNGLE = "ENEMY_TOP_JUNGLE"
ZONE_ENEMY_BOTTOM_JUNGLE = "ENEMY_BOTTOM_JUNGLE"

ZONE_ORDER_V4 = [
    ZONE_UNWALKABLE,
    ZONE_OWN_BASE,
    ZONE_ENEMY_BASE,
    ZONE_TOP_LANE,
    ZONE_MID_LANE,
    ZONE_BOT_LANE_CORE,
    ZONE_RIVER_TOP,
    ZONE_RIVER_BOT,
    ZONE_DRAGON_AREA,
    ZONE_GRUBS_HERALD_AREA,
    ZONE_BARON_AREA,
    ZONE_BOT_SIDE_NEAR,
    ZONE_OWN_TOP_JUNGLE,
    ZONE_OWN_BOTTOM_JUNGLE,
    ZONE_ENEMY_TOP_JUNGLE,
    ZONE_ENEMY_BOTTOM_JUNGLE,
]

ZONE_TO_ID_V4 = {zone: idx for idx, zone in enumerate(ZONE_ORDER_V4)}

TOP_LANE_WIDTH_V4 = 650.0
BOT_LANE_WIDTH_V4 = 650.0
MID_LANE_WIDTH_V4 = 575.0
RIVER_WIDTH_V4 = 725.0
BOT_SIDE_NEAR_WIDTH_V4 = 2450.0
DRAGON_CENTER = (10450.0, 4400.0)
GRUBS_HERALD_CENTER = (4400.0, 10450.0)
BARON_CENTER = (4400.0, 10450.0)
OBJECTIVE_RADIUS_V4 = 950.0
WALKABLE_LOOKUP_RADIUS_CELLS = 2


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


@lru_cache(maxsize=4)
def load_walkable_mask(mask_path: str = str(DEFAULT_MASK_PATH)) -> dict:
    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing v4 walkable mask: {path}. Run build_geometry_v4_artifacts.py first."
        )
    data = np.load(path, allow_pickle=True)
    return {
        "mask": data["walkable_mask"].astype(bool),
        "smooth_density": data["smooth_density"] if "smooth_density" in data else None,
        "bins": int(data["bins"]),
        "map_max": float(data["map_max"]),
        "threshold": float(data["threshold"]) if "threshold" in data else None,
    }


def _coord_to_idx(x: float, y: float, bins: int, map_max: float) -> Optional[Tuple[int, int]]:
    if x < 0.0 or y < 0.0 or x > map_max or y > map_max:
        return None
    ix = min(bins - 1, max(0, int((x / map_max) * bins)))
    iy = min(bins - 1, max(0, int((y / map_max) * bins)))
    return ix, iy


def is_walkable_v4(x: float, y: float, mask_path: str = str(DEFAULT_MASK_PATH)) -> bool:
    data = load_walkable_mask(mask_path)
    idx = _coord_to_idx(float(x), float(y), data["bins"], data["map_max"])
    if idx is None:
        return False
    ix, iy = idx
    mask = data["mask"]
    if bool(mask[ix, iy]):
        return True
    radius = WALKABLE_LOOKUP_RADIUS_CELLS
    x0 = max(0, ix - radius)
    x1 = min(mask.shape[0], ix + radius + 1)
    y0 = max(0, iy - radius)
    y1 = min(mask.shape[1], iy + radius + 1)
    return bool(mask[x0:x1, y0:y1].any())


def is_in_blue_base(x: float, y: float) -> bool:
    return point_in_polygon(float(x), float(y), BLUE_BASE_POLYGON)


def is_in_red_base(x: float, y: float) -> bool:
    return point_in_polygon(float(x), float(y), RED_BASE_POLYGON)


def distance_to_bot_lane_v4(x: float, y: float, team_id: int = BLUE_TEAM_ID) -> float:
    del team_id
    return float(point_to_polyline_distance(float(x), float(y), BOT_LANE_CENTERLINE))


def bot_distance_signal_v4(x: float, y: float, team_id: int = BLUE_TEAM_ID) -> float:
    d = distance_to_bot_lane_v4(x, y, team_id)
    # Smooth signal: near bot lane approaches 0, clear roam distances approach 1.
    return float(1.0 / (1.0 + math.exp(-(d - 1850.0) / 650.0)))


def _is_near_objective(x: float, y: float, center: Tuple[float, float]) -> bool:
    return _dist((x, y), center) <= OBJECTIVE_RADIUS_V4


def _lane_zone(x: float, y: float) -> Optional[str]:
    if point_to_polyline_distance(x, y, BOT_LANE_CENTERLINE) <= BOT_LANE_WIDTH_V4:
        return ZONE_BOT_LANE_CORE
    if point_to_polyline_distance(x, y, MID_LANE_CENTERLINE) <= MID_LANE_WIDTH_V4:
        return ZONE_MID_LANE
    if point_to_polyline_distance(x, y, TOP_LANE_CENTERLINE) <= TOP_LANE_WIDTH_V4:
        return ZONE_TOP_LANE
    return None


def _river_zone(x: float, y: float) -> Optional[str]:
    if point_to_polyline_distance(x, y, RIVER_CENTERLINE) <= RIVER_WIDTH_V4:
        return ZONE_RIVER_TOP if y >= x else ZONE_RIVER_BOT
    return None


def _jungle_fallback(x: float, y: float, team_id: int) -> str:
    blue_side = (x + y) < MAP_CENTER_SUM
    own_or_enemy = "OWN" if (team_id == BLUE_TEAM_ID) == blue_side else "ENEMY"
    top_or_bottom = "TOP" if y >= x else "BOTTOM"
    return f"{own_or_enemy}_{top_or_bottom}_JUNGLE"


def classify_zone_v4(x: float, y: float, team_id: int, mask_path: str = str(DEFAULT_MASK_PATH)) -> str:
    x = float(x)
    y = float(y)
    if x < 0.0 or y < 0.0 or x > MAP_MAX or y > MAP_MAX:
        return ZONE_UNWALKABLE

    if is_in_blue_base(x, y):
        return ZONE_OWN_BASE if team_id == BLUE_TEAM_ID else ZONE_ENEMY_BASE
    if is_in_red_base(x, y):
        return ZONE_OWN_BASE if team_id == RED_TEAM_ID else ZONE_ENEMY_BASE

    if not is_walkable_v4(x, y, mask_path):
        return ZONE_UNWALKABLE

    if _is_near_objective(x, y, DRAGON_CENTER):
        return ZONE_DRAGON_AREA
    if _is_near_objective(x, y, GRUBS_HERALD_CENTER):
        return ZONE_GRUBS_HERALD_AREA

    zone = _lane_zone(x, y)
    if zone is not None:
        return zone

    zone = _river_zone(x, y)
    if zone is not None:
        return zone

    if distance_to_bot_lane_v4(x, y, team_id) <= BOT_SIDE_NEAR_WIDTH_V4 and y <= x + 1200:
        return ZONE_BOT_SIDE_NEAR

    return _jungle_fallback(x, y, team_id)


def is_in_bot_context_v4(x: float, y: float, team_id: int, mask_path: str = str(DEFAULT_MASK_PATH)) -> bool:
    zone = classify_zone_v4(x, y, team_id, mask_path)
    return zone in {
        ZONE_BOT_LANE_CORE,
        ZONE_BOT_SIDE_NEAR,
        ZONE_RIVER_BOT,
        ZONE_DRAGON_AREA,
    }


def zone_id_v4(zone: str) -> int:
    return int(ZONE_TO_ID_V4.get(zone, ZONE_TO_ID_V4[ZONE_UNWALKABLE]))
