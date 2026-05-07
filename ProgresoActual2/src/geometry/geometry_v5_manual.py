#!/usr/bin/env python3
"""
Manual geometry v5 for support roaming analysis.

This geometry is intentionally semantic. It uses broad, hand-traced regions
from the observed player-density annotation instead of trying to model every
wall or passable pixel.
"""

from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_UTILS_DIR = REPO_ROOT / "ProgresoActual" / "src" / "02_data_processing"
sys.path.insert(0, str(SHARED_UTILS_DIR))

from shared_utils import (  # noqa: E402
    BLUE_TEAM_ID,
    MAP_CENTER_SUM,
    MAP_MAX,
    RED_TEAM_ID,
    point_in_polygon,
    point_to_polyline_distance,
)


DEFAULT_CONFIG_PATH = REPO_ROOT / "ProgresoActual2" / "data" / "geometry" / "manual_geometry_v5_config.json"

ZONE_OUT_OF_MAP = "OUT_OF_MAP"
ZONE_UNCLASSIFIED = "UNCLASSIFIED"

ZONE_ORDER_V5 = [
    ZONE_OUT_OF_MAP,
    ZONE_UNCLASSIFIED,
    "BLUE_BASE",
    "RED_BASE",
    "BARON_GRUBS_HERALD_AREA",
    "DRAGON_AREA",
    "MID_LANE",
    "TOP_LANE_CORE",
    "BOT_LANE_CORE",
    "TOP_SIDE_NEAR",
    "BOT_SIDE_NEAR",
    "RIVER_TOP",
    "RIVER_BOT",
    "BLUE_TOP_JUNGLE",
    "BLUE_BOT_JUNGLE",
    "RED_TOP_JUNGLE",
    "RED_BOT_JUNGLE",
]
ZONE_TO_ID_V5 = {zone: idx for idx, zone in enumerate(ZONE_ORDER_V5)}


@lru_cache(maxsize=8)
def load_geometry_v5_config(config_path: str = str(DEFAULT_CONFIG_PATH)) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing manual geometry config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _as_points(points: Iterable[Iterable[float]]) -> List[Tuple[float, float]]:
    return [(float(x), float(y)) for x, y in points]


def _in_circle(x: float, y: float, circle: dict) -> bool:
    cx, cy = circle["center"]
    radius = float(circle["radius"])
    return (x - float(cx)) ** 2 + (y - float(cy)) ** 2 <= radius ** 2


def _dist_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    denom = abx * abx + aby * aby
    if denom <= 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / denom))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


def _distance_to_polygon(x: float, y: float, points: List[Tuple[float, float]]) -> float:
    if point_in_polygon(x, y, points):
        return 0.0
    best = float("inf")
    for i, (ax, ay) in enumerate(points):
        bx, by = points[(i + 1) % len(points)]
        best = min(best, _dist_point_to_segment(x, y, ax, ay, bx, by))
    return float(best)


def _fallback_jungle_zone(x: float, y: float) -> str:
    blue_side = (x + y) < MAP_CENTER_SUM
    top_side = y >= x
    if blue_side and top_side:
        return "BLUE_TOP_JUNGLE"
    if blue_side:
        return "BLUE_BOT_JUNGLE"
    if top_side:
        return "RED_TOP_JUNGLE"
    return "RED_BOT_JUNGLE"


def classify_zone_v5(
    x: float,
    y: float,
    team_id: Optional[int] = None,
    config_path: str = str(DEFAULT_CONFIG_PATH),
    relative: bool = False,
) -> str:
    """Classify an absolute map coordinate into the manual v5 zone set."""
    x = float(x)
    y = float(y)
    if x < 0.0 or y < 0.0 or x > MAP_MAX or y > MAP_MAX:
        return ZONE_OUT_OF_MAP

    config = load_geometry_v5_config(config_path)
    polygons: Dict[str, list] = config.get("polygons", {})
    circles: Dict[str, dict] = config.get("circles", {})
    centerline_zones: Dict[str, dict] = config.get("centerline_zones", {})

    for zone in config["priority"]:
        if zone in circles and _in_circle(x, y, circles[zone]):
            return _to_relative_zone(zone, team_id) if relative else zone
        if zone in centerline_zones:
            centerline_spec = centerline_zones[zone]
            centerline = _as_points(centerline_spec["centerline"])
            width = float(centerline_spec["width"])
            if point_to_polyline_distance(x, y, centerline) <= width:
                return _to_relative_zone(zone, team_id) if relative else zone
            if centerline_spec.get("classification_only", False):
                continue
        if zone in polygons and point_in_polygon(x, y, _as_points(polygons[zone])):
            return _to_relative_zone(zone, team_id) if relative else zone

    zone = _fallback_jungle_zone(x, y)
    return _to_relative_zone(zone, team_id) if relative else zone


def _to_relative_zone(zone: str, team_id: Optional[int]) -> str:
    if team_id not in {BLUE_TEAM_ID, RED_TEAM_ID}:
        return zone
    if zone == "BLUE_BASE":
        return "OWN_BASE" if team_id == BLUE_TEAM_ID else "ENEMY_BASE"
    if zone == "RED_BASE":
        return "OWN_BASE" if team_id == RED_TEAM_ID else "ENEMY_BASE"
    if zone.startswith("BLUE_") and zone.endswith("_JUNGLE"):
        side = "TOP" if "_TOP_" in zone else "BOTTOM"
        return f"{'OWN' if team_id == BLUE_TEAM_ID else 'ENEMY'}_{side}_JUNGLE"
    if zone.startswith("RED_") and zone.endswith("_JUNGLE"):
        side = "TOP" if "_TOP_" in zone else "BOTTOM"
        return f"{'OWN' if team_id == RED_TEAM_ID else 'ENEMY'}_{side}_JUNGLE"
    return zone


def is_walkable_v5(x: float, y: float, config_path: str = str(DEFAULT_CONFIG_PATH)) -> bool:
    return classify_zone_v5(x, y, config_path=config_path) != ZONE_OUT_OF_MAP


def distance_to_bot_lane_v5(x: float, y: float, team_id: int = BLUE_TEAM_ID, config_path: str = str(DEFAULT_CONFIG_PATH)) -> float:
    del team_id
    config = load_geometry_v5_config(config_path)
    if "BOT_LANE_CORE" in config.get("centerline_zones", {}):
        centerline = _as_points(config["centerline_zones"]["BOT_LANE_CORE"]["centerline"])
        return float(point_to_polyline_distance(float(x), float(y), centerline))
    bot_points = _as_points(config["polygons"]["BOT_LANE_CORE"])
    return _distance_to_polygon(float(x), float(y), bot_points)


def bot_distance_signal_v5(x: float, y: float, team_id: int = BLUE_TEAM_ID, config_path: str = str(DEFAULT_CONFIG_PATH)) -> float:
    d = distance_to_bot_lane_v5(x, y, team_id=team_id, config_path=config_path)
    return float(1.0 / (1.0 + math.exp(-(d - 1850.0) / 650.0)))


def is_in_bot_context_v5(x: float, y: float, team_id: int, config_path: str = str(DEFAULT_CONFIG_PATH)) -> bool:
    del team_id
    zone = classify_zone_v5(x, y, config_path=config_path)
    return zone in {"BOT_LANE_CORE", "BOT_SIDE_NEAR", "RIVER_BOT", "DRAGON_AREA"}


def zone_id_v5(zone: str) -> int:
    return int(ZONE_TO_ID_V5.get(zone, ZONE_TO_ID_V5[ZONE_UNCLASSIFIED]))
