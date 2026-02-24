import os
import math
import time
import random

import pandas as pd
from dotenv import load_dotenv
from riotwatcher import LolWatcher, RiotWatcher, ApiError



def sigmoid(x: float) -> float:
    """Sigmoid numerically-stable-ish for moderate x."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

# --- CONFIGURACIÓN ---
load_dotenv('TFG.env')

API_KEY = os.getenv('RIOT_API_KEY')
REGION = os.getenv('REGION', 'europe')          # routing (europe/americas/asia)
MATCH_REGION = os.getenv('MATCH_REGION', 'euw1')  # platform (euw1/na1/...)

GAME_NAME = os.getenv('GAME_NAME', 'adriidz')
TAG_LINE = os.getenv('TAG_LINE', 'diaz')

if not API_KEY:
    raise ValueError("ERROR: No se encontró la variable 'RIOT_API_KEY' en el archivo .env")

# Watchers
watcher = LolWatcher(API_KEY)          # LoL endpoints
account_watcher = RiotWatcher(API_KEY) # Account endpoints

# --- MVP LABEL TUNING CONSTANTS (v4.1) ---
PWR_TAKEDOWN_GOLD = float(os.getenv('PWR_TAKEDOWN_GOLD', '90'))
PWR_SIGMOID_SCALE = float(os.getenv('PWR_SIGMOID_SCALE', '800'))
STYLE_NEAR_DIST = float(os.getenv('STYLE_NEAR_DIST', '2600'))
STYLE_NEAR_ALLIES = int(os.getenv('STYLE_NEAR_ALLIES', '1'))
STYLE_SIDE_MARGIN = float(os.getenv('STYLE_SIDE_MARGIN', '0.25'))
STYLE_SIGMOID_SCALE = float(os.getenv('STYLE_SIGMOID_SCALE', '0.10'))


# -------------------------
# RETRIES 429 / 5XX
# -------------------------

def riot_call_with_retry(fn, *args, max_retries=8, base_sleep=1.5, jitter=0.25, **kwargs):
    """Wrapper para llamadas a RiotWatcher/LolWatcher.

    - 429: respeta Retry-After si existe, si no backoff exponencial.
    - 5xx: backoff exponencial.

    Esto evita que el script se pare por rate limit.
    """
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except ApiError as err:
            status = getattr(err.response, "status_code", None)

            # Rate limit
            if status == 429:
                attempt += 1
                if attempt > max_retries:
                    raise

                retry_after = None
                try:
                    ra = err.response.headers.get("Retry-After")
                    if ra is not None:
                        retry_after = float(ra)
                except Exception:
                    retry_after = None

                sleep_s = retry_after if retry_after is not None else (base_sleep * (2 ** (attempt - 1)))
                sleep_s += random.uniform(0, jitter)
                time.sleep(sleep_s)
                continue

            # Transient server errors
            if status in (500, 502, 503, 504):
                attempt += 1
                if attempt > max_retries:
                    raise
                sleep_s = base_sleep * (2 ** (attempt - 1)) + random.uniform(0, jitter)
                time.sleep(sleep_s)
                continue

            # Other errors -> raise
            raise


def get_match_ids(puuid, count=5, ranked_only=False):
    """Obtiene IDs por puuid con retry."""
    queue = 420 if ranked_only else None
    try:
        return riot_call_with_retry(watcher.match.matchlist_by_puuid, REGION, puuid, count=count, queue=queue)
    except ApiError as err:
        print(f"Error obteniendo IDs: {err}")
        return []


# -------------------------
# POSICIÓN / CLUSTER
# -------------------------

def _euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _safe_pos(obj):
    """obj puede ser participantFrame o event; retorna (x,y) o None."""
    pos = obj.get('position')
    if not pos:
        return None
    x = pos.get('x')
    y = pos.get('y')
    if x is None or y is None:
        return None
    return (float(x), float(y))


def _lane_by_position_xy(x, y):
    """Clasificador simple (robusto) de lane por coordenadas."""
    if abs(x - y) <= 2000 and 3500 <= x <= 11500 and 3500 <= y <= 11500:
        return 'MID'
    if y >= 9000 and x <= 9000:
        return 'TOP'
    if x >= 6000 and y <= 6000:
        return 'BOT'

    centers = {'TOP': (2500, 12500), 'MID': (7500, 7500), 'BOT': (12500, 2500)}
    d = {k: _euclid((x, y), v) for k, v in centers.items()}
    return min(d, key=d.get)


def get_positions_by_minute(timeline, minutes_to_track=(5, 10, 15, 20)):
    """positions[minute][participantId] = (x,y)"""
    frames = timeline['info']['frames']
    positions = {m: {} for m in minutes_to_track}
    max_i = len(frames) - 1

    for m in minutes_to_track:
        idx = min(m, max_i)
        frame = frames[idx]
        for pid_str, pframe in frame['participantFrames'].items():
            pid = int(pid_str)
            positions[m][pid] = _safe_pos(pframe)
    return positions


def compute_team_cluster_metrics(positions_at_minute, participants_info, team_id, grouped_radius=2200.0):
    """Spread medio al centroide + ratio de jugadores dentro del radio + max_pair_dist."""
    pts = []
    for p in participants_info:
        if p['teamId'] != team_id:
            continue
        pid = p['participantId']
        pos = positions_at_minute.get(pid)
        if pos is not None:
            pts.append(pos)

    if len(pts) < 3:
        return {'spread_mean': None, 'grouped_ratio': None, 'max_pair_dist': None}

    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    centroid = (cx, cy)

    dists = [_euclid(pt, centroid) for pt in pts]
    spread_mean = sum(dists) / len(dists)
    grouped_ratio = sum(1 for d in dists if d <= grouped_radius) / len(pts)

    max_pair = 0.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            max_pair = max(max_pair, _euclid(pts[i], pts[j]))

    return {'spread_mean': spread_mean, 'grouped_ratio': grouped_ratio, 'max_pair_dist': max_pair}



# -------------------------
# PLAYER-LEVEL POSITION METRICS (for MVP labels)
# -------------------------

def compute_player_group_side_metrics(timeline, participants, focus_team_id, start_min=14, end_min=25,
                                      near_dist=STYLE_NEAR_DIST):
    """Compute simple per-player grouping vs sidelane metrics in a time window.

    Returns dict keyed by participantId:
      {
        pid: {
          'group_ratio': float,   # fraction of valid minutes where player is near >=2 allies
          'side_ratio': float,    # fraction of valid minutes where player is in TOP/BOT lane zones
          'valid_minutes': int
        }
      }

    Notes:
    - Uses timeline frames (1 frame ~= 1 minute).
    - "Near" is Euclidean distance < near_dist to allies (based on x,y).
    - "Side lane" uses _lane_by_position_xy() and counts TOP/BOT.
    """
    pid_to_team = build_participant_to_team(participants)
    allies = [p['participantId'] for p in participants if p.get('teamId') == focus_team_id]
    out = {pid: {'group_hits': 0, 'side_hits': 0, 'valid': 0} for pid in allies}

    frames = timeline.get('info', {}).get('frames', [])
    if not frames:
        return {pid: {'group_ratio': 0.0, 'side_ratio': 0.0, 'valid_minutes': 0} for pid in allies}

    start_min = max(0, int(start_min))
    end_min = int(end_min)
    if end_min < start_min:
        end_min = start_min

    near2 = max(1, int(STYLE_NEAR_ALLIES))
    near_dist2 = float(near_dist) ** 2

    for minute in range(start_min, min(end_min, len(frames) - 1) + 1):
        fr = frames[minute]
        pframes = fr.get('participantFrames', {})

        # prefetch positions for allies this minute
        pos_cache = {}
        for pid in allies:
            pf = pframes.get(str(pid)) or pframes.get(pid)
            if not pf:
                pos_cache[pid] = None
                continue
            pos = pf.get('position')
            if not pos or 'x' not in pos or 'y' not in pos:
                pos_cache[pid] = None
            else:
                pos_cache[pid] = (pos['x'], pos['y'])

        for pid in allies:
            xy = pos_cache.get(pid)
            if xy is None:
                continue

            out[pid]['valid'] += 1

            # side lane?
            lane = _lane_by_position_xy(xy[0], xy[1])
            if lane in ('TOP', 'BOT'):
                out[pid]['side_hits'] += 1

            # grouped? near at least 2 allies
            near_cnt = 0
            x, y = xy
            for other in allies:
                if other == pid:
                    continue
                oxy = pos_cache.get(other)
                if oxy is None:
                    continue
                dx = x - oxy[0]
                dy = y - oxy[1]
                if (dx * dx + dy * dy) <= near_dist2:
                    near_cnt += 1
                    if near_cnt >= near2:
                        break
            if near_cnt >= near2:
                out[pid]['group_hits'] += 1

    # finalize
    final = {}
    for pid, d in out.items():
        v = d['valid']
        if v <= 0:
            final[pid] = {'group_ratio': 0.0, 'side_ratio': 0.0, 'valid_minutes': 0}
        else:
            final[pid] = {
                'group_ratio': d['group_hits'] / v,
                'side_ratio': d['side_hits'] / v,
                'valid_minutes': v
            }
    return final


def derive_player_mvp_labels(row, role_prefix, side_ratio=None, group_ratio=None):
    """Derive MVP player labels:
      - Y_{role}_PWR_EARLY_PROB / Y_{role}_PWR_CLASS
      - Y_{role}_STYLE_SIDE_PROB / Y_{role}_STYLE_CLASS

    Uses only columns already present in `row` plus optional side/group ratios.
    """
    # --- Power timing (early vs late proxy)
    gd15 = row.get(f"Post_{role_prefix}_GoldDiff_15")
    gd20 = row.get(f"Post_{role_prefix}_GoldDiff_20")
    td15 = row.get(f"Post_{role_prefix}_EarlyTakedowns_15")

    # robust defaults
    gd15 = 0.0 if gd15 is None else float(gd15)
    gd20 = 0.0 if gd20 is None else float(gd20)
    td15 = 0.0 if td15 is None else float(td15)

    # scale takedowns to "gold-ish" units (very rough but stable)
    early_score = gd15 + PWR_TAKEDOWN_GOLD * td15
    late_proxy = gd20

    delta = early_score - late_proxy
    p_early = sigmoid(delta / PWR_SIGMOID_SCALE)  # scale to avoid saturation

    row[f"Y_{role_prefix}_PWR_EARLY_PROB"] = p_early
    row[f"Y_{role_prefix}_PWR_CLASS"] = "PWR_EARLY" if p_early >= 0.5 else "PWR_LATE"

    # --- Style: side vs group
    if side_ratio is None or group_ratio is None:
        row[f"Y_{role_prefix}_STYLE_SIDE_PROB"] = None
        row[f"Y_{role_prefix}_STYLE_CLASS"] = None
        return

    side_ratio = float(side_ratio)
    group_ratio = float(group_ratio)

    # score >0 => more side than grouped
    style_delta = (side_ratio - group_ratio)
    p_side = sigmoid((style_delta - STYLE_SIDE_MARGIN) / STYLE_SIGMOID_SCALE)  # centered on margin

    row[f"Feat_{role_prefix}_SideRatio_14_25"] = side_ratio
    row[f"Feat_{role_prefix}_GroupRatio_14_25"] = group_ratio
    row[f"Feat_{role_prefix}_PosValidMin_14_25"] = int(row.get(f"Feat_{role_prefix}_PosValidMin_14_25") or 0)

    row[f"Y_{role_prefix}_STYLE_SIDE_PROB"] = p_side
    row[f"Y_{role_prefix}_STYLE_CLASS"] = "STYLE_SIDE" if style_delta > STYLE_SIDE_MARGIN else "STYLE_GROUP"


def compute_jungle_presence_and_focus(timeline, participants_info, team_id, start_min=2, end_min=14):
    """Presencia jungla por lane (TOP/MID/BOT) y por mapside (TOPSIDE/BOTSIDE)."""
    jg_pid = None
    for p in participants_info:
        if p['teamId'] == team_id and p.get('teamPosition') == 'JUNGLE':
            jg_pid = p['participantId']
            break

    if jg_pid is None:
        return {
            'lane_top': None, 'lane_mid': None, 'lane_bot': None, 'lane_focus': None,
            'side_top': None, 'side_bot': None, 'side_focus': None,
        }

    frames = timeline['info']['frames']
    max_i = len(frames) - 1
    s = max(0, min(start_min, max_i))
    e = max(0, min(end_min, max_i))

    lane_counts = {'TOP': 0, 'MID': 0, 'BOT': 0}
    side_counts = {'TOPSIDE': 0, 'BOTSIDE': 0, 'NEUTRAL': 0}
    total = 0

    for i in range(s, e + 1):
        pf = frames[i]['participantFrames'].get(str(jg_pid))
        if not pf:
            continue
        pos = _safe_pos(pf)
        if pos is None:
            continue

        lane = _lane_by_position_xy(pos[0], pos[1])
        lane_counts[lane] += 1

        if lane == 'TOP':
            side_counts['TOPSIDE'] += 1
        elif lane == 'BOT':
            side_counts['BOTSIDE'] += 1
        else:
            side_counts['NEUTRAL'] += 1

        total += 1

    if total == 0:
        return {
            'lane_top': 0.0, 'lane_mid': 0.0, 'lane_bot': 0.0, 'lane_focus': 'NEUTRAL',
            'side_top': 0.0, 'side_bot': 0.0, 'side_focus': 'NEUTRAL',
        }

    lane_ratios = {k: v / total for k, v in lane_counts.items()}
    side_ratios = {k: v / total for k, v in side_counts.items()}

    lane_focus = max(lane_ratios, key=lane_ratios.get)
    vals = sorted(lane_ratios.values(), reverse=True)
    if len(vals) >= 2 and (vals[0] - vals[1]) < 0.10:
        lane_focus = 'NEUTRAL'

    side_focus = 'NEUTRAL'
    if (side_ratios['TOPSIDE'] - side_ratios['BOTSIDE']) > 0.10:
        side_focus = 'TOPSIDE'
    elif (side_ratios['BOTSIDE'] - side_ratios['TOPSIDE']) > 0.10:
        side_focus = 'BOTSIDE'

    return {
        'lane_top': lane_ratios['TOP'],
        'lane_mid': lane_ratios['MID'],
        'lane_bot': lane_ratios['BOT'],
        'lane_focus': lane_focus,
        'side_top': side_ratios['TOPSIDE'],
        'side_bot': side_ratios['BOTSIDE'],
        'side_focus': side_focus,
    }


# -------------------------
# EVENTS: OBJETIVOS, FIGHTS, TORRES/PLATES
# -------------------------

def build_participant_to_team(info_participants):
    return {p['participantId']: p['teamId'] for p in info_participants}


def iter_timeline_events(timeline):
    """Yield (event, time_seconds)."""
    for frame in timeline['info']['frames']:
        for ev in frame.get('events', []):
            ts = ev.get('timestamp')
            if ts is None:
                continue
            yield ev, (ts / 1000.0)


def parse_objective_times(timeline, pid_to_team):
    """Objetivos con timestamp y conteos 0-14 (soporta ELITE_MONSTER_KILL)."""
    out = {
        100: {"first_dragon": None, "first_herald": None, "first_grubs": None,
              "drag_0_14": 0, "herald_0_14": 0, "grubs_0_14": 0},
        200: {"first_dragon": None, "first_herald": None, "first_grubs": None,
              "drag_0_14": 0, "herald_0_14": 0, "grubs_0_14": 0},
    }

    for ev, t in iter_timeline_events(timeline):
        et = ev.get("type")
        killer = ev.get("killerId", 0)
        if killer <= 0:
            continue
        team = pid_to_team.get(killer)
        if team not in (100, 200):
            continue

        if et == "ELITE_MONSTER_KILL":
            mtype = ev.get("monsterType")
            if mtype == "DRAGON":
                if out[team]["first_dragon"] is None:
                    out[team]["first_dragon"] = t
                if t <= 14 * 60:
                    out[team]["drag_0_14"] += 1
            elif mtype == "RIFTHERALD":
                if out[team]["first_herald"] is None:
                    out[team]["first_herald"] = t
                if t <= 14 * 60:
                    out[team]["herald_0_14"] += 1
            elif mtype == "HORDE":
                if out[team]["first_grubs"] is None:
                    out[team]["first_grubs"] = t
                if t <= 14 * 60:
                    out[team]["grubs_0_14"] += 1
            continue

        # fallback por si te aparece en tu región
        if et == "DRAGON_KILL":
            if out[team]["first_dragon"] is None:
                out[team]["first_dragon"] = t
            if t <= 14 * 60:
                out[team]["drag_0_14"] += 1
        elif et == "RIFT_HERALD_KILL":
            if out[team]["first_herald"] is None:
                out[team]["first_herald"] = t
            if t <= 14 * 60:
                out[team]["herald_0_14"] += 1
        elif et == "HORDE_KILL":
            if out[team]["first_grubs"] is None:
                out[team]["first_grubs"] = t
            if t <= 14 * 60:
                out[team]["grubs_0_14"] += 1

    return out


def parse_tower_and_plate_events(timeline, pid_to_team):
    """Torres y plates por lane y ventana temporal."""
    windows = {'0_14': (0, 14 * 60), '14_25': (14 * 60, 25 * 60)}

    def init_bucket():
        return {
            'towers': {'TOP': 0, 'MID': 0, 'BOT': 0},
            'plates': {'TOP': 0, 'MID': 0, 'BOT': 0},
            'first_tower_time': None,
        }

    out = {100: {w: init_bucket() for w in windows}, 200: {w: init_bucket() for w in windows}}

    for ev, t in iter_timeline_events(timeline):
        et = ev.get('type')

        if et == 'BUILDING_KILL':
            if ev.get('buildingType') != 'TOWER_BUILDING':
                continue
            killer = ev.get('killerId', 0)
            if killer <= 0:
                continue
            team = pid_to_team.get(killer)
            if team not in (100, 200):
                continue

            if ev.get("laneType") in ("TOP_LANE", "MID_LANE", "BOT_LANE"):
                lane = {"TOP_LANE": "TOP", "MID_LANE": "MID", "BOT_LANE": "BOT"}[ev["laneType"]]
            else:
                pos = _safe_pos(ev)
                if pos is None:
                    continue
                lane = _lane_by_position_xy(pos[0], pos[1])

            ft = out[team]['0_14']['first_tower_time']
            if ft is None or t < ft:
                out[team]['0_14']['first_tower_time'] = t

            for w, (a, b) in windows.items():
                if a <= t <= b:
                    out[team][w]['towers'][lane] += 1

        elif et == 'TURRET_PLATE_DESTROYED':
            killer = ev.get('killerId', 0)
            if killer <= 0:
                continue
            team = pid_to_team.get(killer)
            if team not in (100, 200):
                continue

            if ev.get("laneType") in ("TOP_LANE", "MID_LANE", "BOT_LANE"):
                lane = {"TOP_LANE": "TOP", "MID_LANE": "MID", "BOT_LANE": "BOT"}[ev["laneType"]]
            else:
                pos = _safe_pos(ev)
                if pos is None:
                    continue
                lane = _lane_by_position_xy(pos[0], pos[1])

            if 0 <= t <= 14 * 60:
                out[team]['0_14']['plates'][lane] += 1

    for team in (100, 200):
        ft = out[team]['0_14']['first_tower_time']
        out[team]['14_25']['first_tower_time'] = ft

    return out


def parse_kills_and_fights(timeline, pid_to_team, fight_gap_s=25.0, end_min=25):
    """Fights robustos (0-14 / 14-25 / 0-25)."""
    end_t = end_min * 60

    kill_events = []
    kills_0_14 = {100: 0, 200: 0}
    kills_14_25 = {100: 0, 200: 0}

    for ev, t in iter_timeline_events(timeline):
        if t > end_t:
            continue
        if ev.get("type") != "CHAMPION_KILL":
            continue

        killer = ev.get("killerId", 0)
        if killer <= 0:
            continue
        team = pid_to_team.get(killer)
        if team not in (100, 200):
            continue

        if t <= 14 * 60:
            kills_0_14[team] += 1
        elif 14 * 60 < t <= 25 * 60:
            kills_14_25[team] += 1

        participants = {killer}
        victim = ev.get("victimId", 0)
        if victim > 0:
            participants.add(victim)
        for a in (ev.get("assistingParticipantIds") or []):
            participants.add(a)

        kill_events.append((t, team, participants))

    kill_events.sort(key=lambda x: x[0])

    fights = []
    cur = None
    for t, team, pset in kill_events:
        if cur is None:
            cur = {"start": t, "end": t, "teams": {team}, "kills": 1, "participants": set(pset)}
            continue

        if (t - cur["end"]) <= fight_gap_s:
            cur["end"] = t
            cur["teams"].add(team)
            cur["kills"] += 1
            cur["participants"].update(pset)
        else:
            fights.append(cur)
            cur = {"start": t, "end": t, "teams": {team}, "kills": 1, "participants": set(pset)}
    if cur is not None:
        fights.append(cur)

    def classify_fight(f):
        num_kills = f["kills"]
        uniq = len(f["participants"])
        dur = f["end"] - f["start"]

        if num_kills >= 3 or uniq >= 6:
            return "TEAMFIGHT"
        if (num_kills == 2 and dur <= 10) or uniq == 5:
            return "SKIRMISH"
        return "PICK"

    for f in fights:
        f["type"] = classify_fight(f)
        f["uniq_participants"] = len(f["participants"])
        f["duration"] = f["end"] - f["start"]

    windows = {"0_14": (0, 14 * 60), "14_25": (14 * 60, 25 * 60), "0_25": (0, 25 * 60)}

    def init_team_bucket():
        return {
            "total_fights": 0,
            "pick_fights": 0,
            "skirmish_fights": 0,
            "teamfight_fights": 0,
            "total_kills_in_fights": 0,
            "sum_uniq_participants": 0,
            "sum_duration": 0.0,
            "first_fight_time": None,
            "first_fight_type": None,
        }

    out = {w: {100: init_team_bucket(), 200: init_team_bucket()} for w in windows}

    for f in fights:
        for w, (a, b) in windows.items():
            if not (a <= f["start"] <= b):
                continue
            for team in f["teams"]:
                tb = out[w][team]
                tb["total_fights"] += 1
                tb["total_kills_in_fights"] += f["kills"]
                tb["sum_uniq_participants"] += f["uniq_participants"]
                tb["sum_duration"] += f["duration"]

                if tb["first_fight_time"] is None or f["start"] < tb["first_fight_time"]:
                    tb["first_fight_time"] = f["start"]
                    tb["first_fight_type"] = f["type"]

                if f["type"] == "PICK":
                    tb["pick_fights"] += 1
                elif f["type"] == "SKIRMISH":
                    tb["skirmish_fights"] += 1
                else:
                    tb["teamfight_fights"] += 1

    def finalize(tb, window_minutes):
        total = tb["total_fights"]
        if total == 0:
            return {
                "total_fights": 0, "pick_fights": 0, "skirmish_fights": 0, "teamfight_fights": 0,
                "pick_ratio": 0.0, "skirmish_ratio": 0.0, "teamfight_ratio": 0.0,
                "fights_per_min": 0.0, "avg_kills_per_fight": 0.0, "avg_uniq_participants": 0.0,
                "avg_fight_duration": 0.0, "first_fight_time": tb["first_fight_time"],
                "first_fight_type": tb["first_fight_type"],
            }
        return {
            "total_fights": total,
            "pick_fights": tb["pick_fights"],
            "skirmish_fights": tb["skirmish_fights"],
            "teamfight_fights": tb["teamfight_fights"],
            "pick_ratio": tb["pick_fights"] / total,
            "skirmish_ratio": tb["skirmish_fights"] / total,
            "teamfight_ratio": tb["teamfight_fights"] / total,
            "fights_per_min": total / max(window_minutes, 1e-6),
            "avg_kills_per_fight": tb["total_kills_in_fights"] / total,
            "avg_uniq_participants": tb["sum_uniq_participants"] / total,
            "avg_fight_duration": tb["sum_duration"] / total,
            "first_fight_time": tb["first_fight_time"],
            "first_fight_type": tb["first_fight_type"],
        }

    win_minutes = {"0_14": 14, "14_25": 11, "0_25": 25}
    final = {w: {team: finalize(out[w][team], win_minutes[w]) for team in (100, 200)} for w in windows}

    return {"kills_0_14": kills_0_14, "kills_14_25": kills_14_25, "fights": final}


# -------------------------
# SNAPSHOTS TEMPORALES
# -------------------------

def get_timeline_stats_by_minute(timeline, minutes_to_track=(5, 10, 15, 20)):
    """Snapshots por minuto: gold/xp/cs y takedowns acumulados."""
    frames = timeline['info']['frames']
    snapshots = {}
    takedowns_counter = {pid: 0 for pid in range(1, 11)}

    for i, frame in enumerate(frames):
        for event in frame.get('events', []):
            if event.get('type') == 'CHAMPION_KILL':
                killer_id = event.get('killerId', 0)
                if killer_id > 0:
                    takedowns_counter[killer_id] += 1
                for assist_id in (event.get('assistingParticipantIds') or []):
                    takedowns_counter[assist_id] += 1

        if i in minutes_to_track:
            snapshots[i] = {}
            for pid_str, p_data in frame['participantFrames'].items():
                pid = int(pid_str)
                snapshots[i][pid] = {
                    'gold': p_data['totalGold'],
                    'xp': p_data['xp'],
                    'cs': p_data['minionsKilled'] + p_data['jungleMinionsKilled'],
                    'jungle_cs': p_data['jungleMinionsKilled'],
                    'lane_cs': p_data['minionsKilled'],
                    'takedowns': takedowns_counter[pid],
                }

    last_frame = frames[-1]['participantFrames']
    last_known = {}
    for pid_str, p_data in last_frame.items():
        pid = int(pid_str)
        last_known[pid] = {
            'gold': p_data['totalGold'],
            'xp': p_data['xp'],
            'cs': p_data['minionsKilled'] + p_data['jungleMinionsKilled'],
            'jungle_cs': p_data['jungleMinionsKilled'],
            'lane_cs': p_data['minionsKilled'],
            'takedowns': takedowns_counter[pid],
        }
    for m in minutes_to_track:
        if m not in snapshots:
            snapshots[m] = last_known

    return snapshots


# -------------------------
# PRE-GAME FEATURES
# -------------------------

def encode_summoners_unordered(s1, s2):
    """Multi-hot (orden invariante). (Sin SS_Combo para no ser redundante)"""
    SPELLS = {
        1: "CLEANSE",
        3: "EXHAUST",
        4: "FLASH",
        6: "GHOST",
        7: "HEAL",
        11: "SMITE",
        12: "TELEPORT",
        13: "CLARITY",
        14: "IGNITE",
        21: "BARRIER",
        32: "SNOWBALL",
    }
    present = {s1, s2}
    return {f"SS_{name}": (1 if sid in present else 0) for sid, name in SPELLS.items()}


def extract_keystone(participant):
    perks = participant.get('perks', {})
    styles = perks.get('styles', []) if isinstance(perks, dict) else []
    if not styles:
        return None
    try:
        selections = styles[0].get('selections', [])
        if selections:
            return selections[0].get('perk', None)
    except Exception:
        return None
    return None


# -------------------------
# LABELS (TEAM-LEVEL)
# -------------------------


def derive_team_focus_label(row):
    """Team early focus label (topside vs botside) for MVP.

    Priority:
      1) Use Label_JungleMapSideFocus_2_14 if it's TOPSIDE/BOTSIDE.
      2) Otherwise, use objective bias 0-14 (dragons vs grubs+herald).

    Produces:
      - Y_Team_FOCUS_TOP_PROB
      - Y_Team_FOCUS_CLASS  (FOCUS_TOP_EARLY / FOCUS_BOT_EARLY / FOCUS_NEUTRAL)
    """
    side_focus = row.get('Label_JungleMapSideFocus_2_14')
    top_pres = row.get('Feat_Jungle_TopsidePresence_2_14') or 0.0
    bot_pres = row.get('Feat_Jungle_BotsidePresence_2_14') or 0.0

    # objective proxy
    n_drag = row.get('Feat_Dragons_0_14') or 0
    n_grub = row.get('Feat_Grubs_0_14') or 0
    n_her  = row.get('Feat_Heralds_0_14') or 0

    top_obj = float(n_grub + n_her)
    bot_obj = float(n_drag)

    # presence is already ratios; blend with objectives if neutral
    if side_focus == 'TOPSIDE':
        score = (float(top_pres) - float(bot_pres)) + 0.5
    elif side_focus == 'BOTSIDE':
        score = (float(top_pres) - float(bot_pres)) - 0.5
    else:
        score = (float(top_pres) - float(bot_pres)) + 0.35 * (top_obj - bot_obj)

    p_top = sigmoid(score / 0.35)

    row['Y_Team_FOCUS_TOP_PROB'] = p_top
    if 0.45 <= p_top <= 0.55:
        row['Y_Team_FOCUS_CLASS'] = 'FOCUS_NEUTRAL'
    else:
        row['Y_Team_FOCUS_CLASS'] = 'FOCUS_TOP_EARLY' if p_top > 0.5 else 'FOCUS_BOT_EARLY'

def derive_team_labels(team_id, obj, towers, fights, cluster):
    candidates = []
    if obj[team_id]['first_dragon'] is not None and obj[team_id]['first_dragon'] <= 14 * 60:
        candidates.append(('DRAGON', obj[team_id]['first_dragon']))
    if obj[team_id]['first_grubs'] is not None and obj[team_id]['first_grubs'] <= 14 * 60:
        candidates.append(('GRUBS', obj[team_id]['first_grubs']))
    if obj[team_id]['first_herald'] is not None and obj[team_id]['first_herald'] <= 14 * 60:
        candidates.append(('HERALD', obj[team_id]['first_herald']))
    first_major = 'NONE' if not candidates else min(candidates, key=lambda x: x[1])[0]

    n_drag = obj[team_id]['drag_0_14']
    n_grub = obj[team_id]['grubs_0_14']
    n_her = obj[team_id]['herald_0_14']
    total_obj = n_drag + n_grub + n_her

    if total_obj == 0:
        obj_prio = 'NONE'
    elif n_drag >= (n_grub + n_her) + 1:
        obj_prio = 'DRAGONS'
    elif (n_grub + n_her) >= n_drag + 1:
        obj_prio = 'GRUBS_HERALD'
    else:
        obj_prio = 'MIXED'

    f025 = fights["fights"]["0_25"][team_id]
    total_f = f025["total_fights"]
    pick_f = f025["pick_fights"]
    tf_f = f025["teamfight_fights"]
    sk_f = f025["skirmish_fights"]

    if total_f == 0:
        sk_style = "LOW_ACTION"
    elif tf_f >= pick_f + 2 and tf_f >= sk_f:
        sk_style = "TEAMFIGHT"
    elif pick_f >= tf_f + 2 and pick_f >= sk_f:
        sk_style = "PICK"
    else:
        sk_style = "MIXED"

    ff_time = fights["fights"]["0_14"][team_id]["first_fight_time"]
    ff_type = fights["fights"]["0_14"][team_id]["first_fight_type"]
    if ff_time is None or ff_time > 14 * 60 or ff_type is None:
        first_fight = "NONE"
    else:
        first_fight = ff_type if ff_type in ("PICK", "TEAMFIGHT") else "NONE"

    first_tower = towers[team_id]['0_14']['first_tower_time']
    kills_0_14 = fights['kills_0_14'][team_id]
    obj_0_14 = total_obj

    score = 0
    if first_tower is not None and first_tower <= 12 * 60:
        score += 1
    if obj_0_14 >= 2:
        score += 1
    if kills_0_14 >= 8:
        score += 1
    tempo = 'FAST' if score >= 2 else ('MID' if score == 1 else 'SLOW')

    gr15 = cluster.get('Team_GroupedRatio_15')
    if gr15 is None:
        grouping = 'MIXED'
    elif gr15 >= 0.70:
        grouping = 'GROUP'
    elif gr15 <= 0.55:
        grouping = 'SPLIT'
    else:
        grouping = 'MIXED'

    t14_25 = towers[team_id]['14_25']['towers']
    top_bot = t14_25['TOP'] + t14_25['BOT']
    mid = t14_25['MID']

    if gr15 is None:
        tower_plan = 'UNKNOWN'
    elif gr15 <= 0.55 and top_bot >= mid + 1:
        tower_plan = 'SPLIT_1_3_1'
    elif gr15 >= 0.70 and mid >= top_bot:
        tower_plan = 'SIEGE_GROUP'
    else:
        tower_plan = 'STANDARD'

    p0_14 = towers[team_id]['0_14']['plates']
    if (p0_14['TOP'] + p0_14['MID'] + p0_14['BOT']) == 0:
        lane_bias = 'NEUTRAL'
    else:
        vals = [('TOP', p0_14['TOP']), ('MID', p0_14['MID']), ('BOT', p0_14['BOT'])]
        vals.sort(key=lambda x: x[1], reverse=True)
        lane_bias = vals[0][0] if vals[0][1] >= vals[1][1] + 2 else 'NEUTRAL'

    snowball = 'YES' if ((first_tower is not None and first_tower <= 14 * 60) or obj_0_14 >= 2) else 'NO'

    return {
        'Label_FirstMajorObjective_0_14': first_major,
        'Label_ObjectivePriority_0_14': obj_prio,
        'Label_SkirmishStyle_0_25': sk_style,
        'Label_FirstFightType_0_14': first_fight,
        'Label_TowerPlan_14_25': tower_plan,
        'Label_LanePressureBias_0_14': lane_bias,
        'Label_Tempo_0_20': tempo,
        'Label_Grouping_15': grouping,
        'Label_Snowball_0_14': snowball,
    }


# -------------------------
# MATCH PROCESSING
# -------------------------

def process_match(match_id):
    try:
        details = riot_call_with_retry(watcher.match.by_id, REGION, match_id)
        timeline = riot_call_with_retry(watcher.match.timeline_by_match, REGION, match_id)
    except ApiError as err:
        if getattr(err.response, "status_code", None) == 404:
            print(f"Partida {match_id} no encontrada.")
        else:
            print(f"Error bajando {match_id}: {err}")
        return []

    info = details['info']
    duration_min = info.get('gameDuration', 0) / 60.0

    if duration_min < 15 or info.get('gameMode') != 'CLASSIC':
        return []

    teams_info = {t['teamId']: t for t in info['teams']}
    winning_team_id = 100 if teams_info.get(100, {}).get('win', False) else 200

    team_kills = {100: 0, 200: 0}
    for p in info['participants']:
        team_kills[p['teamId']] += p.get('kills', 0)

    time_minutes = [5, 10, 15, 20]
    snapshots = get_timeline_stats_by_minute(timeline, time_minutes)

    pos_minutes = [5, 10, 15, 20]
    positions = get_positions_by_minute(timeline, pos_minutes)

    pid_to_team = build_participant_to_team(info['participants'])
    obj = parse_objective_times(timeline, pid_to_team)
    towers = parse_tower_and_plate_events(timeline, pid_to_team)
    fights = parse_kills_and_fights(timeline, pid_to_team, fight_gap_s=25.0, end_min=25)

    pmap = {p['participantId']: p for p in info['participants']}
    rows_generated = []

    roles_order = ['TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY']

    for focus_team_id in (100, 200):
        did_win = 1 if focus_team_id == winning_team_id else 0

        row = {
            'matchId': match_id,
            'teamId': focus_team_id,
            'win': did_win,
            'meta_gameDurationMin': duration_min,
            'meta_gameVersion': info.get('gameVersion', None),
            'meta_queueId': info.get('queueId', None),
        }

        valid_team = True


        role_to_pid = {}
        for role in roles_order:
            ally_p = None
            enemy_p = None
            for p in pmap.values():
                if p.get('teamPosition') == role:
                    if p['teamId'] == focus_team_id:
                        ally_p = p
                    else:
                        enemy_p = p

            if not ally_p or not enemy_p:
                valid_team = False
                break

            prefix = role
            pid_ally = ally_p['participantId']
            pid_enemy = enemy_p['participantId']


            role_to_pid[role] = pid_ally
            # ---- PRE-GAME INPUTS (X_)
            row[f"X_{prefix}_AllyChampId"] = ally_p.get('championId')
            row[f"X_{prefix}_EnemyChampId"] = enemy_p.get('championId')

            s1 = ally_p.get('summoner1Id', None)
            s2 = ally_p.get('summoner2Id', None)
            if s1 is not None and s2 is not None:
                row[f"X_{prefix}_SummonerA"] = min(s1, s2)
                row[f"X_{prefix}_SummonerB"] = max(s1, s2)
                for k, v in encode_summoners_unordered(s1, s2).items():
                    row[f"X_{prefix}_{k}"] = v
            else:
                row[f"X_{prefix}_SummonerA"] = None
                row[f"X_{prefix}_SummonerB"] = None

            row[f"X_{prefix}_Keystone"] = extract_keystone(ally_p)

            # ---- IN-GAME (Post_)
            for m in time_minutes:
                s_ally = snapshots[m][pid_ally]
                s_enemy = snapshots[m][pid_enemy]
                row[f"Post_{prefix}_GoldDiff_{m}"] = s_ally['gold'] - s_enemy['gold']
                row[f"Post_{prefix}_XpDiff_{m}"] = s_ally['xp'] - s_enemy['xp']
                row[f"Post_{prefix}_CsDiff_{m}"] = s_ally['cs'] - s_enemy['cs']

            row[f"Post_{prefix}_EarlyTakedowns_15"] = snapshots[15][pid_ally]['takedowns']

            if role == 'JUNGLE':
                row[f"Post_{prefix}_JgCsBefore10"] = snapshots[10][pid_ally]['jungle_cs']
                row[f"Post_{prefix}_EnemyJgInvades"] = ally_p.get('challenges', {}).get('enemyJungleMonsterKills', 0)
            else:
                row[f"Post_{prefix}_LaneCsBefore10"] = snapshots[10][pid_ally]['lane_cs']

            # ---- Post-game analytics (Post_) - opcional pero prefijado
            stats = ally_p
            row[f"Post_{prefix}_DmgTotalPM"] = stats.get('totalDamageDealtToChampions', 0) / max(duration_min, 1e-6)
            row[f"Post_{prefix}_DmgTurretPM"] = stats.get('damageDealtToTurrets', 0) / max(duration_min, 1e-6)
            row[f"Post_{prefix}_DmgObjPM"] = stats.get('damageDealtToObjectives', 0) / max(duration_min, 1e-6)
            row[f"Post_{prefix}_VisionScorePM"] = stats.get('visionScore', 0) / max(duration_min, 1e-6)

            my_kills_assists = stats.get('kills', 0) + stats.get('assists', 0)
            total_team_kills = team_kills[focus_team_id]
            kp = (my_kills_assists / total_team_kills) if total_team_kills > 0 else 0.0
            row[f"Post_{prefix}_KillParticipation"] = round(kp, 3)

        if not valid_team:
            return []

        # ---- TEAM CLUSTER FEATURES (Feat_)
        for m in pos_minutes:
            cm = compute_team_cluster_metrics(positions[m], info['participants'], focus_team_id, grouped_radius=2200.0)
            row[f"Feat_Team_Spread_{m}"] = cm['spread_mean']
            row[f"Feat_Team_GroupedRatio_{m}"] = cm['grouped_ratio']
            row[f"Feat_Team_MaxPairDist_{m}"] = cm['max_pair_dist']

        
        # ---- PLAYER POSITION METRICS (Feat_) for 14-25 window
        pos_metrics = compute_player_group_side_metrics(timeline, info['participants'], focus_team_id,
                                                        start_min=14, end_min=25, near_dist=2000.0)
        for role in roles_order:
            pid = role_to_pid.get(role)
            if not pid:
                continue
            pm = pos_metrics.get(pid, {})
            row[f"Feat_{role}_SideRatio_14_25"] = pm.get('side_ratio')
            row[f"Feat_{role}_GroupRatio_14_25"] = pm.get('group_ratio')
            row[f"Feat_{role}_PosValidMin_14_25"] = pm.get('valid_minutes')

        # ---- PLAYER MVP LABELS (Y_)
        for role in roles_order:
            derive_player_mvp_labels(
                row,
                role,
                side_ratio=row.get(f"Feat_{role}_SideRatio_14_25"),
                group_ratio=row.get(f"Feat_{role}_GroupRatio_14_25"),
            )

        # ---- JUNGLE FOCUS (Feat_ + Label_)
        jg = compute_jungle_presence_and_focus(timeline, info['participants'], focus_team_id, start_min=2, end_min=14)
        row['Feat_Jungle_LanePresenceTop_2_14'] = jg['lane_top']
        row['Feat_Jungle_LanePresenceMid_2_14'] = jg['lane_mid']
        row['Feat_Jungle_LanePresenceBot_2_14'] = jg['lane_bot']
        row['Label_JungleFocusLane_2_14'] = jg['lane_focus']

        row['Feat_Jungle_TopsidePresence_2_14'] = jg['side_top']
        row['Feat_Jungle_BotsidePresence_2_14'] = jg['side_bot']
        row['Label_JungleMapSideFocus_2_14'] = jg['side_focus']

        # ---- TEAM EVENTS FEATURES (Feat_)
        row['Feat_FirstDragonTime_s'] = obj[focus_team_id]['first_dragon']
        row['Feat_FirstGrubsTime_s'] = obj[focus_team_id]['first_grubs']
        row['Feat_FirstHeraldTime_s'] = obj[focus_team_id]['first_herald']
        row['Feat_FirstTowerTime_s'] = towers[focus_team_id]['0_14']['first_tower_time']

        row['Feat_Dragons_0_14'] = obj[focus_team_id]['drag_0_14']
        row['Feat_Grubs_0_14'] = obj[focus_team_id]['grubs_0_14']
        row['Feat_Heralds_0_14'] = obj[focus_team_id]['herald_0_14']
        row['Feat_Objectives_0_14'] = row['Feat_Dragons_0_14'] + row['Feat_Grubs_0_14'] + row['Feat_Heralds_0_14']


        # ---- TEAM MVP LABEL (Y_): early map focus
        derive_team_focus_label(row)
        row['Feat_Kills_0_14'] = fights['kills_0_14'][focus_team_id]
        row['Feat_Kills_14_25'] = fights['kills_14_25'][focus_team_id]

        f014 = fights['fights']['0_14'][focus_team_id]
        row['Feat_Fights_0_14'] = f014['total_fights']
        row['Feat_PickRatio_0_14'] = f014['pick_ratio']
        row['Feat_TeamfightRatio_0_14'] = f014['teamfight_ratio']
        row['Feat_AvgKillsPerFight_0_14'] = f014['avg_kills_per_fight']
        row['Feat_AvgUniqParticipants_0_14'] = f014['avg_uniq_participants']
        row['Feat_FirstFightTime_0_14_s'] = f014['first_fight_time']
        row['Feat_FirstFightType_0_14'] = f014['first_fight_type']

        f1425 = fights['fights']['14_25'][focus_team_id]
        row['Feat_Fights_14_25'] = f1425['total_fights']
        row['Feat_PickRatio_14_25'] = f1425['pick_ratio']
        row['Feat_TeamfightRatio_14_25'] = f1425['teamfight_ratio']
        row['Feat_AvgKillsPerFight_14_25'] = f1425['avg_kills_per_fight']
        row['Feat_AvgUniqParticipants_14_25'] = f1425['avg_uniq_participants']

        f025 = fights['fights']['0_25'][focus_team_id]
        row['Feat_Fights_0_25'] = f025['total_fights']
        row['Feat_PickRatio_0_25'] = f025['pick_ratio']
        row['Feat_TeamfightRatio_0_25'] = f025['teamfight_ratio']
        row['Feat_AvgKillsPerFight_0_25'] = f025['avg_kills_per_fight']
        row['Feat_AvgUniqParticipants_0_25'] = f025['avg_uniq_participants']

        for lane in ('TOP', 'MID', 'BOT'):
            row[f"Feat_TowerKills_{lane}_0_14"] = towers[focus_team_id]['0_14']['towers'][lane]
            row[f"Feat_TowerKills_{lane}_14_25"] = towers[focus_team_id]['14_25']['towers'][lane]
            row[f"Feat_Plates_{lane}_0_14"] = towers[focus_team_id]['0_14']['plates'][lane]

        # ---- LABELS (Y)
        cluster_ctx = {'Team_GroupedRatio_15': row.get('Feat_Team_GroupedRatio_15')}
        row.update(derive_team_labels(focus_team_id, obj, towers, fights, cluster_ctx))

        # ---- Team totals (Post_)
        t_stats = next(t for t in info['teams'] if t['teamId'] == focus_team_id)
        objs = t_stats.get('objectives', {})

        row['Post_Team_Dragons'] = objs.get('dragon', {}).get('kills', 0)
        row['Post_Team_Barons'] = objs.get('baron', {}).get('kills', 0)
        row['Post_Team_Towers'] = objs.get('tower', {}).get('kills', 0)
        row['Post_Team_Inhibitors'] = objs.get('inhibitor', {}).get('kills', 0)
        row['Post_Team_RiftHeralds'] = objs.get('riftHerald', {}).get('kills', 0)
        row['Post_Team_VoidGrubs'] = objs.get('horde', {}).get('kills', 0)

        sample = next((p for p in info['participants'] if p['teamId'] == focus_team_id), None)
        row['Post_Team_ElderDragons'] = sample.get('challenges', {}).get('teamElderDragonKills', 0) if sample else 0

        rows_generated.append(row)

    return rows_generated


if __name__ == '__main__':
    try:
        acc = riot_call_with_retry(account_watcher.account.by_riot_id, REGION, GAME_NAME, TAG_LINE)
        target_puuid = acc.get('puuid')
    except ApiError as err:
        print(f"ERROR: No se pudo encontrar el PUUID para {GAME_NAME}#{TAG_LINE}: {err}")
        target_puuid = None

    if target_puuid:
        print(f"Iniciando prueba de extracción para {GAME_NAME}#{TAG_LINE}...")
        match_ids = get_match_ids(target_puuid, count=10, ranked_only=True)

        dataset = []
        for i, m in enumerate(match_ids):
            print(f"[{i + 1}/{len(match_ids)}] ", end='')
            rows = process_match(m)
            if rows:
                dataset.extend(rows)
            time.sleep(1.3)

        if dataset:
            os.makedirs('Data', exist_ok=True)
            pd.DataFrame(dataset).to_csv('Data/dataset_test_v4.csv', index=False)
            print('Prueba finalizada con éxito.')
    else:
        print('No hay PUUID; abortando.')
