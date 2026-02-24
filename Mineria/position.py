import math

from ..Raw_Data.api import (
    STYLE_NEAR_DIST,
    STYLE_NEAR_ALLIES,
    SUP_ROAM_START_MIN,
    SUP_ROAM_END_MIN,
)

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
    pid_to_team = {p['participantId']: p['teamId'] for p in participants}
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


def compute_support_roam_ratio(timeline, participants_info, team_id, start_min=SUP_ROAM_START_MIN, end_min=SUP_ROAM_END_MIN):
    """Compute how much the support leaves bot lane in early window.

    Returns (roam_ratio, bot_ratio, valid_minutes).
      - roam_ratio: fraction of valid minutes where support is NOT in BOT lane
      - bot_ratio: fraction of valid minutes in BOT lane
    """
    sup_pid = None
    for p in participants_info:
        if p["teamId"] == team_id and p.get("teamPosition") == "UTILITY":
            sup_pid = p["participantId"]
            break
    if sup_pid is None:
        return 0.0, 0.0, 0

    frames = timeline["info"]["frames"]
    max_i = len(frames) - 1
    s = max(0, min(start_min, max_i))
    e = max(0, min(end_min, max_i))

    bot = 0
    valid = 0
    for i in range(s, e + 1):
        pf = frames[i]["participantFrames"].get(str(sup_pid))
        if not pf:
            continue
        pos = _safe_pos(pf)
        if pos is None:
            continue
        lane = _lane_by_position_xy(pos[0], pos[1])
        valid += 1
        if lane == "BOT":
            bot += 1

    if valid <= 0:
        return 0.0, 0.0, 0
    bot_ratio = bot / valid
    roam_ratio = 1.0 - bot_ratio
    return roam_ratio, bot_ratio, valid


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
