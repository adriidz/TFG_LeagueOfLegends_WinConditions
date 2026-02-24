import math

from ..Raw_Data.api import (
    PWR_TAKEDOWN_GOLD,
    PWR_SIGMOID_SCALE,
    STYLE_SIDE_MARGIN,
    STYLE_SIGMOID_SCALE,
    JG_GANK_PROB_SCALE,
    JG_GANK_CLASS_THRESHOLD,
    SUP_ROAM_MIN_VALID,
    SUP_ROAM_PROB_SCALE,
    SUP_ROAM_CLASS_MARGIN,
)


def sigmoid(x: float) -> float:
    """Sigmoid numerically-stable-ish for moderate x."""
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


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
# PLAYER MVP LABELS
# -------------------------

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

    # --- Style: role-specific (only meaningful for TOP and MIDDLE in MVP)
    if role_prefix not in ("TOP", "MIDDLE"):
        row[f"Y_{role_prefix}_STYLE_SIDE_PROB"] = None
        row[f"Y_{role_prefix}_STYLE_CLASS"] = None
        return

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


def derive_jungle_gank_labels(row):
    """MVP label for Jungle: GANKY_EARLY vs FARMY_EARLY.

    Uses only already available post/feat columns:
      - Post_JUNGLE_EarlyTakedowns_15
      - Feat_Jungle_LanePresenceTop_2_14 / Mid / Bot (ratios in [0,1])
      - Feat_Jungle_TopsidePresence_2_14 / BotsidePresence_2_14
    """
    td15 = row.get("Post_JUNGLE_EarlyTakedowns_15") or 0.0
    td15 = float(td15)

    lane_top = float(row.get("Feat_Jungle_LanePresenceTop_2_14") or 0.0)
    lane_mid = float(row.get("Feat_Jungle_LanePresenceMid_2_14") or 0.0)
    lane_bot = float(row.get("Feat_Jungle_LanePresenceBot_2_14") or 0.0)

    top_side = float(row.get("Feat_Jungle_TopsidePresence_2_14") or 0.0)
    bot_side = float(row.get("Feat_Jungle_BotsidePresence_2_14") or 0.0)

    # Heuristics:
    # - More early takedowns -> more ganky
    # - More concentrated time in a lane (max lane ratio) -> more ganky
    # - Strong side commitment (abs(top-bot)) -> more ganky
    lane_conc = max(lane_top, lane_mid, lane_bot) - (1.0 / 3.0)
    side_commit = abs(top_side - bot_side)

    score = (td15 * 0.8) + (lane_conc * 2.0) + (side_commit * 2.0)
    p_gank = sigmoid((score - JG_GANK_CLASS_THRESHOLD) / JG_GANK_PROB_SCALE)

    row["Y_JUNGLE_GANKY_EARLY_PROB"] = p_gank
    row["Y_JUNGLE_STYLE_CLASS"] = "JG_GANKY_EARLY" if p_gank >= 0.5 else "JG_FARMY_EARLY"


def derive_support_roam_labels(row, roam_ratio, bot_ratio, valid_minutes):
    """MVP label for Support: ROAMY_EARLY vs LANE_EARLY."""
    row["Feat_UTILITY_RoamRatio_2_14"] = roam_ratio
    row["Feat_UTILITY_BotLaneRatio_2_14"] = bot_ratio
    row["Feat_UTILITY_PosValidMin_2_14"] = valid_minutes

    if valid_minutes < SUP_ROAM_MIN_VALID:
        row["Y_UTILITY_ROAMY_EARLY_PROB"] = None
        row["Y_UTILITY_STYLE_CLASS"] = None
        return

    # Center prob around a small margin: roam a bit more than staying bot
    delta = (roam_ratio - bot_ratio) - SUP_ROAM_CLASS_MARGIN
    p_roam = sigmoid(delta / SUP_ROAM_PROB_SCALE)

    row["Y_UTILITY_ROAMY_EARLY_PROB"] = p_roam
    row["Y_UTILITY_STYLE_CLASS"] = "SUP_ROAMY_EARLY" if p_roam >= 0.5 else "SUP_LANE_EARLY"


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
