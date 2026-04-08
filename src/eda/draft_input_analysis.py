import os
import json
import math
import argparse
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

DEFAULT_RAW_ROOT = os.path.join("data/raw", "raw")
DEFAULT_REGION = "europe"
DEFAULT_OUT_DIR = os.path.join("data/clean", "draft_input_reports")
CANONICAL_ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
BLUE_TEAM_ID = 100
RED_TEAM_ID = 200


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def list_match_dirs(base: str) -> List[str]:
    if not os.path.isdir(base):
        raise SystemExit(f"No existe el directorio RAW: {base}")
    out = []
    for name in os.listdir(base):
        mdir = os.path.join(base, name)
        if os.path.isdir(mdir):
            out.append(mdir)
    return out


def load_match_json(match_dir: str) -> dict:
    path = os.path.join(match_dir, "match.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_info(match: dict) -> dict:
    return (match or {}).get("info", {}) or {}


def game_duration_minutes(info: dict) -> Optional[float]:
    dur = info.get("gameDuration")
    if dur is None:
        return None
    try:
        dur = float(dur)
    except Exception:
        return None

    # Match-V5 suele venir en segundos. Si aparece un valor enorme en ms, lo normalizamos.
    if dur > 100000:
        return dur / 1000.0 / 60.0
    if dur > 1000:
        return dur / 60.0
    return dur / 60.0 if dur > 100 else dur


# Algunos parches / regiones han usado nombres distintos para las void grubs.
VOID_GRUB_KEYS = (
    "horde",
    "voidGrubs",
    "void_grubs",
    "voidgrubs",
    "grubs",
)


def get_team_map(info: dict) -> Dict[int, dict]:
    teams = info.get("teams") or []
    out = {}
    for team in teams:
        team_id = team.get("teamId")
        if team_id is not None:
            out[int(team_id)] = team
    return out


def get_participants(info: dict) -> List[dict]:
    return list(info.get("participants") or [])


def has_perfect_roles(info: dict) -> bool:
    participants = get_participants(info)
    for team_id in (BLUE_TEAM_ID, RED_TEAM_ID):
        roles = [
            (p.get("teamPosition") or "").strip().upper()
            for p in participants
            if p.get("teamId") == team_id
        ]
        counts = Counter(r for r in roles if r)
        if set(counts.keys()) != set(CANONICAL_ROLES):
            return False
        if any(counts.get(role, 0) != 1 for role in CANONICAL_ROLES):
            return False
    return True


def get_side_win(info: dict) -> Optional[str]:
    teams = get_team_map(info)
    blue = teams.get(BLUE_TEAM_ID)
    red = teams.get(RED_TEAM_ID)
    if not blue or not red:
        return None
    blue_win = blue.get("win")
    red_win = red.get("win")
    if blue_win is True and red_win is False:
        return "BLUE"
    if red_win is True and blue_win is False:
        return "RED"
    if blue_win is True:
        return "BLUE"
    if red_win is True:
        return "RED"
    return None


def get_first_objective_side(info: dict, objective_key: str) -> Optional[str]:
    teams = get_team_map(info)
    blue = teams.get(BLUE_TEAM_ID) or {}
    red = teams.get(RED_TEAM_ID) or {}
    blue_obj = (((blue.get("objectives") or {}).get(objective_key)) or {})
    red_obj = (((red.get("objectives") or {}).get(objective_key)) or {})
    blue_first = blue_obj.get("first")
    red_first = red_obj.get("first")

    if blue_first is True and red_first is not True:
        return "BLUE"
    if red_first is True and blue_first is not True:
        return "RED"
    return None


def get_first_void_grubs_side(info: dict) -> Optional[str]:
    for key in VOID_GRUB_KEYS:
        side = get_first_objective_side(info, key)
        if side is not None:
            return side
    return None


def extract_champion_names(info: dict) -> List[str]:
    champs = []
    for p in get_participants(info):
        name = p.get("championName") or p.get("championId")
        if name is not None:
            champs.append(str(name))
    return champs


def extract_team_draft(info: dict, team_id: int) -> Tuple[str, ...]:
    """
    Devuelve el draft de un equipo como tupla ordenada alfabéticamente de 5 campeones.
    Lo ordenamos para medir composición, no orden de pick.
    """
    champs = []
    for p in get_participants(info):
        if p.get("teamId") == team_id:
            name = p.get("championName") or p.get("championId")
            if name is not None:
                champs.append(str(name))
    return tuple(sorted(champs))


def extract_match_draft(info: dict) -> Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """
    Devuelve el champ select de la partida como:
        (draft_blue, draft_red)
    siendo cada draft una tupla de 5 campeones ordenados alfabéticamente.

    Esto hace la representación dependiente del lado (side-aware),
    que suele ser lo más correcto en LoL.
    """
    blue_draft = extract_team_draft(info, BLUE_TEAM_ID)
    red_draft = extract_team_draft(info, RED_TEAM_ID)

    if len(blue_draft) != 5 or len(red_draft) != 5:
        return None
    return (blue_draft, red_draft)


def format_team_draft(draft: Tuple[str, ...]) -> str:
    return " | ".join(draft)


def format_match_draft(match_draft: Tuple[Tuple[str, ...], Tuple[str, ...]]) -> str:
    blue_draft, red_draft = match_draft
    return f"BLUE[{format_team_draft(blue_draft)}]  vs  RED[{format_team_draft(red_draft)}]"


def pct(n: int, d: int) -> Optional[float]:
    if d == 0:
        return None
    return 100.0 * n / d


def fmt_pct(n: int, d: int) -> str:
    if d == 0:
        return "n/a"
    return f"{100.0 * n / d:.2f}%"


def plot_top_champions(counter: Counter, out_path: str, top_n: int) -> None:
    if not counter:
        return
    items = counter.most_common(top_n)
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(14, 7))
    plt.bar(labels, values)
    plt.title(f"Campeones más pickeados (top {top_n})")
    plt.xlabel("Campeón")
    plt.ylabel("Número de picks")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_rank_frequency(counter: Counter, out_path: str) -> None:
    if not counter:
        return
    freqs = sorted(counter.values(), reverse=True)
    ranks = list(range(1, len(freqs) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(ranks, freqs, marker="o", markersize=3)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Distribución rank-frequency de picks de campeones")
    plt.xlabel("Rango del campeón (1 = más pickeado)")
    plt.ylabel("Frecuencia de picks")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_side_rates(values: Dict[str, float], out_path: str, title: str, ylabel: str) -> None:
    labels = ["BLUE", "RED"]
    ys = [values.get("BLUE", 0.0), values.get("RED", 0.0)]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, ys)
    plt.ylim(0, 100)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def print_header(title: str) -> None:
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))


def summarize_long_tail(champion_counter: Counter) -> None:
    total_picks = sum(champion_counter.values())
    unique_champs = len(champion_counter)
    top_10 = champion_counter.most_common(10)
    top_20 = champion_counter.most_common(20)
    top_10_picks = sum(v for _, v in top_10)
    top_20_picks = sum(v for _, v in top_20)

    print_header("PICKS DE CAMPEONES")
    print(f"Campeones únicos observados: {unique_champs}")
    print(f"Picks totales contabilizados: {total_picks}")
    print(f"Cobertura top-10: {top_10_picks} picks ({fmt_pct(top_10_picks, total_picks)})")
    print(f"Cobertura top-20: {top_20_picks} picks ({fmt_pct(top_20_picks, total_picks)})")
    print("Top 20 campeones por frecuencia:")
    for i, (champ, count) in enumerate(top_20, start=1):
        print(f"  {i:>2}. {champ:<20} {count}")

    rare_5 = sum(1 for _, c in champion_counter.items() if c < 5)
    rare_10 = sum(1 for _, c in champion_counter.items() if c < 10)
    rare_25 = sum(1 for _, c in champion_counter.items() if c < 25)
    print()
    print(f"Campeones con menos de 5 picks:  {rare_5}")
    print(f"Campeones con menos de 10 picks: {rare_10}")
    print(f"Campeones con menos de 25 picks: {rare_25}")


def summarize_team_drafts(team_draft_counter: Counter) -> None:
    total_team_drafts = sum(team_draft_counter.values())
    unique_team_drafts = len(team_draft_counter)

    print_header("FRECUENCIA DE DRAFTS ÚNICOS POR EQUIPO")
    print(f"Drafts de equipo observados: {total_team_drafts}")
    print(f"Drafts únicos de equipo: {unique_team_drafts}")
    print(f"Porcentaje de unicidad: {fmt_pct(unique_team_drafts, total_team_drafts)}")

    singleton = sum(1 for _, c in team_draft_counter.items() if c == 1)
    rare_3 = sum(1 for _, c in team_draft_counter.items() if c < 3)
    rare_5 = sum(1 for _, c in team_draft_counter.items() if c < 5)

    print(f"Drafts de equipo vistos 1 sola vez: {singleton} ({fmt_pct(singleton, unique_team_drafts)})")
    print(f"Drafts de equipo con frecuencia < 3: {rare_3} ({fmt_pct(rare_3, unique_team_drafts)})")
    print(f"Drafts de equipo con frecuencia < 5: {rare_5} ({fmt_pct(rare_5, unique_team_drafts)})")

    print("\nTop 15 drafts de equipo más frecuentes:")
    for i, (draft, count) in enumerate(team_draft_counter.most_common(15), start=1):
        print(f"  {i:>2}. [{count:>6}] {format_team_draft(draft)}")


def summarize_match_drafts(match_draft_counter: Counter) -> None:
    total_match_drafts = sum(match_draft_counter.values())
    unique_match_drafts = len(match_draft_counter)

    print_header("FRECUENCIA DE CHAMP SELECTS ÚNICOS POR PARTIDA")
    print(f"Champ selects observados: {total_match_drafts}")
    print(f"Champ selects únicos: {unique_match_drafts}")
    print(f"Porcentaje de unicidad: {fmt_pct(unique_match_drafts, total_match_drafts)}")

    singleton = sum(1 for _, c in match_draft_counter.items() if c == 1)
    rare_3 = sum(1 for _, c in match_draft_counter.items() if c < 3)
    rare_5 = sum(1 for _, c in match_draft_counter.items() if c < 5)

    print(f"Champ selects vistos 1 sola vez: {singleton} ({fmt_pct(singleton, unique_match_drafts)})")
    print(f"Champ selects con frecuencia < 3: {rare_3} ({fmt_pct(rare_3, unique_match_drafts)})")
    print(f"Champ selects con frecuencia < 5: {rare_5} ({fmt_pct(rare_5, unique_match_drafts)})")

    print("\nTop 10 champ selects más frecuentes:")
    for i, (draft, count) in enumerate(match_draft_counter.most_common(10), start=1):
        print(f"  {i:>2}. [{count:>6}] {format_match_draft(draft)}")


def print_side_table(title: str, counts: Dict[str, int], denom: int) -> None:
    print_header(title)
    blue = counts.get("BLUE", 0)
    red = counts.get("RED", 0)
    print(f"BLUE: {blue} ({fmt_pct(blue, denom)})")
    print(f"RED:  {red} ({fmt_pct(red, denom)})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis del espacio de entrada del draft desde match.json")
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-duration-minutes", type=float, default=15.0)
    parser.add_argument("--require-perfect-roles", action="store_true")
    parser.add_argument("--top-n", type=int, default=150)
    args = parser.parse_args()

    base = os.path.join(args.raw_root, args.region)
    ensure_dir(args.out_dir)
    match_dirs = list_match_dirs(base)

    total_seen = 0
    total_kept = 0
    bad_json = 0
    missing_teams = 0
    filtered_short = 0
    filtered_roles = 0

    champion_counter = Counter()
    team_draft_counter = Counter()
    match_draft_counter = Counter()

    side_wins = Counter()
    first_blood_side = Counter()
    first_dragon_side = Counter()
    first_grubs_side = Counter()

    matches_with_fb = 0
    matches_with_dragon = 0
    matches_with_grubs = 0

    for mdir in match_dirs:
        total_seen += 1
        try:
            match = load_match_json(mdir)
        except Exception:
            bad_json += 1
            continue

        info = get_info(match)
        if not info:
            bad_json += 1
            continue

        teams = get_team_map(info)
        if BLUE_TEAM_ID not in teams or RED_TEAM_ID not in teams:
            missing_teams += 1
            continue

        dur_min = game_duration_minutes(info)
        if dur_min is None or dur_min < args.min_duration_minutes:
            filtered_short += 1
            continue

        if args.require_perfect_roles and not has_perfect_roles(info):
            filtered_roles += 1
            continue

        total_kept += 1

        champion_counter.update(extract_champion_names(info))

        blue_team_draft = extract_team_draft(info, BLUE_TEAM_ID)
        red_team_draft = extract_team_draft(info, RED_TEAM_ID)
        if len(blue_team_draft) == 5:
            team_draft_counter[blue_team_draft] += 1
        if len(red_team_draft) == 5:
            team_draft_counter[red_team_draft] += 1

        match_draft = extract_match_draft(info)
        if match_draft is not None:
            match_draft_counter[match_draft] += 1

        winner = get_side_win(info)
        if winner is not None:
            side_wins[winner] += 1

        fb_side = get_first_objective_side(info, "champion")
        if fb_side is not None:
            first_blood_side[fb_side] += 1
            matches_with_fb += 1

        dragon_side = get_first_objective_side(info, "dragon")
        if dragon_side is not None:
            first_dragon_side[dragon_side] += 1
            matches_with_dragon += 1

        grubs_side = get_first_void_grubs_side(info)
        if grubs_side is not None:
            first_grubs_side[grubs_side] += 1
            matches_with_grubs += 1

    print_header("RESUMEN DEL ANÁLISIS DE FEATURES DEL DRAFT")
    print(f"RAW base: {base}")
    print(f"Partidas vistas: {total_seen}")
    print(f"Bad JSON / ilegibles: {bad_json}")
    print(f"Sin teams azul/rojo identificables: {missing_teams}")
    print(f"Filtradas por duración < {args.min_duration_minutes:.1f} min: {filtered_short}")
    if args.require_perfect_roles:
        print(f"Filtradas por roles no perfectos: {filtered_roles}")
    print(f"Partidas analizadas: {total_kept}")

    summarize_long_tail(champion_counter)
    summarize_team_drafts(team_draft_counter)
    summarize_match_drafts(match_draft_counter)

    matches_with_winner = side_wins.get("BLUE", 0) + side_wins.get("RED", 0)
    print_side_table("VICTORIA GLOBAL POR LADO", side_wins, matches_with_winner)
    if matches_with_winner:
        blue_wr = 100.0 * side_wins.get("BLUE", 0) / matches_with_winner
        red_wr = 100.0 * side_wins.get("RED", 0) / matches_with_winner
        print(f"Sesgo neto hacia BLUE: {blue_wr - red_wr:+.2f} puntos porcentuales")

    print_side_table("FIRST BLOOD POR LADO", first_blood_side, matches_with_fb)
    print_side_table("PRIMER DRAGÓN POR LADO", first_dragon_side, matches_with_dragon)
    print_side_table("PRIMERAS LARVAS / VOID GRUBS POR LADO", first_grubs_side, matches_with_grubs)

    win_rates = {
        "BLUE": pct(side_wins.get("BLUE", 0), matches_with_winner) or 0.0,
        "RED": pct(side_wins.get("RED", 0), matches_with_winner) or 0.0,
    }
    fb_rates = {
        "BLUE": pct(first_blood_side.get("BLUE", 0), matches_with_fb) or 0.0,
        "RED": pct(first_blood_side.get("RED", 0), matches_with_fb) or 0.0,
    }
    dragon_rates = {
        "BLUE": pct(first_dragon_side.get("BLUE", 0), matches_with_dragon) or 0.0,
        "RED": pct(first_dragon_side.get("RED", 0), matches_with_dragon) or 0.0,
    }
    grubs_rates = {
        "BLUE": pct(first_grubs_side.get("BLUE", 0), matches_with_grubs) or 0.0,
        "RED": pct(first_grubs_side.get("RED", 0), matches_with_grubs) or 0.0,
    }

    plot_top_champions(
        champion_counter,
        os.path.join(args.out_dir, "champion_pick_frequencies_topN.png"),
        top_n=args.top_n,
    )
    plot_rank_frequency(
        champion_counter,
        os.path.join(args.out_dir, "champion_pick_rank_frequency.png"),
    )
    plot_side_rates(
        win_rates,
        os.path.join(args.out_dir, "side_win_rate.png"),
        title="Porcentaje global de victoria por lado",
        ylabel="Win rate (%)",
    )
    plot_side_rates(
        fb_rates,
        os.path.join(args.out_dir, "side_first_blood_rate.png"),
        title="Porcentaje de First Blood por lado",
        ylabel="Partidas con First Blood (%)",
    )
    plot_side_rates(
        dragon_rates,
        os.path.join(args.out_dir, "side_first_dragon_rate.png"),
        title="Porcentaje de primer dragón por lado",
        ylabel="Partidas con primer dragón (%)",
    )
    plot_side_rates(
        grubs_rates,
        os.path.join(args.out_dir, "side_first_void_grubs_rate.png"),
        title="Porcentaje de primeras larvas por lado",
        ylabel="Partidas con primeras larvas (%)",
    )

    print_header("PNG GENERADOS")
    print(os.path.join(args.out_dir, "champion_pick_frequencies_topN.png"))
    print(os.path.join(args.out_dir, "champion_pick_rank_frequency.png"))
    print(os.path.join(args.out_dir, "side_win_rate.png"))
    print(os.path.join(args.out_dir, "side_first_blood_rate.png"))
    print(os.path.join(args.out_dir, "side_first_dragon_rate.png"))
    print(os.path.join(args.out_dir, "side_first_void_grubs_rate.png"))


if __name__ == "__main__":
    main()