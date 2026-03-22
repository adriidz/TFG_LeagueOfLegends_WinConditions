import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

try:
    from Raw_Datamining.raw_store import RAW_ROOT_DEFAULT
except Exception:
    RAW_ROOT_DEFAULT = "Data_raw/raw"

CANONICAL_ROLES = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
EMPTY_ROLE_VALUES = {None, "", "INVALID", "NONE"}
DEFAULT_REGION = os.getenv("REGION", "europe")
DEFAULT_OUTPUT_DIR = os.path.join("Data_clean", "qc_reports")


@dataclass
class TeamRoleSummary:
    team_id: int
    participant_count: int
    role_counts: Counter
    missing_roles: List[str]
    duplicate_roles: List[str]
    unexpected_roles: List[str]
    empty_role_count: int
    perfect: bool


@dataclass
class MatchAnalysis:
    match_id: str
    path: str
    ok_json: bool
    error: Optional[str]
    duration_seconds: Optional[float]
    duration_minutes: Optional[float]
    queue_id: Optional[int]
    game_version: Optional[str]
    team_summaries: Dict[int, TeamRoleSummary]
    both_teams_perfect: bool
    usable_basic: bool


def normalize_duration_seconds(raw_value, game_creation=None, game_end_timestamp=None) -> Optional[float]:
    """Return game duration in seconds, handling both sec and ms representations."""
    if raw_value is not None:
        try:
            value = float(raw_value)
            if value > 100000:  # defensivo: si viene en ms
                value /= 1000.0
            if value >= 0:
                return value
        except Exception:
            pass

    try:
        if game_creation is not None and game_end_timestamp is not None:
            return max(0.0, (float(game_end_timestamp) - float(game_creation)) / 1000.0)
    except Exception:
        pass
    return None


def summarize_team_roles(participants: List[dict], team_id: int) -> TeamRoleSummary:
    team_participants = [p for p in participants if p.get("teamId") == team_id]
    normalized_roles: List[Optional[str]] = []
    for p in team_participants:
        raw = p.get("teamPosition")
        if isinstance(raw, str):
            raw = raw.strip().upper()
        normalized_roles.append(raw)

    role_counts = Counter(r for r in normalized_roles if r not in EMPTY_ROLE_VALUES)
    empty_role_count = sum(1 for r in normalized_roles if r in EMPTY_ROLE_VALUES)

    missing_roles = [role for role in CANONICAL_ROLES if role_counts.get(role, 0) == 0]
    duplicate_roles = [role for role in CANONICAL_ROLES if role_counts.get(role, 0) > 1]
    unexpected_roles = sorted(role for role in role_counts.keys() if role not in CANONICAL_ROLES)

    perfect = (
        len(team_participants) == 5
        and all(role_counts.get(role, 0) == 1 for role in CANONICAL_ROLES)
        and not unexpected_roles
        and empty_role_count == 0
    )

    return TeamRoleSummary(
        team_id=team_id,
        participant_count=len(team_participants),
        role_counts=role_counts,
        missing_roles=missing_roles,
        duplicate_roles=duplicate_roles,
        unexpected_roles=unexpected_roles,
        empty_role_count=empty_role_count,
        perfect=perfect,
    )


def analyze_match(match_path: str, min_duration_minutes: float) -> MatchAnalysis:
    match_id = os.path.basename(os.path.dirname(match_path))
    try:
        with open(match_path, "r", encoding="utf-8") as f:
            match = json.load(f)
    except Exception as exc:
        return MatchAnalysis(
            match_id=match_id,
            path=match_path,
            ok_json=False,
            error=f"{type(exc).__name__}: {exc}",
            duration_seconds=None,
            duration_minutes=None,
            queue_id=None,
            game_version=None,
            team_summaries={},
            both_teams_perfect=False,
            usable_basic=False,
        )

    info = (match or {}).get("info") or {}
    metadata = (match or {}).get("metadata") or {}
    match_id = metadata.get("matchId") or match_id
    participants = info.get("participants") or []

    duration_seconds = normalize_duration_seconds(
        info.get("gameDuration"),
        info.get("gameCreation"),
        info.get("gameEndTimestamp"),
    )
    duration_minutes = duration_seconds / 60.0 if duration_seconds is not None else None

    team_ids = sorted({p.get("teamId") for p in participants if p.get("teamId") is not None})
    if not team_ids:
        team_ids = [100, 200]

    team_summaries = {team_id: summarize_team_roles(participants, team_id) for team_id in team_ids}
    both_teams_perfect = bool(team_summaries) and all(ts.perfect for ts in team_summaries.values())

    usable_basic = (
        duration_minutes is not None
        and duration_minutes >= min_duration_minutes
        and both_teams_perfect
    )

    return MatchAnalysis(
        match_id=match_id,
        path=match_path,
        ok_json=True,
        error=None,
        duration_seconds=duration_seconds,
        duration_minutes=duration_minutes,
        queue_id=info.get("queueId"),
        game_version=info.get("gameVersion"),
        team_summaries=team_summaries,
        both_teams_perfect=both_teams_perfect,
        usable_basic=usable_basic,
    )


def find_match_json_paths(raw_root: str, region: str) -> List[str]:
    base = os.path.join(raw_root, region)
    if not os.path.isdir(base):
        raise SystemExit(f"No existe la carpeta RAW: {base}")

    paths: List[str] = []
    for match_id in os.listdir(base):
        mdir = os.path.join(base, match_id)
        if not os.path.isdir(mdir):
            continue
        match_path = os.path.join(mdir, "match.json")
        if os.path.exists(match_path):
            paths.append(match_path)
    return paths


def percentile(sorted_values: List[float], p: float) -> float:
    idx = max(0, min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * p))))
    return sorted_values[idx]


def pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def format_counter(counter: Counter) -> str:
    if not counter:
        return "{}"
    items = ", ".join(f"{k}: {v}" for k, v in sorted(counter.items()))
    return "{" + items + "}"


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_duration_histogram(
    durations: List[float],
    out_path: str,
    cutoff_min: float,
    zoom_max: Optional[float] = None,
) -> None:
    if not durations:
        return

    plt.figure(figsize=(10, 6))
    bins = 80 if zoom_max is None else 50
    plot_values = durations if zoom_max is None else [x for x in durations if x <= zoom_max]

    plt.hist(plot_values, bins=bins)
    plt.axvline(cutoff_min, linestyle="--", linewidth=2, label=f"corte {cutoff_min:.0f} min")
    plt.xlabel("Duración de la partida (minutos)")
    plt.ylabel("Número de partidas")

    title = "Distribución de gameDuration"
    if zoom_max is not None:
        title += f" (zoom 0–{zoom_max:.0f} min)"
    plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_counter(
    counter: Counter,
    out_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
    ordered_keys: Optional[List[str]] = None,
) -> None:
    if not counter:
        return

    if ordered_keys is None:
        keys = list(counter.keys())
    else:
        keys = [k for k in ordered_keys if k in counter] + [k for k in counter.keys() if k not in ordered_keys]

    values = [counter[k] for k in keys]

    plt.figure(figsize=(10, 6))
    plt.bar(keys, values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Control de calidad y sanity checks sobre match.json")
    parser.add_argument("--raw-root", default=os.getenv("RAW_ROOT", RAW_ROOT_DEFAULT))
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-duration-minutes", type=float, default=15.0)
    parser.add_argument("--show-bad-json", type=int, default=10, help="Número máximo de bad json a mostrar")
    parser.add_argument("--show-role-issue-samples", type=int, default=10, help="Número máximo de partidas problemáticas a mostrar")
    args = parser.parse_args()
    ensure_dir(args.out_dir)

    match_paths = find_match_json_paths(args.raw_root, args.region)
    total_dirs = len(match_paths)

    analyses: List[MatchAnalysis] = []
    bad_json_sample: List[Tuple[str, str]] = []
    durations: List[float] = []
    role_counts_global: Counter = Counter()
    team_perfect_counter: Counter = Counter()
    missing_role_counter: Counter = Counter()
    duplicate_role_counter: Counter = Counter()
    unexpected_role_counter: Counter = Counter()
    empty_role_teams = 0
    teams_with_duplicate_jungle = 0
    teams_with_missing_utility = 0
    duration_buckets = {"lt_5": 0, "lt_10": 0, "lt_12": 0, "lt_15": 0, "lt_20": 0}
    short_duration_sample: List[Tuple[str, float]] = []
    non_perfect_match_sample: List[str] = []
    duplicate_jungle_sample: List[str] = []
    missing_utility_sample: List[str] = []

    for match_path in match_paths:
        analysis = analyze_match(match_path, args.min_duration_minutes)
        analyses.append(analysis)

        if not analysis.ok_json:
            if len(bad_json_sample) < args.show_bad_json:
                bad_json_sample.append((analysis.match_id, analysis.error or ""))
            continue

        if analysis.duration_minutes is not None:
            durations.append(analysis.duration_minutes)
            if analysis.duration_minutes < 5:
                duration_buckets["lt_5"] += 1
            if analysis.duration_minutes < 10:
                duration_buckets["lt_10"] += 1
            if analysis.duration_minutes < 12:
                duration_buckets["lt_12"] += 1
            if analysis.duration_minutes < 15:
                duration_buckets["lt_15"] += 1
            if analysis.duration_minutes < 20:
                duration_buckets["lt_20"] += 1
            if analysis.duration_minutes < args.min_duration_minutes and len(short_duration_sample) < args.show_role_issue_samples:
                short_duration_sample.append((analysis.match_id, analysis.duration_minutes))

        if not analysis.both_teams_perfect and len(non_perfect_match_sample) < args.show_role_issue_samples:
            non_perfect_match_sample.append(analysis.match_id)

        for team_summary in analysis.team_summaries.values():
            for role, count in team_summary.role_counts.items():
                role_counts_global[role] += count

            team_perfect_counter["perfect"] += int(team_summary.perfect)
            team_perfect_counter["not_perfect"] += int(not team_summary.perfect)

            for role in team_summary.missing_roles:
                missing_role_counter[role] += 1
            for role in team_summary.duplicate_roles:
                duplicate_role_counter[role] += 1
            for role in team_summary.unexpected_roles:
                unexpected_role_counter[role] += 1

            if team_summary.empty_role_count > 0:
                empty_role_teams += 1

            if "JUNGLE" in team_summary.duplicate_roles:
                teams_with_duplicate_jungle += 1
                if len(duplicate_jungle_sample) < args.show_role_issue_samples:
                    duplicate_jungle_sample.append(analysis.match_id)

            if "UTILITY" in team_summary.missing_roles:
                teams_with_missing_utility += 1
                if len(missing_utility_sample) < args.show_role_issue_samples:
                    missing_utility_sample.append(analysis.match_id)

    good_json = sum(1 for a in analyses if a.ok_json)
    bad_json = total_dirs - good_json
    both_teams_perfect = sum(1 for a in analyses if a.both_teams_perfect)
    usable_basic_count = sum(1 for a in analyses if a.usable_basic)
    total_team_summaries = sum(len(a.team_summaries) for a in analyses if a.ok_json)

    print("=== QC SANITY CHECKS ===")
    print(f"RAW base:              {os.path.join(args.raw_root, args.region)}")
    print(f"Match dirs:            {total_dirs}")
    print(f"Good json:             {good_json}")
    print(f"Bad json:              {bad_json}")
    print(f"Bad json sample:       {bad_json_sample}")

    print_section("Duración de la partida")
    if durations:
        durations_sorted = sorted(durations)
        print(f"min    = {min(durations):.2f}")
        print(f"p05    = {percentile(durations_sorted, 0.05):.2f}")
        print(f"p25    = {percentile(durations_sorted, 0.25):.2f}")
        print(f"median = {percentile(durations_sorted, 0.50):.2f}")
        print(f"p75    = {percentile(durations_sorted, 0.75):.2f}")
        print(f"p95    = {percentile(durations_sorted, 0.95):.2f}")
        print(f"max    = {max(durations):.2f}")
        print(f"mean   = {sum(durations) / len(durations):.2f}")
    print(f"<5 min                     = {duration_buckets['lt_5']}")
    print(f"<10 min                    = {duration_buckets['lt_10']}")
    print(f"<12 min                    = {duration_buckets['lt_12']}")
    print(f"<15 min                    = {duration_buckets['lt_15']}")
    print(f"<20 min                    = {duration_buckets['lt_20']}")
    print(f"Sample <{args.min_duration_minutes:.1f} min = {short_duration_sample}")

    print_section("Integridad de roles")
    print(f"Teams analyzed:                {total_team_summaries}")
    print(f"Perfect teams:                 {team_perfect_counter['perfect']} ({pct(team_perfect_counter['perfect'], total_team_summaries):.2f}%)")
    print(f"Matches with both teams clean: {both_teams_perfect} ({pct(both_teams_perfect, good_json):.2f}%)")
    print(f"Teams with empty teamPosition: {empty_role_teams}")
    print(f"Teams with duplicate JUNGLE:   {teams_with_duplicate_jungle}")
    print(f"Teams missing UTILITY:         {teams_with_missing_utility}")
    print(f"Global teamPosition counts:    {format_counter(role_counts_global)}")
    print(f"Missing roles by team:         {format_counter(missing_role_counter)}")
    print(f"Duplicate roles by team:       {format_counter(duplicate_role_counter)}")
    if unexpected_role_counter:
        print(f"Unexpected roles by team:      {format_counter(unexpected_role_counter)}")
    print(f"Sample non-perfect matches:    {non_perfect_match_sample}")
    print(f"Sample duplicate JUNGLE:       {duplicate_jungle_sample}")
    print(f"Sample missing UTILITY:        {missing_utility_sample}")

    print_section("Filtro básico sugerido")
    print(
        "Criterio: good_json + "
        f"duration>={args.min_duration_minutes:.1f} + roles perfectos en ambos equipos"
    )
    print(f"Usable matches: {usable_basic_count} / {good_json} ({pct(usable_basic_count, good_json):.2f}%)")

    plot_duration_histogram(
        durations,
        os.path.join(args.out_dir, "duration_histogram.png"),
        cutoff_min=args.min_duration_minutes,
    )

    plot_duration_histogram(
        durations,
        os.path.join(args.out_dir, "duration_histogram_zoom_0_30.png"),
        cutoff_min=args.min_duration_minutes,
        zoom_max=30.0,
    )

    plot_counter(
        role_counts_global,
        os.path.join(args.out_dir, "team_position_counts.png"),
        title="Conteo global de teamPosition",
        xlabel="teamPosition",
        ylabel="Frecuencia",
        ordered_keys=list(CANONICAL_ROLES),
    )

    plot_counter(
        missing_role_counter,
        os.path.join(args.out_dir, "missing_roles_by_team.png"),
        title="Roles ausentes por equipo",
        xlabel="Rol",
        ylabel="Número de equipos",
        ordered_keys=list(CANONICAL_ROLES),
    )

    plot_counter(
        duplicate_role_counter,
        os.path.join(args.out_dir, "duplicate_roles_by_team.png"),
        title="Roles duplicados por equipo",
        xlabel="Rol",
        ylabel="Número de equipos",
        ordered_keys=list(CANONICAL_ROLES),
    )


if __name__ == "__main__":
    main()
