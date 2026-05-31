#!/usr/bin/env python3
"""
11_qualitative_match_context.py -- Real match context for top label/model errors.

This complements frame-level label diagnostics with qualitative game evidence
from Riot match/timeline JSON: early kills, deaths, assists, objective events,
and final participant stats for the botlane involved in each top error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOP_ERRORS = REPO_ROOT / "final" / "analysis" / "error_analysis" / "top_abs_errors.csv"
DEFAULT_LABEL_SUMMARY = (
    REPO_ROOT / "final" / "analysis" / "label_error_diagnostics" / "label_error_case_summary.csv"
)
DEFAULT_FRAME_TIMELINE = (
    REPO_ROOT / "final" / "analysis" / "label_error_diagnostics" / "label_error_case_frame_timeline.csv"
)
DEFAULT_RAW_ROOT = REPO_ROOT / "data" / "raw" / "raw" / "europe"
DEFAULT_OUTDIR = REPO_ROOT / "final" / "analysis" / "qualitative_match_context"

EARLY_MAX_MINUTE = 12.0
LABEL_START_MINUTE = 5.0
LABEL_MAX_MINUTE = 12.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract qualitative match context for top errors.")
    p.add_argument("--top-errors", default=str(DEFAULT_TOP_ERRORS))
    p.add_argument("--label-summary", default=str(DEFAULT_LABEL_SUMMARY))
    p.add_argument("--frame-timeline", default=str(DEFAULT_FRAME_TIMELINE))
    p.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    p.add_argument("--top-n", type=int, default=20)
    return p.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def event_minute(event: Dict[str, Any]) -> float:
    return float(event.get("timestamp", 0.0)) / 60000.0


def participant_maps(match: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], Dict[Tuple[int, str], int]]:
    participants = (match.get("info") or {}).get("participants") or []
    by_pid: Dict[int, Dict[str, Any]] = {}
    role_to_pid: Dict[Tuple[int, str], int] = {}
    for p in participants:
        pid = p.get("participantId")
        if not isinstance(pid, int):
            continue
        by_pid[pid] = p
        team_id = p.get("teamId")
        role = str(p.get("teamPosition") or "").upper()
        if isinstance(team_id, int) and role:
            role_to_pid[(team_id, role)] = pid
    return by_pid, role_to_pid


def pid_label(pid: Optional[int], by_pid: Dict[int, Dict[str, Any]]) -> str:
    if not isinstance(pid, int) or pid <= 0:
        return ""
    p = by_pid.get(pid, {})
    role = str(p.get("teamPosition") or "")
    champion = str(p.get("championName") or f"pid{pid}")
    team = p.get("teamId", "")
    return f"{champion}({role},T{team},pid{pid})"


def relation(pid: Optional[int], team_id: int, support_pid: int, adc_pid: int, by_pid: Dict[int, Dict[str, Any]]) -> str:
    if not isinstance(pid, int) or pid <= 0:
        return ""
    if pid == support_pid:
        return "ally_support"
    if pid == adc_pid:
        return "ally_adc"
    p = by_pid.get(pid, {})
    if p.get("teamId") == team_id:
        return f"ally_{str(p.get('teamPosition') or '').lower()}"
    return f"enemy_{str(p.get('teamPosition') or '').lower()}"


def iter_events(timeline: Dict[str, Any], max_minute: float) -> List[Dict[str, Any]]:
    frames = (timeline.get("info") or {}).get("frames") or timeline.get("frames") or []
    out: List[Dict[str, Any]] = []
    for frame in frames:
        for event in frame.get("events") or []:
            if event_minute(event) <= max_minute:
                out.append(event)
    return sorted(out, key=lambda e: e.get("timestamp", 0))


def summarize_participant(p: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "championName",
        "teamPosition",
        "kills",
        "deaths",
        "assists",
        "goldEarned",
        "champLevel",
        "totalMinionsKilled",
        "neutralMinionsKilled",
        "visionScore",
        "wardsPlaced",
        "wardsKilled",
        "totalDamageDealtToChampions",
        "totalHeal",
        "totalDamageShieldedOnTeammates",
        "win",
    ]
    return {key: p.get(key) for key in keys}


def kill_event_row(
    case: pd.Series,
    event: Dict[str, Any],
    by_pid: Dict[int, Dict[str, Any]],
    support_pid: int,
    adc_pid: int,
) -> Dict[str, Any]:
    team_id = int(case["team_id"])
    killer = event.get("killerId")
    victim = event.get("victimId")
    assists = [int(x) for x in event.get("assistingParticipantIds") or [] if isinstance(x, int)]
    involved = set([pid for pid in [killer, victim] if isinstance(pid, int)]) | set(assists)
    pos = event.get("position") or {}
    return {
        "error_rank": int(case["error_rank"]),
        "match_id": case["match_id"],
        "team_id": team_id,
        "minute": event_minute(event),
        "event_type": "CHAMPION_KILL",
        "killer": pid_label(killer, by_pid),
        "victim": pid_label(victim, by_pid),
        "assists": "; ".join(pid_label(pid, by_pid) for pid in assists),
        "killer_relation": relation(killer, team_id, support_pid, adc_pid, by_pid),
        "victim_relation": relation(victim, team_id, support_pid, adc_pid, by_pid),
        "assist_relations": "; ".join(relation(pid, team_id, support_pid, adc_pid, by_pid) for pid in assists),
        "ally_support_involved": support_pid in involved,
        "ally_adc_involved": adc_pid in involved,
        "ally_support_died": victim == support_pid,
        "ally_adc_died": victim == adc_pid,
        "ally_support_assist": support_pid in assists,
        "ally_adc_assist": adc_pid in assists,
        "x": pos.get("x"),
        "y": pos.get("y"),
    }


def objective_event_row(
    case: pd.Series,
    event: Dict[str, Any],
    by_pid: Dict[int, Dict[str, Any]],
    support_pid: int,
    adc_pid: int,
) -> Optional[Dict[str, Any]]:
    typ = event.get("type")
    if typ not in {"ELITE_MONSTER_KILL", "BUILDING_KILL", "TURRET_PLATE_DESTROYED"}:
        return None
    team_id = int(case["team_id"])
    killer = event.get("killerId")
    assists = [int(x) for x in event.get("assistingParticipantIds") or [] if isinstance(x, int)]
    involved = set([pid for pid in [killer] if isinstance(pid, int)]) | set(assists)
    pos = event.get("position") or {}
    return {
        "error_rank": int(case["error_rank"]),
        "match_id": case["match_id"],
        "team_id": team_id,
        "minute": event_minute(event),
        "event_type": typ,
        "monster_type": event.get("monsterType"),
        "building_type": event.get("buildingType"),
        "lane_type": event.get("laneType"),
        "killer": pid_label(killer, by_pid),
        "killer_relation": relation(killer, team_id, support_pid, adc_pid, by_pid),
        "assists": "; ".join(pid_label(pid, by_pid) for pid in assists),
        "ally_support_involved": support_pid in involved,
        "ally_adc_involved": adc_pid in involved,
        "x": pos.get("x"),
        "y": pos.get("y"),
    }


def count_kda(events: List[Dict[str, Any]], pid: int) -> Dict[str, int]:
    kills = deaths = assists = 0
    for e in events:
        if e.get("type") != "CHAMPION_KILL":
            continue
        if e.get("killerId") == pid:
            kills += 1
        if e.get("victimId") == pid:
            deaths += 1
        if pid in (e.get("assistingParticipantIds") or []):
            assists += 1
    return {"kills": kills, "deaths": deaths, "assists": assists}


def nearest_frame_context(frames: pd.DataFrame, minute: float) -> Dict[str, Any]:
    if frames.empty:
        return {}
    idx = (frames["minute"] - minute).abs().idxmin()
    row = frames.loc[idx]
    return {
        "nearest_frame_minute": float(row["minute"]),
        "support_zone_v5_abs": row.get("support_zone_v5_abs"),
        "adc_zone_v5_abs": row.get("adc_zone_v5_abs"),
        "dist_to_adc": row.get("dist_to_adc"),
        "out_bot_context_v5": row.get("out_bot_context_v5"),
        "far_from_adc_v5": row.get("far_from_adc_v5"),
    }


def interpret_case(row: pd.Series) -> str:
    parts: List[str] = []
    if row["actual"] >= 0.9 and row["prediction"] <= 0.35:
        parts.append("modelo infrapredice un roaming extremo")
    if row["outside_ratio_v5"] >= 0.9 and row["far_ratio_v5"] >= 0.9:
        parts.append("la etiqueta parece consistente: fuera de bot y lejos del ADC casi toda la ventana")
    elif row["outside_ratio_v5"] >= 0.7:
        parts.append("la etiqueta viene sobre todo de presencia fuera de bot")
    elif row["far_ratio_v5"] >= 0.7:
        parts.append("la etiqueta viene sobre todo de distancia al ADC")
    if row["support_early_deaths"] > 0:
        parts.append(f"support muere {int(row['support_early_deaths'])} vez/veces antes de 12")
    if row["adc_early_deaths"] > 0:
        parts.append(f"ADC muere {int(row['adc_early_deaths'])} vez/veces antes de 12")
    if row["support_early_assists"] + row["support_early_kills"] > 0:
        parts.append("support participa en kills tempranas")
    if row["label_diagnostic"] in {"low_valid_support_frames", "low_valid_coop_frames", "possible_adc_death_base_coop_artifact"}:
        parts.append(f"requiere cautela por diagnostico de etiqueta: {row['label_diagnostic']}")
    return "; ".join(parts) if parts else "caso mixto; revisar timeline"


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            display[col] = display[col].fillna("").astype(str)
    headers = list(display.columns)
    rows = display.astype(str).values.tolist()
    widths = [max(len(str(h)), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    lines = [
        "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |",
        "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows)
    return "\n".join(lines)


def short_kill_line(row: pd.Series) -> str:
    details = []
    if row.get("ally_support_died"):
        details.append("muere support aliado")
    if row.get("ally_adc_died"):
        details.append("muere ADC aliado")
    if row.get("ally_support_assist"):
        details.append("asiste support aliado")
    if row.get("ally_adc_assist"):
        details.append("asiste ADC aliado")
    if row.get("ally_support_involved") and not row.get("ally_support_died") and not row.get("ally_support_assist"):
        details.append("participa support aliado")
    if row.get("ally_adc_involved") and not row.get("ally_adc_died") and not row.get("ally_adc_assist"):
        details.append("participa ADC aliado")
    suffix = f" ({'; '.join(details)})" if details else ""
    return f"- min {row['minute']:.2f}: {row['killer']} mata a {row['victim']}; assists: {row.get('assists') or '-'}{suffix}"


def case_notes(summary_df: pd.DataFrame, event_df: pd.DataFrame, top_k: int = 5, max_events: int = 8) -> List[str]:
    notes: List[str] = []
    for _, case in summary_df.head(top_k).iterrows():
        match_id = str(case["match_id"])
        error_rank = int(case["error_rank"])
        events = event_df[
            (event_df["match_id"].astype(str) == match_id)
            & (
                event_df["ally_support_involved"].fillna(False)
                | event_df["ally_adc_involved"].fillna(False)
                | event_df["killer_relation"].astype(str).str.startswith("enemy_")
                | event_df["victim_relation"].astype(str).str.startswith("enemy_")
            )
        ].sort_values("minute")
        direct_events = events[
            events["ally_support_involved"].fillna(False) | events["ally_adc_involved"].fillna(False)
        ].head(max_events)
        if direct_events.empty:
            direct_events = events.head(max_events)

        notes.extend(
            [
                f"### Case {error_rank}: {case['match_id']} ({case['ally_support']} + {case['ally_adc']})",
                "",
                (
                    f"Predicho={case['prediction']:.3f}, real={case['actual']:.3f}, "
                    f"error={case['abs_error']:.3f}. Draft: {case['ally_top']}/{case['ally_jungle']}/"
                    f"{case['ally_middle']}/{case['ally_adc']}/{case['ally_support']} vs "
                    f"{case['enemy_top']}/{case['enemy_jungle']}/{case['enemy_middle']}/"
                    f"{case['enemy_adc']}/{case['enemy_support']}."
                ),
                (
                    f"Lectura: {case['qualitative_reading']}. KDA final botlane aliada: "
                    f"support {case['support_final_kda']}, ADC {case['adc_final_kda']}."
                ),
                "",
                "Eventos tempranos relevantes:",
            ]
        )
        notes.extend(short_kill_line(row) for _, row in direct_events.iterrows())
        notes.append("")
    return notes


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cases = pd.read_csv(args.top_errors).head(args.top_n)
    label_summary = pd.read_csv(args.label_summary)
    frame_timeline = pd.read_csv(args.frame_timeline)
    cases = cases.merge(
        label_summary[
            [
                "match_id",
                "team_id",
                "outside_ratio_v5",
                "far_ratio_v5",
                "xp_gap_v5",
                "valid_support_frames_v5",
                "valid_coop_frames_v5",
                "label_diagnostic",
            ]
        ],
        on=["match_id", "team_id"],
        how="left",
    )

    event_rows: List[Dict[str, Any]] = []
    objective_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for _, case in cases.iterrows():
        match_id = str(case["match_id"])
        team_id = int(case["team_id"])
        match_path = Path(args.raw_root) / match_id / "match.json"
        timeline_path = Path(args.raw_root) / match_id / "timeline.json"
        if not match_path.exists() or not timeline_path.exists():
            summary_rows.append(
                {
                    "error_rank": int(case["error_rank"]),
                    "match_id": match_id,
                    "team_id": team_id,
                    "raw_available": False,
                }
            )
            continue

        match = load_json(match_path)
        timeline = load_json(timeline_path)
        by_pid, role_to_pid = participant_maps(match)
        support_pid = role_to_pid.get((team_id, "UTILITY"))
        adc_pid = role_to_pid.get((team_id, "BOTTOM"))
        enemy_team = 200 if team_id == 100 else 100
        enemy_support_pid = role_to_pid.get((enemy_team, "UTILITY"))
        enemy_adc_pid = role_to_pid.get((enemy_team, "BOTTOM"))
        if support_pid is None or adc_pid is None:
            continue

        early_events = iter_events(timeline, EARLY_MAX_MINUTE)
        kill_events = [e for e in early_events if e.get("type") == "CHAMPION_KILL"]
        label_events = [e for e in early_events if LABEL_START_MINUTE <= event_minute(e) <= LABEL_MAX_MINUTE]
        label_kill_events = [e for e in label_events if e.get("type") == "CHAMPION_KILL"]

        case_frames = frame_timeline[
            (frame_timeline["match_id"].astype(str) == match_id)
            & (frame_timeline["team_id"].astype(int) == team_id)
        ].copy()

        for event in kill_events:
            row = kill_event_row(case, event, by_pid, support_pid, adc_pid)
            row.update(nearest_frame_context(case_frames, row["minute"]))
            event_rows.append(row)

        for event in early_events:
            row = objective_event_row(case, event, by_pid, support_pid, adc_pid)
            if row:
                row.update(nearest_frame_context(case_frames, row["minute"]))
                objective_rows.append(row)

        support_kda = count_kda(kill_events, support_pid)
        adc_kda = count_kda(kill_events, adc_pid)
        enemy_support_kda = count_kda(kill_events, enemy_support_pid or -1)
        enemy_adc_kda = count_kda(kill_events, enemy_adc_pid or -1)
        support_label_kda = count_kda(label_kill_events, support_pid)
        adc_label_kda = count_kda(label_kill_events, adc_pid)

        support_final = summarize_participant(by_pid.get(support_pid, {}))
        adc_final = summarize_participant(by_pid.get(adc_pid, {}))
        enemy_support_final = summarize_participant(by_pid.get(enemy_support_pid or -1, {}))
        enemy_adc_final = summarize_participant(by_pid.get(enemy_adc_pid or -1, {}))

        bot_related_kills = 0
        support_related_kills = 0
        adc_related_kills = 0
        for e in kill_events:
            involved = set([e.get("killerId"), e.get("victimId")]) | set(e.get("assistingParticipantIds") or [])
            if involved & {support_pid, adc_pid, enemy_support_pid, enemy_adc_pid}:
                bot_related_kills += 1
            if support_pid in involved:
                support_related_kills += 1
            if adc_pid in involved:
                adc_related_kills += 1

        row = {
            "error_rank": int(case["error_rank"]),
            "match_id": match_id,
            "team_id": team_id,
            "raw_available": True,
            "side": case.get("side"),
            "patch": case.get("patch"),
            "game_version": case.get("game_version"),
            "prediction": float(case["prediction"]),
            "actual": float(case["actual"]),
            "abs_error": float(case["abs_error"]),
            "ally_top": case.get("ally_top_champion_name"),
            "ally_jungle": case.get("ally_jungle_champion_name"),
            "ally_middle": case.get("ally_middle_champion_name"),
            "ally_support": support_final.get("championName"),
            "ally_adc": adc_final.get("championName"),
            "enemy_top": case.get("enemy_top_champion_name"),
            "enemy_jungle": case.get("enemy_jungle_champion_name"),
            "enemy_middle": case.get("enemy_middle_champion_name"),
            "enemy_support": enemy_support_final.get("championName"),
            "enemy_adc": enemy_adc_final.get("championName"),
            "outside_ratio_v5": case.get("outside_ratio_v5"),
            "far_ratio_v5": case.get("far_ratio_v5"),
            "xp_gap_v5": case.get("xp_gap_v5"),
            "valid_support_frames_v5": case.get("valid_support_frames_v5"),
            "valid_coop_frames_v5": case.get("valid_coop_frames_v5"),
            "label_diagnostic": case.get("label_diagnostic"),
            "support_early_kills": support_kda["kills"],
            "support_early_deaths": support_kda["deaths"],
            "support_early_assists": support_kda["assists"],
            "adc_early_kills": adc_kda["kills"],
            "adc_early_deaths": adc_kda["deaths"],
            "adc_early_assists": adc_kda["assists"],
            "support_label_window_kills": support_label_kda["kills"],
            "support_label_window_deaths": support_label_kda["deaths"],
            "support_label_window_assists": support_label_kda["assists"],
            "adc_label_window_kills": adc_label_kda["kills"],
            "adc_label_window_deaths": adc_label_kda["deaths"],
            "adc_label_window_assists": adc_label_kda["assists"],
            "enemy_support_early_kda": f"{enemy_support_kda['kills']}/{enemy_support_kda['deaths']}/{enemy_support_kda['assists']}",
            "enemy_adc_early_kda": f"{enemy_adc_kda['kills']}/{enemy_adc_kda['deaths']}/{enemy_adc_kda['assists']}",
            "bot_related_kill_events_0_12": bot_related_kills,
            "support_related_kill_events_0_12": support_related_kills,
            "adc_related_kill_events_0_12": adc_related_kills,
            "support_final_kda": f"{support_final.get('kills')}/{support_final.get('deaths')}/{support_final.get('assists')}",
            "adc_final_kda": f"{adc_final.get('kills')}/{adc_final.get('deaths')}/{adc_final.get('assists')}",
            "support_final_vision": support_final.get("visionScore"),
            "support_final_gold": support_final.get("goldEarned"),
            "adc_final_gold": adc_final.get("goldEarned"),
            "support_final_win": support_final.get("win"),
        }
        row["qualitative_reading"] = interpret_case(pd.Series(row))
        summary_rows.append(row)

    event_df = pd.DataFrame(event_rows)
    objective_df = pd.DataFrame(objective_rows)
    summary_df = pd.DataFrame(summary_rows).sort_values("error_rank")

    event_df.to_csv(outdir / "early_kill_events_0_12.csv", index=False)
    objective_df.to_csv(outdir / "early_objective_events_0_12.csv", index=False)
    summary_df.to_csv(outdir / "qualitative_case_summary.csv", index=False)

    md_cols = [
        "error_rank",
        "side",
        "patch",
        "ally_support",
        "ally_adc",
        "enemy_support",
        "enemy_adc",
        "prediction",
        "actual",
        "outside_ratio_v5",
        "far_ratio_v5",
        "support_early_kills",
        "support_early_deaths",
        "support_early_assists",
        "adc_early_deaths",
        "bot_related_kill_events_0_12",
        "label_diagnostic",
        "qualitative_reading",
    ]
    md = [
        "# Qualitative Match Context for Top Errors",
        "",
        "This file adds real match evidence from `match.json` and `timeline.json`: early kills, deaths, assists, objective events and final botlane stats. It is the qualitative layer on top of the label reconstruction.",
        "",
        markdown_table(summary_df[[c for c in md_cols if c in summary_df.columns]]),
        "",
        "## Case notes",
        "",
        "The notes below are generated from real `CHAMPION_KILL` events in the Riot timeline. They are not inferred from the draft model.",
        "",
        *case_notes(summary_df, event_df),
        "## How to use",
        "",
        "Use cases marked `consistent_full_roam_label` plus clear early event context as examples of unpredictable in-game variance. Treat cases marked `low_valid_*` or `possible_adc_death_base_coop_artifact` as cautionary label-limit examples.",
        "",
    ]
    (outdir / "qualitative_match_context.md").write_text("\n".join(md), encoding="utf-8")

    meta = {
        "top_errors_path": str(Path(args.top_errors).resolve()),
        "label_summary_path": str(Path(args.label_summary).resolve()),
        "frame_timeline_path": str(Path(args.frame_timeline).resolve()),
        "raw_root": str(Path(args.raw_root).resolve()),
        "outdir": str(outdir.resolve()),
        "top_n": args.top_n,
        "cases": int(len(summary_df)),
        "cases_with_raw": int(summary_df["raw_available"].fillna(False).sum()) if not summary_df.empty else 0,
        "kill_event_rows": int(len(event_df)),
        "objective_event_rows": int(len(objective_df)),
    }
    (outdir / "qualitative_match_context_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[Cases] {meta['cases_with_raw']}/{meta['cases']} raw matches available")
    print(f"[Events] kills={len(event_df)} objectives={len(objective_df)}")
    print(f"[Saved] {outdir.resolve()}")


if __name__ == "__main__":
    main()
