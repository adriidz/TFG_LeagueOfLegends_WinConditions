#!/usr/bin/env python3
"""
Build a combined champion reference table for support roam scores.

Official fields come from Riot Data Dragon champion.json. Expert fields come
from ProgresoActual/references/manual_support_champion_reference.csv.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

DEFAULT_MANUAL_PATH = os.path.join("ProgresoActual", "references", "manual_support_champion_reference.csv")
DEFAULT_OUT_PATH = os.path.join("ProgresoActual", "references", "champion_support_reference.csv")
VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
CHAMPION_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/data/{language}/champion.json"


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_name(value: object) -> str:
    return str(value).strip().lower().replace(" ", "").replace("'", "").replace(".", "")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge Data Dragon champion metadata with manual support references.")
    p.add_argument("--manual-path", default=DEFAULT_MANUAL_PATH)
    p.add_argument("--out-path", default=DEFAULT_OUT_PATH)
    p.add_argument("--version", default="latest", help="Data Dragon version or 'latest'.")
    p.add_argument("--language", default="en_US")
    p.add_argument("--manual-only", action="store_true",
                   help="Skip Data Dragon fetch and only normalize the manual table.")
    return p.parse_args()


def fetch_latest_version() -> str:
    response = requests.get(VERSIONS_URL, timeout=30)
    response.raise_for_status()
    versions = response.json()
    if not versions:
        raise RuntimeError("Data Dragon versions response is empty.")
    return str(versions[0])


def fetch_data_dragon_champions(version: str, language: str) -> pd.DataFrame:
    url = CHAMPION_URL.format(version=version, language=language)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    rows = []
    for champion_id, item in payload.get("data", {}).items():
        info = item.get("info", {}) or {}
        rows.append({
            "champion_id_slug": champion_id,
            "champion_key": item.get("key"),
            "champion_name": item.get("name"),
            "official_title": item.get("title"),
            "official_tags": "|".join(item.get("tags", []) or []),
            "official_info_attack": info.get("attack"),
            "official_info_defense": info.get("defense"),
            "official_info_magic": info.get("magic"),
            "official_info_difficulty": info.get("difficulty"),
            "ddragon_version": version,
            "ddragon_language": language,
        })
    return pd.DataFrame(rows)


def load_manual(path: str) -> pd.DataFrame:
    manual = pd.read_csv(path)
    required = {"champion_name", "expert_archetype", "expert_support_roam_score", "expert_confidence"}
    missing = sorted(required - set(manual.columns))
    if missing:
        raise SystemExit(f"Manual reference missing columns: {missing}")
    manual["expert_support_roam_score"] = pd.to_numeric(manual["expert_support_roam_score"], errors="coerce")
    manual["expert_confidence"] = pd.to_numeric(manual["expert_confidence"], errors="coerce")
    bad = manual[~manual["expert_support_roam_score"].between(0.0, 1.0, inclusive="both")]
    if not bad.empty:
        raise SystemExit(f"Expert scores must be in [0, 1]. Bad champions: {bad['champion_name'].tolist()}")
    return manual


def main() -> None:
    args = parse_args()
    manual = load_manual(args.manual_path)
    manual["_join_name"] = manual["champion_name"].map(normalize_name)

    official: Optional[pd.DataFrame]
    version = args.version
    if args.manual_only:
        official = None
    else:
        if version == "latest":
            version = fetch_latest_version()
        official = fetch_data_dragon_champions(version, args.language)
        official["_join_name"] = official["champion_name"].map(normalize_name)

    if official is None:
        out = manual.copy()
        out.insert(0, "reference_source", "manual_only")
    else:
        out = official.merge(
            manual.drop(columns=["champion_name"]),
            on="_join_name",
            how="left",
            validate="one_to_one",
        )
        out.insert(0, "reference_source", "data_dragon_plus_manual")

    out = out.drop(columns=["_join_name"], errors="ignore")
    ensure_dir(str(Path(args.out_path).parent))
    out.to_csv(args.out_path, index=False)
    print(f"Saved champion reference: {os.path.abspath(args.out_path)}")
    print(f"Rows: {len(out)}")
    if "expert_support_roam_score" in out.columns:
        print(f"Manual expert labels present: {int(out['expert_support_roam_score'].notna().sum())}")


if __name__ == "__main__":
    main()
