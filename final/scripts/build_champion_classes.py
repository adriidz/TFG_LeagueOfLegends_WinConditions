#!/usr/bin/env python3
"""
Generate champion_classes.json from Riot Data Dragon.

Maps champion_id (key) → primary_class (tags[0]) for use in archetype-based
ceiling analysis. Only needs to be run once; output is checked into the repo.
"""

import json
import urllib.request
from pathlib import Path

DDRAGON_URL = "https://ddragon.leagueoflegends.com/cdn/14.10.1/data/en_US/champion.json"
OUTPATH = Path(__file__).resolve().parent.parent / "data" / "champion_classes.json"


def main():
    print(f"[Fetch] {DDRAGON_URL}")
    with urllib.request.urlopen(DDRAGON_URL) as resp:
        raw = json.loads(resp.read().decode("utf-8"))

    mapping = {}
    for champ_data in raw["data"].values():
        champ_id = int(champ_data["key"])
        tags = champ_data.get("tags", [])
        primary = tags[0] if tags else "Unknown"
        mapping[str(champ_id)] = {
            "name": champ_data["name"],
            "primary_class": primary,
            "classes": tags,
        }

    OUTPATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPATH.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    classes = {}
    for v in mapping.values():
        c = v["primary_class"]
        classes[c] = classes.get(c, 0) + 1
    print(f"[Saved] {OUTPATH}")
    print(f"  Champions: {len(mapping)}")
    for c, n in sorted(classes.items(), key=lambda x: -x[1]):
        print(f"    {c}: {n}")


if __name__ == "__main__":
    main()
