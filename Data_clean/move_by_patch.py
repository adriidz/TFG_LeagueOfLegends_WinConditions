import json
import shutil
from pathlib import Path

RAW_ROOT = Path("Data/raw/europe")
DEST_ROOT = Path("Data/raw_archive/europe")

MIN_MAJOR = 16
MIN_MINOR = 2

DRY_RUN = False  # 1) deja True para revisar, 2) pon False para mover

def parse_mm(game_version: str):
    try:
        parts = str(game_version).split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return None

def at_least(mm, min_major, min_minor) -> bool:
    if mm is None:
        return False
    major, minor = mm
    return (major > min_major) or (major == min_major and minor >= min_minor)

def main():
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    moved = 0
    kept = 0
    skipped = 0

    for match_dir in RAW_ROOT.iterdir():
        if not match_dir.is_dir():
            continue

        match_json = match_dir / "match.json"
        if not match_json.exists():
            skipped += 1
            continue

        try:
            data = json.loads(match_json.read_text(encoding="utf-8"))
            gv = (data.get("info") or {}).get("gameVersion")
            mm = parse_mm(gv)
        except Exception:
            skipped += 1
            continue

        if at_least(mm, MIN_MAJOR, MIN_MINOR):
            kept += 1
            continue

        target = DEST_ROOT / match_dir.name
        if DRY_RUN:
            print(f"[DRY] MOVE {match_dir} -> {target} (gameVersion={gv})")
        else:
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(match_dir), str(target))
        moved += 1

    print("\n=== RESUMEN ===")
    print(f"Kept (>= {MIN_MAJOR}.{MIN_MINOR}): {kept}")
    print(f"Moved (<  {MIN_MAJOR}.{MIN_MINOR}): {moved}")
    print(f"Skipped (sin match.json/errores): {skipped}")

if __name__ == "__main__":
    main()