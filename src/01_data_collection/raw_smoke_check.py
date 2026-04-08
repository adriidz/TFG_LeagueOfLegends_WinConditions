import os
import json
import random

RAW_ROOT = os.getenv("RAW_ROOT", "data/raw/raw")
REGION = os.getenv("REGION", "europe")

def main(n=1000):
    base = os.path.join(RAW_ROOT, REGION)
    if not os.path.isdir(base):
        raise SystemExit(f"No existe {base}. ¿Has recolectado algo?")

    match_dirs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    if not match_dirs:
        raise SystemExit("No hay matches guardados.")

    sample = random.sample(match_dirs, min(n, len(match_dirs)))
    ok = 0
    for md in sample:
        mp = os.path.join(md, "match.json")
        tp = os.path.join(md, "timeline.json")
        if not (os.path.exists(mp) and os.path.exists(tp)):
            print("MISSING", md)
            continue
        with open(mp, "r", encoding="utf-8") as f:
            m = json.load(f)
        with open(tp, "r", encoding="utf-8") as f:
            t = json.load(f)
        frames = (((t or {}).get("info") or {}).get("frames") or [])
        dur = (((m or {}).get("info") or {}).get("gameDuration"))
        print(os.path.basename(md), "frames=", len(frames), "dur=", dur)
        ok += 1
    print(f"OK {ok}/{len(sample)}")

if __name__ == "__main__":
    main()