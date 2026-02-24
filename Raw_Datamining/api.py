import os
import time
import random

from dotenv import load_dotenv
from riotwatcher import LolWatcher, RiotWatcher, ApiError

# --- CONFIGURACIÓN ---
load_dotenv('TFG.env')

API_KEY = os.getenv('RIOT_API_KEY')
REGION = os.getenv('REGION', 'europe')            # routing (europe/americas/asia)
MATCH_REGION = os.getenv('MATCH_REGION', 'euw1')  # platform (euw1/na1/...)

GAME_NAME = os.getenv('GAME_NAME', 'adriidz')
TAG_LINE = os.getenv('TAG_LINE', 'diaz')

if not API_KEY:
    raise ValueError("ERROR: No se encontró la variable 'RIOT_API_KEY' en el archivo .env")

# Watchers
watcher = LolWatcher(API_KEY)           # LoL endpoints
account_watcher = RiotWatcher(API_KEY)  # Account endpoints

# --- MVP LABEL TUNING CONSTANTS (v4.1) ---
PWR_TAKEDOWN_GOLD = float(os.getenv('PWR_TAKEDOWN_GOLD', '90'))
PWR_SIGMOID_SCALE = float(os.getenv('PWR_SIGMOID_SCALE', '800'))
STYLE_NEAR_DIST = float(os.getenv('STYLE_NEAR_DIST', '2600'))
STYLE_NEAR_ALLIES = int(os.getenv('STYLE_NEAR_ALLIES', '1'))
STYLE_SIDE_MARGIN = float(os.getenv('STYLE_SIDE_MARGIN', '0.25'))
STYLE_SIGMOID_SCALE = float(os.getenv('STYLE_SIGMOID_SCALE', '0.10'))

# Role-specific MVP thresholds (override via env if you want)
SUP_ROAM_START_MIN = int(os.getenv('SUP_ROAM_START_MIN', '2'))
SUP_ROAM_END_MIN = int(os.getenv('SUP_ROAM_END_MIN', '14'))
SUP_ROAM_MIN_VALID = int(os.getenv('SUP_ROAM_MIN_VALID', '6'))  # minutes with valid position required
SUP_ROAM_PROB_SCALE = float(os.getenv('SUP_ROAM_PROB_SCALE', '0.08'))
SUP_ROAM_CLASS_MARGIN = float(os.getenv('SUP_ROAM_CLASS_MARGIN', '0.15'))  # roam_ratio - bot_ratio

JG_GANK_PROB_SCALE = float(os.getenv('JG_GANK_PROB_SCALE', '0.9'))
JG_GANK_CLASS_THRESHOLD = float(os.getenv('JG_GANK_CLASS_THRESHOLD', '0.0'))


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
