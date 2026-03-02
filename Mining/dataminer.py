"""
Facade module.

Mantiene la API pública original:
  - API_KEY, REGION, MATCH_REGION
  - process_match(match_id)
  - get_match_ids(puuid, count=5, ranked_only=False)

El resto de utilidades se han repartido en:
  - api.py (env/watchers/retry)
  - position.py (métricas de posición)
  - timeline.py (parsers de timeline)
  - labels.py (labels/encoders)
  - processing.py (orquestación process_match)
"""

from ..Raw_Data.api import (
    API_KEY,
    REGION,
    MATCH_REGION,
    watcher,
    account_watcher,
    riot_call_with_retry,
    get_match_ids,
)

from .processing import process_match


if __name__ == '__main__':
    # Conserva el comportamiento de "test" al ejecutar este archivo directamente.
    from .processing import _main_test
    _main_test()
