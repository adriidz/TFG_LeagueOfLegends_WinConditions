
# Pipeline ML v1 para la etiqueta del jungla

## Qué hace el script
`train_jungle_presence_model.py` entrena una primera versión de modelo para predecir la etiqueta del jungla a partir del draft.

Compara tres niveles:

1. `majority_baseline`
   - siempre predice la clase mayoritaria
   - sirve como suelo mínimo

2. `jungle_only_logreg`
   - usa solo el campeón del jungla propio, más `side` y `patch` si existen
   - sirve para medir cuánta señal viene solo del pick del jungla

3. `full_draft_logreg`
   - usa el draft completo de ambos equipos
   - sirve para medir si el contexto de picks del resto añade señal

## Para qué sirve
Sirve para responder estas preguntas:

- ¿hay señal predictiva en el draft?
- ¿esa señal está casi toda en el campeón jungla?
- ¿el contexto de aliados y enemigos ayuda?
- ¿side y patch aportan algo?

## Esquema recomendado del parquet de entrada
Una fila por `(match_id, team_id)`.

Columnas mínimas recomendadas:

- `match_id`
- `team_id`
- `jungle_presence_label`
- `ally_top_champion_name`
- `ally_jungle_champion_name`
- `ally_middle_champion_name`
- `ally_bottom_champion_name`
- `ally_utility_champion_name`
- `enemy_top_champion_name`
- `enemy_jungle_champion_name`
- `enemy_middle_champion_name`
- `enemy_bottom_champion_name`
- `enemy_utility_champion_name`
- `side`
- `patch`

Opcionales útiles:
- `jungle_presence_score`
- `patch_major`
- `patch_minor`
- `queue_id`

## Cómo encaja con tu pipeline
Orden natural:

1. `build_jungle_labels.py`
   - produce score y etiqueta del jungla

2. `build_draft_features.py`
   - produce parquet con picks por rol, side, patch

3. join por `(match_id, team_id)`
   - crea `model_input.parquet`

4. `train_jungle_presence_model.py`
   - entrena y evalúa los primeros modelos

## Cómo ejecutarlo
Ejemplo:

```bash
python train_jungle_presence_model.py   --input-path Data_clean/model_input_jungle.parquet   --output-dir Data_clean/ml/jungle_presence_v1   --class-weight-balanced
```

## Qué outputs genera
- `metrics_summary.json`
- `model_ranking.csv`
- `model_ranking.parquet`
- `predictions_majority_baseline.parquet`
- `predictions_jungle_only_logreg.parquet`
- `predictions_full_draft_logreg.parquet`
- `model_majority_baseline.joblib`
- `model_jungle_only_logreg.joblib`
- `model_full_draft_logreg.joblib`

## Cómo interpretar el resultado
La comparación clave es esta:

- si `full_draft_logreg` no mejora casi nada a `jungle_only_logreg`, entonces la mayor parte de la señal viene del campeón jungla
- si `full_draft_logreg` mejora de forma clara, entonces el contexto de draft sí aporta
- si ambos mejoran al baseline mayoritario, entonces el draft contiene señal no trivial

## Cómo continuar después
La siguiente iteración lógica sería:

1. crear `build_draft_features.py`
2. hacer el join con las labels
3. correr este script
4. mirar:
   - balanced accuracy
   - ROC-AUC
   - F1
5. después pasar a:
   - validación temporal por parche
   - gradient boosting
   - importancia de variables
