# Especificación técnica — Fase Final

Documento de referencia para ejecutar los scripts de `final/`. Contiene rutas
exactas, columnas relevantes, formato de salida y criterios de cada paso.

## Entradas externas (solo lectura)

### Draft features

```
Ruta: ProgresoActual/data/clean/features/draft_features.parquet
```

Columnas clave:

- `match_id` — identificador de partida
- `team_id` — 100 (blue) o 200 (red)
- `side` — "blue" o "red"
- `patch` — versión del parche
- `ally_top_champion_id`, `ally_jungle_champion_id`, `ally_middle_champion_id`,
  `ally_bottom_champion_id`, `ally_utility_champion_id` — campeones aliados
- `enemy_top_champion_id`, ..., `enemy_utility_champion_id` — campeones enemigos
- `ally_*_summoner1_id`, `ally_*_summoner2_id` — hechizos de invocador
- `enemy_*_summoner1_id`, `enemy_*_summoner2_id`
- `ally_*_keystone_id` — runa keystone
- `ally_*_primary_style_id`, `ally_*_sub_style_id` — árboles de runas
- `ally_ban_1_champion_id` ... `ally_ban_5_champion_id` — bans aliados
- `enemy_ban_1_champion_id` ... `enemy_ban_5_champion_id` — bans enemigos

Unidad de análisis: `(match_id, team_id)` — una fila por equipo por partida.

### Support scores v5

```
Ruta: ProgresoActual2/data/clean/scores/support_scores_v5_geometry_m12.parquet
```

Columnas clave:

- `match_id`, `team_id` — join key
- `raw_support_roam_score_v5_geometry` — score bruto antes de gamma
- `support_roam_score_v5_geometry` — score con gamma 0.75 (etiqueta principal)
- `support_champion_name` — nombre del campeón support (útil para análisis)

### Referencia experta

```
Ruta: ProgresoActual/references/manual_support_champion_reference.csv
```

Columnas: `champion_name`, `expert_archetype`, `expert_support_roam_score`,
`expert_confidence`, `notes`.

47 campeones. Solo para validación cualitativa de ranking, no para training.

---

## Script 01 — Preparar dataset final

**Input**: draft features + support scores v5.

**Operación**:

1. Leer ambos parquets.
2. Join inner por `(match_id, team_id)`.
3. Renombrar `support_roam_score_v5_geometry` → `support_roam_score` (target canónico).
4. Conservar también `raw_support_roam_score_v5_geometry` para referencia.
5. Split por `match_id` con `GroupShuffleSplit(test_size=0.2, random_state=42)`.
6. Fittear `QuantileTransformer(output_distribution='uniform', n_quantiles=1000)`
   SOLO sobre los targets del split de train.
7. Crear columna `support_roam_score_quantile` en train (fit_transform) y val
   (transform). Para la variante zero-preserved: fittear solo sobre rows con
   score > 0 en train, dejar score=0 como 0.
8. Guardar como dos parquets separados:
   - `final/data/training/train.parquet`
   - `final/data/training/val.parquet`
9. Guardar `final/data/training/quantile_transformer.joblib` para reproducibilidad.
10. Guardar `final/data/training/split_summary.json` con recuentos y estadísticas.

**Columna target para modelos**:

- Modelos raw: `support_roam_score`
- Modelos quantile: `support_roam_score_quantile`

---

## Script 02 — Baseline de media por campeón

**Input**: `train.parquet`, `val.parquet`.

**Operación**:

1. Calcular media de `support_roam_score` por `ally_utility_champion_id` en train.
2. Predecir en val mapeando por campeón. Si campeón no visto → media global.
3. Calcular métricas.
4. Repetir con `support_roam_score_quantile` como target.

**Output**: `final/baselines/champion_mean_metrics.json`

---

## Script 03 — HistGradientBoostingRegressor

**Input**: `train.parquet`, `val.parquet`.

**Operación**:

1. Usar las mismas feature columns que la MLP (ver abajo).
2. HistGBT maneja categorías nativamente → pasar columnas como categorías
   ordinales, no One-Hot.
3. Entrenar con defaults razonables + un mini grid si hay tiempo.
4. Feature importance.
5. Calcular métricas.
6. Repetir con target quantile.

**Output**: `final/models/gbt/` con modelo, métricas, feature importance.

---

## Script 04 — MLP OneHot

**Input**: `train.parquet`, `val.parquet`.

**Operación**:

1. Reproducir la misma arquitectura que
   `ProgresoActual/scripts/train_support_mlp_regression.py`:
   - OneHotEncoder(handle_unknown='ignore')
   - MLP: Linear(dim→256) → ReLU → BN → Dropout(0.2) → Linear(256→128) →
     ReLU → BN → Dropout(0.2) → Linear(128→1)
   - AdamW, lr=1e-3, weight_decay=1e-4, batch_size=512
   - MSELoss, early stopping patience=15
2. Entrenar con target raw y con target quantile.
3. Guardar best_model.pt, preprocess.joblib, metrics.json, history.csv.

**Output**: `final/models/mlp_onehot/`

---

## Script 05 — Techo empírico

**Input**: `train.parquet` o el dataset completo.

**Operación**:

1. Agrupar por combinaciones frecuentes:
   - solo `ally_utility_champion_id`
   - `ally_utility_champion_id` + `ally_bottom_champion_id` (botlane)
   - `ally_utility_champion_id` + `side`
   - top-50 composiciones completas de 10 campeones
2. Calcular varianza intra-grupo y varianza inter-grupo.
3. Calcular ICC (Intraclass Correlation Coefficient) donde sea viable.
4. Exportar tabla y conclusión sobre cuánta varianza es explicable desde
   composición vs ruido de partida.

**Output**: `final/analysis/ceiling/`

---

## Script 06 — Feature importance

**Input**: modelo GBT entrenado.

**Operación**:

1. Extraer `feature_importances_` del GBT.
2. Agrupar importancias por tipo: campeones aliados, campeones enemigos,
   summoner spells, keystones, rune styles, bans, side.
3. Top-20 features individuales + importancia por grupo.
4. Plot.

**Output**: `final/analysis/feature_importance/`

---

## Script 07 — Comparación de modelos

**Input**: métricas de todos los modelos/baselines.

**Operación**:

1. Tabla comparativa con columnas:
   - Modelo
   - Target (raw / quantile)
   - MSE, RMSE, MAE, R², Pearson, Spearman
   - std(predicciones) / std(target) — ratio de compresión
2. Tabla en CSV, JSON y Markdown.
3. Plot de barras comparativo.

**Output**: `final/analysis/model_comparison/`

---

## Métricas estándar para TODOS los modelos

```python
{
    "mse": ...,
    "rmse": ...,
    "mae": ...,
    "r2": ...,
    "pearson_corr": ...,
    "spearman_corr": ...,
    "pred_std": ...,          # std de las predicciones
    "target_std": ...,        # std del target real
    "compression_ratio": ..., # pred_std / target_std
    "n_train": ...,
    "n_val": ...,
}
```

## Feature groups (referencia)

```python
ROLE_KEYS = ("top", "jungle", "middle", "bottom", "utility")
SIDES = ("ally", "enemy")

# Preset "standard" (el que usaba la MLP actual):
standard = ["champions", "summoner_spells", "context"]

# champions = [f"{s}_{r}_champion_id" for s in SIDES for r in ROLE_KEYS]  → 10 cols
# summoner_spells = [f"{s}_{r}_summoner{i}_id" for s in SIDES for r in ROLE_KEYS for i in (1,2)]  → 20 cols
# context = ["side"]  → 1 col
```
