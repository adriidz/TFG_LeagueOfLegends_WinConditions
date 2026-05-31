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
5. Split a tres niveles por `match_id`:
   - Primer `GroupShuffleSplit(test_size=0.15, random_state=42)` → separa test.
   - Segundo `GroupShuffleSplit(test_size=0.176, random_state=42)` sobre el resto
     → separa val (~15% del total). Resultado: ~70/15/15.
6. Fittear `QuantileTransformer(output_distribution='uniform', n_quantiles=1000)`
   SOLO sobre los targets de train.
7. Crear columna `support_roam_score_quantile` en train (fit_transform), val y
   test (transform). Para la variante zero-preserved: fittear solo sobre rows con
   score > 0 en train, dejar score=0 como 0.
8. Guardar como tres parquets separados:
   - `final/data/training/train.parquet`
   - `final/data/training/val.parquet`
   - `final/data/training/test.parquet`
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
2. Codificar con `OrdinalEncoder(handle_unknown='use_encoded_value',
   unknown_value=-1)` fitteado SOLO en train.
3. Declarar `categorical_features=True` (o lista de índices) al construir
   `HistGradientBoostingRegressor` para que el modelo trate cada columna como
   categórica y no imponga orden artificial entre IDs.
4. Entrenar con defaults razonables + un mini grid si hay tiempo.
5. Calcular métricas en val.
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

1. Usar `sklearn.inspection.permutation_importance` sobre el GBT y el set de
   val. Esto mide el impacto real de cada feature en la métrica, sin depender
   de la implementación interna del modelo.
2. Agrupar importancias por tipo: campeones aliados, campeones enemigos,
   summoner spells, keystones, rune styles, bans, side.
3. Top-20 features individuales + importancia por grupo.
4. Plot.

**Output**: `final/analysis/feature_importance/`

---

## Script 07 — Comparación de modelos

**Input**: métricas de todos los modelos/baselines.

**Operación**:

1. Dos tablas separadas:
   - **Tabla A**: modelos evaluados en escala raw.
   - **Tabla B**: modelos evaluados en escala quantile.
   Para modelos entrenados con target quantile, inverse-transform las
   predicciones a escala raw con el `QuantileTransformer` guardado y añadir
   sus métricas también a la Tabla A. Esto permite comparación directa.
2. Columnas de cada tabla:
   - Modelo
   - MSE, RMSE, MAE, R², Pearson, Spearman
   - std(predicciones) / std(target) — ratio de compresión
3. Métrica común para ranking final: **Spearman** (invariante a transformaciones
   monótonas).
4. Evaluación final en **test** (no val). Val solo se usa durante desarrollo.
5. Tabla en CSV, JSON y Markdown + plot de barras comparativo.

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
    "n_eval": ...,             # filas del split evaluado (val o test)
    "eval_split": ...,         # "val" durante desarrollo, "test" en evaluación final
}
```

---

## Script 08 - SHAP analysis

**Input**: `final/models/gbt/gbt_model_raw.joblib`,
`final/models/gbt/preprocess.joblib`, `train.parquet`, `test.parquet`.

**Operacion**:

1. Cargar el HistGBT base entrenado sobre `support_roam_score`.
2. Reutilizar exactamente el `OrdinalEncoder` y las `feature_columns` guardadas.
3. Muestrear train como background (`--background-size`, default 200) y test
   como muestra explicada (`--sample-size`, default 2000).
4. Intentar `shap.TreeExplainer`; si la version de SHAP/sklearn no soporta el
   estimador o produce valores no aditivos para este HistGBT categorico, usar
   `shap.PermutationExplainer` como fallback reproducible.
5. Exportar importancia global SHAP, summary plots, dependencia categorica por
   support/ADC aliado y waterfalls locales.

**Output**: `final/analysis/shap/`

- `shap_global_importance.csv`
- `shap_summary_bar.png`
- `shap_summary_beeswarm.png`
- `shap_dependence_ally_utility_champion_id.png`
- `shap_dependence_ally_bottom_champion_id.png`
- `shap_local_top_cases.csv`
- `shap_waterfall_case_*.png`
- `shap_metadata.json`

**Limitacion de lectura**: las features categoricas estan codificadas con
ordinales internos del modelo. Los SHAP values deben interpretarse como
contribuciones asociativas del modelo, no como efectos causales ni como orden
real entre campeones.

---

## Script 09 - Auditoria cualitativa consolidada

**Input**: `test.parquet`, `gbt_model_raw.joblib`, `preprocess.joblib`,
`support_scores_v5_geometry_m12.parquet`, `support_frame_state.parquet`,
geometria v5 y JSON raw de Riot (`match.json`, `timeline.json`).

**Operacion**:

1. Evaluar el HistGBT base sobre test y calcular `prediction`, `actual`,
   `signed_error` y `abs_error`.
2. Seleccionar 20 mayores errores y 20 menores errores. Los menores errores se
   estratifican por score real: very-low, low-mid, high-mid y very-high.
3. Unir componentes de etiqueta (`outside_ratio_v5`, `far_ratio_v5`,
   `xp_gap_v5`, frames validos y confidence).
4. Recuperar frames minuto 5-12 con posiciones `support_x/y` y `adc_x/y`,
   zonas v5, distancia support-ADC y flags `out_bot_context_v5` /
   `far_from_adc_v5`.
5. Reconstruir la etiqueta desde frames para verificar que coincide con el score
   guardado.
6. Extraer eventos reales del timeline entre minuto 0 y 12: kills, assists,
   muertes, objetivos, placas y estructuras.
7. Generar mapas cronologicos por caso con trayectoria de support y ADC sobre la
   geometria v5, mas una figura temporal de distancia/flags/XP.

**Output**: `final/analysis/qualitative_case_audit/`

- `case_index.csv`
- `case_event_timeline.csv`
- `case_frame_timeline.csv`
- `case_notes.md`
- `case_plots/*_map.png`
- `case_plots/*_timeline.png`
- `metadata.json`

**Criterio de calidad**: `max_score_reconstruction_delta` y
`max_raw_score_reconstruction_delta` deben ser 0 o numericamente despreciables.
Los tags de evidencia (`chaotic_early_game`, `clean_roam_like_candidate`,
`label_quality_caution`, `accurate_low/mid/high`) son ayudas conservadoras para
revision, no etiquetas causales.

**Nota de limpieza**: `09_error_analysis.py`, `10_label_error_diagnostics.py` y
`11_qualitative_match_context.py` quedan como legado hasta que se archive la
iteracion anterior; el informe final debe citar el consolidado.

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
