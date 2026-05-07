# Progreso 2026-05-06 - geometry v5 y etiqueta quantile

## Contexto

Carpeta activa de trabajo: `ProgresoActual2`.

Regla de seguridad mantenida: `ProgresoActual` se usa como fuente de lectura
para `frame_state`, `draft_features` y utilidades antiguas, pero los nuevos
artefactos se escriben en `ProgresoActual2`.

Objetivo del dia: mejorar la geometria semantica para `support_roam_score` y
explorar una etiqueta mejor distribuida para entrenamiento.

## Geometria v5 manual

Fuente visual:

- `ProgresoActual2/mapa_editado.png`

Configuracion activa:

- `ProgresoActual2/data/geometry/manual_geometry_v5_config.json`
- version: `geometry_v5_manual_redraw_from_annotation_2`

Modulo de clasificacion:

- `ProgresoActual2/src/geometry/geometry_v5_manual.py`

Render/diagnostico:

- `ProgresoActual2/scripts/plot_geometry_v5_manual.py`
- `ProgresoActual2/analysis/geometry_v5_manual/`

Decisiones importantes:

- `MID_LANE` gana en el cruce central mid-rio-junglas.
- `RIVER_BOT` dejo de ser una diagonal fina atravesando mid.
- `RIVER_BOT` ahora es una transicion corta hacia dragon.
- `RIVER_TOP` queda como pieza superior del rio alrededor de herald/baron.
- `BOT_SIDE_NEAR`, `BLUE_BOT_JUNGLE` y `RED_BOT_JUNGLE` se ajustaron para no romper `RIVER_BOT`.
- Las zonas de carril `TOP_LANE_CORE` y `BOT_LANE_CORE` siguen usando `centerline + width` para clasificacion.

Comandos utiles:

```powershell
.venv\Scripts\python.exe ProgresoActual2\scripts\plot_geometry_v5_manual.py --grid-size 520

.venv\Scripts\python.exe ProgresoActual2\scripts\plot_geometry_v5_manual.py `
  --density-path ProgresoActual2\data\geometry\observed_player_density_5_12.npz `
  --tag m5_12 `
  --grid-size 520
```

Nota: la capa clasificada debe ser igual para `m0_14` y `m5_12` si el JSON y el
`grid-size` son los mismos. Lo que cambia es el heatmap de fondo.

## Distribuciones frame-level de zonas

Script creado:

- `ProgresoActual2/scripts/build_geometry_v5_frame_state_distributions.py`

Salida:

- `ProgresoActual2/analysis/geometry_v5_manual/frame_state_distributions/`

Ventana `m5_12`:

- frames vivos clasificados: `2,245,763`
- match_ids: `168,564`
- match-team keys: `337,128`
- `support_in_bot_context_v5_share`: `0.741890`
- legacy `support_in_bot_extended_share`: `0.760169`

Ventana `m0_14`:

- frames vivos clasificados: `4,525,059`
- match_ids: `168,564`
- match-team keys: `337,128`
- `support_in_bot_context_v5_share`: `0.719144`
- legacy `support_in_bot_extended_share`: `0.732074`

Comandos:

```powershell
.venv\Scripts\python.exe ProgresoActual2\scripts\build_geometry_v5_frame_state_distributions.py `
  --start-minute 5 `
  --max-minute 12 `
  --tag m5_12 `
  --chunk-size 750000

.venv\Scripts\python.exe ProgresoActual2\scripts\build_geometry_v5_frame_state_distributions.py `
  --start-minute 0 `
  --max-minute 14 `
  --tag m0_14 `
  --chunk-size 750000
```

Importante: esto no es todavia la etiqueta agregada. Es distribucion frame-level
de zonas y del booleano `support_in_bot_context_v5`.

## Etiqueta agregada con geometria v5

Script creado:

- `ProgresoActual2/scripts/build_support_roam_score_v5_distribution.py`

Parquet exportado:

- `ProgresoActual2/data/clean/scores/support_scores_v5_geometry_m12.parquet`

Salida de analisis:

- `ProgresoActual2/analysis/support_roam_score_v5_geometry/`

Receta usada:

```text
raw = 0.45 * outside_ratio_v5
    + 0.35 * far_ratio_v5
    + 0.20 * xp_gap_v5

support_roam_score_v5_geometry = raw ** 0.75
```

La receta es la misma idea seleccionada en v3, pero usando geometria v5:

- `outside_ratio_v5` se calcula con `support_in_bot_context_v5`
- dentro de bot para v5: `BOT_LANE_CORE`, `BOT_SIDE_NEAR`, `RIVER_BOT`, `DRAGON_AREA`
- bases de support y ADC se excluyen con la geometria manual v5

Resumen full:

- filas: `337,104`
- coverage: `0.999929`
- mean: `0.392561`
- median: `0.389645`
- q05: `0.091024`
- q95: `0.711345`
- q99: `0.839280`
- share_eq_0: `0.018522`
- share_eq_1: `0.000727`
- row_corr_vs_v3: `0.940635`
- mean_delta_v5_minus_v3: `+0.020020`
- median_delta_v5_minus_v3: `0.000000`

Comando:

```powershell
.venv\Scripts\python.exe ProgresoActual2\scripts\build_support_roam_score_v5_distribution.py --export-scores
```

## Transformacion quantile

Motivacion:

La transformacion `gamma=0.75` estira la etiqueta, pero sigue siendo una
calibracion manual. Se probo una alternativa tipo `QuantileTransformer` para
aplanar la distribucion de la etiqueta.

Script creado:

- `ProgresoActual2/scripts/build_support_roam_score_v5_quantile_labels.py`

Parquet exportado:

- `ProgresoActual2/data/clean/scores/support_scores_v5_quantile_m12.parquet`

Salida de analisis:

- `ProgresoActual2/analysis/support_roam_score_v5_quantile/`

Columnas nuevas:

- `support_roam_score_v5_quantile`
- `support_roam_score_v5_quantile_zero_preserved`

La version recomendada para probar primero es:

- `support_roam_score_v5_quantile_zero_preserved`

Razon:

- mantiene `raw == 0` como `0`
- aplana los scores positivos
- conserva el orden de ranking de la etiqueta raw
- es menos arbitraria que `raw ** 0.75`

Resumen:

| columna | mean | median | q05 | q95 | share_eq_0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `raw_support_roam_score_v5_geometry` | 0.303160 | 0.284594 | 0.040946 | 0.635000 | 0.018522 |
| `support_roam_score_v5_geometry` | 0.392561 | 0.389645 | 0.091024 | 0.711345 | 0.018522 |
| `support_roam_score_v5_quantile` | 0.499826 | 0.499986 | 0.049997 | 0.949950 | 0.018522 |
| `support_roam_score_v5_quantile_zero_preserved` | 0.490736 | 0.490574 | 0.032082 | 0.948949 | 0.018525 |

Comando:

```powershell
.venv\Scripts\python.exe ProgresoActual2\scripts\build_support_roam_score_v5_quantile_labels.py
```

Nota metodologica:

Para exploracion y construir un primer model input, la transformacion quantile
global es aceptable. Para una evaluacion estricta tipo `TransformedTargetRegressor`,
el quantile transformer debe ajustarse solo con el split de train y aplicarse a
valid/test.

## Como adaptar al entrenamiento

El entrenamiento actual principal no usa `RandomForestRegressor` con sklearn
puro, sino un pipeline de parquet + model input + MLP/PyTorch. Por eso la forma
mas limpia de adaptar la idea es:

1. Elegir una columna de score en el parquet de support.
2. Construir model input renombrandola a la columna canonica `support_roam_score`.
3. Entrenar el modelo igual que antes.

Para probar quantile zero-preserved como target:

```powershell
.venv\Scripts\python.exe ProgresoActual\src\02_data_processing\build_support_model_input.py `
  --support-scores-path ProgresoActual2\data\clean\scores\support_scores_v5_quantile_m12.parquet `
  --support-score-source-col support_roam_score_v5_quantile_zero_preserved `
  --out-path ProgresoActual2\data\training\model_input_support_regression_v5_quantile_zero_m12.parquet `
  --summary-dir ProgresoActual2\data\training\model_input_support_regression_v5_quantile_zero_m12_analysis `
  --join-how inner
```

Luego entrenar apuntando a ese model input:

```powershell
.venv\Scripts\python.exe ProgresoActual\scripts\train_support_mlp_regression.py `
  --input ProgresoActual2\data\training\model_input_support_regression_v5_quantile_zero_m12.parquet `
  --outdir ProgresoActual2\models\support_mlp_regression_v5_quantile_zero_m12 `
  --target-col support_roam_score `
  --feature-groups standard
```

Nota: estos comandos leen scripts en `ProgresoActual`, pero escriben outputs en
`ProgresoActual2`.

## Artefactos principales

Geometria:

- `ProgresoActual2/data/geometry/manual_geometry_v5_config.json`
- `ProgresoActual2/src/geometry/geometry_v5_manual.py`
- `ProgresoActual2/analysis/geometry_v5_manual/geometry_v5_manual_zone_layer_m0_14.png`
- `ProgresoActual2/analysis/geometry_v5_manual/geometry_v5_manual_zone_layer_m5_12.png`

Scores:

- `ProgresoActual2/data/clean/scores/support_scores_v3_m12.parquet`
- `ProgresoActual2/data/clean/scores/support_scores_v5_geometry_m12.parquet`
- `ProgresoActual2/data/clean/scores/support_scores_v5_quantile_m12.parquet`

Analisis:

- `ProgresoActual2/analysis/support_roam_score_v5_geometry/`
- `ProgresoActual2/analysis/support_roam_score_v5_quantile/`
- `ProgresoActual2/analysis/geometry_v5_manual/frame_state_distributions/`

Scripts nuevos:

- `ProgresoActual2/scripts/build_geometry_v5_frame_state_distributions.py`
- `ProgresoActual2/scripts/build_support_roam_score_v5_distribution.py`
- `ProgresoActual2/scripts/build_support_roam_score_v5_quantile_labels.py`

## Proximo paso recomendado

1. Construir model input con `support_roam_score_v5_quantile_zero_preserved`.
2. Entrenar una corrida MLP comparable a la anterior.
3. Comparar:
   - MSE/MAE/R2/Spearman en validacion
   - distribucion de predicciones
   - ranking medio por campeon
   - estabilidad blue vs red
4. Si quantile zero-preserved funciona bien, documentar que la etiqueta final es
   una etiqueta relativa/percentilizada y no una magnitud fisica absoluta.
