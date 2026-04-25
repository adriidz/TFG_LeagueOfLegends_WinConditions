# ProgresoActual - Reinicio support-only

Este directorio contiene el reinicio ordenado del TFG centrado en regresion continua
del comportamiento de support. La idea es conservar el trabajo anterior como
historial, pero avanzar desde aqui con un pipeline mas pequeno, medible y
defendible.

## Flujo principal

Regla de aislamiento: todo script que viva dentro de `ProgresoActual/` debe
guardar sus salidas dentro de `ProgresoActual/`. Las carpetas `data/` y
`data_new/` pueden usarse como entradas externas heredadas, pero no como destino
por defecto del reinicio.

Cuando se regeneren todas las etiquetas y features del reinicio, `data_new/`
dejara de ser una entrada normal. Solo debe usarse como puente temporal mientras
falte algun generador nuevo equivalente dentro de `ProgresoActual/`.

Orden conceptual del pipeline nuevo:

```text
collector/raw
  -> support_frame_state + draft_features
  -> support_scores continuos
  -> model_input support-only
  -> train MLP regression
  -> champion/reference analysis + reportes
```

## Ejecucion recomendada

Para ejecutar el pipeline limpio support-only con rutas aisladas:

```powershell
.\ProgresoActual\run_support_pipeline.ps1 -SampleFrac 0.05 -Epochs 40
```

Para una prueba rapida sin entrenamiento largo:

```powershell
.\ProgresoActual\run_support_pipeline.ps1 -SampleFrac 0.05 -MaxMatches 200 -Epochs 2
```

## Pasos manuales equivalentes

1. Extraer una cache de frames de support una sola vez:

```powershell
python ProgresoActual\src\02_data_processing\new_02a_extract_support_frame_state.py --raw-root data\raw\raw --region europe --sample-frac 0.05
```

2. Construir `draft_features` desde `match.json`:

```powershell
python ProgresoActual\src\02_data_processing\build_draft_features.py `
  --raw-root data\raw\raw `
  --region europe `
  --sample-frac 0.05
```

3. Probar heuristicas rapidamente desde la cache y exportar una configuracion:

```powershell
python ProgresoActual\src\02_data_processing\new_02b_grid_support_scores.py `
  --sample-frac 0.05 `
  --start-minutes 4 5 6 `
  --max-minutes 10 11 12 `
  --far-adc-thresholds 2200 2500 2800 `
  --weight-triplets 0.45,0.35,0.20 0.35,0.45,0.20 `
  --champion-summary `
  --export-best score_iqr `
  --write-config-json
```

4. Construir el model input support-only:

```powershell
python ProgresoActual\src\02_data_processing\build_support_model_input.py `
  --draft-path ProgresoActual\data\clean\features\draft_features_sample5.parquet `
  --support-scores-path ProgresoActual\data\clean\scores\support_scores_sample5_m11.parquet `
  --out-path ProgresoActual\data\training\model_input_support_regression_sample5_m11.parquet
```

5. Entrenar el MLP support-only:

```powershell
python ProgresoActual\scripts\train_support_mlp_regression.py `
  --input ProgresoActual\data\training\model_input_support_regression_sample5_m11.parquet `
  --feature-groups standard `
  --epochs 40 `
  --wandb `
  --wandb-mode offline `
  --support-config-json ProgresoActual\data\clean\scores\selected_support_score_config.json
```

6. Construir referencia de campeones y comparar:

```powershell
python ProgresoActual\scripts\build_champion_support_reference.py --manual-only

python ProgresoActual\scripts\compare_support_champion_reference.py `
  --support-scores-path ProgresoActual\data\clean\scores\support_scores_sample5_m11.parquet `
  --reference-path ProgresoActual\references\champion_support_reference.csv
```

## Artefactos esperados

- `ProgresoActual/data/clean/frame_state/support_frame_state*.parquet`: cache cara desde raw.
- `ProgresoActual/analysis/support_grid/*`: grid de heuristicas de support.
- `ProgresoActual/data/clean/scores/support_scores*_mXX.parquet`: score seleccionado.
- `ProgresoActual/data/training/model_input_support_regression*.parquet`: entrada de entrenamiento aislada.
- `ProgresoActual/models/support_mlp_regression*`: modelos, metricas e historiales.
- `ProgresoActual/analysis/champion_reference/*`: comparacion por campeon.
- `ProgresoActual/docs/informe_progreso_reinicio.md`: informe editable.

## Diario de iteraciones

Cada sesion de trabajo se resume en `ProgresoActual/docs/iteration_log.md`.
Antes de mandar resultados nuevos a una ruta no cubierta por esta politica, se
debe decidir explicitamente la ruta de destino.
