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
  -> label distribution analysis
  -> model_input support-only
  -> train MLP regression
  -> champion/reference analysis + reportes
```

## Ejecucion recomendada en cluster

El flujo normal del reinicio se ejecuta en dos jobs Slurm. El primero no reserva
GPU y deja preparado el parquet de entrenamiento; el segundo reserva GPU y
entrena el MLP.

```bash
sbatch ProgresoActual/scripts/prepare_cluster_support_data.sh
sbatch ProgresoActual/scripts/train_cluster_support_mlp.sh
```

Si quieres encadenarlos para que el entrenamiento espere a que la preparacion
termine correctamente:

```bash
jid=$(sbatch --parsable ProgresoActual/scripts/prepare_cluster_support_data.sh)
sbatch --dependency=afterok:$jid ProgresoActual/scripts/train_cluster_support_mlp.sh
```

Los defaults actuales son `sample5` y ventana `m12`, por lo que las salidas
principales esperadas son:

```text
ProgresoActual/data/clean/frame_state/support_frame_state_sample5.parquet
ProgresoActual/data/clean/features/draft_features_sample5.parquet
ProgresoActual/data/clean/scores/support_scores_sample5_m12.parquet
ProgresoActual/data/training/model_input_support_regression_sample5_m12.parquet
ProgresoActual/analysis/support_label_distribution/sample5_m12/
ProgresoActual/models/support_mlp_sample5_m12/
```

## Ejecucion local opcional

El PowerShell local queda como preparacion/smoke test hasta justo antes del
entrenamiento:

```powershell
.\ProgresoActual\run_support_pipeline.ps1 -SampleFrac 0.05
```

Para una prueba rapida de preparacion:

```powershell
.\ProgresoActual\run_support_pipeline.ps1 -SampleFrac 0.05 -MaxMatches 200
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
  --support-scores-path ProgresoActual\data\clean\scores\support_scores_sample5_m12.parquet `
  --out-path ProgresoActual\data\training\model_input_support_regression_sample5_m12.parquet
```

5. Generar graficas de distribucion de la etiqueta:

```powershell
python ProgresoActual\scripts\plot_support_label_distribution.py `
  --support-scores-path ProgresoActual\data\clean\scores\support_scores_sample5_m12.parquet `
  --outdir ProgresoActual\analysis\support_label_distribution\sample5_m12
```

6. Entrenar el MLP support-only en cluster:

```bash
sbatch ProgresoActual/scripts/train_cluster_support_mlp.sh
```

7. Construir referencia de campeones y comparar:

```powershell
python ProgresoActual\scripts\build_champion_support_reference.py --manual-only

python ProgresoActual\scripts\compare_support_champion_reference.py `
  --support-scores-path ProgresoActual\data\clean\scores\support_scores_sample5_m12.parquet `
  --reference-path ProgresoActual\references\champion_support_reference.csv
```

## Artefactos esperados

- `ProgresoActual/data/clean/frame_state/support_frame_state*.parquet`: cache cara desde raw.
- `ProgresoActual/analysis/support_grid/*`: grid de heuristicas de support.
- `ProgresoActual/data/clean/scores/support_scores*_mXX.parquet`: score seleccionado.
- `ProgresoActual/data/training/model_input_support_regression*.parquet`: entrada de entrenamiento aislada.
- `ProgresoActual/analysis/support_label_distribution/*`: resumenes y graficas de distribucion de etiquetas.
- `ProgresoActual/models/support_mlp_regression*`: modelos, metricas e historiales.
- `ProgresoActual/analysis/champion_reference/*`: comparacion por campeon.
- `ProgresoActual/docs/informe_progreso_reinicio.md`: informe editable.

## Diario de iteraciones

Cada sesion de trabajo se resume en `ProgresoActual/docs/iteration_log.md`.
Antes de mandar resultados nuevos a una ruta no cubierta por esta politica, se
debe decidir explicitamente la ruta de destino.
