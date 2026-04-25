# Diseno experimental 170k

Este documento aterriza el paso de una prueba simple sobre `sample5` a un flujo
defendible sobre el dataset completo actual. La idea es crear una cache cara una
vez, analizar salud de etiquetas y datos, entrenar una baseline y luego hacer
tuning `one-at-a-time`.

## 1. Congelar snapshot completo

El raw se queda en local. Este paso lee todos los JSON actuales y genera las dos
tablas base del experimento:

```powershell
.\ProgresoActual\scripts\run_full_snapshot.ps1
```

Salidas:

```text
ProgresoActual/data/clean/frame_state/support_frame_state.parquet
ProgresoActual/data/clean/features/draft_features.parquet
```

Este es el cuello de botella. Por defecto el `support_frame_state.parquet` se
escribe como dataset Parquet por partes dentro de una carpeta con ese nombre,
con un marcador `_SUCCESS` al finalizar. Esto reduce presion de memoria y deja
partes parciales en disco durante el proceso. Una vez termine, no hay que volver
a leer raw para probar pesos, ventanas o hiperparametros sobre este snapshot.

Si quieres cambiar el tamano de escritura parcial:

```powershell
.\ProgresoActual\scripts\run_full_snapshot.ps1 -FrameStateChunkMatches 2500
```

Si necesitas el comportamiento antiguo monolitico:

```powershell
.\ProgresoActual\scripts\run_full_snapshot.ps1 -FrameStateWriteMode single
```

## 2. Baseline full m12

Con el snapshot ya creado:

```powershell
.\ProgresoActual\run_support_pipeline.ps1 -SampleFrac 1 -SkipFrameState -SkipDraftFeatures
```

Salidas:

```text
ProgresoActual/data/clean/scores/support_scores_m12.parquet
ProgresoActual/data/training/model_input_support_regression_m12.parquet
ProgresoActual/analysis/support_label_distribution/full_m12/
```

Revisar:

```text
support_label_histogram.png
support_label_cdf.png
support_label_by_champion_summary.csv
support_label_distribution_summary.json
```

Criterio de salud: target en `[0,1]`, cobertura alta, distribucion no colapsada
y sin masa excesiva en los extremos.

## 3. Baseline training en cluster

Subir artefactos full:

```powershell
.\ProgresoActual\scripts\sync_support_artifacts_to_cluster.ps1 -SampleTag full -WindowTag m12
```

Entrenar en cluster:

```bash
SAMPLE_TAG=full WINDOW_TAG=m12 sbatch ProgresoActual/scripts/train_cluster_support_mlp.sh
```

Salida esperada:

```text
ProgresoActual/models/support_mlp_full_m12/
```

Revisar:

```text
history.csv
metrics.json
diagnostics/loss_curve.png
diagnostics/true_vs_pred_scatter.png
diagnostics/residual_histogram.png
```

El objetivo es comprobar que el modelo aprende antes de gastar tiempo en tuning.

## 4. Tuning OAT full

Preparar variantes en local:

```powershell
.\ProgresoActual\scripts\run_support_oat_tuning.ps1 -ExperimentName support_oat_full_m12 -SampleTag full -SampleFrac 1
```

Subir al cluster:

```powershell
.\ProgresoActual\scripts\sync_support_oat_to_cluster.ps1 -ExperimentName support_oat_full_m12
```

En cluster:

```bash
N=$(($(wc -l < ProgresoActual/experiments/support_oat/support_oat_full_m12/runs_manifest.csv)-1))
EXPERIMENT_NAME=support_oat_full_m12 sbatch --array=1-$N ProgresoActual/scripts/train_support_oat_array.sh
```

El tuning cambia una cosa cada vez:

```text
label_weights
time_window
train_hparams
```

La metrica principal es `val_mse`.

## 5. Agregacion y seleccion

Tras copiar de vuelta los modelos del cluster:

```powershell
scp -P 55022 -r adiaz@158.109.75.51:/fhome/adiaz/TFG_LeagueOfLegends_WinConditions/ProgresoActual/models/oat_tuning/support_oat_full_m12 .\ProgresoActual\models\oat_tuning\
```

Agregar:

```powershell
.\.venv\Scripts\python.exe ProgresoActual\scripts\aggregate_support_oat_results.py --experiment-name support_oat_full_m12
```

Salidas:

```text
ProgresoActual/analysis/oat_tuning/support_oat_full_m12/experiments_summary.csv
ProgresoActual/analysis/oat_tuning/support_oat_full_m12/experiments_summary.md
ProgresoActual/analysis/oat_tuning/support_oat_full_m12/best_by_phase.json
ProgresoActual/analysis/oat_tuning/support_oat_full_m12/val_mse_ranking.png
ProgresoActual/analysis/oat_tuning/support_oat_full_m12/metric_vs_parameter_plots.png
```

## Resultado esperado

Al final del ciclo tendras:

- snapshot completo reutilizable;
- baseline full con salud de etiqueta y metricas;
- diagnosticos visuales de entrenamiento;
- ranking OAT por `val_mse`;
- configuracion candidata para defender en la memoria;
- protocolo claro para repetir cuando el collector llegue a 250k.

## Ciclo 250k

Cuando el collector llegue a 250k, repetir:

```powershell
.\ProgresoActual\scripts\run_full_snapshot.ps1
.\ProgresoActual\run_support_pipeline.ps1 -SampleFrac 1 -SkipFrameState -SkipDraftFeatures
```

Despues reentrenar la configuracion candidata y solo los experimentos clave,
no necesariamente todo el grid.
