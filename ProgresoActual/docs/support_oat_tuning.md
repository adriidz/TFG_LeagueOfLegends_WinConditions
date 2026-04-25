# Support OAT tuning

Este flujo sirve para responder de forma ordenada a la pregunta: con que
definicion de etiqueta y con que hiperparametros aprende mejor el modelo de
support.

## Idea

El tuning es `one-at-a-time`:

```text
1. label_weights: cambia solo pesos de la etiqueta
2. time_window: cambia solo ventana start/max
3. train_hparams: cambia solo hiperparametros de entrenamiento
```

La metrica principal para ordenar experimentos es `val_mse`.

## Preparacion local

Antes de lanzar el tuning, deben existir:

```text
ProgresoActual/data/clean/frame_state/support_frame_state_sample5.parquet
ProgresoActual/data/clean/features/draft_features_sample5.parquet
```

Se generan con:

```powershell
.\ProgresoActual\run_support_pipeline.ps1 -SampleFrac 0.05
```

Despues, preparar el tuning:

```powershell
.\ProgresoActual\scripts\run_support_oat_tuning.ps1
```

Smoke test:

```powershell
.\ProgresoActual\scripts\run_support_oat_tuning.ps1 -Smoke
```

El smoke usa por defecto el experimento `support_oat_sample5_m12_smoke` para no
pisar el manifest del tuning completo.

Esto crea:

```text
ProgresoActual/experiments/support_oat/support_oat_sample5_m12/runs_manifest.csv
ProgresoActual/data/clean/scores/oat_tuning/support_oat_sample5_m12/
ProgresoActual/data/training/oat_tuning/support_oat_sample5_m12/
ProgresoActual/analysis/support_label_distribution/oat_tuning/support_oat_sample5_m12/
```

## Sync al cluster

```powershell
.\ProgresoActual\scripts\sync_support_oat_to_cluster.ps1
```

El sync copia solo manifest, scores, model inputs y graficas de etiqueta. No
copia raw, `.venv`, `.venv_cluster` ni modelos antiguos.

## Entrenamiento en cluster

En el cluster, despues de `git pull` o `fetch/reset`:

```bash
wc -l ProgresoActual/experiments/support_oat/support_oat_sample5_m12/runs_manifest.csv
sbatch --array=1-N ProgresoActual/scripts/train_support_oat_array.sh
```

`N` debe ser el numero de filas del manifest menos la cabecera. Para un smoke de
tres runs:

```bash
sbatch --array=1-3 ProgresoActual/scripts/train_support_oat_array.sh
```

Cada run guarda resultados en:

```text
ProgresoActual/models/oat_tuning/support_oat_sample5_m12/<experiment_id>/
```

El job solo activa `.venv_cluster`; no lo crea ni lo modifica.

## Agregacion local

Tras copiar de vuelta `ProgresoActual/models/oat_tuning/support_oat_sample5_m12/`
desde el cluster:

```powershell
python ProgresoActual\scripts\aggregate_support_oat_results.py
```

Salidas:

```text
ProgresoActual/analysis/oat_tuning/support_oat_sample5_m12/experiments_summary.csv
ProgresoActual/analysis/oat_tuning/support_oat_sample5_m12/experiments_summary.md
ProgresoActual/analysis/oat_tuning/support_oat_sample5_m12/best_by_phase.json
ProgresoActual/analysis/oat_tuning/support_oat_sample5_m12/val_mse_ranking.png
ProgresoActual/analysis/oat_tuning/support_oat_sample5_m12/metric_vs_parameter_plots.png
```
