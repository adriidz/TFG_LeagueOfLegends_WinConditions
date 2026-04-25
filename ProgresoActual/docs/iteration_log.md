# Diario de iteraciones

Este archivo documenta las decisiones y cambios realizados en cada iteracion de
trabajo sobre el reinicio del TFG.

## Iteracion 1 - Plan de reinicio support-only

- Se definio el nuevo foco metodologico: support-only, score continuo en `[0, 1]`
  y regresion en lugar de clasificacion.
- Se decidio usar `support_frame_state` como cache reutilizable para no releer
  los JSON raw en cada variante de etiqueta.
- Se eligio comparar la distribucion agregada por campeon contra una referencia
  experta/oficial, entendida como prior de campeon y no como ground truth por
  partida.

## Iteracion 2 - Primera implementacion funcional

- Se extendio el scorer de support para probar grids de heuristicas y exportar
  una configuracion como `support_scores.parquet`.
- Se creo un trainer support-only con `OneHotEncoder + MLP + MSELoss`.
- Se anadieron scripts para construir referencia de campeones y comparar medias
  de score generado contra el prior experto.
- Se redacto un primer informe editable y una planificacion de 8 semanas.
- Se ejecutaron smoke tests con `sample5` para comprobar scorer, model input,
  trainer y comparacion por campeon.

## Iteracion 3 - Separacion fisica del repositorio

- Se creo `ProgresoActual/` como espacio del reinicio.
- Se archivo el trabajo previo visible en `PropuestaInicial/`.
- Se mantuvieron `data/` y `data_new/` en raiz como datos/caches reutilizables y
  pesadas, no como espacio de resultados nuevos.
- Se movieron los scripts minimos del reinicio a `ProgresoActual/src/`.

## Iteracion 4 - Politica de aislamiento de artefactos

- Se acordo que todo artefacto generado por codigo dentro de `ProgresoActual/`
  debe guardarse tambien dentro de `ProgresoActual/`.
- Se actualizaron defaults de scripts para escribir en:
  - `ProgresoActual/data/clean/frame_state/`
  - `ProgresoActual/analysis/support_grid/`
  - `ProgresoActual/data/clean/scores/`
  - `ProgresoActual/data/training/`
  - `ProgresoActual/models/`
  - `ProgresoActual/analysis/champion_reference/`
- Queda como norma preguntar antes si aparece una salida nueva cuya ubicacion no
  este clara.

## Iteracion 5 - Contrato del pipeline nuevo

- Se acordo que, si se redefinen todas las etiquetas, tambien debe regenerarse
  desde cero todo lo que entra al modelo.
- A partir de esa decision, `data_new/` pasa a ser solo un puente temporal para
  smoke tests o comparaciones historicas, no una dependencia normal del reinicio.
- El pipeline estable debera leer y escribir dentro de `ProgresoActual/`, salvo
  dos excepciones explicitas:
  - raw externo, si se decide no duplicar las partidas por tamano;
  - credenciales/configuracion local, como `TFG.env`.
- El orden conceptual nuevo queda definido como:
  collector/raw -> frame_state + draft_features -> support_scores continuos ->
  model_input support-only -> entrenamiento MLP -> analisis por campeon/reportes.

## Iteracion 6 - `build_draft_features`

- Se creo `ProgresoActual/src/02_data_processing/build_draft_features.py`.
- El script lee solo `match.json`, no timelines, y genera una fila por
  `(match_id, team_id)` con perspectiva ally/enemy.
- La salida por defecto queda aislada en
  `ProgresoActual/data/clean/features/draft_features*.parquet`.
- Se mantiene compatibilidad de columnas con el trainer actual:
  campeones, summoner spells, runas, bans, side, patch y metadatos de partida.

## Iteracion 7 - Model input support-only y orquestador

- Se creo `ProgresoActual/src/02_data_processing/build_support_model_input.py`.
- El builder nuevo une solo `draft_features + support_scores`; ya no requiere
  `jungle_scores` ni `team_scores`.
- Se ajusto el trainer para que su input por defecto sea
  `ProgresoActual/data/training/model_input_support_regression.parquet`.
- Se creo `ProgresoActual/run_support_pipeline.ps1` como orquestador del flujo:
  frame-state, draft-features, support-scores, model-input, training y analisis
  de campeones.

## Iteracion 8 - Separacion local/cluster

- Se ajusto `ProgresoActual/run_support_pipeline.ps1` para que solo prepare
  datos hasta `model_input_support_regression`.
- El entrenamiento queda fuera del PowerShell local y pasa al script Slurm
  `ProgresoActual/scripts/train_cluster_support_mlp.sh`.
- Se adapto el `.sh` del cluster para usar:
  `ProgresoActual/scripts/train_support_mlp_regression.py` y
  `ProgresoActual/data/training/model_input_support_regression_sample5_m11.parquet`.
- El `.sh` guarda resultados en `ProgresoActual/models/` y metadatos de run en
  `ProgresoActual/cluster_run_metadata/`.

## Iteracion 9 - Dos jobs Slurm y ventana m12

- Se separo el flujo del cluster en dos trabajos:
  `prepare_cluster_support_data.sh` para preparar datos sin GPU y
  `train_cluster_support_mlp.sh` para entrenar con GPU.
- El job de preparacion ejecuta, dentro de `ProgresoActual/`, la construccion de
  `draft_features`, la extraccion de `support_frame_state`, el grid/export de
  `support_scores` y el `model_input` support-only.
- Se cambio el default de ventana a `m12`, manteniendo `sample5` como benchmark
  inicial.
- Las salidas por defecto del flujo cluster quedan:
  - `ProgresoActual/data/clean/frame_state/support_frame_state_sample5.parquet`
  - `ProgresoActual/data/clean/features/draft_features_sample5.parquet`
  - `ProgresoActual/data/clean/scores/support_scores_sample5_m12.parquet`
  - `ProgresoActual/data/training/model_input_support_regression_sample5_m12.parquet`
- El PowerShell local sigue existiendo como preparacion o smoke test, tambien con
  `m12` por defecto.

## Iteracion 10 - Graficas de distribucion de etiquetas

- Se creo `ProgresoActual/scripts/plot_support_label_distribution.py` para
  resumir y visualizar la etiqueta continua de support antes de entrenar.
- El script genera resumen global CSV/JSON, histograma, CDF empirica,
  comparacion por componentes heuristicas, distribucion por side y resumen/boxplot
  por campeon cuando hay suficientes muestras.
- Se integro esta fase en `prepare_cluster_support_data.sh`, despues de exportar
  `support_scores` y construir el `model_input`.
- Se integro tambien en `run_support_pipeline.ps1` para que los smoke tests
  locales de preparacion dejen las mismas figuras dentro de `ProgresoActual/`.

## Iteracion 11 - Pipeline local real y cluster solo entrenamiento

- Se decidio que el raw completo no se copia al cluster porque pesa mucho y se
  actualiza constantemente durante la recoleccion.
- `run_support_pipeline.ps1` queda como pipeline local oficial de preparacion:
  raw local -> caches/features -> scores -> model input -> graficas.
- Se creo `sync_support_artifacts_to_cluster.ps1` para copiar al cluster solo el
  `model_input`, los `support_scores`, la config de etiqueta y las graficas de
  distribucion.
- `train_cluster_support_mlp.sh` queda como unico pipeline cluster oficial y
  solo entrena en GPU sobre artefactos ya sincronizados.
- Se retiro `prepare_cluster_support_data.sh` porque dependia de tener raw en
  cluster e inducia a un flujo que no encaja con el entorno real.
- Se anadio `.venv_cluster/` a `.gitignore`; el job de entrenamiento solo activa
  ese entorno y falla si no existe, sin crearlo ni sobreescribirlo.
