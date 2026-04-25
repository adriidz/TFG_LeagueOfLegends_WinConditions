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

## Iteracion 12 - Visualizacion de training y tuning OAT

- Se anadio `plot_training_run_diagnostics.py` para generar curvas de loss,
  scatter true-vs-pred, residuos y error por bins a partir de cada entrenamiento.
- `train_support_mlp_regression.py` llama a esos diagnosticos al final de cada
  run, salvo que se use `--skip-diagnostics-plots`.
- Se creo `run_support_oat_tuning.ps1` para preparar experimentos `one-at-a-time`
  sobre pesos de etiqueta, ventanas temporales e hiperparametros de entrenamiento
  sin releer raw.
- Se creo `train_support_oat_array.sh` para ejecutar esos experimentos como Slurm
  array en el cluster, activando solamente `.venv_cluster`.
- Se creo `sync_support_oat_to_cluster.ps1` para subir manifest, scores, model
  inputs y graficas de etiqueta al cluster sin copiar raw ni entornos.
- Se creo `aggregate_support_oat_results.py` para comparar runs por `val_mse` y
  producir resumenes CSV/Markdown, ranking y mejores configs por fase.
- Se documento el procedimiento en `ProgresoActual/docs/support_oat_tuning.md`.

## Iteracion 13 - Diseno experimental full 170k

- Se documento el protocolo completo para pasar de `sample5` al dataset completo
  en `ProgresoActual/docs/experimental_design_170k.md`.
- Se creo `run_full_snapshot.ps1` como wrapper local del paso caro:
  `raw -> support_frame_state.parquet + draft_features.parquet`.
- Se ajusto `train_cluster_support_mlp.sh` para que `SAMPLE_TAG=full` resuelva
  por defecto `model_input_support_regression_m12.parquet`, sin requerir
  `INPUT_PATH` manual.
- Se actualizo el mensaje de `sync_support_artifacts_to_cluster.ps1` para mostrar
  el comando de entrenamiento correcto tambien para `full`.

## Iteracion 14 - Frame-state por partes

- Se detecto que la extraccion full perdia velocidad al acumular millones de
  filas en memoria antes de escribir el parquet final.
- Se anadio a `new_02a_extract_support_frame_state.py` el modo
  `--write-mode dataset`, que escribe partes Parquet incrementales dentro de
  `support_frame_state.parquet` y crea `_SUCCESS` al completar.
- `run_full_snapshot.ps1` usa por defecto ese modo con
  `-FrameStateChunkMatches 5000`, manteniendo la misma logica de extraccion.
- `new_02b_grid_support_scores.py` valida que el dataset por partes tenga
  `_SUCCESS` antes de leerlo, para evitar usar snapshots incompletos.
- Se probo un smoke con `-MaxMatches 20 -FrameStateChunkMatches 5` y el scorer
  pudo leer correctamente el directorio Parquet.

## Iteracion 15 - Logs de rendimiento por intervalo

- Se comprobo que el log `rate` del extractor era una media acumulada desde el
  inicio y no una tasa desde el ultimo print.
- Se ajusto `new_02a_extract_support_frame_state.py` para que el progreso cada
  1000 directorios muestre `rate_last_print`, calculado solo desde el print
  anterior.
- Se ajusto el log de chunks para separar `chunk_elapsed`, `write_elapsed` y
  `elapsed_total`; ahora el tiempo del chunk mide desde el guardado anterior y
  no desde el inicio del proceso.
- El manifiesto `_manifest.json` tambien guarda campos del ultimo chunk para
  poder diagnosticar si el coste esta en parsear JSON o en escribir Parquet.

## Iteracion 16 - Baseline full m12 de etiqueta

- Se genero el baseline completo `m12` a partir del snapshot full:
  `support_scores_m12.parquet`, `model_input_support_regression_m12.parquet` y
  graficas en `analysis/support_label_distribution/full_m12/`.
- La cobertura del scoring fue muy alta: `337094` keys exportadas sobre `337130`
  keys de draft, dejando `36` keys solo en draft.
- La etiqueta queda dentro de `[0, 1]`, sin nulos, con media `0.2828`, mediana
  `0.2636`, desviacion `0.1722`, `1.97%` exactos en `0` y `0.037%` exactos en
  `1`.
- La distribucion no parece degenerada: aproximadamente `14.94%` de filas quedan
  por debajo de `0.10` y `11.47%` por encima de `0.50`.
- El ranking por campeon es interpretable al filtrar por soporte suficiente:
  supports de roaming/enganche como `Pyke`, `Bard`, `Poppy`, `Pantheon`,
  `Alistar` y `Rell` aparecen por encima de enchanters mas estaticos como
  `Yuumi`, `Milio`, `Senna`, `Seraphine`, `Soraka` o `Lulu`.
- Se observa una pequena diferencia por lado (`blue` media `0.2881`, `red`
  media `0.2775`). Este hallazgo es coherente con un analisis temprano del
  proyecto donde el equipo azul tomaba las larvas primero en torno al `60%` de
  las partidas: el acceso y la prioridad alrededor de objetivos neutrales no son
  perfectamente simetricos en League of Legends, por lo que una ligera asimetria
  del roaming observado puede reflejar estructura real del mapa y no
  necesariamente un bug de etiqueta.
- Se genero tambien la comparacion full contra la referencia manual experta en
  `analysis/champion_reference/full_m12/`, con `min_count=100`. La correlacion
  por campeon fue alta en ranking (`Spearman=0.8751`, `Pearson=0.8951`,
  `n=30` campeones comparados), aunque la escala observada es mas comprimida que
  la escala experta.

## Iteracion 17 - Aclaracion de referencia experta

- La referencia actual de campeones no es un score oficial de Riot. Riot/Data
  Dragon solo puede aportar metadatos oficiales generales como tags de campeon,
  titulo e info basica.
- El campo `expert_support_roam_score` procede de
  `references/manual_support_champion_reference.csv`, una tabla manual curada
  con arquetipo, score esperado `[0,1]`, confianza y notas.
- La comparacion `observada vs experta` debe interpretarse como una validacion
  cualitativa de orden/ranking por campeon, no como ajuste exacto de escala. La
  observada resume comportamiento real en partidas entre minuto `5` y `12`,
  mientras que la experta describe identidad/prior general del campeon.

## Iteracion 18 - Linea futura secuencial

- Se anadio al informe la idea de explorar RNN/GRU/LSTM como linea futura,
  siguiendo la sugerencia del tutor.
- La motivacion es que la timeline de Riot puede entenderse como una secuencia
  de snapshots del estado de la partida. La MLP actual usa features tabulares de
  draft y una etiqueta agregada; una arquitectura secuencial podria explotar
  directamente la evolucion temporal.
- Se deja esta opcion como trabajo futuro, no como sustituto inmediato, porque
  primero se necesita una baseline MLP solida para comparar.

## Iteracion 19 - Correccion path full en cluster

- Se detecto que el job de cluster podia buscar por error
  `model_input_support_regression_full_m12.parquet`.
- En el flujo actual, `full` no se guarda con sufijo `_full`; el artefacto
  canonico es `model_input_support_regression_m12.parquet`.
- Se blindo `train_cluster_support_mlp.sh` para usar explicitamente ese path
  cuando `SAMPLE_TAG=full` y para avisar si recibe el nombre antiguo mediante
  `INPUT_PATH`.
