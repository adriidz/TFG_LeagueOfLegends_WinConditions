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
