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

## Iteracion 20 - Primera MLP full m12

- Se analizaron los resultados del entrenamiento `support_mlp_full_m12` traido
  del cluster.
- La MLP usa `337094` filas, split por partida con `269674` filas de train y
  `67420` de validacion, `OneHotEncoder` de dimension `1796`, capas `256 -> 128`
  y `MSELoss`.
- Resultado en validacion: `MSE=0.02543`, `RMSE=0.15947`, `MAE=0.12721`,
  `R2=0.13068`, `Pearson=0.3633`, `Spearman=0.3568`.
- Frente a predecir la media de validacion (`MSE=0.02925`, `MAE=0.13791`), la
  MLP mejora aproximadamente `13.1%` en MSE y `7.8%` en MAE. Por tanto aprende
  senal real, pero todavia limitada.
- La mejor epoca fue la `6`. A partir de ahi el train loss sigue bajando y el
  validation loss sube, senal clara de sobreajuste temprano.
- Las predicciones quedan comprimidas: target de validacion `std=0.1710`, pero
  predicciones `std=0.0681`, con maximo predicho `0.7191`. El modelo captura
  priors de campeon/composicion, pero tiende a regresar hacia la media y falla
  especialmente en scores altos.
- Por bins, el error absoluto medio es bajo en scores medios (`0.2-0.4`) y crece
  mucho en extremos, especialmente por encima de `0.6`. Esto sugiere que la MLP
  tabular predice bien el perfil promedio, pero no la intensidad concreta de
  roaming de una partida.

## Iteracion 21 - Informe de progreso completo

- Se creo `ProgresoActual/docs/informe_progreso_completo.md` como version amplia
  del informe de progreso para la entrega del `27/04/2026`.
- El informe ratifica el objetivo inicial del TFG, pero justifica el cambio de
  clasificacion a regresion continua porque la regresion penaliza los errores
  segun distancia al target.
- Se incorporaron resultados completos del snapshot full, salud de etiqueta
  `m12`, comparacion observada vs experta y primera MLP full.
- Se enlazaron las figuras generadas: distribucion de etiqueta, CDF,
  distribucion por lado, boxplot por campeon, comparacion experta, curvas de
  entrenamiento, scatter true-vs-pred, residuos y error por bins.
- El planning final incluye entregables esperados, criterios de exito,
  dependencias y estado, manteniendo como objetivo final el prototipo terminal
  interpretable.

## Iteracion 22 - Pulido visual y metodologico del informe

- Se ampliaron las figuras del informe con distribucion experta, comparacion
  experta vs observada y un esquema propio de mapa para contextualizar la
  asimetria blue/red en objetivos neutrales.
- Se ajusto el boxplot por campeon para filtrar picks raros usando
  `n >= 500`.
- Se explicaron con mas detalle la CDF, los residuos y el error por bins, y se
  aclaro que no se busca una distribucion artificialmente plana del score.
- Se actualizo el planning para compactar el tuning OAT de MLP y etiqueta en la
  misma semana, adelantando embeddings/feature enrichment antes del Informe de
  Progreso II.
- Se anadieron salidas auxiliares en la comparacion experta:
  `expert_reference_head3.csv`, `expert_support_score_histogram.png` y
  `expert_vs_observed_distribution.png`.

## Iteracion 23 - Contexto minimo de League of Legends

- Se anadio al inicio de `informe_progreso_completo.md` una seccion breve de
  contexto del juego antes del resumen ejecutivo.
- La seccion explica solo los conceptos necesarios para entender el modelo:
  objetivo de la partida, mapa, lineas, roles, draft, jungla, botlane, support,
  roaming, objetivos neutrales y timeline.
- La explicacion se vincula explicitamente con el pipeline de deep learning:
  draft como input pregame y timeline como fuente para construir etiquetas, no
  como entrada del modelo actual.

## Iteracion 24 - Imagen de mapa y origen de referencia experta

- Se inserto `images/minimapa.png` en la seccion inicial de contexto del juego
  para explicar visualmente mapa, bases, lineas, rio y jungla antes de entrar en
  resultados.
- Se retiro del cuerpo de asimetria por side la figura esquematica generada de
  objetivos neutrales, dejando alli solo la interpretacion textual.
- Se aclaro en el informe que la referencia experta de campeones no procede de
  internet ni de Riot: es una curacion manual inicial basada en conocimiento del
  dominio, usada como contraste cualitativo y no como ground truth oficial.

## Iteracion 25 - Ampliacion de referencia experta de supports

- Se amplio `ProgresoActual/references/manual_support_champion_reference.csv`
  de 30 a 47 campeones con presencia habitual o plausible en support.
- Se anadieron arquetipos, score esperado de roaming, confianza y notas para
  picks adicionales como Amumu, Maokai, Poppy, Shaco, Zilean, Mel o Zoe.
- Se normalizo `Renata Glasc` a `Renata` para que coincida con los nombres
  observados en los datos.
- Se regenero `champion_support_reference.csv` en modo manual y la comparacion
  contra `support_scores_m12.parquet`.
- La nueva comparacion mantiene una correlacion alta pese a mayor cobertura:
  `Pearson=0.7947`, `Spearman=0.8251`, `n=47`. La caida frente a la tabla de 30
  campeones es esperable al incluir picks menos canonicos y con menor confianza.
- Se actualizo `informe_progreso_completo.md` para que el texto, la tabla
  `head(3)` y las metricas reflejen la referencia ampliada.

## Iteracion 26 - Aclaracion de frame-state y draft features

- Se amplio la seccion de snapshot del informe para explicar que el
  `frame-state` es una tabla observacional derivada de las timelines, usada para
  calcular etiquetas y no como entrada directa del modelo actual.
- Se anadio una aclaracion previa a la tabla de `draft_features`, explicando que
  estas variables representan la informacion pregame que si se usa como base de
  entrada del modelo.
- La redaccion refuerza la separacion metodologica entre datos usados para
  construir el target y datos disponibles antes de la partida para entrenar y
  evaluar predicciones.

## Iteracion 27 - Figuras legibles en formato de dos columnas

- Se ajustaron los scripts de visualizacion para generar figuras con tipografia
  mas grande, leyendas mas legibles, ticks de mayor tamano y mayor resolucion.
- Los cambios afectan a:
  `plot_support_label_distribution.py`,
  `compare_support_champion_reference.py` y
  `plot_training_run_diagnostics.py`.
- Se regeneraron las figuras principales del informe: distribuciones de
  etiqueta, CDF, distribucion por lado, boxplot por campeon, comparacion experta
  y diagnosticos de entrenamiento de la MLP full `m12`.
- El objetivo es que los ejes, numeros y leyendas sigan siendo legibles si las
  imagenes se insertan en una memoria o informe con maquetacion a dos columnas.

## Iteracion 28 - Aumento adicional de tipografia en figuras

- Se incremento de nuevo el tamano de fuentes en los plots para soportar mejor
  una insercion reducida en formato de dos columnas.
- Se subieron titulos, etiquetas de ejes, ticks, leyendas, grosor de lineas y
  tamano de puntos en scatter plots.
- Se regeneraron las mismas figuras principales de distribucion, comparacion
  experta y diagnosticos de entrenamiento manteniendo `220 dpi`.

## Iteracion 29 - Confianza experta en scatter observado vs experto

- Se modifico `compare_support_champion_reference.py` para que el scatter
  `generated_vs_expert_scatter.png` use color como tercera variable.
- El gradiente rojo-verde representa `expert_confidence`: rojo para etiquetas
  menos seguras y verde para etiquetas mas seguras.
- Se anadio una diagonal discontinua `y=x` para leer rapidamente la desviacion
  entre media observada y score experto.
- Se regenero la comparativa full `m12` y se amplio la explicacion del informe
  para interpretar confianza, diagonal y desviaciones.

## Iteracion 30 - Rango util en scatter observado vs experto

- Se ajusto el eje Y del scatter observado vs experto para mantener el rango
  util de la media observada, que no alcanza valores cercanos a `1`.
- La diagonal se dibuja solo dentro del rango visible, evitando que el grafico
  aplaste la nube de puntos por reservar espacio vertical no informativo.
- Se regenero `generated_vs_expert_scatter.png` con el color de confianza
  experto y el eje Y adaptado al maximo observado.

## Iteracion 31 - Scatter experto vs observado mas limpio

- Se ajusto visualmente la diagonal `y=x` para que sea una referencia gris,
  discontinua y menos dominante que los puntos.
- Se anadio una etiqueta pequena `y=x` dentro del rango visible para dejar claro
  que la diagonal representa coincidencia perfecta entre experto y observado.
- Se cambio la anotacion de campeones: ahora solo se etiquetan puntos extremos
  por score experto, media observada o desviacion absoluta, evitando saturar el
  scatter con texto.
- Se regenero `generated_vs_expert_scatter.png` manteniendo el gradiente de
  confianza experta.

## Iteracion 32 - Referencia escalada del scatter experto vs observado

- Se corrigio la linea discontinua del scatter para que apunte al limite util
  del eje observado (`0.45`) y no al antiguo rango completo `[0,1]`.
- La linea deja de etiquetarse como `y=x`, porque ahora es una referencia visual
  escalada desde `(0,0)` hasta `(1,0.45)`.
- Se actualizo la explicacion del informe para aclarar que el experto usa escala
  `[0,1]`, mientras que las medias observadas por campeon quedan comprimidas por
  debajo de `0.45`.

## Iteracion 33 - Limpieza de etiqueta en scatter escalado

- Se elimino la etiqueta textual `scaled ref.` del scatter observado vs experto.
- La linea discontinua queda como referencia visual sin texto superpuesto, para
  reducir ruido en la figura final del informe.

## Iteracion 34 - Reordenacion de proximos pasos y planning

- Se reordeno `informe_progreso_completo.md` para que `Proximos pasos tecnicos`
  sea el apartado 7.
- El `Planning hasta final de proyecto` pasa a ser el apartado 8, manteniendo el
  mismo contenido y funcionando como concrecion temporal de los pasos tecnicos.
- La bibliografia se mantiene como apartado 9.

## Iteracion 35 - Mitigaciones de la distancia draft-comportamiento

- Se amplio la seccion 6.1 del informe para no limitarse a describir la
  dificultad entre intencion de draft y comportamiento observado.
- Se corrigio la redaccion para centrar las mitigaciones en la reduccion de la
  distancia entre intencion y observacion: uso de partidas de jugadores de alto
  nivel, metricas espaciales y ventana temprana acotada.
- Se explico que evitar diferencias directas de oro, experiencia o recursos
  entre equipos ayuda a que la etiqueta no mida simplemente ventaja de partida.
- La redaccion refuerza que el proyecto no ignora el ruido entre predisposicion
  estrategica y ejecucion real, sino que lo trata como una limitacion central
  del diseno experimental.

## Iteracion 36 - Reajuste del planning final

- Se elimino la semana dedicada exclusivamente a consolidar la baseline MLP,
  porque los resultados ya estan analizados y documentados en el informe.
- El tuning OAT conjunto de MLP y etiqueta support pasa a empezar el
  `28/04-03/05`.
- Embeddings y feature enrichment se adelantan una semana, dejando tambien una
  semana adicional para refinar representacion y cerrar la etiqueta support
  candidata antes del Informe de Progreso II.
- La parte final del planning se reestructura para dar mas tiempo a nuevas
  etiquetas: jungla ocupa `25/05-31/05` y equipo ocupa `01/06-07/06`.
- La semana `08/06-14/06` queda para integrar etiquetas, decidir con el tutor la
  via RNN/GRU/LSTM y consolidar el modelo candidato.

## Iteracion 37 - Figura conceptual del pipeline

- Se creo `ProgresoActual/scripts/plot_report_pipeline_overview.py` para generar
  una figura conceptual reproducible del pipeline actual.
- La figura separa explicitamente la rama de entrada pre-partida
  (`draft -> features -> MLP -> score predicho`) de la rama observada
  (`timeline -> frame-state -> etiqueta real`).
- Se genero
  `ProgresoActual/analysis/report_figures/fig2_pipeline_draft_timeline.png`,
  pensada para insertarse como figura 2 del informe y reforzar que la timeline
  no se usa como input del modelo, sino para construir el valor objetivo.

## Iteracion 38 - Rediseño de la figura conceptual del pipeline

- Se rediseño `fig2_pipeline_draft_timeline.png` con una composicion en dos
  carriles: datos antes de la partida y datos observados despues.
- La nueva version muestra con mas claridad que el flujo superior alimenta el
  modelo y el flujo inferior genera la etiqueta real usada para evaluar.
- Se mantuvo la generacion por codigo en `plot_report_pipeline_overview.py`, sin
  usar ningun motor de imagenes generativo.

## Iteracion 39 - Nueva version diferenciada de figura 2

- Se genero una version alternativa con nombre nuevo:
  `fig2_pipeline_draft_timeline_v2.png`, para evitar problemas de cache visual.
- La composicion se cambio a un flujo en forma de convergencia: la rama
  predictiva produce el score estimado y la rama observada produce la etiqueta
  real; ambas se juntan solo en la comparacion de entrenamiento.
- La figura sigue generandose con Matplotlib desde
  `plot_report_pipeline_overview.py`.

## Iteracion 40 - Primer prototipo terminal de champ select

- Se creo `ProgresoActual/scripts/predict_support_roam_cli.py` como primer
  prototipo por terminal del entregable final.
- El CLI carga la baseline `support_mlp_full_m12` ya entrenada mediante
  `model_config.json`, `preprocess.joblib` y `best_model.pt`, reutilizando el
  mismo `OneHotEncoder` que se uso en entrenamiento.
- El usuario puede introducir side, campeones aliados/enemigos y hechizos de
  invocador por rol de forma interactiva o por argumentos.
- La salida muestra score estimado, percentil frente a las predicciones de
  validacion, lectura textual y contraste con la referencia experta para el
  support aliado y para el support enemigo.
- La prediccion enemiga se obtiene invirtiendo la perspectiva `ally/enemy` y el
  side, de forma equivalente a ejecutar el mismo modelo para el otro equipo.
- Se documento el uso en `ProgresoActual/docs/terminal_prototype.md` y se
  anadio un acceso rapido desde `ProgresoActual/README.md`.
