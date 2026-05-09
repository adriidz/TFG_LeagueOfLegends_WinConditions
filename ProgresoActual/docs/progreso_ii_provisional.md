# Progreso II provisional

Documento de trabajo para guiar la redaccion del Informe de Progreso II.

Periodo cubierto: desde la entrega del Informe de Progreso I hasta el estado de
trabajo del 07/05/2026.

Este documento no sustituye al Informe de Progreso I. Su funcion es ordenar lo
ocurrido despues de esa entrega, comparar el avance real con el planning previsto
y dejar claras las decisiones metodologicas tomadas antes de redactar el Informe
de Progreso II.

## 1. Punto de partida tras el Informe de Progreso I

El Informe de Progreso I cerraba una primera etapa del proyecto con las
siguientes decisiones ya asentadas:

- abandonar la clasificacion discreta inicial y trabajar con regresion continua;
- centrar temporalmente el TFG en `support-only`;
- separar features de draft de etiquetas derivadas de timeline;
- usar `support_roam_score` como etiqueta continua principal;
- tomar la MLP full `m12` como baseline inicial;
- validar la etiqueta con una referencia experta manual por campeon;
- planificar el siguiente bloque alrededor de tuning OAT, embeddings/feature
  enrichment y preparacion del Informe de Progreso II.

La conclusion tecnica del Informe I era que el modelo ya aprendia senal desde el
draft, pero todavia presentaba limitaciones claras: sobreajuste temprano,
predicciones comprimidas hacia valores medios y dependencia fuerte de la calidad
semantica de la etiqueta.

## 2. Planning previsto para el periodo 28/04-10/05

El planning del Informe I marcaba dos bloques principales:

| Fechas | Objetivo previsto | Estado real |
|---|---|---|
| 28/04-03/05 | Tuning OAT conjunto de MLP y etiqueta support | Preparado, documentado y versionado, pero no ejecutado en cluster |
| 04/05-10/05 | Embeddings y feature enrichment inicial | No abordado como embeddings; se sustituyo parcialmente por refinamiento de etiqueta/geometria |

La desviacion principal se debe a que el entorno de cluster/graficas no estaba
operativo para cerrar el tuning con ejecucion completa. En vez de bloquear el
avance esperando al cluster, se avanzo por una ruta alternativa de bajo coste
local: mejorar la geometria de la etiqueta, explorar una transformacion quantile
y montar un primer prototipo terminal.

Esta desviacion no cambia la direccion del TFG. Cambia el orden de ejecucion:
primero se ha reforzado la definicion del target y el entregable aplicado; el
tuning OAT queda preparado para ejecutarse cuando el cluster vuelva a estar
operativo.

## 3. Avance cronologico realizado

### 3.1 Preparacion del tuning OAT

Se preparo el experimento `support_oat_full_m12` para comparar de forma
controlada variantes de etiqueta e hiperparametros.

Artefacto principal:

```text
ProgresoActual/OAT/support_oat_full_m12/experiments/runs_manifest.csv
```

El manifest contiene 20 runs organizadas en tres fases:

| Fase | Runs | Objetivo |
|---|---:|---|
| `label_weights` | 5 | Aislar el efecto de cambiar los pesos de la etiqueta |
| `time_window` | 9 | Comparar ventanas temporales de observacion |
| `train_hparams` | 6 | Probar hiperparametros de la MLP manteniendo la etiqueta baseline |

La decision metodologica fue usar un diseno `one-at-a-time` en vez de una
busqueda factorial completa. Esto permite interpretar mejor cada cambio y reduce
el coste computacional. La limitacion es que no mide interacciones entre
parametros.

Documentacion asociada:

```text
ProgresoActual/docs/support_oat_tuning.md
ProgresoActual/docs/oat_manifest_explanation.md
```

Estado actual:

- los inputs del tuning estan preparados;
- el cluster puede ejecutar las runs leyendo el manifest;
- los modelos/resultados del tuning todavia no existen;
- la comparacion OAT queda pendiente hasta recuperar el flujo de cluster.

### 3.2 Primer prototipo terminal

Se adelanto un primer prototipo por terminal para probar inferencia sobre champ
select usando la baseline MLP full `m12`.

Script principal:

```text
ProgresoActual/scripts/predict_support_roam_cli.py
```

Documentacion:

```text
ProgresoActual/docs/terminal_prototype.md
```

El prototipo permite:

- introducir manualmente un draft aliado y enemigo;
- estimar el score de roaming del support aliado y enemigo;
- usar defaults razonables para hechizos si el usuario no los indica;
- ejecutar inferencia desde una partida real mediante `match_id` y `team_id`;
- mostrar etiqueta real y error absoluto cuando se usa una fila ya etiquetada;
- traducir el score a una lectura interpretativa mediante percentiles frente a
  validacion.

Importancia para el TFG:

- convierte el modelo en un primer entregable aplicado;
- permite explicar el resultado final esperado sin esperar a la interfaz final;
- demuestra que el pipeline de entrenamiento produce artefactos reutilizables en
  inferencia;
- adelanta parte del objetivo que inicialmente estaba previsto para junio.

Limitacion:

- usa todavia la baseline `m12`;
- no incorpora los resultados futuros del OAT;
- no usa embeddings ni feature enrichment;
- predice predisposicion desde draft, no comportamiento observado en partida.

### 3.3 Refinamiento de geometria de la etiqueta support

Como el tuning no podia cerrarse todavia, se avanzo en mejorar la parte mas
critica del problema: la definicion semantica de la etiqueta de roaming.

Se creo una nueva linea de trabajo en:

```text
ProgresoActual2/
```

Esta carpeta funciona como sandbox de Progreso II. `ProgresoActual` se mantiene
como fuente estable del pipeline anterior, mientras que los nuevos artefactos de
geometria y etiqueta se escriben en `ProgresoActual2`.

La geometria v5 manual parte de una anotacion visual del mapa y define zonas
semanticas mas controladas:

```text
ProgresoActual2/data/geometry/manual_geometry_v5_config.json
ProgresoActual2/src/geometry/geometry_v5_manual.py
ProgresoActual2/scripts/plot_geometry_v5_manual.py
```

Decisiones importantes:

- `MID_LANE` tiene prioridad en el cruce central entre mid, rio y junglas;
- `RIVER_BOT` deja de ser una diagonal amplia y pasa a ser una transicion corta
  hacia dragon;
- `RIVER_TOP` queda separado alrededor de herald/baron;
- `BOT_SIDE_NEAR`, `BLUE_BOT_JUNGLE` y `RED_BOT_JUNGLE` se ajustan para no romper
  la lectura de `RIVER_BOT`;
- los nucleos de carril siguen usando una representacion tipo `centerline +
  width`.

Se generaron diagnosticos de zonas y distribuciones frame-level para comprobar
que la nueva geometria se comporta razonablemente.

Documentacion asociada:

```text
ProgresoActual2/docs/geometry_v5_manual.md
ProgresoActual2/docs/geometry_v5_manual_annotation_workflow.md
ProgresoActual2/docs/progress_2026-05-06_geometry_v5_quantile.md
```

### 3.4 Analisis frame-level con geometria v5

Se creo el script:

```text
ProgresoActual2/scripts/build_geometry_v5_frame_state_distributions.py
```

Objetivo:

- clasificar posiciones de supports en las zonas v5;
- comparar el nuevo booleano `support_in_bot_context_v5` frente a la definicion
  legacy;
- revisar ventanas `m5_12` y `m0_14`;
- validar que la geometria nueva no produce comportamientos degenerados.

Resultados documentados:

| Ventana | Frames vivos | Match-team keys | `support_in_bot_context_v5_share` | Legacy share |
|---|---:|---:|---:|---:|
| `m5_12` | 2,245,763 | 337,128 | 0.741890 | 0.760169 |
| `m0_14` | 4,525,059 | 337,128 | 0.719144 | 0.732074 |

Interpretacion:

- la geometria v5 mantiene una lectura parecida a la anterior, pero algo mas
  restrictiva;
- no rompe la cobertura de datos;
- permite separar mejor bot, rio, mid, objetivos y junglas;
- es una base mas defendible para recalcular la etiqueta agregada.

### 3.5 Nueva etiqueta agregada con geometria v5

Se creo el script:

```text
ProgresoActual2/scripts/build_support_roam_score_v5_distribution.py
```

Salida principal:

```text
ProgresoActual2/data/clean/scores/support_scores_v5_geometry_m12.parquet
```

La receta conserva la idea de la etiqueta anterior:

```text
raw = 0.45 * outside_ratio_v5
    + 0.35 * far_ratio_v5
    + 0.20 * xp_gap_v5

support_roam_score_v5_geometry = raw ** 0.75
```

Cambios frente a versiones previas:

- `outside_ratio_v5` usa `support_in_bot_context_v5`;
- el contexto bot incluye `BOT_LANE_CORE`, `BOT_SIDE_NEAR`, `RIVER_BOT` y
  `DRAGON_AREA`;
- bases del support y del ADC se excluyen con la geometria manual v5.

Resumen full documentado:

| Metrica | Valor |
|---|---:|
| Filas | 337,104 |
| Coverage | 0.999929 |
| Mean | 0.392561 |
| Median | 0.389645 |
| Q05 | 0.091024 |
| Q95 | 0.711345 |
| Q99 | 0.839280 |
| Share score = 0 | 0.018522 |
| Share score = 1 | 0.000727 |
| Correlacion fila vs v3 | 0.940635 |
| Mean delta v5 - v3 | +0.020020 |
| Median delta v5 - v3 | 0.000000 |

Interpretacion:

- la nueva etiqueta es muy cercana a la anterior en ranking global;
- la geometria v5 desplaza ligeramente la media hacia arriba;
- se mantiene cobertura practicamente completa;
- el cambio parece refinamiento semantico, no ruptura del pipeline.

### 3.6 Transformacion quantile de la etiqueta

Se exploro una alternativa a la transformacion manual `raw ** 0.75` usando una
transformacion tipo quantile.

Motivacion:

La etiqueta `support_roam_score` no es una magnitud fisica observada de forma
directa, sino un score construido a partir de heuristicas: salir del contexto de
bot, separarse del ADC y acumular experiencia de forma distinta al ADC. Por
ello, la escala exacta de la etiqueta no tiene una interpretacion absoluta tan
fuerte como la tendria una variable medida directamente. Un valor `0.60` no
significa necesariamente "el doble" de roaming que `0.30`; significa que, bajo
la definicion elegida, esa partida-equipo muestra mas evidencia de roaming.

En las variantes previas, la distribucion de la etiqueta quedaba concentrada en
zonas medias. Esto puede empujar a la MLP a aprender predicciones demasiado
centradas, especialmente si la mayoria de ejemplos se agrupan alrededor de
valores parecidos. La transformacion quantile se plantea como una forma de
convertir el score en una escala relativa: en vez de predecir intensidad absoluta
de roaming, el modelo aprende la posicion aproximada del caso dentro de la
distribucion observada.

La consecuencia metodologica es importante: una etiqueta quantile debe
interpretarse como propension relativa o percentil de roaming temprano, no como
cantidad fisica absoluta. Esto puede ser adecuado para el TFG porque el objetivo
aplicado es ordenar y comparar drafts segun su tendencia esperada al roaming,
no medir una unidad natural de roaming.

Script:

```text
ProgresoActual2/scripts/build_support_roam_score_v5_quantile_labels.py
```

Salida:

```text
ProgresoActual2/data/clean/scores/support_scores_v5_quantile_m12.parquet
```

Columnas nuevas:

```text
support_roam_score_v5_quantile
support_roam_score_v5_quantile_zero_preserved
```

La variante recomendada para probar primero es
`support_roam_score_v5_quantile_zero_preserved`, porque:

- mantiene los casos `raw == 0` como cero;
- aplana la distribucion de scores positivos;
- conserva el orden relativo de la etiqueta original;
- reduce la arbitrariedad de elegir una potencia manual;
- separa conceptualmente los casos sin evidencia de roaming de los casos con
  roaming bajo pero positivo.

Resumen documentado:

| Columna | Mean | Median | Q05 | Q95 | Share score = 0 |
|---|---:|---:|---:|---:|---:|
| `raw_support_roam_score_v5_geometry` | 0.303160 | 0.284594 | 0.040946 | 0.635000 | 0.018522 |
| `support_roam_score_v5_geometry` | 0.392561 | 0.389645 | 0.091024 | 0.711345 | 0.018522 |
| `support_roam_score_v5_quantile` | 0.499826 | 0.499986 | 0.049997 | 0.949950 | 0.018522 |
| `support_roam_score_v5_quantile_zero_preserved` | 0.490736 | 0.490574 | 0.032082 | 0.948949 | 0.018525 |

Nota metodologica importante:

La transformacion quantile global es aceptable para exploracion inicial y para
crear un primer model input. Para una evaluacion estricta, el transformador debe
ajustarse solo con train y aplicarse despues a valid/test.

Por tanto, la transformacion no se considera una "trampa" si se documenta como
una redefinicion relativa del target y se evita fuga de informacion entre splits.
Si se ajustara usando todo el dataset antes de evaluar, entonces las metricas de
validacion/test quedarian contaminadas por informacion de la distribucion global.

### 3.7 Preparacion de la siguiente prueba de entrenamiento

Se dejo documentado como construir un model input usando la etiqueta quantile:

```powershell
.venv\Scripts\python.exe ProgresoActual\src\02_data_processing\build_support_model_input.py `
  --support-scores-path ProgresoActual2\data\clean\scores\support_scores_v5_quantile_m12.parquet `
  --support-score-source-col support_roam_score_v5_quantile_zero_preserved `
  --out-path ProgresoActual2\data\training\model_input_support_regression_v5_quantile_zero_m12.parquet `
  --summary-dir ProgresoActual2\data\training\model_input_support_regression_v5_quantile_zero_m12_analysis `
  --join-how inner
```

Y entrenar una MLP comparable:

```powershell
.venv\Scripts\python.exe ProgresoActual\scripts\train_support_mlp_regression.py `
  --input ProgresoActual2\data\training\model_input_support_regression_v5_quantile_zero_m12.parquet `
  --outdir ProgresoActual2\models\support_mlp_regression_v5_quantile_zero_m12 `
  --target-col support_roam_score `
  --feature-groups standard
```

Estado actual:

- la ruta esta preparada conceptualmente;
- falta ejecutar el model input y entrenamiento comparativo;
- falta comparar contra baseline `m12`;
- falta decidir si la etiqueta final sera magnitud directa, gamma transform o
  quantile zero-preserved.

### 3.8 Limpieza y reorganizacion del repositorio

Se realizo una limpieza estructural para que el repositorio refleje mejor el
estado actual del proyecto.

Se eliminaron del indice Git artefactos que no debian estar versionados:

- `.venv/`;
- `*.pyc`;
- `__pycache__/`;
- `data_new/`;
- caches y artefactos generados antiguos.

Tambien se retiraron del arbol activo scripts del pipeline antiguo que ya no
representaban la direccion actual del TFG:

```text
src/02_data_processing/
src/03_training/
src/eda/
```

Se mantuvo:

```text
src/01_data_collection/
```

Motivo:

- el recolector/raw sigue siendo util;
- el pipeline vivo esta ahora en `ProgresoActual`;
- `ProgresoActual2` contiene la linea experimental reciente;
- `PropuestaInicial` queda como archivo documental ligero.

Esta limpieza no es un resultado cientifico principal, pero si mejora la
reproducibilidad y evita que el repositorio mezcle codigo vivo con restos de
experimentos ya superados.

## 4. Que no se ha hecho todavia

### 4.1 OAT ejecutado

El OAT esta preparado, pero no ejecutado. Por tanto no se pueden presentar aun:

- ranking real de runs;
- mejor configuracion por `val_mse`;
- comparacion de curvas train/valid;
- scatter true-vs-pred del mejor OAT;
- conclusion empirica sobre si conviene cambiar pesos, ventana o hiperparametros.

En el Informe II debe decirse claramente que el avance fue preparatorio, no de
resultado final.

### 4.2 Embeddings

No se han implementado embeddings de campeon, runas o hechizos.

Lo que se ha hecho en su lugar es refinar la etiqueta y su geometria, que es una
forma distinta de feature/target enrichment. Aun asi, no debe presentarse como
"embeddings realizados".

Forma honesta de redactarlo:

> El bloque previsto de embeddings no se ejecuto todavia. Se pospuso porque la
> prioridad tecnica paso a ser mejorar la definicion de la etiqueta y preparar
> una comparacion experimental mas fiable.

### 4.3 Comparacion final de modelos

Todavia falta entrenar y comparar:

- baseline MLP full `m12`;
- mejor OAT, cuando exista;
- MLP con etiqueta v5 geometry;
- MLP con etiqueta v5 quantile zero-preserved;
- posibles variantes enriched/embeddings.

Sin esa comparacion, Progreso II debe presentarse como avance metodologico y de
preparacion experimental, no como cierre definitivo del modelo.

## 5. Decisiones metodologicas tomadas

### Decision 1: no bloquear el avance por el cluster

El cluster era necesario para ejecutar el OAT completo, pero no para seguir
mejorando la definicion de la etiqueta ni el prototipo. Por eso se avanzo en
tareas locales que reducen riesgo metodologico.

### Decision 2: mantener `support-only`

No se ha vuelto a abrir jungla/equipo. La prioridad sigue siendo cerrar una
etiqueta support defendible antes de escalar.

### Decision 3: separar trabajo estable y sandbox

`ProgresoActual` queda como linea estable del pipeline support-only.
`ProgresoActual2` queda como sandbox de Progreso II para geometria v5 y etiqueta
quantile.

### Decision 4: tratar OAT como experimento preparado, no concluido

El OAT debe aparecer en Progreso II como diseno experimental listo para ejecutar,
no como resultado computacional ya obtenido.

### Decision 5: adelantar prototipo terminal

Aunque el prototipo estaba previsto mas adelante, se adelanto porque era viable
con la baseline actual y ayuda a conectar el trabajo experimental con el
entregable final.

### Decision 6: priorizar calidad del target antes que embeddings

Antes de enriquecer entradas, se ha decidido mejorar la etiqueta. Si el target
esta mal calibrado o mal definido, embeddings pueden mejorar metricas sin
resolver el problema conceptual.

## 6. Comparacion contra el planning inicial

| Bloque previsto | Resultado real | Lectura para Informe II |
|---|---|---|
| Tuning OAT | Preparado y documentado; pendiente de ejecucion | Avance parcial. El diseno esta listo, pero faltan metricas |
| Embeddings/feature enrichment | Embeddings no realizados; target enrichment si | Desviacion justificada por prioridad metodologica |
| Refinar etiqueta support | Realizado con geometria v5 y quantile | Avance importante no previsto con ese detalle |
| Prototipo terminal | Primer prototipo implementado | Avance adelantado respecto al planning |
| Reorganizacion repo | Limpieza realizada | Mejora de reproducibilidad y mantenimiento |

Mensaje central:

> El planning no se cumplio literalmente en orden, pero el proyecto si avanzo.
> El tuning OAT quedo preparado para cluster, los embeddings se pospusieron, y el
> esfuerzo local se redirigio hacia dos piezas de alto valor: una etiqueta support
> mas defendible y un primer prototipo terminal.

## 7. Artefactos principales para citar en el Informe II

### OAT

```text
ProgresoActual/OAT/support_oat_full_m12/experiments/runs_manifest.csv
ProgresoActual/docs/support_oat_tuning.md
ProgresoActual/docs/oat_manifest_explanation.md
```

### Prototipo terminal

```text
ProgresoActual/scripts/predict_support_roam_cli.py
ProgresoActual/docs/terminal_prototype.md
```

### Geometria v5

```text
ProgresoActual2/data/geometry/manual_geometry_v5_config.json
ProgresoActual2/src/geometry/geometry_v5_manual.py
ProgresoActual2/scripts/plot_geometry_v5_manual.py
ProgresoActual2/scripts/build_geometry_v5_frame_state_distributions.py
ProgresoActual2/docs/geometry_v5_manual.md
ProgresoActual2/docs/progress_2026-05-06_geometry_v5_quantile.md
```

### Etiqueta v5 y quantile

```text
ProgresoActual2/scripts/build_support_roam_score_v5_distribution.py
ProgresoActual2/scripts/build_support_roam_score_v5_quantile_labels.py
ProgresoActual2/data/clean/scores/support_scores_v5_geometry_m12.parquet
ProgresoActual2/data/clean/scores/support_scores_v5_quantile_m12.parquet
ProgresoActual2/analysis/support_roam_score_v5_geometry/
ProgresoActual2/analysis/support_roam_score_v5_quantile/
```

### Limpieza del repositorio

```text
README.md
.gitignore
PropuestaInicial/
ProgresoActual/
ProgresoActual2/
src/01_data_collection/
```

## 8. Como redactarlo en el Informe de Progreso II

Estructura recomendada:

1. Recapitulacion breve del estado tras Progreso I.
2. Comparacion directa con el planning.
3. Bloque OAT preparado.
4. Bloque geometria v5 y nueva etiqueta.
5. Bloque prototipo terminal.
6. Decisiones tomadas y justificacion de desviaciones.
7. Riesgos pendientes.
8. Plan ajustado hasta la entrega final.

No conviene vender el periodo como "se completo el tuning". La formulacion
correcta es:

> Se preparo el tuning OAT completo y reproducible, pero su ejecucion quedo
> pendiente por disponibilidad del cluster. Para evitar detener el avance, se
> redirigio el trabajo local al refinamiento de la etiqueta support y a un primer
> prototipo de inferencia por terminal.

## 9. Siguientes pasos antes de cerrar Progreso II

Prioridad alta:

1. Ejecutar una MLP con `support_roam_score_v5_quantile_zero_preserved`.
2. Compararla contra baseline `m12`.
3. Ejecutar o desbloquear OAT en cluster.
4. Preparar una tabla de planning: previsto, hecho, pendiente, motivo.

Prioridad media:

1. Generar 2-3 figuras claras para Informe II:
   - comparacion de distribuciones v3/v5/quantile;
   - esquema OAT;
   - ejemplo de salida del prototipo terminal.
2. Decidir si `ProgresoActual2` se integra en `ProgresoActual` o queda como
   sandbox documentado.

Prioridad baja:

1. Empezar embeddings solo cuando haya baseline comparable con la nueva etiqueta.
2. Mejorar el prototipo terminal tras elegir la etiqueta/modelo candidato.

## 10. Resumen ejecutivo provisional

Tras el Informe de Progreso I, el proyecto no avanzo exactamente en el orden
previsto, pero si avanzo en piezas relevantes. El tuning OAT fue preparado con un
manifest reproducible de 20 runs, aunque no se ejecuto todavia por la situacion
del cluster. El bloque de embeddings quedo pospuesto. En su lugar se refino la
geometria de la etiqueta de support mediante una version manual v5, se genero
una nueva etiqueta agregada, se exploro una transformacion quantile
zero-preserved y se implemento un primer prototipo terminal de inferencia sobre
champ select. Ademas, se limpio el repositorio para separar codigo vivo,
artefactos experimentales y archivo historico. La siguiente fase debe cerrar la
comparacion empirica entre baseline, OAT y etiqueta v5/quantile, y convertir
estas decisiones en una narrativa clara para el Informe de Progreso II.
