# Informe de Progreso II - base de redaccion

Este documento adapta los hallazgos empiricos de la fase final a la narrativa
del Informe de Progreso II. Su objetivo no es sustituir el informe definitivo,
sino servir como base estructurada para redactarlo. La organizacion sigue los
minimos pedidos para el ultimo informe de progreso:

- seguimiento de la planificacion prevista y ajustes realizados;
- explicacion de la metodologia seguida finalmente;
- exposicion y valoracion de resultados;
- conclusiones provisionales;
- fuentes de informacion consultadas.

La idea central que debe sostener todo el informe es que el objetivo del TFG no
ha cambiado, sino que se ha refinado. El proyecto sigue buscando inferir patrones
de early-game en League of Legends a partir de informacion pregame/draft. Lo que
ha cambiado es el alcance inmediato: en esta fase se ha priorizado una evaluacion
rigurosa del roaming del support antes de ampliar el sistema a nuevas etiquetas o
a una integracion multi-output.

## 1. Introduccion

El Informe de Progreso I cerro una primera etapa del TFG en la que se reformulo
el problema desde una clasificacion discreta inicial hacia una tarea de regresion
continua. En esa etapa ya se habia construido un pipeline completo de
recoleccion, procesamiento, construccion de etiquetas y entrenamiento, y se habia
definido `support_roam_score` como primera etiqueta continua de comportamiento
early-game.

La primera MLP entrenada sobre informacion de draft mostro que existia senal
predictiva, pero tambien dejo varias limitaciones claras: sobreajuste temprano,
predicciones comprimidas hacia valores medios y dificultad para interpretar el
rendimiento sin baselines adicionales. Por tanto, el trabajo posterior al
Informe I se ha orientado a convertir una primera prueba funcional en una
evaluacion experimental mas rigurosa.

El TFG se encuentra ya en su fase final de desarrollo. En este contexto, el
Informe de Progreso II debe explicar no solo que se ha implementado, sino tambien
que conclusiones se pueden extraer del trabajo realizado y que ajustes se han
introducido respecto a la planificacion anterior.

## 2. Punto de partida tras el Informe de Progreso I

Al cerrar el Informe I, el proyecto tenia asentadas las siguientes decisiones:

- usar informacion pregame como entrada del modelo;
- separar estrictamente features de draft y etiquetas derivadas de timeline;
- abandonar la clasificacion discreta como enfoque principal;
- trabajar con regresion continua;
- centrar temporalmente el desarrollo en roaming del support;
- usar `support_roam_score` como etiqueta principal;
- entrenar una primera MLP OneHot como baseline inicial;
- validar cualitativamente la etiqueta mediante una referencia experta por
  campeon;
- planificar el siguiente bloque alrededor de tuning OAT, embeddings/feature
  enrichment y posterior extension a nuevas etiquetas.

La conclusion tecnica del Informe I era prudente: el draft parecia contener
senal, pero el modelo inicial no bastaba para cerrar la evaluacion. Faltaba
comparar contra baselines simples, analizar modelos tabulares alternativos y
estimar que parte de la varianza del comportamiento podia explicarse realmente
desde variables disponibles antes de la partida.

## 3. Seguimiento de la planificacion y obstaculos encontrados

### 3.1 Planificacion prevista

El planning del Informe I preveia el siguiente recorrido:

| Periodo | Objetivo previsto | Criterio de exito |
| --- | --- | --- |
| 28/04-03/05 | Tuning OAT conjunto: MLP + etiqueta support | Tabla comparativa por `val_mse`, ranking de heuristicas |
| 04/05-10/05 | Embeddings y feature enrichment inicial | Comparacion OneHot vs enriched/embeddings |
| 11/05-17/05 | Refinar representacion y cerrar soporte | Feature set candidato + decision sobre representacion |
| 18/05-24/05 | Informe de Progreso II | Documento listo con tuning + embeddings |
| 25/05-31/05 | Redefinir etiqueta de jungla | Label continua de jungla + plots de salud |
| 01/06-07/06 | Redefinir etiqueta de equipo | Label continua de equipo + plots de salud |
| 08/06-14/06 | Integracion multi-output y decision RNN/GRU/LSTM | Modelo candidato, decision secuencial |
| 15/06-21/06 | Prototipo terminal e interpretacion | CLI usable con lectura interpretable |
| 22/06-28/06 | Cierre final | Memoria + presentacion |

### 3.2 Grado de seguimiento

La planificacion no se ha seguido literalmente en el orden previsto. El motivo
principal es doble: por un lado, la revision critica posterior al Informe I
mostro que era necesario introducir baselines y techo empirico antes de seguir
escalando el sistema; por otro, la indisponibilidad del cluster durante mas de
una semana bloqueo la ejecucion completa del tuning OAT de la MLP.

| Bloque previsto | Estado actual | Grado de seguimiento | Lectura para el Informe II |
| --- | --- | --- | --- |
| Tuning OAT de MLP + etiqueta | Preparado y documentado; pendiente de ejecucion completa/refinamiento final | Parcial | El experimento sigue previsto, pero su ejecucion depende de que el cluster vuelva a estar operativo |
| Embeddings / feature enrichment | Pospuesto | Parcial | Se retrasa porque antes era necesario contextualizar la MLP con baselines, GBT y techo empirico |
| Refinar representacion y cerrar support | Redefinido como evaluacion support-only rigurosa | Alto | Incluye dataset final, baselines, GBT, quantile, techo empirico e interpretacion |
| Informe de Progreso II | En preparacion para el 24/05/2026 | En curso | El contenido pasa de "tuning + embeddings" a "evaluacion empirica del alcance real del draft" |
| Redefinir etiqueta de jungla | Replanificado como trabajo futuro | No ejecutado | Se evita abrir una etiqueta grande sin tiempo suficiente para validarla bien |
| Redefinir etiqueta de equipo | Replanificado como trabajo futuro | No ejecutado | Misma justificacion que jungla |
| Integracion multi-output | Replanificada como trabajo futuro | No ejecutado | Depende de disponer de varias etiquetas maduras |
| Prototipo terminal | Adelantado | Alto | Se implementa antes de lo previsto para mantener la dimension aplicada del TFG |
| Cierre final | Se mantiene | En curso | Memoria, conclusiones, prototipo actualizado y presentacion |

### 3.3 Obstaculo tecnico: indisponibilidad del cluster

Un obstaculo importante de esta etapa ha sido que el cluster no ha funcionado
correctamente durante mas de una semana. Esto ha frenado especialmente el tuning
OAT de la MLP, ya que el experimento estaba pensado para ejecutar varias runs de
entrenamiento de forma reproducible y ordenada.

La decision tomada fue no detener el avance del proyecto mientras el cluster
seguia bloqueado. En lugar de esperar a poder cerrar el OAT, se priorizaron
tareas ejecutables en local y de alto valor metodologico:

- revision critica del alcance del proyecto;
- consolidacion del dataset final;
- creacion de un split persistido train/validation/test por `match_id`;
- ejecucion de baselines simples;
- entrenamiento de modelos tabulares HistGBT;
- calculo de techo empirico/ICC;
- desarrollo adelantado del prototipo por terminal.

El OAT no desaparece del plan. Antes de la entrega del Informe de Progreso II
del 24/05/2026 esta previsto dejarlo refinado y listo para entrenar cuando el
cluster vuelva a estar operativo. Su papel pasa a ser el de comprobacion
experimental de la MLP y de variantes de etiqueta/hiperparametros, no el unico
eje de la narrativa del informe.

## 4. Ajustes metodologicos respecto a la propuesta inicial

Tras el Informe I se realizo una revision critica del estado del proyecto,
documentada en:

```text
final/docs/analysis_results.md
final/docs/decisions.md
final/docs/technical_spec.md
```

Esa revision permitio identificar tres riesgos principales:

1. El alcance original era demasiado amplio para cerrar con rigor varias
   etiquetas, modelos y prototipo en el tiempo restante.
2. La MLP inicial no podia interpretarse sin compararla contra baselines simples
   y modelos alternativos.
3. Era necesario reservar un test final persistido para evitar evaluar el
   proyecto sobre la misma particion usada durante el desarrollo.

El ajuste principal ha sido concentrar el trabajo inmediato en `support_roam_score`.
Esto no debe presentarse como abandono del objetivo inicial, sino como
refinamiento metodologico. El TFG sigue investigando la inferencia de
comportamiento early-game desde draft; el caso de roaming del support funciona
como caso de estudio consolidado para medir con rigor hasta donde llega esa
inferencia con informacion exclusivamente pregame.

Las etiquetas de jungla/equipo, la integracion multi-output y las arquitecturas
secuenciales quedan como extensiones futuras. La prioridad actual es obtener una
conclusion defendible sobre una tarea bien definida antes de escalar el sistema.

## 5. Metodologia seguida finalmente

### 5.1 Unidad de analisis

La unidad de analisis es `(match_id, team_id)`. Cada partida genera dos
observaciones: una desde la perspectiva del equipo azul y otra desde la
perspectiva del equipo rojo.

Esta formulacion permite usar el draft de cada equipo como entrada y asociarlo a
un comportamiento observado posteriormente en la timeline. Tambien permite
mantener separadas las perspectivas aliada y enemiga para el prototipo.

### 5.2 Separacion entre input pregame y target postgame

El input del modelo procede de informacion disponible antes de la partida:
campeones, roles, lado, bans, hechizos, runas y composiciones.

La etiqueta se calcula a partir de la timeline, es decir, a partir de lo que
ocurrio durante la partida. La timeline no se usa como entrada del modelo. Esta
separacion es esencial para evitar leakage: el sistema debe inferir una
predisposicion o tendencia desde el draft, no reconstruir una partida que ya ha
observado.

### 5.3 Etiqueta principal

La etiqueta principal es `support_roam_score`, una variable continua en `[0, 1]`
que resume el comportamiento observado del support durante el early-game. Valores
bajos indican un support mas anclado a botlane y al botlaner; valores altos
indican mayor evidencia de desplazamientos fuera del contexto de bot.

La etiqueta mide comportamiento observado, no intencion optima. Esta distincion
es importante para interpretar los resultados: el draft puede sugerir una
predisposicion estrategica, pero la ejecucion real depende de muchos factores
dinamicos no disponibles antes de la partida.

### 5.4 Dataset final y split

El dataset final se separa por `match_id`, evitando que las dos perspectivas de
una misma partida caigan en particiones distintas. Todos los modelos usan el
mismo split persistido:

| split | filas |
| --- | ---: |
| train | 268423 |
| validation | 57335 |
| test | 57489 |

La particion de validation se usa para desarrollo y seleccion de configuraciones.
La particion de test queda reservada para la comparacion final.

### 5.5 Variante quantile

Ademas de la escala raw, se crea `support_roam_score_quantile` mediante un
`QuantileTransformer` ajustado exclusivamente sobre train. Despues se aplica a
validation y test.

Esta variante permite comprobar si una escala mas uniforme facilita el
aprendizaje. No sustituye al target raw como referencia principal, porque la
escala raw es mas interpretable. Cuando un modelo se entrena sobre quantile, sus
predicciones se inverse-transforman a escala raw para calcular metricas
comparables e interpretables.

### 5.6 Modelos y baselines

La comparacion actual incluye:

| Modelo | Papel en el analisis |
| --- | --- |
| Global Mean | Baseline trivial: predice siempre la media global |
| Champion Mean | Baseline por campeon de support |
| HistGBT | Modelo tabular principal sobre variables pregame |
| HistGBT + Archetypes | Variante con arquetipos de campeones |
| HistGBT + Pair TE | Variante con target encodings de pares/interacciones |
| Variantes quantile | Mismos modelos entrenados sobre target quantile e inverse-transformados a raw |
| MLP/OAT | Linea prevista para comprobar hiperparametros y variantes de etiqueta cuando el cluster lo permita |

## 6. Trabajo realizado en esta etapa

### 6.1 Revision critica y documentacion tecnica

La primera parte de esta etapa consistio en revisar el estado del TFG despues del
Informe I. De esa revision salieron tres documentos clave:

- `analysis_results.md`, con diagnostico, small wins y plan revisado;
- `decisions.md`, con decisiones de fase final;
- `technical_spec.md`, con rutas, scripts, columnas y criterios de evaluacion.

Este bloque fue importante porque permitio convertir una intuicion general
("hay que mejorar el modelo") en una lista concreta de comprobaciones: baselines,
GBT, split final, target quantile, techo empirico e interpretacion del prototipo.

### 6.2 Preparacion y refinamiento del OAT

El tuning OAT de la MLP sigue formando parte del plan. Su objetivo es responder
de forma controlada a dos preguntas:

- si la mejora viene de cambiar la definicion de la etiqueta;
- si la mejora viene de cambiar hiperparametros de entrenamiento de la MLP.

El manifest principal es:

```text
ProgresoActual/OAT/support_oat_full_m12/experiments/runs_manifest.csv
```

El experimento contiene 20 runs:

| Fase | Runs | Objetivo |
| --- | ---: | --- |
| `label_weights` | 5 | Aislar el efecto de cambiar pesos de la etiqueta |
| `time_window` | 9 | Comparar ventanas temporales de observacion |
| `train_hparams` | 6 | Probar hiperparametros de la MLP manteniendo la etiqueta baseline |

El diseno es `one-at-a-time`: cada run cambia una dimension y mantiene las demas
constantes. Esto reduce el coste computacional y facilita la interpretacion. La
limitacion es que no mide interacciones entre parametros.

Por la indisponibilidad del cluster, el OAT no se ha podido ejecutar todavia de
forma completa. Antes del Informe II debe presentarse como experimento preparado
y en refinamiento, no como resultado cerrado.

### 6.3 Consolidacion del dataset final

Se creo la fase `final/`, con scripts reproducibles para preparar el dataset,
entrenar modelos, calcular baselines y generar analisis:

```text
final/scripts/01_prepare_final_dataset.py
final/scripts/02_baseline_champion_mean.py
final/scripts/03_train_gbt.py
final/scripts/03b_train_gbt_enriched.py
final/scripts/03c_train_gbt_interactions.py
final/scripts/05_empirical_ceiling.py
final/scripts/06_feature_importance.py
final/scripts/07_model_comparison.py
```

Esta fase marca un cambio importante respecto al Informe I: los resultados ya no
dependen de una unica particion de validacion ni de una unica arquitectura. El
objetivo pasa a ser comparar enfoques bajo un contrato comun de datos.

### 6.4 Baselines

Se implementaron dos baselines fundamentales:

- `Global Mean`, que predice siempre la media global del target;
- `Champion Mean`, que predice la media historica de `support_roam_score` para
  el campeon support aliado.

La baseline por campeon es especialmente importante porque mide cuanto se puede
explicar solo con la identidad del support. Sin esta comparacion, una MLP con
R2 moderado podria parecer mas informativa de lo que realmente es.

### 6.5 Modelos tabulares alternativos

Se entrenaron modelos `HistGradientBoostingRegressor` sobre las mismas variables
pregame. Tambien se probaron variantes con arquetipos e interacciones por pares.

La motivacion fue comprobar si un modelo tabular fuerte capturaba senal adicional
respecto a la media por campeon y si el feature engineering interpretable
aportaba mejoras materiales.

### 6.6 Techo empirico

Se calculo un analisis tipo ICC / media por grupo para estimar cuanta varianza
puede explicarse mediante agrupaciones directas del draft. Este bloque es clave
para interpretar un R2 bajo: permite distinguir entre falta de capacidad del
modelo y limitacion de la informacion disponible antes de la partida.

### 6.7 Prototipo por terminal

El prototipo por terminal se adelanto respecto al planning inicial. Su script
principal es:

```text
ProgresoActual/scripts/predict_support_roam_cli.py
```

El prototipo permite introducir composiciones de champ select y obtener una
prediccion interpretable de la tendencia de roaming de los supports. Tambien
puede ejecutarse sobre una partida real ya etiquetada para mostrar la etiqueta
observada y el error absoluto.

Este avance mantiene la dimension aplicada del TFG durante el bloqueo del
cluster. El prototipo no debe presentarse como un oraculo de comportamiento
futuro, sino como una herramienta de apoyo pregame basada en tendencias
historicas.

## 7. Resultados obtenidos y valoracion

### 7.1 Resultados estrictos en test

La tabla siguiente resume los resultados principales sobre `support_roam_score`
en escala raw:

| Modelo | R2 | Spearman | RMSE | MAE |
| --- | ---: | ---: | ---: | ---: |
| HistGBT + Pair TE | 0.1614 | 0.3882 | 0.1745 | 0.1408 |
| HistGBT | 0.1613 | 0.3881 | 0.1745 | 0.1409 |
| HistGBT + Archetypes | 0.1611 | 0.3880 | 0.1745 | 0.1409 |
| HistGBT + Pair TE (quantile -> raw) | 0.1522 | 0.3883 | 0.1755 | 0.1420 |
| HistGBT (quantile -> raw) | 0.1518 | 0.3879 | 0.1755 | 0.1421 |
| Champion Mean | 0.1249 | 0.3360 | 0.1783 | 0.1440 |
| Global Mean | 0.0000 | n/a | 0.1906 | 0.1552 |

La lectura principal es que el draft contiene senal real, pero limitada. El
HistGBT mejora la baseline por campeon, aunque no por un margen muy grande. Esto
indica que una parte importante de la senal viene ya dada por la identidad del
campeon de support. La composicion y el matchup aportan informacion adicional,
pero no transforman radicalmente el problema.

Un R2 alrededor de 0.16 no significa que el modelo "acierte un 16% de las
veces". Significa que explica aproximadamente un 16% de la varianza del score
observado. La mayor parte de la variacion partida a partida depende de factores
no disponibles en el draft: estado de las oleadas, recalls, pathing real del
jungla, prioridad de mid, vision, eventos tempranos, coordinacion y decisiones
individuales.

### 7.2 Techo empirico observado

Los resultados mas relevantes del analisis ICC / media por grupo son:

| Agrupacion | ICC | R2 media por grupo |
| --- | ---: | ---: |
| support champion | 0.1214 | 0.1212 |
| support champion + side | 0.1211 | 0.1217 |
| botlane champions | 0.1393 | 0.1606 |
| botlane champions + side | 0.1390 | 0.1726 |
| support vs enemy support | 0.1315 | 0.1535 |
| support archetype | 0.0836 | 0.0811 |
| botlane archetypes | 0.0932 | 0.0952 |

Este es uno de los hallazgos centrales del trabajo. Al agrupar solo por campeon
de support se explica alrededor del 12% de la varianza. Al incorporar la botlane
aliada y el lado, el techo practico fiable sube hasta aproximadamente 16-17%.
El HistGBT queda muy cerca de ese rango, lo que sugiere que el modelo tabular
captura gran parte de la senal pregame disponible en las variables actuales.

Los arquetipos no mejoran el techo. Esto es coherente: un arquetipo es una
representacion mas general que el `champion_id`, por lo que pierde informacion
especifica. Puede ser util para explicar resultados, pero no contiene mas
informacion que la identidad concreta del campeon.

Algunas agrupaciones de altisima cardinalidad pueden mostrar R2 muy altos, pero
no deben usarse como techo fiable si generan grupos pequenos o casi unicos. En
el informe conviene centrar la discusion en agrupaciones interpretables y con
suficiente soporte: support, botlane, side y matchup de supports.

### 7.3 Metricas practicas tolerantes

La regresion estricta penaliza errores numericos exactos, pero para un uso
pregame tipo coach no siempre importa distinguir entre scores muy proximos. Por
eso se anadio una evaluacion complementaria en escala raw.

Estas metricas responden a otra pregunta:

> Aunque el modelo no prediga el score exacto, situa la composicion en una zona
> estrategica razonable?

Resultados principales:

| Modelo | within 0.10 | within 0.15 | within 0.20 | fixed bin acc | fixed adjacent acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| HistGBT + Pair TE | 0.4182 | 0.5980 | 0.7418 | 0.4836 | 0.9710 |
| HistGBT | 0.4184 | 0.5975 | 0.7408 | 0.4831 | 0.9708 |
| HistGBT + Archetypes | 0.4181 | 0.5968 | 0.7404 | 0.4825 | 0.9711 |
| Champion Mean | 0.4108 | 0.5841 | 0.7289 | 0.4794 | 0.9678 |
| Global Mean | 0.3781 | 0.5435 | 0.6883 | 0.4651 | 0.9667 |

La metrica `within 0.20` es util para la narrativa del prototipo: el HistGBT
queda a menos de 0.20 puntos del score real en aproximadamente el 74% de las
partidas de test. Esto no convierte al modelo en un predictor exacto, pero si
apoya su uso como herramienta interpretativa de tendencias.

La accuracy por bins fijos usa cortes `[0.00, 0.25, 0.50, 0.75, 1.00]`. La
accuracy exacta de bin ronda el 48%, mientras que la accuracy adyacente supera
el 97%. Esta metrica debe interpretarse con cuidado: es alta en parte porque la
mayoria de errores caen en el bin vecino, pero tambien porque el target esta
concentrado en zonas medias. Es una metrica de apoyo, no el resultado principal.

### 7.4 Metricas ordinales

Tambien se incorporaron metricas ordinales para evaluar si el modelo distingue
zonas de roaming bajo, medio y alto sin exigir precision decimal:

| Metrica | Que mide |
| --- | --- |
| Spearman continuo | Correlacion de ranking entre scores exactos reales y predichos |
| Spearman sobre bins | Ranking entre clases ordinales, ignorando diferencias internas del bin |
| Kendall tau sobre bins | Concordancia por pares entre zonas reales y predichas |
| Quadratic Weighted Kappa | Acuerdo ordinal, penalizando poco errores de un bin y mas errores lejanos |

Resultados principales:

| Modelo | Spearman continuo | fixed bin Spearman | fixed bin Kendall | fixed bin QWK | quantile bin Spearman | quantile bin QWK |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HistGBT | 0.3881 | 0.2369 | 0.2218 | 0.1623 | 0.3282 | 0.2788 |
| HistGBT + Pair TE | 0.3882 | 0.2370 | 0.2219 | 0.1608 | 0.3285 | 0.2779 |
| HistGBT + Archetypes | 0.3880 | 0.2338 | 0.2190 | 0.1578 | 0.3279 | 0.2775 |
| Champion Mean | 0.3360 | 0.2098 | 0.1965 | 0.1413 | 0.2878 | 0.2339 |

La lectura correcta es:

1. El modelo aprende una ordenacion parcial de las composiciones.
2. Esa ordenacion es mas clara con bins por cuantiles de train, porque las
   clases quedan mas balanceadas.
3. El acuerdo ordinal sigue siendo moderado, no alto.
4. Champion Mean queda cerca, reforzando que el campeon de support explica una
   parte importante de la utilidad practica.

El resultado defendible no es que el modelo clasifique perfectamente zonas de
roaming, sino que conserva senal ordinal y tolerante suficiente para alimentar
un prototipo interpretativo.

### 7.5 Valoracion global

El hallazgo principal es que el draft contiene senal, pero no determina la
ejecucion. La identidad del campeon de support, la botlane y ciertos matchups
permiten anticipar una predisposicion estrategica al roam. Sin embargo, el score
real observado depende tambien de variables dinamicas no observables pregame.

Por tanto, el resultado no debe presentarse como un fracaso del modelo, sino
como una conclusion empirica del TFG:

> Con informacion exclusivamente pregame, es posible estimar tendencias de
> roaming de support, pero no recuperar con alta precision la ejecucion exacta
> observada en partida.

## 8. Prototipo terminal

El prototipo por terminal es un entregable importante porque conecta el analisis
offline con la dimension aplicada del TFG. Su valor no depende de predecir el
score exacto con precision decimal. En un uso tipo coach o analista, el objetivo
es convertir el draft en una lectura estrategica previa:

- si la composicion sugiere mas o menos roaming;
- si el support esta por encima o por debajo de una tendencia esperada;
- como comparar la predisposicion del support aliado y enemigo;
- como traducir el output numerico a una interpretacion comprensible.

Una formulacion recomendable para el informe:

> El sistema no pretende sustituir el analisis humano ni predecir decisiones
> exactas del jugador. Su funcion es ofrecer una estimacion contextual de la
> predisposicion al roaming a partir del draft, util como apoyo interpretativo en
> fase pregame.

El prototipo debera actualizarse con el mejor modelo final cuando se cierre la
comparacion completa, y podra incorporar los resultados del OAT si el cluster
vuelve a estar disponible a tiempo.

## 9. Limitaciones y riesgos pendientes

Las principales limitaciones actuales son:

- El target mide comportamiento observado, no intencion optima ni plan
  estrategico declarado.
- El draft no contiene informacion dinamica como wave state, recalls, pathing
  del jungla, prioridad de mid, vision, eventos tempranos o coordinacion.
- El cluster ha retrasado la ejecucion completa del OAT de la MLP.
- No se ha demostrado el limite absoluto del problema; los experimentos acotan
  el alcance del enfoque pregame con las variables actuales.
- Arquetipos e interacciones simples no mejoran materialmente el GBT base, lo
  que sugiere que el cuello de botella no esta solo en la capacidad del modelo.
- La transformacion quantile es util como comprobacion experimental, pero no
  mejora el rendimiento final al volver a escala raw.

Extensiones posibles:

- ejecutar el OAT completo cuando el cluster vuelva a funcionar;
- probar MLPs con embeddings como comprobacion defensiva;
- hacer un analisis Challenger-only si se recupera `sourceTier`;
- explorar etiquetas mas cercanas a intencion o propension estrategica;
- ampliar a jungla/equipo cuando haya tiempo para validar nuevas etiquetas con
  el mismo rigor.

Estas extensiones no deben desplazar la conclusion principal del Informe II: el
caso support roaming ya permite evaluar de forma solida el alcance del enfoque
draft-only.

## 10. Plan revisado hasta la entrega final

El plan revisado hasta la entrega final prioriza cerrar el caso support-only:

| Periodo | Trabajo previsto | Resultado esperado |
| --- | --- | --- |
| Hasta 24/05/2026 | Refinar OAT, completar Informe de Progreso II, actualizar resultados disponibles | Informe II entregable y OAT preparado para cluster |
| Finales de mayo | Ejecutar OAT si el cluster vuelve a estar operativo | Comparacion MLP/OAT contra baselines y GBT |
| Inicio de junio | Actualizar prototipo con el modelo candidato final | CLI alineado con la evaluacion final |
| Junio | Redaccion de memoria, conclusiones y presentacion | Documento final y defensa preparados |

Jungla/equipo, multi-output, Challenger-only y nuevas etiquetas quedan como
trabajo futuro o comprobaciones secundarias. La prioridad es cerrar una
evaluacion rigurosa del caso support roaming.

## 11. Conclusiones provisionales

Conclusiones que conviene defender en el Informe de Progreso II:

1. El objetivo central sigue siendo inferir patrones de early-game desde draft.
2. El alcance inmediato se ha refinado hacia support roaming para obtener una
   evaluacion mas solida.
3. El draft contiene senal predictiva real: Global Mean queda claramente por
   debajo de Champion Mean y de HistGBT.
4. La senal esta concentrada en la identidad del campeon de support y, en menor
   medida, en botlane, side y matchup.
5. El techo empirico practico de agrupaciones pregame defendibles ronda el
   16-17% de varianza explicada.
6. El HistGBT se situa muy cerca de ese techo, lo que sugiere que el limite no
   esta solo en la arquitectura.
7. Arquetipos, interacciones simples y quantile no cambian materialmente la
   conclusion.
8. El prototipo terminal es valioso si se presenta como herramienta de apoyo
   pregame basada en tendencias, no como predictor exacto de decisiones
   individuales.
9. El bloqueo del cluster ha retrasado la ejecucion del OAT, pero no ha detenido
   el avance metodologico ni el desarrollo aplicado del proyecto.

Texto de cierre sugerido:

> El proyecto mantiene su objetivo de inferir patrones early-game desde
> informacion pregame, pero esta fase ha permitido evaluar con mas rigor su
> alcance real. El caso de roaming del support muestra que el draft proporciona
> una senal medible y util para generar una lectura estrategica interpretable,
> aunque limitada para predecir con precision la ejecucion observada en cada
> partida.

## 12. Fuentes de informacion consultadas

Para el Informe II conviene separar fuentes externas y documentacion interna.

Fuentes externas:

- Riot Games Developer Documentation y Data Dragon, para datos de partida,
  campeones y recursos estaticos.
- Documentacion de scikit-learn:
  - `GroupShuffleSplit`;
  - `QuantileTransformer`;
  - `HistGradientBoostingRegressor`;
  - metricas de regresion y correlacion.
- Documentacion de PyTorch, para MLP, funcion de perdida MSE y entrenamiento.
- Articulos academicos ya citados en el Informe I sobre prediccion en League of
  Legends, recomendacion de draft y analisis de comportamiento.

Documentacion interna:

```text
final/docs/analysis_results.md
final/docs/decisions.md
final/docs/technical_spec.md
final/analysis/model_comparison/comparison_tables.md
final/analysis/ceiling/ceiling_summary.md
ProgresoActual/docs/support_oat_tuning.md
ProgresoActual/docs/oat_manifest_explanation.md
ProgresoActual/docs/terminal_prototype.md
```

