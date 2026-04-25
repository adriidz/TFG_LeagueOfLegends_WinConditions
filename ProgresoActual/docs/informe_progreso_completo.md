# Informe de progreso completo del TFG

**Titulo del TFG:** Inferencia de tendencias tempranas en el videojuego League of Legends mediante aprendizaje automatico  
**Autor:** Adrian Diaz Garcia  
**Fecha de preparacion:** 25/04/2026  
**Entrega prevista del informe de progreso:** 27/04/2026  

## 1. Resumen ejecutivo

El objetivo inicial del TFG era construir un sistema de aprendizaje automatico
capaz de inferir tendencias tempranas observables en partidas de League of
Legends a partir de informacion disponible antes del inicio de la partida,
principalmente el draft. La propuesta original planteaba tres tareas:
comportamiento del jungla, perfil de roaming del support y tendencia espacial
del equipo. La salida final esperada sigue siendo un prototipo ejecutable por
terminal donde el usuario pueda introducir manualmente campeones, runas,
hechizos de invocador y contexto del draft, y obtener una lectura interpretable
de los comportamientos esperados antes de que empiece la partida.

Este objetivo se alinea con la literatura previa, aunque desplaza el foco del
resultado final de partida hacia tendencias tempranas e interpretables. Parte de
la investigacion sobre League of Legends se ha centrado en predecir victoria
[1], recomendar campeones durante el draft [2] o analizar la senal estadistica
de picks y bans [3]. En cambio, este trabajo intenta aprovechar esa senal del
draft para aproximar comportamientos espaciales tempranos, conectando con
trabajos que estudian patrones de comportamiento y evolucion espacio-temporal en
MOBAs [5]-[7].

Durante el desarrollo se construyo un primer pipeline completo de recoleccion,
procesamiento, etiquetado y entrenamiento. Sin embargo, los primeros
experimentos mostraron que la dificultad principal no estaba solo en la
arquitectura del modelo, sino en la definicion de las etiquetas. En particular,
la formulacion como clasificacion generaba una perdida importante de
informacion: en clasificacion, todos los errores penalizan igual aunque uno este
muy cerca del umbral y otro sea semanticamente opuesto. Por este motivo, el
enfoque actual pasa a formular el problema como regresion continua. En
regresion, el error se mide segun la distancia entre la prediccion y el valor
real, lo que encaja mejor con etiquetas que originalmente ya son scores
continuos.

El reinicio metodologico se ha centrado primero en una tarea concreta:
predecir un score continuo de roaming del support en `[0,1]`. Esta reduccion de
alcance no elimina las tareas de jungla y equipo del objetivo final, sino que
establece una primera fase controlada para validar metodologia, datos, etiqueta
y modelo. Una vez estabilizada la tarea de support, se plantea reintroducir
nuevas versiones continuas de las etiquetas de jungla y equipo.

El estado actual del proyecto incluye un dataset completo de `171386` partidas
detectadas, `337130` observaciones de draft a nivel `(match_id, team_id)`, una
etiqueta continua de support para `337094` observaciones y una primera MLP
entrenada sobre el dataset completo. La MLP mejora en torno a un `13.1%` el MSE
respecto a predecir siempre la media de validacion, lo que indica que existe
senal aprendible desde el draft, aunque tambien se observa una compresion clara
de las predicciones hacia valores medios.

## 2. Objetivos iniciales y alineacion actual

La propuesta inicial definia como objetivo principal inferir tendencias
tempranas observables a partir del draft. Ese objetivo se mantiene. Lo que ha
cambiado es la forma de llegar a el. Inicialmente se propuso una clasificacion
multi-output con tres salidas. Actualmente se trabaja con una regresion
support-only para validar con mayor rigor una primera etiqueta continua antes de
volver a escalar el sistema.

### 2.1 Objetivos que se mantienen

- Usar informacion pregame del draft como entrada predictiva.
- Trabajar con observaciones a nivel `(match_id, team_id)`.
- Usar datos oficiales de Riot Games como fuente de partidas, timelines y
  metadatos.
- Inferir comportamientos tempranos, no resultado final de partida.
- Producir una salida final interpretable para jugadores.
- Mantener la posibilidad de extender el sistema a varias tareas.

### 2.2 Objetivos que se reformulan

La clasificacion multi-output queda aplazada. La razon principal es que las
etiquetas no nacen como clases naturales, sino como scores continuos derivados
de comportamiento observado en el timeline. Discretizar esos scores obliga a
introducir umbrales y una clase intermedia `ambiguous`, lo que dificulta la
evaluacion y hace que errores muy distintos se midan de la misma manera.

La regresion continua permite conservar la estructura gradual de la etiqueta.
Si una partida tiene un score real de `0.62`, una prediccion de `0.58` debe ser
mejor que una prediccion de `0.15`. En una clasificacion binaria o ternaria, esa
diferencia puede quedar ocultada por la asignacion a clases. Esta propiedad
justifica el cambio metodologico y permite evaluar el modelo con metricas como
MSE, RMSE, MAE, R2 y correlaciones.

### 2.3 Objetivo final del prototipo

El objetivo aplicado sigue siendo construir un prototipo por terminal. La idea
final es que el usuario introduzca manualmente un draft, incluyendo campeones,
runas, summoner spells y contexto relevante, y que el sistema devuelva scores
estimados junto con frases interpretables. Por ejemplo, en la tarea de support,
el sistema no deberia limitarse a devolver `0.37`, sino traducirlo a una lectura
del estilo: "perfil de roaming moderado; el support puede moverse, pero el draft
no sugiere una tendencia extrema a abandonar bot".

Este objetivo es importante porque evita que el proyecto se convierta solo en
una comparacion de metricas offline. Las metricas sirven para validar el modelo,
pero el entregable final debe ser comprensible para un jugador antes de empezar
la partida.

## 3. Trabajo previo realizado

### 3.1 Pipeline inicial

En la primera etapa se desarrollo un pipeline completo que incluia:

- recoleccion de partidas y timelines desde la API de Riot;
- transformacion del raw JSON en features de draft y etiquetas;
- calculo de scores tempranos de jungla, support y equipo;
- discretizacion de esos scores en clases;
- entrenamiento de modelos multi-output;
- analisis exploratorio, graficas y comparaciones entre ventanas.

Ese trabajo no se descarta. Sirve como base empirica para justificar el cambio
de enfoque. De hecho, el valor principal de la propuesta inicial ha sido
mostrar que el problema contiene senal, pero que la definicion del target
condiciona mas los resultados que pequenos cambios de arquitectura.

### 3.2 Fase 1: ventanas temporales y estabilidad

La Fase 1 estudio si la ventana temporal usada para construir las etiquetas
afectaba a la estabilidad y al rendimiento. Se probaron ventanas de early game
como 6, 8, 10, 12 y 14 minutos. La conclusion principal fue que las tareas no
se estabilizan al mismo ritmo:

- la etiqueta de jungla tendia a estabilizarse antes;
- la etiqueta de support necesitaba una ventana algo mas amplia, alrededor de
  12 minutos;
- la tendencia de equipo era la mas inestable.

Este resultado fue importante porque mostro que no era adecuado tratar las tres
tareas como equivalentes. Tambien mostro que parte de la inestabilidad no
procedia de cambios directos entre extremos, sino de transiciones hacia o desde
la zona central `ambiguous`.

### 3.3 Fase 2: problema de la clase `ambiguous`

La Fase 2 analizo el impacto de la clase intermedia. Los experimentos mostraron
que eliminar o reducir la clase `ambiguous` mejoraba las metricas de
clasificacion, especialmente cuando se conservaban ejemplos extremos. Esto
confirmo que la clase central representaba una zona de ruido o indefinicion para
el modelo.

No obstante, esta conclusion tambien revelo una limitacion mas profunda: si la
mejor forma de clasificar era eliminar gran parte de la zona central, entonces
la formulacion discreta estaba desaprovechando informacion. El score continuo
contenia una gradacion que la clasificacion obligaba a cortar artificialmente.
Por ello, el siguiente paso natural fue dejar de predecir clases y volver a
predecir directamente el score.

## 4. Metodologia actual

### 4.1 Datos y unidad de analisis

El proyecto trabaja con partidas de League of Legends obtenidas a partir de la
API de Riot Games. Riot proporciona endpoints para datos de partida y Data
Dragon para datos estaticos y assets como campeones, hechizos, runas e iconos
[9]. La unidad de analisis sigue siendo `(match_id, team_id)`, de manera que
cada partida genera dos observaciones: una desde la perspectiva del equipo azul
y otra desde la perspectiva del equipo rojo.

La entrada del modelo se limita a informacion disponible antes de la partida:
campeones aliados y enemigos, summoner spells, side y contexto del draft. Las
variables derivadas del timeline se usan solo para construir la etiqueta, no
como entrada del modelo. Esta separacion evita data leakage: el modelo no ve
datos del futuro, sino que aprende a aproximar el comportamiento temprano
observado a partir del draft.

### 4.2 Nueva etiqueta continua de support

La etiqueta actual mide el perfil de roaming del support entre minuto `5` y
minuto `12`. El score se define en `[0,1]`, donde valores bajos indican un
support mas anclado a bot/ADC y valores altos indican mayor tendencia observada
a abandonar la linea o jugar lejos del ADC.

La heuristica combina tres componentes:

- proporcion de frames fuera de la zona extendida de bot;
- proporcion de frames lejos del ADC;
- componente asociado a diferencia relativa de experiencia con el ADC.

La configuracion baseline `m12` usa los pesos:

| Componente | Peso |
|---|---:|
| Fuera de bot extendido | 0.45 |
| Lejos del ADC | 0.35 |
| Ratio/diferencia de experiencia | 0.20 |

Esta etiqueta no debe interpretarse como personalidad fija del campeon. Mide el
comportamiento observado de ese support en esa partida concreta. Por eso es
normal que la distribucion agregada este desplazada hacia valores bajos: incluso
campeones con identidad de roaming no abandonan bot de forma extrema en todas
las partidas.

### 4.3 Comparacion experta por campeon

Para comprobar si la etiqueta observada tiene sentido cualitativo, se construyo
una referencia experta manual para 30 campeones habitualmente considerados
supports. Esta tabla esta en
`ProgresoActual/references/manual_support_champion_reference.csv` e incluye:

- nombre del campeon;
- arquetipo experto;
- score esperado de roaming en `[0,1]`;
- confianza de la etiqueta;
- notas justificativas.

Data Dragon no contiene un score oficial de roaming. Solo aporta metadatos
oficiales generales. Por tanto, la comparacion observada vs experta no debe
presentarse como validacion contra ground truth oficial, sino como validacion
cualitativa de ranking: comprobar si los campeones que deberian aparecer como
mas roamers, como Bard, Pyke, Nautilus o Rakan, quedan por encima de campeones
mas anclados al ADC, como Yuumi, Soraka, Sona o Milio.

### 4.4 Modelo MLP

La primera baseline usa una MLP con codificacion One-Hot. Esta decision es
razonable como punto de partida porque la mayor parte de variables de entrada
son categoricas: campeones, summoner spells y side. La documentacion de
scikit-learn describe `OneHotEncoder` como un transformador que convierte
variables categoricas en columnas binarias, una representacion comun para
alimentar modelos de aprendizaje automatico [10].

El modelo usa `MSELoss` como funcion de perdida. PyTorch define MSELoss como el
error cuadratico medio entre prediccion y target [11]. Esta funcion es adecuada
para una primera regresion porque penaliza mas los errores grandes que los
pequenos, alineandose con la motivacion principal del cambio desde
clasificacion.

El split de entrenamiento y validacion se hace por `match_id`, no por filas
aleatorias independientes. Esta decision evita que una misma partida pueda
aparecer parcialmente en train y parcialmente en validacion. scikit-learn
proporciona `GroupShuffleSplit` precisamente para separar conjuntos usando una
variable de grupo externa [12].

## 5. Resultados obtenidos

### 5.1 Snapshot completo del dataset

El snapshot completo actual contiene:

| Metrica | Valor |
|---|---:|
| Partidas detectadas | 171386 |
| Partidas validas en frame-state | 168564 |
| Filas de frame-state | 9823624 |
| Partidas descartadas por duracion corta | 2805 |
| `bad_match` | 7 |
| `bad_tl` | 1 |
| `bad_roles` | 9 |

La tasa de partidas validas es muy alta. La mayor parte de descartes procede de
partidas demasiado cortas, lo que es esperable en un pipeline que intenta medir
comportamiento temprano hasta una ventana concreta.

Las `draft_features` completas contienen:

| Metrica | Valor |
|---|---:|
| Partidas procesadas | 171386 |
| Partidas validas | 168565 |
| Filas de draft | 337130 |
| Observaciones unicas `(match_id, team_id)` | 337130 |

Esto confirma que se esta generando una observacion por equipo. La diferencia
de una partida entre draft y frame-state se debe a una timeline invalida
(`bad_tl=1`), que no afecta de forma relevante al analisis completo.

### 5.2 Salud de la etiqueta `m12`

La etiqueta continua de support baseline `m12` se genero para `337094`
observaciones, con una cobertura de `337094/337130`. No hay nulos y todos los
valores se mantienen dentro de `[0,1]`.

| Metrica | Valor |
|---|---:|
| Filas etiquetadas | 337094 |
| Media | 0.2828 |
| Desviacion tipica | 0.1722 |
| Minimo | 0.0000 |
| Q05 | 0.0385 |
| Q25 | 0.1500 |
| Mediana | 0.2636 |
| Q75 | 0.3917 |
| Q95 | 0.5929 |
| Q99 | 0.7490 |
| Maximo | 1.0000 |
| Porcentaje exacto en 0 | 1.97% |
| Porcentaje exacto en 1 | 0.037% |

La distribucion esta concentrada hacia la izquierda, pero no colapsada. Esto no
es necesariamente negativo: el score mide intensidad de roaming real en una
partida concreta, y la mayoria de supports no juegan roaming extremo en todos
los frames entre minuto 5 y 12.

![Histograma de la etiqueta](../analysis/support_label_distribution/full_m12/support_label_histogram.png)

La CDF permite observar que existe masa suficiente en zonas intermedias, lo que
justifica usar regresion continua en lugar de convertir el problema de nuevo a
clases. La CDF, o funcion de distribucion acumulada, muestra para cada valor `x`
que porcentaje de observaciones tienen un score menor o igual que `x`. Por
ejemplo, permite responder preguntas del tipo: "que proporcion de supports tiene
un score de roaming inferior a 0.30?". Es una grafica util porque hace visible
si la etiqueta esta concentrada en una zona estrecha o si, por el contrario,
existe una gradacion progresiva. En este caso, aunque la distribucion esta
desplazada hacia valores bajos, la curva no indica un colapso total de la
etiqueta.

![CDF de la etiqueta](../analysis/support_label_distribution/full_m12/support_label_cdf.png)

Tambien aparece una ligera asimetria por lado:

| Side | Media | Mediana | Desviacion |
|---|---:|---:|---:|
| Blue | 0.2881 | 0.2700 | 0.1731 |
| Red | 0.2775 | 0.2577 | 0.1711 |

Esta diferencia es coherente con observaciones tempranas del proyecto: el
equipo azul tomaba las larvas primero aproximadamente en el 60% de partidas
analizadas. Dado que los accesos a dragones, larvas, heraldo y baron no son
perfectamente simetricos, parte de la asimetria puede reflejar estructura real
del mapa y no un error de etiqueta.

![Distribucion por lado](../analysis/support_label_distribution/full_m12/support_label_by_side_histogram.png)

El siguiente esquema resume la intuicion espacial que ayuda a interpretar esta
asimetria. Los supports empiezan normalmente ligados a botlane junto al ADC. El
dragon esta en la zona inferior del rio, mientras que las larvas y la zona de
baron/heraldo quedan en la parte superior. Si el equipo azul toma las primeras
larvas en torno al `59.38%` de las partidas, es plausible que sus supports
realicen ligeramente mas movimientos hacia zonas alejadas de botlane durante el
early game. En el analisis antiguo, el primer dragon aparecia mas a menudo para
red side (`RED 60.77%`), mientras que las primeras larvas aparecian mas a menudo
para blue side (`BLUE 59.38%`). Esta lectura no demuestra causalidad, pero ofrece un
contexto de mapa razonable para no interpretar automaticamente la diferencia
blue/red como ruido.

![Contexto de mapa y objetivos neutrales](../analysis/map_context/support_objective_context.png)

La distribucion por campeon tambien es interpretable. Los campeones con mayor
media observada incluyen perfiles de roaming o engage, mientras que los
enchanters mas vinculados al ADC aparecen con medias inferiores. Para evitar que
picks muy raros distorsionen la lectura, el boxplot mostrado filtra campeones
con `n >= 500` observaciones.

![Boxplot por campeones frecuentes](../analysis/support_label_distribution/full_m12/support_label_top_champion_boxplot.png)

### 5.3 Comparacion observada vs experta

La comparacion contra la tabla experta manual se ejecuto con `min_count=100`.
Antes de comparar, se construyo una tabla de referencia con 30 campeones
habitualmente usados como support. Un ejemplo de las tres primeras filas es:

| Campeon | Arquetipo experto | Score experto | Confianza | Nota |
|---|---|---:|---:|---|
| Alistar | engage_roamer | 0.82 | 0.90 | Strong engage and roam identity after lane setup. |
| Bard | roaming_specialist | 0.95 | 0.95 | Design explicitly rewards leaving lane and impacting map. |
| Blitzcrank | pick_roamer | 0.78 | 0.85 | Hook pressure translates well to river and mid roams. |

Cada campo tiene una funcion concreta. `champion_name` identifica el campeon;
`expert_archetype` resume su identidad funcional; `expert_support_roam_score`
es el prior subjetivo de roaming en `[0,1]`; `expert_confidence` indica la
seguridad con la que se asigno ese prior; y `notes` documenta la razon de la
etiqueta. Esta tabla no es oficial de Riot: es una curacion manual que sirve
como referencia cualitativa.

Tambien conviene mirar la distribucion de esos scores expertos. No se espera
que sea perfectamente plana, porque el roster de supports tampoco esta
uniformemente repartido entre campeones hiper-roamers y campeones totalmente
anclados a linea. Esta figura ayuda a comparar el prior teorico de campeon con
la distribucion observada en partidas reales.

![Distribucion del score experto](../analysis/champion_reference/full_m12/expert_support_score_histogram.png)

El resultado de la comparacion fue:

| Metrica | Valor |
|---|---:|
| Campeones en score table con `min_count>=100` | 75 |
| Campeones con referencia experta | 30 |
| Pearson | 0.8951 |
| Spearman | 0.8751 |

La correlacion alta indica que el ranking agregado por campeon se parece mucho
al ranking experto. Esta es una evidencia importante de que la etiqueta no es
arbitraria. Sin embargo, la escala observada es mas comprimida que la experta:
por ejemplo, Bard y Pyke tienen scores expertos cercanos a `1`, pero su media
observada ronda `0.38`. Esto no invalida la etiqueta; simplemente muestra la
diferencia entre identidad teorica de campeon y comportamiento medio observado
en partidas reales.

![Comparacion observada vs experta](../analysis/champion_reference/full_m12/generated_vs_expert_scatter.png)

La comparacion de distribuciones refuerza esa interpretacion: la referencia
experta esta mas extendida porque codifica identidad teorica, mientras que las
medias observadas por campeon se comprimen al promediar muchas partidas reales.
Esta compresion es esperable y no implica que la etiqueta observada deba
estirarse artificialmente hasta ocupar todo el rango `[0,1]`.

![Distribucion experta vs observada](../analysis/champion_reference/full_m12/expert_vs_observed_distribution.png)

Las mayores desviaciones se concentran en campeones de alto roaming teorico,
porque el score experto representa un prior de identidad de campeon y la media
observada representa comportamiento medio real. Esta diferencia debe explicarse
en la memoria como una cuestion de escala y no como fallo directo de la
heuristica.

![Mayores desviaciones](../analysis/champion_reference/full_m12/largest_deviations.png)

### 5.4 Primera MLP full `m12`

La primera MLP se entreno sobre el dataset completo con:

| Elemento | Valor |
|---|---:|
| Filas train | 269674 |
| Filas validacion | 67420 |
| Dimension OneHot | 1796 |
| Capas ocultas | 256 -> 128 |
| Dropout | 0.2 |
| Loss | MSELoss |
| Mejor epoca | 6 |

Metricas de validacion:

| Metrica | Valor |
|---|---:|
| MSE | 0.02543 |
| RMSE | 0.15947 |
| MAE | 0.12721 |
| R2 | 0.13068 |
| Pearson | 0.3633 |
| Spearman | 0.3568 |

Frente a predecir siempre la media de validacion (`MSE=0.02925`,
`MAE=0.13791`), la MLP mejora aproximadamente `13.1%` en MSE y `7.8%` en MAE.
Esto confirma que el draft contiene senal predictiva, aunque limitada.

La curva de entrenamiento muestra sobreajuste temprano. El loss de train sigue
bajando despues de la epoca 6, pero el loss de validacion empieza a subir. Esto
indica que el modelo aprende patrones especificos del conjunto de entrenamiento
que no generalizan igual de bien.

![Curva de entrenamiento](../models/support_mlp_full_m12/diagnostics/loss_curve.png)

El scatter true-vs-pred confirma una tendencia clara a la regresion a la media.
Las predicciones se concentran en una banda central: sobreestiman scores bajos y
subestiman scores altos.

![True vs predicted](../models/support_mlp_full_m12/diagnostics/true_vs_pred_scatter.png)

La distribucion de residuos muestra el valor `prediccion - valor_real` para cada
observacion de validacion. Esta grafica es importante porque permite detectar si
el modelo tiene un sesgo sistematico. Si los residuos se concentran alrededor de
`0`, el modelo no esta desplazado globalmente hacia arriba o hacia abajo. Si la
distribucion tiene una cola negativa, significa que hay casos donde el modelo
predice bastante por debajo del valor real; si tiene una cola positiva, ocurre
lo contrario. En esta run, los residuos estan relativamente centrados, pero
aparece una cola negativa asociada a partidas con roaming real alto que el
modelo infraestima.

![Residuos](../models/support_mlp_full_m12/diagnostics/residual_histogram.png)

El error por bins muestra que la MLP se comporta mejor en la zona media del
score y falla especialmente en los extremos:

| Rango real | Filas | MAE medio |
|---|---:|---:|
| 0.0-0.1 | 9942 | 0.1900 |
| 0.1-0.2 | 14174 | 0.1123 |
| 0.2-0.3 | 15188 | 0.0563 |
| 0.3-0.4 | 12305 | 0.0742 |
| 0.4-0.5 | 8185 | 0.1435 |
| 0.5-0.6 | 4496 | 0.2286 |
| 0.6-0.7 | 2002 | 0.3145 |
| 0.7-0.8 | 764 | 0.4019 |
| 0.8-0.9 | 267 | 0.4920 |
| 0.9-1.0 | 97 | 0.5905 |

![Error por bins](../models/support_mlp_full_m12/diagnostics/abs_error_by_score_bin.png)

La interpretacion principal es que la MLP captura priors de campeon y
composicion, pero no la intensidad concreta del roaming de una partida. Esto es
razonable: desde el draft se puede estimar tendencia, pero la ejecucion real
depende tambien de matchup, estado de linea, pathing del jungla, decisiones
tempranas y eventos no disponibles antes de comenzar la partida.

El hecho de que el error sea menor en la zona media no significa necesariamente
que haya que forzar una distribucion plana del target. Hay dos razones para ese
comportamiento. Primero, hay mas observaciones en la zona media, por lo que el
modelo recibe mas ejemplos de esos rangos. Segundo, la MLP tiende a predecir
valores cercanos a la media, lo que reduce el error en scores intermedios pero
lo aumenta en extremos. Aplanar artificialmente la distribucion podria mejorar
alguna metrica, pero tambien podria romper la semantica del problema: no todas
las partidas generan roaming intenso y no todos los campeones tienen el mismo
perfil. Por ello, el criterio no debe ser llenar uniformemente `[0,1]`, sino
mantener una etiqueta natural, no colapsada, coherente con el criterio experto y
aprendible por el modelo.

## 6. Dificultades encontradas y decisiones tomadas

### 6.1 Distancia entre intencion de draft y comportamiento observado

El proyecto intenta predecir comportamiento temprano a partir del draft. Sin
embargo, la etiqueta se calcula a partir de comportamiento observado en partida.
Esto introduce una diferencia conceptual: el draft representa una predisposicion
o contexto estrategico, mientras que el timeline refleja una ejecucion concreta
afectada por muchos factores. Esta dificultad explica por que una baseline
tabular puede aprender senal promedio, pero no todos los extremos.

### 6.2 Problemas de discretizacion

La clasificacion obligaba a convertir scores continuos en clases. Esta decision
tuvo dos problemas:

- los umbrales introducian arbitrariedad;
- todos los errores de clase penalizaban igual, sin considerar distancia.

La regresion continua conserva mas informacion y permite medir errores con una
nocion de cercania. Esta es la decision metodologica mas importante del
reinicio.

### 6.3 Clase `ambiguous`

La clase `ambiguous` fue util para analizar la zona central de la distribucion,
pero se convirtio en un cuello de botella para el aprendizaje. En muchos casos,
la clase central agrupaba ejemplos heterogeneos, cercanos a diferentes extremos
o poco definidos temporalmente. La evidencia de Fase 2 mostro que quitar o
reducir esa clase mejoraba el rendimiento de clasificacion, reforzando la idea
de que el problema debia reformularse.

### 6.4 Predicciones comprimidas de la MLP

La primera MLP full aprende senal, pero sus predicciones tienen mucha menos
varianza que el target. En validacion:

| Variable | Desviacion tipica |
|---|---:|
| Target real | 0.1710 |
| Prediccion MLP | 0.0681 |

Esto indica que el modelo tiende a la media. La siguiente fase debe comprobar si
esto mejora con tuning, regularizacion, embeddings, feature enrichment o una
representacion secuencial del timeline.

## 7. Planning hasta final de proyecto

El informe de progreso se entrega el 27/04/2026, con retraso respecto a la
fecha inicial del 19/04. El calendario siguiente usa las fechas formales
extraidas de la propuesta inicial: Informe de Progreso II el 24/05, propuesta de
informe final el 14/06, propuesta de presentacion el 21/06 y entrega final el
28/06. Si se incorpora `TFG/fechas.pdf`, este planning debera ajustarse a sus
horarios exactos.

| Periodo | Objetivo | Tareas | Entregable esperado | Criterio de exito | Riesgo/dependencia | Estado |
|---|---|---|---|---|---|---|
| 25/04-27/04 | Entregar Informe de Progreso I retrasado | Pulir recapitulacion, resultados, figuras nuevas y planning | `informe_progreso_completo.md` corregido | Informe entregable y coherente con evidencia generada | Tiempo corto | En curso |
| 28/04-03/05 | Consolidar baseline MLP | Revisar metricas, figuras, conclusiones y baseline media | Seccion de resultados MLP cerrada | Conclusion clara sobre que aprende y que no aprende | Predicciones comprimidas | Pendiente |
| 04/05-10/05 | Tuning OAT conjunto: MLP + etiqueta support | Probar regularizacion, capacidad, dropout, LR, batch size, pesos, ventana, start minute y umbral distancia ADC | Tabla comparativa por `val_mse`, curvas, ranking de heuristicas y etiqueta candidata | Mejorar o justificar configuracion base y seleccionar etiqueta candidata | Coste cluster y tradeoff distribucion/metrica | Pendiente |
| 11/05-17/05 | Embeddings y feature enrichment | Disenar features semanticas de campeones, runas, spells, arquetipos y tags | Comparacion OneHot vs enriched/embeddings | Mejora medible o descarte justificado | Diseno de features y disponibilidad de metadatos | Pendiente |
| 18/05-24/05 | Informe de Progreso II | Integrar tuning, embeddings preliminares, decisiones y figuras finales parciales | Informe II revisado | Documento listo para entrega del 24/05 | Depende de tuning y primeras pruebas de enrichment | Pendiente |
| 25/05-31/05 | Profundizar embeddings/feature enrichment y preparar expansion multi-tarea | Refinar features enriquecidas, analizar importancia, preparar contrato para nuevas etiquetas | Feature set candidato y diseno de reintroduccion jungle/team | Representacion de entrada mas informativa y plan multi-task definido | Puede no superar OneHot | Pendiente |
| 01/06-07/06 | Reintroducir jungle/team labels | Definir nuevas versiones continuas de jungle/team y generar smoke | Propuesta de etiquetas + smoke de generacion | Labels coherentes y compatibles con pipeline | Complejidad multi-task | Pendiente |
| 08/06-14/06 | Decidir RNN/GRU/LSTM con tutor y consolidar modelo candidato | Reunion, diseno secuencial si procede, smoke minimo o descarte documentado | Decision documentada, modelo candidato y estructura de memoria | Siguiente paso aprobado o descartado por alcance | Reunion tutor/tiempo | Pendiente |
| 15/06-21/06 | Prototipo terminal e interpretacion | CLI de entrada manual y traduccion score-texto | CLI minimo usable + frases interpretables | Usuario introduce draft y recibe lectura clara | Integracion features | Pendiente |
| 22/06-28/06 | Cierre final | Conclusiones, limitaciones, memoria, presentacion | Dossier final + informe + presentacion | Entrega final revisada | Tiempo de redaccion | Pendiente |

## 8. Proximos pasos tecnicos

El siguiente paso inmediato es ejecutar el tuning OAT conjunto. La idea es
probar, de forma controlada y una variable cada vez, tanto hiperparametros de la
MLP como variantes de la etiqueta support. En la parte de modelo se revisaran
regularizacion, capacidad, dropout, learning rate y batch size. En la parte de
etiqueta se probaran pesos, ventana temporal, minuto de inicio y umbral de
distancia al ADC. Aunque ambos bloques se planifiquen en la misma semana, cada
experimento debe modificar solo una dimension para mantener la interpretacion.

Si la MLP mejora de forma consistente, el siguiente bloque de trabajo sera
feature enrichment o embeddings. La motivacion es que One-Hot representa cada
campeon como una categoria independiente y no codifica relaciones semanticas:
por ejemplo, que Nautilus, Leona y Rell comparten arquetipo de engage, o que
Yuumi y Milio estan mas vinculados a proteccion del ADC. Embeddings o features
manuales podrian facilitar que el modelo generalice entre campeones con roles
similares.

La exploracion RNN/GRU/LSTM queda condicionada a resultados y tiempo. Su
motivacion es clara: la timeline de Riot es una secuencia de snapshots del
estado de partida. Si se decide explorarlo con el tutor, la pregunta seria si
modelar directamente la evolucion temporal de posiciones, experiencia y zonas
permite explicar mejor el roaming que una MLP tabular basada solo en draft.

Finalmente, tras estabilizar support, se reintroduciran nuevas versiones
continuas de las etiquetas de jungla y equipo. La experiencia obtenida con
support debe servir para evitar repetir los problemas de discretizacion,
ambiguedad y targets poco estables.

## 9. Bibliografia

[1] J.-A. Hitar-Garcia, L. Moran-Fernandez, and V. Bolon-Canedo, "Machine
Learning Methods for Predicting League of Legends Game Outcome," *IEEE
Transactions on Games*, vol. 15, no. 2, pp. 171-181, 2023.

[2] H. Lee, D. Hwang, H. Kim, B. Lee, and J. Choo, "DraftRec: Personalized Draft
Recommendation for Winning in Multi-Player Online Battle Arena Games," in
*Proc. ACM Web Conf. 2022 (WWW '22)*, pp. 3428-3439, 2022.

[3] L. M. Costa, R. G. Mantovani, F. C. M. Souza, and G. Xexeo, "Feature
Analysis to League of Legends Victory Prediction on the Picks and Bans Phase,"
in *2021 IEEE Conference on Games (CoG)*, pp. 1-5, 2021.

[4] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern
Neural Networks," in *Proc. 34th Int. Conf. Machine Learning (ICML)*, vol. 70,
pp. 1321-1330, 2017.

[5] A. M. Rama, V. Rodriguez-Fernandez, and D. Camacho, "Finding Behavioural
Patterns Among League of Legends Players Through Hidden Markov Models," in
*Applications of Evolutionary Computation*, Lecture Notes in Computer Science,
vol. 12104, pp. 419-430, 2020.

[6] G. Wallner, L. Wang, and C. Dormann, "Visualizing the Spatio-Temporal
Evolution of Gameplay using Storyline Visualization: A Study with League of
Legends," *Proc. ACM Hum.-Comput. Interact.*, vol. 7, CHI PLAY, pp. 1002-1024,
2023.

[7] Y. Chen, J. Wu, Y. Wu, and D. Liu, "T-Foresight: Interpret moving strategies
based on context-aware trajectory prediction," *Visual Informatics*, vol. 9,
no. 3, art. 100261, 2025.

[8] R. Caruana, "Multitask Learning," *Machine Learning*, vol. 28, no. 1,
pp. 41-75, 1997.

[9] Riot Games, "League of Legends Developer Documentation: Data Dragon,"
Riot Developer Portal. [Online]. Available:
https://developer.riotgames.com/docs/lol. Accessed: Apr. 25, 2026.

[10] scikit-learn developers, "OneHotEncoder," *scikit-learn documentation*.
[Online]. Available:
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html.
Accessed: Apr. 25, 2026.

[11] PyTorch contributors, "MSELoss," *PyTorch documentation*. [Online].
Available: https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html.
Accessed: Apr. 25, 2026.

[12] scikit-learn developers, "GroupShuffleSplit," *scikit-learn documentation*.
[Online]. Available:
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html.
Accessed: Apr. 25, 2026.

[13] S. Hochreiter and J. Schmidhuber, "Long Short-Term Memory," *Neural
Computation*, vol. 9, no. 8, pp. 1735-1780, 1997, doi:
10.1162/neco.1997.9.8.1735.

[14] R. Wirth and J. Hipp, "CRISP-DM: Towards a Standard Process Model for Data
Mining," in *Proc. 4th Int. Conf. Practical Applications of Knowledge Discovery
and Data Mining*, Manchester, U.K., pp. 29-40, 2000.
