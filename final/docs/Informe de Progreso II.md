# Informe de Progreso II — Borrador revisado

## Trabajo de Fin de Grado: Análisis de la relación entre la selección de personajes y el comportamiento temprano del rol de apoyo en el videojuego League of Legends

**Autor**: Adrián Díaz García  
**Tutor/a**: [Nombre del tutor/a]  
**Fecha**: mayo de 2026  
**Grado**: Ingeniería de Datos  
**Universidad**: Universitat Autònoma de Barcelona

---

## 1. Resumen de esta fase

Este informe recoge el trabajo realizado desde la entrega del Informe de Progreso I y las conclusiones provisionales que se pueden obtener a partir de esta última etapa. En el momento de escribir este documento, el TFG se encuentra ya en una fase avanzada de desarrollo. La parte principal de experimentación sobre la tarea de roaming del support está prácticamente cerrada, aunque todavía quedan tareas de integración final, redacción de la memoria y pulido del prototipo aplicado.

El punto de partida de esta fase era el resultado presentado en el Informe de Progreso I. En aquel momento, el proyecto ya había cambiado de una formulación inicial de clasificación multi-output a una tarea de regresión continua centrada en el support. Ese cambio no fue solo un cambio técnico. El problema era que el comportamiento que se quería predecir no encajaba bien en clases fijas como “bajo”, “medio” o “alto”. En realidad, primero se calculaba un score continuo a partir de la timeline de la partida, y después se intentaba convertir ese score en clases. Al hacer esa conversión se perdía información y aparecía una zona intermedia difícil de interpretar, que acababa afectando tanto al entrenamiento como a la evaluación.

Durante esta segunda fase, el proyecto se ha centrado todavía más en una pregunta concreta: **hasta qué punto la composición del draft permite anticipar el comportamiento temprano del support en League of Legends**. Para responderla, ha sido importante no mezclar dos tipos de información. Por un lado está el draft, disponible antes de que empiece la partida y usado como entrada del modelo. Por otro lado está la timeline, que describe lo que ocurrió durante la partida y se utiliza solo después para construir una etiqueta observada de roaming. Por tanto, el modelo no intenta adivinar cada movimiento del jugador. Lo que intenta es estimar si, antes de que empiece la partida, una composición hace más probable que el support abandone botlane durante los primeros minutos.

Esta diferencia entre predisposición y ejecución ha sido una de las ideas centrales de esta fase. El draft puede dar pistas sobre lo que sería esperable, pero no decide por completo cómo se va a jugar la partida. Dos botlanes parecidas pueden acabar generando scores distintos si hay muertes tempranas, recalls forzados, descoordinación, ventajas acumuladas o incluso jugadores que dejan de jugar de forma normal. Por este motivo, el resultado del modelo debe leerse con prudencia: no como una predicción exacta de la ejecución, sino como una estimación de tendencia basada en información pre-partida.

El dataset final utilizado en esta etapa contiene 383.247 observaciones de tipo match-team, procedentes de aproximadamente 191.000 partidas. Se ha revisado la geometría del mapa usada para construir la etiqueta `support_roam_score`, se ha comparado la etiqueta con una referencia experta de campeones de support, se han comparado modelos tabulares y redes neuronales, se han construido baselines sencillas y se ha estimado una referencia empírica de cuánta variabilidad parece repetible dentro de grupos de draft similares.

El mejor modelo entrenado es una variante de HistGradientBoosting con Pair Target Encoding, que alcanza en test **R² = 0.161** y **Spearman = 0.388**. Como referencia, predecir únicamente la media histórica del campeón support obtiene **R² = 0.125**, mientras que la referencia empírica por botlane+side se sitúa en **R² = 0.173**. Esta comparación es importante porque sitúa el resultado en contexto. El modelo mejora claramente una baseline sencilla, pero queda cerca de una referencia práctica estimada a partir de agrupaciones de draft.

Otro resultado importante de esta fase es que la etiqueta construida desde datos observados se alinea bien con la intuición experta inicial del proyecto. La media de `support_roam_score` por campeón alcanza una correlación de Spearman cercana a **0.82** con una referencia manual de campeones de support ordenados por tendencia esperada al roaming. Esta comparación no elimina las limitaciones de la etiqueta, pero sí da una señal positiva: aunque el score sea una aproximación imperfecta, su ranking agregado por campeón coincide bastante bien con lo que se esperaba desde conocimiento del juego. Por tanto, la conclusión principal no es que el draft determine el roaming del support. La conclusión es más limitada: el draft contiene cierta señal y permite ordenar composiciones con algo de sentido, pero una parte importante del comportamiento observado depende de lo que pasa dentro de la partida.

---

## 2. Seguimiento de la planificación y ajustes realizados

### 2.1. Punto de partida respecto al Informe de Progreso I

El Informe de Progreso I planteaba una planificación de trabajo hasta la entrega final. En ese momento, el proyecto ya había dejado atrás la clasificación multi-output inicial y se había centrado en una primera tarea de regresión continua para el roaming del support. El objetivo era validar bien esta primera tarea antes de reintroducir nuevas etiquetas de jungla y equipo.

La planificación prevista incluía, de forma resumida, los siguientes bloques: tuning de la MLP y de la etiqueta de support, pruebas con embeddings y enriquecimiento de features, cierre de la tarea de support, redacción del Informe II, redefinición de etiquetas de jungla y equipo, integración multi-output, prototipo por terminal y preparación de la entrega final.

El seguimiento de esta planificación ha sido parcial. Algunos bloques se han completado, otros se han reformulado y otros se han dejado como trabajo futuro. El cambio principal no consiste en abandonar el objetivo general, sino en acotar mejor el trabajo: antes de ampliar el sistema a varias tareas, hacía falta entender qué parte del comportamiento del support podía aprenderse realmente desde el draft y qué parte dependía de lo que ocurría después en la partida.

### 2.2. Resumen de seguimiento de la planificación

| Bloque previsto | Seguimiento |
|---|---|
| MLP y etiqueta support | Reformulado: se sustituyó el OAT inicial por baselines, búsqueda de hiperparámetros, transformación quantile y variantes de etiqueta. |
| Embeddings y feature enrichment | Ejecutado: se probaron embeddings y variantes enriquecidas, pero no mejoraron claramente al modelo tabular. |
| Cierre de support | Ampliado: se añadieron ICC, explicabilidad, revisión de errores y análisis de partidas caóticas. |
| Jungla, equipo y multi-output | Replanificado: queda como trabajo futuro para no abrir nuevas tareas sin una evaluación equivalente. |
| Prototipo por terminal | Adelantado: ya existe una versión funcional con carga del modelo final, salida interpretable y modos de uso interactivo/no interactivo. |

En conjunto, la planificación se ha seguido de forma parcial. La tarea de support ha ocupado más espacio del previsto porque los primeros resultados mostraron que no bastaba con entrenar un modelo y comparar métricas. Antes de ampliar el sistema a nuevas tareas, hacía falta entender con qué baselines comparar el resultado, si la etiqueta era estable y qué parte del error podía venir de partidas con desarrollo anómalo. Por este motivo, parte del tiempo previsto para jungla, equipo e integración multi-output se reasignó a cerrar mejor la tarea de support.

### 2.3. Justificación de los ajustes

El primer ajuste relevante afectó al tuning OAT previsto para la MLP y la etiqueta. Se había diseñado una batería de experimentos donde se modificaba una variable cada vez: pesos de la etiqueta, ventanas temporales y parámetros de la red. La ejecución completa estaba prevista en el cluster de la universidad, pero quedó bloqueada durante varios días por problemas de disponibilidad. Este problema afectó a la secuencia prevista de trabajo, aunque no fue la razón principal del cambio de rumbo.

La razón más importante fue que el resultado heredado del Informe I, una primera MLP con R² aproximado de 0.13, todavía era difícil de interpretar. El número por sí solo no decía si el modelo estaba funcionando razonablemente bien, si apenas mejoraba una regla trivial o si estaba lejos de lo que podía esperarse con información pre-partida. Por eso, en lugar de esperar a completar el OAT original, se decidió responder primero a una pregunta más básica: **con qué había que comparar ese R² para saber si era un resultado razonable o no**.

A partir de ahí, el trabajo se reorientó hacia tres líneas. En primer lugar, se construyeron baselines sencillas, como predecir siempre la media global o usar la media histórica del campeón support. En segundo lugar, se entrenaron modelos tabulares adecuados para variables categóricas, especialmente HistGradientBoosting, para comprobar si una arquitectura distinta aprovechaba mejor el draft. En tercer lugar, se estimó una referencia empírica mediante agrupaciones de draft, como campeón support, botlane y botlane+side, para aproximar cuánta variabilidad parecía repetible dentro de condiciones pre-partida similares.

También se abordó parte del tuning previsto mediante una búsqueda local de hiperparámetros de la MLP. Se probó un grid de 108 configuraciones, variando tamaño de capas, dropout, learning rate y weight decay. La mejor configuración mejoró solo alrededor de 0.005 puntos de Spearman en validación respecto a la configuración de referencia, y en test siguió sin superar al modelo tabular. Este resultado refuerza una lectura importante: la principal limitación no parece estar en un ajuste fino insuficiente de la red, sino en la dificultad de la etiqueta y en la cantidad de información realmente disponible antes de la partida.

Esta decisión permitió entender mejor los resultados. La pregunta dejó de ser solo “¿puedo mejorar la MLP?” y pasó a ser “¿cuánta información útil hay realmente en el draft para esta tarea?”. Esta segunda pregunta ayudaba más a dirigir el proyecto, porque permitía decidir si tenía sentido seguir complicando el modelo, revisar la etiqueta o aceptar que parte del límite venía del propio problema.

El segundo ajuste importante fue posponer las tareas de jungla y equipo. En la planificación original, estas etiquetas debían reintroducirse después de cerrar support. Sin embargo, al profundizar en support quedó claro que incluso una sola etiqueta requería bastante trabajo: revisar la geometría del mapa, comprobar la fórmula, construir baselines, estimar una referencia empírica, explicar el modelo y revisar errores concretos. Repetir ese nivel de análisis para tres tareas en el tiempo restante habría obligado a hacer una extensión más superficial.

Por este motivo, se decidió cerrar con más rigor una tarea concreta antes de ampliar el alcance. Esta decisión no elimina el objetivo general de estudiar patrones tempranos desde el draft, pero sí acota el desarrollo final del TFG. En lugar de presentar varias salidas menos trabajadas, el proyecto se centra en una tarea de support más completa, mejor evaluada y con limitaciones más claras.

### 2.4. Trabajo incorporado durante esta fase

Aunque algunas tareas previstas se replanificaron, durante esta etapa se incorporó trabajo adicional que no aparecía con ese nivel de detalle en el calendario inicial.

En primer lugar, se revisó la geometría del mapa usada para construir la etiqueta de roaming. La geometría inicial derivada automáticamente de densidades de jugadores fue sustituida por una geometría manual más fácil de interpretar, trazada sobre el mapa del juego. Este cambio fue importante porque la etiqueta depende directamente de qué posiciones se consideran contexto de botlane, río, midlane, jungla o zona de dragón.

En segundo lugar, se construyeron baselines y una referencia empírica mediante ICC. Esto permitió contextualizar el rendimiento del modelo frente a reglas sencillas y frente a la variabilidad repetible dentro de grupos de draft similares.

En tercer lugar, se realizó una revisión cualitativa de errores. Esta revisión mostró que muchos errores extremos aparecen en partidas con desarrollo temprano anómalo: por ejemplo, partidas donde una botlane empieza muy mal, acumula muchas muertes y los jugadores dejan de comportarse de forma cooperativa. En esos casos, el score observado puede subir porque el support se separa del ADC o abandona su zona habitual, pero esa separación ya no refleja necesariamente una predisposición estratégica del draft. Refleja más bien una partida que se ha desordenado por cómo han jugado los jugadores.

En cuarto lugar, se probaron variantes de la etiqueta y embeddings de campeones. Las variantes de etiqueta permitieron comprobar si pequeñas modificaciones de pesos o componentes cambiaban las conclusiones. También se exploraron variantes basadas en eventos productivos, como kills, asistencias, objetivos, placas o estructuras fuera del contexto de botlane. Estas variantes no se mantuvieron como etiqueta principal porque desplazaban la pregunta hacia el roaming que acaba produciendo eventos visibles, que depende mucho más del desarrollo concreto de la partida. Los embeddings sirvieron para estudiar si una representación numérica de campeones aprendía relaciones útiles entre ellos. En ambos casos, los resultados fueron útiles más por su interpretación que por la mejora directa de métricas: las variantes no cambiaron sustancialmente la señal, y los embeddings no superaron a los modelos tabulares.

También se probó una transformación quantile de la etiqueta para comprobar si una escala más uniforme facilitaba el aprendizaje. Como no mejoró los resultados finales en la escala original del score, se mantuvo la etiqueta raw como referencia principal.

Finalmente, se adelantó una versión funcional del prototipo por terminal. El prototipo permite introducir una composición de draft, rellenar hechizos de invocador por defecto cuando no se proporcionan, cargar el modelo final y devolver una lectura interpretable del perfil esperado de roaming. Además, puede ejecutarse de forma interactiva, por argumentos o en modo batch, lo que facilita usarlo tanto como demostración aplicada como para pruebas rápidas sobre varios drafts.

### 2.5. Planning actualizado hasta la entrega final

|  |  |
|---|---|
| **Periodo** | 21/05 – 24/05 |
| **Objetivo** | Informe de Progreso II |
| **Tarea** | Integrar los resultados de esta fase, revisar la redacción y ajustar el documento al formato final. |
| **Entregable esperado** | Informe de Progreso II revisado |
| **Criterio de éxito** | Documento listo para entrega |

Este bloque consiste en transformar los resultados técnicos de la fase actual en un documento claro y entregable. La prioridad no es añadir nuevos experimentos, sino explicar bien los cambios de alcance, la metodología final y la interpretación de los resultados.

|  |  |
|---|---|
| **Periodo** | 25/05 – 31/05 |
| **Objetivo** | Pulido del prototipo por terminal |
| **Tarea** | Revisar la entrada manual del draft, cargar el modelo candidato y traducir el score a frases interpretables. |
| **Entregable esperado** | CLI funcional con salida comprensible |
| **Criterio de éxito** | El usuario introduce un draft y recibe una lectura clara del perfil de roaming |

En este bloque se pulirá el prototipo por terminal. La funcionalidad principal ya existe: el sistema carga el modelo entrenado, acepta drafts manuales o por argumentos, completa hechizos por defecto cuando hace falta y devuelve una predicción para la tendencia de roaming del support. La tarea pendiente es dejar la salida más limpia para la entrega final, de forma que el usuario no reciba solo un valor como 0.37, sino una lectura comprensible del perfil esperado: laning, mixto, roaming moderado o roaming intenso.

|  |  |
|---|---|
| **Periodo** | 01/06 – 07/06 |
| **Objetivo** | Redacción inicial de la memoria |
| **Tarea** | Redactar contexto, objetivos, metodología y construcción de la etiqueta. |
| **Entregable esperado** | Primer borrador de capítulos iniciales |
| **Criterio de éxito** | Estructura de memoria consolidada y revisable |

La memoria final reutilizará parte del material de los informes de progreso, pero con una estructura más limpia. En esta fase se priorizarán los capítulos de contexto, objetivos, metodología y construcción de la etiqueta, porque son los que explican mejor cómo ha evolucionado el proyecto.

|  |  |
|---|---|
| **Periodo** | 08/06 – 13/06 |
| **Objetivo** | Resultados, discusión y figuras finales |
| **Tarea** | Integrar tablas, figuras, comparación de modelos, limitaciones y conclusiones. |
| **Entregable esperado** | Borrador completo de memoria |
| **Criterio de éxito** | Documento completo enviado al tutor para revisión |

En este bloque se integrarán las tablas y figuras finales, especialmente la comparación de modelos, la referencia empírica, la importancia de variables y la revisión de errores. También sería deseable añadir un pequeño estudio de ablación del modelo tabular para cuantificar mejor qué aporta cada grupo de variables.

|  |  |
|---|---|
| **Periodo** | 14/06 – 28/06 |
| **Objetivo** | Cierre final y defensa |
| **Tarea** | Incorporar correcciones, preparar presentación y revisar la defensa oral. |
| **Entregable esperado** | Memoria final, prototipo y presentación |
| **Criterio de éxito** | Entrega final revisada y defensa preparada |

El cierre final consistirá en incorporar las correcciones del tutor, revisar la memoria, preparar la presentación y ensayar la defensa. En esta parte será importante insistir en la idea principal del proyecto: no se ha construido un predictor exacto de comportamiento, sino una herramienta para estudiar cuánta información útil hay en el draft antes de la partida y hasta dónde llega esa información.

---

## 3. Diseño y evolución de la etiqueta de support

### 3.1. Qué se intenta medir

La construcción de la etiqueta `support_roam_score` ha sido una de las partes más importantes del proyecto. En los datos de Riot Games no existe una variable que diga directamente si un support está roameando. La API proporciona posiciones, eventos y recursos, pero no interpreta tácticamente lo que está haciendo cada jugador. Por eso, la etiqueta debe construirse de forma aproximada a partir de lo que se observa en la timeline.

En League of Legends, el roaming del support puede entenderse como el movimiento del support fuera de la botlane para influir en otras zonas del mapa. Un support puede abandonar temporalmente a su ADC para ayudar al jungler, presionar la línea central, colocar visión, participar en objetivos neutrales o generar ventaja en otra parte del mapa. Sin embargo, no toda separación entre support y ADC significa roaming estratégico. A veces el support se separa porque el ADC ha muerto, porque ambos han hecho recall en momentos distintos, porque la línea está perdida o porque la partida se ha vuelto caótica.

Por este motivo, la etiqueta no debe interpretarse como una medida perfecta de la intención del jugador. Es una aproximación basada en comportamiento observado. Mide hasta qué punto el support aparece fuera del contexto de botlane o lejos del ADC durante una ventana temprana de la partida. Esta diferencia entre intención y observación ha sido una limitación constante del proyecto.

### 3.2. De clasificación discreta a score continuo

La propuesta inicial del TFG planteaba una clasificación discreta de patrones tempranos. En el caso del support, esto implicaba clasificar las observaciones en categorías como support anclado a línea, support ambiguo o support roamer. Este planteamiento resultó problemático porque el comportamiento real no se separaba bien en grupos tan claros. Muchos ejemplos quedaban en una zona intermedia: no eran claramente supports anclados a la línea, pero tampoco roamers extremos.

En las fases anteriores del proyecto se observó que esta clase intermedia dificultaba el aprendizaje. El problema no era solo técnico. Si el roaming se calcula primero como una puntuación gradual, convertirlo después en clases obliga a fijar cortes artificiales. Además, dos partidas con scores muy parecidos pueden acabar en clases distintas si caen a lados opuestos de un umbral, mientras que dos partidas mucho más diferentes pueden recibir una penalización similar si se evalúan solo como clases.

Por eso, en el Informe de Progreso I se cambió el planteamiento a regresión continua. En lugar de predecir una clase, el modelo predice un score en el rango [0, 1]. Valores bajos indican un support más ligado a botlane y al ADC; valores altos indican una mayor tendencia observada a abandonar botlane o jugar lejos del ADC durante los primeros minutos.

### 3.3. Evolución de la geometría del mapa

Para calcular el score de roaming es necesario clasificar la posición del support en distintas zonas del mapa. Esta decisión es más importante de lo que parece, porque determina qué se considera “estar en botlane” y qué se considera “estar fuera de bot”.

Las primeras versiones usaban una geometría derivada automáticamente de la densidad observada de jugadores. La idea era razonable como punto de partida: las zonas más transitadas por ciertos roles podían ayudar a identificar líneas, jungla y río. Sin embargo, este enfoque tenía dos limitaciones. En primer lugar, dependía demasiado de la propia muestra: una zona poco transitada no es necesariamente irrelevante o imposible de recorrer, sino quizá poco usada en esas partidas. En segundo lugar, las fronteras resultantes no siempre coincidían con cómo se entiende el mapa dentro del juego. Por ejemplo, el río podía quedar como una diagonal demasiado fina y algunas zonas cercanas al dragón no quedaban bien diferenciadas.

Durante esta fase se sustituyó esa geometría automática por una geometría manual trazada sobre el mapa del juego. Se definieron zonas como BOT_LANE_CORE, BOT_SIDE_NEAR, RIVER_BOT, DRAGON_AREA, MID_LANE, junglas por cuadrante y bases. Esta geometría busca representar mejor cómo se entiende el mapa dentro del juego, no solo dónde aparecen más posiciones en los datos.

La decisión más relevante fue definir el contexto de botlane de forma que tuviera sentido dentro del juego. No incluye solo la línea inferior estricta, sino también zonas cercanas que forman parte del comportamiento normal de una botlane, como el área próxima al dragón o el río inferior. En cambio, posiciones en midlane, jungla superior o zonas alejadas del ADC se interpretan como señales más compatibles con roaming.

La geometría final se muestra en la Figura 1. La figura ayuda a ver por qué esta definición no es solo un detalle técnico: las zonas marcadas determinan cuándo una posición del support se considera cercana a botlane y cuándo se interpreta como un desplazamiento hacia otra parte del mapa.

*Figura 1*

**Fig. 1. Geometría manual utilizada para clasificar las posiciones del support.** La figura muestra las zonas del mapa usadas para decidir si el support permanece en el contexto de botlane o se desplaza hacia otras zonas.

### 3.4. Fórmula principal del `support_roam_score`

La etiqueta principal utilizada en la comparación experimental final combina tres componentes:

```
score_raw = 0.45 × outside_ratio + 0.35 × far_ratio + 0.20 × xp_gap
score = score_raw ^ 0.75
```

El primer componente, `outside_ratio`, mide la proporción de frames válidos en los que el support está fuera del contexto de botlane. Es la señal espacial principal: si el support pasa más tiempo en zonas alejadas de botlane, el score aumenta.

El segundo componente, `far_ratio`, mide la proporción de frames en los que el support está lejos del ADC. Este componente es importante porque un support puede estar físicamente dentro de una zona cercana a botlane pero separado de su compañero, o puede moverse con el ADC y no estar realmente roameando. La distancia support-ADC ayuda a distinguir estos casos.

El tercer componente, `xp_gap`, utiliza la diferencia relativa de experiencia entre support y ADC como señal auxiliar. La intuición es que un support que se separa a menudo del ADC puede acumular experiencia de forma distinta. Aun así, este componente se mantiene con menor peso porque puede estar más influido por el desarrollo concreto de la partida.

La transformación final `score_raw ^ 0.75` sirve para ajustar la escala. No cambia el orden de los ejemplos, pero estira ligeramente la distribución para que los valores no queden tan concentrados en la zona baja. Esto es útil porque la mayoría de supports no roamean de forma extrema durante toda la ventana analizada.

También se probó una transformación quantile zero-preserved. La idea era sencilla: si la etiqueta está muy concentrada en una zona del rango, una distribución más plana podría facilitar el entrenamiento de algunos modelos, especialmente las redes neuronales. Esta transformación conserva los casos con score cero y reescala los valores positivos según su posición en la distribución. Para evitar data leakage, el transformador se ajustó únicamente con el conjunto de train y después se aplicó a validation y test. La prueba fue útil para descartar que el problema viniera solo de la forma de la distribución, pero no mejoró la comparación final en la escala original.

La distribución final de la etiqueta se muestra en la Figura 2. Como se observa, el score no queda repartido de forma uniforme en todo el rango [0, 1], pero tampoco se colapsa en un único valor. Esto es coherente con el fenómeno que se está midiendo: la mayoría de supports permanecen cerca de botlane durante buena parte de los primeros minutos, y solo una parte de las observaciones muestra roaming intenso.

*Figura 2*

**Fig. 2. Distribución final de la etiqueta `support_roam_score`.** La mayoría de observaciones se concentran en valores bajos o medios, lo cual es esperable en una métrica de roaming temprano.

### 3.5. Comparación experta, robustez y revisión de variantes

Después de definir la etiqueta principal, se revisó si el score tenía sentido desde tres puntos de vista: alineación con conocimiento experto, robustez frente a cambios de fórmula y utilidad para el aprendizaje.

La comprobación cualitativa más importante fue la comparación con una referencia experta de campeones de support. Esta referencia se construyó manualmente a partir de la intuición inicial del proyecto: qué campeones deberían estar más asociados a roaming y cuáles deberían estar más ligados a botlane. Al comparar la media empírica del `support_roam_score` por campeón con esa referencia, se obtuvo una correlación de Spearman cercana a **0.82**. Este resultado es especialmente relevante porque no evalúa una predicción caso por caso, sino si la etiqueta recupera el orden esperado entre campeones. Campeones como Bard, Pyke o Alistar aparecen en la zona alta, mientras que Yuumi, Lulu o Sona quedan en la zona baja. Para un proyecto donde la etiqueta es necesariamente una aproximación, esta alineación es una de las señales más fuertes de que el score está capturando una parte real del fenómeno.

Después se probaron variantes para comprobar si el resultado dependía demasiado de una decisión concreta de pesos o componentes. Se modificaron pesos, se añadieron pequeñas señales auxiliares y se realizaron ablaciones de componentes. Todas las variantes quedaron muy correlacionadas con la etiqueta principal, con correlaciones superiores a 0.99.

Esto no significa que la etiqueta sea perfecta. Significa que, dentro de las variantes probadas, pequeños cambios en la fórmula no cambian mucho la señal medida. Por tanto, el rendimiento limitado de los modelos no parece explicarse simplemente por haber elegido mal un peso concreto.

También se probaron variantes más orientadas a medir roaming productivo observado, incorporando eventos como kills, asistencias, objetivos, placas o estructuras. Estas variantes son interesantes, pero cambian un poco la pregunta. Miden mejor si el roaming produjo eventos visibles, pero eso depende mucho más de cómo fue la partida y menos de lo que se podía esperar solo mirando el draft. Por ejemplo, una jugada puede acabar en kill o asistencia por coordinación, estado de oleada, intervención del jungla o error rival, factores que no están disponibles en la selección de campeones. Para el objetivo actual, centrado en información pre-partida, se mantiene la etiqueta espacial como opción principal y se dejan las variantes basadas en eventos como una línea futura o complementaria.

La conclusión de esta revisión es que la etiqueta principal sigue siendo la mejor aproximación disponible para la pregunta del TFG: medir predisposición espacial al roaming desde el draft. La etiqueta conserva ruido, especialmente en partidas caóticas, pero las alternativas probadas no ofrecen una mejora clara sin cambiar el significado del objetivo.

La comparación entre variantes se resume en la Figura 3. Como las diferencias son pequeñas, no parece que la conclusión principal dependa de una única elección concreta de pesos o componentes.

*Figura 3*

**Fig. 3. Comparación entre variantes de la etiqueta de roaming.** Las variantes probadas producen resultados muy parecidos, por lo que cambiar ligeramente la fórmula no modifica la interpretación general.

---

## 4. Metodología seguida finalmente

### 4.1. Datos y unidad de análisis

Los datos proceden de la API de Riot Games. Para cada partida se utiliza, por un lado, la información de composición y metadatos de la partida, y por otro, la timeline con posiciones y eventos a lo largo del tiempo. La muestra final procede del servidor EUW, de partidas Ranked Solo/Duo de nivel alto y de los parches 16.2 a 16.8. La unidad de análisis se mantiene como `(match_id, team_id)`, igual que en el Informe de Progreso I. Esto significa que cada partida genera dos observaciones: una desde la perspectiva del equipo azul y otra desde la perspectiva del equipo rojo.

Esta formulación permite analizar tendencias de equipo sin reducir cada partida a un único resultado global. Además, obliga a cuidar el split de datos: las dos observaciones de una misma partida no pueden quedar separadas entre entrenamiento y test, porque eso introduciría fuga de información.

El dataset final contiene 383.247 observaciones match-team. Las variables de entrada son exclusivamente pre-partida: 10 campeones aliados/enemigos por rol, 20 hechizos de invocador y el lado del mapa. La timeline no se usa como entrada del modelo. Solo se utiliza para construir la etiqueta observada de roaming.

### 4.2. Pipeline de procesamiento

El pipeline final separa claramente cinco pasos:

1. **Construcción del estado por frame.** A partir de las timelines se extraen posiciones del support y del ADC durante la ventana temprana de análisis. Cada posición se clasifica usando la geometría manual del mapa.

2. **Construcción de variables de draft.** A partir de la información pre-partida se generan las variables categóricas del modelo: campeones, roles, hechizos de invocador y lado.

3. **Cálculo de la etiqueta.** A partir del estado observado en la timeline se calcula `support_roam_score`, combinando posición fuera de botlane, distancia al ADC y diferencia relativa de experiencia.

4. **Preparación de splits y escalas de entrenamiento.** Se genera la escala original del score y una versión quantile zero-preserved ajustada solo en train. La escala original se mantiene como referencia final porque es más interpretable y obtiene los mejores resultados principales, mientras que la quantile sirve para comprobar si aplanar la distribución facilita el aprendizaje.

5. **Marcado de partidas caóticas.** Se añaden variables auxiliares como una marca de partida caótica (`chaos_flag`), pesos de muestra (`sample_weight`), número de frames válidos y confianza final. Estas variables no forman parte del draft de entrada, sino del protocolo de entrenamiento y análisis de calidad.

La separación entre entrada y etiqueta es fundamental. El modelo solo recibe información que estaría disponible antes del minuto cero. La etiqueta se calcula después, observando lo que ocurrió realmente durante los primeros minutos. Esto evita data leakage y mantiene el objetivo aplicado del proyecto: analizar hasta qué punto el draft contiene señal sobre patrones tempranos.

La Figura 4 resume este flujo de trabajo. La parte importante es que el draft y la timeline se usan para cosas distintas: el draft genera las variables de entrada, mientras que la timeline se usa solo para calcular la etiqueta observada.

*Figura 4*

**Fig. 4. Pipeline final de datos y entrenamiento.** El modelo utiliza información pre-partida como entrada, mientras que la timeline se emplea únicamente para construir la etiqueta de roaming.

### 4.3. Splits y evaluación

Los datos se dividen en train, validation y test usando grupos por `match_id`. De esta manera, las dos observaciones asociadas a una misma partida quedan siempre en el mismo split. Esta decisión evita que el modelo vea durante entrenamiento una perspectiva de una partida y sea evaluado sobre la otra. El split persistido contiene aproximadamente 268.000 observaciones en train, 57.000 en validation y 57.000 en test.

El conjunto de validation se utiliza para comparar configuraciones y decidir qué modelos pasan a la comparación final. El conjunto de test se reserva para la evaluación final, de forma que todas las métricas principales se calculen sobre una partición no usada durante el ajuste.

Además, se marca un subconjunto de partidas caóticas. Estas partidas no se eliminan, porque forman parte del fenómeno observado, pero reciben menor peso durante el entrenamiento. La razón es que en ellas el score puede estar más contaminado por ejecución anómala que por predisposición del draft.

El `chaos_flag` se define a partir de eventos tempranos de botlane. Una observación se considera caótica si support y ADC acumulan al menos seis muertes antes del minuto 12, si el ADC muere al menos cinco veces, o si el support muere al menos cuatro veces sin acciones fuera de bot que sugieran una intención de roam productivo. También se exige un mínimo de tres frames válidos del support para conservar una observación. Con este criterio, alrededor del 26.5% de las observaciones quedan marcadas como caóticas. Las observaciones limpias reciben `sample_weight = 1.0`, mientras que las caóticas reciben `sample_weight = 0.2` durante el entrenamiento.

Esta decisión intenta reflejar una característica importante del dominio. League of Legends es un juego muy volátil: una línea puede decidirse en pocos minutos por una pelea mal jugada, un gank, un dive o una cadena de recalls y muertes. En esos casos, la posición posterior del support puede dejar de representar una decisión estratégica de roaming y pasar a ser una consecuencia de que la partida se ha desordenado. El filtrado por caos no elimina esa realidad, pero evita que esos casos dominen de forma desproporcionada el ajuste del modelo.

### 4.4. Modelos y baselines comparados

La comparación incluye baselines sencillas, modelos tabulares y redes neuronales.

Las baselines son necesarias para interpretar los resultados. La primera, Global Mean, predice siempre la media global del score. Es una referencia mínima: cualquier modelo útil debería mejorarla. La segunda, Champion Mean, predice la media histórica del campeón support en train. Esta baseline es especialmente importante porque en League of Legends el propio campeón support ya contiene mucha información sobre el estilo de juego esperado.

Como modelo tabular principal se utiliza HistGradientBoosting. Este tipo de modelo es adecuado para variables categóricas y puede capturar interacciones no lineales entre campeones, hechizos y lado del mapa. Se prueban tres variantes: una versión base, una versión con arquetipos expertos de support y una versión con Pair Target Encoding para resumir el comportamiento histórico de parejas support-ADC sin utilizar información de test.

También se entrenan varias MLPs. La primera usa codificación One-Hot, como continuación natural del Informe I. Las otras incorporan embeddings de campeones, tanto compartidos como separados por rol. La motivación de los embeddings es permitir que el modelo aprenda una representación numérica de los campeones, en lugar de tratarlos solo como categorías independientes.

Para comprobar si la red estaba limitada por una mala configuración concreta, se ejecutó además un grid de hiperparámetros de 108 combinaciones sobre la MLP con embeddings por rol e interacciones. Se variaron tamaños de capas, dropout, learning rate y weight decay. La mejora de la mejor configuración frente al modelo de referencia fue muy pequeña, por lo que el resultado se interpreta como una señal más de que el límite principal no está en el ajuste fino de la MLP.

### 4.5. Métricas

La evaluación no se basa en una única métrica. Se utiliza R² para medir qué proporción de la varianza observada explica el modelo respecto a predecir siempre la media. También se utiliza Spearman, que mide si el modelo ordena correctamente los drafts de menor a mayor tendencia al roaming. Esta métrica es importante porque, para una herramienta de análisis previo a la partida, puede ser útil ordenar composiciones aunque el valor exacto del score no sea perfecto.

Además, se calculan MAE y RMSE para medir error numérico medio, y métricas de cercanía como el porcentaje de predicciones dentro de ±0.10 o ±0.20. Estas métricas ayudan a traducir el rendimiento a una escala más interpretable dentro del rango [0, 1].

También se comparan modelos entrenados sobre la escala original y sobre la escala quantile. Esta comparación no cambia la métrica principal del informe, pero permite separar dos cuestiones: si el modelo falla por la forma de la distribución del target o si el límite viene de la propia información disponible. Como las variantes quantile no mejoran la evaluación final en escala original, la interpretación principal se mantiene sobre la etiqueta original.

Por último, se calcula una referencia empírica mediante ICC y medias por grupo. Esta referencia no es un modelo que prediga caso por caso, ni un límite matemático absoluto. Sirve como aproximación práctica: si dentro de grupos de draft parecidos sigue habiendo mucha variabilidad, entonces una parte de esa variabilidad probablemente no se puede explicar solo con información pre-partida.

---

## 5. Resultados obtenidos

### 5.1. Comparación principal en test

La tabla siguiente resume la comparación final en el conjunto de test. La primera fila corresponde a la referencia empírica por botlane+side. No debe interpretarse igual que los modelos, porque no es un predictor entrenado de la misma forma, sino una referencia para contextualizar el orden de magnitud de la varianza explicable.

| Modelo | R² | Spearman | MAE | within ±0.10 | within ±0.20 |
|---|---:|---:|---:|---:|---:|
| Referencia empírica: media por botlane+side | 0.173 | — | — | — | — |
| HistGBT + Pair Target Encoding | **0.161** | **0.388** | **0.141** | 41.8% | 74.2% |
| HistGBT + Archetypes | 0.161 | 0.388 | 0.141 | — | — |
| HistGBT base | 0.160 | 0.387 | 0.141 | — | — |
| MLP OneHot | 0.155 | 0.381 | 0.141 | 41.9% | 74.1% |
| MLP Per-Role + Interactions | 0.154 | 0.381 | 0.141 | 41.8% | 74.1% |
| MLP Embed compartido | 0.150 | 0.376 | 0.142 | — | — |
| Champion Mean | 0.125 | 0.336 | 0.144 | 41.1% | 72.9% |
| Global Mean | 0.000 | — | 0.155 | 37.8% | 68.8% |

La comparación muestra tres ideas principales. La primera es que el draft contiene señal: todos los modelos útiles mejoran claramente la media global. La segunda es que una parte importante de esa señal ya está en el campeón support, como demuestra la baseline Champion Mean. La tercera es que el resto del draft añade información, pero con un margen limitado: el mejor modelo mejora la media por campeón, aunque no se aleja mucho de ella.

El mejor resultado lo obtiene HistGBT + Pair Target Encoding, con R² = 0.161 y Spearman = 0.388. La mejora respecto a HistGBT base es muy pequeña, por lo que no debe interpretarse como una diferencia fuerte entre variantes. La conclusión más importante no es que una versión concreta sea claramente superior, sino que los modelos tabulares se sitúan ligeramente por encima de las MLPs probadas y cerca de la referencia empírica disponible.

La misma comparación se muestra visualmente en la Figura 5. La figura permite ver con más claridad que los modelos tabulares quedan ligeramente por encima de las MLPs probadas, aunque las diferencias entre los mejores modelos son pequeñas.

*Figura 5*

**Fig. 5. Comparación de modelos en el conjunto de test.** Los modelos tabulares obtienen los mejores resultados, aunque la mejora respecto a algunas variantes neuronales es moderada.

### 5.2. Techo empírico mediante agrupaciones de draft

Para contextualizar los resultados se calcularon referencias por grupos de draft. La idea es sencilla: si se agrupan partidas por el mismo campeón support, la misma botlane o la misma botlane jugando en el mismo lado, se puede observar cuánta variabilidad del score parece estable dentro de esas condiciones.

| Agrupación | ICC | R² usando media por grupo |
|---|---:|---:|
| Support champion | 0.121 | 0.121 |
| Botlane champions | 0.139 | 0.161 |
| Botlane champions + side | 0.139 | **0.173** |
| Support archetype | 0.084 | 0.081 |

La referencia más alta aparece al agrupar por botlane+side, con R² = 0.173. El mejor modelo alcanza R² = 0.161, por lo que queda cerca de esta referencia. Esto no significa que sea imposible mejorar. El ICC depende de las agrupaciones elegidas y no representa un límite teórico absoluto. Sin embargo, sí sugiere que, con las variables pre-partida actuales, buena parte de la señal repetible ya está siendo capturada.

Esta lectura es importante para dirigir el proyecto. Si el modelo estuviera muy lejos de la referencia empírica, tendría sentido buscar una mejora grande en arquitectura o representación. En cambio, al quedar relativamente cerca, parece más razonable interpretar que el límite observado no se debe solo a una mala elección de modelo. Parte de la variabilidad del roaming observado depende de cómo se desarrolla la partida.

La Figura 6 muestra esta comparación de forma visual. La referencia por botlane+side es la más alta, lo que sugiere que la pareja de botlane y el lado del mapa recogen buena parte de la información que se repite en condiciones pre-partida parecidas.

*Figura 6*

**Fig. 6. Referencia empírica mediante agrupaciones de draft.** Agrupar por botlane y lado ayuda a estimar cuánta variabilidad del score parece repetirse bajo condiciones pre-partida parecidas.

### 5.3. Comparación con referencia experta

Uno de los resultados más positivos de esta fase aparece al comparar la etiqueta con una referencia experta construida al inicio del proyecto. Esta referencia ordena campeones de support según la tendencia esperada al roaming desde conocimiento del juego. No se utiliza para entrenar el modelo ni para ajustar la fórmula del score; se usa únicamente como contraste externo de sentido.

La comparación se realiza agregando el `support_roam_score` por campeón y midiendo si el orden empírico coincide con el orden experto. La correlación de Spearman obtenida es aproximadamente **0.82**. Este valor es alto para una etiqueta construida automáticamente desde posiciones minutales y es uno de los resultados más esperanzadores del trabajo. Resulta importante por dos motivos. Primero, indica que el score no está capturando solo ruido: cuando se agregan muchas partidas, los campeones que intuitivamente deberían roamear más tienden a aparecer arriba. Segundo, ofrece una comprobación independiente de la escala exacta del score, porque lo que se evalúa es el ranking.

La lectura cualitativa también es coherente. Supports diseñados para moverse por el mapa, como Bard, Pyke o Alistar, aparecen con medias más altas. En cambio, campeones más ligados al ADC o a la fase de línea, como Yuumi, Lulu o Sona, aparecen en la zona baja. Este resultado no resuelve todos los problemas de la etiqueta, especialmente en partidas caóticas, pero sí da una base sólida para defender que la aproximación tiene información de dominio.

### 5.4. Importancia de variables y explicación del modelo

El análisis de importancia por permutación muestra que la señal principal procede de los campeones aliados, especialmente del campeón support. Esto es coherente con el dominio: el propio campeón support condiciona mucho su tendencia natural a moverse por el mapa. Un campeón como Bard o Pyke está diseñado para moverse y generar presión fuera de botlane, mientras que campeones más ligados al ADC tienden a quedarse cerca de la línea.

| Grupo de features | Importancia total aproximada |
|---|---:|
| Campeones aliados | 0.255 |
| Campeones enemigos | 0.033 |
| Summoner spells | 0.001 |
| Side | ~0 |

La variable individual más importante es el campeón support aliado. Después aparecen, con menor peso, el ADC aliado y el support enemigo. Este orden tiene sentido: el estilo del support depende principalmente de su propio kit, pero también de la pareja de botlane y del emparejamiento contra la botlane rival.

También se utilizó SHAP para inspeccionar el comportamiento del modelo. SHAP no se usa aquí para demostrar causalidad, sino como herramienta de explicación: permite ver qué variables empujan una predicción hacia scores más altos o más bajos. Los resultados son coherentes con la importancia por permutación y no muestran que el modelo dependa principalmente de variables extrañas o poco interpretables.

Esta lectura también se observa en la Figura 7. La mayor parte de la importancia se concentra en los campeones aliados, especialmente en el support, lo cual es coherente con el objetivo del modelo.

*Figura 7*

**Fig. 7. Importancia de los grupos de variables en el modelo tabular.** La información más relevante procede de los campeones aliados, especialmente del campeón support.

### 5.5. Auditoría cualitativa de errores

Además de las métricas agregadas, se revisaron casos concretos con errores altos. Esta revisión es importante porque un error grande no siempre significa que el modelo haya ignorado una señal clara del draft. A veces el score real aumenta por cosas que no podían conocerse antes de empezar la partida.

En varios errores extremos aparecen partidas donde la botlane empieza muy mal y acumula muchas muertes durante los primeros minutos. En este tipo de situaciones, los jugadores pueden frustrarse y dejar de jugar de forma ordenada o cooperativa. En el lenguaje habitual de la comunidad, esto se conoce como “trolear”: jugar mal de forma deliberada, desordenada o poco cooperativa, ignorando el desarrollo normal de la partida. Para el modelo, estas partidas son especialmente difíciles. El draft puede sugerir un support poco orientado al roaming, pero la timeline acaba mostrando mucha separación entre support y ADC porque la fase de líneas se ha desordenado por completo.

Un caso revisado corresponde a una botlane con un support que, por diseño, debería permanecer bastante unido a su ADC. El modelo predice un score bajo, coherente con esa expectativa. Sin embargo, el score observado es muy alto porque el ADC muere repetidamente antes del minuto 12 y la botlane deja de comportarse como una pareja estable. En ese caso, la etiqueta mide una separación real, pero esa separación no parece proceder de una intención estratégica de roaming derivada del draft.

Este análisis motivó la distinción entre partidas limpias y partidas caóticas. En la auditoría de errores, la mayoría de casos extremos estaban asociados a desarrollos tempranos anómalos. En partidas limpias, el HistGBT alcanza R² = 0.171. En partidas caóticas, baja a R² = 0.122. La diferencia refuerza la interpretación principal: una parte importante del error no se explica solo por limitaciones del modelo, sino por la distancia entre lo que el draft sugiere antes de empezar y lo que los jugadores acaban haciendo en la partida.

Un ejemplo de este tipo de caso se muestra en la Figura 8. La figura ayuda a entender por qué el modelo puede fallar aunque la predicción inicial tenga sentido: la partida acaba mostrando una separación alta entre support y ADC por un desarrollo anómalo que no estaba disponible en el draft.

*Figura 8*

**Fig. 8. Ejemplo de caso con error alto en la revisión cualitativa.** El score observado es alto por una separación real entre support y ADC, pero esa separación parece estar causada por el desarrollo anómalo de la partida y no por una intención de roaming prevista desde el draft.

### 5.6. Robustez de la etiqueta

El label variant sweep mostró que las variantes de la etiqueta producen resultados muy parecidos. Cambiar pesos, eliminar componentes o añadir algunas señales de eventos no modificó de forma sustancial el ranking ni el rendimiento de los modelos.

Este resultado debilita una posible explicación alternativa: que el R² limitado se deba únicamente a una mala definición concreta de la etiqueta. La etiqueta puede mejorarse, y sigue siendo una aproximación imperfecta, pero dentro de las variantes probadas no aparece una formulación que cambie la conclusión general.

Las variantes orientadas a eventos productivos tampoco sustituyen bien a la etiqueta principal. Miden un fenómeno interesante, pero más cercano al roaming que acaba produciendo eventos visibles que a la predisposición espacial desde draft. Una etiqueta que premia kills, asistencias, objetivos, placas o estructuras puede ser más intuitiva para estudiar impacto real, pero también introduce más dependencia de la ejecución concreta: si una rotación acaba en kill o no depende de coordinación, ventaja previa, respuesta rival y estado de la partida. Como el objetivo del TFG es trabajar con información pre-partida, se mantiene la etiqueta espacial como referencia principal.

La transformación quantile tampoco cambió la conclusión. Aplanar la distribución podía facilitar el entrenamiento al reducir la concentración de valores en la zona baja-media, pero los modelos entrenados sobre la escala quantile e invertidos a escala original no mejoraron a los modelos entrenados directamente sobre la escala original. Esto sugiere que el límite no se debe solo a una distribución incómoda del target, sino a la información disponible y al ruido observacional de la propia tarea.

### 5.7. Embeddings e hiperparámetros de la MLP

Los embeddings se probaron para comprobar si una representación numérica de campeones podía capturar relaciones que One-Hot no expresa directamente. La hipótesis era que campeones con estilos parecidos podrían quedar cerca en ese espacio aprendido y facilitar el aprendizaje.

En la práctica, las MLPs con embeddings no superaron a los modelos tabulares. La mejor variante neuronal queda alrededor de R² = 0.154, por debajo del HistGBT. Además, el análisis visual y de vecinos cercanos no mostró una separación limpia por arquetipos expertos de support.

Aun así, los embeddings no fueron completamente irrelevantes. La distancia entre campeones en el espacio aprendido muestra una relación positiva débil con la diferencia de roaming medio. Esto sugiere que el modelo aprende algo sobre la función de los campeones, pero esa información no es lo bastante fuerte como para mejorar el rendimiento final frente a modelos tabulares.

Además de probar embeddings, se ejecutó una búsqueda de hiperparámetros para la MLP con embeddings por rol e interacciones. El grid incluyó 108 configuraciones y modificó tamaño de capas, dropout, learning rate y weight decay. La mejor configuración apenas mejoró la referencia en validation, y en test quedó alrededor de R² = 0.155 y Spearman = 0.384, todavía por debajo del HistGBT. Este resultado refuerza la idea de que la principal limitación no está en encontrar una combinación concreta de hiperparámetros, sino en que la etiqueta y la información pre-partida imponen un límite bastante fuerte.

La conclusión es prudente: los embeddings contienen cierta información, pero en esta tarea concreta no justifican sustituir el enfoque tabular. Pueden ser útiles en fases futuras, especialmente si se amplía el modelo a más tareas o se incorporan más variables semánticas, pero no han sido el factor limitante principal en esta etapa.

### 5.8. Prototipo por terminal

Como salida aplicada del proyecto, se ha desarrollado un prototipo por terminal que permite introducir una composición de draft y obtener una lectura del perfil esperado de roaming del support. El prototipo carga el modelo entrenado, transforma los campeones, hechizos y lado al mismo formato usado durante el entrenamiento y devuelve un score continuo acompañado de una interpretación por bandas.

El prototipo puede usarse de forma interactiva, mediante argumentos de consola o en modo batch con ficheros CSV/JSON. Cuando el usuario no introduce hechizos de invocador, el sistema utiliza valores por defecto razonables por rol y muestra esa suposición en la salida. También puede generar una lectura comparativa entre el support aliado y el support enemigo, lo que encaja con el objetivo aplicado del TFG: no predecir exactamente la partida, sino ayudar a leer la tendencia de una composición antes de que empiece.

De cara a la entrega final, el trabajo pendiente del prototipo no es reconstruir la lógica principal, sino pulir la salida: hacer más clara la traducción del score a frases interpretables, revisar ejemplos de uso y asegurar que el prototipo se presenta como herramienta de análisis de tendencia, no como predictor exacto de ejecución.

---

## 6. Valoración de resultados

### 6.1. Interpretación del R² obtenido

El mejor modelo obtiene R² = 0.161 en test. Este valor debe interpretarse con cuidado. No significa que el modelo acierte el 16.1% de las partidas, sino que explica aproximadamente el 16.1% de la varianza observada del score respecto a predecir siempre la media.

Leído de forma aislada, puede parecer un resultado modesto. Sin embargo, el contexto cambia la interpretación. La baseline Champion Mean ya alcanza R² = 0.125 usando solo el campeón support, lo que confirma que el campeón elegido contiene una parte importante de la señal. El HistGBT mejora esa referencia hasta R² = 0.161, por lo que el resto del draft aporta información adicional, aunque limitada.

La comparación con la referencia por botlane+side también es relevante. Esta referencia alcanza R² = 0.173, y el mejor modelo queda cerca de ella. Esto sugiere que, con las variables actuales, el margen de mejora puede no estar principalmente en ajustar más la arquitectura. Una parte de la variabilidad del score parece depender de factores no observables antes de la partida: muertes tempranas, coordinación, decisiones individuales, recalls, estado de la línea y comportamiento de los jugadores.

Por tanto, la lectura más razonable no es que el modelo prediga con precisión el roaming de cada partida. La lectura correcta es más limitada, pero también más defendible: **el draft contiene una señal parcial y permite ordenar composiciones según su predisposición al roaming del support**.

La comparación con la referencia experta ayuda a sostener esta interpretación. Aunque el R² de predicción caso por caso sea limitado, el ranking agregado por campeón se parece mucho al ranking esperado al inicio del proyecto. Esto indica que el trabajo no ha encontrado una señal puramente accidental: la etiqueta recoge una estructura reconocible desde el conocimiento del juego, pero esa estructura se degrada cuando se intenta anticipar la ejecución concreta de una partida individual.

### 6.2. Qué aportan los modelos frente a las baselines

Las baselines permiten separar qué parte del resultado es trivial y qué parte requiere aprendizaje. Predecir siempre la media global no utiliza ninguna información del draft y marca el punto cero de comparación. Predecir la media histórica del campeón support ya mejora mucho, porque el campeón define en gran medida el estilo del support.

El modelo tabular mejora esa baseline porque incorpora interacciones adicionales: qué ADC acompaña al support, qué support juega el rival, qué campeones aparecen en otros roles y en qué lado del mapa juega el equipo. La mejora no es enorme, pero sí consistente. Esto indica que el draft aporta más información que el campeón support aislado, aunque no suficiente para explicar la mayor parte de la ejecución observada.

Las MLPs no mejoraron a los modelos tabulares. Esto no significa que las redes neuronales no sean útiles para este tipo de problema en general. Significa que, con las variables actuales, el volumen de información disponible y las arquitecturas probadas, los modelos basados en árboles han aprovechado mejor las variables categóricas del draft. Además, la búsqueda de hiperparámetros de la MLP no cambió esta conclusión: ajustar más fino la red apenas movió las métricas. Esto refuerza que la principal limitación no está en el optimizador o en una configuración concreta, sino en la etiqueta, la resolución temporal y el alcance de la información pre-partida.

### 6.3. Limitaciones principales

La primera limitación es la resolución temporal de la timeline. La API proporciona posiciones a intervalos de aproximadamente un minuto. En la ventana 5-12, esto deja pocos frames válidos por partida. Por tanto, un movimiento puntual, una muerte o un recall pueden tener bastante peso en el score.

La segunda limitación es de interpretación. La etiqueta mide separación observada, no intención pura. Un support puede aparecer lejos del ADC porque está roameando de forma planificada, pero también porque la botlane ha perdido el control, porque un jugador ha muerto o porque la partida se ha vuelto desordenada.

La tercera limitación es que la referencia experta por campeones procede del conocimiento del autor. Es útil para comprobar si el ranking general tiene sentido y, de hecho, la alineación obtenida es una señal positiva fuerte. Aun así, no sustituye una anotación externa independiente realizada por varios evaluadores.

La cuarta limitación es el tratamiento de partidas caóticas. El `chaos_flag` reduce el peso de partidas con muertes tempranas extremas o pocos frames útiles, pero no resuelve por completo el problema de fondo: la etiqueta sigue midiendo una separación observada, y en League of Legends esa separación puede aparecer por razones muy distintas a un roam planificado.

La quinta limitación es el alcance del dataset. Los datos proceden de un contexto concreto de servidor, cola y nivel de jugadores. Los patrones podrían cambiar en otros rangos, regiones o entornos competitivos.

La sexta limitación es el propio techo empírico. El ICC y las medias por grupo son referencias útiles, pero no límites teóricos. Dependen de las agrupaciones elegidas y de la variabilidad presente en el dataset.

---

## 7. Conclusiones provisionales

La primera conclusión es que el draft contiene señal predictiva para estimar la tendencia de roaming del support, pero esta señal es parcial. El mejor modelo alcanza R² = 0.161 y Spearman = 0.388 en test. Esto indica que el modelo no solo reduce error respecto a predecir la media, sino que también ordena los drafts con cierta coherencia según la tendencia esperada al roaming.

La segunda conclusión es que una parte importante de la señal procede del propio campeón support. La baseline basada en la media histórica del campeón ya alcanza R² = 0.125. El resto del draft mejora ese resultado, pero con un margen limitado. Esto encaja con el conocimiento del juego: el kit y la identidad del support condicionan mucho su estilo, aunque la pareja de botlane y el contexto del draft también aportan información.

La tercera conclusión es que la etiqueta, pese a sus limitaciones, tiene una comparación de dominio relevante. La correlación de Spearman de aproximadamente 0.82 frente a la referencia experta por campeón indica que el score agregado recupera bastante bien la intuición inicial sobre qué supports tienden a roamear más y cuáles tienden a permanecer más ligados al ADC.

La cuarta conclusión es que los modelos tabulares han funcionado mejor que las redes neuronales probadas. HistGradientBoosting aprovecha bien las variables categóricas y sus interacciones, mientras que las MLPs con One-Hot o embeddings no han aportado una mejora clara. Los embeddings contienen cierta señal, pero no han aprendido una estructura suficientemente útil como para superar al enfoque tabular. El grid de hiperparámetros de la MLP tampoco cambia esta lectura.

La quinta conclusión es que la etiqueta principal parece razonablemente estable. Las variantes probadas no cambian sustancialmente la señal ni mejoran de forma clara el rendimiento. La transformación quantile fue útil para comprobar si aplanar la distribución facilitaba el entrenamiento, pero no mejoró la comparación principal. Las versiones más orientadas a eventos productivos miden un fenómeno interesante, pero se alejan del objetivo actual porque dependen más de la ejecución de la partida que de la predisposición desde draft.

La sexta conclusión es que la replanificación del alcance ha sido necesaria. En lugar de extender el sistema a jungla y equipo sin disponer de una evaluación equivalente, se ha preferido cerrar con más rigor la tarea de support. Esta decisión permite presentar un resultado más acotado, pero también más defendible: una tarea concreta, con etiqueta revisada, comparación experta, baselines, modelos comparados, techo empírico, análisis de errores, filtrado de partidas caóticas, prototipo aplicado y limitaciones explícitas.

En conjunto, el proyecto ha pasado de una idea inicial amplia —estudiar varios patrones tempranos desde el draft— a un caso de estudio más concreto y mejor evaluado. El objetivo general se mantiene, pero el desarrollo final se centra en demostrar qué parte del comportamiento del support puede anticiparse antes de que empiece la partida y qué parte queda fuera del alcance de la información pre-partida.

---

## 8. REFERENCIAS

[1] Riot Games, “Riot Developer Portal,” Riot Games. [Online]. Available: https://developer.riotgames.com/

[2] Riot Games, “Data Dragon,” Riot Developer Portal. [Online]. Available: https://developer.riotgames.com/docs/lol

[3] J.-A. Hitar-Garcia, L. Moran-Fernandez, and V. Bolon-Canedo, “Machine Learning Methods for Predicting League of Legends Game Outcome,” *IEEE Transactions on Games*, vol. 15, no. 2, pp. 171–181, 2023.

[4] H. Lee, D. Hwang, H. Kim, B. Lee, and J. Choo, “DraftRec: Personalized Draft Recommendation for Winning in Multi-Player Online Battle Arena Games,” in *Proc. ACM Web Conf. 2022 (WWW ’22)*, 2022, pp. 3428–3439.

[5] A. M. Rama, V. Rodriguez-Fernandez, and D. Camacho, “Finding Behavioural Patterns Among League of Legends Players Through Hidden Markov Models,” in *Applications of Evolutionary Computation*, Lecture Notes in Computer Science, vol. 12104, 2020, pp. 419–430.

[6] G. Wallner, L. Wang, and C. Dormann, “Visualizing the Spatio-Temporal Evolution of Gameplay using Storyline Visualization: A Study with League of Legends,” *Proc. ACM Hum.-Comput. Interact.*, vol. 7, CHI PLAY, pp. 1002–1024, 2023.

[7] Y. Chen, J. Wu, Y. Wu, and D. Liu, “T-Foresight: Interpret moving strategies based on context-aware trajectory prediction,” *Visual Informatics*, vol. 9, no. 3, Art. no. 100261, 2025.

[8] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, “LightGBM: A Highly Efficient Gradient Boosting Decision Tree,” in *Advances in Neural Information Processing Systems*, vol. 30, 2017.

[9] S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” in *Advances in Neural Information Processing Systems*, vol. 30, 2017.

[10] K. O. McGraw and S. P. Wong, “Forming Inferences About Some Intraclass Correlation Coefficients,” *Psychological Methods*, vol. 1, no. 1, pp. 30–46, 1996.

[11] F. Pedregosa *et al.*, “Scikit-learn: Machine Learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[12] A. Paszke *et al.*, “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” in *Advances in Neural Information Processing Systems*, vol. 32, 2019.

