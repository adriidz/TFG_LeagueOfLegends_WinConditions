# Preparacion de reunion con el tutor

Fecha de preparacion: 31/05/2026  
Base: Informe de Progreso I e Informe de Progreso II  
Objetivo: explicar al tutor que se ha hecho, por que se tomaron las decisiones principales, que resultados hay y como se va a cerrar el TFG.

## 1. Mensaje central de la reunion

El TFG ha pasado de una idea inicial amplia, basada en predecir varios patrones de early-game, a un caso de estudio mas acotado y mejor evaluado: estimar la tendencia de roaming del support a partir del draft.

La conclusion principal no es que el modelo prediga exactamente como se jugara cada partida. La conclusion defendible es que el draft contiene una señal real pero parcial: permite ordenar composiciones por predisposicion al roaming del support, pero una parte importante del comportamiento observado depende de la ejecucion dentro de la partida.

Frase para abrir la reunion:

> "Queria ensenarle como ha evolucionado el trabajo desde el Informe I. Al principio tenia una primera MLP y una etiqueta continua prometedora. En el Informe II he cerrado mejor la tarea de support: he anadido baselines, un modelo tabular fuerte, comparacion experta, techo empirico, analisis de errores y una version funcional del prototipo. La lectura final es que el draft si aporta senal, pero el limite principal parece estar en la informacion pre-partida, no solo en la arquitectura del modelo."

## 2. Agenda sugerida

Duracion estimada: 15-20 minutos.

1. Recordatorio del objetivo del TFG: 2 minutos.
2. Que se descubrio en el Informe de Progreso I: 4 minutos.
3. Por que se cambio el enfoque y el alcance: 4 minutos.
4. Resultados principales del Informe de Progreso II: 6 minutos.
5. Conclusiones, limitaciones y cierre del proyecto: 3 minutos.
6. Preguntas concretas al tutor: 2 minutos.

## 3. Resumen para ensenar en la reunion

### 3.1 Objetivo original y objetivo actual

Objetivo original:

- Inferir patrones tempranos de juego en League of Legends usando informacion disponible antes de empezar la partida.
- La entrada principal es el draft: campeones, roles, lado del mapa y variables pre-partida.
- Inicialmente se planteaban tres salidas: jungler, roaming del support y tendencia espacial de equipo.

Objetivo actual:

- Centrarse en una primera tarea bien evaluada: `support_roam_score`.
- El score mide en [0, 1] hasta que punto el support abandona el contexto de botlane o se separa del ADC entre los minutos 5 y 12.
- La timeline se usa solo para construir la etiqueta observada; no entra como input del modelo. Esto evita data leakage y mantiene el planteamiento pre-partida.

### 3.2 Que aporto el Informe de Progreso I

El Informe I sirvio para detectar que la dificultad principal no era solo entrenar un modelo, sino definir bien que se queria predecir.

Resultados y hallazgos clave:

| Punto | Resultado |
| --- | --- |
| Dataset inicial | 337.130 observaciones `(match_id, team_id)` en `draft_features` |
| Frame-state | 9.823.624 filas derivadas de timelines |
| Etiqueta support | 337.094 observaciones etiquetadas, sin nulos, rango [0, 1] |
| Distribucion etiqueta | Media 0.2828, mediana 0.2636, desviacion 0.1722 |
| Comparacion experta | Pearson 0.795, Spearman 0.825 sobre 47 campeones |
| Primera MLP | R2 0.13068, Pearson 0.3633, Spearman 0.3568 |
| Mejora frente a media | Aproximadamente 13% en MSE y 8% en MAE |

Lectura del Informe I:

- La etiqueta no parecia ruido: al agregar por campeon, el ranking se parecia mucho a la intuicion experta.
- La MLP aprendia senal, pero con predicciones comprimidas hacia la media.
- El enfoque de clasificacion discreta generaba problemas por la clase ambigua y por umbrales artificiales.
- Por eso se paso a regresion continua.

### 3.3 Decision 1: pasar de clasificacion a regresion continua

Por que se tomo:

- Los scores nacen como valores continuos calculados desde la timeline.
- Discretizar en clases "bajo / medio / alto" introducia umbrales artificiales.
- La clase intermedia era dificil de interpretar y de aprender.
- En clasificacion, fallar por poco y fallar por mucho puede penalizarse igual si se cruza un umbral.

Como explicarlo:

> "El comportamiento del support no aparece en grupos perfectamente separados. Tiene una escala gradual. Por eso tiene mas sentido predecir un score continuo que forzar clases. Asi la zona intermedia deja de ser una clase problematica y pasa a ser parte natural de la escala."

### 3.4 Decision 2: centrar el TFG en support y posponer jungla/equipo

Por que se tomo:

- En el Informe I ya se vio que cada tarea tenia ritmos y problemas distintos.
- La etiqueta de support necesitaba revisar geometria, formula, baselines, robustez y errores.
- Reintroducir jungla, equipo y multi-output en el tiempo disponible habria dado resultados mas superficiales.

Como defenderlo:

> "No he abandonado el objetivo general, lo he acotado para cerrarlo con mas rigor. En lugar de presentar tres etiquetas poco evaluadas, presento una tarea concreta con etiqueta, baselines, modelos, comparacion experta, techo empirico, analisis de errores, limitaciones y prototipo."

### 3.5 Decision 3: separar estrictamente draft y timeline

Por que se tomo:

- El objetivo aplicado es estimar tendencias antes de empezar la partida.
- El draft es informacion pre-partida y puede usarse como input.
- La timeline describe lo ocurrido durante la partida y solo debe usarse para construir la etiqueta observada.
- Usar timeline como input daria una ventaja artificial y cambiaria la pregunta experimental.

Como explicarlo:

> "El modelo no intenta ver el futuro ni reconstruir la partida minuto a minuto. Aprende desde el draft y se evalua contra una etiqueta calculada despues. Asi puedo medir hasta que punto la composicion anticipa un patron temprano observable."

### 3.6 Decision 4: revisar la geometria del mapa

Por que se tomo:

- La etiqueta depende de clasificar si una posicion pertenece al contexto de botlane o a zonas compatibles con roaming.
- Una geometria automatica basada en densidad podia no coincidir bien con la interpretacion real del mapa.
- Se sustituyo por una geometria manual mas interpretable.

Lectura:

- Botlane no se reduce a la linea inferior estricta.
- Se incluyen zonas cercanas como rio inferior y zona de dragon.
- Midlane, jungla superior o zonas alejadas del ADC cuentan mas como senales de roaming.

### 3.7 Formula final de la etiqueta

La etiqueta final combina tres componentes:

```text
score_raw = 0.45 * outside_ratio
          + 0.35 * far_ratio
          + 0.20 * xp_gap

support_roam_score = score_raw ^ 0.75
```

Interpretacion:

- `outside_ratio`: proporcion de frames en los que el support esta fuera del contexto de botlane.
- `far_ratio`: proporcion de frames en los que el support esta lejos del ADC.
- `xp_gap`: diferencia relativa de experiencia entre support y ADC, con peso menor porque depende mas del desarrollo de la partida.
- La potencia `0.75` ajusta la escala sin cambiar el orden de los ejemplos.

Idea importante:

> "La etiqueta mide separacion observada, no intencion pura. Puede capturar roams reales, pero tambien partidas donde la botlane se desordena. Por eso despues se hizo auditoria cualitativa y filtrado de caos."

## 4. Resultados principales del Informe de Progreso II

### 4.1 Dataset final

| Elemento | Valor |
| --- | --- |
| Observaciones finales | 383.247 filas partida-equipo |
| Partidas aproximadas | 191.000 |
| Servidor / cola / nivel | EUW, Ranked Solo/Duo, Master-Grandmaster-Challenger |
| Parches | 16.2 a 16.8 |
| Split | Aproximadamente 268k train, 57k validation, 57k test |
| Split por grupos | Por `match_id`, para evitar leakage |

Variables de entrada:

- 10 campeones aliados/enemigos por rol.
- 20 hechizos de invocador.
- Lado del mapa.
- Solo informacion disponible antes de la partida.

### 4.2 Comparacion principal de modelos

| Modelo / referencia | R2 | Spearman | MAE |
| --- | ---: | ---: | ---: |
| ICC / referencia empirica botlane+lado | 0.173 | - | - |
| HistGBT + Pair Target Encoding | 0.161 | 0.388 | 0.141 |
| HistGBT + arquetipos | 0.161 | 0.388 | 0.141 |
| HistGBT base | 0.160 | 0.387 | 0.141 |
| MLP OneHot | 0.155 | 0.381 | 0.141 |
| MLP por rol + interacciones | 0.154 | 0.381 | 0.141 |
| MLP embeddings | 0.150 | 0.376 | 0.142 |
| Media por campeon support | 0.125 | 0.336 | 0.144 |
| Media global | 0.000 | - | 0.155 |

Lectura:

- Todos los modelos utiles mejoran la media global.
- La media por campeon support ya explica una parte importante de la senal.
- El modelo tabular mejora la baseline de campeon y queda cerca de la referencia empirica.
- Las MLPs no superan al HistGBT.

Frase para el tutor:

> "El R2 puede parecer modesto si se mira aislado, pero cambia la interpretacion al compararlo con baselines y con el techo empirico. La media por campeon ya llega a 0.125; el mejor modelo sube a 0.161; y la referencia por botlane+lado esta en 0.173. Eso sugiere que gran parte de la senal pre-partida capturable ya esta siendo aprovechada."

### 4.3 Metricas de tolerancia

| Modelo | Predicciones dentro de ±0.10 | Predicciones dentro de ±0.20 |
| --- | ---: | ---: |
| HistGBT + Pair Target | 41.8% | 74.2% |
| MLP OneHot | 41.9% | 74.1% |
| MLP por rol + interacciones | 41.8% | 74.1% |
| Media por campeon | 41.1% | 72.9% |
| Media global | 37.8% | 68.8% |

Lectura:

- Para una herramienta interpretativa, no solo importa el valor exacto.
- Tambien importa si el modelo situa el draft en una zona razonable del score.
- Alrededor de tres cuartas partes de las predicciones caen dentro de ±0.20.

### 4.4 Techo empirico mediante agrupaciones de draft

| Agrupacion | ICC | R2 con media por grupo |
| --- | ---: | ---: |
| Support | 0.121 | 0.121 |
| Botlane `(botlaner + support)` | 0.139 | 0.161 |
| Botlane + lado | 0.139 | 0.173 |
| Arquetipo de support | 0.084 | 0.081 |

Lectura:

- El campeon support importa mucho.
- La pareja de botlane y el lado aportan algo mas.
- Incluso dentro de drafts parecidos sigue habiendo mucha variabilidad.
- Esa variabilidad probablemente depende de ejecucion: muertes tempranas, coordinacion, estado de linea, decisiones individuales, pathing de jungla, recalls, vision, etc.

Importante:

> "No presentaria el ICC como un techo teorico absoluto, sino como una referencia practica. Sirve para mostrar que el mejor modelo esta cerca de la senal repetible observable con las variables actuales."

### 4.5 Importancia de variables

| Grupo de features | Importancia total aproximada |
| --- | ---: |
| Campeones aliados | 0.255 |
| Campeones enemigos | 0.033 |
| Hechizos | 0.001 |
| Lado del mapa | ~0 |

Lectura:

- La variable mas importante es el campeon support aliado.
- Despues aparecen el botlaner aliado y el support enemigo.
- Esto encaja con el dominio: el estilo del support depende mucho de su kit, de su ADC y del matchup de botlane.

### 4.6 Auditoria cualitativa de errores

Hallazgo:

- Muchos errores extremos aparecen en partidas caoticas.
- Ejemplo: una Yuumi deberia permanecer cerca del ADC, el modelo predice bajo, pero el score observado sale alto porque el ADC muere repetidamente y la botlane deja de comportarse como pareja estable.

Resultado cuantitativo:

- En partidas limpias, HistGBT alcanza aproximadamente R2 = 0.171.
- En partidas caoticas, baja a aproximadamente R2 = 0.122.

Lectura:

> "El modelo puede fallar aunque la prediccion tenga sentido desde el draft, porque la timeline acaba reflejando una separacion causada por un desarrollo anomalo de la partida, no por una intencion estrategica de roaming."

### 4.7 Robustez de la etiqueta

Se probaron variantes:

- Cambios de pesos.
- Ablaciones de componentes.
- Transformacion quantile zero-preserved.
- Variantes mas orientadas a eventos: kills, asistencias, objetivos, estructuras.

Conclusiones:

- Las variantes de formula quedaron muy correlacionadas con la etiqueta principal, con correlaciones superiores a 0.99.
- La transformacion quantile no mejoro la evaluacion final en escala original.
- Las variantes por eventos miden algo interesante, pero cambian la pregunta: se acercan al roaming que produce eventos visibles, no a la predisposicion espacial desde draft.

### 4.8 Embeddings e hiperparametros de la MLP

Resultados:

- Las MLPs con embeddings no superan a los modelos tabulares.
- La mejor variante neuronal queda alrededor de R2 = 0.154-0.155.
- El grid de hiperparametros de 108 configuraciones apenas mejora la validacion.

Lectura:

> "La limitacion principal no parece ser encontrar un learning rate o dropout concreto. Con las variables actuales, los modelos tabulares aprovechan mejor la senal disponible."

## 5. Conclusiones que conviene transmitir

1. El draft contiene senal predictiva real, pero parcial.
2. El campeon support concentra una parte importante de esa senal.
3. La etiqueta tiene sentido a nivel agregado: Spearman ≈ 0.825 con referencia experta por campeon.
4. El mejor modelo, HistGBT, mejora las baselines y queda cerca de la referencia empirica.
5. Las redes neuronales no superan al enfoque tabular en esta tarea concreta.
6. La etiqueta es razonablemente robusta frente a variantes de formula, aunque no mide intencion pura.
7. Los errores grandes se explican a menudo por partidas caoticas o ejecucion no predecible desde el draft.
8. La decision de acotar el TFG a support esta justificada por rigor, tiempo y claridad experimental.

Conclusion final para decirla tal cual:

> "El resultado final no es un predictor perfecto de comportamiento individual. Es una evaluacion empirica de cuanta informacion pre-partida hay en el draft para anticipar una tendencia temprana concreta. Esa informacion existe, se puede modelar y tiene sentido de dominio, pero tiene un limite claro porque el roaming observado tambien depende de como se desarrolla cada partida."

## 6. Que ensenar en pantalla

Orden recomendado:

1. Objetivo del proyecto y cambio de enfoque.
2. Figura o tabla de la etiqueta `support_roam_score`.
3. Resultado del Informe I: primera MLP y comparacion experta.
4. Tabla final de comparacion de modelos del Informe II.
5. Referencia empirica / ICC por agrupaciones.
6. Importancia de variables.
7. Caso cualitativo de error alto por partida caotica.
8. Prototipo CLI, si quieres cerrar con algo aplicado.

Material disponible en el repo:

- `final/docs/Informe de Progreso II.docx`
- `final/docs/Informe de Progreso II.pdf`
- `final/docs/figures/report_style/`
- `final/entrevista/`

Carpeta especialmente util:

- `final/entrevista/00_summary/`: resumen general y tabla comparativa.
- `final/entrevista/01_model_comparison/`: comparacion de modelos.
- `final/entrevista/03_ceiling/`: techo empirico.
- `final/entrevista/06_qualitative/`: ejemplos cualitativos.
- `final/entrevista/08_feature_importance/`: importancia de variables.

## 7. Guion de 10 diapositivas

### Diapositiva 1 - Titulo

Titulo: "Inferencia de roaming del support a partir del draft"

Mensaje:

- El TFG estudia si la informacion pre-partida permite anticipar patrones tempranos.
- El caso final se centra en support roaming.

### Diapositiva 2 - Pregunta de investigacion

Mensaje:

- Input: draft antes de empezar.
- Target: `support_roam_score` calculado desde la timeline.
- Pregunta: que parte del roaming temprano se puede anticipar antes de la partida.

### Diapositiva 3 - Cambio respecto a la propuesta inicial

Mensaje:

- Inicialmente: clasificacion multi-output.
- Problema: etiquetas continuas discretizadas, clase ambigua, perdida de informacion.
- Decision: regresion continua y tarea support-only.

### Diapositiva 4 - Etiqueta final

Mostrar:

- Formula.
- Geometria del mapa.
- Distribucion del score.

Mensaje:

- La etiqueta mide separacion observada, no intencion perfecta.
- Es interpretable y robusta frente a variantes.

### Diapositiva 5 - Resultado del Informe I

Mostrar:

- Primera MLP: R2 0.13068, Spearman 0.3568.
- Comparacion experta: Pearson 0.795, Spearman 0.825.

Mensaje:

- La etiqueta tenia sentido a nivel de dominio.
- La MLP aprendia algo, pero faltaba contexto para interpretar el R2.

### Diapositiva 6 - Baselines y modelos finales

Mostrar tabla:

- Media global: R2 0.000.
- Media por campeon: R2 0.125.
- HistGBT: R2 0.160-0.161.
- MLPs: R2 0.150-0.155.

Mensaje:

- El draft aporta senal.
- El campeon support explica mucho.
- El modelo tabular aprovecha algo mas del draft.

### Diapositiva 7 - Techo empirico

Mostrar:

- Botlane + lado: R2 0.173.
- Mejor modelo: R2 0.161.

Mensaje:

- El modelo queda cerca de una referencia practica.
- El cuello de botella parece estar en la informacion disponible antes de la partida.

### Diapositiva 8 - Explicabilidad

Mostrar:

- Importancia por grupos.

Mensaje:

- La senal principal viene de campeones aliados, sobre todo el support.
- Esto coincide con conocimiento del juego.

### Diapositiva 9 - Limitaciones

Mensaje:

- Timeline minutal: pocos frames entre minuto 5 y 12.
- Etiqueta mide separacion, no intencion.
- Partidas caoticas aumentan el error.
- Dataset restringido a una region, cola y nivel.

### Diapositiva 10 - Cierre y siguientes pasos

Mensaje:

- Cerrar memoria con support como caso principal.
- Presentar jungla/equipo como extensiones futuras.
- Pulir CLI y salida interpretable.
- Pedir feedback sobre enfoque, narrativa y alcance.

## 8. Preguntas para hacer al tutor

1. "Le parece bien cerrar el TFG centrado en support-only, dejando jungla/equipo como trabajo futuro?"
2. "Como prefiere que presente el R2: como resultado modesto o como resultado contextualizado por baselines y techo empirico?"
3. "Le parece adecuada la distincion entre predisposicion del draft y ejecucion observada?"
4. "Quiere que la memoria enfatice mas la construccion de la etiqueta o la comparacion de modelos?"
5. "Para la defensa, enseno el prototipo CLI como cierre aplicado o lo dejo como anexo?"

## 9. Riesgos de comunicacion y como evitarlos

Evitar decir:

- "El modelo predice el roaming del support."
- "El ICC es el techo teorico maximo."
- "La etiqueta mide la intencion del jugador."
- "Las redes neuronales no sirven."

Mejor decir:

- "El modelo estima una tendencia de roaming esperada desde el draft."
- "El ICC es una referencia empirica practica para contextualizar el resultado."
- "La etiqueta mide separacion observada compatible con roaming."
- "En esta tarea concreta, con estas variables, los modelos tabulares han funcionado mejor que las MLPs probadas."

## 10. Cierre recomendado de la reunion

> "Mi propuesta para cerrar el TFG es presentar support roaming como caso de estudio principal. La memoria explicaria la evolucion desde clasificacion a regresion, la construccion de la etiqueta, la validacion experta, las baselines, los modelos comparados, el techo empirico, la auditoria de errores y el prototipo. Jungla, equipo y modelos secuenciales quedarian como lineas futuras, porque abrirlos ahora reduciria el rigor del resultado principal."

