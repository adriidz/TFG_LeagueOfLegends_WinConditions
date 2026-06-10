# Análisis de la Relación Entre la Configuración Inicial de Agentes y su Movilidad Temprana en Entornos Multiagente Espaciales

**Autor:** Adrián Díaz García  
**Grado:** Ingeniería de Datos  
**Universidad:** Universitat Autònoma de Barcelona (UAB)  
**Fecha:** Junio de 2026  

---

## Resumen Ejecutivo
Este TFG estudia hasta qué punto la configuración inicial de una partida de League of Legends permite anticipar la movilidad temprana del agente de soporte, usando el videojuego como caso de estudio de un sistema multiagente espacial. Para ello se construyó un pipeline de datos desde la API de Riot Games, se definió una etiqueta continua (`support_roam_score`) a partir de la trayectoria temporal del soporte entre los minutos 5 y 12, y se compararon modelos tabulares y neuronales bajo un protocolo experimental común.

El conjunto experimental final contiene **383.104 observaciones** de tipo `(match_id, team_id)` tras filtros de calidad, con particiones por partida para evitar fuga de información. El mejor modelo principal, un **HistGradientBoostingRegressor**, alcanza **R² = 0.1595 ± 0.0004**, **Spearman = 0.3877 ± 0.0004** y **MAE = 0.1408** en test. La media histórica del campeón de soporte ya explica una parte relevante de la señal (**R² = 0.1243**), pero el draft completo mejora esa referencia. La conclusión principal es que la configuración prepartida contiene señal real pero limitada: permite estimar predisposición, no determinar la ejecución concreta de la partida.

---

## Glosario y Correspondencia de Conceptos
*Para facilitar la lectura al tribunal y evitar el uso innecesario de jerga específica del videojuego League of Legends, a lo largo de este documento usamos la siguiente correspondencia de términos:*
*   **Agente (o Agente Funcional):** Personaje del juego / Campeón.
*   **Configuración Inicial (o Prepartida):** Draft / Selección de personajes.
*   **Entorno Espacial (o Escenario):** Mapa del juego (Summoner's Rift).
*   **Agente de Soporte (o de Apoyo):** Rol de Support (posicionado en el rol de *utility*).
*   **Compañero Principal de Zona (o Tirador):** Rol de ADC (tirador posicionado en el carril inferior).
*   **Zona de Asignación Inicial (o Zona Inferior):** Carril inferior o *botlane*.
*   **Movilidad Temprana (o Desplazamiento):** Comportamiento de *roaming* durante los primeros minutos de juego.
*   **Comportamiento No Cooperativo (o Anómalo):** El término informal del dominio *"trollear"* (muertes consecutivas o inactividad del tirador).
*   **Ejecución del Sistema:** Transcurso de la partida real in-game (datos del *timeline*).

---

## Capítulo 1: Introducción y Objetivos

### 1.1. Contexto y Motivación
Los sistemas multiagente (MAS) cooperativos estudian cómo se coordinan varios agentes autónomos para lograr objetivos comunes. Un entorno excelente para investigar esto son los videojuegos por equipos de estilo MOBA (*Multiplayer Online Battle Arena*). En este escenario competitivo de 5 contra 5, cada partida se divide en dos fases separadas:
1.  **Fase de Configuración Inicial (Prepartida):** Antes de jugar, cada equipo elige una combinación de 5 agentes a partir de un catálogo de más de 170 personajes disponibles. Esta selección se conoce como *draft*.
2.  **Fase de Ejecución (Partida):** Los agentes se despliegan en el escenario y actúan de forma autónoma bajo el control de los usuarios humanos.

En este trabajo nos apoyamos en una separación estricta entre estas dos fases. Queremos investigar si las decisiones tomadas en la configuración inicial permiten anticipar un patrón de comportamiento espacial muy concreto: la **movilidad temprana del agente de soporte**.

El escenario espacial (minimapa) está estructurado en tres carriles o zonas de responsabilidad (superior, medio e inferior) interconectados por ríos y zonas boscosas (jungla). 

```
[Figura 1: Estructura del escenario espacial (minimapa) de juego. Se muestran las tres zonas principales (carril superior, medio e inferior), las bases de inicio y las regiones poligonales de la zona inferior (botlane) y del río trazadas manualmente para clasificar las posiciones.]
```

Como se observa en la **Figura 1**, la zona inferior de asignación inicial está pensada para albergar a dos agentes aliados: el agente de soporte (orientado a tareas de protección) y su compañero principal (el tirador). Sin embargo, durante los primeros minutos de juego, el agente de soporte tiene libertad táctica para abandonar esta zona y realizar desplazamientos por el resto del mapa (comportamiento llamado *roaming* temprano) para ayudar a otros compañeros de equipo o capturar objetivos.

Este comportamiento de movilidad no se registra directamente en las estadísticas de las partidas; debemos construirlo a partir de las coordenadas del recorrido temporal de los agentes. Hemos elegido este juego como caso de estudio porque cuenta con una API pública masiva y porque nos permite formular una pregunta de predicción limpia: usar datos de configuración inicial como entrada y datos de ejecución temporal como salida.

---

### 1.2. Pregunta de Investigación
El núcleo de este trabajo se resume en la siguiente pregunta:

> *¿Hasta qué punto la configuración de agentes seleccionados antes del inicio de la partida permite anticipar la movilidad temprana del agente de soporte fuera de su zona asignada?*

El objetivo no es construir un predictor infalible que adivine el movimiento exacto del jugador en cada segundo de la partida. Lo que buscamos es **cuantificar la señal de tendencia que contiene el draft**. Encontrar que la selección inicial explica un porcentaje determinado de la varianza del comportamiento es ya una respuesta científica válida para resolver esta pregunta.

---

### 1.3. Objetivos Específicos
Para resolver la pregunta de investigación, planteamos los siguientes objetivos:
*   **O1. Construcción y Validación del Indicador de Movilidad:** Diseñar una puntuación numérica gradual entre $0$ y $1$ que mida la movilidad temprana del soporte observando sus posiciones in-game, y validar que este indicador coincide con la intuición de los expertos del dominio.
*   **O2. Comparación de Modelos bajo un Protocolo Común:** Evaluar modelos basados en árboles de decisión y redes neuronales densas con representaciones categóricas, asegurando que todos compiten en igualdad de condiciones de entrada.
*   **O3. Estimación de Referencias Empíricas:** Calcular referencias predictivas simples basadas en agrupaciones históricas para contextualizar el rendimiento absoluto de los modelos de Machine Learning.
*   **O4. Desarrollo de un Prototipo CLI Aplicado:** Diseñar una herramienta de consola que tome un draft manual y devuelva una lectura cualitativa y cuantitativa de la tendencia de roaming esperada antes de que empiece la partida.

---

## Capítulo 2: Trabajo Relacionado (Estado del Arte)

La literatura científica en videojuegos MOBA se ha enfocado tradicionalmente en predecir el resultado final del juego o en recomendar personajes óptimos durante la selección. Por ejemplo, Hitar-Garcia et al. (2023) evalúan modelos para predecir la victoria o derrota a partir de la composición del draft. Sus resultados muestran una precisión máxima cercana al 58% (donde la línea base aleatoria es el 50%). Esto demuestra lo difícil que es extraer señal predictiva de la configuración inicial, ya que el resultado final depende de innumerables factores impredecibles durante el juego. En cuanto a sistemas de recomendación, *DraftRec* (Lee et al., 2022) propone redes neuronales para sugerir personajes durante el draft que maximicen la probabilidad de victoria estimada, sin predecir comportamientos espaciales específicos.

Por otra parte, existen trabajos descriptivos enfocados en analizar el comportamiento espacial de los jugadores una vez iniciada la partida. Rama et al. (2020) aplican Modelos de Markov Ocultos (HMM) para identificar patrones de comportamiento táctico de los jugadores a partir de logs in-game, pero con una perspectiva estrictamente a posteriori (describir la partida una vez finalizada, no anticiparla). Wallner et al. (2023) proponen técnicas de visualización espacio-temporal para resumir las trayectorias de los jugadores de forma analítica. Finalmente, Chen et al. (2025) introducen modelos basados en redes recurrentes para predecir trayectorias de agentes in-game, pero utilizando como entrada las posiciones de los minutos anteriores de la misma partida.

Este TFG se sitúa en la intersección de estas áreas. A diferencia de predecir la victoria global (Hitar-Garcia) o de predecir trayectorias usando datos del propio juego en tiempo real (Chen), nuestro trabajo busca **anticipar una tendencia de comportamiento espacial intermedio utilizando exclusivamente la información estática del draft**.

---

## Capítulo 3: Metodología

### 3.1. Datos e Infraestructura ELT (Extracción, Carga y Transformación)
A diferencia de otros trabajos que utilizan datasets públicos ya preparados, en este proyecto **diseñamos e implementamos una infraestructura ELT a medida** para construir el dataset desde cero. 

```
[Figura 2: Arquitectura del pipeline de datos ELT. Muestra la extracción concurrente desde la API de Riot Games, la ingesta en JSON, el filtrado de calidad Master+, la clasificación espacial minutal de las coordenadas en base a polígonos, y la consolidación del dataset final en formato Parquet.]
```

Como ilustra la **Figura 2**, el flujo de trabajo consta de las siguientes etapas:
1.  **Extracción e Ingesta:** Desarrollamos un script de descarga concurrente que realiza peticiones a la API oficial de Riot Games [1], [2]. Descargamos dos tipos de ficheros por partida: los metadatos del draft (fichero de partida) y el historial de coordenadas minuto a minuto (fichero de *timeline*).
2.  **Filtrado de Calidad:** Restringimos la muestra a partidas clasificatorias individuales del servidor de Europa Occidental (EUW), de nivel alto (rango *Master* en adelante) de los parches 16.2 a 16.8 del juego. Esto nos asegura que el comportamiento observado responde a decisiones tácticas coordinadas y no a errores erráticos de principiantes. La muestra consolidada inicial cuenta con **383.247 observaciones** de tipo `(match_id, team_id)`; tras los filtros finales de calidad usados en el entrenamiento quedan **383.104 observaciones**.
3.  **Procesamiento Espacial y Consolidación:** Programamos un módulo geométrico que toma las coordenadas de posición de cada minuto del soporte y del tirador, comprueba a qué zona del mapa pertenecen (según los polígonos definidos manualmente) y consolida las variables agregadas en archivos en formato Parquet para su lectura eficiente en Python.

---

### 3.2. Construcción del Indicador de Movilidad (`support_roam_score`)
Para medir el roaming del soporte, construimos una puntuación continua entre $0$ y $1$ analizando sus desplazamientos durante la **ventana de los minutos 5 a 12 de la partida**.

La fórmula del indicador combina tres componentes espaciales calculados a lo largo de estos 8 minutos:
1.  **`outside_ratio` (Peso: 0.45):** Fracción del tiempo en la que el soporte se encuentra fuera de la zona inferior (botlane) delimitada por los polígonos manuales.
2.  **`far_ratio` (Peso: 0.35):** Fracción del tiempo en la que la distancia en línea recta entre el soporte y su tirador es superior a 3.000 unidades físicas del mapa.
3.  **`xp_gap` (Peso: 0.20):** Diferencia relativa de experiencia acumulada. Si el soporte permanece en línea con el tirador compartiendo los recursos del carril inferior, la diferencia es baja; si se desplaza a otras áreas del mapa, el tirador gana experiencia en solitario y la diferencia aumenta.

La asignación de pesos (0.45 para la presencia fuera de carril, 0.35 para la distancia física al tirador y 0.20 para el diferencial de experiencia) se planteó como una propuesta heurística inicial basada en el conocimiento del dominio. En el análisis exploratorio de datos (EDA) previo, se observó que la variable `outside_ratio` presentaba la mayor varianza y representaba la métrica más directa del comportamiento de abandono del carril, seguida de la distancia física y la experiencia. Para comprobar la sensibilidad de esta elección, evaluamos 15 combinaciones lineales alternativas de pesos; todas las etiquetas resultantes mantuvieron una correlación lineal superior a $0.99$ entre sí, lo que confirma que el indicador es robusto y que la señal espacial subyacente no depende críticamente de la parametrización exacta de la fórmula.

El score final aplica una transformación gamma de 0.75 para suavizar la distribución y mitigar la asimetría de la cola derecha (característica de distribuciones con sesgo positivo donde la mayoría de partidas tienen roaming bajo y unas pocas registran roaming extremo):

$$\text{roam\_score} = \left( 0.45 \cdot \text{outside\_ratio} + 0.35 \cdot \text{far\_ratio} + 0.20 \cdot \text{xp\_gap} \right)^{0.75}$$

Esta transformación no altera el orden de las observaciones, porque es monótona, pero sí estabiliza la varianza de los residuos en el entrenamiento de los algoritmos (optimizando la convergencia de MSE/Huber). Se comprobó la sensibilidad con $\gamma = 0.5$, $\gamma = 0.75$ y $\gamma = 1.0$: la media de la etiqueta cambia de **0.5177** a **0.3916** y **0.3023**, respectivamente, pero la correlación por campeón frente a la referencia experta se mantiene estable (**Spearman 0.8210**, **0.8215** y **0.8211**). Esto indica que gamma afecta principalmente a la legibilidad de la escala y a la estabilidad del gradiente, no a la señal ordinal del fenómeno.

Para realizar un control de calidad inicial del indicador (test de cordura o *sanity check*), se clasificó previamente a **47 campeones de soporte** según su propensión teórica al roaming basada en criterio de dominio y en clasificaciones públicas del juego (como las directrices de Riot Games y tier lists de analíticas como Mobalytics y U.GG, que agrupan a los soportes de iniciación y roaming frente a los protectores pasivos o enchanters). Al cruzar la media empírica de roaming de cada campeón con este ranking de referencia, se obtuvo una correlación de Spearman de **0.82**. Aunque se reconoce la limitación del sesgo de autor en el diseño inicial del ranking de control, la elevada correlación con clasificaciones externas del meta corrobora que la métrica numérica construida geométricamente captura la coherencia y la lógica táctica del videojuego.

---

### 3.3. Protocolo Experimental y Definiciones Estadísticas
Para entender la evaluación del modelo, es fundamental definir conceptualmente dos términos estadísticos que marcan la validez del entrenamiento:
*   **Datos Dentro de Muestra (In-sample):** Hace referencia a los datos de entrenamiento (train). Aquí el modelo ajusta sus parámetros conociendo las respuestas correctas. Un rendimiento alto en este conjunto puede deberse a que el modelo ha memorizado los datos de memoria (sobreajuste o *overfitting*).
*   **Datos Fuera de Muestra (Out-of-sample):** Son datos nuevos que el modelo no ha visto durante el entrenamiento (splits de validación y test). Evaluar el rendimiento aquí es lo único que garantiza si el modelo ha aprendido patrones reales capaces de generalizar.

#### Decisiones del Protocolo:
*   **Partición por partida (Group Split):** Dividimos los datos en train (70%), validación (15%) y test (15%) agrupando por la identidad de la partida (`match_id`). Esto evita que la perspectiva del equipo azul y del rojo de una misma partida queden en splits separados, lo que provocaría fuga de datos (fuga in-sample a out-of-sample). El test set se reservó intacto hasta la evaluación final.
*   **Filtro de Volatilidad (Partidas Caóticas):** El juego presenta partidas donde el desarrollo temprano se rompe de forma anómala (ej: el tirador aliado muere 5 o más veces antes del minuto 12, comportamiento denominado informalmente en el dominio como *"trollear"*). En estas situaciones (26.5% del dataset), el soporte abandona la línea inferior por colapso táctico y no por decisión estratégica del draft. Durante el entrenamiento, aplicamos un peso de muestra (`sample_weight`) de **0.40** a estas partidas caóticas y de **1.0** a las partidas limpias (valor seleccionado mediante un barrido experimental sistemático en validación, habiendo probado inicialmente un peso de 0.20). Esto reduce el impacto del ruido de ejecución in-game sin descartar por completo la información que aportan.

La Tabla 1 resume el protocolo experimental final. Esta tabla es importante porque fija qué información puede usar el modelo y qué queda reservado únicamente para construir la etiqueta.

#### Tabla 1: Protocolo experimental final

| Elemento | Valor |
| :--- | :--- |
| Unidad de análisis | `(match_id, team_id)` |
| Observaciones finales | 383.104 |
| Split | Train: 268.322 / Validación: 57.314 / Test: 57.468 |
| Criterio de partición | Agrupación por `match_id` |
| Partidas por split | Train: 134.221 / Validación: 28.669 / Test: 28.746 |
| Entrada de modelos principales | 10 IDs de campeón + lado del mapa |
| Variable objetivo | `support_roam_score` en escala [0, 1] |
| Ventana de etiqueta | Minutos 5 a 12 |
| Seeds de modelos aprendidos | 42, 123, 456 |
| Pesos de muestra | Limpias: 1.0 / Caóticas: 0.40 (optimizado por sweep) |
| Porcentaje de caóticas en test | 26.66% |

---

### 3.4. Funcionamiento Interno de los Modelos Comparados
Para demostrar al tribunal que comprendemos la mecánica interna de las arquitecturas y no las tratamos como cajas negras, desglosamos su funcionamiento algorítmico:

#### 3.4.1. HistGradientBoostingRegressor (HistGBT)
Es un modelo basado en un conjunto secuencial de árboles de decisión. Cada árbol se ajusta sobre los errores residuales que dejan los árboles anteriores, de modo que el modelo final combina muchas reglas simples. Para procesar eficientemente variables categóricas con alta cardinalidad (como los 173 campeones), el algoritmo puede ordenar las categorías a partir de estadísticos de la señal o del residuo dentro del nodo:
1.  Para cada categoría $c$ (champion ID), calcula un estadístico asociado a las observaciones que pertenecen a esa categoría.
2.  Ordena las categorías de menor a mayor según dicho estadístico: $c_{(1)}, c_{(2)}, ..., c_{(C)}$.
3.  En lugar de evaluar las $2^{C-1} - 1$ combinaciones posibles de partición (inviable computacionalmente), evalúa únicamente los $C-1$ puntos de corte en la secuencia ordenada:
    $$\text{Split } k: \quad \{c_{(1)}, ..., c_{(k)}\} \quad \text{vs} \quad \{c_{(k+1)}, ..., c_{(C)}\}$$
    Esto le permite agrupar categorías que presentan efectos parecidos sobre la predicción mediante reglas condicionales simples (ej: `¿soporte ∈ {Bard, Pyke, Thresh}?`).

#### 3.4.2. MLP Per-Role + Interactions (Modelo Neuronal)
Esta red neuronal densa procesa los datos categóricos mapeándolos a un espacio vectorial continuo:
*   **Embeddings por Rol:** Cada uno de los 10 slots de personajes del draft cuenta con su propia matriz de pesos de embedding $E_{\text{rol}} \in \mathbb{R}^{173 \times 16}$. Cuando se presenta un champion ID $c$, la red realiza un lookup de fila para extraer un vector denso de 16 números continuos. Al inicio del entrenamiento, estos vectores toman valores aleatorios de una distribución normal estándar $\mathcal{N}(0, 1)$ y se actualizan dinámicamente mediante backpropagation.
*   **Interacciones por Producto Escalar:** Para forzar a la red a evaluar el emparejamiento directamente, calculamos el producto escalar entre los embeddings de personajes clave:
    $$\text{Sinergia Botlane} = \vec{v}_{\text{soporte}} \cdot \vec{v}_{\text{tirador}} = \sum_{d=1}^{16} E_5[c_5, d] \cdot E_4[c_4, d]$$
    Geométricamente, el producto escalar mide la alineación (similitud) entre ambos vectores. Si la combinación de estos dos campeones favorece el roaming, el gradiente forzará a que sus vectores apunten en direcciones similares, reduciendo la distancia en el espacio vectorial de 16 dimensiones.

---

### 3.5. Métricas y Referencias Empíricas
Evaluamos los modelos con R², Spearman y MAE en el test set (fuera de muestra). Para contextualizar el rendimiento, calculamos también el **$R^2$ Group-Mean OOS (Out-of-Sample)**:
1.  Calculamos la media del roam score para cada grupo (ej: misma botlane en el lado azul) usando **únicamente los datos de train**.
2.  Usamos estas medias fijas para predecir las observaciones del **test set** (fuera de muestra), asignando la media global si el grupo es nuevo.
3.  Calculamos el $R^2$ predictivo resultante. Este valor funciona como una referencia simple de lookup y es directamente comparable con el de los modelos. 

*(El ICC descriptivo in-sample de train se calcula por ANOVA para verificar la consistencia interna, pero queda descartado para la comparación directa en test).*

La Tabla 2 muestra esta separación entre ICC y R² OOS. El ICC resume consistencia dentro de grupos en train; el R² OOS simula una predicción real entrenando medias en train y aplicándolas sobre test.

#### Tabla 2: ICC descriptivo frente a R² group-mean OOS

| Agrupación | ICC train | R² group-mean OOS |
| :--- | :---: | :---: |
| Support champion | 0.1214 | 0.1249 |
| Botlane champions | 0.1394 | 0.1239 |
| Botlane champions + side | 0.1391 | 0.1132 |
| Support vs enemy support | 0.1316 | 0.1200 |

---

## Capítulo 4: Resultados

### 4.1. Comparación de Modelos
La Tabla 3 resume las métricas de rendimiento obtenidas en el conjunto de test (fuera de muestra) para los modelos bajo el protocolo de entrada común de 11 variables: 10 campeones y lado del mapa.

#### Tabla 3: Comparación de Modelos en el Test Set (OOS)
*Los modelos entrenados reportan la media sobre las 3 semillas. Las desviaciones estándar son inferiores a 0.001 en todas las celdas, por lo que su variación es insignificante y se omite para simplificar la lectura.*

| Modelo | $R^2$ | Spearman | MAE (Error medio) | within ±0.10 | within ±0.20 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Global Mean** | -0.0008 | — | 15.51% | 37.9% | 68.9% |
| **Champion Mean** | 0.1243 | 0.3362 | 14.38% | 41.2% | 73.1% |
| **HistGBT** | **0.1595** | **0.3877** | **14.08%** | **42.0%** | **74.2%** |
| **MLP OneHot** | 0.1536 | 0.3801 | 14.12% | 41.9% | 74.0% |
| **MLP Embed Shared** | 0.1507 | 0.3763 | 14.15% | 41.7% | 73.9% |
| **MLP Per-Role + Inter.** | 0.1527 | 0.3783 | 14.14% | 41.8% | 74.0% |

*Nota de rigor metodológico:* Todas las métricas presentadas en esta tabla se calculan sobre el conjunto de test de forma estrictamente **no ponderada** (es decir, aplicando un peso de $1.0$ a todas las muestras, incluyendo las partidas clasificadas como caóticas). Aunque durante el entrenamiento se usaron pesos diferenciados para mitigar el ruido en el gradiente, la evaluación en test se realiza sin ponderaciones para medir fielmente la capacidad de generalización del modelo frente a la distribución real de partidas en producción.

#### Análisis de los Resultados:
1.  **Hay señal en el draft:** Todos los modelos mejoran el error de la media global ($R^2 > 0$). Esto responde a la pregunta de investigación: la selección inicial de personajes influye en el comportamiento de movilidad posterior.
2.  **El soporte define la base:** La baseline Champion Mean (que solo conoce la identidad del soporte) explica el **12.43%** de la varianza. Esto demuestra que la mayor parte del roaming predecible está determinado por la predisposición del personaje de apoyo elegido.
3.  **El draft completo añade contexto:** El HistGBT eleva el $R^2$ al **15.95%** y el Spearman a **0.3877**, demostrando que considerar el resto del draft (tirador, oponentes y lado) añade información táctica valiosa.
4.  **Los árboles superan a las redes:** El HistGBT se posiciona por delante de la MLP en Spearman y R² de forma consistente en todas las semillas.

#### Ablación de entrada: soporte + tirador aliado
Para comprobar si los 10 campeones eran realmente necesarios, se entrenó una versión adicional del HistGBT usando únicamente dos variables: el soporte aliado y el tirador aliado. Esta ablación permite separar la señal de botlane de la señal aportada por el resto del draft.

| Configuración | Variables de entrada | $R^2$ | Spearman | MAE |
| :--- | :--- | :---: | :---: | :---: |
| Champion Mean | Soporte aliado | 0.1243 | 0.3362 | 0.1438 |
| HistGBT 2 variables | Soporte + tirador aliados | 0.1393 ± 0.0001 | 0.3587 ± 0.0001 | 0.1426 |
| HistGBT completo | 10 campeones + lado | 0.1595 ± 0.0004 | 0.3877 ± 0.0004 | 0.1408 |

El resultado muestra que la botlane aliada concentra una parte importante de la señal, pero no toda. Añadir el resto del draft mejora el rendimiento en **+0.0202 de R²** y **+0.0290 de Spearman** frente al modelo de dos variables. Además, una tabla de medias exactas por pareja soporte+tirador no generaliza tan bien ($R^2 \approx 0.1205$), porque muchas combinaciones son raras; el HistGBT de dos variables funciona mejor al aprender efectos separados del soporte y del tirador en vez de memorizar cada pareja.

#### Compresión de las Predicciones:
El MAE es muy parecido para todos los modelos (~14.1%). Esto responde a un fenómeno matemático esperado: al enfrentarse a una señal explicable limitada ($R^2 \approx 16\%$) sobre una variable con alta varianza in-game irreductible ($\sigma \approx 0.190$), cualquier estimador que busque minimizar una función de pérdida cuadrática (MSE) o lineal (MAE) tenderá a comprimir sus predicciones hacia la media del target ($\sigma_{\text{pred}} \approx 0.074$). El modelo evita deliberadamente hacer predicciones agresivas en los extremos para minimizar la penalización por error absoluto o cuadrático. En términos prácticos, esto significa que el modelo no pretende adivinar con precisión el valor exacto de la movilidad en cada partida individual (lo que justificaría un despliegue inviable si ese fuera el fin). En su lugar, el modelo funciona como una guía táctica de ordenación de composiciones. Por este motivo, el coeficiente de correlación de rangos de Spearman (0.3877) es la métrica de rendimiento más adecuada y transparente para evaluar este sistema multiagente, ya que mide si el modelo es capaz de clasificar correctamente qué drafts exhiben una propensión mayor o menor a la movilidad relativa de sus agentes.

---

### 4.2. Referencias OOS y Experimento Residual
La Tabla 4 muestra las referencias de grupo evaluadas fuera de muestra (test set) en comparación con el HistGBT.

#### Tabla 4: HistGBT vs. Referencias de Grupo OOS en Test
*Las referencias de grupo se calculan entrenando las medias en train y prediciendo en test.*

| Referencia de Grupo OOS | $R^2$ en Test Set |
| :--- | :---: |
| Support Champion | 0.1249 |
| Botlane Champions | 0.1239 |
| Botlane Champions + Side | 0.1132 |
| **Modelo HistGBT** | **0.1595** |

El HistGBT supera claramente la referencia de lookup por botlane+side ($R^2 = 0.1132$). Esto indica que el modelo está aprovechando relaciones del draft que una media fija por grupo no captura bien fuera de muestra, especialmente cuando aparecen combinaciones raras o no vistas en test.

#### El Experimento Residual (Aislamiento de la señal del soporte):
Para comprobar si el modelo de verdad aprende del resto de personajes más allá del support aliado, realizamos un **experimento en dos etapas**:
1.  Calculamos la media suavizada del score de roaming del support aliado en train (R² de base = 0.1240).
2.  Definimos un nuevo target restando este efecto base: $y_{\text{residual}} = y - \text{media\_support}$.
3.  Entrenamos un HistGBT en base a este residuo, utilizando como entrada el resto del draft (excluyendo la variable del support directo). El modelo residual obtuvo un R² de **0.0386** y un Spearman de **0.1892** sobre el residuo.
4.  **Modelo Aditivo Completo:** Sumamos la predicción base del support y la predicción del HistGBT residual. El modelo combinado alcanzó un **R² de 0.1584** y un **Spearman de 0.3854** en test.

*Significado:* Al separar explícitamente la identidad del support, el resto del draft aporta un incremento predictivo neto (lift) de **+0.0343 en R²** y de **+0.0498 en Spearman**. Esto aporta evidencia de que el resto del draft contiene contexto táctico real, aunque limitado, sobre el comportamiento del soporte.

---

### 4.3. Importancia de Variables y Efecto del Lado
Al medir la importancia de las variables del HistGBT por permutación, los campeones aliados dominan con un peso de **0.255**, seguidos de los enemigos con **0.033**. La variable más importante es el soporte aliado, seguida del tirador aliado y del soporte enemigo.

#### Lectura del Lado del Mapa (`side`):
El lado del mapa se mantiene como variable prepartida porque el escenario no es perfectamente simétrico: los accesos a objetivos tempranos y las rutas seguras hacia río o jungla pueden cambiar según el lado. Aun así, en la evaluación final su importancia es muy pequeña frente a la identidad del soporte y de la botlane. En el conjunto final, la media del score es solo ligeramente mayor en rojo que en azul (aprox. **0.3928** frente a **0.3904**), por lo que este efecto debe interpretarse como un matiz del dominio y no como una señal dominante.

```
[Figura 3: Distribución superpuesta de la etiqueta support_roam_score según el lado del mapa (azul vs. rojo). Se observa que ambas distribuciones son muy parecidas, con una diferencia media pequeña.]
```

Esta lectura es útil para justificar por qué `side` entra en el protocolo común de entrada, pero los resultados indican que el modelo no depende principalmente de esta variable.

---

### 4.4. Rendimiento en Partidas Limpias vs. Caóticas
La Tabla 5 muestra la evaluación del HistGBT segmentada según el filtro de volatilidad in-game (`chaos_flag`).

#### Tabla 5: Rendimiento del HistGBT en Partidas Limpias vs. Caóticas
*Métricas obtenidas sobre el test set.*

| Subconjunto de Test | Observaciones ($n$) | $R^2$ | Spearman | MAE |
| :--- | :---: | :---: | :---: | :---: |
| **Todo el Test set** | 57.468 | 0.1605 | 0.3882 | 0.1408 |
| **Partidas Limpias** | 42.147 | **0.1719** | **0.3986** | 0.1384 |
| **Partidas Caóticas** | 15.321 | 0.1220 | 0.3630 | 0.1473 |

El rendimiento del modelo sube a un **R² de 0.1719** y un **Spearman de 0.3986** sobre las partidas limpias (donde el juego se desarrolla con normalidad). En cambio, en las caóticas (donde la línea inferior sufre muertes masivas que fuerzan la separación física de los jugadores, simulando un *"trolleo"*), el R² baja a **0.1220**. Esto confirma que el caos in-game es un factor importante que degrada la capacidad predictiva del draft.

Un ejemplo cualitativo ilustra esta limitación. En una partida con un soporte diseñado para permanecer cerca del tirador, el modelo predice correctamente un score bajo o moderado según el draft. Sin embargo, si el tirador muere repetidamente antes del minuto 12, la botlane deja de comportarse como una pareja estable y la etiqueta observada puede subir mucho porque el soporte aparece separado físicamente. En ese caso, el error no significa que el modelo haya ignorado una señal clara: la timeline está midiendo una separación real causada por una ejecución anómala que no estaba disponible antes de empezar.

---

### 4.5. Diagnóstico del Límite de Predictibilidad
Dado que el R² ronda el 16%, analizamos cuatro hipótesis para descartar que este límite se debiera a fallos de diseño:
*   **Hipótesis 1 (Complejidad del Modelo):** Evaluamos 108 configuraciones de hiperparámetros de la MLP. La mejor red neuronal apenas mejoró un 0.005 de Spearman a la MLP base y no superó al HistGBT. El límite no es de la arquitectura.
*   **Hipótesis 2 (Representación Categórica):** Probamos embeddings de 16 dimensiones. No superaron a la codificación One-Hot básica en la MLP y quedaron por detrás del HistGBT. El límite no está en cómo codificamos los personajes.
*   **Hipótesis 3 (Arbitrariedad de la Fórmula):** Entrenamos modelos sobre 15 variaciones de la etiqueta de roaming. Todas correlacionaron $\ge 0.99$ entre sí, demostrando que cualquier combinación lineal de desplazamientos espaciales mide esencialmente la misma señal.
*   **Hipótesis 4 (Definición del Fenómeno):** Construimos una etiqueta basada en eventos productivos (asistencias y objetivos) en lugar de separación espacial. El R² del HistGBT cayó a **0.091**, demostrando que los sucesos dinámicos de la partida son mucho menos previsibles desde el draft. La etiqueta espacial captura la máxima señal de predisposición disponible.

Este análisis demuestra que el límite del 16% es una **propiedad intrínseca del problema**: el draft inicial define una predisposición táctica, pero la ejecución de la partida introduce un 84% de variabilidad que no se puede capturar antes del minuto cero.

---

### 4.6. Prototipo CLI
Además de la evaluación experimental, se desarrolló un prototipo por consola que permite aplicar el modelo final a un draft introducido manualmente. La herramienta carga el modelo entrenado, recibe los 10 campeones de la partida y el lado del mapa, genera una predicción de `support_roam_score` y traduce el valor numérico a una lectura cualitativa del perfil esperado.

La interpretación no se hace con umbrales fijos sobre la escala absoluta $[0,1]$, ya que las predicciones del modelo están comprimidas alrededor de la media. En su lugar, el prototipo calibra la lectura usando la distribución de predicciones del propio modelo sobre el split de validación. En el HistGBT final, los cortes aproximados de predicción son: percentil 20 = **0.323**, percentil 40 = **0.364**, percentil 60 = **0.402**, percentil 80 = **0.450** y percentil 90 = **0.487**. Así, un score de 0.42 no se interpreta como "42% de roaming", sino como una predicción situada por encima de la mayoría de drafts comparables.

Un ejemplo simplificado de uso sería:

```text
Entrada:
  lado = blue
  draft aliado = Ornn, Lee Sin, Ahri, Ezreal, Rakan
  draft enemigo = Renekton, Sejuani, Syndra, Jinx, Lulu

Salida:
  support_roam_score = 0.4244
  percentil_predicción = 70.8
  interpretación = perfil medio-alto
```

Este prototipo no pretende sustituir el análisis estadístico del informe, sino demostrar que el pipeline completo puede integrarse en una herramienta interpretable: entrada prepartida, modelo entrenado, percentil calibrado y salida comprensible para un usuario del dominio.

---

## Capítulo 5: Discusión y Limitaciones

### 5.1. Significado del Límite del 16%
El modelo final explica un 16% de la varianza del roaming temprano del soporte. El 84% restante depende de factores in-game que no se pueden observar en el draft (coordinación, muertes tempranas, control de oleadas y decisiones individuales). 

Este resultado no es un fracaso de los modelos; es la respuesta real a la pregunta de investigación. Conseguir que el HistGBT supere las referencias de grupo out-of-sample ($0.1595$ frente a $0.1132$ para botlane+side) demuestra que el draft completo aporta información estructurada, aunque esté acotada por la propia naturaleza del juego.

---

### 5.2. ¿Por qué los Árboles Superan a las Redes Neuronales?
En este proyecto, el HistGBT superó a las MLPs de forma consistente. Esto pasa porque en entornos con señal predictiva débil (R² < 0.20) y variables de alta cardinalidad (173 campeones), los árboles de decisión explotan relaciones lógicas simples mediante splits condicionales directos en sus nodos. 

En cambio, las redes neuronales necesitan proyectar las categorías a coordenadas continuas y aprender sus combinaciones multiplicando matrices de pesos. Esto introduce una cantidad enorme de parámetros que, ante una señal débil y ruidosa, provoca que la red sobreajuste rápidamente (alcanzando su mejor época de validación entre la época 6 y 18) en lugar de generalizar.

---

### 5.3. Limitaciones
*   **Resolución temporal:** La timeline proporciona posiciones a intervalos de un minuto. En la ventana de 5 a 12 minutos, esto nos da solo 8 capturas por partida, haciendo que cualquier suceso accidental (como una muerte) altere significativamente el score.
*   **Proxy de comportamiento:** El score mide separación física y recursos, pero no puede leer la intención táctica del jugador.
*   **Validación de referencia experta:** La ordenación de soporte utilizada para validar la etiqueta fue construida por el propio autor basándose en su criterio de dominio, lo que podría introducir sesgo de confirmación.
*   **Filtro de caos:** El peso asignado a las partidas caóticas se optimizó mediante barrido en validación, pero la definición inicial del `chaos_flag` se apoya en umbrales de muertes definidos mediante criterio táctico.
*   **Alcance del dataset:** Los datos proceden de EUW, partidas clasificatorias Solo/Duo de nivel alto y parches 16.2 a 16.8. Los patrones podrían variar en otros rangos, regiones, colas competitivas o versiones futuras del juego.

---

## Capítulo 6: Conclusiones y Trabajo Futuro

### 6.1. Conclusiones
*   **C1:** El draft inicial contiene una señal parcial pero real sobre la movilidad temprana del soporte. El HistGBT final explica un **15.95% ± 0.04%** de la varianza y ordena los drafts con una correlación de Spearman de **0.3877 ± 0.0004** (con un peso de caóticas optimizado a 0.40).
*   **C2:** La identidad del soporte es la variable dominante del draft ($R^2 = 0.1243$). Añadir el tirador aliado eleva el rendimiento a $R^2 = 0.1393$, pero el modelo completo con los 10 campeones y el lado del mapa alcanza $R^2 = 0.1595$. Además, mediante el experimento residual demostramos que el contexto del resto del draft aporta un lift predictivo neto de **+0.0343 en R²** y de **+0.0498 en Spearman**.
*   **C3:** El modelo final supera las referencias out-of-sample de grupo, incluida botlane+side ($R^2 = 0.1132$), lo que indica que generaliza mejor que una tabla de medias históricas.
*   **C4:** Los modelos tabulares (HistGBT) son más adecuados que las redes neuronales para procesar drafts bajo baja señal, reduciendo el sobreajuste.
*   **C5:** El límite de predictibilidad prepartida es estructural: el 84% de la varianza pertenece a la ejecución in-game. Al intentar predecir una etiqueta más estricta de roaming basada en eventos in-game, el R² cayó a **0.091**, confirmando que el draft captura predisposición y no ejecución.

### 6.2. Trabajo Futuro
*   **Modelado secuencial del draft:** Incorporar el orden de selección temporal de los agentes (secuencia de selecciones y bloqueos) mediante arquitecturas de Redes Recurrentes (RNN/LSTM) o Transformers, lo que permitiría capturar las intenciones tácticas de reacción directa (blind picks vs. counters).
*   **Estabilidad temporal y MLOps:** Diseñar un pipeline de reentrenamiento continuo (frecuencia mensual o por parche del juego) para mitigar la degradación del rendimiento ocasionada por los desajustes de datos y conceptos (*data drift* y *concept drift*), típicos de entornos competitivos que sufren modificaciones continuas de balance de agentes (meta).
*   **Modelado multiagente:** Adaptar la metodología para predecir de forma conjunta la movilidad de otros roles clave, como el jungla o el carrilero central.
*   **Eventos tácticos complementarios:** Incorporar a la etiqueta espacial una métrica de impacto directo en objetivos neutrales o escaramuzas fuera de la zona inferior para ponderar el "roaming con impacto táctico".
*   **Generalización trans-servidor:** Ampliar el dataset a regiones como Corea del Sur (KR) o Norteamérica (NA) para comprobar si la señal predictiva del draft varía según la cultura de juego regional.

---

## Capítulo 7: Bibliografía

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
