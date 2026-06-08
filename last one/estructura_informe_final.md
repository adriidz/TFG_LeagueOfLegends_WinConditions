# Estructura del Informe Final — TFG Support Roaming

> **Restricciones**: 8–10 páginas de contenido + 4 páginas de anexos.
> **Secciones obligatorias**: Objetivos, Estado del arte, Metodología, Resultados, Conclusiones, Bibliografía.

---

## Visión general del reparto de espacio

| Sección | Páginas estimadas |
|---|---:|
| 1. Introducción y objetivos | 1.25 |
| 2. Estado del arte | 0.75 |
| 3. Metodología | 2.5 |
| 4. Resultados | 3.0 |
| 5. Discusión y limitaciones | 1.0 |
| 6. Conclusiones | 0.5 |
| Bibliografía | 0.5 |
| **Total contenido** | **~9.5** |
| Anexo A — Arquitecturas y embeddings | 1.5 |
| Anexo B — Figuras complementarias | 1.5 |
| Anexo C — Prototipo CLI | 1.0 |
| **Total anexos** | **~4.0** |

---

## Cuerpo principal (8–10 páginas)

### 1. Introducción y objetivos (~1.25 páginas)

#### 1.1 Contexto y motivación

Abrir con una descripción general del problema sin asumir que el tribunal conoce el juego. Puntos que deben quedar cubiertos:

- **MOBA como sistema multiagente**: Explicar en 2-3 frases que League of Legends es un videojuego competitivo por equipos (5v5), donde antes de empezar la partida cada equipo selecciona sus agentes (campeones). Esta selección previa se denomina *draft*. No hace falta explicar reglas del juego — solo que hay una fase de configuración (draft) y una fase de ejecución (partida).
- **Separación input/target**: Dejar claro desde el principio que el TFG se apoya en esta dicotomía: lo que se sabe antes de empezar (draft) vs. lo que ocurre después (timeline). El modelo usa lo primero como entrada y lo segundo solo para construir la variable que intenta predecir.
- **LoL como caso de estudio**: Una frase justificando por qué LoL y no otro juego: API pública con datos de composición y posiciones temporales, escala masiva de partidas, y un problema documentado en la literatura.
- **El roaming del support**: Explicar en 2-3 frases accesibles qué es el agente de apoyo (uno de los cinco roles, empieza acompañando al tirador en la zona inferior del mapa) y qué significa que "roamee" (que abandone esa zona para generar ventajas en otras partes del mapa). Usar vocabulario genérico: "zona inferior", "agente de apoyo", "tirador". No decir "botlane", "ADC", "drake" ni jerga específica sin traducir.

#### 1.2 Pregunta de investigación

Un solo párrafo corto que contenga:

- **Pregunta central formulada de forma explícita**: *"¿Hasta qué punto la composición de agentes seleccionados antes de la partida permite anticipar la movilidad temprana del agente de apoyo?"*
- **Encuadre como cuantificación, no predicción perfecta**: Aclarar que el objetivo no es construir un oráculo que adivine cada movimiento del jugador, sino medir cuánta información predictiva contiene la configuración previa. La pregunta admite una respuesta del tipo "el draft explica X% de la varianza" y eso ya es un resultado válido, sea X alto o bajo.

#### 1.3 Objetivos específicos

Listar cuatro objetivos redactados sin jerga técnica no explicada previamente:

- **O1**: Construir un indicador numérico de la movilidad temprana (*roaming*) del *support* a partir de datos observados en partida y comprobar su coherencia con conocimiento experto del dominio.
- **O2**: Comparar modelos tabulares y neuronales bajo un protocolo experimental común para medir cuánta señal predictiva es posible extraer de la composición previa de campeones (*draft*).
- **O3**: Estimar un techo empírico de predictibilidad para determinar el límite práctico de lo que se puede anticipar antes de iniciar la partida.
- **O4**: Desarrollar un prototipo aplicado que traduzca una composición previa a una lectura cualitativa de tendencia de movilidad.

> [!IMPORTANT]
> No usar ICC, OOS, R², proxy ni ningún acrónimo técnico en los objetivos. Esos conceptos aparecen cuando se explican en la metodología (§3.5). Los objetivos deben ser legibles para un evaluador que no haya leído el resto del informe.

> [!NOTE]
> **Qué sale**: Toda la narrativa de evolución (clasificación → regresión) se elimina del cuerpo. La historia del proyecto no es el tema del informe final — el tema es la pregunta, el método y la respuesta.

---

### 2. Estado del arte (~0.75 páginas)

#### 2.1 Predicción de resultados y recomendación en la selección

Describir los dos bloques principales de la literatura y posicionar el TFG respecto a ellos:

- **Predicción de victoria desde draft**: Citar a Hitar-Garcia et al. (2023) con su dato concreto: predicen si un equipo ganará la partida a partir del draft con una precisión de ~58% (la línea base aleatoria es 50%). Esto permite al tribunal entender que incluso predecir una variable binaria simple como victoria/derrota desde el draft es difícil. Incluir la lectura: *"esto ilustra la dificultad intrínseca de extraer señal predictiva de la configuración previa"*.
- **Recomendación de draft**: Citar DraftRec (Lee et al. 2022) como ejemplo de un sistema que sugiere qué campeón elegir, no que prediga qué ocurrirá. Dejar claro que no tiene métricas de varianza explicada comparables con las nuestras.
- **Frase de posicionamiento**: *"Mientras la literatura existente se centra en predecir el resultado final de la partida (victoria o derrota) o en recomendar selecciones óptimas, este TFG plantea una pregunta diferente: predecir un comportamiento intermedio (la movilidad del agente de apoyo) que no es una variable observada directamente sino un indicador construido a posteriori."* Esta frase es la que justifica la novedad del trabajo.

#### 2.2 Análisis de patrones de comportamiento y trayectorias

Describir brevemente los trabajos descriptivos/analíticos existentes y explicar cómo se diferencian:

- **Rama et al. (2020)**: Usan modelos de Markov ocultos (HMM) para identificar patrones de comportamiento en partidas de LoL. Su enfoque es descriptivo y post-partida: caracterizan lo que ocurrió, no lo anticipan.
- **Wallner et al. (2023)**: Visualización de la evolución espacio-temporal de partidas. Herramienta analítica, no predictiva.
- **Chen et al. (2025)**: Predicción de trayectorias en MOBA, pero usando información in-game (posiciones previas durante la partida), no información pre-partida.
- **Frase de diferenciación**: *"Los trabajos previos analizan patrones de comportamiento durante o después de la partida. No existe trabajo previo que intente predecir un comportamiento espacial intermedio a partir exclusivamente de la composición previa. El proyecto abre una línea nueva en este sentido."* Esta frase es importante para el tribunal: no hay un estado del arte contra el que competir con R²=0.80 — se abre una línea.

#### 2.3 Herramientas y técnicas utilizadas

Párrafo breve (4-5 líneas) citando las herramientas metodológicas que se usan en el TFG y que tienen base en la literatura:

- Gradient boosting con categóricas nativas: [Ke et al. 2017, LightGBM]; sklearn HistGBT. Explicar en 1 frase que este tipo de modelo maneja de forma nativa variables categóricas con muchas categorías (como los ~170 campeones del juego).
- Embeddings categóricos: mencionar que la idea de representar entidades discretas como vectores densos proviene de NLP (Word2Vec) y sistemas de recomendación.
- SHAP [Lundberg & Lee 2017]: método de explicabilidad para entender qué variables contribuyen más a cada predicción.
- ICC [McGraw & Wong 1996]: medida estadística de consistencia intra-grupo, usada aquí para estimar el techo empírico.

> [!NOTE]
> **Qué sale**: No se dedica espacio a explicar la API de Riot (va en metodología), ni a describir el juego en detalle (ya se hizo en §1.1 con lo mínimo necesario). No se hace una revisión exhaustiva — se citan los 6-8 trabajos más relevantes y se posiciona el TFG respecto a ellos.

---

### 3. Metodología (~2.5 páginas)

#### 3.1 Datos y unidad de análisis

Describir el dataset y justificar las decisiones de recogida:

- **Fuente**: API de Riot Games, endpoints match-v5 (metadatos de partida y composición) y timeline (posiciones y eventos a lo largo del tiempo). Mencionar Data Dragon como fuente de datos estáticos de campeones.
- **Muestra**: ~191k partidas, lo que genera ~383k observaciones match-team. Servidor EUW, modo Ranked Solo/Duo, nivel alto (Master+). Parches 16.2 a 16.8. Explicar brevemente por qué nivel alto: se asume que la ejecución se aproxima mejor a la intención estratégica del draft que en niveles bajos.
- **Unidad de análisis**: (match_id, team_id). Explicar que cada partida genera dos observaciones independientes: una desde la perspectiva de cada equipo. Esto es necesario porque un mismo campeón puede ser aliado o enemigo dependiendo del punto de vista.
- **Variables de entrada (features)**: 10 IDs de campeón (5 aliados asignados por rol + 5 enemigos asignados por rol) + lado del mapa (blue/red). La timeline **no** se usa como entrada — solo para construir la etiqueta. Esta separación es la garantía principal contra data leakage.

#### 3.2 Construcción de la etiqueta

Esta es posiblemente la sección más importante de la metodología. Hay que construir una narrativa que lleve al lector desde la intuición del fenómeno hasta la fórmula concreta. Estructura recomendada:

1. **Párrafo cualitativo introductorio** (antes de la fórmula): *"El mapa posee una zona inferior donde operan dos agentes aliados: el de apoyo (support) y el tirador (ADC). Durante los primeros minutos, el agente de apoyo puede permanecer en esa zona o desplazarse a otras partes del mapa para generar ventajas. El indicador mide tres señales cuantitativas de dicho desplazamiento: (1) la proporción de tiempo que el apoyo pasa fuera de su zona asignada, (2) la distancia física respecto al tirador, y (3) la diferencia de recursos acumulados entre ambos, que refleja indirectamente si comparten la misma zona."* No hace falta explicar dragones, jungla ni objetivos — solo zona inferior, apoyo, tirador y desplazamiento.

2. **Fórmula como formalización del párrafo anterior**: `score = (0.45 × outside_ratio + 0.35 × far_ratio + 0.20 × xp_gap) ^ 0.75`. Explicar brevemente cada componente: `outside_ratio` es la fracción de snapshots en los que el apoyo está fuera de la zona inferior; `far_ratio` es la fracción de snapshots en los que está lejos del tirador; `xp_gap` es la diferencia relativa de experiencia acumulada, que indica si comparten recursos. La transformación gamma (^0.75) comprime ligeramente los valores extremos.

3. **Ventana temporal**: Minutos 5 a 12 de partida. Una frase justificando: los primeros 5 minutos se excluyen porque los agentes aún están en la fase inicial de la partida; después del minuto 12 el juego cambia de fase y el indicador pierde relevancia como medida de movilidad "temprana". Mencionar en 1 frase que se probaron varias ventanas (6, 8, 10, 12, 14 min) y se eligió 5-12 tras análisis de estabilidad.

4. **Geometría manual del mapa**: Definición de zonas (botlane, río, jungla, mid, bases) trazadas manualmente sobre el mapa para clasificar cada posición. Referenciar la figura al Anexo B.2.

5. **Validación del indicador con referencia experta**: Este es uno de los argumentos más fuertes del TFG. Redactar con cuidado: *"Para comprobar que el indicador recoge el fenómeno esperado, se construyó —antes de evaluar los modelos— una ordenación de referencia experta de 47 agentes de apoyo según su tendencia teórica al desplazamiento. La correlación de Spearman entre la media empírica del score por agente y esta referencia es de 0.82. Los agentes diseñados para moverse por el mapa (ej. Bard, Pyke) aparecen en la zona alta del ranking, y los más ligados a su zona (ej. Yuumi, Soraka) en la zona baja."* Destacar que se hizo antes de ver resultados de modelos, que mide ranking (no valores absolutos), y que 0.82 es una correlación fuerte. El scatter detallado va al Anexo B.5.

6. **Robustez de la fórmula**: Una frase: *"Se probaron 15 variantes de pesos y combinaciones de componentes, todas con correlación lineal ≥0.99 entre sí, lo que indica que la señal capturada no depende de una elección arbitraria de pesos."*

#### 3.3 Protocolo experimental

Describir las decisiones experimentales que garantizan la validez de la comparación:

- **Split por match_id**: train (70%) / val (15%) / test (15%) usando GroupShuffleSplit de scikit-learn. Explicar por qué se agrupa por partida: evitar que las dos perspectivas de una misma partida caigan en particiones distintas (data leakage). El test se reserva intacto hasta la comparación final.
- **Partidas caóticas (chaos_flag)**: Definir en 1-2 frases qué es una partida caótica: aquella en la que la zona inferior sufre un colapso temprano (≥6 muertes de apoyo+tirador antes del minuto 12, o el tirador muere ≥5 veces, o el apoyo muere ≥4 veces sin acciones productivas fuera de su zona). Representan ~26.5% de las observaciones. Se les asigna sample_weight = 0.2 durante el entrenamiento (vs. 1.0 para las limpias) para mitigar su efecto sin eliminarlas.
- **Lado del mapa (Side)**: *"El mapa no es simétrico: los dos equipos juegan desde lados opuestos, lo que afecta al acceso a ciertos objetivos y a la geometría de los desplazamientos. Se codifica como variable binaria."*
- **Hechizos de invocador (Summoner spells)**: *"Cada agente selecciona dos habilidades adicionales antes de la partida que condicionan ligeramente su estilo de juego. Se codifican como variables categóricas."* Aclarar que se probaron en el protocolo completo y se verificó que su importancia es prácticamente nula (~0.001), por lo que se excluyen de la comparación principal para mantenerla limpia.
- **Protocolo de features**: Idéntico para todos los modelos en la comparación principal: 10 champion IDs + side. Ningún modelo recibe información adicional.
- **Reproducibilidad**: Cada modelo se entrena con 3 semillas (42, 123, 456) y se reporta media ± desviación estándar. Seguimiento con Weights & Biases (WandB).

#### 3.4 Modelos comparados (4 variantes bajo protocolo común)

Describir cada modelo con suficiente detalle para que el lector entienda qué hace y por qué se incluye, pero sin entrar en hiperparámetros (van al Anexo A):

- **Global Mean**: Predice siempre la media global del target calculada en train. Es la línea base mínima: si ningún modelo la supera, no hay señal aprendible. Sirve como referencia de "cero información".
- **Champion Mean**: Para cada observación, predice la media histórica del score de roaming del campeón de apoyo aliado calculada en train. Es la línea base de dominio: mide cuánto se puede explicar solo sabiendo qué personaje juega el apoyo, sin considerar el resto del draft. Si el modelo final apenas supera esta baseline, significa que el draft completo añade poca información más allá de la identidad del apoyo.
- **HistGBT**: Mejor modelo tabular. Usa HistGradientBoostingRegressor de scikit-learn, que acepta variables categóricas de forma nativa sin necesidad de one-hot encoding. Captura interacciones no lineales entre campeones mediante splits condicionales de árbol (ej.: "si el apoyo es Bard Y el tirador es Ezreal → score alto"). Los hiperparámetros principales (max_iter=300, max_depth=6, lr=0.05, min_samples_leaf=50) se detallan en Anexo A.
- **MLP Per-Role + Interactions**: Mejor modelo neuronal. Cada uno de los 10 slots de campeón tiene su propia tabla de embedding (173 campeones × 16 dimensiones), lo que permite al modelo aprender representaciones diferentes para un mismo campeón según su rol. Tras la concatenación de embeddings, pasa por capas densas (192→96) con dropout (0.35), weight decay (5e-4) y optimizador AdamW. Incluye 2 productos escalares explícitos (apoyo-vs-apoyo_rival, apoyo-vs-tirador_aliado) como interacciones de primer orden. Representa el límite neuronal bajo protocolo común: se probaron también variantes más simples (OneHot, Shared Embeddings) que obtuvieron resultados inferiores, por lo que esta variante las subsume.

> [!NOTE]
> **Justicia de la comparación**: La comparación es justa porque ambos modelos (HistGBT y MLP Per-Role) acceden exactamente a las mismas 10+1 variables. El HistGBT captura interacciones de forma nativa mediante splits condicionales en sus árboles; la MLP lo hace mediante embeddings y productos escalares. Son aproximaciones representacionales diferentes al mismo problema, y que uno supere al otro es un hallazgo legítimo.

#### 3.5 Métricas y techo empírico

Explicar qué se mide y cómo se interpreta cada métrica:

- **Métricas principales**:
  - **R²**: Fracción de varianza del target explicada por el modelo. R²=0 equivale a predecir siempre la media; R²=1 sería predicción perfecta. Sensible a valores extremos.
  - **Spearman**: Correlación de ranking entre predicciones y valores reales. Mide si el modelo ordena correctamente las composiciones de menor a mayor roaming, independientemente de los valores absolutos. Es la métrica más informativa para este problema.
  - **MAE**: Error absoluto medio. Fácil de interpretar ("en promedio el modelo se equivoca en X puntos del score") pero, como se verá en resultados, puede ser engañoso cuando el modelo comprime predicciones.
- **Métricas complementarias**: within ±0.10 y within ±0.20 (% de predicciones que caen dentro de ese margen del valor real). Útiles como apoyo narrativo pero deben interpretarse con cautela por la compresión de predicciones.
- **Techo empírico**: Para contextualizar un R² "bajo", se calculan dos referencias paralelas:
  - **ICC** (Intraclass Correlation Coefficient): Consistencia in-sample por agrupación de draft (ej. "todas las partidas donde el apoyo es Bard y el tirador es Ezreal en lado azul"). Es una medida descriptiva de cuánta variabilidad del score se debe al grupo vs. a la variabilidad interna de cada partida.
  - **R² group-mean OOS**: Se calculan las medias por grupo en train y se usan como predicción en test (con fallback a la media global para grupos no vistos). **Este** es el valor directamente comparable con el R² de los modelos, porque se evalúa en el mismo conjunto de test.
  - **Aclaración importante**: ICC ≠ R² group-mean. Son métricas paralelas sobre los mismos grupos. No deben mezclarse al comparar con modelos.

> [!NOTE]
> **Qué sale**: El pipeline detallado paso a paso (5 etapas del Informe II §4.2) se comprime a un diagrama en Anexo B.1. Los hiperparámetros detallados de cada modelo van en tabla en Anexo A.1. En el cuerpo solo se describe la idea de cada modelo en 2-3 frases.

---

### 4. Resultados (~3 páginas)

#### 4.1 Comparación principal bajo protocolo común

Esta es la tabla central del informe. Presentarla limpia y acompañarla de lectura narrativa:

| Modelo | R² | Spearman | MAE | ±0.10 | ±0.20 |
|---|---:|---:|---:|---:|---:|
| HistGBT | 0.160 ± 0.000 | 0.387 ± 0.000 | 0.141 | 42.0% | 74.2% |
| MLP Per-Role + Inter | 0.153 ± 0.001 | 0.378 ± 0.001 | 0.141 | 41.8% | 74.0% |
| Champion Mean | 0.124 | 0.336 | 0.144 | 41.1% | 72.9% |
| Global Mean | –0.001 | — | 0.155 | 37.9% | 68.9% |

**Lecturas que deben quedar explícitas en el texto** (en este orden):
1. **Existe señal**: Todos los modelos superan claramente la Global Mean → la composición previa contiene información predictiva real sobre el roaming. Esto ya responde parcialmente la pregunta de investigación.
2. **El campeón de apoyo domina**: Champion Mean ya alcanza R²=0.124 y Spearman=0.336. El salto de Global Mean a Champion Mean es mayor que el salto de Champion Mean a HistGBT. Esto demuestra que la identidad del personaje de apoyo es el factor predictivo dominante.
3. **El draft completo añade señal**: El HistGBT mejora sobre Champion Mean (Spearman 0.336→0.387), lo que prueba que considerar los 10 campeones y sus interacciones aporta información adicional, aunque limitada.
4. **Los árboles superan a las redes**: El HistGBT supera consistentemente a la MLP Per-Role (la mejor variante neuronal) con desviación estándar pequeña entre semillas.

**Honestidad sobre el MAE** — un párrafo imprescindible: *"El MAE es prácticamente idéntico (~0.141) para todos los modelos. Esto no es casualidad: la desviación estándar de las predicciones (σ ≈ 0.074) es muy inferior a la del target (σ ≈ 0.190). El modelo comprime sus predicciones hacia el centro de la distribución porque, en un régimen de señal limitada (R² ≈ 0.16), arriesgar con predicciones extremas empeoraría la pérdida cuadrática media. Todos los modelos se equivocan de forma similar en promedio porque todos predicen cerca del centro. Las diferencias entre modelos aparecen en R² y Spearman, que son sensibles al ranking y a los extremos."*

**Argumento principal — Spearman, no within**: *"La métrica within ±0.20 (74.2%) puede parecer alta, pero exagera el rendimiento real por efecto de la compresión. La mejora respecto a la Global Mean (68.9% → 74.2%) es de solo 5.3 puntos porcentuales. El Spearman (0.39) ofrece una lectura más informativa: el modelo ordena las composiciones según su propensión al roaming con coherencia moderada, mejorando la baseline por campeón (0.34)."*

#### 4.2 Techo empírico y posición del modelo

Contextualizar el R² del modelo frente a referencias group-mean OOS:

| Referencia group-mean (OOS) | R² |
|---|---:|
| Support champion | 0.125 |
| Botlane champions | 0.124 |
| Botlane + side | 0.113 |
| **HistGBT** | **0.160** |

Puntos que deben quedar claros en el texto:
- **El modelo supera todas las referencias group-mean OOS**: Esto significa que combina información de los 10 slots del draft de forma no trivial. Una simple tabla de medias por subgrupo no alcanza el mismo rendimiento predictivo que el HistGBT.
- **ICC in-sample vs R² group-mean OOS**: Mencionar que el ICC in-sample de botlane+side es ≈0.139, pero que no es directamente comparable con el R² de modelos porque se calcula in-sample. El R² group-mean OOS (0.113 para botlane+side) sí es comparable y el HistGBT lo supera claramente.
- **Interpretación del límite**: El modelo extrae más señal que cualquier tabla de medias por subconjunto del draft. La varianza restante proviene de la ejecución individual: decisiones del jugador, muertes tempranas, coordinación, estado de la línea — factores no observables antes de la partida.

#### 4.3 Importancia de variables

Presentar la estructura de importancia y conectarla con el dominio:

- **Importancia por permutación agrupada**:
  - Campeones aliados: 0.255 (grupo dominante)
  - Campeones enemigos: 0.033
  - Summoner spells: ~0.001
  - Side: ~0.000
- **Ranking individual**: `ally_utility_champion_id` (apoyo aliado) es la variable individual más importante, seguida de `ally_bottom_champion_id` (tirador aliado) y `enemy_utility_champion_id` (apoyo rival). Incluir una frase interpretativa: *"Esto concuerda con la lógica del juego: el personaje de apoyo define su predisposición intrínseca a desplazarse, el tirador condiciona la viabilidad de abandonar la zona inferior (un tirador autosuficiente facilita el roaming), y el apoyo rival determina el nivel de presión en la zona."*
- **Asimetría del mapa (Side)** — párrafo clave para la defensa: *"Aunque la importancia por permutación del lado sea prácticamente nula en el modelo global, existe una asimetría física en el mapa que influye en los desplazamientos. Un análisis exploratorio confirmó que los apoyos de lado rojo presentan un score promedio ligeramente superior (0.3928 vs. 0.3904 en lado azul). La explicación reside en la geometría: la distribución de muros, accesos y la posición del arbusto lateral (tribush) en el cuadrante inferior obliga a los apoyos de lado rojo a realizar recorridos más largos para incorporarse al río o rotar hacia la zona media. No obstante, este efecto es mucho menor que el del propio campeón de apoyo (importancia 0.255 vs. ~0.000), por lo que queda eclipsado en las métricas de importancia global."* Apoyarse en una fotografía/captura del mapa durante la defensa para señalar la asimetría.
- **SHAP**: Confirma la misma jerarquía de variables. Los gráficos detallados (beeswarm, waterfall) van al Anexo B.4.

#### 4.4 Partidas limpias vs. caóticas

Mostrar la tabla de subconjuntos y extraer la lectura:

| Subconjunto | n | R² | Spearman |
|---|---:|---:|---:|
| Todas | 57.468 | 0.161 | 0.388 |
| Limpias | 42.147 | 0.172 | 0.399 |
| Caóticas | 15.321 | 0.122 | 0.363 |

Puntos a desarrollar:
- **El modelo predice mejor en partidas limpias**: R² sube de 0.161 a 0.172 y Spearman de 0.388 a 0.399. Esto confirma que el caos temprano (colapso de la zona inferior, muertes masivas) introduce varianza que no puede anticiparse desde la composición previa.
- **Auditoría cualitativa**: En 2-3 frases, describir el análisis de los 20 errores más extremos del modelo. Resultado: 17 de los 20 corresponden a partidas caóticas. Incluir un ejemplo concreto: *"Un agente de apoyo con perfil bajo de desplazamiento (ej. Soraka) obtiene un score observado alto porque la zona inferior colapsó con muertes masivas tempranas, forzando una separación física no intencional. El modelo predice correctamente la tendencia esperada del personaje, pero no puede anticipar el colapso."*
- **Lectura para la conclusión**: El caos temprano es varianza no capturable desde el draft. El filtro de partidas caóticas mitiga el ruido, pero no lo elimina — lo cual es una limitación reconocida.

#### 4.5 Investigación sistemática del techo predictivo

Esta sección es clave para el tribunal. El modelo apenas mejora sobre un lookup por campeón en MAE (0.144 → 0.141). La pregunta natural es: *¿por qué?* Se realizaron cuatro líneas de investigación para descartar hipótesis alternativas y demostrar que el techo es inherente al fenómeno, no a una carencia del trabajo.

**Hipótesis 1 — ¿El modelo es demasiado simple?**
Se compararon 3 variantes de MLP (OneHot, Shared Embeddings, Per-Role + Interactions) y el HistGBT. Las tres MLPs obtienen resultados muy similares entre sí; la variante Per-Role es la mejor pero por un margen marginal. El HistGBT supera a todas consistentemente (Spearman 0.387 vs. 0.378, std ≈ 0.001). Se probaron 108 configuraciones de hiperparámetros de la MLP: la mejor mejora ~0.005 en Spearman. **Conclusión: el límite no es arquitectural.** No se resolverá cambiando capas, dropout, learning rate ni representación de los campeones.

**Hipótesis 2 — ¿La representación de los campeones es el cuello de botella?**
Los embeddings per-role (10 tablas de 173×16) no superan a la codificación one-hot simple. Los vectores aprendidos muestran correlación débil entre distancia vectorial y diferencia de roaming medio. Capturan algo de estructura semántica (campeones similares quedan cercanos en el espacio), pero eso no basta para superar los splits categóricos nativos de los árboles. **Conclusión: el cuello de botella no está en la representación.**

**Hipótesis 3 — ¿La fórmula de la etiqueta es arbitraria o subóptima?**
Se probaron 15 variantes de pesos y combinaciones de componentes (outside_ratio, far_ratio, xp_gap, evidencia de combate, control de visión). Todas las variantes correlacionan ≥0.99 entre sí. La mejor variante mejora 0.002 en Spearman respecto a v5. **Conclusión: el límite no está en la fórmula.** Con ~8 snapshots minutales, cualquier combinación lineal de los componentes espaciales mide esencialmente la misma señal.

**Hipótesis 4 — ¿La etiqueta es demasiado laxa?**
Se construyó una variante alternativa de la etiqueta que, en lugar de medir solo la separación espacial, exige evidencia de acciones concretas del apoyo fuera de su zona durante la partida. Con esta definición más estricta, la predictibilidad desde el draft **bajó** (R² de 0.160 a 0.091). La lectura es directa: los eventos concretos que ocurren durante la partida dependen más de la ejecución que de la composición previa, y por tanto son menos anticipables. La etiqueta espacial captura precisamente la parte predecible del fenómeno — la predisposición agregada del campeón — mientras que lo que ocurre en cada partida concreta queda dominado por factores no observables antes de empezar.

**Lectura conjunta para el tribunal**: Cada hipótesis razonable fue probada y descartada. El techo no proviene ni del modelo, ni de la representación, ni de la fórmula, ni de una definición demasiado laxa de la etiqueta. Proviene de la naturaleza del fenómeno: la ejecución individual introduce ~84% de varianza que no es capturable desde la composición previa.

> [!NOTE]
> **Qué sale a anexos**: Los detalles de t-SNE/UMAP/vecinos de embeddings van a Anexo B. Las curvas de entrenamiento de MLPs a Anexo A. El prototipo CLI va entero al Anexo C.

---

### 5. Discusión y limitaciones (~1 página)

#### 5.1 Qué significa el resultado y límite del problema

Sección narrativa que conecta los números con la pregunta de investigación. Esta sección debe convencer al tribunal de que el resultado modesto no es por falta de esfuerzo, sino por la naturaleza del problema — y que demostrarlo es precisamente la contribución.

- **El 16% y el 84%**: *"El modelo explica aproximadamente un 16% de la varianza del indicador de movilidad. El 84% restante depende de factores que solo se conocen durante la partida: coordinación entre jugadores, muertes tempranas, estado de las oleadas de unidades, recalls, pathing del jungla, prioridad de la zona media, control de visión y decisiones individuales."*
- **La compresión como consecuencia matemática**: *"En un régimen de señal limitada, el modelo tiende a predecir valores cercanos a la media global. La desviación de las predicciones (σ ≈ 0.07) es menos de la mitad de la del target (σ ≈ 0.19). Esto implica que la métrica within ±0.20 (74.2%) exagera el rendimiento aparente por efecto de concentración. El coeficiente de Spearman (0.39) ofrece la lectura más honesta: el modelo ordena las composiciones según su propensión al desplazamiento con coherencia moderada, mejorando la baseline por campeón (0.34 → 0.39)."*
- **La investigación del techo como contribución central**: Conectar con §4.5: *"Se investigaron cuatro hipótesis para explicar por qué el modelo apenas mejora sobre una tabla de medias por campeón: complejidad insuficiente (descartada), representación subóptima (descartada), fórmula arbitraria (descartada) y definición demasiado laxa de la etiqueta (una variante más estricta basada en eventos concretos en partida resultó ser menos predecible, no más, confirmando que la ejecución domina sobre la predisposición). La conclusión es que el techo es inherente al fenómeno."*
- **No es un fracaso — es la respuesta**: *"Cuantificar este límite no representa un fracaso del modelo, sino el núcleo de la investigación. La pregunta era hasta qué punto el draft permite anticipar la movilidad, y la respuesta es: parcialmente, con un Spearman de 0.39 y un R² de 0.16. Que el modelo supere todas las baselines de tabla de medias (R² group-mean OOS 0.113-0.125) demuestra que el draft completo aporta señal más allá de la identidad del apoyo, aunque esta aportación adicional sea limitada."*

#### 5.2 Por qué los árboles superan a las redes

Un párrafo explicando la intuición técnica:

- Con ~170 campeones categóricos y señal débil (R² < 0.2), los árboles de gradient boosting explotan interacciones directamente mediante splits categóricos nativos. Cada nodo puede condicionar sobre un campeón concreto sin necesidad de representación intermedia.
- Las MLPs necesitan representar cada campeón como vector (sea one-hot de 173 dimensiones o embedding de 16 dimensiones) y aprender combinaciones a través de capas densas. Esto implica más parámetros para capturar la misma estructura, lo que en un régimen de señal débil lleva a sobreajuste más rápido. De hecho, las MLPs alcanzan su mejor epoch entre la 6 y la 18 de 150 posibles, confirmando la rapidez del sobreajuste.

#### 5.3 Limitaciones

Lista explícita y honesta. Cada limitación en 1-2 frases:

1. **Resolución temporal**: La timeline proporciona ~8 snapshots por partida en la ventana 5-12 min. Un solo evento (una muerte, un recall) puede mover el score significativamente con tan pocos puntos de datos.
2. **Etiqueta como indicador indirecto**: El score mide separación espacial, no intención estratégica. No toda separación del tirador es roaming deliberado: una muerte forzada o un colapso de la zona inferior también generan separación.
3. **Validación subjetiva**: La referencia experta fue construida por el autor a partir de conocimiento del dominio, no por un panel independiente de evaluadores. Esto introduce potencial sesgo de confirmación, aunque la correlación de 0.82 sugiere que la ordenación es razonable.
4. **Alcance del dataset**: Un servidor (EUW), una cola (Ranked Solo/Duo), un rango de nivel (alto). Los patrones podrían diferir en otros servidores, modos de juego o niveles de habilidad.
5. **Filtro de partidas caóticas**: Mitiga el ruido de partidas anómalas, pero la frontera limpia/caótica es heurística. Los umbrales (≥6 muertes, ≥5 del tirador, ≥4 del apoyo) fueron elegidos por criterio de dominio, no optimizados.

---

### 6. Conclusiones (~0.5 páginas)

Cinco conclusiones numeradas, cada una en 1-2 frases. Deben poder leerse como resumen autónomo:

- **C1**: El draft contiene señal parcial pero real sobre la movilidad temprana del apoyo. El HistGBT explica ~16% de la varianza y ordena composiciones con Spearman ~0.39.
- **C2**: El campeón de apoyo es la variable dominante. El resto del draft (tirador, matchup rival, equipo completo) añade señal adicional pero limitada.
- **C3**: El modelo supera todas las referencias group-mean OOS → combina información de los 10 slots del draft de forma no trivial.
- **C4**: Los modelos tabulares (HistGBT) superan a las redes neuronales probadas (MLP con embeddings por rol). Los embeddings no compensan su coste computacional en este régimen de señal débil.
- **C5**: El límite principal es inherente al problema: la ejecución individual introduce ~84% de varianza no capturable desde la composición previa.
- **C6**: Se descartaron sistemáticamente cuatro hipótesis alternativas (complejidad del modelo, representación, fórmula de la etiqueta, definición del fenómeno). Al construir una etiqueta más estricta basada en acciones productivas, la predictibilidad desde el draft baja — lo que confirma que el draft captura predisposición, no ejecución, y que el techo observado refleja la naturaleza del problema.

#### Trabajo futuro

Lista de extensiones posibles (4-5 líneas):
- Construir indicadores para otros roles (jungla, equipo) usando la misma metodología validada.
- Explorar resolución sub-minutal si la API lo permite en el futuro.
- Complementar el indicador espacial con indicadores basados en eventos productivos (asistencias fuera de zona, presencia en objetivos).
- Ampliar a otros servidores y rangos para evaluar la generalización de los patrones.

---

### Bibliografía (~0.5 páginas)

Formato IEEE o APA consistente. Las 12 referencias principales:

[1] Riot Games, Developer Portal.
[2] Riot Games, Data Dragon.
[3] Hitar-Garcia et al. 2023 — Win prediction.
[4] Lee et al. 2022 — DraftRec.
[5] Rama et al. 2020 — Behavioural patterns HMM.
[6] Wallner et al. 2023 — Spatio-temporal visualization.
[7] Chen et al. 2025 — Trajectory prediction.
[8] Ke et al. 2017 — LightGBM / GBT.
[9] Lundberg & Lee 2017 — SHAP.
[10] McGraw & Wong 1996 — ICC.
[11] Pedregosa et al. 2011 — scikit-learn.
[12] Paszke et al. 2019 — PyTorch.

---

## Anexos (4 páginas)

### Anexo A — Arquitecturas de modelos y detalle técnico (~1.5 páginas)

#### A.1 Tabla de hiperparámetros por modelo
Tabla con todos los HPs de cada modelo. Para las MLPs: hidden dims, dropout, weight decay, learning rate, epochs, patience, batch size. Para el GBT: max_iter, max_depth, learning rate, min_samples_leaf. También incluir las variantes de MLP que se probaron (OneHot, Shared Embeddings) pero que no se detallan en el cuerpo.

#### A.2 Diagrama de arquitectura MLP + Embeddings
Un diagrama visual que muestre las 3 variantes de MLP lado a lado: (1) OneHot → Dense, (2) Shared Embedding → Dense, (3) Per-Role Embeddings + Dot Products → Dense. Señalar los dot products explícitos y cómo se concatenan con el vector de embedding antes de las capas densas.

#### A.3 Embedding: cómo funciona
Explicación concisa para lectores no familiarizados: tabla de lookup entrenable E ∈ ℝ^{173×16}, donde cada campeón se mapea a un vector de 16 dimensiones. Backpropagation actualiza los vectores durante el entrenamiento. El cuello de botella dimensional (173 → 16) actúa como regularización implícita. Diferencia entre shared (1 tabla para todos los slots) y per-role (10 tablas, una por posición).

#### A.4 Curvas de entrenamiento
2-3 plots de train/val loss de las MLPs. Mostrar el gap train/val creciente y la epoch del mejor modelo (6-18 de 150). Estos gráficos apoyan la narrativa del §5.2 sobre el sobreajuste rápido.

---

### Anexo B — Figuras y análisis complementario (~1.5 páginas)

#### B.1 Diagrama del pipeline de datos
Diagrama visual del flujo completo: API de Riot → Raw JSON → Frame state (posiciones, experiencia por minuto) → Draft features (composición pregame) → Score v5 (etiqueta) → Splits (train/val/test) → Modelos → Evaluación.

#### B.2 Geometría del mapa
Figura de las zonas manuales del mapa con leyenda: BOT_LANE_CORE, BOT_SIDE_NEAR, RIVER_BOT, DRAGON_AREA, MID_LANE, junglas por cuadrante, bases. Esta figura es la referencia visual para entender §3.2 (qué se considera "fuera de la zona inferior").

#### B.3 Distribución de la etiqueta
Histograma del support_roam_score con las observaciones limpias y caóticas diferenciadas por color. Permite ver que la distribución está concentrada hacia la izquierda pero no colapsada, y que las caóticas tienen una distribución ligeramente distinta.

#### B.4 SHAP
Beeswarm o bar plot de importancia global SHAP. Confirma la jerarquía: apoyo aliado >> tirador aliado > apoyo rival >> resto. Opcionalmente, 1 waterfall de un caso individual representativo para mostrar cómo contribuye cada variable a una predicción concreta.

#### B.5 Referencia experta
Scatter plot de score medio observado vs. score experto por campeón (47 campeones). Color por confianza experta. La línea de referencia escalada al rango observado. Esta figura apoya la validación de §3.2.

---

### Anexo C — Prototipo aplicado (~1 página)

#### C.1 Descripción funcional
Qué hace el prototipo: acepta un draft (10 campeones por nombre o ID), carga el modelo HistGBT final, y devuelve un score continuo acompañado de una interpretación por bandas (ej. "perfil de roaming bajo / moderado / alto"). Tres modos de uso: interactivo (introduce campeones uno a uno), argumentos CLI (un solo comando), batch (lee CSV o JSON). Asume hechizos de invocador por defecto (Flash + hechizo estándar del rol) si no se especifican.

#### C.2 Ejemplo de uso
1 captura de pantalla o bloque de texto formateado con un draft concreto (ej. Bard + Ezreal vs. Leona + Jinx) y la salida del prototipo: score numérico, banda interpretativa, y comparación del perfil de roaming del apoyo aliado vs. el enemigo. La lectura debe ser comprensible para un jugador sin formación técnica.

---

## Decisiones explícitas: qué se queda y qué se descarta

### ✅ Se queda en el cuerpo

| Elemento | Ubicación |
|---|---|
| Pregunta de investigación | §1.2 |
| Dataset (383k obs, split, features) | §3.1 |
| Párrafo cualitativo + fórmula + validación experta (Spearman 0.82) | §3.2 |
| Protocolo común (features, weights, seeds, WandB) | §3.3 |
| 4 modelos bajo protocolo común | §3.4 |
| Tabla principal de resultados (multi-seed) | §4.1 |
| Honestidad MAE / compresión / Spearman como argumento principal | §4.1 |
| Ceiling OOS (corregido, no el 0.173 in-sample) | §4.2 |
| Importancia de variables + asimetría del mapa | §4.3 |
| Clean vs chaotic + auditoría de errores extremos | §4.4 |
| Investigación sistemática del techo: 4 hipótesis descartadas | §4.5 |
| Experimento v8 (etiqueta productiva baja predictibilidad) | §4.5 |
| 15 variantes de etiqueta (correlación ≥0.99) | §4.5 |
| HP search 108 configs (mejora ~0.005 Spearman) | §4.5 |
| GBT > MLP como hallazgo | §4.5 |
| Limitaciones explícitas | §5.3 |

### 📎 Se mueve a anexos

| Elemento | Destino |
|---|---|
| Arquitecturas detalladas de MLP + embeddings | Anexo A |
| Hiperparámetros de todos los modelos | Anexo A |
| Curvas de entrenamiento | Anexo A |
| Pipeline de datos (diagrama) | Anexo B |
| Geometría del mapa (figura) | Anexo B |
| SHAP (beeswarm + waterfalls) | Anexo B |
| Distribución de la etiqueta | Anexo B |
| Scatter experto | Anexo B |
| Prototipo CLI completo | Anexo C |

### ❌ Se descarta del informe final

| Elemento | Motivo |
|---|---|
| Evolución clasificación → regresión | Historia del proyecto, no resultado. Ya está en informes de progreso |
| Seguimiento de planificación | No es sección obligatoria del informe final |
| Geometría v4 y evolución a v5 | Solo se presenta la v5 final |
| OAT del cluster | No produjo resultado usable |
| Quantile transform | 1 frase: "no mejoró los resultados" |
| GBT enriched / Pair TE | Fuera del protocolo común, mencionables en 1 frase |
| Embeddings detallados (t-SNE, UMAP, vecinos) | SHAP + resumen en §4.5 bastan; detalles visuales a Anexo B |
| 40 casos de auditoría cualitativa | 2-3 frases con 1 ejemplo representativo |
| Prototipo CLI en detalle | Va a Anexo C |
| Ventanas temporales (6, 8, 10, 12, 14 min) | 1 frase: "se eligió 5-12 min tras análisis de estabilidad" |
| ICC in-sample de la tabla antigua (0.173) | **Reemplazar** por ceiling OOS correcto |
| Multi-output / jungla / equipo | 1 frase en trabajo futuro |
