# Análisis Crítico del TFG: Inferencia de Early Game en League of Legends

> [!NOTE]
> Este análisis se basa en la lectura completa de: la propuesta inicial (README), el informe de progreso I completo, el iteration log (40 iteraciones), el progreso II provisional, toda la documentación de `ProgresoActual2` (geometry v4/v5, quantile labels, variant comparison), y el código fuente de entrenamiento, etiquetado y pipeline.

---

## 1. Diagnóstico General: ¿Dónde estás?

### Lo que tienes
- **~171k partidas recolectadas** (y sigues recolectando, rumbo a 250k)
- **Pipeline completo funcional**: collector → frame-state → draft features → scores → model input → MLP → CLI prototype
- **Una etiqueta support continua** (`support_roam_score`) con buena cobertura (~337k observaciones) y correlación alta con referencia experta (Spearman 0.82)
- **Una MLP baseline** que mejora ~13% MSE sobre predecir la media (R²=0.13, Pearson=0.36)
- **Geometría v5 manual** mejorada semánticamente + exploración de etiqueta quantile
- **Prototipo terminal** funcional
- **OAT preparado** (20 runs) pero no ejecutado
- **Documentación exhaustiva** (esto es genuinamente bueno)

### Lo que NO tienes
- Resultados de OAT ejecutado
- Embeddings probados
- Etiquetas de jungla/equipo redefinidas
- Comparación MLP vs baseline con etiqueta v5/quantile
- Un modelo que realmente capture varianza significativa (R²=0.13 es muy bajo)

---

## 2. Lo que está BIEN hecho (y debes preservar)

### ✅ Decisión de cambiar de clasificación a regresión
Esto fue absolutamente correcto. La argumentación es sólida: los scores nacen continuos, discretizar pierde información, y la zona "ambiguous" era artificialmente problemática. Esto demuestra madurez metodológica.

### ✅ Separación estricta entre input pregame y target postgame
La arquitectura conceptual (draft como input, timeline como target builder) es limpia y evita data leakage. Esto está bien pensado y bien implementado.

### ✅ Split por `match_id` con `GroupShuffleSplit`
Correcto. Evita que las dos observaciones de la misma partida caigan en train y validation. En la fase final se amplía a train/val/test (70/15/15) para evitar sobreajuste implícito al iterar sobre val.

### ✅ Validación cualitativa con referencia experta
El Spearman 0.82 contra tu referencia manual es una señal positiva de que la etiqueta captura algo real. La honestidad de presentarla como "validación cualitativa de ranking, no ground truth" es acertada.

### ✅ Pipeline reproducible y documentado
El nivel de documentación (`iteration_log.md`, READMEs, design notes) es excepcionalmente bueno para un TFG. Esto facilita la reproducibilidad y demuestra rigor.

### ✅ Priorizar calidad del target antes que embeddings
Esta decisión (Decision 6 del Progreso II) es correcta. Si el target está mal, mejorar el input no sirve de nada.

---

## 3. Problemas CRÍTICOS que debes abordar

### 🔴 Problema 1: R²=0.13 es un resultado débil para un TFG

> [!CAUTION]
> Un R²=0.13 significa que tu modelo explica solo el 13% de la varianza del target. El 87% restante es ruido o señal no capturada. Si presentas esto como resultado final, un tribunal puede cuestionar la viabilidad del enfoque.

**¿Por qué ocurre?** Hay una razón fundamental que ya identificas parcialmente pero no afrontas del todo: **el draft no determina el comportamiento early-game**. El draft crea una *predisposición*, pero la ejecución depende de:
- Matchup lane-level (quién pushea primero, quién tiene prio)
- Pathing del jungla enemigo y aliado
- Decisiones humanas en tiempo real
- Estado de la wave, timings de recall, nivel de los jugadores

**Esto no invalida tu TFG**, pero necesitas ajustar cómo lo presentas y qué esperas del modelo. Tu techo predictivo desde draft puro probablemente está en torno a R²=0.15-0.25, no mucho más.

**Acción recomendada:**
1. **Calcula un "techo empírico"**: agrupa por composición completa (mismos 10 campeones, mismo side) y mide la varianza intragrupo. Si la varianza intragrupo es enorme, demuestra empíricamente que el draft no determina el roaming.
2. **Presenta esto como hallazgo, no como fracaso**: "El draft predice ~13% de la varianza del comportamiento de roaming temprano. Esto sugiere que el draft impone una predisposición medible pero limitada, coherente con la naturaleza estratégica del juego."
3. **Compara con la literatura**: los modelos de predicción de victoria desde draft suelen reportar accuracies del 55-60% (vs 50% aleatorio). Eso es también una señal débil. Tu hallazgo es coherente con el campo.

### 🔴 Problema 2: La MLP predice la media — pero NO has intentado soluciones serias

Tu MLP tiene std(predicciones) = 0.068 vs std(target) = 0.171. Predice esencialmente la media por campeón. Has documentado este problema extensamente pero las únicas soluciones planteadas son:

- OAT de hiperparámetros (no ejecutado)
- Embeddings (no ejecutado)
- RNN/LSTM (aplazado)

**Falta probar cosas mucho más simples primero:**

1. **¿Qué aprende la MLP realmente?** Haz un análisis de contribución: ¿la predicción se explica casi completamente por el campeón del support aliado? Si es así, tu MLP está aprendiendo `mean_score[champion]`, que es un lookup trivial, no ML.

2. **Feature engineering básico que no has hecho:**
   - **Interacciones de draft**: matchup support vs support, matchup support vs ADC enemigo, sinergia support-ADC aliados. Estos son features que un jugador usa mentalmente para decidir si roamear.
   - **Arquetipos como features**: en vez de One-Hot puro de 170+ campeones, añade features categóricas como `support_archetype` (engage, enchanter, mage, etc.) que ayuden a generalizar.
   - **Densidad de CC, engage potencial, poke**: features derivadas del kit del draft que son relevantes para la decisión de roaming.

3. **Un modelo mucho más simple como comparación**: ¿cuánto da un `mean(score) por campeón support`? Si da R²=0.11 y tu MLP da 0.13, la MLP aporta casi nada sobre un lookup básico. Esto es un dato **crítico** que debes reportar.

### 🔴 Problema 3: Demasiado tiempo invertido en la etiqueta, poco en el modelo

> [!WARNING]
> Has iterado la etiqueta 5 veces (v1→v2→v3→v4 design→v5 geometry→v5 quantile), has construido geometría manual, anotación de mapa, transformación quantile, calibración gamma... pero solo has entrenado **una MLP** con la configuración por defecto.

La calidad de la etiqueta es importante, pero la iteración v2→v3→v5 produjo cambios mínimos:
- Correlación v5 vs v3 a nivel fila: **0.94**
- Mean delta: **+0.02**

Esto significa que tus mejoras de geometría apenas cambiaron la etiqueta. El problema no está en si "bot" incluye Dragon Area o no. El problema está en que **el modelo es demasiado simple para capturar la señal disponible**, y no has explorado alternativas.

---

## 4. Small Wins de ALTO impacto (implementables en días)

### 🏆 Win 1: Baseline de media por campeón (1-2 horas)

```python
# Calcula en train
champion_means = df_train.groupby('ally_utility_champion_id')['support_roam_score'].mean()
# Predice en val (desarrollo) o test (evaluación final)
y_pred_baseline = df_eval['ally_utility_champion_id'].map(champion_means).fillna(y_train.mean())
# Compara métricas
```

Si esto da R²=0.10-0.12, tu MLP solo aporta un 1-3% marginal, lo que revela que no está aprendiendo interacciones relevantes. Si da R²=0.05, tu MLP sí está capturando algo de composición. **Este dato es esencial para el informe.**

### 🏆 Win 2: Gradient Boosting como comparación (2-3 horas)

```python
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
```

Un GBT sobre features codificadas con `OrdinalEncoder` + `categorical_features=True` captura interacciones automáticamente sin One-Hot. Si un GBT da R²=0.18 donde tu MLP da 0.13, la narrativa cambia: "GBT captura interacciones de draft mejor que la MLP tabular".

`HistGradientBoostingRegressor` de sklearn es rápido y adecuado para este volumen de datos.

### 🏆 Win 3: Importancia de features (1-2 horas)

Con un GBT entrenado, usas `sklearn.inspection.permutation_importance` sobre val para descubrir qué variables del draft realmente predicen roaming. Esto da contenido para el TFG sin necesidad de cluster.

### 🏆 Win 4: Tabla de "techo predictivo" (2-3 horas)

Agrupa por las combinaciones más comunes de (support_champion + ADC_champion + side) y calcula:
- Varianza intra-grupo del score
- Varianza inter-grupo
- ICC (Intraclass Correlation Coefficient)

Esto cuantifica empíricamente cuánto del roaming es "predecible desde composición" vs "ruido de partida". Es un resultado metodológico muy defendible.

### 🏆 Win 5: Ejecutar OAT inmediatamente en local (si el cluster no funciona)

20 runs de MLP en 337k filas con One-Hot de 1796 dims en CPU tarda probablemente ~4-8 horas. No necesitas cluster para esto. Ejecuta en local con batch más pequeño si hace falta. **No tener el OAT ejecutado para el Progreso II es un hueco importante.**

---

## 5. Cambios de Enfoque Recomendados

### 📐 Reencuadrar el objetivo del TFG

Tu TFG no debería ser "construir el mejor predictor de roaming". Debería ser:

> **"Cuantificar hasta qué punto el draft predice comportamiento temprano en LoL, y construir un sistema interpretable que aproveche esa señal"**

Este reencuadre convierte un R²=0.13 de "resultado pobre" a "hallazgo científico legítimo": **el draft impone una predisposición limitada pero medible**. La contribución del TFG pasa a ser:

1. **Metodológica**: diseño de una etiqueta continua observacional validada cualitativamente
2. **Empírica**: cuantificación de la señal predictiva del draft (con baselines, techo empírico, importancia de features)
3. **Aplicada**: prototipo que traduce la señal en lectura interpretable

### 📐 Abandonar jungla y equipo — centrar el TFG en support

> [!IMPORTANT]
> El planning actual prevé redefinir etiquetas de jungla (25/05-31/05) y equipo (01/06-07/06). **No lo hagas.** No hay tiempo suficiente para hacer tres etiquetas bien. Es mejor hacer UNA etiqueta excelentemente validada, con múltiples modelos comparados, feature engineering serio, y análisis de techo predictivo, que tres etiquetas superficiales.

**Argumento ante el tribunal**: "Se decidió profundizar en la tarea de support porque es la más clara semánticamente, tiene la validación experta más fuerte, y permite un análisis más riguroso. Las tareas de jungla y equipo quedan como trabajo futuro con el framework ya validado."

### 📐 Priorizar diversidad de modelos sobre diversidad de etiquetas

En vez de v5, v5-quantile, v5-gamma, v4... usa UNA etiqueta final y compara:

| Modelo | Qué prueba |
|--------|-----------|
| Media por campeón | Baseline trivial |
| MLP One-Hot (actual) | Baseline neural |
| GBT / HistGBT | Captura automática de interacciones |
| MLP con feature engineering | Matchups, arquetipos, sinergias |
| MLP con embeddings aprendidos | Representación densa de campeones |

Esta tabla es **mucho más defendible** que 5 variantes de la misma MLP con 5 variantes de etiqueta.

---

## 6. Problemas menores pero importantes

### ⚠️ Transformación quantile: precaución y oportunidad

La transformación quantile debe fittearse SOLO en train (ya recogido en `decisions.md` y `technical_spec.md`). Bien hecha, no es solo una precaución sino un **eje experimental legítimo**: comparar modelos con target raw vs quantile demuestra si la distribución del target condiciona el aprendizaje. Para comparar métricas entre ambas escalas, usar Spearman (invariante a transformaciones monótonas) o inverse-transform las predicciones quantile a escala raw.

### ⚠️ Referencia experta manual: riesgo de circularidad

Tu referencia experta la hiciste tú mismo. Esto no es malo (no hay alternativa obvia), pero debes:
1. Declarar explícitamente que es subjetiva
2. Idealmente, pedir a otra persona con conocimiento del juego que la valide independientemente
3. No usarla para calibrar la etiqueta, solo para validar ranking

### ⚠️ Sobreingeniería del pipeline

40 iteraciones, 3 carpetas de progreso, scripts de sync a cluster, OAT framework... La infraestructura es impresionante pero desproporcionada para los resultados obtenidos. **Un tribunal valora resultados y análisis, no la complejidad del pipeline.**

### ⚠️ Recolección continua a 250k: ¿para qué?

Tienes 171k partidas. Si la señal es débil con 171k, no será mucho mejor con 250k. El cuello de botella no es la cantidad de datos sino la riqueza del input (solo draft) y la naturaleza del problema. **No pierdas más tiempo recolectando; trabaja con lo que tienes.**

---

## 7. Planning: original vs realidad vs plan revisado

### 7.1 Qué planteaba el Informe de Progreso I (27/04)

| Periodo | Objetivo previsto | Criterio de éxito |
|---------|-------------------|-------------------|
| 28/04-03/05 | Tuning OAT conjunto: MLP + etiqueta support | Tabla comparativa por `val_mse`, ranking de heurísticas |
| 04/05-10/05 | Embeddings y feature enrichment inicial | Comparación OneHot vs enriched/embeddings |
| 11/05-17/05 | Refinar representación y cerrar soporte | Feature set candidato + decisión sobre representación |
| 18/05-24/05 | Informe de Progreso II | Documento listo con tuning + embeddings |
| 25/05-31/05 | Redefinir etiqueta de jungla | Label continua de jungla + plots de salud |
| 01/06-07/06 | Redefinir etiqueta de equipo | Label continua de equipo + plots de salud |
| 08/06-14/06 | Integración multi-output y decisión RNN/GRU/LSTM | Modelo candidato, decisión secuencial |
| 15/06-21/06 | Prototipo terminal e interpretación | CLI usable con lectura interpretable |
| 22/06-28/06 | Cierre final | Memoria + presentación |

### 7.2 Qué pasó realmente (28/04 → 09/05)

| Previsto | Hecho | Veredicto |
|----------|-------|-----------|
| Tuning OAT ejecutado | OAT **preparado** (manifest de 20 runs) pero **no ejecutado** — cluster no disponible | ❌ Resultado no obtenido, infraestructura lista |
| Embeddings / feature enrichment | **No se hizo**. Se sustituyó por refinamiento de geometría (v5 manual) y exploración de etiqueta quantile | ❌ Desvío; la geometría v5 correlaciona 0.94 con v3 → impacto mínimo |
| — (no previsto) | **Prototipo terminal** adelantado desde junio | ✅ Ganancia real: entregable aplicado disponible antes de tiempo |
| — (no previsto) | **Limpieza del repositorio**: eliminación de artefactos viejos, separación ProgresoActual / ProgresoActual2 | ⚠️ Útil pero no produce resultados experimentales |
| — (no previsto) | **Transformación quantile zero-preserved** explorada | ✅ Buena idea; faltaba entrenar con ella para evaluar impacto |

**Diagnóstico**: se avanzó en infraestructura y calidad del target, pero no se
produjeron resultados experimentales nuevos (ni OAT, ni modelos alternativos,
ni baselines triviales). El TFG tiene hoy los mismos resultados que el
27/04: una MLP con R²=0.13.

### 7.3 Qué sigue en pie y qué no

| Bloque del Informe I | ¿Sigue en pie? | Razón |
|----------------------|:--------------:|-------|
| Tuning OAT de MLP + etiqueta | ⚠️ **Parcialmente** | Los 20 runs preparados pueden ejecutarse en local (no necesitan cluster para ~337k filas). Pero el OAT ya no es la prioridad: primero hay que tener baselines triviales y GBT para contextualizar. Sin esos datos, el OAT no aporta narrativa |
| Embeddings / feature enrichment | ✅ **Sí, pero más tarde** | Sigue siendo relevante como paso 4-5, no como paso 1. Primero hay que demostrar que la MLP One-Hot supera (o no) una baseline trivial |
| Refinar representación y cerrar support | ✅ **Sí, redefinido** | "Cerrar support" ahora incluye: baselines, GBT, techo empírico, feature importance. No solo elegir una etiqueta |
| Informe de Progreso II | ✅ **Sí** | Fecha: 24/05. El contenido cambia: en vez de "tuning completado", será "comparación de modelos y cuantificación de señal" |
| Redefinir etiqueta de jungla | ❌ **No** | No hay tiempo para hacer dos etiquetas nuevas bien. Profundizar en support produce un TFG más fuerte que tres tareas superficiales |
| Redefinir etiqueta de equipo | ❌ **No** | Misma razón. Quedan como trabajo futuro |
| Integración multi-output | ❌ **No** | Sin jungla/equipo no hay multi-output. Se presenta como trabajo futuro con framework ya validado |
| Decisión RNN/GRU/LSTM con tutor | ⚠️ **Depende** | Si hay tiempo después de embeddings (semana 25/05-01/06), puede explorarse como experimento adicional. No es prioritario |
| Prototipo terminal | ✅ **Ya hecho** | Adelantado. Solo necesita actualizarse con el mejor modelo final |
| Cierre final | ✅ **Sí** | Se mantiene la fecha del 28/06 |

### 7.4 Planning revisado (desde hoy, 09/05)

| Periodo | Objetivo | Entregable | Relación con plan original |
|---------|----------|------------|---------------------------|
| **09-11/05** | Esperar 200k + preparar dataset final | `final/data/training/{train,val,test}.parquet` con split persistido y columnas quantile | **Nuevo**: paso inexistente en el plan original |
| **12-14/05** | Baselines críticas | Baseline media-por-campeón + HistGBT + techo empírico + feature importance | **Nuevo**: el plan original no preveía baselines triviales ni GBT |
| **15-18/05** | MLP OneHot reproducida + comparación raw vs quantile | MLP entrenada en `final/`, tabla comparativa parcial con 3+ modelos | **Fusiona**: tuning OAT + refinar representación → ahora es comparación de modelos |
| **19-24/05** | Informe de Progreso II | Narrativa: señal del draft cuantificada, comparación de modelos, decisiones | **Se mantiene**: misma fecha, contenido más rico |
| **25/05-01/06** | Feature engineering + embeddings | Matchup features, arquetipos, embedding layer aprendido | **Se mantiene con delay**: era 04/05-17/05, ahora 25/05-01/06 |
| **02-08/06** | Consolidación de resultados | Tabla final de modelos en test, prototipo terminal actualizado | **Fusiona**: lo que antes era "integración multi-output" ahora es "consolidación support-only" |
| **09-14/06** | Propuesta de informe final | Estructura completa de la memoria con todas las figuras | **Se mantiene**: misma fecha |
| **15-21/06** | Redacción de memoria | Texto final | **Se mantiene**: misma fecha |
| **22-28/06** | Presentación + revisión | Slides, ensayo, entrega | **Se mantiene**: misma fecha |

### 7.5 Cambios clave respecto al plan original

1. **Se eliminan 3 semanas** dedicadas a jungla (25/05-31/05), equipo
   (01/06-07/06) e integración multi-output (08/06-14/06). Ese tiempo se
   reasigna a baselines, diversificación de modelos y consolidación.

2. **Se añaden baselines triviales y GBT** como paso previo a cualquier otra
   cosa. Esto no existía en el plan original y es el cambio más importante:
   sin contexto sobre qué aporta la MLP sobre un lookup por campeón, los
   resultados actuales no son defendibles.

3. **Se adelanta el prototipo terminal** (ya hecho) y se retrasan los
   embeddings (de semana 2 a semana 4). Priorizar baselines y comparación de
   modelos es más urgente que enriquecer el input.

4. **Se reencuadra el TFG** de "sistema multi-output de predicción de early
   game" a "cuantificación de la señal predictiva del draft sobre roaming de
   support". Esto cambia la narrativa del Informe II y de la memoria final.

5. **El OAT pasa de ser un bloque independiente a ser absorbido** por la
   comparación de modelos. Los hiperparámetros se prueban como parte natural
   de entrenar cada modelo, no como framework separado.

---

## 8. Resumen Ejecutivo

### Lo bueno
Tu TFG tiene una **base sólida**: buen pipeline, buena documentación, etiqueta razonable, separación limpia entre input y target. El cambio de clasificación a regresión fue correcto. El prototipo terminal es un buen entregable.

### Lo que necesita cambio urgente
1. **Ejecuta baselines triviales** (media por campeón, GBT) para contextualizar la MLP
2. **Deja de iterar la etiqueta** — la v5 está bien, úsala y avanza
3. **Diversifica modelos**, no etiquetas
4. **Calcula el techo empírico** para defender un R² bajo
5. **Abandona jungla/equipo** — no hay tiempo
6. **Reencuadra el TFG como cuantificación de señal**, no como maximización de R²

### Diferencia entre un TFG malo y uno bueno con este material

| TFG Malo | TFG Bueno |
|----------|-----------|
| "Entrené una MLP con R²=0.13" | "Cuantifiqué que el draft aporta ~13-18% de varianza explicada sobre roaming temprano, comparable con la señal reportada en la literatura de predicción de victoria" |
| "Probé 5 variantes de etiqueta" | "Validé la etiqueta cualitativamente (Spearman 0.82 vs experto) y demostré que cambios de geometría menores no alteran significativamente la señal" |
| "No tuve tiempo para jungla/equipo" | "Profundicé en support para obtener conclusiones metodológicas transferibles a otras tareas" |
| "La MLP predice la media" | "Demostré empíricamente que la mejora sobre una baseline trivial de lookup por campeón es limitada, sugiriendo que el draft impone predisposiciones a nivel de campeón más que interacciones complejas" |

---

---

## 9. Explicabilidad y auditoria cualitativa

La comparacion de modelos debe presentar la MLP con el mismo peso narrativo que
los modelos tabulares: fue la primera hipotesis acordada con el tutor y sirve
como referencia neural del proyecto. La lectura final no es "la MLP no importa",
sino "la MLP, los baselines y el HistGBT cuantifican juntos cuanta senal
pre-game hay en el draft".

Para explicabilidad se usa el `HistGBT` base en escala raw porque es mas
interpretable: usa solo draft pre-game codificado con las 31 features canonicas.

### SHAP como explicabilidad asociativa

`final/scripts/08_shap_analysis.py` genera importancia global SHAP, summary
plots, dependencias categoricas para support/ADC aliado y waterfalls locales.
El script intenta `TreeExplainer`, pero valida aditividad y cae a
`PermutationExplainer` si la combinacion SHAP/sklearn no conserva
`prediccion = base + suma(SHAP)`.
La interpretacion correcta es: el modelo aprende predisposiciones de draft,
especialmente identidad del support, ADC aliado/enemigo y contexto de botlane.
No debe presentarse como causalidad del campeon ni como orden semantico entre
IDs, porque las categorias se codifican ordinalmente para el estimador.

### Auditoria cualitativa consolidada

`final/scripts/09_qualitative_case_audit.py` sustituye los analisis separados de
errores, diagnostico de etiqueta y contexto raw. El script exporta 20 mayores
errores y 20 menores errores, estos ultimos estratificados por score real. Para
cada partida une prediccion, etiqueta, draft completo, componentes de score,
frames minuto 5-12, eventos reales minuto 0-12 y mapas cronologicos support/ADC.

El run completo genero 40 casos, 1475 eventos de timeline, 280 frames de
etiqueta y 40 mapas + 40 timelines. La reconstruccion de etiqueta coincide
exactamente con los scores guardados (`max_score_reconstruction_delta = 0.0` y
`max_raw_score_reconstruction_delta = 0.0`).

El hallazgo principal es que 17/20 top errores estan marcados como
`chaotic_early_game`: muchas muertes o eventos tempranos en bot generan
separacion support-ADC que el draft no puede anticipar. En contraste, los bottom
errores muestran casos donde el modelo acierta scores bajos, medios y altos.

### Uso recomendado en el informe

1. Incluir `shap_summary_bar.png` como figura de interpretabilidad global.
2. Incluir una dependencia categorica de support o ADC aliado para mostrar
   lectura de dominio sin tratar IDs como continuos.
3. Elegir 2-3 casos de `case_notes.md` y abrir sus mapas en `case_plots/` para
   comprobar visualmente posiciones, zonas y orden temporal.

### Lectura para la memoria

No conviene llamar al target "roam real" en todos los casos. La lectura mas
precisa es `roam-like displacement` o separacion support-ADC: el score captura
presencia fuera de contexto bot, distancia al ADC y gap de XP. Eso incluye roams
limpios, pero tambien colapsos de botlane, muertes y resets que separan a los
jugadores.

Ejemplos defendibles:

- `EUW1_7831489390`: Yuumi + Smolder vs Pyke + Velkoz. Predicho 0.209, real
  1.000. Antes del minuto 12, Yuumi muere 4 veces y Smolder 7; el timeline
  muestra kills repetidas de Velkoz/Pyke sobre la botlane aliada y separacion
  sostenida support-ADC. Es un outlier real de snowball/caos temprano.
- `EUW1_7706461344`: Yuumi + Zeri vs Sona + Lucian. Predicho 0.174, real 0.930.
  Zeri muere 6 veces y Yuumi 3 antes de minuto 12; la KDA final de la botlane
  aliada es 0/5/0 y 0/9/0. La etiqueta alta refleja colapso de botlane, no una
  predisposicion de draft capturable pre-game.
- `EUW1_7708715292`: Senna + Caitlyn vs Blitzcrank + Tristana. Predicho 0.310,
  real 1.000. Senna muere 6 veces antes de minuto 12 pero tambien asiste kills;
  es una partida de fights constantes que separan al support de la posicion
  esperada junto al ADC.

Lectura para la memoria: el modelo aprende predisposiciones del draft, pero los
errores extremos demuestran varianza no observable pre-game. Esto refuerza, no
debilita, la conclusion central: el draft predispone, pero no determina.

> [!TIP]
> **Próximos 3 días**: Implementa las 5 small wins listadas en la sección 4. Son todas ejecutables en local, no requieren cluster, y transforman la calidad del análisis. Especialmente Win 1 (media por campeón) y Win 2 (GBT). Si esos resultados son mejores que la MLP, tienes una narrativa mucho más interesante.
