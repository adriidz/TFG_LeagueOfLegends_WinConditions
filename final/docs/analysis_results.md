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
Correcto. Evita que las dos observaciones de la misma partida caigan en train y validation.

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
# Predice en validation
y_pred_baseline = df_val['ally_utility_champion_id'].map(champion_means).fillna(y_train.mean())
# Compara métricas
```

Si esto da R²=0.10-0.12, tu MLP solo aporta un 1-3% marginal, lo que revela que no está aprendiendo interacciones relevantes. Si da R²=0.05, tu MLP sí está capturando algo de composición. **Este dato es esencial para el informe.**

### 🏆 Win 2: Gradient Boosting como comparación (2-3 horas)

```python
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
```

Un GBT sobre features ordinally-encoded (no One-Hot) suele capturar interacciones automáticamente. Si un GBT da R²=0.18 donde tu MLP da 0.13, la narrativa cambia radicalmente: "GBT captura interacciones de draft mejor que la MLP tabular".

**Esto es un paper-quality result** y cuesta muy poco. `HistGradientBoostingRegressor` de sklearn maneja categorías nativamente y es rápido.

### 🏆 Win 3: Importancia de features (1-2 horas)

Con un GBT entrenado, extraes `feature_importances_` y descubres qué variables del draft realmente predicen roaming. Esto da contenido para el TFG sin necesidad de cluster.

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

### ⚠️ Transformación quantile: nota de precaución

La transformación quantile global (fitteada sobre todo el dataset) introduce data leakage si no se refittea solo en train. Ya lo notas en la documentación, pero si la usas en resultados finales, **debes fitear solo en train**. Si no lo haces, el tribunal lo puede señalar.

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

## 7. Planning Sugerido hasta Entrega Final (28/06)

| Periodo | Objetivo | Entregable |
|---------|----------|------------|
| **09-11/05** | Baselines críticas | Media-por-campeón baseline, GBT comparativo, tabla de techo predictivo |
| **12-18/05** | Feature engineering + modelos | Matchup features, arquetipos, comparación MLP vs GBT vs enriched |
| **19-24/05** | Informe Progreso II | Narrativa: señal limitada pero real, comparación de modelos, decisiones |
| **25/05-01/06** | Embeddings de campeón | Embedding layer aprendido, comparar contra One-Hot |
| **02-08/06** | Consolidación de resultados | Mejor modelo final, prototipo terminal actualizado |
| **09-14/06** | Propuesta de informe final | Estructura completa de la memoria |
| **15-21/06** | Redacción de memoria | Texto final con figuras y tablas |
| **22-28/06** | Presentación + revisión | Slides, ensayo de presentación, entrega |

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

> [!TIP]
> **Próximos 3 días**: Implementa las 5 small wins listadas en la sección 4. Son todas ejecutables en local, no requieren cluster, y transforman la calidad del análisis. Especialmente Win 1 (media por campeón) y Win 2 (GBT). Si esos resultados son mejores que la MLP, tienes una narrativa mucho más interesante.
