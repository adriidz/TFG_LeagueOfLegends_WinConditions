# Resumen del TFG — Para email al tutor

> [!NOTE]
> Este documento resume todo el trabajo realizado en el TFG desde el Informe de Progreso I hasta hoy (19 mayo 2026). Está pensado como referencia para que puedas redactar un email conciso a tu profesor, centrado en **qué se ha hecho**, **qué se ha descubierto** (sobre todo limitaciones) y **qué dudas tienes sobre la dirección del trabajo**.

---

## 1. Recordatorio del objetivo

**Tema**: Inferir patrones de early-game en League of Legends a partir de información pregame (draft).

**Caso concreto**: Predecir la tendencia de roaming del support durante el early-game (minutos 5-12) usando únicamente la composición de campeones seleccionados antes de la partida.

**Etiqueta**: `support_roam_score` — variable continua [0, 1] que mide separación observada support-ADC (posición fuera de bot, distancia al ADC, gap de XP). Validada contra referencia experta con Spearman 0.82.

---

## 2. Cronología de lo hecho

### Informe de Progreso I (27 abril)
- Pipeline completo: recolección → frame-state → draft features → scores → modelo.
- Cambio de clasificación discreta a **regresión continua** (decisión clave).
- Primera MLP OneHot entrenada: R² ≈ 0.13, Pearson ≈ 0.36.
- Prototipo de terminal (CLI) adelantado.
- Planning: OAT tuning → embeddings → jungla → equipo → multi-output → memoria.

### Revisión crítica post-Informe I (9 mayo)
- Documento `analysis_results.md`: diagnóstico exhaustivo.
- Conclusión: R²=0.13 sin contexto de baselines no es defendible. Faltaban baselines triviales, techo empírico y diversidad de modelos.
- **Decisión importante**: abandonar jungla/equipo/multi-output. Centrar el TFG en support-only para obtener conclusiones rigurosas y profundas.
- Reencuadre: el TFG pasa de "construir el mejor predictor" a "**cuantificar hasta qué punto el draft predice comportamiento temprano**".

### Fase final: lo implementado (9 mayo → hoy)

| Qué | Script(s) | Resultado clave |
|---|---|---|
| Dataset final | `01_prepare_final_dataset.py` | 383k obs, split train/val/test persistido por match_id |
| Refinamiento etiqueta v5 | scripts de geometría/score v5 | score continuo basado en outside/far/xp_gap, minutos 5-12 |
| OAT preparado | `ProgresoActual/OAT/support_oat_full_m12/experiments/runs_manifest.csv` | 20 runs diseñadas; ejecución completa bloqueada por cluster |
| Baseline Global Mean | `02_baseline_champion_mean.py` | R² = 0.000 (referencia nula) |
| Baseline Champion Mean | `02_baseline_champion_mean.py` | R² = 0.125 (solo identidad del support) |
| HistGBT base | `03_train_gbt.py` | **R² = 0.160**, Spearman = 0.387 |
| HistGBT + arquetipos | `03b_train_gbt_enriched.py` | R² = 0.161 — no mejora |
| HistGBT + Pair TE | `03c_train_gbt_interactions.py` | R² = 0.161 — mejora marginal |
| MLP OneHot | `04a_train_mlp_onehot.py` | R² ≈ 0.140, Spearman ≈ 0.364 |
| MLP Embeddings (compartido) | `04b_train_mlp_embed.py` | R² = 0.150, Spearman = 0.376 |
| MLP Per-Role + Interactions | `04c_train_mlp_per_role.py` | R² = 0.154, Spearman = 0.381 |
| Techo empírico (ICC / media por grupo) | `05_empirical_ceiling.py` | **R² ≈ 0.173** (botlane support-ADC + lado) |
| Feature importance | `06_feature_importance.py` | Campeón support aliado domina |
| Comparación de modelos | `07_model_comparison.py` | Tabla completa con métricas tolerantes |
| SHAP | `08_shap_analysis.py` | Importancia global, beeswarm, dependencias |
| Auditoría cualitativa | `09_qualitative_case_audit.py` | 40 casos, 17/20 top errors = partidas caóticas |
| Chaos filtering | `16_add_chaos_filter_weights.py` | chaos_flag + sample_weight para reducir ruido |
| Label variant sweep | `14_train_label_variant_sweep.py` | 15 variantes de la fórmula → correlación ≥0.99 con v5 |
| Análisis de embeddings | `17_embedding_analysis.py` | No clusters por arquetipo, pero gradiente de roam en t-SNE |
| **HP search MLP** (en curso) | `18_mlp_hp_search.py` | 80/108 configs evaluadas; **Spearman ≈ 0.377 (mejor) vs 0.372 (default)** |

---

## 3. Resultados y experimentos principales

### Tabla comparativa final (test set, escala raw)

| Modelo | R² | Spearman | MAE |
|---|---|---|---|
| **ICC Ceiling** (techo empírico/práctico) | **0.173** | — | — |
| HistGBT + Pair TE | **0.161** | **0.388** | **0.141** |
| HistGBT | 0.160 | 0.387 | 0.141 |
| MLP Per-Role + Interactions | 0.154 | 0.381 | 0.141 |
| MLP Embed (compartido) | 0.150 | 0.376 | 0.142 |
| MLP OneHot | ~0.140 | 0.364 | 0.142 |
| Champion Mean | 0.125 | 0.336 | 0.144 |
| Global Mean | 0.000 | — | 0.155 |

**Lectura**: el HistGBT está al **93% del techo empírico ICC**. Queda ~1 punto de R² de mejora práctica si se usan solo variables de draft puro, así que la evidencia apunta a que el límite principal no es la arquitectura sino la información disponible antes de la partida.

### Qué se ha probado y qué significa cada experimento

Esta es la parte que conviene transmitir al tutor, porque lo último que conoce es el Informe de Progreso I: entonces había una primera MLP, una etiqueta continua prometedora y un plan bastante abierto. Desde entonces el trabajo se ha convertido en una evaluación más completa del caso support-only. La pregunta ya no es solo "¿puedo entrenar una red?", sino "¿cuánta señal aporta realmente el draft para anticipar roaming de support?".

#### Dataset final y split por partida

**Qué se hizo**: se construyó un dataset final con unas 383k observaciones y splits train/validation/test persistidos por `match_id`.

**Cómo se hizo**: se juntaron las features de draft con la etiqueta `support_roam_score`, manteniendo la separación por partida para evitar que información de una misma partida aparezca a la vez en entrenamiento y test.

**Qué significa**: esto convierte los resultados posteriores en una comparación más limpia. Antes del Informe I había una prueba de concepto; ahora hay una base experimental estable para comparar modelos, baselines y análisis.

#### Refinamiento de la etiqueta `support_roam_score`

**Qué se hizo**: se consolidó la etiqueta final de roaming del support, centrada en minutos 5-12 y basada en separación respecto al contexto de botlane.

**Cómo se hizo**: la etiqueta combina tres señales observadas en los snapshots de Riot: si el support está fuera de la zona de bot, si está lejos del ADC y si aparece un gap de experiencia entre support y ADC. La versión final usa la geometría v5 del mapa y una transformación gamma para obtener una escala continua más útil.

**Qué significa**: la etiqueta es más informativa que una clase discreta tipo "roamea/no roamea", pero sigue teniendo una limitación importante: mide separación observada, no intención. Por eso puede capturar tanto roams reales como situaciones caóticas donde la botlane se rompe.

#### Preparación del OAT de etiqueta y MLP

**Qué se hizo**: se preparó un experimento OAT ("one-at-a-time") para variar de forma controlada pesos de etiqueta, ventana temporal e hiperparámetros de la MLP.

**Cómo se hizo**: el manifest previsto contiene 20 runs: 5 variantes de pesos de etiqueta, 9 ventanas temporales y 6 cambios de hiperparámetros. La idea era cambiar una dimensión cada vez y mantener el resto fijo para poder atribuir cualquier mejora a una causa concreta.

**Qué significa**: este experimento estaba alineado con lo acordado tras el Informe I, pero quedó parcialmente bloqueado por la disponibilidad del cluster. En lugar de detener el proyecto esperando al OAT completo, se priorizaron experimentos ejecutables localmente con más valor metodológico inmediato: baselines, GBT, techo empírico, SHAP y análisis de errores. En la narrativa conviene presentarlo como diseño preparado y parcialmente absorbido por los experimentos finales, no como un resultado cerrado.

#### Baselines: media global y media por campeón

**Qué se hizo**: se añadieron dos referencias simples. La primera predice siempre la media global del score. La segunda predice la media histórica del campeón support aliado.

**Cómo se hizo**: para cada campeón de support se calculó su `support_roam_score` medio en train y se usó ese valor como predicción en validación/test.

**Qué significa**: esta comparación es importante porque evita vender como "modelo inteligente" algo que quizá solo esté aprendiendo que Pyke, Bard o Rakan tienden a moverse más que otros supports. La media global da R²=0.000, mientras que la media por campeón llega a R²≈0.125. Eso muestra que hay señal real en el draft, pero también que gran parte de la señal está concentrada en la identidad del support.

#### HistGBT: modelo tabular fuerte

**Qué se hizo**: se entrenó un `HistGradientBoostingRegressor` como benchmark tabular más fuerte que la MLP inicial.

**Cómo se hizo**: se usaron las variables disponibles antes de la partida: campeones aliados/enemigos, lado del mapa y features de draft. También se probaron variantes con arquetipos y con interacciones de pareja/matchup.

**Qué significa**: el HistGBT alcanza R²≈0.160-0.161 y Spearman≈0.388. Es el mejor modelo práctico hasta ahora. Además, las variantes enriquecidas apenas mejoran, lo que sugiere que añadir arquetipos manuales o target encoding de parejas no cambia sustancialmente el techo de señal.

#### MLP OneHot, embeddings y embeddings por rol

**Qué se hizo**: se continuó la línea acordada en el Informe I: probar redes neuronales para representar el draft. Se evaluaron tres familias: MLP con one-hot, MLP con embeddings compartidos y MLP con embeddings separados por rol más interacciones explícitas.

**Cómo se hizo**: la versión one-hot codifica cada campeón como variable categórica expandida. La versión de embeddings aprende una representación densa de los campeones. La versión per-role intenta distinguir el significado de un campeón según aparezca como support, ADC, jungla, etc., y añade señales de interacción entre botlanes/matchups.

**Qué significa**: las MLPs mejoran respecto a la primera prueba, pero no superan al HistGBT. La mejor MLP queda en R²≈0.154 frente a R²≈0.160 del GBT. La lectura sensata no es que "la MLP no sirve", sino que para este problema tabular, con señal débil y muchas categorías, los árboles parecen capturar mejor las regularidades disponibles.

#### HP search de la MLP

**Qué se hizo**: se lanzó una búsqueda de hiperparámetros para comprobar si la MLP estaba limitada por una mala configuración.

**Cómo se hizo**: se probaron combinaciones de tamaño de capas, dropout, learning rate y weight decay sobre la MLP per-role.

**Qué significa**: las mejoras son pequeñas. La mejor configuración mejora alrededor de +0.005 en Spearman de validación frente al default. Esto refuerza que el problema no parece estar en el tuning fino, sino en la cantidad de señal pregame disponible.

#### Techo empírico ICC / media por grupo

**Qué se hizo**: se estimó un techo empírico para responder a una duda clave: si el modelo alcanza R²≈0.16, ¿es porque el modelo es insuficiente o porque el draft solo permite anticipar una parte pequeña del comportamiento temprano?

**Cómo se hizo**: se agruparon partidas que comparten información pregame comparable y se midió cuánta variabilidad del `support_roam_score` se explica por el grupo frente a cuánta queda dentro del grupo. Se probaron agrupaciones como mismo campeón support, support+lado, pareja support-ADC, pareja support-ADC+lado y matchup support aliado vs support enemigo. Para cada agrupación se calculó ICC / descomposición de varianza y también el R² de predecir la media histórica del grupo.

**Qué significa**: agrupar solo por campeón support explica alrededor del 12% de la varianza. Con pareja support-ADC sube a ~16%, y con pareja support-ADC + lado llega a R²≈0.173. Por eso el HistGBT con R²≈0.161 se interpreta como cercano al techo práctico: captura casi toda la señal estable que aparece en el draft. Lo que queda fuera parece depender de ejecución en partida: wave state, recalls, muertes tempranas, pathing del jungla, prioridad de mid, visión y coordinación.

#### Feature importance y SHAP

**Qué se hizo**: se analizó qué variables usa el modelo para predecir.

**Cómo se hizo**: se calcularon importancias por permutación y explicaciones SHAP sobre el modelo GBT.

**Qué significa**: el campeón support aliado domina la predicción, seguido por el ADC aliado y el support enemigo. Esto es coherente con el dominio: el roaming del support depende primero de quién es el support, después de con qué ADC juega y contra qué tipo de botlane/matchup se enfrenta. También ayuda a justificar que el modelo no está aprendiendo una señal absurda o accidental.

#### Auditoría cualitativa de errores

**Qué se hizo**: se revisaron manualmente casos donde el modelo falla mucho.

**Cómo se hizo**: se seleccionaron errores grandes y se miró el contexto de partida: muertes tempranas, botlane colapsada, trayectorias de support/ADC y eventos entre minuto 5 y 12.

**Qué significa**: muchos errores grandes no parecen deberse a un patrón de draft mal aprendido, sino a partidas caóticas. Por ejemplo, si la botlane muere varias veces o se rompe el estado normal de línea, el support puede aparecer separado del ADC por razones que no equivalen a un roam planificado. Esto apoya la tesis de que el draft predice predisposición, no ejecución exacta.

#### Chaos filtering y pesos de muestra

**Qué se hizo**: se creó una señal de "partida caótica" para reducir el peso de ejemplos donde la etiqueta puede mezclar roaming real con colapso de partida.

**Cómo se hizo**: se marcaron casos con muchas muertes de botlane o patrones de eventos incompatibles con un roam limpio, y se asignó menor peso a esos ejemplos durante el entrenamiento/análisis.

**Qué significa**: no elimina el problema de fondo, pero lo hace explícito. La etiqueta mide separación support-ADC, no intención. El filtro ayuda a explicar una limitación metodológica importante y da una forma razonable de mitigar ruido sin ocultarlo.

#### Label variant sweep

**Qué se hizo**: se probaron variantes de la fórmula de la etiqueta.

**Cómo se hizo**: se modificaron pesos y componentes del `support_roam_score` y se compararon 15 variantes con la versión principal.

**Qué significa**: las variantes correlacionan muy alto con la etiqueta actual (≥0.99). Esto sugiere que las conclusiones no dependen de un ajuste frágil de la fórmula. En otras palabras, el límite observado no parece venir de haber elegido mal un peso concreto del score.

#### Análisis de embeddings

**Qué se hizo**: se inspeccionó si los embeddings aprendidos por la MLP capturan estructura interpretable de campeones.

**Cómo se hizo**: se visualizaron embeddings con t-SNE/UMAP, se midieron vecinos cercanos y se comparó con arquetipos humanos de support.

**Qué significa**: no aparecen clusters claros por arquetipo humano, pero sí cierto gradiente relacionado con roaming. Esto encaja con el resto del trabajo: la red aprende algo de señal continua, pero no descubre categorías limpias y separables. Para la memoria, esto sirve como resultado negativo útil, no como fracaso.

#### Prototipo CLI

**Qué se hizo**: se mantuvo una salida aplicada del TFG: un prototipo por terminal que permite introducir una composición y obtener una predicción interpretable.

**Cómo se hizo**: el prototipo carga el modelo entrenado y transforma una composición pregame en un score estimado de tendencia de roaming.

**Qué significa**: el prototipo no debe presentarse como oráculo de partida, sino como herramienta de apoyo: dado un draft, sitúa la composición en una zona de tendencia esperada. Es coherente con la conclusión general del TFG: el draft orienta, pero no determina.

---

## 4. Hallazgos clave (los buenos)

1. **El draft contiene señal predictiva real**: Global Mean → Champion Mean → GBT muestra una escalera clara. No es azar.
2. **El techo empírico cuantifica el límite**: el análisis ICC / media por grupo muestra que, incluso agrupando por botlane support-ADC y lado, la señal estable ronda R²≈0.17. Esto convierte un R² "bajo" en un resultado defendible: el modelo alcanza ~93% de ese techo.
3. **La etiqueta es robusta**: 15 variantes de la fórmula del score correlacionan ≥0.99 con la v5. El problema no es la definición del score.
4. **El campeón del support domina la señal** (SHAP): la identidad del support explica la mayor parte, seguida del ADC aliado y el support enemigo. Coherente con el dominio.
5. **La auditoría cualitativa explica los errores**: los mayores errores son partidas caóticas (botlane colapsada, muchas muertes). El draft no puede predecir caos de ejecución → refuerza la tesis.
6. **Métricas tolerantes útiles**: within-0.20 ≈ 74%, adjacent bin accuracy ≈ 97%. Para uso pregame tipo coach, la predicción sitúa la composición en una zona estratégica razonable.

---

## 5. Limitaciones principales (lo más importante para el email)

### 5.1 El draft no determina la ejecución
~16% de varianza explicada. El 84% restante depende de wave state, recalls, pathing del jungla, prioridad de mid, visión, eventos tempranos, coordinación y decisiones individuales. Esto es una **limitación inherente del planteamiento pregame-only**, no un fallo del modelo.

### 5.2 La MLP no supera al GBT
- La MLP (todas las variantes: OneHot, Embed, Per-Role) queda **por debajo del HistGBT** en todas las métricas.
- Incluso con embeddings por rol e interacciones explícitas de matchup, los árboles siguen siendo superiores.
- Hipótesis: con ~170 campeones y señal débil, el GBT explora interacciones de forma combinatoria (splits), mientras que la MLP necesita que se le especifiquen.
- El **HP search en curso (80 configs evaluadas)** confirma que la MLP es robusta a la configuración: la mejor config mejora solo +0.005 Spearman sobre el default. No es un problema de tuning.

### 5.3 Los embeddings no capturan estructura de arquetipos
- Los embeddings aprendidos no forman clusters por categoría humana (engage, enchanter, mage…). Silhouette ≈ -0.15.
- Sí codifican un gradiente continuo de roaming (correlación distancia↔score: Pearson 0.17, significativo).
- Pero la falta de clusters se explica por (1) señal débil del draft y (2) embedding compartido entre 10 slots.

### 5.4 Resolución temporal de la API
La etiqueta se construye con ~8 snapshots minutales (API de Riot). Con 5-7 frames válidos, un solo frame atípico cambia el score significativamente. Esta limitación es **inherente a la fuente de datos** y no se puede resolver con más partidas.

### 5.5 La etiqueta mide separación, no intención
`support_roam_score` captura separación support-ADC, que incluye roams limpios pero también colapsos de botlane. El chaos filtering mitiga esto parcialmente (sample_weight = 0.2 para partidas caóticas), pero no lo elimina.

### 5.6 Alcance reducido respecto a la propuesta original
- Se descartaron las etiquetas de jungla y equipo.
- Se descartó la integración multi-output.
- Se descartó la exploración de RNN/GRU/LSTM.
- **Justificación**: profundizar en una tarea bien definida aporta más que abrir tres tareas con análisis superficial. El framework queda validado para extensiones futuras.

---

## 6. La MLP — Estado actual y contexto para el tutor

### Qué se acordó en el Informe I
La MLP era la **primera hipótesis** del TFG: usar una red neuronal densa con representación one-hot del draft para predecir roaming. El plan era hacer tuning OAT, probar embeddings, y luego decidir si explorar RNN/LSTM.

### Qué ha pasado realmente
1. **MLP OneHot** (baseline neural): R² ≈ 0.140. Predice la media por campeón con poca variación.
2. **MLP Embeddings** (embeddings compartidos dim=16): R² = 0.150. Mejora sobre one-hot pero no sobre GBT.
3. **MLP Per-Role + Interactions** (embeddings por rol + dot-products de matchup): R² = 0.154. La mejor MLP, pero sigue por debajo del GBT (0.160).
4. **HP Search** (en ejecución): grid de 108 configs (4 hidden_dims × 3 dropout × 3 lr × 3 weight_decay). 80 completadas. Mejor Spearman val: 0.377 vs default 0.372. **Delta ≈ +0.005, por debajo del umbral de mejora significativa.**

### La narrativa para la memoria
La lectura no es "la MLP fracasa", sino: **la MLP, las baselines, el HistGBT y el techo ICC cuantifican juntos cuánta señal pregame hay en el draft.** La MLP fue la primera hipótesis acordada con el tutor. El GBT funciona como benchmark tabular fuerte. El ICC / media por grupo estima cuánta repetibilidad hay cuando se repiten condiciones de draft comparables. Juntos sugieren que el cuello de botella no es principalmente la arquitectura, sino la señal disponible antes de que empiece la partida.

---

## 7. Qué queda por hacer

| Tarea | Fecha objetivo | Estado |
|---|---|---|
| Terminar HP search MLP | 19-20 mayo | 🔄 En ejecución (80/108) |
| Informe de Progreso II | **24 mayo** | ⬜ Por redactar |
| CLI prototipo pulido | 31 mayo | ⬜ |
| Ablation study GBT | 31 mayo | ⬜ (nice to have) |
| Memoria capítulos 1-4 | 7 junio | ⬜ |
| Memoria capítulos 5-7 + figuras | 13 junio | ⬜ |
| **Propuesta de memoria final** | **14 junio** | ⬜ |

---

## 8. Preguntas sugeridas para el email al tutor

> [!IMPORTANT]
> Estas son las preguntas clave que podrías formular en el email:

1. **¿La narrativa "cuantificar la señal del draft" es adecuada?** — El TFG ha pasado de "construir el mejor predictor" a "medir cuánta señal predictiva hay realmente". ¿Esto es válido como contribución de TFG?

2. **¿La MLP (que era la hipótesis inicial acordada) queda suficientemente cubierta?** — Hemos probado 3 variantes de MLP + un HP search de 108 configs. En todas, el GBT gana. ¿Esto cierra la cuestión o se espera algo más de la parte neural?

3. **¿R²=0.16 al 93% del techo ICC es un resultado defendible?** — El techo se ha estimado agrupando partidas por configuraciones pregame repetidas, sobre todo botlane support-ADC + lado. ¿Esta forma de justificar el límite empírico te parece adecuada?

4. **¿Haber descartado jungla/equipo/multi-output es un problema serio?** — Lo hemos justificado como "profundizar sobre amplitud", pero la propuesta original los incluía.

5. **¿Sería útil quedar para revisar la dirección antes del Informe II (24 mayo)?** — Tienes dudas sobre si lo que estás haciendo añade valor suficiente.

---

## 9. Borrador de puntos clave para el email (resumido)

> Hola [tutor],
>
> Te escribo para ponerte al día del estado del TFG y consultarte sobre la dirección que está tomando.
>
> **Desde el Informe I**, he implementado la fase final con:
> - Dataset de 383k observaciones, split persistido train/val/test.
> - Baselines (Global Mean, Champion Mean), HistGBT (varias variantes), tres arquitecturas de MLP (OneHot, Embeddings, Per-Role con interacciones), y un HP search de 108 configuraciones de MLP en ejecución.
> - Techo empírico mediante ICC / media por grupo (R²≈0.17), SHAP, auditoría cualitativa, chaos filtering y label variant sweep.
>
> **El hallazgo principal** es que el draft contiene señal predictiva real pero limitada. Para contextualizarlo he calculado un techo empírico: agrupo partidas que comparten condiciones pregame comparables (por ejemplo la misma pareja support-ADC y el mismo lado del mapa) y miro cuánta variabilidad del `support_roam_score` se explica por esa agrupación frente a cuánta queda como variabilidad interna de la partida. Con botlane+side el techo práctico queda en R²≈0.173, mientras que el HistGBT obtiene R²≈0.161, alrededor del 93% de ese techo.
>
> Mi interpretación provisional es que el modelo no está limitado tanto por la arquitectura como por la propia información disponible en draft. Las MLPs, incluso con embeddings por rol y matchup features, no superan al GBT. El HP search confirma además que no parece ser un problema de tuning.
>
> **Las limitaciones principales** son: (1) la señal del draft es inherentemente débil para predecir ejecución de partida, (2) la MLP no aporta ventaja sobre métodos tabulares para este problema, (3) se ha descartado jungla/equipo/multi-output por falta de tiempo.
>
> **Lo que me preocupa**: ¿esta línea de trabajo aporta valor suficiente al TFG? ¿La narrativa de "cuantificar señal" en vez de "maximizar predicción" te parece válida? ¿La forma de estimar el techo empírico mediante ICC / medias por grupo es defendible? ¿La cobertura de la MLP es suficiente? ¿Sería bueno quedar antes del Informe II (24 mayo) para revisar si voy por el buen camino?
>
> Quedo a tu disposición para lo que necesites.
