# Estructura del Informe de Progreso II

## Dinámica general

El Informe I contaba una historia de *descubrimiento*: "partimos de clasificación, vimos que no funcionaba, reformulamos a regresión continua, y hay señal". El Informe II cuenta una historia de *cuantificación*: "medimos cuánta señal hay exactamente, con qué modelos, contra qué techo, y qué significa para el TFG".

El tono cambia de exploratorio a conclusivo. El lector debe terminar el informe sabiendo:
1. Qué se puede y qué no se puede predecir desde el draft.
2. Qué modelo es mejor y por qué.
3. Qué queda para la entrega final (poco: redacción + prototipo CLI).

**Extensión recomendada**: 15-20 páginas (sin contar bibliografía ni anexos). El Informe I tenía ~18 páginas efectivas; este debe ser similar o ligeramente más corto porque hay menos contexto nuevo que explicar.

---

## Estructura de apartados

### 0. Portada + Datos identificativos
- Título, autor, fecha, tutor.

### 1. Resumen ejecutivo (~1 página)

Párrafo denso que responde: ¿qué se ha hecho desde el Informe I y cuál es la conclusión principal?

**Incluir**:
- Dataset final: 383k observaciones, 195k partidas, parches 14.x-16.8.
- Comparación de 5 familias de modelos sobre el mismo split test.
- Resultado principal: HistGBT alcanza R²=0.16, al 93% del techo empírico ICC (0.17).
- Conclusión clave: el draft contiene señal predictiva de roaming, pero limitada (~16% de la varianza). Las MLPs con embeddings no superan a los árboles.
- Estado: el proyecto está en su fase final. Resta redactar la memoria y cerrar el prototipo CLI.

### 2. Seguimiento de la planificación (~2 páginas)

> *Requerido: "Indicación del nivel de seguimiento de la planificación prevista y de los ajustes efectuados, junto con su justificación."*

**Incluir**:
- Tabla del planning del Informe I (§8) con estado actual de cada bloque.
- Qué se cumplió: tuning, baselines, GBT, SHAP, techo empírico, auditoría.
- Qué se desvió: no se reintrodujeron etiquetas de jungla/equipo. Justificación: se priorizó profundidad sobre amplitud — cuantificar el techo de la señal del draft para una tarea bien definida aporta más que abrir tres tareas con análisis superficial.
- Qué se añadió (no previsto): SHAP, auditoría cualitativa de errores, chaos filtering, label variant sweep.
- Planning actualizado hasta la entrega final (28/06).

**Tabla sugerida**:

| Bloque planificado (Inf. I) | Estado | Nota |
|---|---|---|
| Tuning OAT (MLP + etiqueta) | ✅ Completado | Label variant sweep con 15 variantes; regularización MLP |
| Embeddings y feature enrichment | ✅ Completado | MLP Embed (dim=16) + GBT con arquetipos + Pair TE |
| Informe de Progreso II | 🔄 En curso | Este documento |
| Redefinir etiqueta jungla | ❌ Descartado | Priorización de profundidad sobre amplitud |
| Redefinir etiqueta equipo | ❌ Descartado | Ídem |
| Integración multi-output | ❌ Descartado | No justificable sin etiquetas validadas |
| Exploración RNN/GRU/LSTM | ❌ Descartado | El cuello de botella es la señal del draft, no la arquitectura |
| Prototipo terminal | ⏳ Pendiente | CLI funcional con el mejor modelo |
| Memoria final | ⏳ Pendiente | Estructura definida, redacción por hacer |

### 3. Metodología final (~4-5 páginas)

> *Requerido: "Explicación general de la metodología que se ha seguido finalmente y de los cambios respecto a la propuesta inicial."*

#### 3.1 Pipeline de datos
- Recogida (API Riot) → Frame state → Draft features → Scores v5 → Splits.
- Dataset final: 383k obs, split 70/15/15 por match_id, sin leakage.
- Diagrama del pipeline (reusar o actualizar el del Informe I).

#### 3.2 Etiqueta `support_roam_score`
- Receta: 0.45 outside + 0.35 far + 0.20 xp_gap, gamma 0.75.
- Resolución: 8 frames minutales (limitación API Riot).
- Validación: Spearman 0.82 vs referencia experta (47 campeones).
- **Nuevo**: Chaos filtering. Explicar por qué y cómo (chaos_flag + sample_weight=0.2). Resultado: 26.5% de partidas marcadas como caóticas.
- **Nuevo**: Robustez. El label sweep con 15 variantes (v6 events) demostró que cambiar la fórmula produce correlaciones ≥0.99 con v5 → la señal está limitada por la resolución minutal, no por los pesos.

#### 3.3 Modelos comparados
- **Global Mean**: baseline trivial.
- **Champion Mean**: baseline de lookup por support champion.
- **HistGBT**: HistGradientBoostingRegressor con OrdinalEncoder + categorical_features.
  - Variantes: base, + arquetipos, + Pair TE.
- **MLP OneHot**: red densa con one-hot de champion IDs.
- **MLP Embeddings**: red densa con embeddings aprendidos (dim=16).
- Todos entrenados y evaluados sobre el mismo split con sample_weight.

#### 3.4 Techo empírico
- ICC (Intraclass Correlation Coefficient) por botlane+side.
- R² máximo teórico: 0.173. El GBT alcanza el 93%.

#### 3.5 Análisis complementarios
- SHAP: importancia global y dependencias.
- Auditoría cualitativa: 40 casos, reconstrucción del score, diagnóstico de errores.
- Clean vs chaotic: evaluación segregada.

### 4. Resultados (~4-5 páginas)

> *Requerido: "Exposición y valoración de los resultados."*

#### 4.1 Tabla comparativa principal

La tabla de `comparison_tables.md` (Table A - Raw Scale), formateada para el informe. Esta es la tabla central del TFG.

| Modelo | R² | Spearman | MAE |
|---|---|---|---|
| ICC Ceiling | 0.173 | — | — |
| HistGBT + Pair TE | 0.161 | 0.388 | 0.141 |
| HistGBT | 0.160 | 0.387 | 0.141 |
| MLP Embed | 0.143 | 0.366 | 0.142 |
| MLP OneHot | 0.140 | 0.364 | 0.142 |
| Champion Mean | 0.125 | 0.336 | 0.144 |
| Global Mean | 0.000 | — | 0.155 |

#### 4.2 Interpretación por bloques
- **Baselines → GBT**: el GBT mejora 28% en R² sobre Champion Mean. Confirma que hay interacciones entre campeones que un lookup simple no captura.
- **GBT vs MLPs**: los árboles ganan ~2 puntos de R². Los embeddings mejoran ligeramente sobre one-hot (+0.003) pero no alcanzan al GBT. Hipótesis: con ~170 campeones y señal débil, los árboles explotan interacciones mejor que redes densas.
- **Enrichment**: arquetipos y Pair TE no mejoran significativamente. La señal ya la captura el GBT base.
- **Quantile vs Raw**: el scale del target no cambia las conclusiones. Resultado presentable como "la distribución de la etiqueta no es el cuello de botella".
- **Techo ICC**: R²=0.173. El GBT está al 93%. Queda ~1 punto de R² de mejora teórica → no hay arquitectura que cambie sustancialmente esto con datos de draft.

#### 4.3 Clean vs Chaotic
- Incluir tabla de `clean_vs_chaotic.md`.
- Interpretación: el modelo predice mejor en partidas "normales" → el caos early-game es varianza no capturable desde el draft. Resultado que refuerza la tesis.

#### 4.4 SHAP y feature importance
- 2-3 figuras clave: beeswarm, top features, dependencia de ally_utility_champion_id.
- El campeón support aliado domina la importancia, seguido del ADC aliado y el support enemigo. Esto es coherente con la tarea.

#### 4.5 Auditoría cualitativa
- 2-3 casos ejemplo (top errors) que muestran que los errores grandes corresponden a partidas caóticas, no a fallos de la etiqueta.
- Caso Yuumi: score=1.0 en partida donde la botlane colapsó (4+7 muertes). El draft no puede predecir esto.

#### 4.6 Training curves de las MLPs
- Incluir plots de `training_curves/`. Mostrar cómo la regularización mejorada reduce el gap train/val.

### 5. Conclusiones provisionales (~1-2 páginas)

> *Requerido: "Conclusiones provisionales."*

**Incluir estas 5 conclusiones**:

1. **El draft contiene señal predictiva de roaming del support**, pero explica solo el ~16% de la varianza observada. El 84% restante depende de la ejecución en partida.

2. **HistGradientBoostingRegressor es el mejor modelo**, superando tanto a las baselines como a las MLPs. Los árboles capturan interacciones entre campeones de forma más eficiente que las redes densas para este problema.

3. **Los embeddings de campeones no aportan ganancia significativa** sobre one-hot en esta tarea. La señal del draft es demasiado débil para que representaciones densas revelen estructura latente no capturable por métodos tabulares.

4. **La etiqueta es robusta a cambios de definición** (15 variantes probadas, correlación ≥0.99 con v5), y el principal factor de ruido son las partidas caóticas con botlane colapsada, no la fórmula del score.

5. **El techo predictivo empírico (ICC=0.173) está prácticamente alcanzado.** Esto cierra la pregunta experimental del TFG: el draft define una predisposición, no un destino.

### 6. Trabajo restante para la entrega final (~0.5 páginas)

- Redacción de la memoria final.
- Prototipo CLI funcional (script `predict_cli.py` ya existe).
- Limpieza del repositorio.
- Preparación de la presentación.

### 7. Fuentes de información consultadas (~1 página)

> *Requerido: "Fuentes de información consultadas."*

Reutilizar la bibliografía del Informe I [1]-[14] y añadir:
- [15] Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," NeurIPS 2017. (para justificar GBT)
- [16] Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017. (para SHAP)
- [17] McGraw & Wong, "Forming Inferences About Some Intraclass Correlation Coefficients," Psych Methods 1996. (para ICC)
- scikit-learn docs para HistGradientBoostingRegressor.
- PyTorch docs para nn.Embedding.

---

## Figuras a incluir (mínimo)

1. Diagrama del pipeline (actualizado).
2. Tabla comparativa de modelos (la central).
3. Training curves MLP OneHot y MLP Embed (con regularización mejorada).
4. SHAP beeswarm o top features.
5. Clean vs Chaotic: tabla comparativa.
6. 1-2 scatter true-vs-pred (GBT vs MLP).
7. Distribución de la etiqueta con chaos_flag marcado.

---

## Diferencias clave con el Informe I

| Aspecto | Informe I | Informe II |
|---|---|---|
| Tono | Exploratorio | Conclusivo |
| Modelo | Solo MLP OneHot | 5 familias comparadas |
| Baselines | Solo Global Mean | Global Mean + Champion Mean + ICC |
| Etiqueta | Presentada sin validar mucho | Validada + sweep + chaos filter |
| Conclusión | "Hay señal" | "Hay exactamente esta cantidad de señal y este es el techo" |
| Planning | 8 bloques futuros | 4 tareas de cierre |
