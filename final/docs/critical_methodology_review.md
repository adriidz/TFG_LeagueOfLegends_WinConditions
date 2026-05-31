# Crítica Técnica Severa: Metodología de Roaming de Supports

> [!CAUTION]
> Este documento no contiene consejos genéricos de ML. Cada argumento está respaldado con evidencia concreta de tu código, tus datos y tus resultados. Lo que sigue es doloroso pero necesario.

---

## Veredicto ejecutivo

Tu proyecto tiene un **error de diseño fundamental** que no se resuelve cambiando hiperparámetros, probando 15 variantes de label, ni filtrando partidas caóticas. El problema es:

**Estás intentando predecir un fenómeno de ejecución (in-game) usando exclusivamente información de draft (pre-game), con un target que confunde señal con ruido por diseño.**

Tu propio ceiling analysis lo demuestra: el ICC del `botlane_champions+side` es **0.139** y el R² del group mean es **0.173**. Esto significa que el **82.7% de la varianza del target no está explicada por la composición de equipos**, ni siquiera con una lookup table infinita. Tu mejor modelo (HistGBT + Pair TE) logra R²=0.161 y Spearman=0.388 — está prácticamente en el techo teórico del problema tal como lo has planteado.

> [!IMPORTANT]
> **No tienes un problema de modelo. Tienes un problema de definición del fenómeno.** El modelo funciona correctamente dentro de los límites de lo que le estás pidiendo. Lo que le pides es fundamentalmente inadecuado.

---

## Fallo 1: "Estar fuera de botlane" es una proxy terrible para roaming

### Lo que haces

Tu [label_quality.md](file:///c:/Users/adria/Desktop/TFG/final/docs/label_quality.md) define el score como:

```
score_raw = 0.45 × outside_ratio + 0.35 × far_ratio + 0.20 × xp_gap
score = score_raw ^ 0.75
```

Donde `outside_ratio` = fracción de frames (minuto 5-12) donde el support está fuera de `BOT_LANE_CORE, BOT_SIDE_NEAR, RIVER_BOT, DRAGON_AREA, bases`.

### Por qué es incorrecto

"Estar fuera de botlane" NO es roaming. Es la unión de al menos 6 fenómenos distintos:

| Fenómeno | outside_ratio | far_ratio | xp_gap | ¿Es roaming? |
|---|---|---|---|---|
| Roam intencional a mid/top | Alto | Alto | Alto | **Sí** |
| Support muere y aparece en base | Bajo* | Variable | Variable | **No** |
| ADC muere, support vaguea | Alto | Alto | Medio | **No** |
| Swap de lanes (rotación estratégica) | Alto | Alto | Bajo | **Depende** |
| Chase/persecución después de teamfight | Alto | Alto | Bajo | **No** |
| Recall largo + pathing de vuelta | Variable | Variable | Bajo | **No** |
| Yuumi attached a jungla | Alto | Alto | Alto | **No es roaming convencional** |

*Frames con support muerto o en base se descartan, pero con solo 8 frames máximos, un support que muere en min 5, reaparece en min 6 y está de vuelta en bot en min 7 pierde 2 frames → 25% de resolución perdida.

### La evidencia de tu propio dataset

Tu [case_notes.md](file:///c:/Users/adria/Desktop/TFG/final/analysis/qualitative_case_audit/case_notes.md) muestra que **17 de 20 top errors** son `chaotic_early_game`. Mira el caso #1:

- **Yuumi + Smolder vs Pyke + Vel'koz**
- Yuumi muere en min 1.36, Smolder muere en min 1.48, Smolder muere de nuevo en min 3.03, otra vez en min 3.87, Yuumi muere en min 3.99
- Score = **1.000** (máximo roaming posible)
- Predicción = 0.209

Yuumi NO roameó. **La botlane fue masacrada.** El score de 1.0 es artefactual: Yuumi estaba muerta o en respawn la mayor parte del tiempo, los pocos frames válidos la pillan fuera de bot (porque va caminando de vuelta, o está en base, o está attached a otro ally post-colapso).

### El problema fundamental

Tú mismo lo reconoces en [label_quality.md](file:///c:/Users/adria/Desktop/TFG/final/docs/label_quality.md#L31): "La etiqueta mide separación, no intención". Pero luego dices "El problema no es la definición del score sino la contaminación por outliers caóticos" — esto es incorrecto. **El 27% de tu dataset (15,321 partidas) son "caóticas"** según tu propio flag. Eso no son outliers. Es un cuarto de la distribución. Tu "chaos_flag" con `sample_weight=0.2` es un parche que no resuelve el problema porque:

1. El caos no es binario. Partidas con 5 muertes combinadas no son "limpias" solo porque no llegan al threshold de 6.
2. El umbral es arbitrario y no está validado contra ground truth.
3. Incluso en partidas "limpias", el score sigue sin distinguir roaming de pathing, recalls, o swaps.

---

## Fallo 2: Resolución temporal catastrófica

### Los números

Del minuto 5 al 12 tienes **como máximo 8 snapshots** (uno por minuto). Tras filtrar frames con support muerto o en base, tu propio análisis muestra que los casos típicos tienen **5-7 frames válidos**. Tu [label_quality.md](file:///c:/Users/adria/Desktop/TFG/final/docs/label_quality.md#L43) dice: "cada frame vale entre 14% y 20% del score".

### Implicación

Un support que:
1. Está en bot lane en min 5, 6, 7, 8, 9
2. Roamea a mid lane en min 10 y 11
3. Está fuera en min 12 (river de vuelta)

Tiene `outside_ratio = 3/8 = 0.375` si se le cuentan todos los frames, o potencialmente diferente si alguno se descarta.

Otro support que:
1. Muere en min 5 (frame descartado)
2. Camina de vuelta en min 6 (fuera de bot)
3. Está en bot min 7, 8, 9
4. Muere en min 10 (frame descartado)
5. Base min 11 (frame descartado)
6. Caminando de vuelta min 12 (fuera de bot)

Tiene `outside_ratio = 2/5 = 0.40` sobre 5 frames válidos.

**Ambos tienen outside_ratio similar pero el primero es un roamer activo y el segundo es un feeder.** Y con 5-8 datos binarios por partida, tu score tiene una resolución efectiva de ~0.14-0.20 por step. Es cuantización brutal.

### Por qué esto importa para tu modelo

Con 8 frames y 3 componentes (outside, far, xp), tu target tiene **resolución discreta** que aparenta ser continua por el gamma 0.75. Pero la entropía real del target es mucho menor que la de un continuo en [0,1]. Tu `target_std = 0.190` pero la distribución está concentrada en el rango medio-bajo (`mean ≈ 0.39`). El modelo naturalmente converge a predecir la media porque la varianza residual dominante es ruido de ejecución que no puede predecirse.

---

## Fallo 3: Gap información features ↔ target insalvable tal como está planteado

### Lo que usas como features

Según tu [technical_spec.md](file:///c:/Users/adria/Desktop/TFG/final/docs/technical_spec.md#L310-L321):

```python
# 10 champion IDs + 20 summoner spell IDs + 1 side = 31 features categóricas
```

### Lo que predices

Un target que depende de: ejecución in-game, snowball de lane, decisiones individuales del jugador, pathing del jungler, timing de ganks, timing de backs...

### El ceiling lo confirma

| Agrupación | ICC | R² group mean |
|---|---|---|
| support_champion | 0.121 | 0.121 |
| botlane_champions | 0.139 | 0.161 |
| botlane_champions+side | 0.139 | 0.173 |
| all_10_champions | Insuficiente | — |

El ICC de 0.139 para `botlane_champions` significa que solo el **13.9% de la varianza total se explica por la composición de la botlane**. El 86.1% restante es varianza intra-grupo: **partidas con la misma botlane pero resultados completamente distintos**. Esto es un bound teórico que ningún modelo puede superar usando solo draft.

Tu mejor modelo alcanza R²=0.161, que está **a 0.012 del ceiling** de `botlane_champions+side` (R²=0.173). **Ya estás en el techo.** No hay gap modelo-señal. Hay gap señal-fenómeno.

### Tu SHAP confirma la misma historia

Del [shap_global_importance.csv](file:///c:/Users/adria/Desktop/TFG/final/analysis/shap/shap_global_importance.csv):

- `ally_utility_champion_id`: mean SHAP = **0.058** (dominante)
- `ally_bottom_champion_id`: mean SHAP = **0.016**
- Todo lo demás: < 0.014

El modelo esencialmente hace una lookup de la media por support champion con ajustes menores. Es exactamente lo que un baseline de champion mean haría con un poco más de contexto. Y tu champion mean baseline tiene R²=0.125 y Spearman=0.336. Tu mejor modelo sube eso a R²=0.161 y Spearman=0.388. **Toda la maquinaria de ML te da +0.05 de Spearman sobre una tabla de medias.**

---

## Fallo 4: Yuumi y champions atípicos destruyen la coherencia del target

### El problema de Yuumi

Yuumi es un caso que **invalida tu definición de roaming a nivel conceptual**. Yuumi se attacha a un ally y su posición coincide con la de ese ally. Si la botlane colapsa y Yuumi se attacha al jungler o al mid, su posición será "fuera de bot" pero Yuumi NO roameó — siguió haciendo su función de enchanter attached.

Tu [case_notes.md](file:///c:/Users/adria/Desktop/TFG/final/analysis/qualitative_case_audit/case_notes.md) muestra:
- Caso #1: Yuumi score=1.000, prediction=0.209 → error=0.791
- Caso #2: Yuumi score=0.930, prediction=0.174 → error=0.756
- Caso #8: Yuumi score=0.817, prediction=0.158 → error=0.658
- Caso #13: Yuumi score=0.777, prediction=0.131 → error=0.647

Yuumi es un support con `champion_mean=0.149` y `expert_score=0.080` (el más bajo posible). Pero en partidas caóticas, su score sube a 0.8-1.0 porque su ADC muere y ella se attacha a otro. **Tu label dice "máximo roaming" para el champion que por diseño NO puede roamear independientemente.**

### Otros champions problemáticos

- **Bard**: diseñado para roamear (passive de chimes), su movimiento por el mapa es natural y constante. ¿Bard caminando a recoger chimes es "roaming táctico"? Parcialmente sí, parcialmente no.
- **Pyke**: assassin support que roamea agresivamente. Su score debería ser alto. Pero "estar fuera de bot" por dying en una gank fallida ≠ roaming exitoso.
- **Senna**: su scoring como marksman_support cambia dependiendo de si juega como ADC o como support. Tu modelo no distingue estos roles.
- **Twitch support, Brand support, Vel'koz support**: "supports" mage/adc que pueden estar en bot lane haciendo daño pero cuyo patrón de movimiento es diferente al de un enchanter.

### Impacto en el modelo

El modelo ve "Yuumi" → predice ~0.15 (su media). Cuando la partida real da 1.0 por caos, el modelo tiene un error de 0.85. Estos errores extremos dominan tu MSE y tu MAE. Pero el modelo está **haciendo lo correcto**: la mejor predicción pre-game para Yuumi IS ~0.15. El label es el que está mal, no el modelo.

---

## Fallo 5: El label variant sweep demostró que el problema es la definición, no la fórmula

Tu [sweep_config.json](file:///c:/Users/adria/Desktop/TFG/final/analysis/label_variant_sweep/sweep_config.json) probó **16 variantes** de label con diferentes pesos de frame/combat/vision. Los resultados del [sweep_top40_by_spearman.csv](file:///c:/Users/adria/Desktop/TFG/final/analysis/label_variant_sweep/sweep_top40_by_spearman.csv):

- Mejor variante: `events_tiny_90_07_03` con Spearman=0.383
- V5 original: Spearman=0.381
- Rango total del sweep: Spearman [0.379, 0.383]

**Diferencia entre la mejor y la peor variante: 0.004 de Spearman.** Esto demuestra que el problema NO está en los pesos de la fórmula. Con 8 frames minutales, cualquier combinación lineal de outside/far/xp es esencialmente la misma variable. Tu [label_quality.md](file:///c:/Users/adria/Desktop/TFG/final/docs/label_quality.md#L88) ya lo dice: "correlación v5↔v7 ≈ 0.99".

Pero la conclusión correcta NO es "la señal está limitada por la resolución minutal". La conclusión correcta es: **tu target mide algo diferente de lo que crees medir, y ningún rerrecetado de los mismos ingredientes cambiará eso.**

---

## Fallo 6: No separas la pregunta "¿cuánto roaming hay?" de "¿quién está predispuesto a roamear?"

Tu modelo intenta predecir cuánto roaming *observado* habrá en una partida concreta, dado el draft. Pero esto mezcla dos componentes:

1. **Predisposición del draft al roaming** (señal estable por composición)
2. **Ejecución estocástica de la partida** (ruido irreductible)

El ICC de 0.139 te dice que solo el 14% es componente (1). Pero lo que probablemente quieres medir — y lo que sería útil — es la **predisposición**, no la ejecución concreta.

### Lo que deberías hacer vs. lo que haces

| Enfoque | Target | Predictor | ¿Funciona? |
|---|---|---|---|
| **Lo que haces** | score por partida individual | draft | No, 86% es ruido |
| **Lo que deberías hacer** | probabilidad/tendencia de roaming del draft | draft | Potencialmente mejor |

Esto es como intentar predecir si un tirador de basketball va a encestar cada tiro individual (alta varianza) vs. predecir su porcentaje de tiro en la temporada (baja varianza, más predecible). Tú estás haciendo lo primero.

---

## Fallo 7: Tu evaluación no evalúa lo que importa

### Métricas actuales

R², MAE, RMSE, Pearson, Spearman — todas evalúan predicción puntual a nivel de partida individual. Pero:

- R²=0.16 en un target con 86% de varianza irreductible es **buen rendimiento**, no malo.
- Spearman=0.39 significa que el modelo ordena razonablemente los casos, dado que el fenómeno que mide es ruidoso.
- `compression_ratio=0.39` (pred_std/target_std) indica que el modelo comprime mucho las predicciones — lo cual es **correcto** dado que la media por champion es la mejor predicción y tiene baja varianza.

### Lo que no evalúas

1. **Calibración por rangos de champion**: ¿El modelo predice Yuumi < Lulu < Thresh < Pyke < Bard? Esto sería más informativo que R².
2. **Validez del ranking de drafts**: dado dos drafts distintos con el mismo support, ¿el modelo predice correctamente cuál tendrá más roaming?
3. **Concordancia con ranking experto**: tienes una referencia experta de 47 champions. ¿Cuál es el Spearman entre las predicciones medias por champion y ese ranking?
4. **Fairness por tipo de champion**: ¿El error del modelo es sistemáticamente peor para enchanters que para tanks? Tu SHAP sugiere que sí.

---

## Fallo 8: El chaos filter es un hack que oculta un problema de diseño

Tu [16_add_chaos_filter_weights.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/16_add_chaos_filter_weights.py) aplica `sample_weight=0.2` a partidas caóticas. Los resultados del [clean_vs_chaotic.md](file:///c:/Users/adria/Desktop/TFG/final/analysis/clean_vs_chaotic/clean_vs_chaotic.md):

| Subset | R² | Spearman |
|---|---|---|
| All | 0.160 | 0.387 |
| Clean | 0.171 | 0.397 |
| Chaotic | 0.122 | 0.363 |

La mejora en "clean" es real pero pequeña (+0.010 Spearman). Y el R² de "chaotic" sigue siendo positivo (0.122) — el draft sigue explicando algo incluso en partidas caóticas, lo cual sugiere que el caos no es completamente aleatorio sino que ciertos drafts son más propensos a colapsos de bot.

Pero el filtro **no resuelve el problema fundamental**: incluso en partidas "limpias", tu target sigue sin distinguir roaming intencional de otros movimientos fuera de bot.

---

## Propuesta Concreta: Nueva Definición de Roaming

### Principio rector

> **Roaming** = movimiento del support fuera de botlane con la **intención observable** de generar impacto en otra zona del mapa, evidenciado por participación en eventos fuera de bot.

No podemos medir intención directamente, pero podemos definir **indicadores observables con la timeline data de Riot** que distingan roaming de ruido.

### Definición: Roam Event

Un **roam event** es una secuencia temporal que cumple:

```
ROAM EVENT = (
    support sale de botlane context
    → NO está muerto, en recall, ni en base
    → participa en al menos 1 evento "productivo" fuera de botlane:
        - Kill/assist en CHAMPION_KILL fuera de bot zone
        - Participación en ELITE_MONSTER_KILL (Herald, Dragon con posición fuera de bot)
        - BUILDING_KILL o TURRET_PLATE_DESTROYED fuera de bot
        - WARD_PLACED en zona no-bot (si la posición está disponible)
)
```

### Score propuesto (v8)

```python
# Conteo de roam events productivos del support en minutos 0-14
productive_roams = count(CHAMPION_KILL assists fuera de bot)
                 + count(CHAMPION_KILL kills fuera de bot)
                 + 0.5 * count(ELITE_MONSTER assists)
                 + 0.3 * count(BUILDING_KILL/PLATE fuera de bot)

# Normalización relativa al tiempo de juego útil
time_alive_outside_bot = frames vivos fuera de bot / total frames vivos

# Score compuesto
raw_score_v8 = 0.60 * min(productive_roams / 3.0, 1.0)   # saturar en 3 eventos
             + 0.30 * time_alive_outside_bot               # presencia fuera de bot
             + 0.10 * xp_deficit_vs_adc_normalized          # evidencia complementaria

roam_score_v8 = raw_score_v8 ^ 0.75
```

### Por qué esto es mejor

1. **Requiere evidencia de impacto**: un support muerto que aparece fuera de bot no suma puntos.
2. **Distingue roaming de caos**: morir repetidamente genera `time_alive_outside_bot` bajo, no alto.
3. **La participación en kills/objectives fuera de bot es observable** directamente en los timeline events de Riot.
4. **Es más estable**: un soporte que consigue 2 assists fuera de bot es un roamer independientemente de si el frame minutal lo captura.

### Lo que necesitas de la Riot API

De la timeline (ya lo tienes en tu `support_event_context`):
- `CHAMPION_KILL` events con `position.x`, `position.y` y lista de assists → ya los capturas
- `ELITE_MONSTER_KILL` con assists → ya los capturas
- `BUILDING_KILL` / `TURRET_PLATE_DESTROYED` → ya los capturas

Lo que necesitas **verificar**: que las posiciones de estos eventos son confiables y que tu definición geométrica de "fuera de bot" incluye correctamente la zona de dragon como bot (o no, según tu criterio).

### Tratamiento de champions atípicos

| Champion | Tratamiento |
|---|---|
| **Yuumi** | Usar posición del host. Si Yuumi attached a mid y participa en kill fuera de bot, SÍ cuenta como roam. Yuumi attached a ADC en bot NO cuenta. Necesitas detectar attached state (posición idéntica a otro ally). |
| **Bard** | Recoger chimes no genera eventos de kill/objective → no infla el score. Solo cuenta si participa en impacto fuera de bot. |
| **Pyke** | Sus kills fuera de bot cuentan naturalmente. Sus muertes no. |
| **Senna ADC** | Si Senna es la carry (está en BOTTOM role), excluir del análisis de support. |

---

## Plan Experimental Paso a Paso

### Fase 1: Validar el target (1-2 días)

> [!IMPORTANT]
> No toques el modelo hasta que valides el target.

#### Experimento 1.1: Auditoría manual de 50 partidas

1. Muestrea 50 partidas estratificadas:
   - 10 con score_v5 < 0.15 (supuestos non-roamers)
   - 10 con score_v5 entre 0.35-0.45 (zona media ambigua)
   - 10 con score_v5 > 0.80 (supuestos heavy roamers)
   - 10 con chaos_flag=True y score_v5 > 0.70 (sospechosos de false positive)
   - 10 con Yuumi/Bard/Pyke en composiciones variadas

2. Para cada partida, usa tu [09_qualitative_case_audit.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/09_qualitative_case_audit.py) para extraer el timeline, mapa y eventos.

3. Clasifica manualmente cada partida en:
   - `genuine_roam`: el support se movió intencionalmente para generar impacto
   - `forced_displacement`: caos, muertes, swap, colapso
   - `mixed`: algo de ambos
   - `measurement_artifact`: score alto por artefacto de la resolución temporal

4. Calcula el % de concordancia entre tu label v5 y tu clasificación manual:
   - Si > 80% de scores altos son `genuine_roam`: tu label es aceptable
   - Si < 60%: tu label está fundamentalmente roto

#### Experimento 1.2: Comparar v5 vs v8 en la misma muestra

1. Implementa el `roam_score_v8` basado en eventos productivos
2. Calcula ambos scores para las 50 partidas auditadas
3. Compara cuál correlaciona mejor con tu clasificación manual
4. Calcula ICC de v8 vs v5 agrupando por support champion

### Fase 2: Construir el nuevo target (2-3 días)

#### Experimento 2.1: Implementar v8 a escala

1. Usa tus datos de `support_event_context_m12.parquet` que ya tienen los eventos
2. Implementa el score v8 como un nuevo script `19_build_support_roam_score_v8_productive.py`
3. Verifica:
   - Distribución de v8 (debe tener más masa en 0 y más separación)
   - Correlación v5↔v8 por champion mean (debería ser alta) y por partida (debería ser más baja que v5↔v6)
   - ICC de v8 por champion (debería ser > 0.139 si mide mejor la predisposición)

#### Experimento 2.2: Ceiling analysis de v8

1. Repite tu [05_empirical_ceiling.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/05_empirical_ceiling.py) con v8 como target
2. Si `ICC(v8, botlane_champions) > ICC(v5, botlane_champions)` → v8 es más predecible desde draft → has mejorado el target
3. Si ICC similar → el problema no era la definición, era genuinamente que el draft no determina el roaming

### Fase 3: Entrenar y evaluar (1-2 días)

#### Experimento 3.1: Mismo modelo, diferente target

1. Entrena el HistGBT con target v8 usando el mismo pipeline que [03_train_gbt.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/03_train_gbt.py)
2. Compara métricas: si Spearman mejora → el target era el cuello de botella

#### Experimento 3.2: Reglas heurísticas como baseline

Antes de ML, implementa un sistema de reglas:

```python
def heuristic_roam_score(champion_name, adc_champion_name, enemy_support_name):
    base = CHAMPION_MEAN_LOOKUP[champion_name]  # tu tabla de medias ya existente
    
    # Ajuste por ADC: ADC con self-peel permite más roaming
    if adc_champion_name in SAFE_ADCS:  # Ezreal, Caitlyn, Tristana
        base += 0.03
    if adc_champion_name in VULNERABLE_ADCS:  # Kog'Maw, Twitch, Aphelios
        base -= 0.03
    
    # Ajuste por enemy support: contra engage, más difícil roamear
    if enemy_support_name in HEAVY_ENGAGE:  # Leona, Nautilus, Blitzcrank
        base -= 0.02
    
    return clip(base, 0, 1)
```

Mide Spearman de esta heurística. Si es comparable al modelo → **el ML no está aportando nada** sobre knowledge domain.

### Fase 4: Diagnóstico de labeling vs. datos vs. modelo (1 día)

#### Experimento 4.1: Test de ruido del label

1. Toma partidas que se repiten con la MISMA composición de botlane (hay ~104 partidas de media por botlane pair en tu dataset)
2. Calcula la varianza de `roam_score` dentro de cada grupo
3. Compara v5 vs v8: si v8 tiene menor varianza intra-grupo → es un mejor indicador de predisposición

#### Experimento 4.2: Label noise injection

1. Toma tu label v5 y añade ruido gaussiano controlado: `v5_noisy = v5 + N(0, σ)`
2. Entrena modelos con σ = {0.05, 0.10, 0.15, 0.20}
3. Si la métrica del modelo no cambia significativamente al añadir ruido → tu label ya tiene tanto ruido inherente que el ruido adicional es irrelevante → **confirma que el label es el problema**

#### Experimento 4.3: Oracle test con features de ejecución

1. Añade features de ejecución (muertes tempranas, gold diff min 5) como inputs al modelo
2. Si el R² sube significativamente → confirma que la varianza no explicada es de ejecución
3. Si no sube mucho → la varianza es puramente aleatoria (timing, micro-decisiones)

> [!WARNING]
> Este test es SOLO diagnóstico. Nunca uses features de ejecución en el modelo final porque no las tendrías en pre-game.

---

## Resumen: Priorización

```
┌─────────────────────────────────────────────────────────┐
│ URGENCIA ALTA                                           │
│                                                         │
│ 1. Auditar 50 partidas manualmente (Exp 1.1)            │
│ 2. Construir v8 basado en eventos productivos (Exp 2.1) │
│ 3. Comparar ICC v5 vs v8 (Exp 2.2)                      │
│                                                         │
│ Si ICC(v8) > ICC(v5): éxito, has mejorado el target     │
│ Si ICC(v8) ≈ ICC(v5): el draft genuinamente no          │
│    determina el roaming → tu TFG debería argumentar     │
│    esto como conclusión (hallazgo negativo válido)       │
└─────────────────────────────────────────────────────────┘
```

> [!TIP]
> Un resultado negativo bien documentado ("el draft explica solo el 14% de la varianza del roaming observado, y esto es un bound inherente del fenómeno") es un resultado académico perfectamente válido y más interesante que forzar un R² de 0.16 con trucos. **Documenta el techo, muestra que tu modelo lo alcanza, y argumenta por qué eso es lo mejor posible.**

---

## Conclusiones por pregunta original

| Tu pregunta | Mi respuesta |
|---|---|
| ¿"Fuera de botlane" es mala proxy? | **Sí.** Confunde 6+ fenómenos distintos. |
| ¿Cómo separar roaming de caos? | **Requiriendo evidencia de impacto** (kill/assist/objective fuera de bot), no solo posición. |
| ¿Cómo tratar Yuumi? | **Detectar attached state** y no penalizar por posición del host. O excluirla como caso especial. |
| ¿Modelar eventos en vez de score global? | **Sí, el score v8 propuesto se basa en count de eventos productivos**, no en ratio de frames. |
| ¿Usar reglas heurísticas antes de ML? | **Sí.** Tu champion_mean ya es una heurística potente. El ML solo añade +0.05 Spearman. |
| ¿Cómo construir un target más robusto? | **v8: productive_roams + time_alive_outside + xp_gap**, con saturación y sin contar frames de muerte/base. |
| ¿Cómo evaluar manualmente? | **50 partidas estratificadas**, clasificadas en genuine_roam / forced_displacement / mixed / artifact. |
| ¿Qué features concretas de la API? | Ya tienes las necesarias: kill events con posición, assists, objectives. Solo necesitas combinarlas como score. |
| ¿Cómo saber si es labeling, datos o modelo? | **Experiments 4.1-4.3**: noise injection, oracle test, varianza intra-grupo. Tu evidencia actual apunta fuertemente a labeling. |
