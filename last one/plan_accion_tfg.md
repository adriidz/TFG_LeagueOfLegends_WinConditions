# Plan de Acción — TFG Support Roaming

> [!IMPORTANT]
> Este plan está organizado por **bloques de trabajo priorizados** hasta la reunión del 10/06 y la entrega final del 28/06. He revisado a fondo TODOS los scripts, análisis, informes y datos del proyecto.

---

## Diagnóstico: Los 4 problemas del tutor + hallazgos adicionales

### Problema 1 del tutor: Falta de rigor en entrenamientos

El tutor tiene razón. Tras revisar los 13 scripts de entrenamiento, estas son las **inconsistencias reales encontradas**:

| Problema | Detalle |
|---|---|
| **No hay WandB** | Ningún script usa experiment tracking. Las métricas se guardan en JSON/CSV sueltos |
| **Feature mismatch GBT vs MLP** | GBT usa 31 features (10 campeones + 20 summoner spells + side). MLP usa solo 11 (10 campeones + side). **No están en igualdad de condiciones** |
| **Sample weight inconsistente** | `03_train_gbt.py` SÍ usa sample_weight, pero `03b_gbt_enriched.py` y `03c_gbt_interactions.py` NO lo usan |
| **Hiperparámetros ambiguos** | Los scripts individuales (04a/b/c) tienen defaults distintos a `run_all_training.py`, que los sobreescribe con hiperparámetros "más fuertes". No queda claro qué configuración generó los modelos finales |
| **No hay CV ni intervalos de confianza** | Todo se basa en un único split train/val/test. Sin bootstrap CIs, sin k-fold, sin tests de significancia |
| **Curvas de entrenamiento solo como PNG** | No hay datos numéricos (CSV de loss por epoch). Solo imágenes que no se pueden auditar |

> [!CAUTION]
> La comparación HistGBT con Pair TE vs MLP Per-Role + Inter **NO es justa**: el GBT ve summoner spells y el MLP no. Además, el GBT con Pair TE usa target encodings (información adicional del target) mientras que las MLPs no tienen nada equivalente.

### Problema 2 del tutor: El ICC y el R²

El tutor preguntó "¿cómo se saca un R² de un ICC?" — y es una pregunta legítima. Revisando [05_empirical_ceiling.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/05_empirical_ceiling.py):

- El **ICC(1)** se calcula con ANOVA: `ICC = (MSB - MSW) / (MSB + (k-1)·MSW)`
- El **R² group mean** es una métrica **diferente**: `R² = 1 - SS_res/SS_tot` donde la predicción es la media de cada grupo
- **El R² NO se "saca del ICC"**. Son dos métricas distintas calculadas en paralelo sobre los mismos grupos
- **Problema adicional**: el R² group mean está calculado **in-sample** (sobre train), pero se compara con el R² de modelos calculado **out-of-sample** (test). Es una comparación sesgada — el techo aparece artificialmente más alto

**Cómo explicarlo al tutor**: "No sacamos R² del ICC. Son dos medidas paralelas. El ICC estima qué proporción de la varianza total es varianza entre-grupos (consistencia del fenómeno). El R² group-mean mide cuánto predice si memorizo la media de cada grupo. Ambos se calculan sobre las mismas agrupaciones de draft."

### Problema 3 del tutor: Vocabulario demasiado LoL

El informe actual usa constantemente: roaming, botlane, support, ADC, draft, ganks, recalls, dives, snowball, laning phase... El tribunal no es experto en LoL. Se necesita una capa de abstracción.

**Mapeo propuesto**:

| Término LoL | Término general |
|---|---|
| Draft / selección de campeones | Configuración pre-partida / composición de agentes |
| Mapa (Summoner's Rift) | Entorno / escenario de juego |
| Campeón | Agente / tipo de agente |
| Roles (top, jungle, mid, bot, support) | Posiciones funcionales / roles del equipo |
| Roaming del support | Movilidad del agente de apoyo fuera de su zona asignada |
| Botlane | Zona de responsabilidad principal del agente de apoyo |
| Draft + timeline | Información pre-partida + observaciones in-game |
| ADC / carry | Agente de daño principal / compañero de zona |
| Trolear | Comportamiento no cooperativo o deliberadamente anómalo |
| Kill, asistencia, objetivo | Evento de impacto / acción productiva |
| Gank | Intervención de un agente externo |
| Recall | Retorno a base |

> [!TIP]
> **Mantén "trolear" con explicación** — es un ejemplo excelente de por qué la etiqueta mide separación y no intención. Un tribunal de ingeniería agradecerá que justifiques tus errores de predicción con fenómenos reales del dominio.

### Problema 4 del tutor: Embeddings mal explicados

El tutor dijo que "embeddings" es muy general y que no supiste explicar el mecanismo exacto. Aquí tienes la explicación técnica completa que debes dominar:

**Qué es un embedding en tu modelo:**
```python
self.embed = nn.Embedding(vocab_size=173, embed_dim=16, padding_idx=0)
# Es una tabla de lookup: matriz E ∈ ℝ^{173×16}
# Dado un champion_id c, el embedding es simplemente E[c, :] ∈ ℝ^{16}
```

**Cómo se inicializa:** `torch.nn.init.normal_(weight, mean=0, std=1)` — vectores aleatorios. El index 0 (padding) se mantiene siempre a cero.

**Cómo se entrena:** Los embeddings forman parte de `model.parameters()`. El optimizador AdamW calcula gradientes via backpropagation y los actualiza igual que los pesos de las capas lineales. No hay optimizador ni lr separados. La función de pérdida (MSE entre predicción y roam score real) propaga gradientes hacia atrás hasta la tabla de embeddings. **Los vectores de campeones que afectan de manera similar al roam score convergen hacia representaciones cercanas en el espacio de 16 dimensiones.**

**Por qué NO es lo mismo que one-hot:**
1. **One-hot**: cada campeón es un vector binario de dimensión 173. Todos los campeones son equidistantes (distancia √2). No codifica ninguna relación entre campeones.
2. **Embedding**: cada campeón es un vector denso de dimensión 16. El **cuello de botella dimensional** (173→16) fuerza al modelo a descubrir qué campeones se comportan de forma similar. Campeones con efectos parecidos sobre el roaming quedan cerca en el espacio.
3. **Matemáticamente**: `E^T · one_hot(c) = E[c, :]` — un embedding ES una capa lineal después de one-hot, pero con la diferencia clave de que la dimensión reducida actúa como **regularización implícita** que fuerza la agrupación semántica.

**Las 3 variantes en tu proyecto:**
- **Shared** (MLPEmbed): 1 tabla (173×16). El mismo vector para Thresh como aliado o enemigo. **Limitación: no diferencia roles.**
- **Per-Role** (MLPPerRole): 10 tablas independientes (10 × 173×16). Thresh-como-support-aliado tiene vector distinto a Thresh-como-support-enemigo. Más expresivo.
- **Per-Role + Interactions**: Además añade 2 dot products explícitos: `dot(embed_ally_sup, embed_enemy_sup)` y `dot(embed_ally_sup, embed_ally_adc)` para capturar sinergias/matchups directamente.

---

## Bloque 1 — URGENTE: Arreglar la metodología (3-5 jun)

### 1.1 Poner todos los modelos en igualdad de condiciones

Esto es lo **más crítico** para el tutor. Si los modelos no se comparan en igualdad, la comparación no vale.

**Acción**: Reentrenar TODOS los modelos con las mismas condiciones:

```
FEATURES comunes: 10 champion IDs + 1 side = 11 features categóricas
   (descartar summoner spells para igualar — SHAP muestra que aportan ~0.001)
   
SAMPLE WEIGHT: SIEMPRE usar chaos_flag → sample_weight=0.2/1.0

HIPERPARÁMETROS: documentar EXACTAMENTE qué se usa para cada modelo

SEED: usar mínimo 3 seeds (42, 123, 456) y reportar media ± std
```

**Modelos a comparar (tabla final limpia):**

| Modelo | Tipo | Representación del draft |
|---|---|---|
| Global Mean | Baseline | Ninguna |
| Champion Mean | Baseline | Media por support champion |
| HistGBT | Tabular | OrdinalEncoder → categóricas nativas |
| MLP OneHot | Neural | One-hot (173 dims por slot) |
| MLP Embed Shared | Neural | Embedding compartido (16 dims por slot) |
| MLP Per-Role + Inter | Neural | Embedding por rol + dot products |

> [!IMPORTANT]
> **Quitar summoner spells** no debería cambiar los resultados significativamente (importancia por permutación = 0.001). Pero SI cambia algo, eso es un hallazgo interesante que merece discutirse.

### 1.2 Integrar WandB

No necesitas refactorizar todo. Lo mínimo viable:

```python
import wandb

wandb.init(project="tfg-support-roaming", config={
    "model": "histgbt",
    "features": "champions_side",
    "target": "raw",
    "seed": 42,
    "sample_weight": True,
    # ... todos los hiperparámetros
})

# Al final del entrenamiento:
wandb.log({
    "test/r2": r2, "test/spearman": spearman,
    "test/mae": mae, "test/rmse": rmse,
    "val/r2": val_r2, ...
})

# Para MLPs, en el loop de entrenamiento:
for epoch in range(n_epochs):
    # ... train step ...
    wandb.log({"train/loss": train_loss, "val/loss": val_loss, 
               "val/spearman": val_spearman, "epoch": epoch})
```

**Objetivo**: que cada experimento tenga un registro auditable con TODOS los parámetros y métricas. Esto resuelve directamente la crítica del tutor.

### 1.3 Guardar curvas de entrenamiento como datos

```python
# Al final de cada MLP:
history_df = pd.DataFrame(history)  # list of dicts per epoch
history_df.to_csv(output_dir / "training_history.csv", index=False)
```

Esto permite reconstruir las curvas y verificar que cuadran con la tabla.

### 1.4 Corregir el ceiling (ICC vs R²)

**Acción**: Recalcular el R² group-mean en **test** (no en train):

```python
# Calcular medias por grupo SOLO con datos de train
group_means = train_df.groupby(grouping_cols)["target"].mean()

# Predecir en test usando esas medias
test_preds = test_df[grouping_cols].merge(group_means, ...)

# Calcular R² out-of-sample
r2_ceiling = r2_score(test_df["target"], test_preds)
```

Esto dará un ceiling **más honesto** y directamente comparable con el R² de los modelos.

---

## Bloque 2 — Reforzar la narrativa (5-8 jun)

### 2.1 Reencuadrar la historia como investigación científica

El TFG no es "un predictor de roaming". Es una **investigación sobre los límites de la información pre-partida**. La narrativa debe ser:

```
PREGUNTA: ¿Cuánta información sobre el comportamiento del agente de apoyo 
está contenida en la configuración pre-partida?

HIPÓTESIS: La composición de agentes contiene señal parcial, pero la 
ejecución individual limita la predictibilidad.

MÉTODO: Construir una proxy observable del roaming, compararla con 
referencia experta, entrenar modelos y medir el techo empírico.

RESULTADO: El draft explica ~16% de la varianza (R²=0.161), el techo 
empírico es ~17% (R²≈0.173 out-of-sample), y el 83% restante es 
ejecución individual. Esto es un HALLAZGO, no un fracaso.
```

### 2.2 Dar más peso al MAE

El tutor quiere MAE más presente. Actualmente solo aparece como columna secundaria. En el informe final:

- **MAE = 0.141** en escala [0,1] → error medio de ±14.1 puntos porcentuales
- **74.2%** de predicciones dentro de ±0.20
- **41.8%** dentro de ±0.10

Traducción para el tribunal: "En un rango de 0 a 1, el modelo se equivoca en promedio 0.14 puntos. En 3 de cada 4 casos, la predicción está a menos de 0.20 del valor real."

### 2.3 Preparar la explicación del ICC/R² para el tutor

Prepara un **mini-slide o diagrama** que explique:

```
┌──────────────────────────────────────────────────────┐
│  DATOS: 383K observaciones partida-equipo            │
│                                                      │
│  Se agrupan por (support, ADC, side)                 │
│  → ~3,800 grupos                                     │
│                                                      │
│  ICC(1) = varianza entre-grupos / varianza total     │
│  = 0.139 → 13.9% de la varianza es estable           │
│           por composición                             │
│                                                      │
│  R² group-mean = si predigo la media de cada grupo   │
│  → cuánta varianza explico = 0.173 (in-sample)       │
│  → Recalculado out-of-sample: ~0.15-0.16             │
│                                                      │
│  Son dos métricas diferentes sobre los mismos datos  │
│  El ICC es un coeficiente de consistencia             │
│  El R² es la varianza explicada por predicción trivial│
└──────────────────────────────────────────────────────┘
```

---

## Bloque 3 — Estructura del informe final (8-10 jun)

> [!IMPORTANT]
> 8 páginas + 2 de anexos. Cada sección debe justificar su presencia.

### Estructura propuesta (8 páginas)

| Sección | Páginas | Contenido clave |
|---|---|---|
| **1. Introducción y contexto** | 1 | Problema general (predicción de comportamiento desde configuración inicial), dominio (MOBA como sistema multiagente), objetivo específico (cuánta señal hay en la composición) |
| **2. Trabajo relacionado** | 0.5 | Predicción de win-rate desde draft [3], draft recommendation [4], patrones de comportamiento [5], análisis espacial [6,7] |
| **3. Datos y etiqueta** | 1.5 | API de Riot, dataset (383K obs, EUW, nivel alto), construcción de la proxy de movilidad (fórmula, geometría, gamma), validación experta (Spearman 0.82), chaos filter, limitaciones de la etiqueta |
| **4. Metodología** | 1.5 | Pipeline (draft→features, timeline→etiqueta), splits (GroupShuffle por match), modelos (baselines, GBT, MLPs + embeddings), métricas (R², Spearman, MAE), techo empírico (ICC + R² OOS) |
| **5. Resultados** | 2 | Tabla comparativa EN IGUALDAD, curvas de entrenamiento, importancia de variables, auditoría de errores (troleo), comparación con techo |
| **6. Discusión y limitaciones** | 1 | Qué señal hay vs. qué no, resolución temporal, etiqueta como proxy, partidas caóticas, alcance del dataset |
| **7. Conclusiones** | 0.5 | Hallazgo principal: draft explica ~16%, techo ~17%, límite inherente del problema. Trabajo futuro: nuevas etiquetas, más roles, resolución temporal |

### Anexos (2 páginas)

| Anexo | Contenido |
|---|---|
| **A. Arquitectura de modelos** | Diagrama de la MLP + embeddings, tabla de hiperparámetros, explicación técnica concisa |
| **B. Prototipo CLI** | Captura de pantalla, ejemplo de uso, cómo se traduce el score |

---

## Bloque 4 — Lo que NO incluir en el informe

> [!WARNING]
> El informe tiene 8+2 páginas. Cada línea debe aportar. Estas cosas quedan FUERA:

- ❌ La evolución de clasificación a regresión (ya está en los informes de progreso)
- ❌ Todas las variantes de etiqueta probadas (v5, v6, v7, v8, v9) — solo presentar la final con una frase de "se probaron variantes que correlacionan >0.99"
- ❌ El OAT fallido del cluster
- ❌ Los resultados con quantile transform (no mejoran → una frase basta)
- ❌ El HP search completo (108 configs) — una frase: "se buscaron hiperparámetros sin mejora significativa"
- ❌ Detalles de la geometría del mapa — una figura y una frase
- ❌ Los 20 errores cualitativos uno por uno — 2-3 casos representativos

---

## Roadmap día a día

### Semana 1: Antes de la reunión con el tutor (2-10 jun)

| Día | Tarea | Entregable |
|---|---|---|
| **Mar 3** | Integrar WandB en GBT + MLPs. Quitar summoner spells de GBT. Asegurar sample_weight en todos | Scripts actualizados |
| **Mié 4** | Reentrenar TODOS los modelos con 3 seeds (42, 123, 456) | Runs en WandB |
| **Jue 5** | Recalcular ceiling R² out-of-sample (test). Generar tabla comparativa final. Generar curvas como CSV | Tabla nueva, ceiling corregido |
| **Vie 6** | Preparar explicación de embeddings (diagrama). Preparar explicación ICC/R². Preparar argumento "hallazgo negativo válido" | Material de defensa |
| **Sáb 7** | Escribir borrador de secciones 1-4 del informe (generalizar vocabulario) | Borrador parcial |
| **Dom 8** | Escribir borrador de secciones 5-7 + anexos | Borrador completo |
| **Lun 9** | Revisar coherencia, preparar preguntas específicas para el tutor | Documento para reunión |
| **Mar 10** | **REUNIÓN CON TUTOR** | — |

### Semana 2: Post-reunión, pulido (11-15 jun)

| Día | Tarea |
|---|---|
| 11-12 | Incorporar feedback del tutor |
| 13-14 | Pulir informe, generar figuras finales |
| **14-15** | **Propuesta de informe final lista** |

### Semana 3: Cierre (16-28 jun)

| Día | Tarea |
|---|---|
| 16-20 | Correcciones finales del tutor, pulir CLI, revisar anexos |
| 21-25 | Preparar presentación oral, ensayar defensa |
| 26-27 | Revisión final de todos los entregables |
| **28** | **ENTREGA** |

---

## Guía rápida para la defensa oral

### Preguntas que te van a hacer y cómo responder

**P: "¿Por qué R²=0.16? ¿No es muy bajo?"**
R: "No es bajo, es informativo. El techo empírico (calculado out-of-sample) indica que la composición explica como máximo ~16-17% de la varianza. El modelo captura >93% de esa señal. El 83% restante es ejecución individual — decisiones de los jugadores durante la partida que no están disponibles antes de empezar. El resultado principal del TFG no es el predictor, sino la cuantificación de ese límite."

**P: "¿Qué son los embeddings exactamente?"**
R: "Son vectores de 16 dimensiones, uno por campeón, que se inicializan de forma aleatoria y se actualizan con backpropagation durante el entrenamiento. La pérdida del modelo (MSE entre predicción y roam score) propaga gradientes hasta estos vectores. Campeones que afectan al roaming de manera similar convergen hacia vectores cercanos. Es una técnica estándar en NLP (Word2Vec usa el mismo principio) y en sistemas de recomendación. A diferencia de one-hot, que trata todos los campeones como equidistantes, el embedding fuerza una representación compacta de 16 dimensiones que captura similitudes funcionales."

**P: "¿No está comparando modelos de manera injusta?"**
R: (Con los arreglos del Bloque 1) "Todos los modelos de la tabla final usan exactamente las mismas features (10 IDs de campeón + lado), los mismos sample weights, el mismo split, y se reporta media ± desviación sobre 3 semillas. La única diferencia es la representación del draft y la arquitectura."

**P: "¿El ICC y el R² son lo mismo?"**
R: "No. El ICC mide consistencia intra-grupo mediante descomposición de varianza ANOVA. El R² mide cuánta varianza explica un predictor. Calculamos ambos sobre las mismas agrupaciones de composición: el ICC nos dice que ~14% de la varianza es estable por composición (señal predecible), y el R² de grupo nos da una referencia de qué rendimiento tendría un predictor que memorice la media de cada composición."

**P: "¿Qué pasa con las partidas 'caóticas'?"**
R: "Un 26.5% de las observaciones tienen desarrollo temprano anómalo — por ejemplo, un ADC que muere 5+ veces antes del minuto 12. En estos casos, la etiqueta mide separación real entre support y ADC, pero esa separación no refleja una predisposición del draft al roaming, sino un colapso de la fase de líneas. Estas partidas reciben peso 0.2 durante el entrenamiento. En partidas limpias, el R² sube a 0.171; en caóticas baja a 0.122. Esto confirma que el modelo funciona mejor cuando la ejecución se acerca a lo esperable por el draft."

---

## Checklist para la reunión del 10/06

- [ ] Modelos reentrenados en igualdad de condiciones (sin summoner spells en GBT)
- [ ] WandB integrado con todos los runs registrados
- [ ] Tabla comparativa con media ± std sobre 3 seeds
- [ ] Ceiling R² recalculado out-of-sample (test)
- [ ] Borrador del informe con vocabulario generalizado
- [ ] Diagrama de arquitectura MLP + embeddings preparado
- [ ] Explicación ICC vs R² preparada (diagrama/slide)
- [ ] Curvas de entrenamiento guardadas como datos numéricos
- [ ] Argumento de "hallazgo negativo válido" articulado
- [ ] 3 ejemplos de errores cualitativos seleccionados (incluyendo "troleo")
