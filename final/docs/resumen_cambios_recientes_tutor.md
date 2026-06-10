# Resumen de Cambios Recientes Para Reunión Con Tutor

## Objetivo del resumen

Este documento recoge los cambios metodológicos y argumentales introducidos después de las últimas críticas del tutor. La idea no es presentar más resultados por acumular resultados, sino dejar claro que:

- la comparación principal de modelos se ha rehecho bajo un protocolo común;
- el techo empírico ya no se explica como si el R2 saliera del ICC;
- la narrativa se reformula en términos generales de sistemas multiagente;
- la explicación de embeddings se concreta a nivel de mecanismo;
- la cercanía entre modelos y la media por support se interpreta como resultado del dominio, no como fallo sin justificar.

## 1. Rigor Experimental Y Comparación Justa

### Qué estaba mal o era débil antes

La comparación anterior mezclaba familias de modelos y features no equivalentes. En particular, el HistGBT tenía variantes enriquecidas con `Pair Target Encoding` o arquetipos, mientras que las MLP se presentaban con `Per-Role + Interactions`. Esto podía parecer una comparación injusta porque algunos modelos recibían información adicional o una representación más rica que otros.

También había un problema de trazabilidad: las curvas locales de entrenamiento no siempre cuadraban con la tabla final y no estaban todas. Esto debilitaba la defensa de los entrenamientos.

### Qué se ha cambiado

La tabla principal ahora usa solo modelos bajo el mismo protocolo de entrada:

```text
10 champion IDs + side
```

Los modelos principales comparados son:

- Global Mean.
- Champion Mean.
- HistGBT.
- MLP OneHot.
- MLP Embed Shared.
- MLP Per-Role + Interactions.

Todos los modelos aprendidos de la tabla principal usan:

- mismo split train/val/test;
- mismas 11 features de entrada;
- `sample_weight`;
- 3 seeds en los modelos aprendidos;
- métricas recomputadas desde predicciones en el mismo test set.

La auditoría está en:

```text
final/analysis/model_comparison/feature_protocol_audit.csv
```

La tabla principal está en:

```text
final/analysis/model_comparison/final_main_table_raw.md
```

### Resultado principal actual

| Modelo | R2 | Spearman | MAE | pred_std |
|---|---:|---:|---:|---:|
| Champion Mean | 0.1243 | 0.3362 | 0.1438 | 0.0680 |
| HistGBT | 0.1595 ± 0.0004 | 0.3869 ± 0.0004 | 0.1408 | 0.0740 |
| MLP OneHot | 0.1536 ± 0.0010 | 0.3801 ± 0.0005 | 0.1412 | 0.0786 |
| MLP Embed Shared | 0.1507 ± 0.0004 | 0.3763 ± 0.0007 | 0.1415 | 0.0738 |
| MLP Per-Role + Interactions | 0.1527 ± 0.0013 | 0.3783 ± 0.0014 | 0.1414 | 0.0765 |

Lectura:

- El HistGBT sigue siendo el mejor modelo principal.
- Las MLP se acercan, pero no superan al HistGBT.
- La mejora frente a Champion Mean existe pero es moderada.
- La cercanía entre modelos no se debe ocultar: indica que la señal prepartida está dominada por la identidad del agente de apoyo y que el resto del draft aporta señal adicional limitada.

### Cómo se responde a “los resultados están muy juntos”

La respuesta no debe ser “vamos a buscar un modelo más grande”, sino:

> La cercanía entre modelos es un hallazgo: en este problema, gran parte de la señal predecible está contenida en la identidad del agente de apoyo. Al introducir modelos más complejos, la mejora existe pero es pequeña porque el comportamiento real durante la partida depende de ejecución individual, coordinación, muertes tempranas y otros factores no observables antes de empezar.

Además, las predicciones están comprimidas:

```text
target_std ≈ 0.1905
HistGBT pred_std ≈ 0.0740
```

Esto significa que el modelo ordena composiciones de manera razonable, pero evita predecir extremos porque el target tiene mucho ruido no explicable desde draft. Por eso Spearman es especialmente importante.

## 2. Experimento Residual: Señal Más Allá Del Support

### Pregunta que contesta

Si el modelo está dominado por la identidad del support, la pregunta natural es:

> Para un mismo support, ¿el resto del draft empuja el score hacia arriba o hacia abajo respecto a lo esperable?

Para responderla se implementó un modelo aditivo:

```text
support_effect = media suavizada del support aliado
context_residual = HistGBT(resto del draft + interacciones suavizadas)
pred_final = support_effect + context_residual
```

El modelo residual no recibe `ally_utility_champion_id` como feature directa. El support solo aparece dentro de interacciones explícitas como support+ADC o support vs support enemigo.

### Resultados en test

Archivo:

```text
final/analysis/model_comparison/residual_context_diagnostics.csv
```

| Modelo | R2 | Spearman | Lectura |
|---|---:|---:|---|
| Smoothed Support Mean | 0.1240 | 0.3356 | efecto base del support |
| Residual Context GBT | 0.0386 sobre residual | 0.1892 sobre residual | señal contextual tras controlar support |
| Support Mean + Residual Context GBT | 0.1584 | 0.3854 | modelo aditivo completo |

Lift del modelo aditivo sobre la media suavizada del support:

```text
R2:       +0.0343
Spearman: +0.0498
```

Lectura para tutor:

> El modelo no se limita a predecir una media por support. Cuando se separa explícitamente ese efecto principal y se fuerza a otro modelo a predecir solo el residuo, el resto del draft todavía aporta señal. Esa señal no es suficiente para romper mucho el techo del problema, pero sí demuestra que hay información contextual real más allá de la identidad del support.

Este experimento se mantiene como análisis secundario, no como tabla principal, porque su objetivo es diagnóstico e interpretabilidad.

## 3. Pair Target Encoding Y Arquetipos

Las variantes `HistGBT + Pair TE` y `HistGBT + Archetypes` ya no forman parte de la comparación principal.

Se reportan como análisis secundarios:

```text
final/analysis/model_comparison/comparison_secondary_table_raw.csv
```

Resultados secundarios:

| Modelo secundario | R2 | Spearman | Motivo de exclusión de tabla principal |
|---|---:|---:|---|
| HistGBT + Archetypes | 0.1611 | 0.3881 | añade arquetipos/clases externas |
| HistGBT + Pair TE | 0.1604 | 0.3880 | añade target encodings |
| Support Mean + Residual Context GBT | 0.1584 | 0.3854 | descomposición diagnóstica |

Lectura:

> Estas variantes confirman que hay pequeñas mejoras o diagnósticos útiles, pero no cambian la conclusión principal. Por rigor, quedan fuera de la tabla principal porque no usan exactamente el mismo protocolo de entrada.

## 4. Curvas De Entrenamiento Y WandB

Los scripts de entrenamiento ya soportan `--use-wandb` en GBT y MLPs. Las curvas de entrenamiento relevantes pueden enseñarse desde WandB.

También existen curvas locales para las MLP:

```text
final/analysis/training_curves/
  mlp_onehot_raw_curves.png
  mlp_onehot_quantile_curves.png
  mlp_embed_raw_curves.png
  mlp_embed_quantile_curves.png
  mlp_per_role_raw_curves.png
  mlp_per_role_quantile_curves.png
```

Punto importante para la reunión:

> No usaría curvas antiguas que no cuadraban con la tabla final. La defensa debe apoyarse en WandB y en los `history.csv` ligados a los artefactos finales, no en imágenes sueltas generadas durante fases anteriores.

## 5. ICC Y R2: Corrección De La Explicación

### Qué había que corregir

La crítica del tutor era correcta: no se puede decir que “del ICC sacamos un R2”. El ICC es un coeficiente de consistencia; R2 es una métrica predictiva. Son métricas distintas.

### Explicación correcta

Ahora se separan dos cosas:

1. **ICC train**
   - Métrica descriptiva in-sample.
   - Mide consistencia dentro de grupos repetidos de draft.
   - No se compara directamente con el R2 de modelos.

2. **R2 group-mean OOS**
   - Métrica predictiva.
   - Se calculan medias por grupo usando solo train.
   - Se aplican esas medias a test.
   - Grupos no vistos usan la media global de train.
   - Este sí se puede comparar con el R2 de los modelos.

Archivo metodológico:

```text
final/analysis/ceiling/ceiling_methodology_note.md
```

Archivo de resultados:

```text
final/analysis/ceiling/ceiling_oos_summary.csv
```

Ejemplos actuales:

| Agrupación | ICC train | R2 group-mean OOS |
|---|---:|---:|
| support_champion | 0.1214 | 0.1249 |
| botlane_champions | 0.1394 | 0.1239 |
| botlane_champions+side | 0.1391 | 0.1132 |
| sup_vs_enemy_sup_champion | 0.1316 | 0.1200 |

Lectura:

> El ICC no se transforma en R2. Ambos se calculan sobre agrupaciones relacionadas, pero responden preguntas diferentes. El ICC describe cuánta consistencia hay dentro de grupos de draft en train. El R2 group-mean OOS pregunta cuánto predeciría un sistema muy simple que memoriza medias de grupos en train y las aplica a test.

## 6. Vocabulario: Reencuadre General

La memoria final debe usar una capa de abstracción más general. El lector no experto debe entender el problema aunque no conozca League of Legends.

| Término de LoL | Término recomendado |
|---|---|
| campeón | agente |
| draft | configuración prepartida / composición de agentes |
| mapa | entorno / escenario espacial |
| support | agente de apoyo |
| ADC | tirador / compañero principal de zona |
| botlane | zona inferior / zona asignada inicial |
| roaming | movilidad temprana fuera de la zona asignada |
| gank | intervención de un agente externo |
| partida caótica | ejecución temprana anómala |

Frase recomendada:

> Este TFG estudia hasta qué punto la configuración prepartida de un sistema multiagente permite anticipar la movilidad temprana de uno de sus agentes funcionales, el agente de apoyo.

Se puede mantener el ejemplo de “trolear”, pero como explicación cualitativa de errores extremos:

> En algunos casos, la etiqueta mide separación espacial real, pero esa separación no corresponde a una intención estratégica, sino a una ejecución anómala o no cooperativa. Este fenómeno, conocido informalmente por jugadores como “trolear”, justifica que ciertos errores no sean predecibles desde la configuración inicial.

## 7. Explicación Precisa De Embeddings

### Qué es un embedding en este proyecto

En este TFG, un embedding no se usa en sentido genérico. Es una tabla de parámetros entrenable:

```python
nn.Embedding(vocab_size=173, embedding_dim=16, padding_idx=0)
```

Matemáticamente:

```text
E ∈ R^(173 x 16)
```

Cada campeón/agente discreto `c` se representa como:

```text
E[c, :]
```

Es decir, un lookup a una fila de una matriz entrenable.

### Cómo se aprende exactamente

1. Al inicio, los vectores son parámetros de la red.
2. En cada batch, el modelo toma los IDs de los 10 agentes.
3. Cada ID selecciona una fila de la tabla embedding.
4. Esos vectores se concatenan con `side` y, en la variante Per-Role + Interactions, con dos productos escalares:

```text
dot(ally_utility, enemy_utility)
dot(ally_utility, ally_bottom)
```

5. La red produce una predicción.
6. Se calcula la pérdida, en este caso MSE ponderado por `sample_weight`.
7. Backpropagation calcula gradientes no solo para las capas densas, sino también para las filas de embedding usadas en ese batch.
8. AdamW actualiza esas filas.

Por tanto, no son vectores fijos ni definidos manualmente. Se ajustan para reducir el error de predicción del score.

### Diferencia con one-hot

One-hot también es una representación vectorial, pero no es lo mismo que el embedding entrenable usado aquí:

| Representación | Qué es | Qué aprende |
|---|---|---|
| One-hot | vector disperso fijo de dimensión 173 | nada en la representación; aprende la capa posterior |
| Embedding compartido | matriz entrenable 173x16 compartida por todos los slots | similitudes funcionales globales |
| Embedding per-role | 10 matrices entrenables 173x16, una por slot | comportamiento distinto según posición/rol |

Frase para defender:

> Un embedding categórico puede verse como una capa lineal aplicada a un one-hot, pero con una diferencia importante: proyecta la categoría a una dimensión mucho menor y esa matriz se actualiza con backpropagation. El cuello de botella 173→16 fuerza al modelo a representar agentes con efectos parecidos mediante vectores cercanos.

## 8. Mensaje Central Para El Tutor

Resumen oral recomendado:

> He corregido la comparación para que la tabla principal sea justa: todos los modelos aprendidos usan las mismas 11 variables, los mismos splits, pesos y seeds. Las variantes enriquecidas quedan como análisis secundarios. La cercanía entre modelos y la media por support no se oculta: se interpreta como una propiedad del problema. Para comprobar si el modelo miraba más allá del support, he añadido un diagnóstico residual: primero se modela la media suavizada del support, y luego se fuerza a un GBT a predecir solo el residuo usando el resto del draft e interacciones. El modelo aditivo sube de R2=0.1240 a R2=0.1584, lo que demuestra que hay señal contextual adicional, aunque limitada. También he separado ICC de R2 group-mean OOS: el ICC es descriptivo; el R2 comparable se obtiene aplicando medias de grupos entrenadas en train sobre test. Finalmente, la memoria se va a reescribir con vocabulario general de sistemas multiagente y con una explicación técnica precisa de embeddings.

## 9. Qué Enseñar En La Reunión

Llevar estos artefactos:

```text
final/analysis/model_comparison/final_main_table_raw.md
final/analysis/model_comparison/feature_protocol_audit.csv
final/analysis/model_comparison/comparison_secondary_table_raw.csv
final/analysis/model_comparison/residual_context_diagnostics.csv
final/analysis/ceiling/ceiling_methodology_note.md
final/analysis/ceiling/ceiling_oos_summary.csv
final/analysis/training_curves/
WandB runs con curvas de entrenamiento
```

Orden recomendado para explicarlo:

1. Primero, protocolo común y tabla principal.
2. Después, por qué Champion Mean es fuerte y qué significa.
3. Luego, experimento residual para demostrar señal contextual adicional.
4. Después, ICC vs R2 OOS.
5. Finalmente, vocabulario general y explicación de embeddings.
