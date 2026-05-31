# Roadmap TFG — 16 mayo → 14 junio 2026

> [!IMPORTANT]
> Fecha límite dura: **24 mayo** (Informe Progreso II) y **14 junio** (propuesta de memoria final).
> Todo lo que no añada una fila a la tabla comparativa, una figura a la memoria, o un párrafo nuevo a las conclusiones → descartarlo.

---

## Estado actual (16 mayo)

### ✅ Completado
- Dataset final: 383k obs, split persistido train/val/test por match_id
- Baselines: Global Mean, Champion Mean
- HistGBT: base (R²=0.160), + arquetipos, + Pair TE
- MLP OneHot: entrenada con sample_weight y regularización mejorada
- MLP Embeddings: entrenada con sample_weight, embeddings guardados
- Techo empírico ICC: R²=0.173 (GBT al 93%)
- SHAP: importancia global, beeswarm, dependencias
- Auditoría cualitativa: 40 casos, reconstrucción del score
- Chaos filtering: chaos_flag + sample_weight implementado
- Label variant sweep: 15 variantes, robustez confirmada
- Clean vs Chaotic: análisis segregado
- Training curves: plots generados
- Documentación: label_quality.md, progress_check_may15.md, decisions.md

### ❌ Pendiente
- Análisis del espacio de embeddings (t-SNE/UMAP)
- MLP con features de interacción (comparación justa)
- HP search sistemático de MLP
- CLI prototipo pulido
- Ablation study del GBT
- Informe de Progreso II
- Memoria final

---

## Semana 1 — 16-18 mayo (viernes)

**Objetivo**: Cerrar los dos análisis que faltan para poder escribir el Informe II con conclusiones sólidas.

### ✅ Tarea 1A: Análisis del espacio de embeddings — COMPLETADA (16 mayo)
**Script**: `final/scripts/17_embedding_analysis.py`
**Outputs**: `final/analysis/embedding_analysis/`

**Resultados obtenidos**:
- Silhouette por arquetipo de support: **-0.146** → no hay clusters por categoría humana.
- Silhouette por clase Data Dragon: **-0.074** → ídem.
- Top-5 vecinos del mismo arquetipo: **4%** → básicamente azar.
- Vecinos más cercanos incoherentes (Janna→Riven, Nautilus→Jax).
- **Pero**: correlación distancia↔roam score: Pearson **r=0.166** (p=3.2e-06),
  Spearman **r=0.176** (p=7.6e-07). Señal débil pero significativa.
- El t-SNE coloreado por roam score muestra un gradiente suave no aleatorio.

**Conclusión para la memoria**: los embeddings compartidos no reproducen la
clasificación por arquetipos, pero sí codifican una gradación continua de
tendencia al roaming. La falta de clusters se explica por (1) señal débil del
draft (~16% varianza) y (2) embedding compartido entre 10 slots, que impide
especialización por rol.

### ✅ Tarea 1B: MLP con embeddings por rol e interacciones — COMPLETADA (16 mayo)
**Script**: `final/scripts/04c_train_mlp_per_role.py`
**Outputs**: `final/models/mlp_per_role/`, `final/analysis/embedding_analysis_per_role_ally_utility/`

**Resultados obtenidos**:
- MLP Per-Role + Interactions: **R²=0.1544**, Spearman=0.3806
- Mejora sobre MLP Embed compartido (R²=0.1496, +0.005) pero no cierra
  el gap con HistGBT (R²=0.1599-0.1614).
- Los embeddings del slot ally_utility mejoran la vecindad de supports
  (top5_support_neighbor_rate 0.295 vs 0.220 compartido), aunque los
  clusters por arquetipo siguen débiles.
- Feature enrichment descartado: con ~2000 obs por campeón, el modelo ya
  tiene datos suficientes para aprender efectos individuales. El GBT con
  arquetipos solo ganó +0.001 R².

**Conclusión para la memoria**: incluso dando a la MLP la misma granularidad
por slot que el GBT (embeddings separados) e interacciones explícitas de
matchup, los árboles siguen siendo superiores. Esto se explica porque el GBT
explora interacciones de forma combinatoria mediante splits, mientras que la
MLP requiere que se le especifiquen. Con ~170 campeones y señal débil, la
ventaja estructural de los árboles es decisiva.

**Tabla comparativa final (raw scale, test set)**:

| Modelo | R² | Spearman | MAE |
|---|---|---|---|
| ICC Ceiling | 0.173 | — | — |
| HistGBT + Pair TE | **0.161** | **0.388** | **0.141** |
| HistGBT | 0.160 | 0.387 | 0.141 |
| MLP Per-Role + Interactions | 0.154 | 0.381 | 0.141 |
| MLP Embed (compartido) | 0.150 | 0.376 | 0.142 |
| MLP OneHot | — | 0.379 | — |
| Champion Mean | 0.125 | 0.336 | 0.144 |
| Global Mean | 0.000 | — | 0.155 |

---

## Semana 2 — 19-24 mayo (sábado = entrega Informe II)

**Objetivo**: Escribir y entregar el Informe de Progreso II.

### Tarea 2A: HP search de MLP (lunes-martes)
**Prioridad**: 🟡 Media — refuerza la robustez
**Tiempo estimado**: 4 horas (+ tiempo de GPU)
**Script**: `final/scripts/18_mlp_hp_search.py`

**Qué hacer**:
1. Grid search pequeño sobre la mejor arquitectura MLP (embed o interactions):
   ```
   hidden_dims: [[128,64], [192,96], [256,128], [256,128,64]]
   dropout: [0.2, 0.3, 0.4]
   lr: [1e-3, 5e-4, 2e-4]
   weight_decay: [1e-4, 5e-4, 1e-3]
   ```
2. Evaluar en val, reportar Spearman y R² para cada config
3. Seleccionar la mejor config y re-evaluar en test UNA sola vez

**Output esperado**:
- `final/analysis/hp_search/hp_search_results.csv`
- Mejor configuración documentada
- Si mejora → actualizar la tabla comparativa

**Regla**: si el mejor HP no mejora más de 0.005 Spearman sobre el default, reportar "la MLP es robusta a la configuración" y usar el default.

### Tarea 2B: Redacción del Informe II (miércoles-sábado)
**Prioridad**: 🔴 CRÍTICA — es un entregable con fecha
**Tiempo estimado**: 12-16 horas de redacción

Seguir la estructura definida en `estructura_informe_ii.md`:
1. Resumen ejecutivo (1 pág)
2. Seguimiento de planificación (2 págs)
3. Metodología final (4-5 págs)
4. Resultados (4-5 págs) — con tabla comparativa, embeddings, clean vs chaotic
5. Conclusiones provisionales (1-2 págs)
6. Trabajo restante (0.5 págs)
7. Bibliografía (1 pág)

**Figuras mínimas a incluir**:
1. Tabla comparativa de modelos (la central)
2. t-SNE/UMAP de embeddings coloreado por arquetipo
3. Training curves (MLP OneHot + MLP Embed)
4. SHAP beeswarm o top features
5. Clean vs Chaotic tabla
6. Scatter true-vs-pred del GBT

---

## Semana 3 — 25-31 mayo

**Objetivo**: Prototipo CLI + ablation study.

### Tarea 3A: Prototipo CLI pulido
**Prioridad**: 🟡 Media — entregable aplicado del TFG
**Tiempo estimado**: 6-8 horas
**Script**: `final/scripts/predict_cli.py` (ya existe, necesita pulir)

**Qué hacer**:
1. Interfaz: usuario introduce 10 campeones + side
2. El CLI carga el mejor modelo (GBT) y predice el score
3. Traducción a texto interpretable:
   - Score < 0.25: "Perfil de laning — el support tiende a quedarse en botlane"
   - Score 0.25-0.40: "Perfil mixto — roaming ocasional pero anclado a bot"
   - Score 0.40-0.55: "Perfil de roaming moderado — rotaciones frecuentes esperadas"
   - Score > 0.55: "Perfil de roaming intenso — el support abandona bot activamente"
4. Mostrar también: confianza, campeones similares (de embeddings), top features que influyen (SHAP local)
5. Modo interactivo y modo batch (archivo de drafts)

**Output esperado**: CLI funcional que demuestre la aplicación práctica del TFG.

### Tarea 3B: Ablation study del GBT
**Prioridad**: 🟢 Baja — nice to have para la memoria
**Tiempo estimado**: 3-4 horas
**Script**: `final/scripts/19_gbt_ablation.py`

**Qué hacer**:
1. Entrenar GBT con subconjuntos de features:
   - Solo campeones (10 features)
   - Solo ally (5 features)
   - Solo enemy (5 features)
   - Solo botlane (4 features: ally/enemy support + ADC)
   - Solo support+side (3 features)
   - Campeones + summoner spells (30 features)
   - Todo (31 features)
2. Comparar R²/Spearman para cada subset

**Output esperado**:
- `final/analysis/ablation/gbt_ablation.csv`
- Tabla que muestra cuánto contribuye cada grupo de features
- Hallazgo esperado: el campeón support aliado domina, seguido del ADC aliado y la composición enemiga

---

## Semana 4 — 1-7 junio

**Objetivo**: Iniciar redacción de la memoria final.

### Tarea 4A: Estructura de la memoria
- Definir capítulos y secciones
- Reutilizar contenido del Informe I y II como base
- Estructura típica de TFG de ingeniería:
  1. Introducción y motivación
  2. Estado del arte
  3. Metodología
  4. Implementación
  5. Resultados experimentales
  6. Conclusiones y trabajo futuro
  7. Bibliografía
  + Anexos (código, datos, CLI)

### Tarea 4B: Redacción de capítulos 1-4
- Cap 1: adaptar de los Informes I y II
- Cap 2: ampliar bibliografía con papers de sports analytics, GBT, embeddings
- Cap 3-4: descripción técnica del pipeline, etiqueta, modelos

---

## Semana 5 — 8-14 junio (sábado = propuesta memoria final)

### Tarea 5A: Redacción de capítulos 5-7
- Cap 5: resultados — la tabla comparativa es el núcleo
- Cap 6: conclusiones — las 5 conclusiones definitivas
- Cap 7: bibliografía actualizada

### Tarea 5B: Figuras definitivas
- Regenerar todas las figuras en alta resolución
- Formato consistente (misma paleta, mismos ejes, misma tipografía)

### Tarea 5C: Revisión y entrega
- Revisión completa de coherencia
- Entrega de la propuesta de memoria el 14 de junio

---

## Reglas del roadmap

> [!WARNING]
> **Regla 1**: Cada tarea debe pasar el test: *¿añade una fila a la tabla, una figura a la memoria, o un párrafo a las conclusiones?* Si no → no hacerla.

> [!WARNING]
> **Regla 2**: No iterar más la etiqueta. Está cerrada desde el 15 de mayo.

> [!WARNING]
> **Regla 3**: El Informe II del 24 de mayo es intocable. Si una tarea experimental no está lista para el 22 de mayo, se excluye del informe y se documenta como "trabajo en curso".

> [!IMPORTANT]
> **Regla 4**: Los resultados no son "pobres". R²=0.16 al 93% del techo ICC es un resultado fuerte que necesita el framing correcto, no más experimentos.

---

## Checklist de entregables

| Entregable | Fecha | Estado |
|---|---|---|
| Análisis de embeddings (t-SNE/UMAP) | 18 mayo | ✅ 16 mayo |
| MLP per-role + interacciones | 18 mayo | ✅ 16 mayo |
| HP search MLP | 20 mayo | ⬜ |
| Informe de Progreso II | 24 mayo | ⬜ |
| CLI prototipo | 31 mayo | ⬜ |
| Ablation study GBT | 31 mayo | ⬜ |
| Memoria capítulos 1-4 | 7 junio | ⬜ |
| Memoria capítulos 5-7 + figuras | 13 junio | ⬜ |
| Propuesta de memoria final | 14 junio | ⬜ |
