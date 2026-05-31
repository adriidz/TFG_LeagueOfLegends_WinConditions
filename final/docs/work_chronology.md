# Cronología completa del trabajo post-Informe I

**Periodo cubierto**: 9 mayo → 23 mayo 2026
**Punto de partida**: [analysis_results.md](file:///c:/Users/adria/Desktop/TFG/final/docs/analysis_results.md) — análisis crítico del estado del TFG tras entregar Informe de Progreso I (27 abril)

**Estado al comenzar**: Una MLP con R²=0.13, sin baselines, sin techo empírico, sin comparación con modelos alternativos.

---

## Bloque 0 — 28 abril-8 mayo: Preparación OAT, geometría v5 y prototipo inicial

> Este bloque cubre el trabajo realizado entre la entrega del Informe de Progreso I (27 abril) y la creación de la fase `final/` el 9 de mayo. El avance no siguió literalmente el planning previsto: el tuning OAT quedó preparado pero no ejecutado por el bloqueo del cluster, y el trabajo local se redirigió a reforzar la etiqueta, preparar un prototipo aplicado y ordenar el repositorio.

### Trabajo realizado

#### 0.1 Preparación del tuning OAT support-only
- **Artefacto**: [runs_manifest.csv](file:///c:/Users/adria/Desktop/TFG/ProgresoActual/OAT/support_oat_full_m12/experiments/runs_manifest.csv) — 28 abril-3 mayo
- **Docs**: [support_oat_tuning.md](file:///c:/Users/adria/Desktop/TFG/ProgresoActual/docs/support_oat_tuning.md), [oat_manifest_explanation.md](file:///c:/Users/adria/Desktop/TFG/ProgresoActual/docs/oat_manifest_explanation.md)
- **Qué hizo**: dejó diseñado un experimento one-at-a-time con 20 runs para variar pesos de etiqueta, ventanas temporales e hiperparámetros de la MLP.
- **Estado**: preparado y versionado, pero no cerrado como resultado experimental porque la ejecución completa dependía del cluster.
- **Hallazgo/decisión**: el OAT se mantuvo como comprobación planificada, pero dejó de ser el único eje del Informe II.

#### 0.2 Primer prototipo terminal de champ select
- **Script**: [predict_support_roam_cli.py](file:///c:/Users/adria/Desktop/TFG/ProgresoActual/scripts/predict_support_roam_cli.py) — antes del 7 mayo
- **Doc**: [terminal_prototype.md](file:///c:/Users/adria/Desktop/TFG/ProgresoActual/docs/terminal_prototype.md)
- **Qué hizo**: permitió introducir un draft aliado/enemigo y obtener una predicción interpretable de tendencia de roaming usando la baseline MLP full `m12`.
- **Importancia**: adelantó un entregable aplicado que originalmente estaba previsto para junio y demostró que los artefactos entrenados podían reutilizarse en inferencia.
- **Limitación**: todavía usaba la baseline MLP previa, no el modelo final ni los resultados posteriores de `final/`.

#### 0.3 Sandbox `ProgresoActual2` para geometría v5
- **Código**: [geometry_v5_manual.py](file:///c:/Users/adria/Desktop/TFG/ProgresoActual2/src/geometry/geometry_v5_manual.py)
- **Config**: `ProgresoActual2/data/geometry/manual_geometry_v5_config.json`
- **Docs**: [geometry_v5_manual.md](file:///c:/Users/adria/Desktop/TFG/ProgresoActual2/docs/geometry_v5_manual.md), [geometry_v5_manual_annotation_workflow.md](file:///c:/Users/adria/Desktop/TFG/ProgresoActual2/docs/geometry_v5_manual_annotation_workflow.md)
- **Qué hizo**: creó una geometría manual del mapa más defendible para distinguir botlane, río, mid, dragón, junglas y bases.
- **Decisión**: `ProgresoActual` quedó como línea estable; `ProgresoActual2` funcionó como sandbox de Progreso II para experimentar con geometría y etiqueta.

#### 0.4 Análisis frame-level con geometría v5
- **Script**: [build_geometry_v5_frame_state_distributions.py](file:///c:/Users/adria/Desktop/TFG/ProgresoActual2/scripts/build_geometry_v5_frame_state_distributions.py) — 6 mayo
- **Doc**: [progress_2026-05-06_geometry_v5_quantile.md](file:///c:/Users/adria/Desktop/TFG/ProgresoActual2/docs/progress_2026-05-06_geometry_v5_quantile.md)
- **Resultado**: en ventana `m5_12`, `support_in_bot_context_v5_share` ≈ **0.742** frente a legacy ≈ **0.760**.
- **Hallazgo**: la geometría v5 mantuvo una lectura parecida a la anterior, pero algo más restrictiva y semánticamente controlada.

#### 0.5 Nueva etiqueta `support_roam_score_v5_geometry`
- **Script**: [build_support_roam_score_v5_distribution.py](file:///c:/Users/adria/Desktop/TFG/ProgresoActual2/scripts/build_support_roam_score_v5_distribution.py) — 6 mayo
- **Output**: `ProgresoActual2/data/clean/scores/support_scores_v5_geometry_m12.parquet`
- **Fórmula**: `0.45 * outside_ratio_v5 + 0.35 * far_ratio_v5 + 0.20 * xp_gap_v5`, con transformación `raw ** 0.75`.
- **Resultado**: **337,104 filas**, coverage ≈ **0.9999**, media ≈ **0.393**, mediana ≈ **0.390**, correlación fila vs v3 ≈ **0.941**.
- **Hallazgo**: v5 fue un refinamiento semántico, no una ruptura del pipeline. Conservó el ranking global y mejoró la defendibilidad de la etiqueta.

#### 0.6 Transformación quantile zero-preserved
- **Script**: [build_support_roam_score_v5_quantile_labels.py](file:///c:/Users/adria/Desktop/TFG/ProgresoActual2/scripts/build_support_roam_score_v5_quantile_labels.py) — 6 mayo
- **Output**: `ProgresoActual2/data/clean/scores/support_scores_v5_quantile_m12.parquet`
- **Qué hizo**: exploró una escala relativa del target preservando los casos `raw == 0` como cero.
- **Decisión metodológica**: la versión quantile podía usarse como experimento de aprendizaje/ranking, pero para evaluación estricta el transformador debía ajustarse solo con train para evitar leakage. Esta idea se integró después en `final/scripts/01_prepare_final_dataset.py`.

#### 0.7 Limpieza y reorganización del repositorio
- **Commit local de referencia**: `Limpieza` — 7 mayo
- **Qué hizo**: retiró del árbol activo artefactos/caches que no debían representar código vivo, mantuvo `src/01_data_collection/` como recolector útil y separó:
  - `PropuestaInicial/` como archivo documental;
  - `ProgresoActual/` como pipeline support-only estable;
  - `ProgresoActual2/` como sandbox experimental;
  - `final/` como fase final creada después.
- **Importancia**: redujo mezcla entre experimentos antiguos, código vivo y artefactos generados, preparando el salto a una fase final reproducible.

#### 0.8 Documento provisional para Informe II
- **Doc**: [progreso_ii_provisional.md](file:///c:/Users/adria/Desktop/TFG/ProgresoActual/docs/progreso_ii_provisional.md) — 7 mayo
- **Qué contiene**: síntesis del periodo 28/04-07/05, comparación contra planning, decisiones metodológicas y siguientes pasos.
- **Lectura**: deja claro que el periodo temprano no completó el tuning ni embeddings, pero sí produjo tres avances de valor: OAT preparado, etiqueta/geometría refinada y prototipo terminal inicial.

---

## Bloque 1 — 9-10 mayo: Baselines y techo empírico

> El diagnóstico de `analysis_results.md` identificó que los resultados del Informe I no eran defendibles sin contexto. Se priorizó construir baselines y un techo empírico antes de cualquier otra cosa.

### Trabajo realizado

#### 1.1 Regeneración del dataset final (383K observaciones)
- **Script**: [00_regenerate_inputs.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/00_regenerate_inputs.py) — 10 mayo
- **Output**: `final/data/frame_state/`, `final/data/features/`, `final/data/scores/`
- **Qué hizo**: regeneró frame_state, draft_features y scores v5 con el dataset ampliado a ~191K partidas (383,247 observaciones)

#### 1.2 Baseline Champion Mean
- **Script**: [02_baseline_champion_mean.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/02_baseline_champion_mean.py) — 10 mayo
- **Output**: `final/baselines/champion_mean_metrics.json`
- **Resultado**: **R²=0.125, Spearman=0.336, MAE=0.144**
- **Hallazgo**: un simple lookup por campeón support explica el 77% de la varianza que captura el mejor modelo. Dato esencial para contextualizar la MLP.

#### 1.3 HistGBT base
- **Script**: [03_train_gbt.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/03_train_gbt.py) — 10 mayo (primera versión)
- **Output**: `final/models/gbt/`
- **Resultado**: **R²=0.160, Spearman=0.387, MAE=0.141**
- **Hallazgo**: los árboles superan a la MLP desde el primer intento. Capturan interacciones de draft de forma combinatoria.

#### 1.4 Techo empírico ICC
- **Script**: [05_empirical_ceiling.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/05_empirical_ceiling.py) — 10 mayo
- **Output**: [ceiling_summary.md](file:///c:/Users/adria/Desktop/TFG/final/analysis/ceiling/ceiling_summary.md)
- **Resultado principal**: ICC(botlane+side) = **0.139**, R²(group_mean) = **0.173**
- **Hallazgo central**: el HistGBT está al **93% del techo teórico**. Queda ~1 punto de R² de margen. No hay arquitectura que cambie esto sustancialmente.
- **Descomposición del techo**:

| Agrupación | ICC | R²(group mean) |
|---|---|---|
| Support champion solo | 0.121 | 0.121 |
| Botlane (sup+ADC) | 0.139 | 0.161 |
| Botlane + side | 0.139 | **0.173** |
| Support archetype | 0.084 | 0.081 |

#### 1.5 Feature importance por permutación
- **Script**: [06_feature_importance.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/06_feature_importance.py) — 10 mayo
- **Output**: `final/analysis/feature_importance/`
- **Resultado**: `ally_utility_champion_id` domina (importancia = **0.226**), seguido de ADC aliado (0.024) y support enemigo (0.017). Summoner spells y side ≈ 0.

#### 1.6 Label health check
- **Output**: `final/analysis/label_health/` — 10 mayo
- **Qué hizo**: verificación de distribución, nulos, rango [0,1] del score v5 en el dataset ampliado.

---

## Bloque 2 — 11-12 mayo: GBT enriched, SHAP y auditoría cualitativa

### Trabajo realizado

#### 2.1 HistGBT + Arquetipos
- **Script**: [03b_train_gbt_enriched.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/03b_train_gbt_enriched.py) — 11 mayo
- **Output**: `final/models/gbt_enriched/`
- **Resultado**: R²=0.161 (+0.001 vs base)
- **Hallazgo**: los arquetipos no añaden señal que el GBT no capture ya desde los IDs de campeón.

#### 2.2 HistGBT + Pair Target Encoding
- **Script**: [03c_train_gbt_interactions.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/03c_train_gbt_interactions.py) — 11 mayo
- **Output**: `final/models/gbt_interactions/`
- **Resultado**: **R²=0.161, Spearman=0.388** (mejora marginal)
- **Hallazgo**: el Target Encoding de pares support-ADC aporta ~0.001 R². La señal de interacción es real pero mínima.

#### 2.3 Model comparison framework
- **Script**: [07_model_comparison.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/07_model_comparison.py) — 11 mayo (primera versión)
- **Output**: `final/analysis/model_comparison/`
- **Qué hizo**: tabla comparativa unificada con todas las métricas (R², Spearman, MAE, RMSE, within±0.10, within±0.20, QWK) para todos los modelos sobre el mismo test set.

#### 2.4 Documentación de hallazgos empíricos
- **Doc**: [progress_ii_empirical_findings.md](file:///c:/Users/adria/Desktop/TFG/final/docs/progress_ii_empirical_findings.md) — 11 mayo
- **Qué contiene**: primer draft de la narrativa empírica: ceiling, baselines, comparación GBT vs MLP.

#### 2.5 SHAP analysis
- **Script**: [08_shap_analysis.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/08_shap_analysis.py) — 12 mayo
- **Output**: `final/analysis/shap/` (beeswarm, bar plots, dependencias categóricas, waterfalls locales)
- **Hallazgo**: confirma la dominancia del campeón support. Las dependencias categóricas muestran que Bard/Pyke/Alistar empujan SHAP hacia arriba; Yuumi/Lulu/Sona hacia abajo. Coherente con el ranking experto.

#### 2.6 Error analysis y label diagnostics
- **Scripts**: [09_error_analysis.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/09_error_analysis.py), [10_label_error_diagnostics.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/10_label_error_diagnostics.py) — 11-12 mayo
- **Outputs**: `final/analysis/error_analysis/`, `final/analysis/label_error_diagnostics/`

#### 2.7 Auditoría cualitativa de 40 casos
- **Script**: [09_qualitative_case_audit.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/09_qualitative_case_audit.py) — 12 mayo
- **Output**: `final/analysis/qualitative_case_audit/` (40 reportes, mapas, timelines, case_notes.md)
- **Resultado clave**: **17/20 top errors** tienen tag `chaotic_early_game`. Los errores grandes no son fallos del modelo sino partidas donde la botlane colapsó. Ejemplo: Yuumi+Smolder vs Pyke+Vel'Koz, pred=0.209, real=1.000 (Smolder muere 7 veces).
- **Reconstrucción perfecta**: max_score_reconstruction_delta = **0.0** (la etiqueta se reproduce exactamente desde sus componentes).

#### 2.8 Qualitative match context
- **Script**: [11_qualitative_match_context.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/11_qualitative_match_context.py) — 12 mayo
- **Output**: `final/analysis/qualitative_match_context/`
- **Qué hizo**: contexto completo de partidas para los 40 casos auditados (eventos raw de timeline, KDA, items, etc.)

---

## Bloque 3 — 14-15 mayo: Label sweep, chaos filtering y re-entrenamiento

### Trabajo realizado

#### 3.1 Event context builder
- **Script**: [12_build_support_event_context.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/12_build_support_event_context.py) — 14 mayo (primera versión; actualizada 23 mayo)
- **Output**: `final/data/event_context/`
- **Qué hizo**: extrae eventos de timeline (kills, muertes, objetivos) y clasifica si ocurrieron fuera de botlane. Base para v6/v7/chaos filtering.

#### 3.2 Score v6 (eventos agregados)
- **Script**: [13_build_support_roam_score_v6_events.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/13_build_support_roam_score_v6_events.py) — 14 mayo
- **Resultado**: 15 variantes de la fórmula del score con canales de eventos. **Todas correlacionan ≥0.99 con v5**. Mejor mejora: +0.002 Spearman.
- **Hallazgo**: cambiar la fórmula no cambia la señal. La resolución minutal de la API (~8 frames) es la limitación fundamental.

#### 3.3 Label variant sweep
- **Script**: [14_train_label_variant_sweep.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/14_train_label_variant_sweep.py) — 14 mayo
- **Output**: `final/analysis/label_variant_sweep/`, `final/models/label_variant_sweep/`
- **Resultado**: 15 variantes entrenadas y evaluadas. Confirma robustez de v5.

#### 3.4 Score v7 (event snapshots)
- **Script**: [15_build_support_roam_score_v7_event_snapshots.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/15_build_support_roam_score_v7_event_snapshots.py) — 15 mayo
- **Resultado**: correlación v5↔v7 ≈ **0.99**. Los event snapshots no añaden resolución útil.

#### 3.5 Chaos filtering + sample weights
- **Script**: [16_add_chaos_filter_weights.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/16_add_chaos_filter_weights.py) — 15 mayo
- **Output**: splits actualizados en `final/data/training/` con `chaos_flag`, `sample_weight`
- **Decisión documentada en**: [label_quality.md](file:///c:/Users/adria/Desktop/TFG/final/docs/label_quality.md), [decisions.md](file:///c:/Users/adria/Desktop/TFG/final/docs/decisions.md)
- **Estadísticas**: **26.5%** de partidas marcadas como caóticas → reciben sample_weight=0.2
- **Lógica del chaos_flag**: `(sup_deaths + adc_deaths ≥ 6) | (adc_deaths ≥ 5) | (sup_deaths ≥ 4 & 0 acciones fuera de bot)`

#### 3.6 Re-entrenamiento de todos los modelos con chaos filtering
- **Scripts**: [03_train_gbt.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/03_train_gbt.py), [04a_train_mlp_onehot.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/04a_train_mlp_onehot.py), [04b_train_mlp_embed.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/04b_train_mlp_embed.py) — 15 mayo
- **Outputs**: `final/models/gbt/`, `final/models/mlp_onehot/`, `final/models/mlp_embed/`
- **Resultados en test (con sample_weight)**:

| Modelo | R² | Spearman | MAE |
|---|---|---|---|
| HistGBT + Pair TE | **0.161** | **0.388** | **0.141** |
| HistGBT base | 0.160 | 0.387 | 0.141 |
| MLP OneHot | 0.155 | 0.381 | 0.141 |
| MLP Embed (dim=16) | 0.150 | 0.376 | 0.142 |
| Champion Mean | 0.125 | 0.336 | 0.144 |

#### 3.7 Clean vs chaotic evaluation
- **Output**: [clean_vs_chaotic.md](file:///c:/Users/adria/Desktop/TFG/final/analysis/clean_vs_chaotic/clean_vs_chaotic.md) — 15 mayo
- **Resultado**:

| Subset | n | R² | Spearman |
|---|---|---|---|
| all | 57,468 | 0.160 | 0.387 |
| clean (73.3%) | 42,147 | **0.171** | 0.397 |
| chaotic (26.7%) | 15,321 | 0.122 | 0.363 |

- **Hallazgo**: en partidas limpias, el GBT alcanza R²=0.171, prácticamente idéntico al techo ICC (0.173).

#### 3.8 Progress check y documentación
- **Doc**: [progress_check_may15.md](file:///c:/Users/adria/Desktop/TFG/final/docs/progress_check_may15.md) — 15 mayo
- **Scorecard**: 10 de 14 recomendaciones de `analysis_results.md` seguidas.

---

## Bloque 4 — 16 mayo: Embeddings, MLP per-role, HP search y roadmap

### Trabajo realizado

#### 4.1 Embedding analysis (t-SNE/UMAP)
- **Script**: [17_embedding_analysis.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/17_embedding_analysis.py) — 16 mayo
- **Output**: `final/analysis/embedding_analysis/`
- **Resultados**:
  - Silhouette por arquetipo: **-0.146** → no hay clusters por categoría humana
  - Silhouette por clase Data Dragon: **-0.074**
  - Top-5 vecinos del mismo arquetipo: **4%** → básicamente azar
  - Correlación distancia↔roam score: Pearson **r=0.166** (p=3.2e-06)
- **Hallazgo**: los embeddings codifican gradación continua de roaming, pero no reproducen taxonomías humanas. t-SNE muestra gradiente suave.

#### 4.2 MLP Per-Role + Interactions
- **Script**: [04c_train_mlp_per_role.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/04c_train_mlp_per_role.py) — 16 mayo
- **Output**: `final/models/mlp_per_role/`, `final/analysis/embedding_analysis_per_role_ally_utility/`
- **Resultado**: **R²=0.154, Spearman=0.381** (mejora sobre MLP Embed +0.004, pero no cierra gap con GBT)
- **Hallazgo**: dar a la MLP embeddings separados por rol + interacciones explícitas no basta para superar árboles.

#### 4.3 HP search de MLP (108 configuraciones)
- **Script**: [18_mlp_hp_search.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/18_mlp_hp_search.py) — 16 mayo
- **Output**: [hp_search_summary.md](file:///c:/Users/adria/Desktop/TFG/final/analysis/hp_search/hp_search_summary.md)
- **Resultado**: mejor configuración mejora **+0.005 Spearman** sobre default → por debajo del umbral de mejora significativa
- **Decisión**: la MLP es robusta a la configuración; se conserva el default.

#### 4.4 Tabla comparativa final completa
- **Script**: [07_model_comparison.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/07_model_comparison.py) — actualizado 16 mayo
- **Output**: `final/analysis/model_comparison/`
- **Tabla final de 7 modelos + 2 baselines** sobre test set intocable:

| Modelo | R² | Spearman | MAE |
|---|---|---|---|
| **ICC Ceiling** | **0.173** | — | — |
| HistGBT + Pair TE | **0.161** | **0.388** | **0.141** |
| HistGBT + Archetypes | 0.161 | 0.388 | 0.141 |
| HistGBT base | 0.160 | 0.387 | 0.141 |
| MLP OneHot | 0.155 | 0.381 | 0.141 |
| MLP Per-Role + Interactions | 0.154 | 0.381 | 0.141 |
| MLP Embed (compartido) | 0.150 | 0.376 | 0.142 |
| Champion Mean | 0.125 | 0.336 | 0.144 |
| Global Mean | 0.000 | — | 0.155 |

#### 4.5 Training curves
- **Output**: `final/analysis/training_curves/` — 16 mayo
- Plots de loss train/val para MLP OneHot, MLP Embed, MLP Per-Role.

#### 4.6 Roadmap final y estructura del Informe II
- **Docs**: [roadmap_final.md](file:///c:/Users/adria/Desktop/TFG/final/docs/roadmap_final.md), [estructura_informe_ii.md](file:///c:/Users/adria/Desktop/TFG/final/docs/estructura_informe_ii.md) — 16 mayo
- **Qué contiene**: planning revisado hasta 14 junio, estructura detallada del Informe II con secciones y figuras mínimas.

---

## Bloque 5 — 19-20 mayo: CLI prototipo y HP search final

### Trabajo realizado

#### 5.1 CLI prototipo
- **Script**: [predict_cli.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/predict_cli.py) — 19 mayo (39KB)
- **Qué hace**: usuario introduce 10 campeones + side → carga el GBT → predice score → traduce a franjas interpretables (laning / mixto / roaming moderado / roaming intenso) + muestra campeones similares y SHAP local.

#### 5.2 HP search finalizado
- **Output**: `final/analysis/hp_search/` — 20 mayo (evaluación final en test)
- **Resultado final en test**: MLP Per-Role + Interactions HP Best: **R²=0.155, Spearman=0.384**. Confirma: la MLP no supera al GBT.

#### 5.3 Preparación de entrevista/defensa
- **Scripts**: [prepare_entrevista.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/prepare_entrevista.py), [prepare_entrevista_pre_informe_i_style.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/prepare_entrevista_pre_informe_i_style.py) — 20 mayo
- **Doc**: [resumen_para_tutor.md](file:///c:/Users/adria/Desktop/TFG/final/docs/resumen_para_tutor.md) — 20 mayo

---

## Bloque 6 — 21 mayo: Borrador del Informe de Progreso II

### Trabajo realizado

#### 6.1 Borrador completo
- **Doc**: [informe_progreso_ii_borrador.md](file:///c:/Users/adria/Desktop/TFG/final/docs/informe_progreso_ii_borrador.md) — 21 mayo (433 líneas, 37KB)
- **Contenido**: 8 secciones completas incluyendo seguimiento de planificación, metodología, resultados, conclusiones provisionales y trabajo restante.

#### 6.2 Versión alternativa con tono del Informe I
- **Doc**: [informe_progreso_ii_borrador_tono_informe_i.md](file:///c:/Users/adria/Desktop/TFG/final/docs/informe_progreso_ii_borrador_tono_informe_i.md) — 21 mayo (42KB)
- **Qué es**: misma estructura, tono más académico y descriptivo para mantener consistencia con el primer informe.

---

## Bloque 7 — 23 mayo: Revisión crítica de la etiqueta (v8, v9)

> Este bloque surge de una revisión metodológica profunda: ¿la etiqueta v5 mide realmente lo que queremos predecir?

### Trabajo realizado

#### 7.1 Revisión crítica de la metodología de roaming
- **Doc**: [critical_methodology_review.md](file:///c:/Users/adria/Desktop/TFG/final/docs/critical_methodology_review.md) — 23 mayo
- **Diagnóstico**: v5 mide separación espacial (proxy de predisposición), pero está contaminada por muertes, recalls y caos. ¿Se puede mejorar requiriendo evidencia productiva?

#### 7.2 Score v8 (productive)
- **Script**: [19_build_support_roam_score_v8_productive.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/19_build_support_roam_score_v8_productive.py) — 23 mayo
- **Output**: `final/analysis/label_v8_productive/`
- **Fórmula**: 60% productive_event_score + 30% presence_score + 10% xp_gap, con cap a 0.35 si 0 eventos productivos
- **Resultado**: `productive_event_score ↔ v8_score`: Spearman **0.945** → dominado por ejecución
- **Problema**: 44.1% de matches tienen 0 eventos productivos → aplastados a 0.35

#### 7.3 Actualización de event context (buildings + plates)
- **Script**: [12_build_support_event_context.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/12_build_support_event_context.py) — actualizado 23 mayo
- **Qué se añadió**: `BUILDING_KILL` y `TURRET_PLATE_DESTROYED` como eventos productivos
- **Verificación**: cross-check de event_kind strings entre productor (12) y consumidor (19) → 4/4 match exacto

#### 7.4 Ceiling analysis para v8
- **Output**: `final/analysis/ceiling_v8_productive/` — 23 mayo
- **Resultado**: ICC(v8) peor que ICC(v5) → v8 es menos predecible desde draft

#### 7.5 Score v9 (balanced)
- **Script**: [20_build_support_roam_score_v9_balanced.py](file:///c:/Users/adria/Desktop/TFG/final/scripts/20_build_support_roam_score_v9_balanced.py) — 23 mayo
- **Output**: `final/analysis/label_v9_balanced/`
- **Fórmula**: backbone espacial (75% alive_outside_bot_ratio + 25% xp_gap) × modulador productivo × chaos dampener
- **Resultados**:

| Componente ↔ v9 score | Spearman |
|---|---|
| backbone (spatial) | **0.940** ← domina ✅ |
| v5 geometry | 0.742 |
| productive_event_score | 0.610 ← secundario ✅ |
| xp_gap | 0.563 |

- Expert alignment: Spearman v9 vs expert = **0.808** (v5 = 0.822)

#### 7.6 Ceiling analysis para v9
- **Output**: `final/analysis/ceiling_v9_balanced/` — 23 mayo
- **Resultado**: ICC(v9) comparable a v5 pero **no superior**

#### 7.7 Ceiling re-check para v5
- **Output**: `final/analysis/ceiling_v5_ceiling_check/` — 23 mayo
- **Qué hizo**: verificación con el mismo split y seed que v8/v9 para comparación justa

#### 7.8 Conclusión del bloque v8/v9
> v5 mide mejor predisposición espacial.
> v8 mide mejor roaming productivo observado (ejecución).
> v9 recupera parte de la predisposición pero no supera a v5 en ICC.
>
> **Conclusión**: la separación espacial cruda (v5) es la mejor proxy de predisposición al roaming desde draft. El "ruido" de muertes/base está parcialmente correlacionado con identidad del champion y por tanto contribuye a la señal predecible.

---

## Resumen de artefactos producidos

### Scripts nuevos creados (14 scripts)
| Script | Fecha | Propósito |
|---|---|---|
| `00_regenerate_inputs.py` | 10 may | Regenerar dataset ampliado |
| `02_baseline_champion_mean.py` | 10 may | Baseline trivial |
| `05_empirical_ceiling.py` | 10 may | Techo ICC |
| `06_feature_importance.py` | 10 may | Permutation importance |
| `03b_train_gbt_enriched.py` | 11 may | GBT + arquetipos |
| `03c_train_gbt_interactions.py` | 11 may | GBT + Pair TE |
| `08_shap_analysis.py` | 12 may | Explicabilidad SHAP |
| `09_qualitative_case_audit.py` | 12 may | Auditoría 40 casos |
| `12_build_support_event_context.py` | 14 may | Eventos timeline → parquet |
| `13, 14, 15` (v6, sweep, v7) | 14-15 may | Label variants |
| `16_add_chaos_filter_weights.py` | 15 may | Chaos filtering |
| `17_embedding_analysis.py` | 16 may | t-SNE/UMAP embeddings |
| `04c_train_mlp_per_role.py` | 16 may | MLP con embeddings por rol |
| `18_mlp_hp_search.py` | 16 may | HP search 108 configs |
| `19_build_support_roam_score_v8_productive.py` | 23 may | Label v8 |
| `20_build_support_roam_score_v9_balanced.py` | 23 may | Label v9 |

### Análisis producidos (20 carpetas)
`ceiling`, `ceiling_v5_ceiling_check`, `ceiling_v8_productive`, `ceiling_v9_balanced`, `clean_vs_chaotic`, `embedding_analysis`, `embedding_analysis_per_role_ally_utility`, `error_analysis`, `feature_importance`, `hp_search`, `label_error_diagnostics`, `label_health`, `label_v8_productive`, `label_v9_balanced`, `label_variant_sweep`, `model_comparison`, `qualitative_case_audit`, `qualitative_match_context`, `shap`, `training_curves`

### Documentación (14 archivos en `final/docs/`)
`analysis_results.md`, `critical_methodology_review.md`, `decisions.md`, `estructura_informe_ii.md`, `informe_progreso_ii_borrador.md`, `informe_progreso_ii_borrador_tono_informe_i.md`, `label_quality.md`, `progress_check_may15.md`, `progress_ii_empirical_findings.md`, `resumen_para_tutor.md`, `roadmap_final.md`, `support_roam_score_v8_productive.md`, `technical_spec.md`

---

## Las 5 conclusiones que salen de todo esto

1. **El draft contiene señal predictiva de roaming** (~16% varianza), pero limitada. El draft impone predisposición, no destino.

2. **Los árboles ganan a las redes neuronales** para esta tarea. HistGBT > MLP Per-Role > MLP Embed > MLP OneHot.

3. **El techo está prácticamente alcanzado** (93% del ICC). No hay arquitectura que cambie esto.

4. **La etiqueta v5 es robusta y es la mejor opción**: 15 variantes correlacionan ≥0.99, y las alternativas v8/v9 no mejoran la predictibilidad desde draft.

5. **Los errores grandes se explican por caos en partida**, no por fallos del modelo.
