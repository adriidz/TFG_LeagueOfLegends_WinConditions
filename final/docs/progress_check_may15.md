# Progress Check — 15 mayo 2026

Revisión del estado de `final/` medido contra las recomendaciones de
`analysis_results.md` (9 mayo).

## 1. Scorecard de recomendaciones

| Recomendación del 9 mayo | Estado | Veredicto |
|---|---|---|
| Win 1: Baseline de media por campeón | ✅ Hecho | `baselines/champion_mean_metrics.json` |
| Win 2: HistGBT como comparación | ✅ Hecho | `models/gbt/` con raw + quantile |
| Win 3: Feature importance | ✅ Hecho | `analysis/feature_importance/` |
| Win 4: Techo empírico (ICC) | ✅ Hecho | `analysis/ceiling/ceiling_summary.md` |
| Win 5: Ejecutar OAT en local | ⚠️ No ejecutado | Justificado por cluster |
| Comparación de modelos en tabla final | ✅ Hecho | `analysis/model_comparison/comparison_tables.md` |
| SHAP analysis | ✅ Hecho | `analysis/shap/` |
| Auditoría cualitativa | ✅ Hecho | `analysis/qualitative_case_audit/` |
| GBT enriched (arquetipos) | ✅ Hecho | `models/gbt_enriched/` — no mejora |
| GBT interactions (Pair TE) | ✅ Hecho | `models/gbt_interactions/` — mejora marginal |
| Dejar de iterar la etiqueta | ❌ No seguido | Scripts 12-15: v6, v7, label sweep |
| Diversificar modelos, no etiquetas | ❌ No seguido | `models/mlp_onehot/` y `models/mlp_embeddings/` vacíos |
| Abandonar jungla/equipo | ✅ Seguido | Correcto |
| Reencuadrar TFG como cuantificación | ✅ En progreso | `docs/progress_ii_empirical_findings.md` |

Balance: 10 de 14 recomendaciones seguidas. Las no seguidas son las más
importantes para la narrativa: diversidad de modelos y foco en la etiqueta
existente.

## 2. Lo que está bien

- Dataset final con split persistido train/val/test por match_id.
- Baselines Global Mean + Champion Mean implementadas y comparadas.
- HistGBT base: R²=0.161, Spearman=0.388 en test.
- HistGBT + Archetypes: confirma que arquetipos no mejoran.
- HistGBT + Pair TE: mejora marginal.
- Quantile comparison: la escala del target no cambia la conclusión.
- Techo empírico ICC: botlane+side R²≈0.17. HistGBT al 95% del techo.
- SHAP: importancia global, beeswarm, dependencias, waterfalls.
- Auditoría cualitativa: 40 casos, reconstrucción perfecta del score.
- Métricas prácticas: within-0.20≈74%, adjacent bin accuracy≈97%.
- Documentación y narrativa del Informe II bien orientadas.

## 3. Lo que está mal

### Iteración innecesaria de la etiqueta (scripts 12-15)

Los scripts 12-15 construyen variantes v6 (eventos agregados) y v7 (event
snapshots) del score, más un sweep de variantes con ablaciones de features.
El resultado del sweep confirma lo que se predecía: la mejor variante v6
mejora 0.002 de Spearman sobre v5. Cambiar la etiqueta no cambia la señal
capturable desde el draft.

Este trabajo consumió tiempo que debería haberse dedicado a entrenar y
comparar modelos neurales.

### Ausencia de modelos neurales en final/

Las carpetas `models/mlp_onehot/` y `models/mlp_embeddings/` contienen solo
`.gitkeep`. No hay MLP ni embeddings entrenados sobre el split final. La
tabla de comparación tiene solo baselines y HistGBT.

Esto es un hueco crítico: el TFG necesita la comparación tabular vs neural
para cerrar la narrativa.

## 4. Acciones derivadas de este check

1. Implementar chaos filtering para la etiqueta (ver `label_quality.md`).
2. Entrenar MLP OneHot y MLP Embeddings sobre el split final.
3. Regenerar la tabla de comparación con todos los modelos.
4. Completar el Informe de Progreso II con los nuevos resultados.
5. Limpiar directorios `_tmp_*`.

## 5. Nota sobre los scripts 12-15

Los scripts 12-15 no se eliminan. El label variant sweep aporta un hallazgo
rescatable: la etiqueta es robusta a cambios de definición. Este resultado
se menciona brevemente en la memoria (una frase en la sección de validación
de la etiqueta) pero no es un eje experimental del TFG.
