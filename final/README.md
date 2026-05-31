# Final — Fase final del TFG

Carpeta de trabajo definitiva. Todo el análisis, modelado y resultados finales
se generan aquí. Las carpetas anteriores (`ProgresoActual`, `ProgresoActual2`)
quedan congeladas como referencia histórica.

## Entradas externas (solo lectura)

- `ProgresoActual/data/clean/features/draft_features.parquet` — features de draft
- `ProgresoActual/data/clean/frame_state/support_frame_state.parquet` — cache de frames
- `ProgresoActual2/data/clean/scores/support_scores_v5_geometry_m12.parquet` — etiqueta v5
- `ProgresoActual/references/manual_support_champion_reference.csv` — referencia experta

## Regla

No se escribe nada fuera de `final/`. No se modifica `ProgresoActual/` ni
`ProgresoActual2/`. Si hace falta algo de allí, se copia aquí o se lee
directamente.

## Orden de scripts

```
01_prepare_final_dataset.py    — join + split persistido train/val/test
02_baseline_champion_mean.py   — baseline trivial: media por campeón
03_train_gbt.py                — HistGradientBoostingRegressor
04_train_mlp.py                — MLP OneHot (reproducción de la actual)
05_empirical_ceiling.py        — techo predictivo desde composición
06_feature_importance.py       — importancia de features desde GBT
07_model_comparison.py         — tabla comparativa final de todos los modelos
09_qualitative_case_audit.py   — auditoría cualitativa de errores
10_label_error_diagnostics.py  — diagnóstico de errores de etiqueta
12_build_support_event_context.py — extracción de contexto de eventos
13-15                          — iteración de etiqueta (resultado: robustez)
16_add_chaos_filter_weights.py — filtrado de partidas caóticas + sample_weight
predict_cli.py                 — prototipo terminal con el mejor modelo
```
