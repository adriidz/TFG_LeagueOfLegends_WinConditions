# Decisiones — Fase Final

Log breve de decisiones tomadas durante la fase final del TFG.

## Etiqueta

Etiqueta base: `support_roam_score_v5_geometry` (geometría v5, gamma 0.75).

Se comparan dos formulaciones del target:

1. **raw** — el score v5 tal cual. Distribución no uniforme, concentrada en
   rango medio-bajo (media ~0.39, mediana ~0.39).
2. **quantile zero-preserved** — transformación quantile fitteada SOLO en train.
   Distribución uniforme, preserva ranking y mantiene raw=0 como cero.

La comparación entre ambas es un resultado experimental: si el modelo mejora con
quantile, demuestra que la distribución del target condiciona el aprendizaje.
Si no mejora, la señal predictiva del draft es realmente limitada independiente
de la escala.

Precaución: el `QuantileTransformer` debe ajustarse exclusivamente en train y
aplicarse después a val/test para evitar leakage.

## Modelos a comparar

1. **Media por campeón** — baseline trivial para contextualizar la MLP
2. **HistGradientBoostingRegressor** — captura interacciones automáticamente
3. **MLP OneHot** — reproducción de la baseline actual
4. **MLP con feature engineering** — matchups, arquetipos, sinergias
5. **MLP con embeddings** — si da tiempo

## Split

Split a tres niveles por `match_id` con `GroupShuffleSplit`:

- **train** (70%) — entrenamiento de todos los modelos
- **val** (15%) — selección de hiperparámetros, iteración, desarrollo
- **test** (15%) — evaluación final INTOCABLE, solo para la tabla de la memoria

Todos los modelos usan exactamente el mismo split. Se persiste en
`final/data/training/{train,val,test}.parquet`.

## Etiquetas de jungla y equipo

Descartadas para la fase final. El TFG se centra en support para obtener
conclusiones más profundas y defendibles.

## Filtrado de ruido por caos (15 mayo 2026)

La auditoría cualitativa reveló que 17/20 top errors del modelo son partidas
con `chaotic_early_game` donde la botlane colapsó violentamente. El score
alto en esas partidas no refleja predisposición del draft al roaming sino
ejecución caótica que el modelo no puede predecir.

Decisión: NO se cambia la fórmula del score v5. En su lugar se añade
`chaos_flag` + `sample_weight` al entrenamiento:

- `chaos_flag = True` si `support_deaths + adc_deaths >= 6`, o
  `adc_deaths >= 5`, o `support_deaths >= 4` sin acciones activas fuera de
  bot.
- `sample_weight = 0.2` para partidas caóticas, `1.0` para las demás.
- `min_support_frames` sube de 2 a 3.

Justificación: cambiar la fórmula produce variantes que correlacionan 0.99
con v5 (demostrado en el label sweep con 15 variantes de v6). La señal está
limitada por la resolución minutal de la API. Filtrar el ruido es más
efectivo y no cambia la definición de lo que se mide.

Implementación: `scripts/16_add_chaos_filter_weights.py`.
Documentación: `docs/label_quality.md`.
