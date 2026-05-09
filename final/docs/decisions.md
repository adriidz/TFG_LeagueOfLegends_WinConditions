# Decisiones — Fase Final

Log breve de decisiones tomadas durante la fase final del TFG.

## Etiqueta

Etiqueta base: `support_roam_score_v5_geometry` (geometría v5, gamma 0.75).

Se comparan dos formulaciones del target:

1. **raw** — el score v5 tal cual. Distribución sesgada a la izquierda
   (media ~0.39, mediana ~0.39).
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

Un único split train/val persistido por `match_id`, reutilizado por todos los
modelos. Esto garantiza comparaciones justas.

## Etiquetas de jungla y equipo

Descartadas para la fase final. El TFG se centra en support para obtener
conclusiones más profundas y defendibles.
