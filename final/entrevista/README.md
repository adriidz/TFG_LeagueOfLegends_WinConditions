# Material per a l'Entrevista amb el Tutor

Directori preparat automàticament amb totes les gràfiques i visualitzacions
organitzades per tema. Cada carpeta conté les imatges rellevants.

## Estructura

### 📁 `00_summary/` – Resum general i taula comparativa
- `key_findings_infographic.png`
- `summary_table.png`

### 📁 `01_model_comparison/` – Comparació de models (R², Spearman, tolerància)
- `comparison_spearman.png`
- `comparison_tolerance_plot.png`

### 📁 `02_training_curves/` – Corbes d'entrenament MLP (overfitting check)
- `mlp_embed_quantile_curves.png`
- `mlp_embed_raw_curves.png`
- `mlp_onehot_quantile_curves.png`
- `mlp_onehot_raw_curves.png`
- `mlp_per_role_quantile_curves.png`
- `mlp_per_role_raw_curves.png`

### 📁 `03_ceiling/` – Sostre empíric (ICC / R² per agrupació)
- `ceiling_icc_r2_by_grouping.png`

### 📁 `04_shap/` – SHAP – importància de features i waterfall
- `shap_dependence_ally_bottom_champion_id.png`
- `shap_dependence_ally_utility_champion_id.png`
- `shap_summary_bar.png`
- `shap_summary_beeswarm.png`
- `shap_waterfall_case_01_high_pred_low_actual.png`
- `shap_waterfall_case_01_low_pred_high_actual.png`
- `shap_waterfall_case_02_high_accuracy.png`
- `shap_waterfall_case_02_low_pred_high_actual.png`
- `shap_waterfall_case_03_high_accuracy.png`
- `shap_waterfall_case_03_roam_support.png`

### 📁 `05_embeddings/` – Visualització d'embeddings (t-SNE / UMAP)
- `embedding_distance_vs_roam.png`
- `support_nearest_neighbors.png`
- `tsne_by_archetype.png`
- `tsne_by_roam_score.png`
- `tsne_perplexity15_by_archetype.png`
- `umap_by_class.png`
- `umap_n10_by_class.png`

### 📁 `06_qualitative/` – Anàlisi qualitativa – exemples top/bottom error
- `bottom_error_best_map.png`
- `bottom_error_best_timeline.png`
- `mid_error_map.png`
- `mid_error_timeline.png`
- `top_error_worst_map.png`
- `top_error_worst_timeline.png`

### 📁 `07_label_health/` – Salut de l'etiqueta – distribucions
- `support_roam_score_quantile_transform_overlay.png`
- `support_roam_score_v3_vs_v5_distribution_overlay.png`
- `support_roam_score_v5_distribution.png`
- `support_roam_score_v5_minus_v3_delta.png`

### 📁 `08_feature_importance/` – Importància de features (permutació)
- `permutation_importance_groups.png`
- `permutation_importance_top_features.png`

### 📁 `09_label_variant_sweep/` – Robustesa de la fórmula – label variant sweep
- `label_variant_sweep_spearman.png`

### 📁 `10_hp_search/` – Cerca d'hiperparàmetres MLP
- `hp_search_spearman_all_configs.png`

### 📁 `11_tolerance/` – Mètriques de tolerància (±0.10, ±0.20)
- `comparison_tolerance_plot.png`

## Xifres Clau

| Mètrica | Valor |
| --- | --- |
| Dataset | 383k observacions (191k partides) |
| Split | 268k/57k/57k (train/val/test) |
| Millor R² (HistGBT) | 0.1614 |
| Millor Spearman | 0.3882 |
| Sostre ICC (botlane+side) | 0.1726 |
| % del sostre assolit | 93.5% |
| ±0.10 tolerància | 41.8% |
| ±0.20 tolerància | 74.2% |
| HP search Δ Spearman | +0.005 (negligible) |
| Variants de fórmula | 15, totes correlació ≥ 0.99 |
| Errors explicats per caos | 17/20 top errors |

## Missatge Principal

> El draft conté senyal predictiu real però limitat (R²≈0.16).
> El model arriba al 93% del sostre empíric.
> El coll d'ampolla no és l'arquitectura sinó la informació pre-partida.
> R²=0.16 no és un "mal resultat" sinó una **troballa empírica legítima**.