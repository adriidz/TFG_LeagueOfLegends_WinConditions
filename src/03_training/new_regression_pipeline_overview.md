# Rama nueva de regresión continua

## Objetivo
Esta rama paralela deja intacta la línea anterior de clasificación, pero añade un pipeline nuevo orientado a **predecir scores continuos crudos** en vez de clases discretas.

Los tres targets continuos son:

- `jungle_presence_score`
- `support_roam_score`
- `team_side_focus_score`

## Qué cambia respecto a la rama anterior

Antes:
- Se calculaba un score continuo.
- Ese score se discretizaba en clases.
- El modelo entrenaba con `CrossEntropyLoss` sobre clases.

Ahora:
- Se mantiene el score continuo como target.
- El modelo entrena directamente sobre valores numéricos.
- La salida del modelo son **3 valores continuos**, uno por tarea.

## Arquitectura nueva (simple y defendible)

### Unidad de muestra
Cada fila representa un **equipo dentro de una partida**: `(match_id, team_id)`.

### Entrada
Solo se usa información de **draft**:
- campeones
- summoner spells
- runas
- bans
- side

### Preprocesado
Todas las columnas categóricas se codifican con **One-Hot Encoding**.

- dimensión de entrada cruda: número de columnas categóricas seleccionadas
- dimensión real de entrada al modelo: número de columnas tras One-Hot, que depende del train

### Modelo
El trainer nuevo usa por defecto:

- `OneHotEncoder(handle_unknown="ignore")`
- `LinearRegression()` multi-output

Conceptualmente:
`ŷ = XW + b`

Eso equivale a una arquitectura con:

- entrada: vector One-Hot de dimensión `D`
- salida: **3 neuronas lineales**
  - una para jungle
  - una para support
  - una para team

### Loss / criterio
En regresión lineal se optimiza error cuadrático (least squares / MSE).

## Por qué esta versión es mejor para explicar

- No pierde información al discretizar antes de entrenar.
- Penaliza más los errores grandes que los pequeños.
- La arquitectura es muy transparente.
- La pregunta “¿cuántas salidas tiene?” queda clara: **3 salidas continuas**.

## Scripts creados

### 1) `new_02a_build_labels_and_draft_features.py`
Guarda en `data_new/`:
- labels discretas
- draft features
- score tables continuas por tarea y ventana

### 2) `new_02b_build_model_input_regression.py`
Une:
- `draft_features`
- `jungle_scores`
- `support_scores`
- `team_tendency_scores`

y genera:

- `data_new/training/model_input_multioutput_regression.parquet`

### 3) `new_03_train_multioutput_regression.py`
Entrena el modelo simple de regresión con One-Hot + modelo lineal.

## Ejemplo de ejecución

### Paso 1: construir scores y draft features
```bash
python new_02a_build_labels_and_draft_features.py --analysis-max-minutes 8 --sample-frac 0.05
```

### Paso 2: construir model input continuo
```bash
python new_02b_build_model_input_regression.py --score-max-minute 8 --sample-frac 0.05
```

### Paso 3: entrenar regresión continua
```bash
python new_03_train_multioutput_regression.py --score-max-minute 8 --sample-frac 0.05 --feature-groups standard
```

## Artefactos principales del entrenamiento

En `Models_new/...` se guardan:

- `regression_pipeline.joblib`
- `metrics_by_target.csv`
- `validation_predictions.parquet`
- `feature_space_summary.csv`
- `model_config.json`
- `architecture_summary.md`

## Métricas a enseñar
Por target:
- MSE
- RMSE
- MAE
- R²
- correlación de Pearson
