# Arquitectura del modelo de regresión

## Unidad de muestra
Cada fila representa un equipo dentro de una partida: `(match_id, team_id)`.

## Entrada cruda
- Columnas de entrada antes de One-Hot: **31**
- Grupos activos: **champions, summoner_spells, context**

- **champions**: Champion picks de ambos equipos (10 posiciones).
- **summoner_spells**: Hechizos de invocador (Flash, Teleport, Ignite...).
- **context**: Side (blue/red).

## Preprocesado
- Todas las columnas categóricas seleccionadas se transforman con `OneHotEncoder(handle_unknown='ignore')`.
- Dimensión final tras One-Hot en train: **1308**.

## Modelo
- Modelo final: **LinearRegression multi-output**.
- Ecuación conceptual: `ŷ = XW + b`.
- Esto equivale a una única capa lineal con **3 salidas continuas**.

## Salida
- Número de salidas: **3**
- `jungle_presence_score`
- `support_roam_score`
- `team_side_focus_score`

## Por qué esta versión es más simple
- No discretiza los scores antes de entrenar.
- Conserva la magnitud del error.
- La arquitectura completa es fácil de explicar: **One-Hot + capa lineal de 3 salidas**.
