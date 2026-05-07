# Prototipo terminal de champ select

Este prototipo permite introducir datos de champ select y obtener una primera
prediccion interpretable del score de roaming de ambos supports. Usa la baseline
documentada en el Informe de Progreso I:

```text
ProgresoActual/models/support_mlp_full_m12
```

El flujo de inferencia carga `model_config.json`, `preprocess.joblib` y
`best_model.pt`, por lo que reutiliza exactamente el `OneHotEncoder` entrenado.
No reentrena el modelo ni recalcula etiquetas.

## Uso interactivo

```powershell
.\.venv\Scripts\python.exe ProgresoActual\scripts\predict_support_roam_cli.py
```

El script pregunta por:

- side aliado (`blue` o `red`);
- campeones aliados y enemigos por rol;
- hechizos de invocador por rol.

Si se dejan los hechizos en blanco, usa defaults razonables por rol. En la
salida se muestran esas suposiciones para que no queden ocultas.

La inferencia se ejecuta dos veces:

- una desde la perspectiva del equipo aliado;
- otra desde la perspectiva del equipo enemigo, invirtiendo `ally/enemy` y el
  lado del mapa.

Por tanto, la salida incluye `Prediccion support aliado` y
`Prediccion support enemigo`.

## Uso por argumentos

```powershell
.\.venv\Scripts\python.exe ProgresoActual\scripts\predict_support_roam_cli.py `
  --no-interactive `
  --side blue `
  --ally-top Ornn --ally-jungle Viego --ally-middle Ahri --ally-bottom Draven --ally-utility Janna `
  --enemy-top Kled --enemy-jungle Wukong --enemy-middle Lux --enemy-bottom Yunara --enemy-utility Nami `
  --ally-utility-spells flash,heal `
  --enemy-utility-spells flash,heal
```

Tambien se puede probar contra una fila real ya etiquetada:

```powershell
.\.venv\Scripts\python.exe ProgresoActual\scripts\predict_support_roam_cli.py `
  --from-match-id EUN1_3915259648 `
  --team-id 100
```

En ese modo imprime la etiqueta real y el error absoluto, util para comprobar
que la inferencia reproduce el mismo contrato que el entrenamiento.

## Interpretacion

La salida principal de cada equipo es:

- `Score estimado`: valor continuo en `[0,1]`.
- `Percentil vs validacion`: posicion del score respecto a las predicciones de
  validacion de la MLP full `m12`.
- `Lectura`: frase corta basada en ese percentil.
- `Prior experto`: contraste cualitativo del campeon support si existe en la
  referencia manual.

El modelo actual solo usa informacion pregame y no observa eventos de partida.
La etiqueta aprendida resume roaming observado entre minuto 5 y minuto 12.
