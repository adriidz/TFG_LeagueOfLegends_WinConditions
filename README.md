# TFG

Estructura actual del repositorio:

- `ProgresoActual/`: reinicio support-only con regresion continua, scorer rapido, trainer MLP, referencias de campeones e informe editable.
- `ProgresoActual2/`: sandbox reciente para geometria v5 y variantes quantile de la etiqueta support.
- `PropuestaInicial/`: archivo documental del planteamiento anterior. Los artefactos pesados historicos se mantienen fuera de Git.
- `data/` y `data_new/`: datos y caches locales reutilizables, ignorados por Git para evitar duplicar archivos pesados.
- `src/01_data_collection/`: colector raw heredado que sigue siendo util para actualizar datos.

Punto de entrada recomendado:

```powershell
Get-Content ProgresoActual\README.md
```
