# PropuestaInicial

Este directorio actua como archivo conceptual del planteamiento inicial del TFG.
No se han movido aqui los datos pesados ni los resultados generados porque varios
scripts todavia dependen de rutas historicas como `data/`, `Models/`,
`eda_progreso_I/`, `report_figures/` y `progreso_I/`.

## Material asociado al planteamiento inicial

- `FASE 1 COMPLETADA.txt`: estudio de ventanas temporales y estabilidad.
- `FASE 2 COMPLETADA.txt`: estudio de la clase `ambiguous` y formulaciones
  binarias.
- `test labels por ventana de minutos (6, 8, 10, 12, 14).txt`: notas de pruebas
  de labels por ventana.
- `src/03_training/03_p3_train_multioutput*.py`: trainers de clasificacion
  multi-output.
- `src/02_data_processing/02a_*` y `02b_*`: pipeline historico de labels
  discretas.
- `Models/`, `eda_progreso_I/`, `report_figures/` y `progreso_I/`: artefactos,
  graficas y reportes de la primera propuesta.

## Criterio de separacion

El nuevo avance se documenta y ejecuta desde `ProgresoActual/`. La propuesta inicial
queda como evidencia del proceso: decisiones tomadas, resultados obtenidos y
razones para cambiar hacia regresion continua support-only.

Cuando el pipeline nuevo este estable, se puede hacer una reorganizacion fisica
mas agresiva moviendo resultados antiguos a este directorio. Por ahora se evita
para no romper rutas ni perder reproducibilidad.
