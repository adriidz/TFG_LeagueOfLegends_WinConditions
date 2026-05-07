# Geometry v5 Manual

## Resumen

`geometry_v5_manual` convierte la anotacion manual de `mapa_editado.png` en una geometria semantica trazada a mano. Esta version no intenta modelar muros internos de jungla: separa macrozonas utiles para la etiqueta de roaming.

La configuracion actual (`geometry_v5_manual_redraw_from_annotation_2`) se rehizo desde la imagen anotada. El cruce central se resuelve haciendo que `MID_LANE` sea continuo y tenga prioridad sobre rio y junglas. `RIVER_BOT` queda como una transicion corta hacia dragon, no como una diagonal que atraviesa mid.

## Archivos

- Configuracion: `ProgresoActual2/data/geometry/manual_geometry_v5_config.json`
- Modulo: `ProgresoActual2/src/geometry/geometry_v5_manual.py`
- Render: `ProgresoActual2/scripts/plot_geometry_v5_manual.py`
- Diagnosticos: `ProgresoActual2/analysis/geometry_v5_manual/`

## Criterio

- Los outlines de la anotacion se conservan como referencia visual.
- Los carriles usan `centerline + width` para clasificacion, porque los outlines amarillos son bandas grandes y no deben invadir jungla o side-near.
- Los pits se definen como circulos manuales segun la correccion dibujada.
- Si un punto real cae fuera de poligonos explicitos, se usa fallback por cuadrante de jungla para evitar perder cobertura.

## Pendiente

- Ajustar vertices a mano si alguna frontera queda desplazada.
- Validar distribucion de zonas en supports `5-12` antes de usar esta geometria en la etiqueta.
