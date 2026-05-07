# Geometry v5 Manual Annotation Workflow

## Objetivo

Usar los heatmaps observados como plantilla visual para dibujar una geometria semantica manual. Esta fase no intenta detectar muros con precision fisica: busca separar bien bot, rio, mid, objetivos y los cuatro cuadrantes de jungla.

## Archivos Generados

- `ProgresoActual2/analysis/geometry_manual_annotation/geometry_v5_annotation_template_m5_12.png`
- `ProgresoActual2/analysis/geometry_manual_annotation/geometry_v5_annotation_canvas_m5_12.png`
- `ProgresoActual2/analysis/geometry_manual_annotation/geometry_v5_annotation_metadata_m5_12.json`
- `ProgresoActual2/analysis/geometry_manual_annotation/geometry_v5_annotation_template_m0_14.png`
- `ProgresoActual2/analysis/geometry_manual_annotation/geometry_v5_annotation_canvas_m0_14.png`
- `ProgresoActual2/analysis/geometry_manual_annotation/geometry_v5_annotation_metadata_m0_14.json`

La ventana principal para dibujar es `m5_12`, porque coincide con la ventana de la etiqueta. La ventana `m0_14` queda como contraste para comprobar que las fronteras elegidas no dependan demasiado del tramo temporal.

## Como Dibujar

Usar preferentemente el canvas si se quiere que la conversion automatica a poligonos sea mas facil:

- `geometry_v5_annotation_canvas_m5_12.png`

Usar la plantilla si se quiere dibujar con ayuda de coordenadas:

- `geometry_v5_annotation_template_m5_12.png`

Recomendaciones:

- Dibujar outlines gruesos y opacos.
- Evitar fills translucidos.
- Cerrar las regiones siempre que sea posible.
- No intentar recortar muros pequenos dentro de jungla.
- Priorizar fronteras semanticas claras: rio frente a jungla, lane frente a jungla, alcobas frente a jungla, pits frente a rio/jungla generica.

## Colores Sugeridos

- `BOT_LANE_CORE`: amarillo
- `TOP_LANE_CORE`: amarillo-verde
- `BOT_SIDE_NEAR`: naranja
- `TOP_SIDE_NEAR`: naranja claro
- `RIVER_BOT / RIVER_TOP`: turquesa
- `BLUE_BOT_JUNGLE`: azul
- `BLUE_TOP_JUNGLE`: cian
- `RED_BOT_JUNGLE`: rojo
- `RED_TOP_JUNGLE`: magenta
- `MID_LANE`: morado
- `OBJECTIVE/PIT`: verde
- `BASE`: negro

No es obligatorio seguir estos colores, pero ayuda a extraer automaticamente los trazos si se mantienen colores planos y saturados.

## Siguiente Paso

Cuando exista una imagen anotada, el siguiente paso sera convertir los trazos a un archivo de configuracion, por ejemplo:

- `ProgresoActual2/data/geometry/manual_geometry_v5_config.json`

Despues se implementara una geometria manual:

- `ProgresoActual2/src/geometry/geometry_v5_manual.py`

La geometria `v4` se mantiene como diagnostico observado, no como geometria final.
