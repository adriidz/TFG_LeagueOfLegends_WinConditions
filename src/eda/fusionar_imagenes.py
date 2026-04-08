from PIL import Image

# 1. Pon aquí las rutas exactas de las dos fotos que ya tienes generadas
ruta_heatmap = "data/clean/spatial_reports/jungle_heatmap_0_14_balanced.png" 
ruta_partida_con_lineas = "data/clean/geometry_reports/jungle_heatmap_0_14_balanced.png"

try:
    # 2. Abrir las imágenes
    img_heatmap = Image.open(ruta_heatmap).convert("RGBA")
    img_lineas = Image.open(ruta_partida_con_lineas).convert("RGBA")

    # 3. Superponerlas (Blend)
    # El valor alpha (de 0.0 a 1.0) controla el peso de la segunda imagen.
    # 0.5 significa 50% de opacidad para cada una. 
    # Súbelo a 0.6 o 0.7 si quieres que las líneas geométricas se vean más fuertes que el calor.
    img_final = Image.blend(img_heatmap, img_lineas, alpha=0.5)

    # 4. Guardar el resultado
    ruta_salida = "heatmap_con_lineas_superpuestas.png"
    img_final.save(ruta_salida)
    
    print(f"¡Listo! Imagen combinada guardada en: {ruta_salida}")

except FileNotFoundError as e:
    print(f"Error: No se encontró una de las imágenes. Revisa las rutas.\nDetalle: {e}")