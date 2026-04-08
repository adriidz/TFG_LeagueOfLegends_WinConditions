import pandas as pd
import os

def main():
    index_path = os.path.join("data", "clean", "raw_index.csv")
    
    if not os.path.exists(index_path):
        print(f"No se ha encontrado el archivo de índice en {index_path}.")
        return

    print(f"Cargando índice desde {index_path}...")
    df = pd.read_csv(index_path)

    if 'sourceTier' not in df.columns:
        print("El archivo no contiene la columna 'sourceTier'.")
        return

    print("\n--- Distribución de Rangos (sourceTier) ---")
    
    # Calcular frecuencias absolutas y porcentajes
    counts = df['sourceTier'].value_counts(dropna=False)
    percentages = df['sourceTier'].value_counts(dropna=False, normalize=True) * 100
    
    # Crear un DataFrame para mostrarlo mejor
    dist_df = pd.DataFrame({
        'Cantidad': counts,
        'Porcentaje (%)': percentages.round(2)
    })
    
    # Imprimir el resultado
    print(dist_df)
    print("-------------------------------------------\n")
    print(f"Total de partidas procesadas (líneas en el CSV): {len(df)}")

if __name__ == "__main__":
    main()
