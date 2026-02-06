import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

INPUT_JSON = "Data/champion_stats_3.json"
OUTPUT_PCA = "Data/champion_context_pca.json"
N_COMPONENTS = 6 # Comprimimos toda la info del campeón en 6 números

def generate_pca_embeddings():
    with open(INPUT_JSON, 'r') as f:
        stats_dict = json.load(f)

    # 1. Convertir JSON a DataFrame
    # Las filas son "ChampID_Rol", las columnas son las métricas (mean)
    data_rows = []
    ids = []
    
    # Recopilamos todas las métricas posibles primero para alinear columnas
    # (Usamos las keys del primer elemento como referencia)
    first_key = next(iter(stats_dict))
    metric_names = sorted(stats_dict[first_key].keys())
    # Filtramos solo las medias, ignoramos std y games
    metric_names = [m for m in metric_names if isinstance(stats_dict[first_key][m], dict)]
    
    print(f"Usando {len(metric_names)} métricas para generar el contexto.")

    for key, val in stats_dict.items():
        # key ejemplo: "86_TOP"
        try:
            row_vals = []
            for m in metric_names:
                # Extraemos solo la media ('mean')
                row_vals.append(val[m]['mean'])
            
            data_rows.append(row_vals)
            ids.append(key)
        except Exception as e:
            continue
            
    X = np.array(data_rows)
    
    # 2. Estandarización (Z-Score)
    # Esto elimina el problema de "dividir por 1000". 
    # Deja todas las métricas en la misma escala automáticamente.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. PCA
    pca = PCA(n_components=N_COMPONENTS)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Varianza explicada por los {N_COMPONENTS} componentes: {np.sum(pca.explained_variance_ratio_):.2%}")
    # Si la varianza es baja (<70%), sube N_COMPONENTS.
    
    # 4. Guardar Diccionario Mapeado
    pca_dict = {}
    for i, key in enumerate(ids):
        # Convertimos a lista de floats (JSON no traga numpy)
        # Parseamos la key para guardar solo por ID de campeón si queremos simplificar,
        # o mantenemos "ID_ROL" si creemos que Garen Top es distinto a Garen Mid.
        # Para contexto general, ID_ROL es mejor.
        pca_dict[key] = X_pca[i].tolist()
        
    with open(OUTPUT_PCA, 'w') as f:
        json.dump(pca_dict, f)
        
    print(f"Generado {OUTPUT_PCA} con éxito.")

if __name__ == "__main__":
    generate_pca_embeddings()