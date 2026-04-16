import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Configuración visual
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

def parse_args():
    parser = argparse.ArgumentParser(description="Genera histogramas de las métricas continuas con sus cuantiles.")
    parser.add_argument("--labels-dir", default="data/clean/labels", help="Directorio origen de las etiquetas")
    parser.add_argument("--out-dir", default="report_figures", help="Directorio de salida para las figuras")
    parser.add_argument("--sample-frac", type=float, default=None, help="Muestreo usado (p.ej 0.05)")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    suffix = f"_sample{int(args.sample_frac * 100)}" if args.sample_frac and 0.0 < args.sample_frac < 1.0 else ""
    
    tasks = [
        ("Jungle Presence", f"jungle_labels{suffix}_m10.parquet", "jungle_presence_score"),
        ("Support Roam", f"support_labels{suffix}_m10.parquet", "support_roam_score"),
        ("Team Tendency", f"team_tendency_labels{suffix}_m10.parquet", "team_side_focus_score")
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Figura 11: Distribución de Scores y Cuantiles 20/80 (Ventana 10m)', fontsize=14, y=1.05)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (title, filename, score_col) in enumerate(tasks):
        ax = axes[i]
        path = os.path.join(args.labels_dir, filename)
        
        # Fallback a archive si no está en main
        if not os.path.exists(path):
            archive_path = os.path.join(args.labels_dir, "archive", filename)
            if os.path.exists(archive_path):
                path = archive_path
                
        # Smart Fallback: Si no existe el sample exacto, busca CUALQUIER .parquet m10 para esa tarea
        if not os.path.exists(path):
            task_prefix = filename.split("_labels")[0] + "_labels"
            print(f"[Warning] No se encontró {path}. Buscando un fallback para {task_prefix}...")
            
            # Buscar en main
            matches = glob.glob(os.path.join(args.labels_dir, f"{task_prefix}*m10*.parquet"))
            # Buscar en archive
            matches += glob.glob(os.path.join(args.labels_dir, "archive", f"{task_prefix}*m10*.parquet"))
            
            if matches:
                path = matches[0]
                print(f"[Fallback] Usando: {path}")
        
        if not os.path.exists(path):
            print(f"[Error] No se encontró ningún archivo para {title}. Omitiendo gráfica.")
            ax.set_title(title)
            ax.text(0.5, 0.5, 'Archivo no encontrado', ha='center', va='center')
            continue
            
        df = pd.read_parquet(path)
        valid = df[score_col].dropna()
        
        if valid.empty:
            ax.set_title(title)
            ax.text(0.5, 0.5, 'Sin datos válidos', ha='center', va='center')
            continue
            
        # Calcular cuantiles q20 y q80
        q20 = valid.quantile(0.20)
        q80 = valid.quantile(0.80)
        
        # Plotear histograma mejorado (Suavizado si hay seaborn)
        if HAS_SEABORN:
            sns.histplot(valid, bins=35, kde=True, ax=ax, color=colors[i], alpha=0.6, edgecolor='white', linewidth=0.5)
        else:
            ax.hist(valid, bins=35, color=colors[i], alpha=0.7, edgecolor='white', linewidth=0.5)
        
        # Líneas verticales
        ax.axvline(q20, color='red', linestyle='--', linewidth=2, label=f'q20: {q20:.3f}')
        ax.axvline(q80, color='purple', linestyle='--', linewidth=2, label=f'q80: {q80:.3f}')
        
        # Rellenar zona ambigua
        ax.axvspan(q20, q80, color='gray', alpha=0.2, label='Zona Ambigua (60%)')
        
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel('Puntuación Continua' if title != "Team Tendency" else "Focus (-1=Bot, 1=Top)")
        ax.set_ylabel('Nº Partidas')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_file = os.path.join(args.out_dir, "fig11_distribucion_cuantiles.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Histogramas de cuantiles generados en: {out_file}")

if __name__ == "__main__":
    main()
