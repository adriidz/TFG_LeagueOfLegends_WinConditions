import os
import sys
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "02_data_processing"))
from shared_utils import (
    list_match_dirs, load_json, get_timeline_frames, get_participant_frame,
    extract_position, participant_is_alive, MAP_MAX
)
from spatial_target_analysis import draw_spatial_boundaries, draw_zone_reference_overlay

def process_match(mdir: str):
    timeline_path = os.path.join(mdir, "timeline.json")
    if not os.path.exists(timeline_path): return [], []
    try:
        timeline = load_json(timeline_path)
    except:
        return [], []
        
    frames = get_timeline_frames(timeline)
    xs, ys = [], []
    # Minute 0 to 14 (first 15 frames)
    for frame in frames[:15]: 
        for pid in range(1, 11):
            pf = get_participant_frame(frame, pid)
            if participant_is_alive(pf):
                pos = extract_position(pf)
                if pos:
                    xs.append(pos[0])
                    ys.append(pos[1])
    return xs, ys

def main():
    root = "data/raw/raw/europe"
    dirs = list_match_dirs(root)[:10000]
    out_dir = "data/clean/geometry_reports"
    os.makedirs(out_dir, exist_ok=True)
    
    xs_all, ys_all = [], []
    print(f"Extrayendo posiciones de {len(dirs)} partidas...")
    
    # Procesamiento concurrente para máxima velocidad
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for xs, ys in executor.map(process_match, dirs):
            xs_all.extend(xs)
            ys_all.extend(ys)
        
    print(f"Generando heatmap con {len(xs_all):,} puntos mundiales...")
    
    # ------ MAPA 1: HEATMAP + LÍNEAS ------
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # Heatmap
    hb = ax.hexbin(xs_all, ys_all, gridsize=80, extent=(0, MAP_MAX, 0, MAP_MAX), 
                   mincnt=1, cmap="inferno", norm=mcolors.LogNorm(vmin=1.0))
    cb = plt.colorbar(hb)
    cb.set_label("Densidad de Jugadores (Escala Logarítmica)")
    
    # Overlay de contornos
    draw_spatial_boundaries(ax)
    
    plt.xlim(0, MAP_MAX)
    plt.ylim(0, MAP_MAX)
    plt.title("Validación Geometría vs Heatmap Real (10.000 matches)")
    
    out_1 = os.path.join(out_dir, "heatmap_10k_lines_only.png")
    plt.savefig(out_1, dpi=200, bbox_inches="tight")
    plt.close()
    
    # ------ MAPA 2: HEATMAP + ZONES COMPLETAS ------
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    hb = ax.hexbin(xs_all, ys_all, gridsize=80, extent=(0, MAP_MAX, 0, MAP_MAX), 
                   mincnt=1, cmap="inferno", norm=mcolors.LogNorm(vmin=1.0))
    plt.colorbar(hb)
    
    # Polígonos de zonas sólidas translúcidas
    draw_zone_reference_overlay(ax, grid_size=300, alpha=0.35)
    
    plt.xlim(0, MAP_MAX)
    plt.ylim(0, MAP_MAX)
    plt.title("Zonas Completas vs Heatmap Real (10.000 matches)")
    
    out_2 = os.path.join(out_dir, "heatmap_10k_full_zones.png")
    plt.savefig(out_2, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"Visualizaciones guardadas en:")
    print(f"  - {out_1}")
    print(f"  - {out_2}")

if __name__ == "__main__":
    main()
