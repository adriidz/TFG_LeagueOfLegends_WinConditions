import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, ArrowStyle, ConnectionPatch
import matplotlib.patches as patches

# Configuración visual alineada con la "Figura 9" de referencia
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Genera figuras para el reporte.")
    parser.add_argument("--out-dir", default="report_figures", help="Directorio de salida para las figuras")
    return parser.parse_args()

COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'tertiary': '#2ca02c',
    'ambiguous': '#d62728',
    'bg': '#f8f9fa'
}

# ==========================================
# Figura 1: Esquema general del pipeline
# ==========================================
def plot_fig1_pipeline_esquema(out_dir):

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    
    stages = [
        "Riot API\n(Datos Crudos)",
        "Raw Data\n(JSONs/Parquet)",
        "Feature Eng.\n(Espacial/Espacio-tiempo)",
        "Métricas\nContinuas",
        "Discretización\n(Labels)",
        "Model Input\n(Ensamblado)",
        "Multi-output\nTraining"
    ]
    
    x_positions = np.linspace(0.1, 0.9, len(stages))
    y_pos = 0.5
    
    for i, (stage, x) in enumerate(zip(stages, x_positions)):
        bbox = FancyBboxPatch((x - 0.06, y_pos - 0.1), 0.12, 0.2, boxstyle="round,pad=0.02", 
                              ec="black", fc="#e1f5fe" if i < len(stages)-1 else "#c8e6c9", lw=1.5)
        ax.add_patch(bbox)
        ax.text(x, y_pos, stage, ha='center', va='center', fontsize=9, fontweight='bold')
        
        if i < len(stages) - 1:
            ax.annotate('', xy=(x_positions[i+1]-0.07, y_pos), xytext=(x+0.07, y_pos),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
            
    plt.title("Figura 1: Esquema general del pipeline ELT y Machine Learning", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig01_pipeline_arquitectura.png"), dpi=300)
    plt.close()

# ==========================================
# Figura 2: Resumen visual del pipeline (Scripts)
# ==========================================
def plot_fig2_scripts_timeline(out_dir):

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    
    scripts = [
        ("01_run_collector.py", "Descarga y limpieza básica de matches (Riot API)"),
        ("02a_p1_build_labels_*.py", "Construcción de métricas continuas y discretización (Fase 1)"),
        ("02b_p2_build_model_input.py", "Join de features pregame + labels (Target)"),
        ("03_p3_train_multioutput.py", "Entrenamiento del modelo PyTorch")
    ]
    
    for i, (script, desc) in enumerate(scripts):
        y = 0.8 - i*0.2
        ax.text(0.1, y, script, fontsize=11, fontweight='bold', color=COLORS['primary'], family='monospace')
        ax.text(0.35, y, desc, fontsize=10, va='bottom')
        if i < len(scripts) - 1:
            ax.plot([0.15, 0.15], [y-0.02, y-0.12], color='gray', lw=2, linestyle='--')
            
    plt.title("Figura 2: Secuencia de ejecución del Pipeline", pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig02_scripts_timeline.png"), dpi=300)
    plt.close()

# ==========================================
# Figura 3: Rendimiento(F1) por ventana temporal
# ==========================================
def plot_fig3_rendimiento_ventanas(out_dir):

    # Datos reales extraídos de los logs de entrenamiento (schema=ternary, sample5)
    # F1 Macro es más informativo que accuracy en ternary (clases desbalanceadas + ambiguous)
    ventanas = [6, 8, 10, 12, 14]
    
    f1_jungle  = [0.2201, 0.2007, 0.3740, 0.3284, 0.3553]
    f1_support = [0.2799, 0.3010, 0.2849, 0.4117, 0.4098]
    f1_team    = [0.2336, 0.2953, 0.3445, 0.3105, 0.3089]
    
    acc_jungle  = [0.3868, 0.3818, 0.4194, 0.3657, 0.4287]
    acc_support = [0.3506, 0.3703, 0.3086, 0.4715, 0.4367]
    acc_team    = [0.2885, 0.5471, 0.4978, 0.5481, 0.5692]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Figura 3: Rendimiento por ventana temporal (Ternary, sample=5%)', fontsize=13)

    for ax, vals_j, vals_s, vals_t, metric in [
        (axes[0], acc_jungle, acc_support, acc_team, 'Accuracy'),
        (axes[1], f1_jungle,  f1_support,  f1_team,  'F1 Macro')
    ]:
        ax.plot(ventanas, vals_j, marker='o', label='Jungle Presence', lw=2, color=COLORS['primary'])
        ax.plot(ventanas, vals_s, marker='s', label='Support Roam',    lw=2, color=COLORS['secondary'])
        ax.plot(ventanas, vals_t, marker='^', label='Team Tendency',   lw=2, color=COLORS['tertiary'])
        ax.set_xlabel('Ventana temporal (minutos)')
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.set_xticks(ventanas)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.65)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig03_rendimiento_por_ventana.png"), dpi=300)
    plt.close()

# ==========================================
# Figura 4: Distribución de etiquetas por ventana
# ==========================================
def plot_fig4_distribucion_etiquetas(out_dir):

    # Datos reales extraídos de los logs de validación (columnas target, schema=ternary)
    ventanas = ['6m', '8m', '10m', '12m', '14m']

    # ---- Jungle ----
    jun_farm = np.array([38.3, 37.8, 21.9, 25.2, 23.4])
    jun_amb  = np.array([39.3, 38.2, 52.5, 52.1, 53.8])
    jun_map  = np.array([22.3, 23.9, 25.6, 22.7, 22.7])

    # ---- Support ----
    sup_lane = np.array([27.4, 33.3, 26.3, 20.6, 22.1])
    sup_amb  = np.array([39.0, 35.7, 48.3, 54.9, 54.4])
    sup_roam = np.array([33.6, 30.9, 25.4, 24.5, 23.5])

    # ---- Team ----
    tea_bot  = np.array([29.3, 20.2, 22.8, 21.7, 20.7])
    tea_amb  = np.array([44.1, 56.8, 54.1, 57.1, 58.3])
    tea_top  = np.array([26.5, 23.0, 23.1, 21.2, 21.0])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.suptitle('Figura 4: Distribución de etiquetas por ventana temporal (Ternary)', fontsize=13)

    for ax, (c1, amb, c2), (l1, lamb, l2), title in [
        (axes[0], (jun_farm, jun_amb, jun_map),   ('Farm Oriented',  'Ambiguous', 'Map Presence'),   'Jungle'),
        (axes[1], (sup_lane, sup_amb, sup_roam),  ('Lane Anchored',  'Ambiguous', 'Roamer'),          'Support'),
        (axes[2], (tea_bot,  tea_amb, tea_top),   ('Botside',        'Ambiguous', 'Topside'),         'Team Tendency'),
    ]:
        ax.bar(ventanas, c1,            label=l1,   color=COLORS['primary'])
        ax.bar(ventanas, amb, bottom=c1, label=lamb, color='#cccccc')
        ax.bar(ventanas, c2,  bottom=c1+amb, label=l2, color=COLORS['secondary'])
        ax.set_title(title)
        ax.set_xlabel('Ventana')
        ax.tick_params(axis='x', rotation=45)
        if ax == axes[0]:
            ax.set_ylabel('Porcentaje de partidas (%)')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "fig04_distribucion_etiquetas_ventana.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
def plot_fig5_acuerdo_ventanas(out_dir):

    # Datos reales extraídos de all_tasks_pairwise_agreement.csv
    ventanas = ['6→8', '8→10', '10→12', '12→14']

    # agreement_exact (consecutivos)
    acuerdo_jungle  = [0.7070, 0.6887, 0.7816, 0.7920]
    acuerdo_support = [0.6858, 0.7022, 0.7826, 0.8214]
    acuerdo_team    = [0.5961, 0.6328, 0.7471, 0.7705]

    # agreement_non_ambiguous (sólo casos donde ambas ventanas son no-ambiguos)
    nd_jungle  = [1.0000, 0.9992, 1.0000, 1.0000]
    nd_support = [1.0000, 1.0000, 1.0000, 1.0000]
    nd_team    = [0.9758, 0.9824, 0.9961, 0.9978]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Figura 5: Estabilidad de etiquetas entre ventanas consecutivas', fontsize=13)

    for ax, title, j, s, t in [
        (axes[0], 'Acuerdo exacto (incluye ambiguous)',   acuerdo_jungle,  acuerdo_support,  acuerdo_team),
        (axes[1], 'Acuerdo excluyendo ambiguous',         nd_jungle,       nd_support,       nd_team),
    ]:
        ax.plot(ventanas, j, marker='o', label='Jungle',        color=COLORS['primary'],   lw=2)
        ax.plot(ventanas, s, marker='s', label='Support',       color=COLORS['secondary'], lw=2)
        ax.plot(ventanas, t, marker='^', label='Team Tendency', color=COLORS['tertiary'],  lw=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Transición temporal (minutos)')
        ax.set_ylabel('Proporción de acuerdo')
        ax.set_ylim(0.5, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig05_estabilidad_acuerdo.png"), dpi=300)
    plt.close()

def plot_fig6_transiciones(out_dir):

    # Datos reales de all_tasks_transitions.csv – transiciones excluyendo ambiguous
    # Se muestran las 3 tareas lado a lado para el par más revelador: 8→10

    # Jungle 8→10 (ignorando filas con ambiguous en origen o destino)
    jun_matrix = np.array([[3201, 5],
                           [0,    3094]])  # farm→farm, farm→map; map→farm, map→map
    jun_labels = ['Farm (8m)', 'Map (8m)'], ['Farm (10m)', 'Map (10m)']

    # Support 8→10: 0 flips extremos
    sup_matrix = np.array([[3513, 0],
                           [0,    3433]])  # lane→lane, lane→roam; roam→lane, roam→roam
    sup_labels = ['Lane (8m)', 'Roam (8m)'], ['Lane (10m)', 'Roam (10m)']

    # Team 8→10: max flips (75 en total)
    tea_matrix = np.array([[1964, 73],
                           [2,    2217]])  # bot→bot, bot→top; top→bot, top→top
    tea_labels = ['Botside (8m)', 'Topside (8m)'], ['Botside (10m)', 'Topside (10m)']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Figura 6: Matrices de transición directa (8m→10m, excluyendo ambiguous)', fontsize=12)

    for ax, matrix, (row_labels, col_labels), title in [
        (axes[0], jun_matrix, jun_labels, 'Jungle\n(5 flips directos)'),
        (axes[1], sup_matrix, sup_labels, 'Support\n(0 flips directos)'),
        (axes[2], tea_matrix, tea_labels, 'Team Tendency\n(75 flips directos)'),
    ]:
        vmax = matrix.max()
        cax = ax.matshow(matrix, cmap='Blues', vmin=0, vmax=vmax)
        for i in range(2):
            for j in range(2):
                v = matrix[i, j]
                color = 'white' if v > vmax * 0.55 else 'black'
                ax.text(j, i, f'{v:,}', va='center', ha='center', color=color, fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.xaxis.set_ticks_position('bottom')
        ax.set_title(title, fontsize=10, pad=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig06_flips_extremos.png"), dpi=300)
    plt.close()

# ==========================================
# Figura 7: ternary vs binary_clean vs binary_full
# ==========================================
def plot_fig7_comparacion_schemas(out_dir):

    # Datos reales extraídos de los logs (window=m10, sample=5%, sin quantiles explícitos)
    # Ternary: mejor epoch 34 (m10)
    # Binary Full: mejor epoch 35 (m10)
    # Binary Clean: mejor epoch 20 (m10, sin quantile tag – q por defecto igual a q20_80 lógicamente)
    schemas = ['Ternary', 'Binary Full', 'Binary Clean']

    acc_jungle  = [0.4194, 0.5304, 0.5552]
    acc_support = [0.3086, 0.5810, 0.6146]
    acc_team    = [0.4978, 0.5261, 0.5556]

    f1_jungle   = [0.3740, 0.5110, 0.5343]
    f1_support  = [0.2849, 0.5809, 0.6145]
    f1_team     = [0.3445, 0.5244, 0.5515]

    x = np.arange(len(schemas))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Figura 7: Impacto de la exclusión de ambiguous (window=10m, sample=5%)', fontsize=12)

    for ax, (j, s, t), metric in [
        (axes[0], (acc_jungle, acc_support, acc_team), 'Accuracy'),
        (axes[1], (f1_jungle,  f1_support,  f1_team),  'F1 Macro'),
    ]:
        ax.bar(x - width, j, width, label='Jungle',        color=COLORS['primary'])
        ax.bar(x,         s, width, label='Support',       color=COLORS['secondary'])
        ax.bar(x + width, t, width, label='Team Tendency', color=COLORS['tertiary'])
        ax.set_ylabel(f'{metric} (Validación)')
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(schemas)
        if ax == axes[0]:
            ax.legend(title='Tarea')
        ax.set_ylim(0, 0.8)
        ax.axhline(0.5, color='gray', linestyle='--', lw=1, alpha=0.5, label='Baseline (50%)')
        for i, (jv, sv, tv) in enumerate(zip(j, s, t)):
            ax.text(i - width, jv + 0.012, f'{jv:.2f}', ha='center', fontsize=7.5)
            ax.text(i,         sv + 0.012, f'{sv:.2f}', ha='center', fontsize=7.5)
            ax.text(i + width, tv + 0.012, f'{tv:.2f}', ha='center', fontsize=7.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig07_comparacion_schemas.png"), dpi=300)
    plt.close()

# ==========================================
# Figura 8: Ablación binaria q20_80 vs q30_70 vs q40_60
# ==========================================
def plot_fig8_quantiles_ablation(out_dir):

    # Datos reales – sample=5%, window=m10, schema=binary_clean
    # muestras válidas promedio (media de las 3 tareas) sobre val_size=3224
    # q20_80: (1585+1606+1449)/3 = 1546.7 → 48.0%
    # q30_70: (2494+2271+2043)/3 = 2269.3 → 70.4%
    # q40_60: (2920+3206+2736)/3 = 2954.0 → 91.6%
    quantiles = ['q40_60', 'q30_70', 'q20_80']
    cobertura_jun  = [2920/3224*100, 2494/3224*100, 1585/3224*100]
    cobertura_sup  = [3206/3224*100, 2271/3224*100, 1606/3224*100]
    cobertura_tea  = [2736/3224*100, 2043/3224*100, 1449/3224*100]

    # F1 Macro reales por tarea
    f1_jun  = [0.5176, 0.5159, 0.5262]  # q40_60, q30_70, q20_80
    f1_sup  = [0.5725, 0.5949, 0.6145]
    f1_tea  = [0.5270, 0.5214, 0.5515]
    f1_avg  = [(a+b+c)/3 for a, b, c in zip(f1_jun, f1_sup, f1_tea)]
    cob_avg = [(a+b+c)/3 for a, b, c in zip(cobertura_jun, cobertura_sup, cobertura_tea)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Figura 8: Trade-off Pureza/Cobertura en Binary Clean (window=10m)', fontsize=12)

    # Panel izquierdo: F1 por tarea
    x = np.arange(len(quantiles))
    width = 0.25
    axes[0].bar(x - width, f1_jun, width, label='Jungle',        color=COLORS['primary'])
    axes[0].bar(x,         f1_sup, width, label='Support',       color=COLORS['secondary'])
    axes[0].bar(x + width, f1_tea, width, label='Team Tendency', color=COLORS['tertiary'])
    axes[0].set_ylabel('F1 Macro')
    axes[0].set_title('F1 Macro por tarea y quantil')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(quantiles)
    axes[0].set_ylim(0.45, 0.70)
    axes[0].legend(fontsize=8)
    for i, (j, s, t) in enumerate(zip(f1_jun, f1_sup, f1_tea)):
        axes[0].text(i - width, j + 0.004, f'{j:.3f}', ha='center', fontsize=7)
        axes[0].text(i,         s + 0.004, f'{s:.3f}', ha='center', fontsize=7)
        axes[0].text(i + width, t + 0.004, f'{t:.3f}', ha='center', fontsize=7)

    # Panel derecho: trade-off F1 avg vs cobertura
    ax2 = axes[1]
    ax2b = ax2.twinx()
    ax2.bar(quantiles, f1_avg, width=0.4, color=COLORS['primary'], alpha=0.8, label='F1 Macro promedio')
    ax2b.plot(quantiles, cob_avg, color='crimson', marker='o', lw=2.5, label='% Dataset retenido')
    ax2.set_ylabel('F1 Macro (promedio tareas)', color=COLORS['primary'])
    ax2.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax2.set_ylim(0.45, 0.70)
    ax2b.set_ylabel('% muestras válidas (promedio tareas)', color='crimson')
    ax2b.tick_params(axis='y', labelcolor='crimson')
    ax2b.set_ylim(0, 105)
    ax2.set_title('Trade-off F1 vs Cobertura')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig08_quantiles_tradeoff.png"), dpi=300)
    plt.close()

# ==========================================
# Figura 9: Curvas de Train Loss y Validaciones
# ==========================================
def plot_fig9_learning_curves(out_dir):

    # Datos completos extraídos de binary_clean q20_80 (experimento final Kaggle)
    train_loss = [0.6875, 0.6694, 0.6651, 0.6606, 0.6580, 0.6566, 0.6554, 0.6543, 0.6534, 0.6533, 
                  0.6524, 0.6524, 0.6521, 0.6518, 0.6516, 0.6518, 0.6516, 0.6512, 0.6513, 0.6509, 
                  0.6513, 0.6508, 0.6506, 0.6511, 0.6508, 0.6509, 0.6506, 0.6506, 0.6511, 0.6506, 
                  0.6505, 0.6504, 0.6506, 0.6508, 0.6505, 0.6505, 0.6507, 0.6506, 0.6502, 0.6504, 
                  0.6503, 0.6504, 0.6505, 0.6505, 0.6507, 0.6503, 0.6503, 0.6505, 0.6503, 0.6504, 
                  0.6507, 0.6504, 0.6503, 0.6503, 0.6506, 0.6504, 0.6507]
                  
    val_loss = [0.6694, 0.6661, 0.6616, 0.6583, 0.6566, 0.6547, 0.6539, 0.6527, 0.6520, 0.6512, 
                0.6511, 0.6513, 0.6511, 0.6511, 0.6506, 0.6510, 0.6505, 0.6508, 0.6507, 0.6507, 
                0.6504, 0.6503, 0.6512, 0.6510, 0.6504, 0.6506, 0.6505, 0.6508, 0.6503, 0.6502, 
                0.6506, 0.6506, 0.6502, 0.6506, 0.6506, 0.6504, 0.6505, 0.6503, 0.6506, 0.6503, 
                0.6506, 0.6501, 0.6508, 0.6505, 0.6511, 0.6502, 0.6503, 0.6502, 0.6503, 0.6505, 
                0.6506, 0.6502, 0.6505, 0.6508, 0.6511, 0.6508, 0.6505]
                
    j_acc = [0.545, 0.554, 0.573, 0.584, 0.589, 0.591, 0.591, 0.592, 0.595, 0.593, 
             0.593, 0.593, 0.594, 0.595, 0.594, 0.593, 0.592, 0.593, 0.592, 0.591, 
             0.593, 0.593, 0.593, 0.592, 0.591, 0.592, 0.593, 0.593, 0.592, 0.594, 
             0.595, 0.593, 0.593, 0.594, 0.593, 0.592, 0.592, 0.593, 0.592, 0.594, 
             0.594, 0.595, 0.591, 0.590, 0.593, 0.593, 0.591, 0.592, 0.591, 0.592, 
             0.593, 0.596, 0.593, 0.593, 0.591, 0.593, 0.592]
             
    s_acc = [0.641, 0.650, 0.655, 0.658, 0.658, 0.662, 0.663, 0.664, 0.661, 0.666, 
             0.664, 0.661, 0.664, 0.661, 0.663, 0.663, 0.664, 0.664, 0.663, 0.663, 
             0.663, 0.663, 0.660, 0.662, 0.665, 0.663, 0.663, 0.665, 0.664, 0.663, 
             0.662, 0.664, 0.663, 0.663, 0.664, 0.664, 0.664, 0.664, 0.662, 0.661, 
             0.662, 0.665, 0.663, 0.665, 0.659, 0.664, 0.664, 0.666, 0.665, 0.664, 
             0.662, 0.663, 0.664, 0.663, 0.664, 0.664, 0.663]
             
    t_acc = [0.539, 0.548, 0.550, 0.554, 0.556, 0.559, 0.563, 0.571, 0.577, 0.577, 
             0.580, 0.579, 0.582, 0.582, 0.583, 0.582, 0.584, 0.581, 0.584, 0.583, 
             0.585, 0.584, 0.584, 0.583, 0.582, 0.583, 0.586, 0.583, 0.582, 0.581, 
             0.581, 0.584, 0.581, 0.584, 0.583, 0.583, 0.582, 0.586, 0.584, 0.586, 
             0.581, 0.583, 0.583, 0.582, 0.583, 0.582, 0.584, 0.586, 0.583, 0.582, 
             0.584, 0.583, 0.584, 0.584, 0.583, 0.581, 0.589]

    epochs = np.arange(1, len(train_loss) + 1)
    best_epoch = 42
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Figura 9: Curvas de entrenamiento (Experimento final q20_80)', fontsize=14)
    
    # Loss
    axes[0,0].plot(epochs, train_loss, label='Train loss')
    axes[0,0].plot(epochs, val_loss, label='Validation loss')
    axes[0,0].axvline(best_epoch, color='steelblue', linestyle='--', label='Mejor Val Loss')
    axes[0,0].set_title('Global Loss')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_xlabel('Época')
    axes[0,0].legend()
    
    # Jungle
    axes[0,1].plot(epochs, j_acc, color='#1f77b4')
    axes[0,1].set_title('Jungle presence accuracy')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].set_xlabel('Época')
    
    # Support
    axes[1,0].plot(epochs, s_acc, color='#ff7f0e')
    axes[1,0].set_title('Support roam accuracy')
    axes[1,0].set_ylabel('Accuracy')
    axes[1,0].set_xlabel('Época')
    
    # Team
    axes[1,1].plot(epochs, t_acc, color='#2ca02c')
    axes[1,1].set_title('Team tendency accuracy')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].set_xlabel('Época')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_dir, "fig09_learning_curves.png"), dpi=300)
    plt.close()

# ==========================================
# Figura 10: Métricas finales por tarea (Configuración seleccionada)
# ==========================================
def plot_fig10_metricas_finales(out_dir):

    # Valores exactos de la validación del modelo q20_80
    tareas = ['Jungle\n(Map vs Farm)', 'Support\n(Roam vs Anchor)', 'Team Trend\n(Top vs Bot)']
    acc = [0.5949, 0.6646, 0.5825]
    f1 = [0.5835, 0.6634, 0.5788]
    
    x = np.arange(len(tareas))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width/2, acc, width, label='Accuracy', color=COLORS['primary'])
    ax.bar(x + width/2, f1, width, label='F1 Macro', color=COLORS['secondary'])
    
    for i, (a, f) in enumerate(zip(acc, f1)):
        ax.text(i - width/2, a + 0.01, f"{a:.3f}", ha='center', fontsize=9)
        ax.text(i + width/2, f + 0.01, f"{f:.3f}", ha='center', fontsize=9)
    
    ax.set_ylabel('Métrica')
    ax.set_title('Figura 10: Rendimiento Final por Tarea (Binary Clean q20_80, 10m)')
    ax.set_xticks(x)
    ax.set_xticklabels(tareas)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig10_rendimiento_final.png"), dpi=300)
    plt.close()


def main():
    args = parse_args()
    
    # Imports tardíos para no romper arriba, pero os.makedirs se necesita
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("Generando figuras para el informe de progreso del TFG...")
    plot_fig1_pipeline_esquema(args.out_dir)
    plot_fig2_scripts_timeline(args.out_dir)
    plot_fig3_rendimiento_ventanas(args.out_dir)
    plot_fig4_distribucion_etiquetas(args.out_dir)
    plot_fig5_acuerdo_ventanas(args.out_dir)
    plot_fig6_transiciones(args.out_dir)
    plot_fig7_comparacion_schemas(args.out_dir)
    plot_fig8_quantiles_ablation(args.out_dir)
    plot_fig9_learning_curves(args.out_dir)
    plot_fig10_metricas_finales(args.out_dir)
    print(f"Todas las figuras se han guardado correctamente en la carpeta: {args.out_dir}")

if __name__ == "__main__":

    main()
