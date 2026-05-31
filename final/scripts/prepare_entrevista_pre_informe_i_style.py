"""
Prepare entrevista/ directory with all graphics for tutor meeting.

Generates missing charts and copies existing PNGs, organized by theme.
"""
import shutil
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────
FINAL   = Path(r"c:\Users\adria\Desktop\TFG\final")
ANALYSIS = FINAL / "analysis"
OUT     = FINAL / "entrevista"

# ── Theme directories ──────────────────────────────────────────────────
THEMES = {
    "01_model_comparison":      "Comparació de models (R², Spearman, tolerància)",
    "02_training_curves":       "Corbes d'entrenament MLP (overfitting check)",
    "03_ceiling":               "Sostre empíric (ICC / R² per agrupació)",
    "04_shap":                  "SHAP – importància de features i waterfall",
    "05_embeddings":            "Visualització d'embeddings (t-SNE / UMAP)",
    "06_qualitative":           "Anàlisi qualitativa – exemples top/bottom error",
    "07_label_health":          "Salut de l'etiqueta – distribucions",
    "08_feature_importance":    "Importància de features (permutació)",
    "09_label_variant_sweep":   "Robustesa de la fórmula – label variant sweep",
    "10_hp_search":             "Cerca d'hiperparàmetres MLP",
    "11_tolerance":             "Mètriques de tolerància (±0.10, ±0.20)",
}

# ── Helpers ────────────────────────────────────────────────────────────
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_png(src: Path, dst_dir: Path, rename: str = None):
    """Copy a PNG to a destination directory, optionally renaming."""
    ensure_dir(dst_dir)
    name = rename or src.name
    dst = dst_dir / name
    shutil.copy2(src, dst)
    print(f"  [OK] {dst.relative_to(OUT)}")

# ── Style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor':   '#16213e',
    'axes.edgecolor':   '#e94560',
    'axes.labelcolor':  '#eaeaea',
    'text.color':       '#eaeaea',
    'xtick.color':      '#aaaaaa',
    'ytick.color':      '#aaaaaa',
    'grid.color':       '#2a2a4a',
    'grid.alpha':       0.6,
    'font.family':      'sans-serif',
    'font.size':        11,
})

COLORS = ['#e94560', '#0f3460', '#533483', '#16c79a', '#f5a623',
          '#4fc3f7', '#ff6f61', '#7e57c2', '#26a69a', '#ff8a65']

# ═══════════════════════════════════════════════════════════════════════
#  1. COPY existing PNGs
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  COPYING EXISTING GRAPHICS")
print("="*60)

# 01 - Model comparison
mc = ANALYSIS / "model_comparison"
d01 = OUT / "01_model_comparison"
for f in ["comparison_spearman.png", "comparison_tolerance_plot.png"]:
    copy_png(mc / f, d01)

# 02 - Training curves
tc = ANALYSIS / "training_curves"
d02 = OUT / "02_training_curves"
for f in tc.glob("*.png"):
    copy_png(f, d02)

# 04 - SHAP
sh = ANALYSIS / "shap"
d04 = OUT / "04_shap"
for f in sh.glob("*.png"):
    copy_png(f, d04)

# 05 - Embeddings (use the main embedding_analysis, not per_role)
emb = ANALYSIS / "embedding_analysis"
d05 = OUT / "05_embeddings"
for f in emb.glob("*.png"):
    copy_png(f, d05)

# 06 - Qualitative (select 3 representative cases)
cp = ANALYSIS / "qualitative_case_audit" / "case_plots"
d06 = OUT / "06_qualitative"

# Top error #1: worst prediction
copy_png(cp / "top_error_01_EUW1_7831489390_200_map.png", d06, "top_error_worst_map.png")
copy_png(cp / "top_error_01_EUW1_7831489390_200_timeline.png", d06, "top_error_worst_timeline.png")

# Bottom error #1: best (smallest error)
copy_png(cp / "bottom_error_01_EUW1_7739311514_200_map.png", d06, "bottom_error_best_map.png")
copy_png(cp / "bottom_error_01_EUW1_7739311514_200_timeline.png", d06, "bottom_error_best_timeline.png")

# A mid-range example
copy_png(cp / "top_error_10_EUW1_7708270762_200_map.png", d06, "mid_error_map.png")
copy_png(cp / "top_error_10_EUW1_7708270762_200_timeline.png", d06, "mid_error_timeline.png")

# 07 - Label health
lh = ANALYSIS / "label_health"
d07 = OUT / "07_label_health"
for f in lh.glob("*.png"):
    copy_png(f, d07)

# 08 - Feature importance
fi = ANALYSIS / "feature_importance"
d08 = OUT / "08_feature_importance"
for f in fi.glob("*.png"):
    copy_png(f, d08)

# 11 - Tolerance (same as model comparison tolerance plot, re-copy for easy access)
d11 = OUT / "11_tolerance"
copy_png(mc / "comparison_tolerance_plot.png", d11)

# ═══════════════════════════════════════════════════════════════════════
#  2. GENERATE missing charts
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  GENERATING MISSING CHARTS")
print("="*60)

# ── 03 - Ceiling Chart ────────────────────────────────────────────────
d03 = OUT / "03_ceiling"
ensure_dir(d03)

df_ceil = pd.read_csv(ANALYSIS / "ceiling" / "ceiling_summary.csv")

# Filter meaningful groupings (exclude degenerate ones with n_groups=0 or very few)
mask = (df_ceil['n_groups'] > 0) & (df_ceil['n_groups'] < 5000) & (df_ceil['icc'].notna())
df_plot = df_ceil[mask].copy()

# Sort by ICC
df_plot = df_plot.sort_values('icc', ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))

y_pos = np.arange(len(df_plot))
bar_height = 0.35

# ICC bars
bars1 = ax.barh(y_pos - bar_height/2, df_plot['icc'], bar_height,
                label='ICC', color=COLORS[0], alpha=0.9, edgecolor='none')
# R² group mean bars
bars2 = ax.barh(y_pos + bar_height/2, df_plot['r2_group_mean'], bar_height,
                label='R² (group mean)', color=COLORS[3], alpha=0.9, edgecolor='none')

# Add value labels
for bar in bars1:
    width = bar.get_width()
    ax.text(width + 0.003, bar.get_y() + bar.get_height()/2,
            f'{width:.3f}', va='center', fontsize=9, color='#eaeaea')
for bar in bars2:
    width = bar.get_width()
    ax.text(width + 0.003, bar.get_y() + bar.get_height()/2,
            f'{width:.3f}', va='center', fontsize=9, color='#eaeaea')

# Best model reference line
ax.axvline(x=0.1614, color='#f5a623', linestyle='--', linewidth=1.5,
           label='Best Model (HistGBT R²=0.161)', alpha=0.8)

ax.set_yticks(y_pos)
ax.set_yticklabels(df_plot['grouping'].str.replace('_', ' ').str.title(), fontsize=9)
ax.set_xlabel('Score')
ax.set_title('Sostre Empíric: ICC i R² per Agrupació de Composició\n'
             '(fins a quin punt el draft explica la variància del roaming)',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10, framealpha=0.7)
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, max(df_plot['r2_group_mean'].max(), df_plot['icc'].max()) + 0.04)

plt.tight_layout()
fig.savefig(d03 / "ceiling_icc_r2_by_grouping.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ 03_ceiling/ceiling_icc_r2_by_grouping.png")


# ── 09 - Label Variant Sweep Chart ───────────────────────────────────
d09 = OUT / "09_label_variant_sweep"
ensure_dir(d09)

df_sweep = pd.read_csv(ANALYSIS / "label_variant_sweep" / "sweep_metrics.csv")

# Aggregate: best Spearman per variant
best_per_variant = (df_sweep
    .groupby('variant_id')
    .agg({'spearman_corr': 'max', 'r2': 'max'})
    .sort_values('spearman_corr', ascending=True)
    .reset_index())

fig, ax = plt.subplots(figsize=(12, 7))

y_pos = np.arange(len(best_per_variant))
bars = ax.barh(y_pos, best_per_variant['spearman_corr'], 0.6,
               color=COLORS[0], alpha=0.9, edgecolor='none')

# Add value labels
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
            f'{width:.4f}', va='center', fontsize=9, color='#eaeaea')

# Reference: best model Spearman
ax.axvline(x=0.3882, color='#f5a623', linestyle='--', linewidth=1.5,
           label='Best Model Spearman (0.388)', alpha=0.8)

ax.set_yticks(y_pos)
labels = best_per_variant['variant_id'].str.replace('_', ' ').str.title()
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Spearman Correlation')
ax.set_title('Robustesa de la Fórmula: Spearman per Variant d\'Etiqueta\n'
             '(totes les variants donen resultats molt similars)',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='lower right', fontsize=10, framealpha=0.7)
ax.grid(axis='x', alpha=0.3)

# Set x range to emphasize how close they all are
xmin = best_per_variant['spearman_corr'].min() - 0.01
xmax = best_per_variant['spearman_corr'].max() + 0.015
ax.set_xlim(xmin, xmax)

plt.tight_layout()
fig.savefig(d09 / "label_variant_sweep_spearman.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ 09_label_variant_sweep/label_variant_sweep_spearman.png")


# ── 10 - HP Search Chart ─────────────────────────────────────────────
d10 = OUT / "10_hp_search"
ensure_dir(d10)

df_hp = pd.read_csv(ANALYSIS / "hp_search" / "hp_search_results.csv")

# Sort by val spearman
if 'val_spearman' in df_hp.columns:
    spearman_col = 'val_spearman'
elif 'best_val_spearman' in df_hp.columns:
    spearman_col = 'best_val_spearman'
else:
    # Try to find the right column
    spearman_cols = [c for c in df_hp.columns if 'spearman' in c.lower()]
    spearman_col = spearman_cols[0] if spearman_cols else None

if spearman_col:
    df_hp_sorted = df_hp.sort_values(spearman_col, ascending=False).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    x = np.arange(len(df_hp_sorted))
    vals = df_hp_sorted[spearman_col].values
    
    # Color by performance: best = green, worst = red
    colors_hp = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(vals)))
    
    ax.bar(x, vals, color=colors_hp, alpha=0.85, edgecolor='none', width=0.8)
    
    # Reference lines
    default_spearman = 0.372226
    best_spearman = 0.376688
    ax.axhline(y=default_spearman, color='#f5a623', linestyle='--', linewidth=1.5,
               label=f'Default ({default_spearman:.4f})', alpha=0.8)
    ax.axhline(y=best_spearman, color='#16c79a', linestyle='--', linewidth=1.5,
               label=f'Best ({best_spearman:.4f})', alpha=0.8)
    
    ax.set_xlabel('Configuration (sorted by Spearman)', fontsize=11)
    ax.set_ylabel('Validation Spearman', fontsize=11)
    ax.set_title('Cerca d\'Hiperparàmetres MLP: 108 Configuracions\n'
                 f'Δ best-default = {best_spearman - default_spearman:.4f} (negligible)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.7)
    ax.grid(axis='y', alpha=0.3)
    
    # Tight y-range to emphasize similarity
    ax.set_ylim(vals.min() - 0.005, vals.max() + 0.005)
    ax.set_xticks([])
    
    plt.tight_layout()
    fig.savefig(d10 / "hp_search_spearman_all_configs.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ 10_hp_search/hp_search_spearman_all_configs.png")
else:
    print(f"  ⚠ Could not find Spearman column in HP search results. Columns: {list(df_hp.columns)}")

# ── SUMMARY TABLE GRAPHIC ────────────────────────────────────────────
# Create a beautiful summary comparison table as an image
d_summary = OUT / "00_summary"
ensure_dir(d_summary)

# Main comparison data
models_data = [
    ("Global Mean",           0.000,  "—",    0.1552, "—",    "—"),
    ("Champion Mean",         0.1249, 0.3360, 0.1440, "41.1%", "72.9%"),
    ("MLP OneHot",            0.1545, 0.3807, 0.1412, "41.9%", "74.1%"),
    ("MLP Embed",             0.1496, 0.3755, 0.1416, "41.8%", "73.8%"),
    ("MLP Per-Role",          0.1544, 0.3806, 0.1412, "41.8%", "74.1%"),
    ("HistGBT",               0.1599, 0.3874, 0.1408, "41.9%", "74.2%"),
    ("HistGBT + Pair TE",     0.1614, 0.3882, 0.1408, "41.8%", "74.2%"),
    ("ICC Ceiling (botlane)", 0.1726, "—",    "—",    "—",    "—"),
]

fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

col_labels = ['Model', 'R²', 'Spearman', 'MAE', '±0.10', '±0.20']
cell_text = [[m[0]] + [str(v) for v in m[1:]] for m in models_data]

table = ax.table(cellText=cell_text, colLabels=col_labels,
                 cellLoc='center', loc='center')

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 1.8)

# Style header
for j, label in enumerate(col_labels):
    cell = table[0, j]
    cell.set_facecolor('#e94560')
    cell.set_text_props(color='white', fontweight='bold', fontsize=12)

# Style rows
for i, row_data in enumerate(models_data):
    for j in range(len(col_labels)):
        cell = table[i+1, j]
        if i == len(models_data) - 1:  # Ceiling row
            cell.set_facecolor('#533483')
            cell.set_text_props(color='#eaeaea', fontstyle='italic')
        elif i == 0:  # Global Mean
            cell.set_facecolor('#2a2a4a')
            cell.set_text_props(color='#888888')
        elif row_data[0] == "HistGBT + Pair TE":  # Best model
            cell.set_facecolor('#0f3460')
            cell.set_text_props(color='#16c79a', fontweight='bold')
        else:
            cell.set_facecolor('#16213e')
            cell.set_text_props(color='#eaeaea')

ax.set_title('Resum de Models – Test Set (57,468 partides)\n'
             'Dataset: 383k observacions · Target: support_roam_score_v5',
             fontsize=14, fontweight='bold', color='#eaeaea', pad=20)

fig.patch.set_facecolor('#1a1a2e')
plt.tight_layout()
fig.savefig(d_summary / "summary_table.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ 00_summary/summary_table.png")


# ── KEY FINDINGS INFOGRAPHIC ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.patch.set_facecolor('#1a1a2e')

# Panel 1: Escalation from baselines to model to ceiling
ax1 = axes[0, 0]
models = ['Global\nMean', 'Champion\nMean', 'Best MLP', 'Best GBT', 'ICC\nCeiling']
r2_vals = [0.0, 0.1249, 0.1545, 0.1614, 0.1726]
bar_colors = ['#555555', '#f5a623', '#4fc3f7', '#e94560', '#533483']
bars = ax1.bar(models, r2_vals, color=bar_colors, alpha=0.9, edgecolor='none', width=0.6)
for bar, val in zip(bars, r2_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10, color='#eaeaea')
ax1.set_ylabel('R²')
ax1.set_title('Escalació de R²', fontweight='bold', fontsize=12)
ax1.set_ylim(0, 0.22)
ax1.grid(axis='y', alpha=0.3)

# Panel 2: Tolerance distribution
ax2 = axes[0, 1]
tol_labels = ['±0.05', '±0.10', '±0.15', '±0.20']
tol_vals = [21.65, 41.83, 59.81, 74.19]
bars2 = ax2.bar(tol_labels, tol_vals, color=COLORS[3], alpha=0.9, edgecolor='none', width=0.5)
for bar, val in zip(bars2, tol_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, color='#eaeaea')
ax2.set_ylabel('% prediccions')
ax2.set_title('Prediccions dins de Tolerància\n(HistGBT + Pair TE)', fontweight='bold', fontsize=12)
ax2.set_ylim(0, 95)
ax2.grid(axis='y', alpha=0.3)

# Panel 3: Model reaches ceiling
ax3 = axes[0, 2]
labels3 = ['Model R²', 'Ceiling R²']
vals3 = [0.1614, 0.1726]
bars3 = ax3.bar(labels3, vals3, color=[COLORS[0], COLORS[4]], alpha=0.9, width=0.4)
pct = 0.1614 / 0.1726 * 100
for bar, val in zip(bars3, vals3):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.4f}', ha='center', va='bottom', fontsize=11, color='#eaeaea')
ax3.set_title(f'Model arriba al {pct:.0f}% del Sostre', fontweight='bold', fontsize=12)
ax3.set_ylim(0, 0.22)
ax3.grid(axis='y', alpha=0.3)

# Panel 4: HP search futility
ax4 = axes[1, 0]
ax4.text(0.5, 0.6, '108', fontsize=48, fontweight='bold', color=COLORS[0],
         ha='center', va='center', transform=ax4.transAxes)
ax4.text(0.5, 0.35, 'configs avaluades', fontsize=14, color='#aaaaaa',
         ha='center', va='center', transform=ax4.transAxes)
ax4.text(0.5, 0.15, 'Δ Spearman = +0.005', fontsize=16, fontweight='bold',
         color='#f5a623', ha='center', va='center', transform=ax4.transAxes)
ax4.set_title('Cerca d\'Hiperparàmetres', fontweight='bold', fontsize=12)
ax4.set_xticks([])
ax4.set_yticks([])

# Panel 5: Label robustness
ax5 = axes[1, 1]
ax5.text(0.5, 0.6, '15', fontsize=48, fontweight='bold', color=COLORS[3],
         ha='center', va='center', transform=ax5.transAxes)
ax5.text(0.5, 0.35, 'variants de fórmula', fontsize=14, color='#aaaaaa',
         ha='center', va='center', transform=ax5.transAxes)
ax5.text(0.5, 0.15, 'Correlació ≥ 0.99', fontsize=16, fontweight='bold',
         color='#16c79a', ha='center', va='center', transform=ax5.transAxes)
ax5.set_title('Robustesa de l\'Etiqueta', fontweight='bold', fontsize=12)
ax5.set_xticks([])
ax5.set_yticks([])

# Panel 6: Error source
ax6 = axes[1, 2]
ax6.text(0.5, 0.6, '17/20', fontsize=42, fontweight='bold', color=COLORS[6],
         ha='center', va='center', transform=ax6.transAxes)
ax6.text(0.5, 0.35, 'top errors són de\npartides caòtiques', fontsize=13, color='#aaaaaa',
         ha='center', va='center', transform=ax6.transAxes)
ax6.text(0.5, 0.12, 'El draft no pot predir\nel caos a l\'execució', fontsize=11,
         color='#ff8a65', ha='center', va='center', transform=ax6.transAxes,
         fontstyle='italic')
ax6.set_title('Font dels Errors', fontweight='bold', fontsize=12)
ax6.set_xticks([])
ax6.set_yticks([])

fig.suptitle('Resum de Troballes Clau – TFG: Win Conditions i Predicció de Comportament',
             fontsize=16, fontweight='bold', color='#eaeaea', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(d_summary / "key_findings_infographic.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  ✓ 00_summary/key_findings_infographic.png")


# ═══════════════════════════════════════════════════════════════════════
#  3. GENERATE README INDEX
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  GENERATING README INDEX")
print("="*60)

readme_lines = [
    "# Material per a l'Entrevista amb el Tutor",
    "",
    "Directori preparat automàticament amb totes les gràfiques i visualitzacions",
    "organitzades per tema. Cada carpeta conté les imatges rellevants.",
    "",
    "## Estructura",
    "",
]

for theme_dir, desc in sorted(THEMES.items()):
    theme_path = OUT / theme_dir
    if theme_path.exists():
        pngs = sorted(theme_path.glob("*.png"))
        readme_lines.append(f"### 📁 `{theme_dir}/` – {desc}")
        for png in pngs:
            readme_lines.append(f"- `{png.name}`")
        readme_lines.append("")

# Also add summary
summary_path = OUT / "00_summary"
if summary_path.exists():
    pngs = sorted(summary_path.glob("*.png"))
    readme_lines.insert(7, f"### 📁 `00_summary/` – Resum general i taula comparativa")
    for i, png in enumerate(pngs):
        readme_lines.insert(8 + i, f"- `{png.name}`")
    readme_lines.insert(8 + len(pngs), "")

readme_lines.extend([
    "## Xifres Clau",
    "",
    "| Mètrica | Valor |",
    "| --- | --- |",
    "| Dataset | 383k observacions (191k partides) |",
    "| Split | 268k/57k/57k (train/val/test) |",
    "| Millor R² (HistGBT) | 0.1614 |",
    "| Millor Spearman | 0.3882 |",
    "| Sostre ICC (botlane+side) | 0.1726 |",
    "| % del sostre assolit | 93.5% |",
    "| ±0.10 tolerància | 41.8% |",
    "| ±0.20 tolerància | 74.2% |",
    "| HP search Δ Spearman | +0.005 (negligible) |",
    "| Variants de fórmula | 15, totes correlació ≥ 0.99 |",
    "| Errors explicats per caos | 17/20 top errors |",
    "",
    "## Missatge Principal",
    "",
    "> El draft conté senyal predictiu real però limitat (R²≈0.16).",
    "> El model arriba al 93% del sostre empíric.",
    "> El coll d'ampolla no és l'arquitectura sinó la informació pre-partida.",
    "> R²=0.16 no és un \"mal resultat\" sinó una **troballa empírica legítima**.",
])

readme_text = "\n".join(readme_lines)
(OUT / "README.md").write_text(readme_text, encoding='utf-8')
print(f"  ✓ README.md")

print("\n" + "="*60)
print(f"  DONE! Tot el material es troba a: {OUT}")
print("="*60)
