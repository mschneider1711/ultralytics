import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================
# Excel-Datei laden
# ============================================
file_path = "/Users/marcschneider/Library/CloudStorage/OneDrive-THKöln/Desktop/Experiments_Summray.xlsx"
sheet_name = "backbone_plot"
df = pd.read_excel(file_path, sheet_name=sheet_name)

# ============================================
# Werte bereinigen
# ============================================
df['Parameters'] = df['Parameters'].astype(str).str.replace('.', '', regex=False).astype(float) / 1_000_000
df['Inference Time'] = df['Inference Time'].astype(str).str.replace(',', '.', regex=False).astype(float)
df['GFLOPs'] = df['GFLOPs'].astype(str).str.replace(',', '.', regex=False).astype(float)
df['mAP50-95'] = (
    df['mAP50-95'].astype(str)
      .str.replace(',', '.', regex=False).str.replace('%', '', regex=False)
      .astype(float)
)

df = df.dropna(subset=['Parameters', 'Inference Time', 'GFLOPs', 'mAP50-95'])

# ============================================
# Gruppen definieren
# ============================================
def get_group(model_name: str) -> str:
    name = str(model_name).lower()
    if name.startswith('yolov8'):
        return 'YOLOv8'
    if 'biformer' in name:
        return 'BiFormer'
    if 'swin' in name:
        return 'Swin'
    if 'pvt' in name:
        return 'PVT'
    return 'Other'

df['Group'] = df['Model'].apply(get_group)

group_colors = {
    'YOLOv8': '#1f77b4',   # Blau
    'BiFormer': '#d62728', # Rot
    'Swin': '#2ca02c',     # Grün
    'PVT': '#9467bd',      # Violett
    'Other': 'gray'
}

# ============================================
# Labels für YOLOv8 (n, s, m, l, x)
# ============================================
def make_label(row):
    name = str(row['Model']).lower()
    if 'yolov8n' in name: return 'n'
    if 'yolov8s' in name: return 's'
    if 'yolov8m' in name: return 'm'
    if 'yolov8l' in name: return 'l'
    if 'yolov8x' in name: return 'x'
    return ''  # keine Labels für Custom Modelle

df['Label'] = df.apply(make_label, axis=1)

# ============================================
# YOLOv8-Baseline sortieren
# ============================================
is_yolo = df['Group'] == 'YOLOv8'

def get_sorted_xy(xcol):
    sub = df[is_yolo][[xcol, 'mAP50-95']].dropna().copy()
    sub = sub.sort_values(by=xcol)
    return sub[xcol].to_numpy(), sub['mAP50-95'].to_numpy()

# ============================================
# GFLOPs → Punktgröße skalieren
# ============================================
gflops_min, gflops_max = df['GFLOPs'].min(), df['GFLOPs'].max()
df['Size'] = 200 + 1500 * (df['GFLOPs'] - gflops_min) / (gflops_max - gflops_min)

# ============================================
# Plotfunktion
# ============================================
def plot_scatter(ax, xcol, xlabel):
    xs, ys = get_sorted_xy(xcol)

    # Verbindungslinie YOLOv8
    if len(xs) >= 2:
        ax.plot(xs, ys, linestyle='--', linewidth=3.5, color=group_colors['YOLOv8'])

    # Offsets
    x_range = df[xcol].max() - df[xcol].min()
    y_range = 56 - 46
    offset_y = y_range * 0.018

    # Punkte zeichnen
    added_labels = set()
    for group, gdf in df.groupby('Group'):
        label = group
        ax.scatter(
            gdf[xcol],
            gdf['mAP50-95'],
            s=gdf['Size'],
            color=group_colors.get(group, 'gray'),
            alpha=0.85,
            label=(label if label not in added_labels else "")
        )
        added_labels.add(label)

        # Nur YOLOv8 beschriften
        if group == 'YOLOv8':
            for _, row in gdf.iterrows():
                if not row['Label']:
                    continue
                ax.text(
                    row[xcol],
                    row['mAP50-95'] + offset_y * 1.2,
                    row['Label'],
                    fontsize=24,
                    ha='center',
                    va='bottom',
                    fontweight='bold'
                )

    ax.set_xlabel(xlabel, fontsize=28, labelpad=12)
    ax.set_ylim(46, 56)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=24, width=1.5)

# ============================================
# Figure Setup
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 11))

plot_scatter(ax1, 'Parameters', 'Parameters [Million]')
plot_scatter(ax2, 'Inference Time', 'Inference Time [ms] (TensorRT FP16)')

ax1.set_ylabel(r'mAP$_{50–95}$ [%]', fontsize=28, labelpad=12)
ax2.set_ylabel(r'mAP$_{50–95}$ [%]', fontsize=28, labelpad=12)

# ============================================
# Legende
# ============================================
handles, labels = ax1.get_legend_handles_labels()
legend_order = ['YOLOv8', 'BiFormer', 'Swin', 'PVT']
legend_dict = {k: v for k, v in zip(labels, handles)}
ordered_handles = [legend_dict[k] for k in legend_order if k in legend_dict]
ordered_labels = [k for k in legend_order if k in legend_dict]

fig.legend(
    ordered_handles, ordered_labels,
    loc='lower center', bbox_to_anchor=(0.5, -0.02),
    ncol=4, frameon=False,
    fontsize=22, title='Model Group', title_fontsize=24
)

plt.subplots_adjust(left=0.06, right=0.98, bottom=0.23, top=0.94, wspace=0.2)

# ============================================
# Speichern
# ============================================
fig.savefig("backbone_benchmark_gflops_noborder.png",
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.05,
            transparent=True)

plt.show()
