import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# Excel-Datei laden
# ============================================
file_path = "/Users/marcschneider/Library/CloudStorage/OneDrive-THKöln/Desktop/Experiments_Summray.xlsx"
sheet_name = "modular_plot"
df = pd.read_excel(file_path, sheet_name=sheet_name)

# ============================================
# Werte bereinigen
# ============================================
df['Parameters'] = df['Parameters'].astype(str).str.replace('.', '', regex=False).astype(float) / 1_000_000
df['Inference Time'] = df['Inference Time'].astype(str).str.replace(',', '.', regex=False).astype(float)
df['mAP50-95'] = (
    df['mAP50-95'].astype(str)
    .str.replace(',', '.', regex=False).str.replace('%', '', regex=False)
    .astype(float))
df['GFLOPs'] = df['GFLOPs'].astype(str).str.replace(',', '.', regex=False).astype(float)

# ============================================
# Gruppen / Position
# ============================================
def get_group(model_name):
    if 'C2fBF' in model_name:   return 'C2fBF'
    elif 'C2fSTR' in model_name:return 'C2fSTR'
    elif 'C2fPVT' in model_name:return 'C2fPVT'
    elif 'yolov8m' in model_name:return 'YOLOv8m'
    elif 'yolov8l' in model_name:return 'YOLOv8l'
    elif 'yolov8s' in model_name:return 'YOLOv8s'
    elif 'yolov8x' in model_name:return 'YOLOv8x'
    elif 'yolov8n' in model_name:return 'YOLOv8n'
    else:                      return 'Other'

def get_block_position(model_name):
    return model_name.split('_')[1] if '_' in model_name else model_name

def extract_p_positions(text):
    matches = re.findall(r'P\d+', str(text))
    joined = ''.join(matches) if matches else None
    return joined

df['Group'] = df['Model'].apply(get_group)
df['Block_Position'] = df['Model'].apply(get_block_position)
df['P_Pos'] = df['Block_Position'].apply(extract_p_positions)

# ============================================
# Kurzlabels für YOLO-Baselines
# ============================================
yolo_short = {'yolov8n':'n','yolov8s':'s','yolov8m':'m','yolov8l':'l','yolov8x':'x'}

def label_for_row(row):
    if row['Model'] in yolo_short:
        return yolo_short[row['Model']]
    bp = str(row['Block_Position'])
    if 'full' in bp.lower():
        return 'Full'
    return row['P_Pos']

group_colors = {
    'YOLOv8n': '#1f77b4', 'YOLOv8s': '#1f77b4', 'YOLOv8m': '#1f77b4', 'YOLOv8l': '#1f77b4', 'YOLOv8x': '#1f77b4',
    'C2fBF': '#d62728', 'C2fSTR': '#2ca02c', 'C2fPVT': '#9467bd', 'Other': 'gray'
}

# ============================================
# GFLOPs → Punktgröße skalieren
# ============================================
gflops_min, gflops_max = df['GFLOPs'].min(), df['GFLOPs'].max()
df['Size'] = 200 + 1500 * (df['GFLOPs'] - gflops_min) / (gflops_max - gflops_min)

# ============================================
# Baselines (nur originale YOLOv8)
# ============================================
base_order = ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x']
base_df = df[df['Model'].isin(base_order)].copy()
base_df['__order__'] = base_df['Model'].map({m:i for i,m in enumerate(base_order)})
base_df = base_df.sort_values('__order__')

x_base_params = base_df['Parameters'].tolist()
x_base_time   = base_df['Inference Time'].tolist()
y_base        = base_df['mAP50-95'].tolist()

ymin = df['mAP50-95'].min() - 1
ymax = df['mAP50-95'].max() + 1

# ============================================
# Manuelle Label-Positionierung
# ============================================
manual_label_positions = {
    "p3c2fbf": "right",
    "p2p3c2fstr": "left",
    "p2p3c2fbf": "right",
    "p2c2fbf": "left",
    "p2c2fstr": "left",
    "fullc2fstr": "right",
    "p10c2fpvt": "right",
    "p10c2fstr": "right",
    "p3c2fstr": "left",
    "p2p3c2fpbt": "right",
}

# ============================================
# Plotfunktion
# ============================================
def plot_scatter(ax, x_col, xlabel, x_base):
    if len(x_base) >= 2:
        ax.plot(x_base, y_base, linestyle='--', linewidth=3.5, color='#1f77b4')

    x_range = df[x_col].max() - df[x_col].min()
    y_range = ymax - ymin
    offset_x = x_range * 0.01
    offset_y = y_range * 0.01

    added_labels = set()
    for group, group_df in df.groupby('Group'):
        label = "YOLOv8" if group.startswith("YOLOv8") else group
        ax.scatter(
            group_df[x_col],
            group_df['mAP50-95'],
            s=group_df['Size'],
            color=group_colors.get(group, 'gray'),
            alpha=0.85,
            label=(label if label not in added_labels else "")
        )
        added_labels.add(label)

        # Nur YOLOv8 beschriften
        for _, row in group_df.iterrows():
            txt = label_for_row(row)
            if not txt:
                continue
            x, y = row[x_col], row['mAP50-95']
            model_lower = row['Model'].lower()

            matched_key = next((k for k in manual_label_positions if k in model_lower), None)
            pos = manual_label_positions.get(matched_key, "top")

            if pos == "right":
                ax.text(x + offset_x, y, txt, fontsize=18, ha='left', va='center', fontweight='medium')
            elif pos == "left":
                ax.text(x - offset_x, y, txt, fontsize=18, ha='right', va='center', fontweight='medium')
            else:
                ax.text(x, y + offset_y, txt, fontsize=18, ha='center', va='bottom', fontweight='medium')

    ax.set_xlabel(xlabel, fontsize=28, labelpad=12)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=24, width=1.5)

# ============================================
# Figure Setup
# ============================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 11))

plot_scatter(ax1, 'Parameters', 'Parameters [Million]', x_base_params)
ax1.set_ylabel(r'mAP$_{50–95}$ [%]', fontsize=28, labelpad=12)

plot_scatter(ax2, 'Inference Time', 'Inference Time [ms] (PyTorch)', x_base_time)
ax2.set_ylabel(r'mAP$_{50–95}$ [%]', fontsize=28, labelpad=12)

# ============================================
# Legende
# ============================================
handles, labels = ax1.get_legend_handles_labels()
legend_order = ['YOLOv8', 'C2fBF', 'C2fSTR', 'C2fPVT']
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
# Speichern – hochauflösend
# ============================================
fig.savefig("modular_backbone_benchmark_gflops_smalllabels.png",
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.05,
            transparent=True)

plt.show()
