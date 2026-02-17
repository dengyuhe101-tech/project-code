import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import config
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_and_process_data(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: File does not exist {file_path}.")
        return None
    try:
        df = pd.read_csv(file_path)
        df = df[df['GT_Box'].notna()]
        df['GT_Box'] = df['GT_Box'].astype(str)
        invalid_markers = ['0', '[]', 'nan', 'None', '', 'null']
        df = df[~df['GT_Box'].str.strip().isin(invalid_markers)]
        if df.empty:
            print(f"[Warning] {file_path} is empty after cleaning.")
            return None
        
        highest_iou_idx = df.groupby('GT_Box')['IoU'].idxmax()
        return df.loc[highest_iou_idx]
    except Exception as e:
        print(f"[Error] Failed to process {file_path}: {e}")
        return None

def get_smart_xticks(min_val, max_val, base_step, specials=None):
    """
    Generate clean integer xticks, ensuring special values are included.
    """
    if specials is None:
        specials = []
        
    current_step = base_step
   
    while (max_val - min_val) / current_step > 15:
        if current_step < 100: current_step += base_step
        else: current_step *= 2
            
    ticks = np.arange(min_val, max_val + 1, current_step).astype(int).tolist()
    
    final_ticks = []
    params = sorted(list(set(ticks + specials)))
    # Threshold to prevent overlapping (3% of total range)
    threshold = (max_val - min_val) * 0.03 
    
    for val in params:
        if val in specials:
            final_ticks.append(val)
        else:
            is_too_close = False
            for s in specials:
                if abs(val - s) < threshold:
                    is_too_close = True
                    break
            if not is_too_close:
                final_ticks.append(val)
                
    return sorted(list(set(final_ticks)))

def plot_single_chart(data_map, plot_conf):
    min_area = plot_conf['min']
    max_area = plot_conf['max']
    bin_width = plot_conf['bin']
    xtick_step = plot_conf['step']
    highlights = plot_conf.get('highlights', []) 

    print(f"Plotting: {plot_conf['name']} (Range: {min_area}-{max_area})")

    bins = np.arange(min_area, max_area + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, ax1 = plt.subplots(figsize=(14, 8), dpi=300)

    # 1. Background Histogram (Sample Distribution)
    first_df = list(data_map.values())[0]
    counts, _ = np.histogram(first_df['GT_Area'].values, bins=bins)
    ax2 = ax1.twinx()
    ax2.bar(bin_centers, counts, width=bin_width*0.8, color='gray', alpha=0.15, label='Sample Count')
    ax2.set_ylabel('Number of Samples', color='gray')
    ax2.grid(False)

    # 2. Vertical Lines for Highlights
    for val in highlights:
        ax1.axvline(x=val, color='black', linestyle='--', linewidth=1.5, alpha=0.6, zorder=0)
        
    colors = ['#1f77b4', '#d62728', '#2E8B57', '#FFA500', '#FF4500', '#000000', '#7B68EE']
    markers = ['o', 's', '^', 'D', 'x', 'v', '*']
    min_samples = 5 

    for idx, (label, df) in enumerate(data_map.items()):
        areas = df['GT_Area'].values
        ious = df['IoU'].values
        bin_means = []
        for i in range(len(bins) - 1):
            mask = (areas >= bins[i]) & (areas < bins[i + 1])
            if np.sum(mask) >= min_samples:
                bin_means.append(np.mean(ious[mask]))
            else:
                bin_means.append(np.nan)
        
        if not np.all(np.isnan(bin_means)):
             ax1.plot(bin_centers, bin_means, marker=markers[idx%7], markersize=6, 
                     label=label, color=colors[idx%7], linewidth=2, alpha=0.9)

    ax1.set_xlabel('Ground Truth Bounding Box Area(pixels)', fontsize=14)
    ax1.set_ylabel('Average IoU', fontsize=14)
    ax1.set_title(f'Avg IoU vs Area ({plot_conf["name"]})', fontsize=16)
    
    # Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower right', fontsize=12)
    
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 1.05)
    
    # X-Ticks
    custom_xticks = get_smart_xticks(min_area, max_area, xtick_step, specials=highlights)
    ax1.set_xticks(custom_xticks)
    ax1.set_xticklabels(custom_xticks, rotation=0, fontsize=11, fontweight='normal')
    
    for label in ax1.get_xticklabels():
        if label.get_text() in [str(x) for x in highlights]:
            label.set_fontweight('bold') 
            label.set_color('black')
    
    plt.tight_layout()
    save_path = os.path.join(config.OUTPUT_DIR, f'plot_{plot_conf["name"]}.png')
    ensure_dir(config.OUTPUT_DIR)
    plt.savefig(save_path)
    print(f"  -> Chart saved to: {save_path}")
    plt.close()

def run():
    print(">>> [Task] Comparison Plotting Started...")
    data_map = {}
    
    for item in config.COMPARISON_DATA_SOURCES:
        df = load_and_process_data(item['file'])
        if df is not None: 
            data_map[item['label']] = df
            
    if not data_map: 
        print("No valid data loaded. Process terminated.")
        return

    for plot_conf in config.PLOT_CONFIGS:
        plot_single_chart(data_map, plot_conf)
    
    print(">>> [Task] Comparison Plotting Completed.\n")
