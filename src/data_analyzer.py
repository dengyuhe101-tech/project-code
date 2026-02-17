# src/data_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import config

def kneedle(x, y, x_limit):
    """
    Kneedle algorithm to find the knee point of a curve.
    """
    mask = x <= x_limit
    x = x[mask]
    y = y[mask]
    
    if len(x) < 3:
        return None, None

    x = np.asarray(x)
    y = np.asarray(y)

    # Normalization
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    g = y_norm - x_norm
    idx = np.argmax(g)
    return x[idx], y[idx]

def detect_outliers_3sigma(df):
    """
    Detect outliers using 3-Sigma rule on rolling statistics.
    """
    df = df.copy()
    trend_window = 10
    std_window = 20
    
    # Calculate rolling trend and residual
    df['Trend'] = df['Mean'].rolling(window=trend_window, center=True, min_periods=1).mean()
    df['Residual'] = df['Mean'] - df['Trend']
    
    # Calculate rolling bounds
    df['Rolling_Mean_Res'] = df['Residual'].rolling(window=std_window, center=True, min_periods=5).mean()
    df['Rolling_Std'] = df['Residual'].rolling(window=std_window, center=True, min_periods=5).std()
    df['Centered_Residual'] = df['Residual'] - df['Rolling_Mean_Res']   

    df['Upper_Bound'] = config.OUTLIER_SIGMA * df['Rolling_Std']
    df['Lower_Bound'] = -config.OUTLIER_SIGMA * df['Rolling_Std']
    outliers = df[df['Centered_Residual'].abs() > df['Upper_Bound']]
    
    return df, outliers

def plot_kneedle_chart(ax, df, x_limit, knee_x, knee_y, title):
    """
    Draw a single Kneedle analysis chart.
    """
    x = df['Area'].values
    y = df['Mean'].values
    mask = x <= x_limit
    x_plot = x[mask]
    y_plot = y[mask]
    
    ax.plot(x_plot, y_plot, label='IoU Mean Curve', color='#1f77b4', linewidth=2)
    if len(x_plot) > 1:
        ax.plot([x_plot[0], x_plot[-1]], [y_plot[0], y_plot[-1]], 
                color='gray', linestyle='--', label='Reference Chord', alpha=0.7)
    
    if knee_x is not None and knee_y is not None:
        if x_plot[-1] != x_plot[0]:
            m = (y_plot[-1] - y_plot[0]) / (x_plot[-1] - x_plot[0])
            c = y_plot[0] - m * x_plot[0]
            y_on_line = m * knee_x + c
            
            ax.vlines(knee_x, y_on_line, knee_y, colors='red', linestyles='solid', alpha=0.5, label='Max Vertical Dist.')
            ax.scatter([knee_x], [knee_y], color='red', s=100, zorder=5, marker='o', edgecolors='white')
            ax.annotate(f'Knee Point\nArea={knee_x:.0f}', 
                        xy=(knee_x, knee_y), 
                        xytext=(knee_x + (x_plot[-1]-x_plot[0])*0.1, knee_y - (y_plot[-1]-y_plot[0])*0.1),
                        arrowprops=dict(arrowstyle="->", color='red'),
                        fontsize=11, color='red', fontweight='bold')

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(r'Area (pixels)')
    ax.set_ylabel('Mean IoU')
    ax.legend(loc='lower right')
    ax.grid(True, linestyle=':', alpha=0.6)

def plot_outlier_chart(ax, df, outliers):
    """
    Draw Outlier Analysis chart.
    """
    
    ax.plot(df['Area'], df['Centered_Residual'], label='Residual - Rolling Mean', color='blue', linewidth=1)
    ax.plot(df['Area'], df['Upper_Bound'], color='red', linestyle='--', label='3 Sigma Bound')
    ax.plot(df['Area'], df['Lower_Bound'], color='red', linestyle='--')
    
    if not outliers.empty:
        ax.scatter(outliers['Area'], outliers['Centered_Residual'], 
                   color='orange', s=50, zorder=5, label='Outliers')

    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_title('Outlier Detection', fontsize=14)
    ax.set_xlabel('Area (pixels)', fontsize=12)
    ax.set_ylabel('Deviation (Residual - Mean)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def run():
    file_path = config.ANALYSIS_FILE_PATH
    print(f">>> [Task] Data Analysis Started...")
    print(f"  -> Reading file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"[Error] Analysis file not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
        df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
        df['Mean'] = pd.to_numeric(df['Mean'], errors='coerce')
        df = df.dropna(subset=['Area', 'Mean']).sort_values('Area').reset_index(drop=True)
        
        # 1. Kneedle Analysis
        areas = df['Area'].values
        means = df['Mean'].values

        # A. Global Knee Point
        global_knee_x, global_knee_y = kneedle(areas, means, x_limit=config.KNEEDLE_GLOBAL_LIMIT)
        
        # B. Local Knee Point
        local_knee_x, local_knee_y = None, None
        local_limit = None 

        if global_knee_x is not None:
            local_knee_x, local_knee_y = kneedle(areas, means, x_limit=global_knee_x)
        else:
            print("  -> [Info] Global knee not found. Skipping local knee calculation.")

        # Print Results
        print("-" * 50)
        print("Analysis Result 1: Knee Point Detection (Kneedle)")
        print("-" * 50)
        if local_knee_x:
            print(f"  ->(1) Local Knee Point found at Area = {local_knee_x:.1f}")
        else:
            print("  ->[Warning] Local Knee Point: Skipped or Not Detected")
        
        if global_knee_x:
            print(f"  ->(2) Global Knee Point found at Area = {global_knee_x:.1f}")
        else:
            print("  ->[Warning] Global Knee Point: Not Detected")


        # 2. Outlier Analysis
        df_res, outliers = detect_outliers_3sigma(df)
        
        print("-" * 50)
        print("Analysis Result 2: Outlier Detection (3-Sigma)")
        print("-" * 50)
        
        if not outliers.empty:
            first_outlier_area = outliers.iloc[0]['Area']
            print(f"  -> First Outlier (Boundary Point): Area = {first_outlier_area:.1f}")
            
            if len(outliers) > 1:
                print(f"  -> (Note: {len(outliers)-1} additional outliers followed.)")
            else:
                print("  -> (Note: Only 1 isolated outlier detected.)")
                
            # print(f"  -> Debug: All Outliers: {outliers['Area'].values}")
            # Save outliers to CSV if needed
            # outliers.to_csv('outliers_detected.csv', index=False)
        else:
            print("  -> No significant outliers detected.")


        # 3. Plotting
        print("-" * 50)
        print("Generating Analysis Plots...")
        ensure_dir(config.OUTPUT_DIR)
        
        # Plot A: Kneedle
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    
        if global_knee_x is not None:
            plot_kneedle_chart(ax1, df, global_knee_x, local_knee_x, local_knee_y, 
                           title=f'Local Knee Detection\n(Range: Area <= {global_knee_x:.0f})')
        else:
            ax1.text(0.5, 0.5, 'Local Analysis Skipped\n(No Global Knee)', 
                     ha='center', va='center', color='gray')
            ax1.axis('off')
        
        plot_kneedle_chart(ax2, df, config.KNEEDLE_GLOBAL_LIMIT, global_knee_x, global_knee_y, 
                           title=f'Global Knee Detection\n(Range: Area <= {config.KNEEDLE_GLOBAL_LIMIT:.0f})')
        
        plt.tight_layout()
        save_path_kneedle = os.path.join(config.OUTPUT_DIR, 'analysis_kneedle_plot.png')
        plt.savefig(save_path_kneedle)
        plt.close(fig1)
        print(f"  -> Kneedle plot saved to: {save_path_kneedle}")

        # Plot B: Outlier
        fig2, ax_outlier = plt.subplots(figsize=(14, 6), dpi=300)
        plot_outlier_chart(ax_outlier, df_res, outliers)
    
        plt.tight_layout()
        save_path_outlier = os.path.join(config.OUTPUT_DIR, 'analysis_outlier_plot.png')
        plt.savefig(save_path_outlier)
        plt.close(fig2)
        print(f"  -> Outlier plot saved to: {save_path_outlier}")
            
        print("\n>>> [Task] Data Analysis Completed.\n")
        
    except Exception as e:
        print(f"[Error] Analysis failed: {e}")