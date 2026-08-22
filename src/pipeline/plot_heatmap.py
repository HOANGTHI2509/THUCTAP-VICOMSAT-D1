import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("Generating Metric Heatmap...")
    df = pd.read_csv('artifacts/benchmark_summary.csv')
    models = df['Model'].values

    # Metrics list and their preferred direction (1 for higher is better, -1 for lower is better)
    metrics_info = {
        'RMSE (L) ↓': -1,
        'SNR (dB) ↑': 1,
        'Smoothness (L/step)': -1,
        'Refuel F1 ↑': 1,
        'Theft F1 ↑': 1,
        'Delay (steps) ↓': -1,
        'Latency (ms) ↓': -1
    }

    matrix = []
    labels = []
    for metric, direction in metrics_info.items():
        # Handle cases where latency might be '~0' or NaN
        vals = pd.to_numeric(df[metric].astype(str).str.replace('~', ''), errors='coerce').fillna(0).values
        v_min = np.min(vals)
        v_max = np.max(vals)
        
        if v_max == v_min:
            norm = np.ones_like(vals)
        else:
            if direction == 1:
                norm = (vals - v_min) / (v_max - v_min)
            else:
                norm = (v_max - vals) / (v_max - v_min)
        matrix.append(norm)
        
        # Format label cleanly
        formatted_labels = []
        for val in vals:
            if val == 0:
                formatted_labels.append("0.0")
            elif val < 0.01:
                formatted_labels.append(f"{val:.3f}")
            else:
                formatted_labels.append(f"{val:.2f}")
        labels.append(formatted_labels)

    matrix = np.array(matrix)
    labels = np.array(labels)

    # Clean up model names for plotting
    model_names = [m.replace('1D-CNN + Gated Attention', 'CNN-GA').replace('Standard Kalman', 'Standard').replace('Adaptive Kalman', 'Adaptive') for m in models]
    metric_names = [m.replace(' (L)', '').replace(' (dB)', '').replace(' (L/step)', '').replace(' (steps)', '').replace(' (ms)', '') for m in metrics_info.keys()]

    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(matrix, annot=labels, fmt="", cmap="RdYlGn", 
                     xticklabels=model_names, yticklabels=metric_names,
                     cbar_kws={'label': 'Tương quan hiệu suất (0 = Kém nhất, 1 = Tốt nhất)'},
                     vmin=0, vmax=1)

    # Move x-axis labels to top
    ax.xaxis.tick_top()
    plt.xticks(rotation=0, fontsize=11, fontweight='bold')
    plt.yticks(rotation=0, fontsize=11, fontweight='bold')
    plt.title("PERFORMANCE MATRIX: CNN-GA vs KALMAN", pad=40, fontsize=16, fontweight='bold')
    plt.tight_layout()

    os.makedirs('artifacts', exist_ok=True)
    plt.savefig('artifacts/metric_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved heatmap to artifacts/metric_heatmap.png")

if __name__ == '__main__':
    main()
