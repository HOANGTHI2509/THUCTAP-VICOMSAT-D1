import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

CLASSES = [
    "UPWARD_SHIFT",
    "DOWNWARD_SHIFT",
    "GRADUAL_CHANGE",
    "STABLE_JITTER",
    "OSCILLATION_NOISE",
]

def plot_confusion_matrix_svg(cm, title, save_path_svg, save_path_png=None):
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    ax.set(
        xticks=np.arange(len(CLASSES)),
        yticks=np.arange(len(CLASSES)),
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        title=title,
        ylabel="Actual label",
        xlabel="Predicted label",
    )

    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor", fontsize=9.5)
    plt.setp(ax.get_yticklabels(), fontsize=9.5)
    ax.title.set_fontsize(12)
    ax.xaxis.label.set_fontsize(10.5)
    ax.yaxis.label.set_fontsize(10.5)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            ax.text(
                j,
                i,
                f"{int(val):,}",
                ha="center",
                va="center",
                color=color,
                fontsize=9.5,
                fontweight="normal",
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path_svg), exist_ok=True)
    fig.savefig(save_path_svg, format="svg", bbox_inches="tight")
    if save_path_png:
        os.makedirs(os.path.dirname(save_path_png), exist_ok=True)
        fig.savefig(save_path_png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    csv_dir = "reports/confusion_matrices/csv"
    svg_dir = "reports/confusion_matrices/svg"
    png_dir = "reports/confusion_matrices/png"
    os.makedirs(svg_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    print(f"Tìm thấy {len(csv_files)} file ma trận CSV. Bắt đầu xuất SVG...")

    # Xuất cho từng xe
    for f in csv_files:
        df = pd.read_csv(f, index_col=0)
        vname = os.path.basename(f).replace("cm_", "").replace(".csv", "")
        cm = df.values.astype(int)
        title = f"Confusion Matrix - {vname}"
        svg_path = os.path.join(svg_dir, f"cm_{vname}.svg")
        png_path = os.path.join(png_dir, f"cm_{vname}.png")
        plot_confusion_matrix_svg(cm, title, svg_path, png_path)
        print(f"  + Đã xuất SVG: {svg_path}")

    # Xuất các ma trận tổng hợp
    # 1. Test set (3 xe test held-out)
    test_cars = ["90H_03494", "92H_02687", "Car_5"]
    test_cm = np.zeros((5, 5), dtype=int)
    for tc in test_cars:
        f = os.path.join(csv_dir, f"cm_{tc}.csv")
        if os.path.exists(f):
            test_cm += pd.read_csv(f, index_col=0).values.astype(int)
    
    test_svg = os.path.join("reports/confusion_matrices", "test_held_out_confusion_matrix.svg")
    test_png = os.path.join("reports/confusion_matrices", "test_held_out_confusion_matrix.png")
    plot_confusion_matrix_svg(test_cm, "Test Confusion Matrix (Held-out: 90H, 92H, Car 5)", test_svg, test_png)
    print(f"\n=> Đã xuất Ma trận Test Held-out: {test_svg}")

    # 2. Overall Fleet (toàn bộ 34 xe)
    fleet_cm = np.zeros((5, 5), dtype=int)
    for f in csv_files:
        fleet_cm += pd.read_csv(f, index_col=0).values.astype(int)
    
    overall_svg = os.path.join("reports/confusion_matrices", "overall_fleet_confusion_matrix.svg")
    overall_png = os.path.join("reports/confusion_matrices", "overall_fleet_confusion_matrix.png")
    plot_confusion_matrix_svg(fleet_cm, f"Overall Fleet Confusion Matrix (34 Vehicles - {fleet_cm.sum():,} points)", overall_svg, overall_png)
    print(f"=> Đã xuất Ma trận Toàn bộ Hạm đội: {overall_svg}")

    print("\nHoàn tất xuất toàn bộ ảnh SVG không vỡ nét!")

if __name__ == "__main__":
    main()
