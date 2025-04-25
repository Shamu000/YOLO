import os
import pandas as pd
import matplotlib.pyplot as plt

def load_csvs(file_paths):
    datasets = {}
    for path in file_paths:
        if not os.path.isfile(path):
            print(f"Warning: File not found: {path}")
            continue
        # 使用父目录名加文件名作为标签
        parent = os.path.basename(os.path.dirname(path))
        filename = os.path.splitext(os.path.basename(path))[0]
        name = f"{parent}/{filename}"
        df = pd.read_csv(path)
        datasets[name] = df
    return datasets

def plot_metrics(datasets, metrics, max_epoch=None):
    num_metrics = len(metrics)
    num_cols = 3
    num_rows = (num_metrics + num_cols - 1) // num_cols
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*6, num_rows*4))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for idx, (name, df) in enumerate(datasets.items()):
            if metric not in df.columns:
                continue
            epochs = df["epoch"]
            values = df[metric]
            if max_epoch:
                mask = epochs <= max_epoch
                epochs = epochs[mask]
                values = values[mask]
            color = colors[idx % len(colors)]
            ax.plot(epochs, values, label=name, color=color)
        ax.set_title(metric, fontsize=12)
        ax.set_xlabel("", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.grid(True)
        ax.legend(fontsize=8)
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_fontsize(8)
        for label in ax.get_yticklabels():
            label.set_fontsize(8)

    # 隐藏多余子图
    for j in range(len(metrics), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 在这里替换为你的实际 CSV 路径
    csv_files = [
        "/home/marshal_kim/下载/models/4.25.1/results.csv",
        "/home/marshal_kim/下载/models/4.24.3/results.csv"
    ]
    metrics_to_plot = [
        "train/box_loss", "train/pose_loss", "train/cls_loss",
        "val/box_loss", "val/pose_loss", "val/cls_loss",
        "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        "metrics/mAP50(P)", "metrics/mAP50-95(P)"
    ]

    data = load_csvs(csv_files)
    plot_metrics(data, metrics_to_plot)
