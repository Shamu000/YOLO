import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

# 你的短标签映射表
category_mapping = {
    0: "BS",   1: "B1",  2: "B2",  3: "B3",  4: "B4",  5: "B5",
    6: "BO",   7: "BSB", 8: "BLB", 9: "RS", 10: "R1", 11: "R2",
    12: "R3", 13: "R4", 14: "R5", 15: "RO", 16: "RSB",17: "RLB",
    18: "OS", 19: "O1", 20: "O2", 21: "O3", 22: "O4", 23: "O5",
    24: "OO", 25: "OSB",26: "OLB"
}

def evaluate_pose_per_class(model_path, data_yaml, imgsz=640, batch_size=32):
    model = YOLO(model_path).to("cuda")
    metrics = model.val(
        data=data_yaml,
        split="test",
        batch=batch_size,
        imgsz=imgsz,
        verbose=False,
        plots=True,  # ✅ 启用绘图
        save=True  # ✅ 启用结果保存（txt、PR曲线、confusion matrix等）
    )

    names = model.names
    valid_indices = metrics.ap_class_index.tolist()  # 测试集中实际出现过的类别索引

    rows = []
    for cid in valid_indices:
        idx = valid_indices.index(cid)
        bp, br, bm50, bm5095, pp, pr, pm50, pm5095 = metrics.class_result(idx)
        rows.append({
            "ClassID": cid,
            "Short": category_mapping[cid],
            "Box(P)": bp, "Recall(B)": br,
            "mAP50(B)": bm50, "mAP50-95(B)": bm5095,
            "Pose(P)": pp, "Recall(P)": pr,
            "mAP50(P)": pm50, "mAP50-95(P)": pm5095
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    # === 配置部分 ===
    model_paths = [
        # "/home/wyc/Yolo11/runs/pose/2.22.2/weights/best.pt",
        # "/home/wyc/Yolo11/runs/pose/4.25.1/weights/best.pt",
        # "/home/wyc/Yolo11/runs/pose/4.25.2/weights/best.pt",
        # "/home/wyc/Yolo11/runs/pose/4.27.2/weights/best.pt",
        # "/home/wyc/Yolo11/runs/pose/4.28.1/weights/best.pt",
        # "/home/wyc/Yolo11/runs/pose/4.28.2/weights/best.pt",
        # "/home/wyc/Yolo11/runs/pose/4.28.3/weights/best.pt",
        # "/home/wyc/Yolo11/runs/pose/4.29.1/weights/best.pt",
        # "/home/wyc/Yolo11/runs/pose/4.29.2/weights/best.pt",
        # "/home/wyc/Yolo11/runs/pose/4.29.3/weights/best.pt",
        "/home/wyc/Yolo11/runs/pose/4.30/weights/best.pt",
        "/home/wyc/Yolo11/runs/pose/6.27/weights/best.pt",
    ]
    data_yaml = "custom/cfg/datasets.yaml"
    imgsz, batch_size = 640, 32

    # 评估所有模型
    model_dfs = {}
    for mp in model_paths:
        name = os.path.basename(os.path.dirname(os.path.dirname(mp)))
        model_dfs[name] = evaluate_pose_per_class(mp, data_yaml, imgsz, batch_size)

    # 用第一份结果的短标签顺序做 X 轴
    short_labels = model_dfs[next(iter(model_dfs))]["Short"].tolist()

    # 要画的 8 个指标
    metrics = [
        "Box(P)", "Recall(B)", "mAP50(B)", "mAP50-95(B)",
        "Pose(P)", "Recall(P)", "mAP50(P)", "mAP50-95(P)"
    ]

    # 创建 2×4 子图大画布
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for name, df in model_dfs.items():
            ax.plot(short_labels, df[metric], marker='o', label=name)
        ax.set_title(metric)
        ax.set_ylabel(metric)

        # 下面两行：水平显示 x 轴短标签，并且居中对齐
        ax.set_xticks(short_labels)
        ax.set_xticklabels(short_labels, rotation=0, ha='center')

        ax.grid(axis='y')
        if i == 0:
            ax.legend(loc="upper right")
        # 去掉 x 轴标题
        ax.set_xlabel("")

    plt.tight_layout()
    plt.show()

