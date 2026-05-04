import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd

# 你的短标签映射表
category_mapping = {
    0: "Red_Target_Light",
    1: "Red_Target_Off",
    2: "Blue_Target_Light",
    3: "Blue_Target_Off"
}

def evaluate_pose_per_class(model_path, data_yaml, imgsz=640, batch_size=32):
    model = YOLO(model_path).to("cuda")
    metrics = model.val(
        data=data_yaml,
        split="test",
        batch=batch_size,
        imgsz=imgsz,
        conf=0.6,
        iou=0.55,
        max_det=200,
        agnostic_nms=True,
        verbose=False,
        plots=True,  # 启用绘图
        save=True  # 启用结果保存（txt、PR曲线、confusion matrix等）
    )

    names = model.names
    # 测试集中实际出现过的类别索引（有时是 list，有时是 ndarray）
    ap_index = metrics.ap_class_index
    if ap_index is None:
        valid_indices = []
    elif hasattr(ap_index, "tolist"):
        valid_indices = ap_index.tolist()
    else:
        # already a list-like
        valid_indices = list(ap_index)

    rows = []
    for cid in valid_indices:
        # 尝试使用类 ID 获取指标；如果失败则回退到在 valid_indices 中的位置索引
        try:
            bp, br, bm50, bm5095, pp, pr, pm50, pm5095 = metrics.class_result(cid)
        except Exception:
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
        "/home/hithcat/Code/CRT/yolo/runs/pose/train/weights/best.pt",
    ]
    data_yaml = "custom/cfg/datasets.yaml"
    imgsz, batch_size = 640, 32

    # 评估所有模型
    model_dfs = {}
    for mp in model_paths:
        name = os.path.basename(os.path.dirname(os.path.dirname(mp)))
        df = evaluate_pose_per_class(mp, data_yaml, imgsz, batch_size)
        # 如果评估结果为空（例如没有标签），用短标签和 0 值指标填充，避免后续绘图出错
        if df is None or df.empty:
            metrics_fallback = {
                "Box(P)": 2, "Recall(B)": 2, "mAP50(B)": 2, "mAP50-95(B)": 2,
                "Pose(P)": 2, "Recall(P)": 2, "mAP50(P)": 2, "mAP50-95(P)": 2
            }
            rows = []
            for cid in sorted(category_mapping.keys()):
                row = {"ClassID": cid, "Short": category_mapping[cid]}
                row.update(metrics_fallback)
                rows.append(row)
            df = pd.DataFrame(rows)
        model_dfs[name] = df

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

