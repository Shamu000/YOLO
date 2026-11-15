import os
import sys
# 确保项目本地 ultralytics 优先于 pip 安装版本
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ultralytics import YOLO
import matplotlib
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
import matplotlib.pyplot as plt
import pandas as pd
import yaml

# ===== 开关：是否保存所有 val batch 的预测图和标签图 =====
# 默认 plots=True 只保存前 6 个 batch（且每 batch 最多展示 16 张图）到 runs/pose/val{N}/
# 开启后保存全部 batch 至 runs/pose/allBatch/val{N}/，且每 batch 展示 batch_size 张图
SAVE_ALL_BATCHES = True  # 改为 False 关闭

if SAVE_ALL_BATCHES:
    import torch
    from ultralytics.models.yolo.pose.val import PoseValidator
    from ultralytics.utils.plotting import plot_images, output_to_target

    class _SaveAllBatchesPoseValidator(PoseValidator):
        """扩展 PoseValidator：保存所有 batch 图像到 runs/pose/allBatch/"""

        # ---- 重写绘图方法，增加 max_subplots 以展示 batch 中全部图像 ----
        def plot_val_samples(self, batch, ni):
            plot_images(
                batch["img"],
                batch["batch_idx"],
                batch["cls"].squeeze(-1),
                batch["bboxes"],
                kpts=batch["keypoints"],
                paths=batch["im_file"],
                fname=self.save_dir / f"val_batch{ni}_labels.jpg",
                names=self.names,
                on_plot=self.on_plot,
                max_subplots=self.args.batch,  # 展示 batch 中全部图像
            )

        def plot_predictions(self, batch, preds, ni):
            pred_kpts = torch.cat([p[:, 6:].view(-1, *self.kpt_shape) for p in preds], 0) if preds else torch.zeros((0, self.kpt_shape[0], self.kpt_shape[1]))
            plot_images(
                batch["img"],
                *output_to_target(preds, max_det=self.args.max_det),
                kpts=pred_kpts,
                paths=batch["im_file"],
                fname=self.save_dir / f"val_batch{ni}_pred.jpg",
                names=self.names,
                on_plot=self.on_plot,
                max_subplots=self.args.batch,  # 展示 batch 中全部图像
            )

        # ---- 重写 update_metrics：每个 batch 更新完指标后无条件保存图像 ----
        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            if self.args.plots:
                self.plot_val_samples(batch, self.batch_i)
                self.plot_predictions(batch, preds, self.batch_i)


def load_category_names(yaml_path="custom/cfg/datasets.yaml"):
    """从 datasets.yaml 读取类别名映射"""
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('names', {})


category_mapping = load_category_names()

def evaluate_pose_per_class(model_path, data_yaml, imgsz=640, batch_size=32):
    model = YOLO(model_path).to("cuda")

    # 公共参数
    val_kwargs = dict(
        data=data_yaml,
        split="test",
        batch=batch_size,
        imgsz=imgsz,
        conf=0.1,
        iou=0.55,
        max_det=200,
        agnostic_nms=True,
        verbose=False,
        plots=True,  # 启用绘图
        save=True,  # 启用结果保存（txt、PR曲线、confusion matrix等）
    )

    if SAVE_ALL_BATCHES:
        # 将所有 val 输出（含全部 batch 图像）保存到 runs/pose/allBatch/val{N}/
        val_kwargs["project"] = "runs/pose/allBatch"
        metrics = model.val(validator=_SaveAllBatchesPoseValidator, **val_kwargs)
    else:
        metrics = model.val(**val_kwargs)

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
        "/home/hithcat/Code/CRT/yolo/runs/pose/v2-50%数据增强-401代/weights/best.pt",
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

