import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ultralytics import YOLO


model = YOLO('/home/hithcat/Code/tools/models/yolo11n-pose.pt').to("cuda")
model.train(
    # ------------------------ 训练任务配置 ------------------------
    task='pose',                     # 任务类型，关键点检测为 'pose'
    mode='train',                   # 模式，训练用 'train'
    data='custom/cfg/datasets.yaml',    # 数据配置文件路径
    epochs=60,                     # 训练总轮数
    patience=10,                   # 早停等待轮数
    batch=-1,                       # 每批次图像数量，-1 表示自动调整
    imgsz=640,                      # 输入图像尺寸（可设为 int 或 list）
    device='0',                     # 指定训练设备（如 'cpu', '0', '0,1'）
    workers=8,                      # DataLoader 使用的线程数
    pretrained=True,               # 是否加载预训练模型（True / 模型路径）
    optimizer='auto',              # 优化器类型（'SGD', 'Adam', 等）
    cos_lr=True,                 # 是否使用余弦学习率调度器
    close_mosaic=10,              # 关闭 mosaic 增强的 epoch 数（后期 fine-tune）
    resume=False,                 # 是否从上一次训练断点继续训练
    fraction=1.0,                 # 使用数据集的比例
    freeze=None,                  # 冻结前 n 层或指定层索引列表
    multi_scale=False,           # 是否启用多尺度训练
)
