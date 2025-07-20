import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from ultralytics import YOLO


model = YOLO('yolo11n-pose.pt').to("cuda")

model.train(
    # ------------------------ 训练任务配置 ------------------------
    task='pose',                     # 任务类型，关键点检测为 'pose'
    mode='train',                   # 模式，训练用 'train'
    data='../cfg/datasets.yaml',    # 数据配置文件路径
    epochs=300,                     # 训练总轮数
    patience=30,                   # 早停等待轮数
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

    # ------------------------ 分类（Classification）专用参数 ------------------------
    dropout=0.0,                  # dropout 概率（只用于分类任务）

    # ------------------------ 超参数（Hyperparameters） ------------------------
    lr0=0.01,                     # 初始学习率
    lrf=0.01,                     # 最终学习率 = lr0 * lrf
    momentum=0.937,              # SGD 动量 / Adam β1
    weight_decay=0.0005,         # 权重衰减
    warmup_epochs=3.0,           # 学习率预热轮数
    warmup_momentum=0.8,         # 预热初始动量
    warmup_bias_lr=0.1,          # 预热阶段 bias 的学习率

    box=0.5,                     # 边界框损失权重
    cls=7.5,                     # 分类损失权重
    dfl=1.5,                     # DFL 损失权重
    pose=12.0,                   # 关键点损失权重（关键点任务专用）
    kobj=1.0,                    # 关键点对象损失权重

    nbs=64,                      # 标准 batch size（用于学习率调整）
    hsv_h=0.015,                 # HSV 色调增强
    hsv_s=0.3,                   # HSV 饱和度增强
    hsv_v=0.6,                   # HSV 亮度增强
    degrees=1.0,                 # 图像旋转增强（±角度）
    translate=0.1,               # 平移增强（±比例）
    scale=0.5,                   # 缩放增强（±比例）
    shear=0.0,                   # 剪切增强（±角度）
    perspective=0.001,             # 透视变换增强（比例）
    flipud=0.0,                  # 上下翻转概率
    fliplr=0.0,                  # 左右翻转概率
    bgr=0.0,                     # BGR 通道变换概率

    mosaic=1.0,                  # mosaic 数据增强概率
    mixup=0.0,                   # mixup 数据增强概率
    copy_paste=0.0,              # segmentation copy-paste 增强概率
    copy_paste_mode='flip',      # copy_paste 增强方法（flip 或 mixup）

    auto_augment='None',  # 分类任务的自动增强策略
    erasing=0.4                  # 分类任务随机擦除概率
)


#