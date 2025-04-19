import os
from ultralytics import YOLO


model = YOLO('yolo11n-pose.pt')

model.train(
    # ------------------------ 训练任务配置 ------------------------
    task='pose',                     # 任务类型，关键点检测为 'pose'
    mode='train',                   # 模式，训练用 'train'
    data='custom/cfg/datasets.yaml',# 数据配置文件路径
    epochs=100,                     # 训练总轮数
    time=None,                      # 最大训练时长（小时），设置后覆盖 epochs
    patience=100,                   # 早停等待轮数
    batch=-1,                       # 每批次图像数量，-1 表示自动调整
    imgsz=640,                      # 输入图像尺寸（可设为 int 或 list）

    save=True,                      # 是否保存训练检查点
    save_period=-1,                # 每多少 epoch 保存一次模型（-1 表示只保存最后）
    cache=False,                   # 数据是否使用缓存（False, 'ram', 'disk'）
    device='0',                     # 指定训练设备（如 'cpu', '0', '0,1'）
    workers=8,                      # DataLoader 使用的线程数
    exist_ok=False,                # 若路径存在是否允许覆盖

    pretrained=True,               # 是否加载预训练模型（True / 模型路径）
    optimizer='auto',              # 优化器类型（'SGD', 'Adam', 等）
    verbose=True,                  # 是否输出详细训练信息
    seed=0,                         # 随机种子
    deterministic=True,           # 是否使用确定性训练（保证复现性）
    single_cls=False,             # 是否将多类别作为单类别处理
    rect=False,                   # 是否使用矩形训练（加快训练）
    cos_lr=False,                 # 是否使用余弦学习率调度器
    close_mosaic=10,              # 关闭 mosaic 增强的 epoch 数（后期 fine-tune）
    resume=False,                 # 是否从上一次训练断点继续训练
    amp=True,                     # 是否启用混合精度训练
    fraction=1.0,                 # 使用数据集的比例
    profile=False,                # 是否 profile ONNX/TensorRT 推理速度
    freeze=None,                  # 冻结前 n 层或指定层索引列表
    multi_scale=False,           # 是否启用多尺度训练

    # ------------------------ 分割（Segmentation）专用参数 ------------------------
    overlap_mask=True,            # 是否合并 instance 的 mask（只用于 segmentation）
    mask_ratio=4,                 # mask 的下采样倍数

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

    box=7.5,                     # 边界框损失权重
    cls=0.5,                     # 分类损失权重
    dfl=1.5,                     # DFL 损失权重
    pose=12.0,                   # 关键点损失权重（关键点任务专用）
    kobj=1.0,                    # 关键点对象损失权重

    nbs=64,                      # 标准 batch size（用于学习率调整）
    hsv_h=0.015,                 # HSV 色调增强
    hsv_s=0.7,                   # HSV 饱和度增强
    hsv_v=0.4,                   # HSV 亮度增强
    degrees=0.0,                 # 图像旋转增强（±角度）
    translate=0.1,               # 平移增强（±比例）
    scale=0.5,                   # 缩放增强（±比例）
    shear=0.0,                   # 剪切增强（±角度）
    perspective=0.0,             # 透视变换增强（比例）
    flipud=0.0,                  # 上下翻转概率
    fliplr=0.5,                  # 左右翻转概率
    bgr=0.0,                     # BGR 通道变换概率

    mosaic=1.0,                  # mosaic 数据增强概率
    mixup=0.0,                   # mixup 数据增强概率
    copy_paste=0.0,              # segmentation copy-paste 增强概率
    copy_paste_mode='flip',      # copy_paste 增强方法（flip 或 mixup）

    auto_augment='randaugment',  # 分类任务的自动增强策略
    erasing=0.4                  # 分类任务随机擦除概率
)


