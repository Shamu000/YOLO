import os
from ultralytics import YOLO


model = YOLO('/home/wyc/Yolo11/runs/pose/2.22.2/weights/best.pt').to("cuda")

model.val(
    # ------------------------ 训练任务配置 ------------------------
    task='pose',                     # 任务类型，关键点检测为 'pose'
    mode='val',                   # 模式，训练用 'train'
    data='custom/cfg/test.yaml',# 数据配置文件路径
    batch=-1,                       # 每批次图像数量，-1 表示自动调整
    imgsz=640,                      # 输入图像尺寸（可设为 int 或 list）
    device='0',                     # 指定训练设备（如 'cpu', '0', '0,1'）
    workers=8,                      # DataLoader 使用的线程数
)


#