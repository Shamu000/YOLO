import os
from ultralytics import YOLO


model = YOLO('/home/wyc/Yolo11/runs/pose/4.25.2/weights/best.pt').to("cuda")

metrics = model.val(
    data="custom/cfg/datasets.yaml",
    split="test",
    batch=16,
    imgsz=640,
    verbose=True
)
