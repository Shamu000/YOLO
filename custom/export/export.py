from ultralytics import YOLO


# Load a model
model = YOLO("/home/hithcat/Code/CRT/yolo/runs/pose/FinalTrain/weights/best.pt")  # load a custom trained model

# Export the model
model.export(format="onnx",
             imgsz=640,
             dynamic=False,
             simplify=False,
             opset=12,
             batch=1,
             device='0')








