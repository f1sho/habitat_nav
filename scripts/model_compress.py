# from ultralytics import YOLO

# model = YOLO("yolo26n-seg.pt")

# model.info()

# ----------------------------------------------------------

from ultralytics import YOLO

model = YOLO("yolo26n-seg.onnx")

results=model("test.jpg")