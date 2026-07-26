Model: YOLO26n-seg
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
TensorRT: 10.16.1.11
Ultralytics: 8.4.47
Calibration images: 320
Input: fixed batch 1, 640x640
Attempts:
- workspace=2, simplify=True: failed
- workspace=4, simplify=True: failed
- workspace=4, simplify=False: failed

Fatal node:
/model.23/proto/cv3/conv/Conv + Sigmoid + Mul

Conclusion:
Standard TensorRT INT8 PTQ engine could not be built for this model and software/hardware stack.
