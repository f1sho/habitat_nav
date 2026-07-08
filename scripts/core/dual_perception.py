import torch
import cv2
import numpy as np
from ultralytics import YOLO

class DualPerceptionModule:
    def __init__(self, yolo_path="yolo26n-seg.pt", conf_threshold=0.5):
        print("Loading YOLO model...")
        self.yolo = YOLO(yolo_path)
        self.conf_threshold = conf_threshold

        print("Loading MiDaS depth model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load a lightweight depth estimation model from PyTorch Hub
        self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.depth_model.to(self.device)
        self.depth_model.eval()
        
        self.depth_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def process_rgb_for_depth(self, rgb_image):
        # Run the depth estimation model
        input_batch = self.depth_transform(rgb_image).to(self.device)

        with torch.no_grad():
            prediction = self.depth_model(input_batch)
            
            # Resize predicted depth map to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb_image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        
        # Use fixed empirical bounds to prevent depth scale jitter across frames
        FIXED_MIN = 10.0
        FIXED_MAX = 500.0
        
        # Clip extreme values to ensure stability
        depth_map_clipped = np.clip(depth_map, FIXED_MIN, FIXED_MAX)
        
        # Normalize to 10 meters and invert 
        normalized_depth = (FIXED_MAX - depth_map_clipped) / (FIXED_MAX - FIXED_MIN)
        metric_depth = normalized_depth * 10.0
        
        # Apply Gaussian blur to smooth the dense depth map for the local planner
        metric_depth = cv2.GaussianBlur(metric_depth, (5, 5), 0)
            
        return metric_depth.astype(np.float32)

    def process_frame(self, rgba_frame):
        # Process frame to get YOLO detections and predicted depth map
        rgb_image = rgba_frame[..., :3]
        
        predicted_depth = self.process_rgb_for_depth(rgb_image)
        
        results = self.yolo(rgb_image, verbose=False)[0]
        detections = []

        # Create a blank canvas using the image dimensions
        # Fill with 255 to represent the background class
        height, width = rgb_image.shape[:2]
        prediction_mask = np.full((height, width), 255, dtype=np.uint8)
        
        if results.boxes is not None and results.masks is not None:
            for box, mask in zip(results.boxes, results.masks):
                conf = float(box.conf[0])
                if conf >= self.conf_threshold:
                    cls_id = int(box.cls[0])
                    class_name = self.yolo.names[cls_id]
                    
                    # Extract the polygon points of the segmentation mask
                    polygon = mask.xy[0]

                    # Fill the corresponding polygon area with the class ID
                    pts = np.array(polygon, dtype=np.int32)
                    cv2.fillPoly(prediction_mask, [pts], cls_id)
                    
                    detections.append({
                        "class_name": class_name,
                        "confidence": conf,
                        "polygon": polygon
                    })
                    
        return detections, predicted_depth, prediction_mask