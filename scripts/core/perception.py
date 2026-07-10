import numpy as np
from ultralytics import YOLO

class PerceptionModule:
    def __init__(self, model_path="yolo26n-seg.pt", camera_height=1.5, focal_length=800, img_height=480):
        self.model = YOLO(model_path)
        self.camera_height = camera_height  # Height of the camera from the ground in meters
        self.focal_length = focal_length    # Focal length of the camera in pixels
        self.img_height = img_height        # Total height of the image in pixels

    def estimate_distance_ipm(self, polygon):
        """
        Calculates distance using Inverse Perspective Mapping based on the lowest point of the segmentation mask.
        Assumes camera pitch is 0.
        """
        if len(polygon) == 0:
            return float('inf')
            
        # Extract the maximum y-coordinate
        v_max = np.max(polygon[:, 1])
        
        # Calculate the optical center y-coordinate
        c_y = self.img_height / 2.0
        
        # If the lowest point is above the horizon, IPM cannot calculate distance
        if v_max <= c_y:
            return float('inf')
            
        # IPM formula: Z = (Camera Height * Focal Length) / (v_max - c_y)
        distance = (self.camera_height * self.focal_length) / (v_max - c_y)
        return distance

    def process_frame(self, rgb_frame):
        results = self.model(rgb_frame)
        detections = []

        for result in results:
            if result.masks is None:
                continue 

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            
            # Extract polygon coordinates scaled to the original image dimensions
            masks_polygons = result.masks.xy 

            for i, polygon in enumerate(masks_polygons):
                # Calculate metric distance using IPM algorithm
                estimated_dist = self.estimate_distance_ipm(polygon)

                # Add the extracted polygon data into the dictionary for downstream visualization
                class_id = int(classes[i])
                class_name = self.model.names[class_id]

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(scores[i]),
                    "bbox": boxes[i],
                    "estimated_distance": estimated_dist,
                    "polygon": polygon
                })

        return detections