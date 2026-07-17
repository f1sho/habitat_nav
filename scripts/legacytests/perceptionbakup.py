import time
from typing import cast
import torch
import numpy as np
from ultralytics import YOLO


class PerceptionModule:
    def __init__(
        self,
        model_path="yolo26n-seg.pt",
        camera_height=1.5,
        focal_length=800,
        img_height=480,
        device=0,
    ):
        self.model = YOLO(model_path)

        self.camera_height = camera_height
        self.focal_length = focal_length
        self.img_height = img_height
        self.device = device

    def estimate_distance_ipm(self, polygon):
        """
        Estimate object distance using Inverse Perspective Mapping (IPM)
        based on the lowest point of the segmentation polygon.
        """
        polygon = np.asarray(polygon)

        if polygon.size == 0:
            return float("inf")

        v_max = float(np.max(polygon[:, 1]))
        c_y = self.img_height / 2.0

        if v_max <= c_y:
            return float("inf")

        distance = (
            self.camera_height * self.focal_length
        ) / (v_max - c_y)

        return float(distance)

    def process_frame(self, rgb_frame):
        """
        Process one RGB frame.

        latency_ms measures the complete perception pipeline:
        model preprocessing + inference + postprocessing + polygon parsing
        + IPM distance estimation.

        Returns
        -------
        detections : list
            Detection results used by the local planner.

        metrics : dict
            Per-frame metrics used by the Evaluation Framework.
        """
        pipeline_start = time.perf_counter()

        # Disable Ultralytics per-frame console output during evaluation.
        results = self.model.predict(
            source=rgb_frame,
            verbose=False,
            device=self.device,
        )

        detections = []
        confidence_sum = 0.0

        for result in results:
            if result.masks is None or result.boxes is None:
                continue

            boxes_tensor = cast(torch.Tensor, result.boxes.xyxy)
            classes_tensor = cast(torch.Tensor, result.boxes.cls)
            scores_tensor = cast(torch.Tensor, result.boxes.conf)

            boxes = boxes_tensor.detach().cpu().numpy()
            classes = classes_tensor.detach().cpu().numpy()
            scores = scores_tensor.detach().cpu().numpy()
            masks_polygons = result.masks.xy

            detection_number = min(
                len(masks_polygons),
                len(boxes),
                len(classes),
                len(scores),
            )

            for index in range(detection_number):
                polygon = np.asarray(
                    masks_polygons[index],
                    dtype=np.float32,
                )

                estimated_distance = self.estimate_distance_ipm(
                    polygon
                )

                class_id = int(classes[index])
                class_name = self.model.names[class_id]
                confidence = float(scores[index])

                confidence_sum += confidence

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": boxes[index],
                        "estimated_distance": estimated_distance,
                        "polygon": polygon,
                    }
                )

        latency_ms = (
            time.perf_counter() - pipeline_start
        ) * 1000.0

        detection_count = len(detections)

        avg_confidence = (
            confidence_sum / detection_count
            if detection_count > 0
            else 0.0
        )

        # Keep full precision in frame_metrics.csv.
        # Rounding is performed only in the summary.
        metrics = {
            "latency_ms": float(latency_ms),
            "fps": (
                float(1000.0 / latency_ms)
                if latency_ms > 0.0
                else 0.0
            ),
            "detections": int(detection_count),
            "avg_confidence": float(avg_confidence),
        }

        return detections, metrics
