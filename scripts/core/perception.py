import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from ultralytics import YOLO


class PerceptionModule:
    """Run YOLO segmentation and estimate object distance with IPM."""

    SUPPORTED_MODEL_SUFFIXES = {".pt", ".onnx", ".engine"}

    def __init__(
        self,
        model_path="yolo26n-seg.pt",
        camera_height=1.5,
        focal_length=320.0,
        img_height=480,
        device=0,
        imgsz=640,
        confidence_threshold=0.25,
    ):
        self.model_path = Path(model_path).expanduser()

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path.resolve()}"
            )

        model_suffix = self.model_path.suffix.lower()
        if model_suffix not in self.SUPPORTED_MODEL_SUFFIXES:
            raise ValueError(
                "Unsupported model format "
                f"'{model_suffix}'. Supported formats are: "
                f"{sorted(self.SUPPORTED_MODEL_SUFFIXES)}"
            )

        # Explicitly set the task so TensorRT and ONNX models expose
        # segmentation masks and polygons through the same Results interface.
        self.model = YOLO(
            str(self.model_path),
            task="segment",
        )

        self.backend = self._infer_backend(model_suffix)
        self.camera_height = float(camera_height)
        self.focal_length = float(focal_length)
        self.img_height = int(img_height)
        self.device = device
        self.imgsz = int(imgsz)
        self.confidence_threshold = float(confidence_threshold)

        if self.camera_height <= 0.0:
            raise ValueError("camera_height must be greater than zero.")
        if self.focal_length <= 0.0:
            raise ValueError("focal_length must be greater than zero.")
        if self.img_height <= 0:
            raise ValueError("img_height must be greater than zero.")
        if self.imgsz <= 0:
            raise ValueError("imgsz must be greater than zero.")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold must be between 0 and 1."
            )

    @staticmethod
    def _infer_backend(model_suffix):
        """Return a stable backend label from the model file suffix."""
        backend_by_suffix = {
            ".pt": "pytorch",
            ".onnx": "onnx",
            ".engine": "tensorrt",
        }
        return backend_by_suffix[model_suffix]

    def _uses_cuda(self):
        """Return whether the configured inference device is CUDA."""
        if not torch.cuda.is_available():
            return False

        if isinstance(self.device, int):
            return self.device >= 0

        device_name = str(self.device).strip().lower()
        if device_name in {"cpu", "mps"}:
            return False

        return (
            device_name.isdigit()
            or device_name == "cuda"
            or device_name.startswith("cuda:")
        )

    def _synchronize_cuda(self):
        """Synchronize CUDA so latency includes completed GPU execution."""
        if self._uses_cuda():
            torch.cuda.synchronize()

    def _predict(self, rgb_frame):
        """Run one prediction with identical settings across backends."""
        return self.model.predict(
            source=rgb_frame,
            imgsz=self.imgsz,
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device,
        )

    def warmup(self, rgb_frame, iterations=5):
        """
        Warm up the selected inference backend.

        Warm-up predictions are not returned and should not be logged as
        evaluation frames.
        """
        if iterations < 0:
            raise ValueError("iterations must be zero or greater.")

        for _ in range(iterations):
            self._predict(rgb_frame)

        self._synchronize_cuda()

    def estimate_distance_ipm(self, polygon):
        """
        Estimate object distance using Inverse Perspective Mapping (IPM)
        based on the lowest point of the segmentation polygon.
        """
        polygon = np.asarray(polygon, dtype=np.float32)

        if (
            polygon.size == 0
            or polygon.ndim != 2
            or polygon.shape[1] < 2
        ):
            return float("inf")

        v_max = float(np.max(polygon[:, 1]))
        c_y = self.img_height / 2.0

        if not np.isfinite(v_max) or v_max <= c_y:
            return float("inf")

        distance = (
            self.camera_height * self.focal_length
        ) / (v_max - c_y)

        return float(distance)

    def _get_class_name(self, class_id):
        """Return a class name for list- or dict-based model metadata."""
        names = self.model.names

        if isinstance(names, dict):
            return str(names.get(class_id, class_id))

        if 0 <= class_id < len(names):
            return str(names[class_id])

        return str(class_id)

    def process_frame(self, rgb_frame):
        """
        Process one RGB frame.

        latency_ms measures the complete perception pipeline:
        model preprocessing + inference + postprocessing + polygon parsing
        + IPM distance estimation.

        CUDA synchronization is applied before and after the measured region
        so PT, ONNX CUDA, and TensorRT latency values are comparable.

        Returns
        -------
        detections : list
            Detection results used by the local planner.

        metrics : dict
            Per-frame metrics used by the Evaluation Framework.
        """
        if rgb_frame is None:
            raise ValueError("rgb_frame must not be None.")

        self._synchronize_cuda()
        pipeline_start = time.perf_counter()

        results = self._predict(rgb_frame)

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
                class_name = self._get_class_name(class_id)
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

        self._synchronize_cuda()
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
