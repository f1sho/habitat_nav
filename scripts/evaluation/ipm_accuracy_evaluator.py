import csv
import os

import cv2
import numpy as np


class IPMAccuracyEvaluator:
    def __init__(
        self,
        model_name,
        camera_height,
        focal_length,
        image_height,
        save_root="results/ipm_accuracy",
        bottom_band_height=5,
        contact_height_threshold=0.15,
    ):
        self.model_name = self._sanitize_name(model_name)

        self.camera_height = float(camera_height)
        self.focal_length = float(focal_length)
        self.principal_y = float(image_height) / 2.0

        self.bottom_band_height = int(bottom_band_height)
        self.contact_height_threshold = float(
            contact_height_threshold
        )

        self.save_dir = os.path.join(
            save_root,
            self.model_name,
        )

        self.samples = []
        self.total_frames = 0
        self.frames_with_detections = 0
        self.total_detections = 0
        self.skipped_detections = 0

    def update_frame(
        self,
        episode,
        step,
        detections,
        depth_frame,
    ):
        """
        Compare IPM predictions with aligned Habitat depth observations.
        """

        depth_frame = np.asarray(depth_frame)

        if depth_frame.ndim != 2:
            raise ValueError(
                "depth_frame must have shape (H, W), "
                f"but received {depth_frame.shape}"
            )
        
        self.total_frames += 1
        self.total_detections += len(detections)

        if detections:
            self.frames_with_detections += 1

        for detection_index, detection in enumerate(detections):
            sample = self._evaluate_detection(
                episode=episode,
                step=step,
                detection_index=detection_index,
                detection=detection,
                depth_frame=depth_frame,
            )

            if sample is None:
                self.skipped_detections += 1
            else:
                self.samples.append(sample)

    def _evaluate_detection(
        self,
        episode,
        step,
        detection_index,
        detection,
        depth_frame,
    ):
        polygon = detection.get("polygon")
        predicted_distance = detection.get(
            "estimated_distance"
        )

        if polygon is None or predicted_distance is None:
            return None

        polygon = np.asarray(
            polygon,
            dtype=np.float32,
        )

        if (
            polygon.ndim != 2
            or polygon.shape[0] < 3
            or polygon.shape[1] < 2
        ):
            return None

        predicted_distance = float(predicted_distance)

        if (
            not np.isfinite(predicted_distance)
            or predicted_distance <= 0.0
        ):
            return None

        height, width = depth_frame.shape

        polygon_int = np.round(
            polygon[:, :2]
        ).astype(np.int32)

        polygon_int[:, 0] = np.clip(
            polygon_int[:, 0],
            0,
            width - 1,
        )

        polygon_int[:, 1] = np.clip(
            polygon_int[:, 1],
            0,
            height - 1,
        )

        polygon_mask = np.zeros(
            (height, width),
            dtype=np.uint8,
        )

        cv2.fillPoly(
            polygon_mask,
            [polygon_int],
            1,
        )

        bottom_y = int(
            np.max(polygon_int[:, 1])
        )

        top_y = max(
            0,
            bottom_y - self.bottom_band_height + 1,
        )

        bottom_band_mask = np.zeros(
            (height, width),
            dtype=bool,
        )

        bottom_band_mask[
            top_y : bottom_y + 1,
            :
        ] = True

        valid_mask = (
            polygon_mask.astype(bool)
            & bottom_band_mask
            & np.isfinite(depth_frame)
            & (depth_frame > 0.0)
        )

        valid_depths = depth_frame[valid_mask]

        if valid_depths.size == 0:
            return None

        gt_distance = float(
            np.median(valid_depths)
        )

        lowest_index = int(
            np.argmax(polygon_int[:, 1])
        )

        lowest_x = int(
            polygon_int[lowest_index, 0]
        )
        lowest_y = int(
            polygon_int[lowest_index, 1]
        )

        lowest_pixel_depth = float(
            depth_frame[lowest_y, lowest_x]
        )

        if (
            not np.isfinite(lowest_pixel_depth)
            or lowest_pixel_depth <= 0.0
        ):
            lowest_pixel_depth = gt_distance

        vertical_drop = (
            (lowest_y - self.principal_y)
            / self.focal_length
            * lowest_pixel_depth
        )

        lowest_point_height = (
            self.camera_height - vertical_drop
        )

        contact_valid = (
            abs(lowest_point_height)
            <= self.contact_height_threshold
        )

        signed_error = (
            predicted_distance - gt_distance
        )

        absolute_error = abs(signed_error)

        relative_error = (
            absolute_error / gt_distance * 100.0
        )

        return {
            "episode": int(episode),
            "step": int(step),
            "detection_index": int(detection_index),
            "class_id": int(detection["class_id"]),
            "class_name": detection["class_name"],
            "confidence": float(
                detection["confidence"]
            ),
            "predicted_distance_m": predicted_distance,
            "gt_distance_m": gt_distance,
            "signed_error_m": signed_error,
            "absolute_error_m": absolute_error,
            "relative_error_pct": relative_error,
            "bottom_y_px": bottom_y,
            "lowest_x_px": lowest_x,
            "lowest_y_px": lowest_y,
            "lowest_pixel_depth_m": lowest_pixel_depth,
            "lowest_point_height_m": lowest_point_height,
            "contact_valid": bool(contact_valid),
        }

    def save(self):
        os.makedirs(
            self.save_dir,
            exist_ok=True,
        )

        samples_path = os.path.join(
            self.save_dir,
            "ipm_accuracy_samples.csv",
        )

        summary_path = os.path.join(
            self.save_dir,
            "ipm_accuracy_summary.csv",
        )

        episode_summary_path = os.path.join(
            self.save_dir,
            "ipm_accuracy_episode_summary.csv",
        )

        self._save_samples(samples_path)

        summary = self._build_summary()
        episode_summary = self._build_episode_summary()
        self._save_summary(
            summary_path,
            summary,
        )
        self._save_rows(
            episode_summary_path,
            episode_summary,
        )

        print(
            f"[IPM Accuracy] Samples saved to: "
            f"{samples_path}"
        )

        print(
            f"[IPM Accuracy] Summary saved to: "
            f"{summary_path}"
        )

        print(
            f"[IPM Accuracy] Episode summary saved to: "
            f"{episode_summary_path}"
        )

        if summary["total_samples"] > 0:
            print(
                "[IPM Accuracy] "
                f"MAE={summary['all_mae_m']:.3f} m, "
                f"RMSE={summary['all_rmse_m']:.3f} m, "
                f"MAPE={summary['all_mape_pct']:.2f}%"
            )

            print(
                "[IPM Accuracy] "
                f"Contact-valid samples="
                f"{summary['contact_valid_samples']}/"
                f"{summary['total_samples']}"
            )

        return summary

    def _build_summary(self):
        contact_samples = [
            sample
            for sample in self.samples
            if sample["contact_valid"]
        ]

        summary = {
            "model_name": self.model_name,

            "processed_frames": self.total_frames,
            "frames_with_detections": self.frames_with_detections,
            "detection_frame_rate": (
                self.frames_with_detections
                / self.total_frames
                if self.total_frames > 0
                else 0.0
            ),

            "total_detections": self.total_detections,
            "total_samples": len(self.samples),
            "skipped_detections": self.skipped_detections,
            "valid_sample_rate": (
                len(self.samples)
                / self.total_detections
                if self.total_detections > 0
                else 0.0
            ),

            "contact_valid_samples": len(contact_samples),
            "contact_valid_rate": (
                len(contact_samples) / len(self.samples)
                if self.samples
                else 0.0
            ),
        }

        summary.update(
            self._summarize_samples(
                self.samples,
                prefix="all",
            )
        )

        summary.update(
            self._summarize_samples(
                contact_samples,
                prefix="contact_valid",
            )
        )

        return summary
    
    def _build_episode_summary(self):
        """
        Build one IPM accuracy summary row for each episode.
        """

        episode_ids = sorted(
            {
                sample["episode"]
                for sample in self.samples
            }
        )

        rows = []

        for episode_id in episode_ids:
            episode_samples = [
                sample
                for sample in self.samples
                if sample["episode"] == episode_id
            ]

            contact_samples = [
                sample
                for sample in episode_samples
                if sample["contact_valid"]
            ]

            row = {
                "model_name": self.model_name,
                "episode": episode_id,
                "total_samples": len(episode_samples),
                "contact_valid_samples": len(
                    contact_samples
                ),
                "contact_valid_rate": (
                    len(contact_samples)
                    / len(episode_samples)
                    if episode_samples
                    else 0.0
                ),
            }

            row.update(
                self._summarize_samples(
                    episode_samples,
                    prefix="all",
                )
            )

            row.update(
                self._summarize_samples(
                    contact_samples,
                    prefix="contact_valid",
                )
            )

            rows.append(row)

        return rows

    @staticmethod
    def _summarize_samples(
        samples,
        prefix,
    ):
        if not samples:
            return {
                f"{prefix}_mae_m": "",
                f"{prefix}_rmse_m": "",
                f"{prefix}_mape_pct": "",
                f"{prefix}_mean_signed_error_m": "",
            }

        signed_errors = np.asarray(
            [
                sample["signed_error_m"]
                for sample in samples
            ],
            dtype=np.float64,
        )

        absolute_errors = np.asarray(
            [
                sample["absolute_error_m"]
                for sample in samples
            ],
            dtype=np.float64,
        )

        relative_errors = np.asarray(
            [
                sample["relative_error_pct"]
                for sample in samples
            ],
            dtype=np.float64,
        )

        return {
            f"{prefix}_mae_m": float(
                np.mean(absolute_errors)
            ),
            f"{prefix}_rmse_m": float(
                np.sqrt(
                    np.mean(
                        np.square(signed_errors)
                    )
                )
            ),
            f"{prefix}_mape_pct": float(
                np.mean(relative_errors)
            ),
            f"{prefix}_mean_signed_error_m": float(
                np.mean(signed_errors)
            ),
        }

    def _save_samples(self, filepath):
        if not self.samples:
            return

        with open(
            filepath,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=self.samples[0].keys(),
            )

            writer.writeheader()
            writer.writerows(self.samples)

    @staticmethod
    def _save_summary(
        filepath,
        summary,
    ):
        with open(
            filepath,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=summary.keys(),
            )

            writer.writeheader()
            writer.writerow(summary)

    @staticmethod
    def _save_rows(
        filepath,
        rows,
    ):
        """
        Save multiple dictionaries as CSV rows.
        """

        if not rows:
            return

        with open(
            filepath,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=rows[0].keys(),
            )

            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _sanitize_name(name):
        name = os.path.basename(name)
        name = name.replace(".", "_")
        name = name.replace("-", "_")
        return name