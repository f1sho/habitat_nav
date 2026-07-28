import cv2
import numpy as np


class DemoVisualizer:
    """
    Visualize navigation, sensitive-object detections, IPM distance,
    safety distance, and Habitat depth reference.
    """

    SENSITIVE_OBJECTS = {
        "chair",
        "potted plant",
        "tv",
        "bed",
        "sofa",
        "vase",
    }

    def __init__(
        self,
        window_name="Semantic Navigation Demo",
        save_video=True,
        video_path="demo_output.mp4",
        video_fps=8.0,
    ):
        self.window_name = window_name
        self.save_video = save_video
        self.video_path = video_path
        self.video_fps = video_fps
        self.video_writer = None

    @staticmethod
    def _to_float(value):
        """Convert a value to a valid positive distance."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        if not np.isfinite(value) or value <= 0.0:
            return None

        return value

    @classmethod
    def _format_distance(cls, value):
        """Format a distance for display."""
        value = cls._to_float(value)

        if value is None:
            return "N/A"

        return f"{value:.2f}m"

    @classmethod
    def _get_safety_status(
        cls,
        ipm_distance,
        safety_distance,
    ):
        """
        Determine safety status using the IPM distance because IPM is the
        distance information available to the navigation system.
        """
        ipm_distance = cls._to_float(ipm_distance)

        if ipm_distance is None:
            return "UNKNOWN"

        if ipm_distance < safety_distance:
            return "TOO CLOSE"

        return "CLEAR"

    @staticmethod
    def _get_depth_gt(
        polygon,
        depth_frame,
        bottom_band_height=5,
    ):
        """
        Calculate the Habitat depth reference from the bottom region of the
        segmentation polygon.

        The depth reference is used only for visualization and evaluation.
        """
        if polygon is None or depth_frame is None:
            return None

        polygon = np.asarray(
            polygon,
            dtype=np.float32,
        )

        depth_frame = np.asarray(
            depth_frame,
            dtype=np.float32,
        )

        depth_frame = np.squeeze(depth_frame)

        if (
            polygon.ndim != 2
            or polygon.shape[0] < 3
            or polygon.shape[1] < 2
            or depth_frame.ndim != 2
        ):
            return None

        frame_height, frame_width = depth_frame.shape

        polygon_int = np.round(
            polygon[:, :2]
        ).astype(np.int32)

        polygon_int[:, 0] = np.clip(
            polygon_int[:, 0],
            0,
            frame_width - 1,
        )

        polygon_int[:, 1] = np.clip(
            polygon_int[:, 1],
            0,
            frame_height - 1,
        )

        polygon_mask = np.zeros(
            (frame_height, frame_width),
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
            bottom_y - bottom_band_height + 1,
        )

        bottom_band_mask = np.zeros(
            (frame_height, frame_width),
            dtype=bool,
        )

        bottom_band_mask[
            top_y:bottom_y + 1,
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

        return float(
            np.median(valid_depths)
        )

    @staticmethod
    def _draw_text(
        frame,
        text,
        position,
        color,
        font_scale=0.52,
        thickness=2,
    ):
        """Draw readable text with a black outline."""
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def show_frame(
        self,
        rgb_frame,
        detections,
        action,
        step,
        dist_to_wp,
        depth_gt_frame=None,
        current_safe_distance=0.4,
    ):
        """
        Display and optionally save one navigation frame.
        """
        if rgb_frame is None:
            return

        detections = detections or []

        if (
            rgb_frame.ndim == 3
            and rgb_frame.shape[2] == 4
        ):
            bgr_frame = cv2.cvtColor(
                rgb_frame,
                cv2.COLOR_RGBA2BGR,
            )
        else:
            bgr_frame = cv2.cvtColor(
                rgb_frame,
                cv2.COLOR_RGB2BGR,
            )

        frame_height, frame_width = (
            bgr_frame.shape[:2]
        )

        sensitive_rows = []

        for detection in detections:
            class_name = str(
                detection.get(
                    "class_name",
                    "unknown",
                )
            )

            polygon = detection.get(
                "polygon"
            )

            if polygon is None:
                continue

            polygon = np.asarray(
                polygon,
                dtype=np.float32,
            )

            if (
                polygon.ndim != 2
                or polygon.shape[0] < 3
                or polygon.shape[1] < 2
            ):
                continue

            polygon_int = np.round(
                polygon[:, :2]
            ).astype(np.int32)

            polygon_int[:, 0] = np.clip(
                polygon_int[:, 0],
                0,
                frame_width - 1,
            )

            polygon_int[:, 1] = np.clip(
                polygon_int[:, 1],
                0,
                frame_height - 1,
            )

            confidence = detection.get(
                "confidence",
                0.0,
            )

            ipm_distance = detection.get(
                "estimated_distance"
            )

            is_sensitive = (
                class_name.lower()
                in self.SENSITIVE_OBJECTS
            )

            if is_sensitive:
                outline_color = (
                    0,
                    0,
                    255,
                )
                outline_thickness = 3
            else:
                outline_color = (
                    0,
                    255,
                    0,
                )
                outline_thickness = 2

            cv2.polylines(
                bgr_frame,
                [polygon_int],
                isClosed=True,
                color=outline_color,
                thickness=outline_thickness,
                lineType=cv2.LINE_AA,
            )

            label_x = int(
                np.min(polygon_int[:, 0])
            )

            label_y = int(
                np.min(polygon_int[:, 1])
            ) - 10

            label_x = max(
                8,
                min(label_x, frame_width - 210),
            )

            label_y = max(
                20,
                label_y,
            )

            if is_sensitive:
                object_label = (
                    f"SENSITIVE: {class_name}"
                )

                depth_gt = self._get_depth_gt(
                    polygon=polygon,
                    depth_frame=depth_gt_frame,
                )

                status = self._get_safety_status(
                    ipm_distance=ipm_distance,
                    safety_distance=current_safe_distance,
                )

                sensitive_rows.append(
                    {
                        "name": class_name,
                        "ipm": (
                            self._format_distance(
                                ipm_distance
                            )
                        ),
                        "gt": (
                            self._format_distance(
                                depth_gt
                            )
                        ),
                        "status": status,
                    }
                )
            else:
                try:
                    confidence = float(
                        confidence
                    )
                except (TypeError, ValueError):
                    confidence = 0.0

                object_label = (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )

            self._draw_text(
                frame=bgr_frame,
                text=object_label,
                position=(
                    label_x,
                    label_y,
                ),
                color=outline_color,
                font_scale=0.5,
                thickness=2,
            )

        displayed_rows = sensitive_rows[:4]

        try:
            waypoint_distance_text = (
                f"{float(dist_to_wp):.2f}m"
            )
        except (TypeError, ValueError):
            waypoint_distance_text = "N/A"

        hud_lines = [
            (
                f"Step: {step} | "
                f"Action: {action}"
            ),
            (
                "Distance to waypoint: "
                f"{waypoint_distance_text}"
            ),
            (
                "Current safe distance: "
                f"{current_safe_distance:.2f}m"
            ),
        ]

        hud_statuses = [
            None,
            None,
            None,
        ]

        if displayed_rows:
            for row in displayed_rows:
                hud_lines.append(
                    f"{row['name']} | "
                    f"IPM: {row['ipm']} | "
                    f"Depth GT: {row['gt']} | "
                    f"{row['status']}"
                )

                hud_statuses.append(
                    row["status"]
                )
        else:
            hud_lines.append(
                "Sensitive objects: none detected"
            )

            hud_statuses.append(None)

        line_height = 27
        top_padding = 13
        bottom_padding = 12
        first_baseline = 27

        hud_height = (
            top_padding
            + bottom_padding
            + line_height * len(hud_lines)
        )

        hud_height = min(
            hud_height,
            frame_height - 10,
        )

        overlay = bgr_frame.copy()

        cv2.rectangle(
            overlay,
            (5, 5),
            (
                frame_width - 5,
                5 + hud_height,
            ),
            (45, 45, 45),
            -1,
        )

        cv2.addWeighted(
            overlay,
            0.75,
            bgr_frame,
            0.25,
            0,
            bgr_frame,
        )

        for line_index, line in enumerate(
            hud_lines
        ):
            text_y = (
                5
                + top_padding
                + first_baseline
                + line_index * line_height
            )

            status = hud_statuses[
                line_index
            ]

            if line_index == 0:
                text_color = (
                    0,
                    255,
                    255,
                )
            elif status == "TOO CLOSE":
                text_color = (
                    0,
                    0,
                    255,
                )
            elif status == "CLEAR":
                text_color = (
                    0,
                    255,
                    0,
                )
            elif status == "UNKNOWN":
                text_color = (
                    0,
                    165,
                    255,
                )
            else:
                text_color = (
                    255,
                    255,
                    255,
                )

            self._draw_text(
                frame=bgr_frame,
                text=line,
                position=(
                    16,
                    text_y,
                ),
                color=text_color,
                font_scale=0.52,
                thickness=1,
            )

        if self.save_video:
            if self.video_writer is None:
                fourcc = (
                    cv2.VideoWriter_fourcc(
                        *"mp4v"
                    )
                )

                self.video_writer = (
                    cv2.VideoWriter(
                        self.video_path,
                        fourcc,
                        self.video_fps,
                        (
                            frame_width,
                            frame_height,
                        ),
                    )
                )

                if not self.video_writer.isOpened():
                    print(
                        "Warning: Unable to open "
                        f"video writer: {self.video_path}"
                    )
                    self.video_writer = None

            if self.video_writer is not None:
                self.video_writer.write(
                    bgr_frame
                )

        try:
            cv2.imshow(
                self.window_name,
                bgr_frame,
            )

            cv2.waitKey(1)
        except cv2.error:
            pass

    def close(self):
        """Release video and display resources."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

            print(
                f"Demo video saved to: "
                f"{self.video_path}"
            )

        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass