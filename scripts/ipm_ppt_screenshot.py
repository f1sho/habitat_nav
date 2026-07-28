import math
import time
from pathlib import Path

import cv2
import numpy as np

from core.habitat_env import HabitatEnv
from core.perception import PerceptionModule
from core.planning.global_planner import GlobalPlanner
from core.planning.local_planner import DiscreteDWAPlanner
from evaluation.evaluator import Evaluator
from evaluation.model_metrics import ModelMetrics
from evaluation.navigation_metrics import NavigationMetrics
from run_navigation import _build_ipm_distance_map
from utils.trajectory_plotter import plot_topdown_trajectory
from utils.visualizer import DemoVisualizer


MODEL_PATH = "yolo26n-seg.onnx"
SCENE_PATH = (
    "/home/hannah/data/replica_v1/apartment_2/"
    "habitat/mesh_semantic.ply"
)
NAVMESH_PATH = (
    "/home/hannah/data/replica_v1/apartment_2/"
    "habitat/mesh_semantic.navmesh"
)

PPT_SCREENSHOT_PATH = "results/ppt/ppt_navigation_frame.png"
PPT_MASK_ALPHA = 0.32
PPT_BOTTOM_BAND_HEIGHT = 5
PPT_MIN_CONFIDENCE = 0.25
PPT_MIN_MASK_AREA_RATIO = 0.01

SENSITIVE_OBJECTS = {
    "chair",
    "potted plant",
    "tv",
    "bed",
    "sofa",
    "vase",
}


class PresentationFrameCollector:
    """Select and save one presentation-ready navigation frame."""

    def __init__(
        self,
        output_path,
        sensitive_objects,
        general_safe_distance,
        semantic_safe_distance,
        bottom_band_height=5,
        mask_alpha=0.32,
        min_confidence=0.25,
        min_mask_area_ratio=0.01,
    ):
        """Configure the one-file PPT screenshot collector."""
        self.output_path = Path(output_path)
        self.sensitive_objects = set(sensitive_objects)
        self.general_safe_distance = float(general_safe_distance)
        self.semantic_safe_distance = float(semantic_safe_distance)
        self.bottom_band_height = max(1, int(bottom_band_height))
        self.mask_alpha = float(np.clip(mask_alpha, 0.0, 1.0))
        self.min_confidence = float(min_confidence)
        self.min_mask_area_ratio = float(min_mask_area_ratio)

        self.best_score = float("-inf")
        self.best_step = None
        self.best_class_name = None
        self.has_saved = False

    def update(self, rgb_frame, detections, depth_gt_frame, step, action):
        """Update the saved PNG when the current frame is a better candidate."""
        rgb_frame = np.asarray(rgb_frame)
        depth_gt_frame = np.asarray(depth_gt_frame)

        if rgb_frame.ndim != 3 or rgb_frame.shape[2] < 3:
            raise ValueError(
                "rgb_frame must have shape (H, W, 3) or (H, W, 4)."
            )
        if depth_gt_frame.ndim != 2:
            raise ValueError("depth_gt_frame must have shape (H, W).")
        if rgb_frame.shape[:2] != depth_gt_frame.shape:
            raise ValueError(
                "RGB and depth GT frames must have identical resolution."
            )

        prepared_detections = self._prepare_detections(
            detections=detections,
            depth_gt_frame=depth_gt_frame,
        )
        selected_index, selected_score = self._select_candidate(
            prepared_detections=prepared_detections,
            image_shape=rgb_frame.shape[:2],
        )

        if selected_index is None or selected_score <= self.best_score:
            return False

        annotated_frame = self._draw_annotated_frame(
            rgb_frame=rgb_frame,
            prepared_detections=prepared_detections,
            selected_index=selected_index,
            step=step,
            action=action,
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        saved = cv2.imwrite(str(self.output_path), annotated_frame)
        if not saved:
            raise OSError(
                f"Failed to save PPT screenshot: {self.output_path}"
            )

        selected = prepared_detections[selected_index]
        self.best_score = selected_score
        self.best_step = int(step)
        self.best_class_name = selected["class_name"]
        self.has_saved = True

        print(
            "[PPT Screenshot] Updated best frame: "
            f"step={self.best_step}, "
            f"object={self.best_class_name}, "
            f"score={self.best_score:.3f}, "
            f"path={self.output_path}"
        )
        return True

    def print_summary(self):
        """Print the final screenshot result."""
        if self.has_saved:
            print(
                "[PPT Screenshot] Final image saved to: "
                f"{self.output_path}"
            )
            print(
                "[PPT Screenshot] Selected frame: "
                f"step={self.best_step}, "
                f"object={self.best_class_name}"
            )
        else:
            print(
                "[PPT Screenshot] No suitable frame was found. "
                "Run the script again to sample another route."
            )

    def _prepare_detections(self, detections, depth_gt_frame):
        """Validate polygons and compute aligned display measurements."""
        prepared = []

        for detection in detections:
            polygon = detection.get("polygon")
            if polygon is None:
                continue

            polygon = np.asarray(polygon, dtype=np.float32)
            if (
                polygon.ndim != 2
                or polygon.shape[0] < 3
                or polygon.shape[1] < 2
            ):
                continue

            polygon_int = self._clip_polygon(
                polygon=polygon[:, :2],
                image_shape=depth_gt_frame.shape,
            )

            class_name = str(detection.get("class_name", "unknown"))
            confidence = float(detection.get("confidence", 0.0))
            ipm_distance = float(
                detection.get("estimated_distance", float("inf"))
            )
            gt_distance = self._compute_depth_gt(
                polygon_int=polygon_int,
                depth_gt_frame=depth_gt_frame,
            )
            lowest_point = self._get_lowest_point(polygon_int)
            is_sensitive = class_name in self.sensitive_objects
            safety_distance = (
                self.semantic_safe_distance
                if is_sensitive
                else self.general_safe_distance
            )

            prepared.append(
                {
                    "polygon": polygon_int,
                    "class_name": class_name,
                    "confidence": confidence,
                    "ipm_distance": ipm_distance,
                    "gt_distance": gt_distance,
                    "lowest_point": lowest_point,
                    "is_sensitive": is_sensitive,
                    "safety_distance": safety_distance,
                }
            )

        return prepared

    def _select_candidate(self, prepared_detections, image_shape):
        """Choose the most legible valid sensitive-object detection."""
        image_height, image_width = image_shape
        image_area = float(image_height * image_width)
        image_center = np.array(
            [image_width / 2.0, image_height / 2.0],
            dtype=np.float32,
        )
        maximum_center_distance = float(np.linalg.norm(image_center))

        selected_index = None
        selected_score = float("-inf")

        for index, detection in enumerate(prepared_detections):
            if not detection["is_sensitive"]:
                continue
            if detection["confidence"] < self.min_confidence:
                continue
            if not self._is_valid_distance(detection["ipm_distance"]):
                continue
            if not self._is_valid_distance(detection["gt_distance"]):
                continue

            polygon_area = float(
                abs(cv2.contourArea(detection["polygon"]))
            )
            area_ratio = polygon_area / image_area
            if area_ratio < self.min_mask_area_ratio:
                continue

            polygon_center = np.mean(
                detection["polygon"].astype(np.float32),
                axis=0,
            )
            center_distance = float(
                np.linalg.norm(polygon_center - image_center)
            )
            center_score = max(
                0.0,
                1.0 - center_distance / maximum_center_distance,
            )

            area_score = min(area_ratio / 0.15, 1.0)
            confidence_score = detection["confidence"]

            gt_distance = detection["gt_distance"]
            distance_score = max(
                0.0,
                1.0 - abs(gt_distance - 1.8) / 3.0,
            )

            score = (
                2.0 * confidence_score
                + 1.2 * area_score
                + 0.5 * center_score
                + 0.3 * distance_score
            )

            if score > selected_score:
                selected_index = index
                selected_score = score

        return selected_index, selected_score

    def _draw_annotated_frame(
        self,
        rgb_frame,
        prepared_detections,
        selected_index,
        step,
        action,
    ):
        """Draw segmentation masks and presentation annotations."""
        rgb_frame = np.ascontiguousarray(
            rgb_frame[..., :3].astype(np.uint8)
        )
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        palette = [
            (55, 170, 255),
            (70, 210, 120),
            (235, 130, 70),
            (205, 100, 215),
            (90, 215, 225),
            (185, 185, 80),
        ]

        # Keep non-selected detections visible but visually subordinate.
        secondary_mask_layer = frame.copy()
        for index, detection in enumerate(prepared_detections):
            if index == selected_index:
                continue
            color = palette[index % len(palette)]
            cv2.fillPoly(
                secondary_mask_layer,
                [detection["polygon"]],
                color,
            )

        secondary_alpha = min(self.mask_alpha * 0.25, 0.10)
        frame = cv2.addWeighted(
            secondary_mask_layer,
            secondary_alpha,
            frame,
            1.0 - secondary_alpha,
            0.0,
        )

        # Emphasize only the selected object used for the IPM example.
        selected_mask_layer = frame.copy()
        selected_detection = prepared_detections[selected_index]
        cv2.fillPoly(
            selected_mask_layer,
            [selected_detection["polygon"]],
            (0, 215, 255),
        )
        frame = cv2.addWeighted(
            selected_mask_layer,
            self.mask_alpha,
            frame,
            1.0 - self.mask_alpha,
            0.0,
        )

        for index, detection in enumerate(prepared_detections):
            selected = index == selected_index
            color = (
                (0, 215, 255)
                if selected
                else palette[index % len(palette)]
            )
            thickness = 3 if selected else 1

            cv2.polylines(
                frame,
                [detection["polygon"]],
                isClosed=True,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

            # Show a text label only for the selected object.
            if selected:
                class_label = (
                    f"{detection['class_name']} | "
                    f"conf. {detection['confidence']:.2f} | sensitive"
                )
                label_x = int(np.min(detection["polygon"][:, 0]))
                label_y = int(np.min(detection["polygon"][:, 1])) - 6
                self._draw_text_box(
                    frame=frame,
                    lines=[class_label],
                    anchor=(label_x, label_y),
                    accent_color=color,
                    font_scale=0.43,
                )

                lowest_x, lowest_y = detection["lowest_point"]
                cv2.circle(
                    frame,
                    (lowest_x, lowest_y),
                    7,
                    (0, 255, 255),
                    -1,
                    lineType=cv2.LINE_AA,
                )
                cv2.drawMarker(
                    frame,
                    (lowest_x, lowest_y),
                    (0, 0, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=12,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )

                self._draw_text_box(
                    frame=frame,
                    lines=["Lowest polygon point used for IPM"],
                    anchor=(lowest_x + 10, lowest_y - 8),
                    accent_color=(0, 215, 255),
                    font_scale=0.40,
                )

        selected = prepared_detections[selected_index]
        avoidance_triggered = (
            selected["ipm_distance"] < selected["safety_distance"]
        )
        safety_response = (
            "Avoidance triggered"
            if avoidance_triggered
            else "No avoidance required"
        )
        readable_action = str(action).replace("_", " ").title()

        summary_lines = [
            (
                f"Selected object: {selected['class_name']} "
                f"(conf. {selected['confidence']:.2f})"
            ),
            "Sensitive object: Yes",
            f"Safety margin: {selected['safety_distance']:.2f} m",
            (
                "IPM estimate: "
                f"{self._format_distance(selected['ipm_distance'])}"
            ),
            (
                "Depth GT: "
                f"{self._format_distance(selected['gt_distance'])}"
            ),
            f"Safety response: {safety_response}",
            f"Planner action: {readable_action}",
        ]

        self._draw_text_box(
            frame=frame,
            lines=summary_lines,
            anchor=(14, 14),
            accent_color=(0, 215, 255),
            font_scale=0.50,
            anchor_is_top=True,
        )

        return frame

    def _compute_depth_gt(self, polygon_int, depth_gt_frame):
        """Compute median depth in the polygon bottom band."""
        height, width = depth_gt_frame.shape
        polygon_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [polygon_int], 1)

        bottom_y = int(np.max(polygon_int[:, 1]))
        top_y = max(
            0,
            bottom_y - self.bottom_band_height + 1,
        )

        bottom_band_mask = np.zeros((height, width), dtype=bool)
        bottom_band_mask[top_y : bottom_y + 1, :] = True

        valid_mask = (
            polygon_mask.astype(bool)
            & bottom_band_mask
            & np.isfinite(depth_gt_frame)
            & (depth_gt_frame > 0.0)
        )
        valid_depths = depth_gt_frame[valid_mask]

        if valid_depths.size == 0:
            return float("nan")

        return float(np.median(valid_depths))

    @staticmethod
    def _get_lowest_point(polygon_int):
        """Return the polygon vertex with the largest image y coordinate."""
        lowest_index = int(np.argmax(polygon_int[:, 1]))
        return (
            int(polygon_int[lowest_index, 0]),
            int(polygon_int[lowest_index, 1]),
        )

    @staticmethod
    def _clip_polygon(polygon, image_shape):
        """Round and clip polygon coordinates to image boundaries."""
        height, width = image_shape
        polygon_int = np.round(polygon).astype(np.int32)
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
        return polygon_int

    @staticmethod
    def _is_valid_distance(distance):
        """Return whether a distance is finite and positive."""
        return bool(np.isfinite(distance) and distance > 0.0)

    @staticmethod
    def _format_distance(distance):
        """Format a metric distance for display."""
        if np.isfinite(distance) and distance > 0.0:
            return f"{distance:.2f} m"
        return "N/A"

    @staticmethod
    def _draw_text_box(
        frame,
        lines,
        anchor,
        accent_color,
        font_scale,
        anchor_is_top=False,
    ):
        """Draw a clipped opaque text panel with complete background coverage."""
        if not lines:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1
        padding_x = 8
        padding_y = 7
        line_gap = 5

        text_sizes = [
            cv2.getTextSize(
                line,
                font,
                font_scale,
                thickness,
            )[0]
            for line in lines
        ]
        text_width = max(size[0] for size in text_sizes)
        line_height = max(size[1] for size in text_sizes)
        box_width = text_width + 2 * padding_x
        box_height = (
            2 * padding_y
            + len(lines) * line_height
            + (len(lines) - 1) * line_gap
        )

        frame_height, frame_width = frame.shape[:2]
        anchor_x = int(anchor[0])
        anchor_y = int(anchor[1])

        top = anchor_y if anchor_is_top else anchor_y - box_height
        left = int(
            np.clip(
                anchor_x,
                0,
                max(0, frame_width - box_width),
            )
        )
        top = int(
            np.clip(
                top,
                0,
                max(0, frame_height - box_height),
            )
        )
        right = min(frame_width - 1, left + box_width)
        bottom = min(frame_height - 1, top + box_height)

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (20, 20, 20),
            -1,
        )
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            accent_color,
            2,
        )

        text_y = top + padding_y + line_height
        for line in lines:
            cv2.putText(
                frame,
                line,
                (left + padding_x, text_y),
                font,
                font_scale,
                (245, 245, 245),
                thickness,
                cv2.LINE_AA,
            )
            text_y += line_height + line_gap


def main():
    """Run navigation and save the best presentation screenshot."""
    print("--- Initializing Modules ---")
    env = HabitatEnv(
        SCENE_PATH,
        NAVMESH_PATH,
        enable_depth=True,
    )

    dynamic_seed = int(time.time())
    env.sim.seed(dynamic_seed)
    env.sim.pathfinder.seed(dynamic_seed)

    print("Searching for a valid long distance route...")
    start_pos = env.sim.pathfinder.get_random_navigable_point()
    goal_pos = env.sim.pathfinder.get_random_navigable_point()

    while np.linalg.norm(start_pos - goal_pos) < 8.0:
        start_pos = env.sim.pathfinder.get_random_navigable_point()
        goal_pos = env.sim.pathfinder.get_random_navigable_point()

    global_planner = GlobalPlanner(
        env.sim.pathfinder,
        map_height=start_pos[1],
    )

    sim_camera_height = 1.5
    sim_image_width = 640
    sim_image_height = 480
    sim_hfov_deg = 90.0

    sim_focal_length = sim_image_width / (
        2.0
        * math.tan(
            math.radians(sim_hfov_deg) / 2.0
        )
    )

    perception = PerceptionModule(
        model_path=MODEL_PATH,
        camera_height=sim_camera_height,
        focal_length=sim_focal_length,
        img_height=sim_image_height,
    )

    model_metrics = ModelMetrics(
        model_path=MODEL_PATH,
        model=(
            perception.model.model
            if MODEL_PATH.endswith(".pt")
            else None
        ),
    )

    local_planner = DiscreteDWAPlanner(
        safe_distance=0.1,
        semantic_safe_distance=1.2,
    )

    screenshot_collector = PresentationFrameCollector(
        output_path=PPT_SCREENSHOT_PATH,
        sensitive_objects=SENSITIVE_OBJECTS,
        general_safe_distance=local_planner.safe_distance,
        semantic_safe_distance=local_planner.semantic_safe_distance,
        bottom_band_height=PPT_BOTTOM_BAND_HEIGHT,
        mask_alpha=PPT_MASK_ALPHA,
        min_confidence=PPT_MIN_CONFIDENCE,
        min_mask_area_ratio=PPT_MIN_MASK_AREA_RATIO,
    )

    visualizer = DemoVisualizer()
    evaluator = Evaluator(model_name=MODEL_PATH)
    nav_metrics = NavigationMetrics()
    evaluator.log_model(model_metrics)
    evaluator.start_episode()

    print("\n--- Starting Navigation Loop ---")
    print(f"Start: {start_pos} | Goal: {goal_pos}")

    waypoints = global_planner.plan_path(start_pos, goal_pos)
    if not waypoints:
        print("Failed to generate global path.")
        visualizer.close()
        env.close()
        return

    global_planner.visualize_path(start_pos, goal_pos, waypoints)

    agent = env.sim.get_agent(0)
    agent_state = agent.get_state()
    agent_state.position = start_pos
    agent.set_state(agent_state)

    current_wp_idx = 1
    max_steps = 800
    actual_trajectory = []

    shortest_path = 0.0
    for index in range(len(waypoints) - 1):
        shortest_path += np.linalg.norm(
            waypoints[index + 1] - waypoints[index]
        )

    nav_metrics.start_episode(
        start_position=start_pos,
        shortest_path=shortest_path,
    )

    try:
        for step in range(max_steps):
            target_wp = waypoints[current_wp_idx]

            current_state = agent.get_state()
            current_agent_position = current_state.position.copy()
            nav_metrics.update(current_agent_position)
            actual_trajectory.append(current_agent_position)

            if np.linalg.norm(agent.state.position - target_wp) < 0.25:
                print(
                    f"Reached waypoint {current_wp_idx}. "
                    "Moving to next."
                )
                current_wp_idx += 1
                if current_wp_idx >= len(waypoints):
                    print("Goal Reached!")
                    nav_metrics.finish_episode(True)
                    break
                target_wp = waypoints[current_wp_idx]

            obs = env.get_observations()
            rgb_frame = obs["color_sensor"][..., :3]
            depth_gt_frame = obs["depth_sensor"]

            detections, perception_metrics = perception.process_frame(
                rgb_frame
            )

            frame_height, frame_width = rgb_frame.shape[:2]
            ipm_distance_map = _build_ipm_distance_map(
                detections=detections,
                frame_height=frame_height,
                frame_width=frame_width,
            )

            evaluator.update_frame(step, perception_metrics)

            action = local_planner.get_best_action(
                ipm_distance_map,
                detections,
                agent.state,
                target_wp,
            )
            dist_to_wp = np.linalg.norm(
                agent.state.position - target_wp
            )
            print(
                f"Step {step}: Distance to WP={dist_to_wp:.2f}m "
                f"| Action={action}"
            )

            screenshot_collector.update(
                rgb_frame=rgb_frame,
                detections=detections,
                depth_gt_frame=depth_gt_frame,
                step=step,
                action=action,
            )

            visualizer.show_frame(
                rgb_frame=rgb_frame,
                detections=detections,
                action=action,
                step=step,
                dist_to_wp=dist_to_wp,
                depth_gt_frame=depth_gt_frame,
            )

            env.step(action)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[Manual Stop] User interrupted the script.")
    finally:
        print("\nSaving video and cleaning up...")

        if not nav_metrics.success:
            nav_metrics.finish_episode(False)

        evaluator.finish_episode(nav_metrics)
        screenshot_collector.print_summary()

        visualizer.close()
        plot_topdown_trajectory(
            env,
            start_pos,
            goal_pos,
            waypoints,
            actual_trajectory,
        )
        evaluator.save()
        env.close()


if __name__ == "__main__":
    main()
