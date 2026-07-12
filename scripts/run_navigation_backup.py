from __future__ import annotations
import math
import argparse
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np

from core.habitat_env import HabitatEnv
from core.perception import PerceptionModule
from core.planning.global_planner import GlobalPlanner
from core.planning.local_planner import DiscreteDWAPlanner
from evaluation.evaluator import Evaluator
from evaluation.model_metrics import ModelMetrics
from evaluation.navigation_metrics import NavigationMetrics


DEFAULT_SCENE_PATH = (
    "/home/hannah/data/replica_v1/apartment_2/habitat/mesh_semantic.ply"
)
DEFAULT_NAVMESH_PATH = (
    "/home/hannah/data/replica_v1/apartment_2/habitat/mesh_semantic.navmesh"
)

SIM_CAMERA_HEIGHT = 1.5
SIM_IMAGE_WIDTH = 640
SIM_IMAGE_HEIGHT = 480
SIM_HFOV_DEG = 90.0

SIM_FOCAL_LENGTH = SIM_IMAGE_WIDTH / (
    2.0
    * math.tan(
        math.radians(SIM_HFOV_DEG) / 2.0
    )
)


def _model_name_from_path(model_path: str) -> str:
    """Return a stable, backend-specific name for the evaluator."""
    path = Path(model_path)
    base_name = path.stem.replace("-", "_")
    backend_name = path.suffix.lower().lstrip(".").replace("-", "_")

    if backend_name:
        return f"{base_name}_{backend_name}"

    return base_name


def _compute_path_length(waypoints: List[np.ndarray]) -> float:
    """Compute the length of the global path represented by waypoints."""
    return float(
        sum(
            np.linalg.norm(waypoints[index + 1] - waypoints[index])
            for index in range(len(waypoints) - 1)
        )
    )



def _build_ipm_distance_map(
    detections: List[Dict[str, Any]],
    frame_height: int,
    frame_width: int,
    default_distance: float = 10.0,
) -> np.ndarray:
    """
    Convert instance-level IPM estimates into the distance-map interface used
    by the existing DiscreteDWAPlanner.

    Each detected instance occupies its bounding rectangle (or the bounds of
    its segmentation polygon) and is filled with its IPM estimated distance.
    The rest of the image is treated as far free space.

    This is not Habitat depth. It is a synthetic map built only from RGB,
    segmentation geometry, and IPM distance estimates.
    """
    distance_map = np.full(
        (frame_height, frame_width),
        default_distance,
        dtype=np.float32,
    )

    for detection in detections:
        estimated_distance = detection.get("estimated_distance")

        if estimated_distance is None:
            continue

        try:
            estimated_distance = float(estimated_distance)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(estimated_distance) or estimated_distance <= 0.0:
            continue

        estimated_distance = min(estimated_distance, default_distance)

        bounds = None

        bbox = detection.get("bbox")
        if bbox is not None and len(bbox) >= 4:
            bounds = (
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2]),
                float(bbox[3]),
            )
        else:
            polygon = detection.get("polygon")
            if polygon is not None:
                polygon_array = np.asarray(polygon)
                if polygon_array.ndim == 2 and polygon_array.shape[0] > 0:
                    bounds = (
                        float(np.min(polygon_array[:, 0])),
                        float(np.min(polygon_array[:, 1])),
                        float(np.max(polygon_array[:, 0])),
                        float(np.max(polygon_array[:, 1])),
                    )

        if bounds is None:
            continue

        x_min, y_min, x_max, y_max = bounds

        x_min_i = int(np.clip(np.floor(x_min), 0, frame_width - 1))
        x_max_i = int(np.clip(np.ceil(x_max), 0, frame_width - 1))
        y_min_i = int(np.clip(np.floor(y_min), 0, frame_height - 1))
        y_max_i = int(np.clip(np.ceil(y_max), 0, frame_height - 1))

        if x_max_i < x_min_i or y_max_i < y_min_i:
            continue

        region = distance_map[
            y_min_i : y_max_i + 1,
            x_min_i : x_max_i + 1,
        ]
        np.minimum(region, estimated_distance, out=region)

    return distance_map


def _sample_valid_route(
    env: HabitatEnv,
    min_start_goal_distance: float,
    max_route_attempts: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], GlobalPlanner]:
    """
    Sample a navigable start/goal pair and ensure that the global planner
    can generate a path containing at least two waypoints.
    """
    pathfinder = env.sim.pathfinder

    for _ in range(max_route_attempts):
        start_position = pathfinder.get_random_navigable_point()
        goal_position = pathfinder.get_random_navigable_point()

        if not np.all(np.isfinite(start_position)):
            continue
        if not np.all(np.isfinite(goal_position)):
            continue

        straight_line_distance = float(
            np.linalg.norm(start_position - goal_position)
        )
        if straight_line_distance < min_start_goal_distance:
            continue

        global_planner = GlobalPlanner(
            pathfinder,
            map_height=float(start_position[1]),
        )
        waypoints = global_planner.plan_path(start_position, goal_position)

        if waypoints is not None and len(waypoints) >= 2:
            return (
                np.asarray(start_position, dtype=np.float32),
                np.asarray(goal_position, dtype=np.float32),
                list(waypoints),
                global_planner,
            )

    raise RuntimeError(
        "Unable to sample a valid route after "
        f"{max_route_attempts} attempts. "
        "Reduce min_start_goal_distance or check the navmesh."
    )


def _build_result(
    nav_metrics: NavigationMetrics,
    model_name: str,
    episode_id: int,
    seed: int,
    success: bool,
    steps: int,
    start_position: np.ndarray,
    goal_position: np.ndarray,
    shortest_path: float,
) -> Dict[str, Any]:
    """
    Build a lightweight result dictionary for run_evaluation.py.

    NavigationMetrics remains the source of truth for CSV logging. Common
    attributes are copied when they exist, without assuming a specific
    NavigationMetrics implementation.
    """
    result: Dict[str, Any] = {
        "model_name": model_name,
        "episode_id": episode_id,
        "seed": seed,
        "success": bool(success),
        "steps": int(steps),
        "start_position": start_position.tolist(),
        "goal_position": goal_position.tolist(),
        "shortest_path": float(shortest_path),
    }

    candidate_attributes = (
        "spl",
        "navigation_time",
        "path_length",
        "collision_count",
        "success",
    )

    for attribute_name in candidate_attributes:
        if hasattr(nav_metrics, attribute_name):
            value = getattr(nav_metrics, attribute_name)
            if isinstance(value, np.generic):
                value = value.item()
            result[attribute_name] = value

    return result


def run_navigation(
    model_path: str,
    scene_path: str = DEFAULT_SCENE_PATH,
    navmesh_path: str = DEFAULT_NAVMESH_PATH,
    episode_id: int = 0,
    seed: int = 42,
    min_start_goal_distance: float = 8.0,
    max_route_attempts: int = 500,
    max_steps: int = 800,
    waypoint_threshold: float = 0.25,
    safe_distance: float = 0.1,
    semantic_safe_distance: float = 1.2,
    perception: Optional[PerceptionModule] = None,
    evaluator: Optional[Evaluator] = None,
    log_model_metrics: bool = True,
    save_results: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run one headless Habitat navigation episode.

    This function contains no GUI, video writer, trajectory plot, image save,
    debug frame output, or artificial sleep.

    For fair model comparison, use the same episode seeds for every model.

    Parameters
    ----------
    perception:
        Optional preloaded perception module. run_evaluation.py should pass
        one here so the model is loaded once per model rather than once per
        episode.
    evaluator:
        Optional shared evaluator. run_evaluation.py can reuse one evaluator
        across all episodes for the same model and call evaluator.save() once
        after the batch.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be greater than zero.")
    if waypoint_threshold <= 0:
        raise ValueError("waypoint_threshold must be greater than zero.")
    if min_start_goal_distance < 0:
        raise ValueError("min_start_goal_distance cannot be negative.")

    model_name = _model_name_from_path(model_path)
    owns_perception = perception is None
    owns_evaluator = evaluator is None

    env: Optional[HabitatEnv] = None
    quiet_stream: Optional[TextIO] = None
    nav_metrics = NavigationMetrics()

    episode_started = False
    success = False
    steps_executed = 0
    start_position: Optional[np.ndarray] = None
    goal_position: Optional[np.ndarray] = None
    shortest_path = 0.0
    result: Optional[Dict[str, Any]] = None

    try:
        env = HabitatEnv(scene_path, navmesh_path)

        # Fixed seeds make episodes reproducible across model variants.
        env.sim.seed(seed)
        env.sim.pathfinder.seed(seed)

        (
            start_position,
            goal_position,
            waypoints,
            _global_planner,
        ) = _sample_valid_route(
            env=env,
            min_start_goal_distance=min_start_goal_distance,
            max_route_attempts=max_route_attempts,
        )

        # Habitat has already been initialized here.
        # Capture process GPU memory before loading the perception model.
        gpu_memory_before_model = (
            ModelMetrics.get_process_gpu_memory_mb()
        )

        if perception is None:
            perception = PerceptionModule(
                model_path=model_path,
                camera_height=SIM_CAMERA_HEIGHT,
                focal_length=SIM_FOCAL_LENGTH,
                img_height=SIM_IMAGE_HEIGHT,
                device=0,
            )

        if evaluator is None:
            evaluator = Evaluator(model_name=model_name)

        # if log_model_metrics:
        #     inner_model = None
        #     if Path(model_path).suffix.lower() == ".pt":
        #         wrapped_model = getattr(perception, "model", None)
        #         inner_model = getattr(wrapped_model, "model", None)

        #     model_metrics = ModelMetrics(
        #         model_path=model_path,
        #         model=inner_model,
        #     )
        #     evaluator.log_model(model_metrics)

        local_planner = DiscreteDWAPlanner(
            safe_distance=safe_distance,
            semantic_safe_distance=semantic_safe_distance,
        )

        if not verbose:
            quiet_stream = open(os.devnull, "w", encoding="utf-8")

        agent = env.sim.get_agent(0)
        agent_state = agent.get_state()
        agent_state.position = start_position
        agent.set_state(agent_state)
        
        # Perception warm-up
        if owns_perception:
            warmup_observations = env.get_observations()
            warmup_rgb_frame = warmup_observations["color_sensor"][..., :3]

            warmup_frames = 10

            for _ in range(warmup_frames):
                perception.process_frame(warmup_rgb_frame)

        # Capture steady-state GPU memory after model loading and warm-up.
        gpu_memory_after_warmup = (
            ModelMetrics.get_process_gpu_memory_mb()
        )

        if log_model_metrics:
            inner_model = None

            if Path(model_path).suffix.lower() == ".pt":
                wrapped_model = getattr(
                    perception,
                    "model",
                    None,
                )
                inner_model = getattr(
                    wrapped_model,
                    "model",
                    None,
                )

            model_metrics = ModelMetrics(
                model_path=model_path,
                model=inner_model,
                gpu_memory_before_mb=gpu_memory_before_model,
                gpu_memory_after_mb=gpu_memory_after_warmup,
            )

            evaluator.log_model(model_metrics)

        # -------------------------------------------------
        # Check actual inference backend after warm-up
        # -------------------------------------------------

        # predictor = getattr(perception.model, "predictor", None)
        # backend = getattr(predictor, "model", None)
        # session = getattr(backend, "session", None)

        # if session is not None:
        #     print(
        #         "[Runtime] Active ONNX providers:",
        #         session.get_providers(),
        #     )
        #     print(
        #         "[Runtime] ONNX provider options:",
        #         session.get_provider_options(),
        #     )
        # else:
        #     print(
        #         "[Runtime] Backend type:",
        #         type(backend).__name__,
        #     )
        #     print(
        #         "[Runtime] Device:",
        #         getattr(backend, "device", "unknown"),
        #     )

        shortest_path = _compute_path_length(waypoints)

        evaluator.start_episode()
        nav_metrics.start_episode(
            start_position=start_position,
            shortest_path=shortest_path,
        )
        episode_started = True

        current_waypoint_index = 1

        for step in range(max_steps):
            steps_executed = step + 1

            current_state = agent.get_state()
            current_position = current_state.position.copy()
            nav_metrics.update(current_position)

            target_waypoint = waypoints[current_waypoint_index]
            distance_to_waypoint = float(
                np.linalg.norm(current_position - target_waypoint)
            )

            # Advance through all waypoints already reached.
            while distance_to_waypoint < waypoint_threshold:
                current_waypoint_index += 1

                if current_waypoint_index >= len(waypoints):
                    success = True
                    break

                target_waypoint = waypoints[current_waypoint_index]
                distance_to_waypoint = float(
                    np.linalg.norm(current_position - target_waypoint)
                )

            if success:
                break

            observations = env.get_observations()
            rgb_frame = observations["color_sensor"][..., :3]

            if quiet_stream is None:
                detections, perception_metrics = perception.process_frame(
                    rgb_frame
                )
            else:
                # Suppress Ultralytics and planner frame-by-frame console logs
                # during headless batch evaluation.
                with redirect_stdout(quiet_stream), redirect_stderr(quiet_stream):
                    detections, perception_metrics = perception.process_frame(
                        rgb_frame
                    )

            evaluator.update_frame(
                step,
                perception_metrics,
            )

            frame_height, frame_width = rgb_frame.shape[:2]
            ipm_distance_map = _build_ipm_distance_map(
                detections=detections,
                frame_height=frame_height,
                frame_width=frame_width,
            )

            if quiet_stream is None:
                # action = "move_forward"
                action = local_planner.get_best_action(
                    ipm_distance_map,
                    detections,
                    current_state,
                    target_waypoint,
                )
            else:
                with redirect_stdout(quiet_stream), redirect_stderr(quiet_stream):
                    # action = "move_forward"
                    action = local_planner.get_best_action(
                        ipm_distance_map,
                        detections,
                        current_state,
                        target_waypoint,
                    )

            step_observations = env.step(action)

            if (
                not isinstance(step_observations, dict)
                or "collided" not in step_observations
            ):
                raise RuntimeError(
                    "Habitat-Sim step result does not contain "
                    "the 'collided' field."
                )

            collided = bool(
                step_observations["collided"]
            )

            # print(
            #     f"[Collision Debug] "
            #     f"step={step}, "
            #     f"action={action}, "
            #     f"collided={collided}"
            # )

            nav_metrics.update_collision(collided)

            if verbose and collided:
                print(
                    f"[Collision] step={step}, action={action}"
                )

    finally:
        # Log exactly one episode result, including failed and interrupted runs.
        if episode_started:
            nav_metrics.finish_episode(success)
            if evaluator is not None:
                evaluator.finish_episode(nav_metrics)

            if start_position is not None and goal_position is not None:
                result = _build_result(
                    nav_metrics=nav_metrics,
                    model_name=model_name,
                    episode_id=episode_id,
                    seed=seed,
                    success=success,
                    steps=steps_executed,
                    start_position=start_position,
                    goal_position=goal_position,
                    shortest_path=shortest_path,
                )

        # A standalone call owns its evaluator and therefore saves immediately.
        # A batch runner should pass a shared evaluator and save once per model.
        if owns_evaluator and evaluator is not None and save_results:
            evaluator.save()

        if quiet_stream is not None:
            quiet_stream.close()

        if env is not None:
            env.close()

        # PerceptionModule currently has no required close call. This variable
        # documents ownership for future backends such as TensorRT.
        _ = owns_perception

        if verbose and start_position is not None and goal_position is not None:
            print(
                f"[Episode {episode_id}] model={model_name}, seed={seed}, "
                f"success={success}, steps={steps_executed}, "
                f"shortest_path={shortest_path:.3f} m"
            )

    if result is None:
        raise RuntimeError("Navigation episode ended before metrics were created.")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one headless Habitat-Sim navigation episode."
    )
    parser.add_argument(
        "--model-path",
        default="yolo26n-seg.onnx",
        help="Path to the PT, ONNX, FP16, or INT8 model.",
    )
    parser.add_argument("--scene-path", default=DEFAULT_SCENE_PATH)
    parser.add_argument("--navmesh-path", default=DEFAULT_NAVMESH_PATH)
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-distance", type=float, default=8.0)
    parser.add_argument("--max-route-attempts", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--waypoint-threshold", type=float, default=0.25)
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run the episode without writing evaluator CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = run_navigation(
        model_path=args.model_path,
        scene_path=args.scene_path,
        navmesh_path=args.navmesh_path,
        episode_id=args.episode_id,
        seed=args.seed,
        min_start_goal_distance=args.min_distance,
        max_route_attempts=args.max_route_attempts,
        max_steps=args.max_steps,
        waypoint_threshold=args.waypoint_threshold,
        save_results=not args.no_save,
        verbose=True,
    )

    print(result)


if __name__ == "__main__":
    main()
