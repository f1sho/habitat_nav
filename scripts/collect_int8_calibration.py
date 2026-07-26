from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import yaml
from PIL import Image
from habitat_sim.utils.common import quat_from_angle_axis
from ultralytics import YOLO

from core.habitat_env import HabitatEnv
from run_navigation import DEFAULT_NAVMESH_PATH, DEFAULT_SCENE_PATH


def normalize_names(names: object) -> Dict[int, str]:
    """Convert model class metadata into a YAML-friendly dictionary."""
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    raise TypeError(f"Unsupported model.names type: {type(names).__name__}")


def save_dataset_yaml(dataset_root: Path, model_path: Path) -> Path:
    """Create an Ultralytics dataset YAML for calibration images."""
    model = YOLO(str(model_path), task="segment")
    yaml_path = dataset_root / "replica_int8.yaml"
    yaml_content = {
        "path": str(dataset_root.resolve()),
        "train": "images/val",
        "val": "images/val",
        "names": normalize_names(model.names),
    }
    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(yaml_content, file, sort_keys=False, allow_unicode=True)
    return yaml_path


def collect_images(
    scene_path: str,
    navmesh_path: str,
    output_root: Path,
    target_count: int,
    views_per_position: int,
    seed: int,
    overwrite: bool,
) -> int:
    """Collect unlabeled RGB frames from random navigable poses."""
    if target_count <= 0:
        raise ValueError("target_count must be greater than zero.")
    if views_per_position <= 0:
        raise ValueError("views_per_position must be greater than zero.")

    image_dir = output_root / "images" / "val"
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    image_dir.mkdir(parents=True, exist_ok=True)

    saved_count = len(list(image_dir.glob("replica_*.png")))
    if saved_count >= target_count:
        print(f"Already found {saved_count} images; target is {target_count}.")
        return saved_count

    env = None
    rng = np.random.default_rng(seed)
    up_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    try:
        env = HabitatEnv(scene_path, navmesh_path, enable_depth=False)
        env.sim.seed(seed)
        env.sim.pathfinder.seed(seed)

        pathfinder = env.sim.pathfinder
        agent = env.sim.get_agent(0)
        attempts = 0
        max_attempts = target_count * 20

        while saved_count < target_count:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError("Unable to collect enough valid images.")

            position = np.asarray(
                pathfinder.get_random_navigable_point(), dtype=np.float32
            )
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                continue

            base_yaw = float(rng.uniform(-math.pi, math.pi))

            for view_index in range(views_per_position):
                if saved_count >= target_count:
                    break

                yaw = base_yaw + (2.0 * math.pi * view_index) / views_per_position
                state = agent.get_state()
                state.position = position.copy()
                state.rotation = quat_from_angle_axis(yaw, up_axis)
                agent.set_state(state)

                observations = env.get_observations()
                if "color_sensor" not in observations:
                    raise KeyError("Missing 'color_sensor' observation.")

                rgb_frame = np.asarray(observations["color_sensor"])
                if rgb_frame.ndim != 3 or rgb_frame.shape[2] < 3:
                    raise ValueError(
                        f"Unexpected RGB observation shape: {rgb_frame.shape}"
                    )

                rgb_frame = np.ascontiguousarray(
                    rgb_frame[..., :3], dtype=np.uint8
                )
                if float(rgb_frame.std()) < 1.0:
                    continue

                output_path = image_dir / f"replica_{saved_count:04d}.png"
                Image.fromarray(rgb_frame, mode="RGB").save(output_path)
                saved_count += 1

                if saved_count == 1 or saved_count % 25 == 0 or saved_count == target_count:
                    print(f"Saved {saved_count}/{target_count}: {output_path}")
    finally:
        if env is not None:
            env.close()

    return saved_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect Replica RGB images for TensorRT INT8 calibration."
    )
    parser.add_argument("--scene-path", default=DEFAULT_SCENE_PATH)
    parser.add_argument("--navmesh-path", default=DEFAULT_NAVMESH_PATH)
    parser.add_argument("--model-path", default="scripts/yolo26n-seg.pt")
    parser.add_argument("--output-dir", default="scripts/int8_calibration")
    parser.add_argument("--count", type=int, default=320)
    parser.add_argument("--views-per-position", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_path = Path(args.model_path).expanduser()
    output_root = Path(args.output_dir).expanduser()

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path.resolve()}")

    saved_count = collect_images(
        scene_path=args.scene_path,
        navmesh_path=args.navmesh_path,
        output_root=output_root,
        target_count=args.count,
        views_per_position=args.views_per_position,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    yaml_path = save_dataset_yaml(output_root, model_path)

    print("\nCalibration dataset ready.")
    print(f"Images: {saved_count}")
    print(f"Directory: {output_root.resolve()}")
    print(f"YAML: {yaml_path.resolve()}")


if __name__ == "__main__":
    main()
