from __future__ import annotations

import argparse
import math

import numpy as np

from core.perception import PerceptionModule
from evaluation.ipm_accuracy_evaluator import (
    IPMAccuracyEvaluator,
)
from run_navigation import (
    DEFAULT_NAVMESH_PATH,
    DEFAULT_SCENE_PATH,
    run_navigation,
)


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_HEIGHT = 1.5
CAMERA_HFOV_DEG = 90.0

FOCAL_LENGTH = IMAGE_WIDTH / (
    2.0
    * math.tan(
        math.radians(CAMERA_HFOV_DEG) / 2.0
    )
)


def run_ipm_accuracy(
    model_path,
    scene_path,
    navmesh_path,
    episodes,
    seed_start,
    max_steps,
    save_root,
):
    """
    Run navigation episodes while collecting IPM distance accuracy samples.
    """

    perception = PerceptionModule(
        model_path=model_path,
        camera_height=CAMERA_HEIGHT,
        focal_length=FOCAL_LENGTH,
        img_height=IMAGE_HEIGHT,
        device=0,
    )

    # Warm up the perception backend once before collecting samples.
    warmup_frame = np.zeros(
        (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        dtype=np.uint8,
    )

    for _ in range(10):
        perception.process_frame(warmup_frame)

    ipm_evaluator = IPMAccuracyEvaluator(
        model_name=model_path,
        camera_height=CAMERA_HEIGHT,
        focal_length=FOCAL_LENGTH,
        image_height=IMAGE_HEIGHT,
        save_root=save_root,
        bottom_band_height=5,
        contact_height_threshold=0.15,
    )

    episode_results = []

    try:
        for episode_id in range(episodes):
            seed = seed_start + episode_id

            print(
                f"[IPM Accuracy] "
                f"Episode {episode_id + 1}/{episodes}, "
                f"seed={seed}"
            )

            result = run_navigation(
                model_path=model_path,
                scene_path=scene_path,
                navmesh_path=navmesh_path,
                episode_id=episode_id,
                seed=seed,
                max_steps=max_steps,
                perception=perception,
                ipm_accuracy_evaluator=ipm_evaluator,
                log_model_metrics=False,
                save_results=False,
                verbose=False,
            )

            episode_results.append(result)

            print(
                f"[IPM Accuracy] "
                f"success={result['success']}, "
                f"steps={result['steps']}"
            )

    finally:
        summary = ipm_evaluator.save()

    successful_episodes = sum(
        int(result["success"])
        for result in episode_results
    )

    print()
    print("[IPM Accuracy] Evaluation completed")
    print(
        f"[IPM Accuracy] Episodes: "
        f"{len(episode_results)}"
    )
    print(
        f"[IPM Accuracy] Successful episodes: "
        f"{successful_episodes}"
    )
    print(
        f"[IPM Accuracy] Collected samples: "
        f"{summary['total_samples']}"
    )

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate IPM distance accuracy using "
            "aligned Habitat depth observations."
        )
    )

    parser.add_argument(
        "--model-path",
        default="yolo26n-seg.pt",
    )

    parser.add_argument(
        "--scene-path",
        default=DEFAULT_SCENE_PATH,
    )

    parser.add_argument(
        "--navmesh-path",
        default=DEFAULT_NAVMESH_PATH,
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seed-start",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--save-root",
        default="results/ipm_accuracy",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.episodes <= 0:
        raise ValueError(
            "--episodes must be greater than zero."
        )

    run_ipm_accuracy(
        model_path=args.model_path,
        scene_path=args.scene_path,
        navmesh_path=args.navmesh_path,
        episodes=args.episodes,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        save_root=args.save_root,
    )


if __name__ == "__main__":
    main()