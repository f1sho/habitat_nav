from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.perception import PerceptionModule
from evaluation.evaluator import Evaluator
from evaluation.model_metrics import ModelMetrics
from evaluation.statistics import Statistics
from run_navigation import (
    DEFAULT_NAVMESH_PATH,
    DEFAULT_SCENE_PATH,
    SIM_CAMERA_HEIGHT,
    SIM_FOCAL_LENGTH,
    SIM_IMAGE_HEIGHT,
    run_navigation,
)


# Start with models that already exist. Add the engine files after export.
DEFAULT_MODELS = [
    "yolo26n-seg.pt",
    "yolo26n-seg.onnx",
    # "yolo26n-seg_fp16.engine",
    # "yolo26n-seg_int8.engine",
]


def model_name_from_path(model_path: str) -> str:
    path = Path(model_path)
    base_name = path.stem.replace("-", "_")
    backend_name = path.suffix.lower().lstrip(".").replace("-", "_")

    if backend_name:
        return f"{base_name}_{backend_name}"

    return base_name


def read_first_csv_row(csv_path: Path) -> Dict[str, Any]:
    if not csv_path.exists():
        return {}

    with csv_path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)
        row = next(reader, None)

    return dict(row) if row is not None else {}


def build_comparison_summary(
    results_dir: Path,
    model_paths: List[str],
) -> Path:
    """
    Aggregate per-model CSV files into one thesis-table-friendly CSV.
    """
    rows: List[Dict[str, Any]] = []

    for model_path in model_paths:
        model_name = model_name_from_path(model_path)
        model_dir = results_dir / model_name

        model_summary = read_first_csv_row(
            model_dir / "model_metrics.csv"
        )
        perception_summary = Statistics.summarize_frame_csv(
            str(model_dir / "frame_metrics.csv")
        )
        navigation_summary = Statistics.summarize_csv(
            str(model_dir / "episode_metrics.csv")
        )

        if (
            not model_summary
            and not perception_summary
            and not navigation_summary
        ):
            continue

        row: Dict[str, Any] = {
            "model_name": model_name,
            "model_path": model_path,
        }

        row.update(model_summary)
        row.update(perception_summary)

        navigation_summary.pop("episode", None)

        # Rename the averaged binary success field to success_rate.
        if "success" in navigation_summary:
            navigation_summary["success_rate"] = (
                navigation_summary.pop("success")
            )

        row.update(navigation_summary)
        rows.append(row)

    output_path = results_dir / "comparison_summary.csv"

    if not rows:
        return output_path

    fieldnames: List[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def run_worker(args: argparse.Namespace) -> None:
    """
    Evaluate one model in one fresh process.

    The model is loaded exactly once and reused for all episodes assigned to
    this worker. A fresh process per model prevents CUDA/ONNX allocator state
    from contaminating the next model's GPU-memory baseline.
    """
    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file does not exist: {model_path}"
        )

    model_name = model_name_from_path(str(model_path))
    results_dir = Path(args.results_dir)
    model_result_dir = results_dir / model_name

    if args.overwrite and model_result_dir.exists():
        shutil.rmtree(model_result_dir)

    print(
        f"\n========== Evaluating {model_name} ==========",
        flush=True,
    )
    print(
        f"Model: {model_path}",
        flush=True,
    )
    print(
        f"Episodes: {args.episodes}",
        flush=True,
    )
    print(
        f"Seeds: {args.seed_start} "
        f"to {args.seed_start + args.episodes - 1}",
        flush=True,
    )

    # Measure the process before creating the model/runtime.
    gpu_memory_before_mb = (
        ModelMetrics.get_process_gpu_memory_mb()
    )

    perception = PerceptionModule(
        model_path=str(model_path),
        camera_height=SIM_CAMERA_HEIGHT,
        focal_length=SIM_FOCAL_LENGTH,
        img_height=SIM_IMAGE_HEIGHT,
        device=args.device,
    )

    # Warm up once per model. These frames are not logged.
    warmup_frame = np.zeros(
        (SIM_IMAGE_HEIGHT, 640, 3),
        dtype=np.uint8,
    )

    for _ in range(args.warmup_frames):
        perception.process_frame(warmup_frame)

    gpu_memory_after_mb = (
        ModelMetrics.get_process_gpu_memory_mb()
    )

    evaluator = Evaluator(
        model_name=model_name,
        save_dir=str(results_dir),
    )

    inner_model = None

    if model_path.suffix.lower() == ".pt":
        inner_model = getattr(
            perception.model,
            "model",
            None,
        )

    evaluator.log_model(
        ModelMetrics(
            model_path=str(model_path),
            model=inner_model,
            gpu_memory_before_mb=gpu_memory_before_mb,
            gpu_memory_after_mb=gpu_memory_after_mb,
        )
    )

    episode_results: List[Dict[str, Any]] = []

    for episode_index in range(args.episodes):
        seed = args.seed_start + episode_index

        result = run_navigation(
            model_path=str(model_path),
            scene_path=args.scene_path,
            navmesh_path=args.navmesh_path,
            episode_id=episode_index,
            seed=seed,
            min_start_goal_distance=args.min_distance,
            max_route_attempts=args.max_route_attempts,
            max_steps=args.max_steps,
            waypoint_threshold=args.waypoint_threshold,
            safe_distance=args.safe_distance,
            semantic_safe_distance=args.semantic_safe_distance,
            perception=perception,
            evaluator=evaluator,
            log_model_metrics=False,
            save_results=False,
            verbose=False,
        )

        episode_results.append(result)

        print(
            f"[{model_name}] "
            f"episode {episode_index + 1}/{args.episodes}, "
            f"seed={seed}, "
            f"success={result.get('success')}, "
            f"spl={result.get('spl')}, "
            f"collisions={result.get('collision_count')}",
            flush=True,
        )

    evaluator.save()

    model_result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = {
        "model_name": model_name,
        "model_path": str(model_path),
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "seeds": [
            args.seed_start + index
            for index in range(args.episodes)
        ],
        "warmup_frames": args.warmup_frames,
        "scene_path": args.scene_path,
        "navmesh_path": args.navmesh_path,
        "min_start_goal_distance": args.min_distance,
        "max_route_attempts": args.max_route_attempts,
        "max_steps": args.max_steps,
        "waypoint_threshold": args.waypoint_threshold,
        "safe_distance": args.safe_distance,
        "semantic_safe_distance": args.semantic_safe_distance,
        "device": args.device,
        "episode_results": episode_results,
    }

    with (
        model_result_dir / "run_manifest.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"Completed model: {model_name}",
        flush=True,
    )


def run_parent(args: argparse.Namespace) -> None:
    """
    Launch one isolated worker process per model.
    """
    model_paths = (
        args.models
        if args.models
        else DEFAULT_MODELS
    )

    script_path = Path(__file__).resolve()
    successful_models: List[str] = []
    failed_models: List[str] = []

    for model_path in model_paths:
        command = [
            sys.executable,
            str(script_path),
            "--worker",
            "--model-path",
            model_path,
            "--episodes",
            str(args.episodes),
            "--seed-start",
            str(args.seed_start),
            "--warmup-frames",
            str(args.warmup_frames),
            "--scene-path",
            args.scene_path,
            "--navmesh-path",
            args.navmesh_path,
            "--results-dir",
            args.results_dir,
            "--min-distance",
            str(args.min_distance),
            "--max-route-attempts",
            str(args.max_route_attempts),
            "--max-steps",
            str(args.max_steps),
            "--waypoint-threshold",
            str(args.waypoint_threshold),
            "--safe-distance",
            str(args.safe_distance),
            "--semantic-safe-distance",
            str(args.semantic_safe_distance),
            "--device",
            str(args.device),
        ]

        if args.overwrite:
            command.append("--overwrite")

        completed = subprocess.run(
            command,
            check=False,
        )

        if completed.returncode == 0:
            successful_models.append(model_path)
        else:
            failed_models.append(model_path)
            print(
                f"[ERROR] Model failed: {model_path}",
                file=sys.stderr,
                flush=True,
            )

    summary_path = build_comparison_summary(
        results_dir=Path(args.results_dir),
        model_paths=successful_models,
    )

    if successful_models:
        print(
            f"\nComparison summary saved to: {summary_path}",
            flush=True,
        )

    if failed_models:
        raise RuntimeError(
            "One or more models failed: "
            + ", ".join(failed_models)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch evaluation for compressed perception models "
            "in Habitat-Sim navigation."
        )
    )

    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model-path",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Model files evaluated in identical seed order. "
            "Defaults to DEFAULT_MODELS in this script."
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=10,
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
        "--results-dir",
        default="results",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--max-route-attempts",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=800,
    )
    parser.add_argument(
        "--waypoint-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--safe-distance",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--semantic-safe-distance",
        type=float,
        default=1.2,
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing result folders for evaluated models.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero.")

    if args.warmup_frames < 0:
        parser.error(
            "--warmup-frames cannot be negative."
        )

    if args.worker:
        if not args.model_path:
            parser.error(
                "--model-path is required in worker mode."
            )
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
