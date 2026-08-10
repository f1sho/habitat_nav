from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Dict, Iterable, List

import numpy as np


PIXEL_OFFSETS = (-5, -2, -1, 0, 1, 2, 5)


def calculate_focal_length(
    image_width: int,
    hfov_deg: float,
) -> float:
    """Calculate focal length in pixels from horizontal field of view."""
    return image_width / (
        2.0 * math.tan(math.radians(hfov_deg) / 2.0)
    )


def estimate_ipm_distance(
    pixel_y: float,
    camera_height: float,
    focal_length: float,
    principal_y: float,
    camera_pitch_rad: float,
) -> float:
    """
    Estimate horizontal ground distance from a ground-contact image row.

    This implementation mirrors the final perception module. The camera is
    pitched downward, so the ground-ray angle is the sum of the fixed camera
    pitch and the pixel ray angle relative to the optical axis.
    """
    pixel_down_angle = math.atan2(
        pixel_y - principal_y,
        focal_length,
    )

    ground_ray_angle = (
        camera_pitch_rad
        + pixel_down_angle
    )

    if ground_ray_angle <= 0.0:
        return float("nan")

    tangent = math.tan(ground_ray_angle)

    if not math.isfinite(tangent) or tangent <= 0.0:
        return float("nan")

    distance = camera_height / tangent

    if not math.isfinite(distance) or distance <= 0.0:
        return float("nan")

    return float(distance)


def project_ground_contact_to_image_y(
    true_distance: float,
    camera_height: float,
    focal_length: float,
    principal_y: float,
    camera_pitch_rad: float,
) -> float:
    """
    Project an ideal ground-contact point at a known distance into image space.

    The function is the analytical inverse of estimate_ipm_distance before
    pixel quantisation is applied.
    """
    ground_ray_angle = math.atan2(
        camera_height,
        true_distance,
    )

    pixel_down_angle = (
        ground_ray_angle
        - camera_pitch_rad
    )

    projected_y = (
        principal_y
        + focal_length * math.tan(pixel_down_angle)
    )

    return float(projected_y)


def save_rows(
    path: str,
    rows: List[Dict[str, float]],
) -> None:
    """Save dictionaries to a CSV file."""
    if not rows:
        return

    with open(
        path,
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


def summarize(
    rows: Iterable[Dict[str, float]],
) -> Dict[str, float]:
    """Calculate error metrics for ideal quantised IPM samples."""
    rows = list(rows)

    signed_errors = np.asarray(
        [
            row["signed_error_m"]
            for row in rows
        ],
        dtype=np.float64,
    )

    absolute_errors = np.abs(signed_errors)

    true_distances = np.asarray(
        [
            row["true_distance_m"]
            for row in rows
        ],
        dtype=np.float64,
    )

    return {
        "samples": len(rows),
        "mae_m": float(
            np.mean(absolute_errors)
        ),
        "rmse_m": float(
            np.sqrt(
                np.mean(
                    np.square(signed_errors)
                )
            )
        ),
        "mape_pct": float(
            np.mean(
                absolute_errors / true_distances
            )
            * 100.0
        ),
        "mean_signed_error_m": float(
            np.mean(signed_errors)
        ),
        "max_absolute_error_m": float(
            np.max(absolute_errors)
        ),
    }


def run_test(
    args: argparse.Namespace,
) -> None:
    """
    Run ideal-condition and vertical pixel-sensitivity IPM tests.

    The ideal test isolates the calibrated IPM geometry from perception and
    segmentation errors. Ground-contact locations are analytically projected
    into the image, quantised to integer pixel rows, and converted back into
    metric distance using the same pitched-camera geometry as the navigation
    perception module.
    """
    focal_length = calculate_focal_length(
        args.image_width,
        args.hfov_deg,
    )

    principal_y = (
        args.image_height / 2.0
    )

    camera_pitch_rad = math.radians(
        args.camera_pitch_deg
    )

    geometric_horizon_y = (
        principal_y
        + focal_length
        * math.tan(-camera_pitch_rad)
    )

    bottom_row = float(
        args.image_height - 1
    )

    minimum_visible_distance = (
        estimate_ipm_distance(
            pixel_y=bottom_row,
            camera_height=args.camera_height,
            focal_length=focal_length,
            principal_y=principal_y,
            camera_pitch_rad=camera_pitch_rad,
        )
    )

    requested_distances = np.arange(
        args.min_distance,
        args.max_distance
        + args.distance_step / 2.0,
        args.distance_step,
        dtype=np.float64,
    )

    ideal_rows: List[
        Dict[str, float]
    ] = []

    sensitivity_rows: List[
        Dict[str, float]
    ] = []

    for true_distance in requested_distances:
        projected_y = (
            project_ground_contact_to_image_y(
                true_distance=float(
                    true_distance
                ),
                camera_height=args.camera_height,
                focal_length=focal_length,
                principal_y=principal_y,
                camera_pitch_rad=camera_pitch_rad,
            )
        )

        if (
            projected_y < 0.0
            or projected_y
            > args.image_height - 1
        ):
            continue

        quantized_y = float(
            np.rint(projected_y)
        )

        estimated_distance = (
            estimate_ipm_distance(
                pixel_y=quantized_y,
                camera_height=args.camera_height,
                focal_length=focal_length,
                principal_y=principal_y,
                camera_pitch_rad=camera_pitch_rad,
            )
        )

        if not np.isfinite(
            estimated_distance
        ):
            continue

        signed_error = (
            estimated_distance
            - true_distance
        )

        ideal_rows.append(
            {
                "true_distance_m": float(
                    true_distance
                ),
                "projected_y_px": float(
                    projected_y
                ),
                "quantized_y_px": float(
                    quantized_y
                ),
                "estimated_distance_m": float(
                    estimated_distance
                ),
                "signed_error_m": float(
                    signed_error
                ),
                "absolute_error_m": float(
                    abs(signed_error)
                ),
                "relative_error_pct": float(
                    abs(signed_error)
                    / true_distance
                    * 100.0
                ),
            }
        )

        for pixel_offset in PIXEL_OFFSETS:
            observed_y = (
                quantized_y
                + pixel_offset
            )

            if (
                observed_y < 0.0
                or observed_y
                > args.image_height - 1
            ):
                continue

            offset_estimate = (
                estimate_ipm_distance(
                    pixel_y=observed_y,
                    camera_height=(
                        args.camera_height
                    ),
                    focal_length=focal_length,
                    principal_y=principal_y,
                    camera_pitch_rad=(
                        camera_pitch_rad
                    ),
                )
            )

            if not np.isfinite(
                offset_estimate
            ):
                continue

            offset_error = (
                offset_estimate
                - true_distance
            )

            sensitivity_rows.append(
                {
                    "true_distance_m": float(
                        true_distance
                    ),
                    "base_quantized_y_px": float(
                        quantized_y
                    ),
                    "pixel_offset_px": int(
                        pixel_offset
                    ),
                    "observed_y_px": float(
                        observed_y
                    ),
                    "estimated_distance_m": float(
                        offset_estimate
                    ),
                    "signed_error_m": float(
                        offset_error
                    ),
                    "absolute_error_m": float(
                        abs(offset_error)
                    ),
                    "relative_error_pct": float(
                        abs(offset_error)
                        / true_distance
                        * 100.0
                    ),
                }
            )

    if not ideal_rows:
        raise RuntimeError(
            "No visible ground-contact samples were generated. "
            "Check the distance range and camera parameters."
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True,
    )

    ideal_path = os.path.join(
        args.output_dir,
        "ipm_ideal_samples.csv",
    )

    sensitivity_path = os.path.join(
        args.output_dir,
        "ipm_pixel_sensitivity.csv",
    )

    summary_path = os.path.join(
        args.output_dir,
        "ipm_ideal_summary.csv",
    )

    save_rows(
        ideal_path,
        ideal_rows,
    )

    save_rows(
        sensitivity_path,
        sensitivity_rows,
    )

    summary = summarize(
        ideal_rows
    )

    summary.update(
        {
            "camera_height_m": (
                args.camera_height
            ),
            "camera_pitch_deg": (
                args.camera_pitch_deg
            ),
            "image_width_px": (
                args.image_width
            ),
            "image_height_px": (
                args.image_height
            ),
            "hfov_deg": (
                args.hfov_deg
            ),
            "focal_length_px": (
                focal_length
            ),
            "principal_y_px": (
                principal_y
            ),
            "geometric_horizon_y_px": (
                geometric_horizon_y
            ),
            "minimum_visible_ground_distance_m": (
                minimum_visible_distance
            ),
            "requested_min_distance_m": (
                args.min_distance
            ),
            "requested_max_distance_m": (
                args.max_distance
            ),
            "distance_step_m": (
                args.distance_step
            ),
        }
    )

    save_rows(
        summary_path,
        [summary],
    )

    print(
        "[IPM Ideal] Evaluation completed"
    )

    print(
        "[IPM Ideal] "
        f"camera_pitch="
        f"{args.camera_pitch_deg:.1f} deg"
    )

    print(
        "[IPM Ideal] "
        f"focal_length="
        f"{focal_length:.3f} px"
    )

    print(
        "[IPM Ideal] "
        f"geometric horizon row="
        f"{geometric_horizon_y:.3f} px"
    )

    print(
        "[IPM Ideal] "
        "minimum visible ground distance="
        f"{minimum_visible_distance:.3f} m"
    )

    print(
        "[IPM Ideal] "
        f"samples={summary['samples']}"
    )

    print(
        "[IPM Ideal] "
        f"MAE={summary['mae_m']:.6f} m, "
        f"RMSE={summary['rmse_m']:.6f} m, "
        f"MAPE={summary['mape_pct']:.4f}%"
    )

    print(
        "[IPM Ideal] "
        f"Mean signed error="
        f"{summary['mean_signed_error_m']:.6f} m"
    )

    print(
        "[IPM Ideal] "
        f"Maximum absolute error="
        f"{summary['max_absolute_error_m']:.6f} m"
    )

    print(
        f"[IPM Ideal] Samples: "
        f"{ideal_path}"
    )

    print(
        f"[IPM Ideal] Sensitivity: "
        f"{sensitivity_path}"
    )

    print(
        f"[IPM Ideal] Summary: "
        f"{summary_path}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate final pitched-camera IPM geometry "
            "and its sensitivity to vertical "
            "ground-contact pixel errors."
        )
    )

    parser.add_argument(
        "--camera-height",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--camera-pitch-deg",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--image-width",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--image-height",
        type=int,
        default=480,
    )

    parser.add_argument(
        "--hfov-deg",
        type=float,
        default=90.0,
    )

    parser.add_argument(
        "--min-distance",
        type=float,
        default=1.2,
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--distance-step",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/"
            "ipm_ideal_pitch15_final"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_test(
        parse_args()
    )