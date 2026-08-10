from pathlib import Path

import numpy as np

from core.habitat_env import HabitatEnv
from run_navigation import _sample_valid_route


REPLICA_ROOT = Path("/home/hannah/data/replica_v1")

SCENES = [
    "apartment_0",
    "apartment_1",
    "apartment_2",
    "frl_apartment_0",
    "frl_apartment_1",
    "frl_apartment_2",
    "frl_apartment_3",
    "frl_apartment_4",
    "frl_apartment_5",
    "hotel_0",
    "office_0",
    "office_1",
    "office_2",
    "office_3",
    "office_4",
    "room_0",
    "room_1",
    "room_2",
]

MIN_DISTANCE = 8.0
MAX_ROUTE_ATTEMPTS = 500

# Test exactly the same 20 seeds planned for the final evaluation.
SEEDS = list(range(1000, 1020))


def compute_path_length(waypoints):
    """Compute the total length of a waypoint path."""
    if waypoints is None or len(waypoints) < 2:
        return 0.0

    total_length = 0.0

    for i in range(1, len(waypoints)):
        p0 = np.asarray(waypoints[i - 1], dtype=np.float32)
        p1 = np.asarray(waypoints[i], dtype=np.float32)
        total_length += float(np.linalg.norm(p1 - p0))

    return total_length


def check_scene(scene_name):
    """Check route feasibility for one Replica scene."""
    scene_path = (
        REPLICA_ROOT
        / scene_name
        / "habitat"
        / "mesh_semantic.ply"
    )

    navmesh_path = (
        REPLICA_ROOT
        / scene_name
        / "habitat"
        / "mesh_semantic.navmesh"
    )

    if not scene_path.exists():
        return {
            "scene": scene_name,
            "status": "missing scene",
        }

    if not navmesh_path.exists():
        return {
            "scene": scene_name,
            "status": "missing navmesh",
        }

    env = None

    try:
        env = HabitatEnv(
            str(scene_path),
            str(navmesh_path),
            enable_depth=False,
        )

        successful_seeds = []
        failed_seeds = []
        straight_distances = []
        route_lengths = []

        for seed in SEEDS:
            env.sim.seed(seed)
            env.sim.pathfinder.seed(seed)

            try:
                (
                    start_position,
                    goal_position,
                    waypoints,
                    _global_planner,
                ) = _sample_valid_route(
                    env=env,
                    min_start_goal_distance=MIN_DISTANCE,
                    max_route_attempts=MAX_ROUTE_ATTEMPTS,
                )

                straight_distance = float(
                    np.linalg.norm(
                        np.asarray(goal_position)
                        - np.asarray(start_position)
                    )
                )

                route_length = compute_path_length(waypoints)

                successful_seeds.append(seed)
                straight_distances.append(straight_distance)
                route_lengths.append(route_length)

            except RuntimeError:
                failed_seeds.append(seed)

        success_count = len(successful_seeds)

        result = {
            "scene": scene_name,
            "status": "ok",
            "success_count": success_count,
            "failed_count": len(failed_seeds),
            "successful_seeds": successful_seeds,
            "failed_seeds": failed_seeds,
        }

        if straight_distances:
            result["mean_straight_distance"] = float(
                np.mean(straight_distances)
            )
            result["mean_route_length"] = float(
                np.mean(route_lengths)
            )
        else:
            result["mean_straight_distance"] = 0.0
            result["mean_route_length"] = 0.0

        return result

    except Exception as exc:
        return {
            "scene": scene_name,
            "status": f"error: {exc}",
        }

    finally:
        if env is not None:
            env.close()


def main():
    print()
    print(
        f"{'Scene':<20}"
        f"{'Routes':>10}"
        f"{'Failed':>10}"
        f"{'Mean direct':>14}"
        f"{'Mean route':>14}"
    )
    print("-" * 68)

    results = []

    for scene_name in SCENES:
        result = check_scene(scene_name)
        results.append(result)

        if result["status"] != "ok":
            print(
                f"{scene_name:<20}"
                f"{'-':>10}"
                f"{'-':>10}"
                f"{'-':>14}"
                f"{result['status']:>14}"
            )
            continue

        print(
            f"{scene_name:<20}"
            f"{result['success_count']:>7}/20"
            f"{result['failed_count']:>10}"
            f"{result['mean_straight_distance']:>13.2f}m"
            f"{result['mean_route_length']:>13.2f}m"
        )

    print()
    print("Recommended candidates:")
    print("-" * 68)

    candidates = [
        result
        for result in results
        if (
            result["status"] == "ok"
            and result["success_count"] == len(SEEDS)
        )
    ]

    if not candidates:
        print(
            "No scene successfully generated routes for all 20 seeds."
        )
    else:
        for result in candidates:
            print(
                f"{result['scene']}: "
                f"{result['success_count']}/20 routes, "
                f"mean direct distance="
                f"{result['mean_straight_distance']:.2f} m, "
                f"mean route length="
                f"{result['mean_route_length']:.2f} m"
            )

    print()

    for result in results:
        if (
            result.get("status") == "ok"
            and result.get("failed_seeds")
        ):
            print(
                f"{result['scene']} failed seeds: "
                f"{result['failed_seeds']}"
            )


if __name__ == "__main__":
    main()