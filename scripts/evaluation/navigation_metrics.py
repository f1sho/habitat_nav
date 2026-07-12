from __future__ import annotations

import time
from typing import Dict, Optional, Union

import numpy as np


MetricValue = Union[int, float]


class NavigationMetrics:
    """
    Episode-level navigation metrics.

    Collision semantics
    -------------------
    collision_count counts simulator actions for which Habitat-Sim returns
    collided=True. Repeated attempts to move into the same obstacle are counted
    as repeated collision actions.

    collision_rate is:
        collision_count / action_count
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.start_position: Optional[np.ndarray] = None
        self.previous_position: Optional[np.ndarray] = None
        self.shortest_path = 0.0

        self.path_length = 0.0
        self.step_count = 0
        self.action_count = 0

        self.collision_count = 0
        self.collision_rate = 0.0

        self.success = False
        self.spl = 0.0
        self.navigation_time = 0.0

        self._start_time: Optional[float] = None
        self._finished = False

    def start_episode(
        self,
        start_position: np.ndarray,
        shortest_path: float,
    ) -> None:
        """
        Start a new episode and reset all accumulated metrics.
        """
        self.reset()

        start = np.asarray(
            start_position,
            dtype=np.float64,
        ).copy()

        self.start_position = start
        self.previous_position = start
        self.shortest_path = max(
            float(shortest_path),
            0.0,
        )

        self._start_time = time.perf_counter()

    def update(self, current_position: np.ndarray) -> None:
        """
        Record one sampled agent position and accumulate travelled distance.

        run_navigation.py calls this once per navigation-loop iteration.
        """
        if self._start_time is None:
            raise RuntimeError(
                "start_episode() must be called before update()."
            )

        position = np.asarray(
            current_position,
            dtype=np.float64,
        ).copy()

        if self.previous_position is not None:
            travelled_distance = float(
                np.linalg.norm(
                    position - self.previous_position
                )
            )

            if np.isfinite(travelled_distance):
                self.path_length += travelled_distance

        self.previous_position = position
        self.step_count += 1

    def update_collision(self, collided: bool) -> None:
        """
        Record the collision result of one executed simulator action.
        """
        if self._start_time is None:
            raise RuntimeError(
                "start_episode() must be called before "
                "update_collision()."
            )

        self.action_count += 1

        if bool(collided):
            self.collision_count += 1

    # Optional clearer alias for future code.
    record_action = update_collision

    def finish_episode(self, success: bool) -> None:
        """
        Finalize success, SPL, navigation time, and collision rate.
        """
        if self._finished:
            return

        self.success = bool(success)

        if self._start_time is not None:
            self.navigation_time = (
                time.perf_counter() - self._start_time
            )

        if self.success and self.shortest_path > 0.0:
            spl_denominator = max(
                self.path_length,
                self.shortest_path,
            )
            self.spl = (
                self.shortest_path / spl_denominator
                if spl_denominator > 0.0
                else 0.0
            )
        else:
            self.spl = 0.0

        self.collision_rate = (
            self.collision_count / self.action_count
            if self.action_count > 0
            else 0.0
        )

        self._finished = True

    def summary(self) -> Dict[str, MetricValue]:
        """
        Return metrics in the format expected by EvaluationLogger.
        """
        return {
            "success": int(self.success),
            "spl": round(float(self.spl), 3),
            "nav_time": round(
                float(self.navigation_time),
                3,
            ),
            "path_length": round(
                float(self.path_length),
                3,
            ),
            "collision_count": int(
                self.collision_count
            ),
            "collision_rate": round(
                float(self.collision_rate),
                4,
            ),
            "step_count": int(self.step_count),
            "action_count": int(self.action_count),
        }
