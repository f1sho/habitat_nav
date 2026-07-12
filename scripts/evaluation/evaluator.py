import os

from .logger import EvaluationLogger
from .statistics import Statistics


class Evaluator:
    def __init__(self, model_name, save_dir="results"):
        self.logger = EvaluationLogger(
            model_name=model_name,
            save_root=save_dir,
        )

        self.episode = 0

    def start_episode(self):
        self.episode += 1

    def update_frame(self, step, metrics):
        self.logger.log_frame(
            episode=self.episode,
            step=step,
            **metrics,
        )

    def finish_episode(self, nav_metrics):
        """
        nav_metrics : NavigationMetrics object
        """
        self.logger.log_episode(
            episode=self.episode,
            **nav_metrics.summary(),
        )

    def log_model(self, model_metrics):
        """
        model_metrics : ModelMetrics object
        """
        self.logger.log_model(
            **model_metrics.summary(),
        )

    def print_summary(self):
        print("\n========== Evaluation Summary ==========")

        # ---------------- Model ----------------

        model_csv = os.path.join(
            self.logger.save_dir,
            "model_metrics.csv",
        )

        print("\n[Model]")

        model_summary = Statistics.summarize_csv(model_csv)

        for key, value in model_summary.items():
            print(f"{key:20}: {value}")

        # ---------------- Perception ----------------

        frame_csv = os.path.join(
            self.logger.save_dir,
            "frame_metrics.csv",
        )

        print("\n[Perception]")

        # Use a dedicated perception summary. FPS is calculated from
        # total frames / total latency, not mean(frame-level FPS).
        frame_summary = Statistics.summarize_frame_csv(
            frame_csv
        )

        for key, value in frame_summary.items():
            print(f"{key:20}: {value}")

        # ---------------- Navigation ----------------

        episode_csv = os.path.join(
            self.logger.save_dir,
            "episode_metrics.csv",
        )

        print("\n[Navigation]")

        episode_summary = Statistics.summarize_csv(
            episode_csv
        )

        for key, value in episode_summary.items():
            print(f"{key:20}: {value}")

        print("\n========================================")

    def save(self):
        self.logger.save()
        self.print_summary()

    def reset(self):
        self.episode = 0
        self.logger.reset()
