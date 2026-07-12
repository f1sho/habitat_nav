import csv
import os


class EvaluationLogger:
    def __init__(self, model_name="default", save_root="results"):
        self.model_name = self._sanitize_name(model_name)
        self.save_dir = os.path.join(save_root, self.model_name)

        self.reset()

    def reset(self):
        self.frame_logs = []
        self.episode_logs = []
        self.model_logs = []

    def log_frame(self, **kwargs):
        self.frame_logs.append(kwargs)

    def log_episode(self, **kwargs):
        self.episode_logs.append(kwargs)

    def log_model(self, **kwargs):
        self.model_logs.append(kwargs)

    def _sanitize_name(self, name):
        """
        Convert model filename into a valid folder name.

        Example:
            yolo26n-seg.pt
                ↓
            yolo26n_seg_pt
        """
        name = os.path.basename(name)
        name = name.replace(".", "_")
        name = name.replace("-", "_")
        return name

    def _save_csv(self, filename, data):
        if not data:
            return

        os.makedirs(self.save_dir, exist_ok=True)

        filepath = os.path.join(self.save_dir, filename)

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    def save(self):
        self._save_csv("model_metrics.csv", self.model_logs)
        self._save_csv("frame_metrics.csv", self.frame_logs)
        self._save_csv("episode_metrics.csv", self.episode_logs)

        print(f"[Logger] Results saved to: {self.save_dir}")