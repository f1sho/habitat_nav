import os
import csv
import time
import numpy as np

class PureNavigationEvaluator:
    def __init__(self, mode_name):
        self.mode_name = mode_name
        self.episode_logs = []
        self.current_episode_latencies = []

    def record_inference_time(self, latency_seconds):
        # Store latency for later FPS calculation
        self.current_episode_latencies.append(latency_seconds)

    def record_episode(self, episode_id, success, steps, shortest_dist, actual_dist):
        # Calculate SPL score
        if success:
            denominator = max(actual_dist, shortest_dist, 1e-5)
            spl = shortest_dist / denominator
        else:
            spl = 0.0
            
        avg_latency = np.mean(self.current_episode_latencies) if self.current_episode_latencies else 0.0
        fps = 1.0 / avg_latency if avg_latency > 0 else 0.0
        
        # Add clean navigation record
        self.episode_logs.append({
            "mode": self.mode_name,
            "episode": episode_id,
            "success": 1 if success else 0,
            "steps": steps,
            "optimal_distance": round(shortest_dist, 2),
            "actual_distance": round(actual_dist, 2),
            "spl": round(spl, 4),
            "fps": round(fps, 2)
        })
        
        self.current_episode_latencies.clear()

    def export_to_csv(self, output_dir="evaluate/evaluation_results"):
        # Export the logs to a spreadsheet
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if not self.episode_logs:
            print("No navigation data to export.")
            return None
            
        timestamp = time.strftime("%Y%m%d_%H%M")
        filename = f"nav_benchmark_{self.mode_name}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
            
        with open(filepath, mode="w", newline="") as csv_file:
            fieldnames = [
                "mode", "episode", "success", "steps", 
                "optimal_distance", "actual_distance", "spl", "fps"
            ]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.episode_logs)
            
        return filepath