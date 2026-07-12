import csv
import os
from typing import Dict, Union

class Statistics:
    @staticmethod
    def _read_rows(csv_path):
        if not os.path.exists(csv_path):
            return []

        with open(csv_path, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            return list(reader)

    @staticmethod
    def _numeric_values(rows, key):
        values = []

        for row in rows:
            try:
                values.append(float(row[key]))
            except (KeyError, TypeError, ValueError):
                continue

        return values

    @staticmethod
    def summarize_csv(csv_path):
        """
        Generic mean summary for model-level and episode-level CSV files.
        """
        rows = Statistics._read_rows(csv_path)

        if not rows:
            return {}

        summary = {}

        for key in rows[0]:
            values = Statistics._numeric_values(rows, key)

            if values:
                summary[key] = round(
                    sum(values) / len(values),
                    3,
                )

        return summary

    @staticmethod
    def summarize_frame_csv(csv_path):
        """
        Summarize perception frame metrics.

        FPS is calculated as:
            total frames / total measured perception time

        It is intentionally not calculated as the arithmetic mean of
        per-frame FPS values.
        """
        rows = Statistics._read_rows(csv_path)

        if not rows:
            return {}

        latencies_ms = Statistics._numeric_values(
            rows,
            "latency_ms",
        )
        detections = Statistics._numeric_values(
            rows,
            "detections",
        )
        confidences = Statistics._numeric_values(
            rows,
            "avg_confidence",
        )

        summary: Dict[str, Union[int, float]] = {
            "frame_count": len(rows),
        }

        if latencies_ms:
            total_latency_ms = sum(latencies_ms)
            mean_latency_ms = (
                total_latency_ms / len(latencies_ms)
            )

            throughput_fps = (
                len(latencies_ms)
                / (total_latency_ms / 1000.0)
                if total_latency_ms > 0.0
                else 0.0
            )

            summary["mean_latency_ms"] = round(
                mean_latency_ms,
                3,
            )
            summary["fps"] = round(
                throughput_fps,
                3,
            )

        if detections:
            summary["mean_detections"] = round(
                sum(detections) / len(detections),
                3,
            )

        if confidences:
            summary["mean_confidence"] = round(
                sum(confidences) / len(confidences),
                3,
            )

        return summary
