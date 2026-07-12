import time


class PerceptionMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_frames = 0
        self.total_latency = 0.0

        self.total_detections = 0
        self.total_confidence = 0.0

    def start_timer(self):
        self._start_time = time.perf_counter()

    def stop_timer(self):
        latency = (time.perf_counter() - self._start_time) * 1000.0  # ms

        self.total_frames += 1
        self.total_latency += latency

        return latency

    def update_detection(self, detections):
        self.total_detections += len(detections)

        for det in detections:
            self.total_confidence += det["confidence"]

    @property
    def avg_latency(self):
        if self.total_frames == 0:
            return 0.0
        return self.total_latency / self.total_frames

    @property
    def fps(self):
        if self.total_latency == 0:
            return 0.0
        return self.total_frames / (self.total_latency / 1000.0)

    @property
    def avg_detection(self):
        if self.total_frames == 0:
            return 0.0
        return self.total_detections / self.total_frames

    @property
    def avg_confidence(self):
        if self.total_detections == 0:
            return 0.0
        return self.total_confidence / self.total_detections