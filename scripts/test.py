from evaluation.logger import EvaluationLogger

logger = EvaluationLogger()

logger.log_frame(
    episode=1,
    step=1,
    latency_ms=15.8,
    fps=63.2,
    detections=6,
    avg_confidence=0.84
)

logger.log_episode(
    episode=1,
    success=True,
    spl=0.82,
    nav_time=20.4,
    path_length=10.6,
    collision_count=1,
    collision_rate=0.02
)

logger.save()