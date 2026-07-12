import math

from core.habitat_env import HabitatEnv
from core.perception import PerceptionModule
from evaluation.ipm_accuracy_evaluator import (
    IPMAccuracyEvaluator,
)


SCENE_PATH = (
    "/home/hannah/data/replica_v1/"
    "apartment_2/habitat/mesh_semantic.ply"
)

NAVMESH_PATH = (
    "/home/hannah/data/replica_v1/"
    "apartment_2/habitat/mesh_semantic.navmesh"
)

MODEL_PATH = "yolo26n-seg.pt"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_HEIGHT = 1.5
CAMERA_HFOV_DEG = 90.0

FOCAL_LENGTH = IMAGE_WIDTH / (
    2.0
    * math.tan(
        math.radians(CAMERA_HFOV_DEG) / 2.0
    )
)


env = HabitatEnv(
    scene_path=SCENE_PATH,
    navmesh_path=NAVMESH_PATH,
    enable_depth=True,
)

perception = PerceptionModule(
    model_path=MODEL_PATH,
    camera_height=CAMERA_HEIGHT,
    focal_length=FOCAL_LENGTH,
    img_height=IMAGE_HEIGHT,
    device=0,
)

ipm_evaluator = IPMAccuracyEvaluator(
    model_name=MODEL_PATH,
    camera_height=CAMERA_HEIGHT,
    focal_length=FOCAL_LENGTH,
    image_height=IMAGE_HEIGHT,
    bottom_band_height=5,
    contact_height_threshold=0.15,
)

observations = env.get_observations()

rgb_frame = observations["color_sensor"][..., :3]
depth_frame = observations["depth_sensor"]

detections, perception_metrics = perception.process_frame(
    rgb_frame
)

print(
    f"Calculated focal length: "
    f"{FOCAL_LENGTH:.3f} px"
)

print(
    f"Detection count: "
    f"{len(detections)}"
)

ipm_evaluator.update_frame(
    episode=1,
    step=0,
    detections=detections,
    depth_frame=depth_frame,
)

summary = ipm_evaluator.save()

print("Summary:", summary)

env.close()