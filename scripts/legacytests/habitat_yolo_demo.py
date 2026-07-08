import os
import time
import numpy as np
import cv2

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
from ultralytics import YOLO


SCENE_PATH = "/home/hannah/data/versioned_data/habitat_test_scenes/apartment_1.glb"
NAVMESH_PATH = "/home/hannah/data/versioned_data/habitat_test_scenes/apartment_1.navmesh"

OUTPUT_DIR = "yolo_loop_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_sim():
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE_PATH
    sim_cfg.enable_physics = False

    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "color_sensor"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [480, 640]
    rgb_sensor.position = [0.0, 1.5, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor]

    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward",
            habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left",
            habitat_sim.agent.ActuationSpec(amount=20.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right",
            habitat_sim.agent.ActuationSpec(amount=20.0)
        ),
    }

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    if os.path.exists(NAVMESH_PATH):
        sim.pathfinder.load_nav_mesh(NAVMESH_PATH)
        print("Loaded navmesh:", NAVMESH_PATH)

    return sim


def set_random_agent_state(sim):
    agent = sim.initialize_agent(0)
    start = sim.pathfinder.get_random_navigable_point()

    state = habitat_sim.AgentState()
    state.position = start

    yaw = np.random.uniform(0, 2 * np.pi)
    state.rotation = quat_from_angle_axis(yaw, np.array([0, 1, 0]))

    agent.set_state(state)
    print("Agent start position:", start)


def get_rgb(obs):
    rgb = obs["color_sensor"]
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    return rgb


def main():
    seed = int(time.time())
    np.random.seed(seed)

    sim = make_sim()
    set_random_agent_state(sim)

    model = YOLO("yolov8n.pt")

    actions = [
        "turn_left",
        "turn_left",
        "move_forward",
        "turn_right",
        "move_forward",
        "turn_left",
        "move_forward",
    ]

    for step_id, action in enumerate(actions):
        observations = sim.get_sensor_observations()
        rgb = get_rgb(observations)

        results = model(rgb, conf=0.25, verbose=False)
        result = results[0]

        detected = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            detected.append(f"{name}({conf:.2f})")

        print(f"Step {step_id}: action={action}")
        print("Detected:", detected if detected else "No objects")

        annotated = result.plot()
        out_path = os.path.join(
            OUTPUT_DIR,
            f"step_{step_id:02d}_{action}_{seed}.png"
        )
        cv2.imwrite(out_path, annotated)

        sim.step(action)

    print("Saved loop detection images to:", OUTPUT_DIR)
    sim.close()


if __name__ == "__main__":
    main()