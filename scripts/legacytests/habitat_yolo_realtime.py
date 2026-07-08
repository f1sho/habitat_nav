import os
import time
import numpy as np
import cv2

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis
from ultralytics import YOLO


SCENE_PATH = "/home/hannah/data/versioned_data/habitat_test_scenes/apartment_1.glb"
NAVMESH_PATH = "/home/hannah/data/versioned_data/habitat_test_scenes/apartment_1.navmesh"


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
            habitat_sim.agent.ActuationSpec(amount=0.08)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left",
            habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right",
            habitat_sim.agent.ActuationSpec(amount=5.0)
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
    return agent


def get_rgb(observations):
    rgb = observations["color_sensor"]

    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    return rgb


def choose_demo_action(step_id):
    # Simple demo motion pattern, only for showing real-time perception
    pattern = [
        "turn_left",
        "turn_left",
        "move_forward",
        "move_forward",
        "turn_right",
        "move_forward",
        "turn_left",
        "move_forward",
    ]

    return pattern[step_id % len(pattern)]


def main():
    seed = int(time.time())
    np.random.seed(seed)

    sim = make_sim()
    set_random_agent_state(sim)

    model = YOLO("yolov8n.pt")

    cv2.namedWindow("Habitat YOLO Real-time", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Habitat YOLO Real-time", 960, 720)

    step_id = 0

    print("Real-time window started.")
    print("Press q to quit.")
    print("Press s to save current frame.")

    while True:
        observations = sim.get_sensor_observations()
        rgb = get_rgb(observations)

        # YOLO inference
        results = model(rgb, conf=0.25, verbose=False)
        result = results[0]

        # result.plot() returns BGR image, suitable for OpenCV display
        annotated = result.plot()

        detected = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls_id]
            detected.append(f"{name} {conf:.2f}")

        action = choose_demo_action(step_id)

        # Add text overlay
        cv2.putText(
            annotated,
            f"Step: {step_id} | Action: {action}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if detected:
            text = "Detected: " + ", ".join(detected[:5])
        else:
            text = "Detected: none"

        cv2.putText(
            annotated,
            text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Habitat YOLO Real-time", annotated)

        key = cv2.waitKey(100) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            filename = f"realtime_yolo_step_{step_id}.png"
            cv2.imwrite(filename, annotated)
            print("Saved:", filename)

        sim.step(action)
        step_id += 1

    sim.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()