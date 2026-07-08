import os
import cv2
import habitat_sim
import numpy as np

# ===== 路径 =====
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "..", "data", "rgb")
os.makedirs(output_dir, exist_ok=True)

scene_path = "/home/hannah/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"

# ===== 模拟器配置 =====
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = scene_path
sim_cfg.enable_physics = False

# ===== 传感器配置 =====
sensor_cfg = habitat_sim.CameraSensorSpec()
sensor_cfg.uuid = "color_sensor"
sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
sensor_cfg.resolution = [480, 640]
sensor_cfg.position = [0.0, 1.5, 0.0]

# ===== Agent 配置 =====
agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [sensor_cfg]
agent_cfg.action_space = {
    "move_forward": habitat_sim.agent.ActionSpec(
        "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
    ),
    "turn_left": habitat_sim.agent.ActionSpec(
        "turn_left", habitat_sim.agent.ActuationSpec(amount=15.0)
    ),
    "turn_right": habitat_sim.agent.ActionSpec(
        "turn_right", habitat_sim.agent.ActuationSpec(amount=15.0)
    ),
}

# ===== 创建 Simulator =====
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
agent = sim.initialize_agent(0)

# ===== 检查 navmesh/pathfinder =====
print("Pathfinder loaded:", sim.pathfinder.is_loaded)

# ===== 设置初始状态 =====
state = habitat_sim.AgentState()

if sim.pathfinder.is_loaded:
    state.position = sim.pathfinder.get_random_navigable_point()
else:
    print("Warning: pathfinder not loaded, fallback to origin")
    state.position = np.array([0.0, 0.0, 0.0])

agent.set_state(state)
print("Initial position:", agent.get_state().position)

# ===== 简单动作序列 =====
actions = [
    "move_forward",
    "turn_left",
    "move_forward",
]

print("Start stepping...")

for i, action in enumerate(actions):
    obs = sim.step(action)

    rgb = obs["color_sensor"][:, :, :3]
    print(f"[{i}] RGB min/max: {rgb.min()} / {rgb.max()}")

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    save_path = os.path.join(output_dir, f"frame_{i:04d}.png")
    cv2.imwrite(save_path, rgb_bgr)
    print(f"Saved: {save_path}")

    agent_state = agent.get_state()
    print(f"[{i}] action={action}, position={agent_state.position}")

print("Done.")
sim.close()