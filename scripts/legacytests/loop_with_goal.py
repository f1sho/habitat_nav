import time
import habitat_sim
import numpy as np

scene_path = "/home/hannah/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"

# ===== 1. Simulator 配置 =====
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = scene_path
sim_cfg.enable_physics = False

# ===== 2. RGB 相机 =====
sensor_cfg = habitat_sim.CameraSensorSpec()
sensor_cfg.uuid = "color_sensor"
sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
sensor_cfg.resolution = [480, 640]
sensor_cfg.position = [0.0, 1.5, 0.0]

# ===== 3. Agent 配置 =====
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

# ===== 4. 创建 Simulator =====
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
agent = sim.initialize_agent(0)

print("Pathfinder loaded:", sim.pathfinder.is_loaded)

if not sim.pathfinder.is_loaded:
    raise RuntimeError("Pathfinder not loaded. Cannot do goal navigation.")

# ===== 5. 设置 start 和 goal =====
start = sim.pathfinder.get_random_navigable_point()
goal = sim.pathfinder.get_random_navigable_point()

state = habitat_sim.AgentState()
state.position = start
agent.set_state(state)

print("Start:", start)
print("Goal :", goal)

# ===== 工具函数 =====
def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n

def quat_to_forward_vector(rotation):
    rot_matrix = habitat_sim.utils.common.quat_to_magnum(rotation).to_matrix()
    rot_np = np.array(rot_matrix)
    forward = rot_np @ np.array([0.0, 0.0, 1.0])
    return normalize(forward)

def signed_angle_between_2d(v1, v2):
    a = np.array([v1[0], v1[2]])
    b = np.array([v2[0], v2[2]])

    a = normalize(a)
    b = normalize(b)

    dot = np.clip(np.dot(a, b), -1.0, 1.0)
    angle = np.arccos(dot)

    cross = a[0] * b[1] - a[1] * b[0]
    if cross < 0:
        angle = -angle
    return angle

# ===== 6. 主循环 =====
max_steps = 100
goal_threshold = 0.5
turn_threshold = np.deg2rad(10.0)

for step in range(max_steps):
    obs = sim.get_sensor_observations()
    rgb = obs["color_sensor"][:, :, :3]

    agent_state = agent.get_state()
    pos = agent_state.position
    rot = agent_state.rotation

    to_goal = goal - pos
    dist_to_goal = np.linalg.norm(to_goal)

    if dist_to_goal < goal_threshold:
        print(f"[SUCCESS] Reached goal at step {step}, distance={dist_to_goal:.3f}")
        break

    forward = quat_to_forward_vector(rot)
    angle_to_goal = signed_angle_between_2d(forward, to_goal)
    angle_deg = np.rad2deg(angle_to_goal)

    # ===== 关键修正逻辑 =====

    if abs(angle_deg) > 20:
        action = "turn_left" if angle_deg > 0 else "turn_right"
    else:
        action = "move_forward"

    sim.step(action)

    print(
        f"step={step:03d}, action={action:12s}, "
        f"dist={dist_to_goal:.3f}, angle_deg={angle_deg:7.2f}, "
        f"pos={pos}, rgb_min={rgb.min()}, rgb_max={rgb.max()}"
    )

    time.sleep(0.05)
else:
    print("[STOP] Max steps reached without reaching goal.")

sim.close()