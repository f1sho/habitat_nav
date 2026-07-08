import time
import habitat_sim
import numpy as np
from habitat_sim.utils.common import quat_from_angle_axis

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
    raise RuntimeError("Pathfinder not loaded.")

# ===== 5. start / goal =====
start = sim.pathfinder.get_random_navigable_point()
goal = sim.pathfinder.get_random_navigable_point()

state = habitat_sim.AgentState()
state.position = start
agent.set_state(state)

print("Start:", start)
print("Goal :", goal)

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def face_goal(agent, goal):
    state = agent.get_state()
    pos = np.array(state.position)
    goal = np.array(goal)

    direction = goal - pos
    direction[1] = 0.0

    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        return

    direction = direction / norm

    # 修正后的 yaw
    yaw = np.arctan2(-direction[0], -direction[2])

    state.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
    agent.set_state(state)

max_steps = 30
goal_threshold = 0.5

for step in range(max_steps):
    agent_state = agent.get_state()
    pos_before = np.array(agent_state.position)
    dist_before = distance(pos_before, goal)

    if dist_before < goal_threshold:
        print(f"[SUCCESS] reached goal at step {step}, dist={dist_before:.3f}")
        break

    # 先强制朝向 goal，再前进一步
    face_goal(agent, goal)
    sim.step("move_forward")

    pos_after = np.array(agent.get_state().position)
    dist_after = distance(pos_after, goal)

    print(
        f"step={step:02d}, "
        f"dist_before={dist_before:.3f}, dist_after={dist_after:.3f}, "
        f"pos_before={pos_before}, pos_after={pos_after}"
    )

    time.sleep(0.05)
else:
    print("[STOP] max steps reached.")

sim.close()