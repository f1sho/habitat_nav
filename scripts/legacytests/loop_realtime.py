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

# ===== 5. 初始位置 =====
state = habitat_sim.AgentState()
state.position = sim.pathfinder.get_random_navigable_point()
agent.set_state(state)

print("Pathfinder loaded:", sim.pathfinder.is_loaded)
print("Initial position:", agent.get_state().position)

# ===== 6. 一个简单的“假控制器” =====
# 现在先用规则动作，后面这里会换成你的 CNN / planner
actions = ["move_forward", "move_forward", "turn_left", "move_forward", "turn_right"]

# ===== 7. 实时 loop =====
num_steps = 20

for step in range(num_steps):
    # 获取当前观测
    obs = sim.get_sensor_observations()
    rgb = obs["color_sensor"][:, :, :3]

    # ===== 未来这里可以接你的 CNN =====
    # 现在只是简单循环动作
    action = actions[step % len(actions)]

    # 执行动作
    sim.step(action)

    # 打印状态
    agent_state = agent.get_state()
    print(
        f"step={step:02d}, action={action}, "
        f"pos={agent_state.position}, rgb_min={rgb.min()}, rgb_max={rgb.max()}"
    )

    # 让输出别太快刷屏
    time.sleep(0.1)

print("Loop finished.")
sim.close()