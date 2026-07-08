import habitat_sim
import numpy as np
import cv2
import os

# ===== 路径处理 =====
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "..", "data", "rgb")
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, "frame_0001.png")

# ===== 1. 创建模拟器配置 =====
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = "/home/hannah/data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"

# ===== 2. 相机传感器 =====
sensor_cfg = habitat_sim.CameraSensorSpec()
sensor_cfg.uuid = "color_sensor"
sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
sensor_cfg.resolution = [480, 640]
sensor_cfg.position = [0.0, 1.5, 0.0]

# ===== 3. agent配置 =====
agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = [sensor_cfg]

# ===== 4. 创建 simulator =====
sim = habitat_sim.Simulator(
    habitat_sim.Configuration(sim_cfg, [agent_cfg])
)

# ===== 5. 获取一帧 RGB =====
obs = sim.get_sensor_observations()
rgb = obs["color_sensor"]

# ===== 6. 保存图像 =====
rgb = rgb[:, :, :3]
rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

cv2.imwrite(save_path, rgb)

print(f"Saved image at: {save_path}")