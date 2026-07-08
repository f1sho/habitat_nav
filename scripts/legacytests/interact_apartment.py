import habitat_sim
import cv2
import numpy as np

def main():
    scene_path = "/home/hannah/data/versioned_data/habitat_test_scenes/apartment_1.glb"
    
    # Configure Simulator
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    
    # Setup RGB Sensor
    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "color_sensor"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [480, 640]
    rgb_sensor.position = [0.0, 1.5, 0.0]
    
    # Setup Depth Sensor (Crucial for understanding obstacles)
    depth_sensor = habitat_sim.CameraSensorSpec()
    depth_sensor.uuid = "depth_sensor"
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor.resolution = [480, 640]
    depth_sensor.position = [0.0, 1.5, 0.0]
    
    agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]
    
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    
    # Teleport agent to the specific crash site from the video
    agent = sim.agents[0]
    start_pos = np.array([5.55664, -1.60025, -0.620963])
    agent.state.position = start_pos
    
    print("\n--- Interactive Explorer Started ---")
    print("Controls:")
    print("  [W] : Move Forward")
    print("  [A] : Turn Left")
    print("  [D] : Turn Right")
    print("  [Q] : Quit")
    print("------------------------------------\n")

    # Create resizable windows
    cv2.namedWindow("Habitat Explorer (RGB)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Habitat Explorer (Depth)", cv2.WINDOW_NORMAL)

    while True:
        obs = sim.get_sensor_observations()
        
        # Process RGB Image
        rgb = obs["color_sensor"]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        
        # Process Depth Image (Normalize to 0-5 meters for visualization)
        depth = obs["depth_sensor"]
        depth_vis = np.clip(depth, 0.0, 5.0) / 5.0 
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        # Show frames
        cv2.imshow("Habitat Explorer (RGB)", bgr)
        cv2.imshow("Habitat Explorer (Depth)", depth_vis)
        
        # Wait for user input (0 means wait indefinitely)
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('w'):
            sim.step("move_forward")
            print("Action: move_forward")
        elif key == ord('a'):
            sim.step("turn_left")
            print("Action: turn_left")
        elif key == ord('d'):
            sim.step("turn_right")
            print("Action: turn_right")
        elif key == ord('q'):
            print("Exiting Explorer...")
            break
            
    sim.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()