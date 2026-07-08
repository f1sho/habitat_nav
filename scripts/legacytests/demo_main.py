import numpy as np
import time
from core.habitat_env import HabitatEnv
from core.planning.global_planner import GlobalPlanner
from core.perception import PerceptionModule
from core.planning.local_planner import DiscreteDWAPlanner
from utils.visualizer import DemoVisualizer
# import math
# import quaternion

def main():
    scene_path = "/home/hannah/data/versioned_data/habitat_test_scenes/apartment_1.glb"
    navmesh_path = "/home/hannah/data/versioned_data/habitat_test_scenes/apartment_1.navmesh"

    print("--- Initializing Modules ---")
    env = HabitatEnv(scene_path, navmesh_path)

    # Lock the seed to ensure the map and spawn points are fixed
    # seed = 42 
    # env.sim.seed(seed)
    # env.sim.pathfinder.seed(seed)

    # Generate a dynamic seed based on the current system time
    dynamic_seed = int(time.time())
    env.sim.seed(dynamic_seed)
    env.sim.pathfinder.seed(dynamic_seed)

    # Define coordinate points earlier so we can use the height for the global planner
    # start_pos = np.array([5.55664, -1.60025, -0.620963])
    # goal_pos = np.array([3.58175, -1.60025, 6.7737])
    start_pos = env.sim.pathfinder.get_random_navigable_point()
    goal_pos = env.sim.pathfinder.get_random_navigable_point()

    # Pass the exact starting height start_pos[1] to keep the top down map slice consistent
    global_planner = GlobalPlanner(env.sim.pathfinder, map_height=start_pos[1])
    
    # --- MODIFICATION START ---
    # Context: Initialization of the perception module
    # Changes: Removed conf_threshold to match the updated PerceptionModule signature
    SIM_CAMERA_HEIGHT = 1.5
    SIM_FOCAL_LENGTH = 800.0
    SIM_IMG_HEIGHT = 480
    
    perception = PerceptionModule(
        model_path="yolov8n-seg.pt", 
        camera_height=SIM_CAMERA_HEIGHT,
        focal_length=SIM_FOCAL_LENGTH,
        img_height=SIM_IMG_HEIGHT
    )
    # --- MODIFICATION END ---

    local_planner = DiscreteDWAPlanner(safe_distance=0.5, semantic_safe_distance=0.8)
    visualizer = DemoVisualizer()
    
    print("\n--- Starting Navigation Loop ---")

    print(f"Start: {start_pos} | Goal: {goal_pos}")
    waypoints = global_planner.plan_path(start_pos, goal_pos)
    
    if not waypoints:
        print("Failed to generate global path.")
        return

    # Generate visualized path
    global_planner.visualize_path(start_pos, goal_pos, waypoints)

    # Get the proper agent object and update its state inside the C++ simulator engine
    agent = env.sim.get_agent(0)
    agent_state = agent.get_state()
    agent_state.position = start_pos

    # # --- START OF MODIFICATION ---
    # # Construct a corner case Force the agent to spawn facing away from the first waypoint
    # if len(waypoints) > 1:
    #     target_wp = waypoints[1]
    #     dir_vec = target_wp - start_pos
        
    #     # Calculate the angle to the target then add pi to face exactly backwards
    #     yaw = math.atan2(dir_vec[0], dir_vec[2]) + math.pi 
        
    #     # Override the initial rotation
    #     agent_state.rotation = quaternion.from_euler_angles(0, yaw, 0)
    # # --- END OF MODIFICATION ---

    agent.set_state(agent_state)
    
    current_wp_idx = 1 
    max_steps = 200
    
    try:
        for step in range(max_steps):
            
            target_wp = waypoints[current_wp_idx]
            
            if np.linalg.norm(agent.state.position - target_wp) < 0.25:
                print(f"Reached waypoint {current_wp_idx}. Moving to next.")
                current_wp_idx += 1
                if current_wp_idx >= len(waypoints):
                    # Add the print statement before breaking out of the loop
                    print("Goal Reached!")
                    break
                target_wp = waypoints[current_wp_idx]

            obs = env.get_observations()
            rgb_frame = obs["color_sensor"]
            
            # --- MODIFICATION START ---
            # Context: Preprocessing the RGB frame before inference
            # Changes: Sliced the array to keep only the first 3 channels because Habitat outputs RGBA and YOLO expects RGB
            rgb_frame = rgb_frame[..., :3]
            # --- MODIFICATION END ---
            
            depth_frame = obs["depth_sensor"]

            # --- MODIFICATION START ---
            # Context: Processing the RGB frame to get semantic detections
            # Changes: Updated the method call from process_rgb to process_frame to match the updated PerceptionModule
            detections = perception.process_frame(rgb_frame)
            # --- MODIFICATION END ---

            action = local_planner.get_best_action(depth_frame, detections, agent.state, target_wp)
            dist_to_wp = np.linalg.norm(agent.state.position - target_wp)
            print(f"Step {step}: Distance to WP={dist_to_wp:.2f}m | Action={action}")

            visualizer.show_frame(rgb_frame, detections, action, step, dist_to_wp)

            env.step(action)
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n[Manual Stop] User interrupted the script.")
    finally:
        print("\nSaving video and cleaning up...")
        visualizer.close()
        env.close()

if __name__ == "__main__":
    main()