import numpy as np
import time
from core.habitat_env import HabitatEnv
from core.planning.global_planner import GlobalPlanner
from core.dual_perception import DualPerceptionModule
from core.planning.local_planner import DiscreteDWAPlanner
from utils.visualizer import DemoVisualizer
from utils.trajectory_plotter import plot_topdown_trajectory

def main():
    scene_path = "/home/hannah/data/replica_v1/apartment_2/habitat/mesh_semantic.ply"
    navmesh_path = "/home/hannah/data/replica_v1/apartment_2/habitat/mesh_semantic.navmesh"

    print("--- Initializing Modules ---")
    env = HabitatEnv(scene_path, navmesh_path)

    # Use a dynamic seed to allow searching for new long routes
    dynamic_seed = int(time.time())
    env.sim.seed(dynamic_seed)
    env.sim.pathfinder.seed(dynamic_seed)

    print("Searching for a valid long distance route...")
    start_pos = env.sim.pathfinder.get_random_navigable_point()
    goal_pos = env.sim.pathfinder.get_random_navigable_point()
    
    # Loop until the straight line distance is at least 8.0 meters
    while np.linalg.norm(start_pos - goal_pos) < 8.0:
        start_pos = env.sim.pathfinder.get_random_navigable_point()
        goal_pos = env.sim.pathfinder.get_random_navigable_point()

    # Pass the exact starting height to keep the top down map slice consistent
    global_planner = GlobalPlanner(env.sim.pathfinder, map_height=start_pos[1])
    
    SIM_CAMERA_HEIGHT = 1.5
    SIM_FOCAL_LENGTH = 800.0
    SIM_IMG_HEIGHT = 480
    
    perception = DualPerceptionModule(
        yolo_path="yolo26n-seg.pt", 
        conf_threshold=0.5
    )

    # Reduce the semantic safe distance to prevent oscillation in narrow indoor environments
    local_planner = DiscreteDWAPlanner(safe_distance=0.15, semantic_safe_distance=0.4)

    visualizer = DemoVisualizer()
    
    print("\n--- Starting Navigation Loop ---")

    print(f"Start: {start_pos} | Goal: {goal_pos}")
    waypoints = global_planner.plan_path(start_pos, goal_pos)
    
    if not waypoints:
        print("Failed to generate global path.")
        return

    # Generate visualised path
    global_planner.visualize_path(start_pos, goal_pos, waypoints)

    # Get the proper agent object and update its state inside the C++ simulator engine
    agent = env.sim.get_agent(0)
    agent_state = agent.get_state()
    agent_state.position = start_pos

    agent.set_state(agent_state)
    
    current_wp_idx = 1 
    max_steps = 800  

    actual_trajectory = []
    
    try:
        for step in range(max_steps):
            
            target_wp = waypoints[current_wp_idx]

            # Fetch real-time state from the engine and use it consistently for all calculations
            current_state = agent.get_state()
            current_agent_position = current_state.position.copy()
            actual_trajectory.append(current_agent_position)
            
            pos_2d = current_state.position[[0, 2]]
            wp_2d = target_wp[[0, 2]]
            
            if np.linalg.norm(pos_2d - wp_2d) < 0.3:
                print(f"Reached waypoint {current_wp_idx}. Moving to next.")
                current_wp_idx += 1
                if current_wp_idx >= len(waypoints):
                    print("Goal Reached!")
                    break
                target_wp = waypoints[current_wp_idx]

            obs = env.get_observations()
            rgb_frame = obs["color_sensor"]
            
            # Slice the array to keep only the first 3 channels
            rgb_frame = rgb_frame[..., :3]
            
            # Retrieve both detections and the real predicted depth frame
            detections, depth_frame, *_ = perception.process_frame(rgb_frame)

            action = local_planner.get_best_action(depth_frame, detections, current_state, target_wp)
            dist_to_wp = np.linalg.norm(current_state.position[[0, 2]] - target_wp[[0, 2]])
            print(f"Step {step}: Distance to WP={dist_to_wp:.2f}m | Action={action}")

            visualizer.show_frame(rgb_frame, detections, action, step, dist_to_wp)

            env.step(action)
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n[Manual Stop] User interrupted the script.")
    finally:
        print("\nSaving video and cleaning up...")
        visualizer.close()
        plot_topdown_trajectory(env, start_pos, goal_pos, waypoints, actual_trajectory)
        env.close()

if __name__ == "__main__":
    main()