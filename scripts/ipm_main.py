import numpy as np
import time
import math
from core.habitat_env import HabitatEnv
from core.planning.global_planner import GlobalPlanner
from core.perception import PerceptionModule
from core.planning.local_planner import DiscreteDWAPlanner
from utils.visualizer import DemoVisualizer
from utils.trajectory_plotter import plot_topdown_trajectory
from evaluation.navigation_metrics import NavigationMetrics
from evaluation.evaluator import Evaluator
from evaluation.model_metrics import ModelMetrics

def main():
    MODEL_PATH = "yolo26n-seg.onnx"
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

    # Pass the exact starting height start_pos[1] to keep the top down map slice consistent
    global_planner = GlobalPlanner(env.sim.pathfinder, map_height=start_pos[1])
    
    SIM_CAMERA_HEIGHT = 1.5
    SIM_IMAGE_WIDTH = 640
    SIM_IMAGE_HEIGHT = 480
    SIM_HFOV_DEG = 90.0

    SIM_FOCAL_LENGTH = SIM_IMAGE_WIDTH / (
        2.0
        * math.tan(
            math.radians(SIM_HFOV_DEG) / 2.0
        )
    )
    
    perception = PerceptionModule(
        model_path=MODEL_PATH,
        camera_height=SIM_CAMERA_HEIGHT,
        focal_length=SIM_FOCAL_LENGTH,
        img_height=SIM_IMAGE_HEIGHT,
    )

    model_metrics = ModelMetrics(
        model_path=MODEL_PATH,
        model=perception.model.model if MODEL_PATH.endswith(".pt") else None
    )

    # Decrease general safe distance and increase semantic safe distance for pure vision testing
    local_planner = DiscreteDWAPlanner(safe_distance=0.1, semantic_safe_distance=1.2)

    visualizer = DemoVisualizer()

    evaluator = Evaluator(
        model_name = MODEL_PATH
    )
    nav_metrics = NavigationMetrics()
    evaluator.log_model(model_metrics)
    evaluator.start_episode()
    
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

    shortest_path = 0.0

    for i in range(len(waypoints) - 1):
        shortest_path += np.linalg.norm(
            waypoints[i + 1] - waypoints[i]
        )

    nav_metrics.start_episode(
        start_position=start_pos,
        shortest_path=shortest_path
    )
    
    try:
        for step in range(max_steps):
            
            target_wp = waypoints[current_wp_idx]

            # Use get_state to fetch the real time position from the C++ simulator engine
            current_state = agent.get_state()
            current_agent_position = current_state.position.copy()
            nav_metrics.update(current_agent_position)
            actual_trajectory.append(current_agent_position)
            
            if np.linalg.norm(agent.state.position - target_wp) < 0.25:
                print(f"Reached waypoint {current_wp_idx}. Moving to next.")
                current_wp_idx += 1
                if current_wp_idx >= len(waypoints):
                    print("Goal Reached!")
                    nav_metrics.finish_episode(True)
                    break
                target_wp = waypoints[current_wp_idx]

            obs = env.get_observations()
            rgb_frame = obs["color_sensor"]
            
            # Slice the array to keep only the first 3 channels
            rgb_frame = rgb_frame[..., :3]
            
            # Create a dummy depth frame filled with 10.0 meters
            # This prevents KeyError and forces the planner to rely solely on IPM detections
            depth_frame = np.ones((480, 640), dtype=np.float32) * 10.0

            detections, perception_metrics = perception.process_frame(rgb_frame)
            # print(perception_metrics)
            evaluator.update_frame(
                step,
                perception_metrics
            )

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

        if not nav_metrics.success:
            nav_metrics.finish_episode(False)

        evaluator.finish_episode(nav_metrics)

        visualizer.close()
        plot_topdown_trajectory(env, start_pos, goal_pos, waypoints, actual_trajectory)
        evaluator.save()
        env.close()

if __name__ == "__main__":
    main()