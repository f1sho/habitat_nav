import numpy as np
import time
import csv
from core.habitat_env import HabitatEnv
from core.planning.global_planner import GlobalPlanner
from core.planning.local_planner import DiscreteDWAPlanner

# Import both perception modules for seamless switching during evaluation
from core.perception import PerceptionModule as SegIPMPerception
from core.dual_perception import DualPerceptionModule

# Import the decoupled navigation evaluator
from evaluate.nav_evaluator import PureNavigationEvaluator

# === MODIFICATION START: Cleaned Imports ===
# Removed torch and torchmetrics imports as perception metrics are handled offline
# === MODIFICATION END ===

def main():
    scene_path = "/home/hannah/data/replica_v1/apartment_2/habitat/mesh_semantic.ply"
    navmesh_path = "/home/hannah/data/replica_v1/apartment_2/habitat/mesh_semantic.navmesh"

    print("Initializing simulation environment...")
    env = HabitatEnv(scene_path, navmesh_path)
    
    # Configure the perception mode here
    test_mode = "MDE" 
    
    if test_mode == "IPM":
        print("Loading geometric IPM perception module...")
        perception = SegIPMPerception(
            model_path="yolo26n-seg.pt",
            camera_height=1.5,
            focal_length=800.0,
            img_height=480
        )
    else:
        print("Loading neural MDE perception module...")
        perception = DualPerceptionModule(yolo_path="yolo26n-seg.pt", conf_threshold=0.5)
        
    local_planner = DiscreteDWAPlanner(safe_distance=0.1, semantic_safe_distance=0.5)
    
    total_episodes = 20
    max_steps_per_episode = 500
    
    # Initialize the streamlined navigation evaluator
    nav_evaluator = PureNavigationEvaluator(mode_name=test_mode)

    # === MODIFICATION START: Removed Metric Initialization ===
    # The MulticlassJaccardIndex initialization block has been completely deleted
    # to keep the memory footprint light and focus solely on pathfinding efficiency
    # === MODIFICATION END ===
    
    print(f"\nStarting batch evaluation for {total_episodes} episodes using {test_mode} mode.")
    
    for episode in range(total_episodes):
        dynamic_seed = int(time.time()) + episode
        env.sim.seed(dynamic_seed)
        env.sim.pathfinder.seed(dynamic_seed)
        
        start_pos = env.sim.pathfinder.get_random_navigable_point()
        goal_pos = env.sim.pathfinder.get_random_navigable_point()
        
        while np.linalg.norm(start_pos - goal_pos) < 5.0:
            start_pos = env.sim.pathfinder.get_random_navigable_point()
            goal_pos = env.sim.pathfinder.get_random_navigable_point()
            
        global_planner = GlobalPlanner(env.sim.pathfinder, map_height=start_pos[1])
        waypoints = global_planner.plan_path(start_pos, goal_pos)
        
        if not waypoints:
            print(f"Episode {episode + 1}: Failed to generate global path. Skipping.")
            continue
            
        # Calculate theoretical shortest path distance using Euclidean logic
        shortest_path_dist = 0.0
        if len(waypoints) > 1:
            for i in range(1, len(waypoints)):
                shortest_path_dist += np.linalg.norm(waypoints[i] - waypoints[i-1])
            
        agent = env.sim.get_agent(0)
        agent_state = agent.get_state()
        agent_state.position = start_pos
        agent.set_state(agent_state)
        
        current_wp_idx = 1
        step_count = 0
        is_successful = False
        total_path_length = 0.0
        last_position = start_pos
        
        for step in range(max_steps_per_episode):
            step_count += 1
            target_wp = waypoints[current_wp_idx]
            
            current_position = agent.state.position
            distance_to_wp = np.linalg.norm(current_position - target_wp)
            
            total_path_length += np.linalg.norm(current_position - last_position)
            last_position = current_position
            
            if distance_to_wp < 0.25:
                current_wp_idx += 1
                if current_wp_idx >= len(waypoints):
                    is_successful = True
                    break
                target_wp = waypoints[current_wp_idx]
                
            obs = env.get_observations()
            rgb_frame = obs["color_sensor"]

            # === MODIFICATION START: Cleaned Inference Loop ===
            # Removed all semantic sensor data extraction and tensor updates
            inference_start = time.time()
            
            if test_mode == "IPM":
                rgb_sliced = rgb_frame[..., :3]
                depth_frame = np.ones((480, 640), dtype=np.float32) * 10.0
                detections = perception.process_frame(rgb_sliced)
            else:
                # Discard the mask output as it is no longer needed for live scoring
                detections, depth_frame, _ = perception.process_frame(rgb_frame)
                
            inference_end = time.time()
            nav_evaluator.record_inference_time(inference_end - inference_start)
            # === MODIFICATION END ===
                
            action = local_planner.get_best_action(depth_frame, detections, agent.state, target_wp)
            env.step(action)
            
        # Record the physical performance metrics to the evaluator
        nav_evaluator.record_episode(
            episode_id=episode + 1,
            success=is_successful,
            steps=step_count,
            shortest_dist=shortest_path_dist,
            actual_dist=total_path_length
        )
        
        print(f"Episode {episode + 1} finished | Success: {is_successful} | Path Length: {total_path_length:.2f}m")
        
    # Export all aggregated navigation metrics to a single spreadsheet
    csv_filename = f"evaluate/results/evaluation_results_{test_mode}.csv"
    print(f"\nWriting evaluation data to {csv_filename}...")
    nav_evaluator.export_to_csv(output_dir="evaluate/results")
        
    print("Evaluation pipeline completed successfully.")
    env.close()

if __name__ == "__main__":
    main()