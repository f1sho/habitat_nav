import matplotlib
# Force matplotlib to use a headless backend to prevent display errors on servers
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

def plot_topdown_trajectory(env, start_pos, goal_pos, waypoints, actual_trajectory, save_path="actual_trajectory_map.png"):
    """
    Generate a top down map and plot the planned path versus the actual trajectory.
    """
    print(f"Starting trajectory plotting with {len(actual_trajectory)} recorded steps...")
    
    if not actual_trajectory:
        print("Error: actual_trajectory list is empty. Cannot draw map.")
        return

    meters_per_pixel = 0.05
    # Extract the binary top down map from the simulator pathfinder
    topdown_map = env.sim.pathfinder.get_topdown_view(meters_per_pixel, start_pos[1])

    # Get navigation mesh bounds for coordinate conversion
    lower_bound, _ = env.sim.pathfinder.get_bounds()

    def to_grid(pt):
        # Convert 3D world coordinates to 2D map pixel coordinates
        px = (pt[0] - lower_bound[0]) / meters_per_pixel
        pz = (pt[2] - lower_bound[2]) / meters_per_pixel
        return px, pz

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the binary top down map
    ax.imshow(topdown_map, cmap="gray", origin="lower")

    # Extract and plot the planned global path
    if waypoints:
        px_planned, pz_planned = zip(*[to_grid(wp) for wp in waypoints])
        ax.plot(px_planned, pz_planned, color="blue", linewidth=2, label="Planned Path")

    # Extract and plot the actual executed trajectory
    px_actual, pz_actual = zip(*[to_grid(pos) for pos in actual_trajectory])
    ax.plot(px_actual, pz_actual, color="orange", linewidth=2, linestyle="solid", label="Actual Trajectory")

    # Mark start and goal points
    start_px, start_pz = to_grid(start_pos)
    goal_px, goal_pz = to_grid(goal_pos)
    ax.scatter(start_px, start_pz, color="green", s=100, label="Start", zorder=5)
    ax.scatter(goal_px, goal_pz, color="red", s=100, label="Goal", zorder=5)

    ax.set_title("Top-down Map with Actual Trajectory")
    ax.legend()
    ax.axis("off")

    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Successfully saved trajectory map to {save_path}")