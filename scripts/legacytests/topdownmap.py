import habitat_sim
import matplotlib.pyplot as plt
import heapq
from scipy.ndimage import binary_dilation
import numpy as np
import time


def astar(grid, start, goal):
    rows, cols = grid.shape

    def heuristic(a, b):
        # Euclidean heuristic is more suitable for 8-neighbour movement
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    neighbors = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dx, dy in neighbors:
            nr = current[0] + dx
            nc = current[1] + dy

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            if not grid[nr, nc]:
                continue

            move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
            tentative_g = g_score[current] + move_cost
            neighbor = (nr, nc)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None

def simplify_waypoints(waypoints, step=10):
    simplified = waypoints[::step]

    if len(simplified) == 0 or not np.allclose(simplified[-1], waypoints[-1]):
        simplified.append(waypoints[-1])

    return simplified


def main():
    scene_path = "/home/hannah/data/versioned_data/habitat_test_scenes/apartment_1.glb"
    navmesh_path = "/home/hannah/data/versioned_data/habitat_test_scenes/apartment_1.navmesh"

    meters_per_pixel = 0.05

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    sim = habitat_sim.Simulator(cfg)

    # Use a changing seed so start and goal change each run
    seed = int(time.time())
    sim.seed(seed)

    try:
        sim.pathfinder.seed(seed)
    except Exception:
        pass

    print("Seed:", seed)

    loaded = sim.pathfinder.load_nav_mesh(navmesh_path)
    print("Navmesh loaded:", loaded)
    print("Pathfinder loaded:", sim.pathfinder.is_loaded)

    bounds = sim.pathfinder.get_bounds()
    lower_bound = bounds[0]
    print("Bounds:", bounds)

    # Use a valid navigable height for the top-down map
    reference_point = sim.pathfinder.get_random_navigable_point()
    height = reference_point[1]
    print("Using height:", height)

    top_down_map = sim.pathfinder.get_topdown_view(
        meters_per_pixel=meters_per_pixel,
        height=height
    )

    print("Map min/max:", top_down_map.min(), top_down_map.max())

    def world_to_map(point):
        x = int((point[0] - lower_bound[0]) / meters_per_pixel)
        y = int((point[2] - lower_bound[2]) / meters_per_pixel)
        return x, y

    def map_to_world(pixel):
        row, col = pixel
        x = lower_bound[0] + col * meters_per_pixel
        z = lower_bound[2] + row * meters_per_pixel
        y = height
        return np.array([x, y, z])

    # Sample start and goal until they are far enough apart
    while True:
        start = sim.pathfinder.get_random_navigable_point()
        goal = sim.pathfinder.get_random_navigable_point()

        dist = np.linalg.norm(start - goal)

        if dist > 3.0:
            break

    print("Start:", start)
    print("Goal:", goal)
    print("Start-goal distance:", dist)

    start_px = world_to_map(start)
    goal_px = world_to_map(goal)

    print("Start pixel:", start_px)
    print("Goal pixel:", goal_px)

    # Convert from image coordinates (x, y) to numpy coordinates (row, col)
    start_node = (start_px[1], start_px[0])
    goal_node = (goal_px[1], goal_px[0])

    # Optional safety margin around obstacles
    obstacle_map = ~top_down_map
    inflated_obstacles = binary_dilation(obstacle_map, iterations=1)
    safe_map = ~inflated_obstacles

    path = astar(safe_map, start_node, goal_node)

    # If safety margin blocks the path, fall back to original map
    if path is None:
        print("No path on safe_map, trying original map...")
        path = astar(top_down_map, start_node, goal_node)

    print("Path found:", path is not None)

    if path is not None:
        raw_waypoints = [map_to_world(p) for p in path]
        waypoints = simplify_waypoints(raw_waypoints, step=10)

        print("Raw waypoints:", len(raw_waypoints))
        print("Simplified waypoints:", len(waypoints))
        print("First waypoint:", waypoints[0])
        print("Last waypoint:", waypoints[-1])

    # Plot result
    plt.figure(figsize=(8, 8))
    plt.imshow(top_down_map, cmap="gray")

    plt.scatter(start_px[0], start_px[1], c="green", s=80, label="Start")
    plt.scatter(goal_px[0], goal_px[1], c="red", s=80, label="Goal")

    if path is not None:
        path_x = [p[1] for p in path]
        path_y = [p[0] for p in path]
        plt.plot(path_x, path_y, c="blue", linewidth=2, label="A* Path")

    plt.legend()
    plt.axis("off")
    plt.title("Top-down Map with A* Path")

    filename = f"apartment_1_astar_{seed}.png"
    plt.savefig(filename, bbox_inches="tight", pad_inches=0)
    print("Saved:", filename)

    plt.show()
    sim.close()


if __name__ == "__main__":
    main()