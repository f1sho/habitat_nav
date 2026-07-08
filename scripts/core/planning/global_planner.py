import numpy as np
import heapq
from scipy.ndimage import binary_dilation

# --- START OF MODIFICATION ---
# Added distance_transform_edt for costmap generation
from scipy.ndimage import distance_transform_edt
# --- END OF MODIFICATION ---

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os

class GlobalPlanner:
    # Added map_height=None to allow passing a fixed floor height
    def __init__(self, pathfinder, meters_per_pixel=0.05, map_height=None):
        self.pathfinder = pathfinder
        self.meters_per_pixel = meters_per_pixel
        
        self.bounds = self.pathfinder.get_bounds()
        self.lower_bound = self.bounds[0]

        # If a fixed height is provided, use it; otherwise, fallback to a random navigable point
        if map_height is not None:
            self.height = map_height
        else:
            ref_point = self.pathfinder.get_random_navigable_point()
            self.height = ref_point[1]
        
        self.top_down_map = self.pathfinder.get_topdown_view(
            meters_per_pixel=self.meters_per_pixel,
            height=self.height
        )

        obstacle_map = ~self.top_down_map
        
        # --- START OF MODIFICATION ---
        # Create a distance map to penalize moving close to walls
        self.distance_map = distance_transform_edt(self.top_down_map)
        
        # Keep a small binary dilation to prevent physical collisions without blocking narrow doors
        inflated_obstacles = binary_dilation(obstacle_map, iterations=2)
        # --- END OF MODIFICATION ---
        
        self.safe_map = ~inflated_obstacles

    def world_to_map(self, point):
        x = int((point[0] - self.lower_bound[0]) / self.meters_per_pixel)
        y = int((point[2] - self.lower_bound[2]) / self.meters_per_pixel)
        return x, y

    def map_to_world(self, pixel):
        row, col = pixel
        x = self.lower_bound[0] + col * self.meters_per_pixel
        z = self.lower_bound[2] + row * self.meters_per_pixel
        return np.array([x, self.height, z])

    def _get_valid_node(self, grid, node, max_radius=20):
        r, c = node
        rows, cols = grid.shape
        r = max(0, min(r, rows - 1))
        c = max(0, min(c, cols - 1))
        
        if grid[r, c]:
            return (r, c)
            
        queue = [((r, c), 0)]
        visited = set([(r, c)])
        
        while queue:
            (curr_r, curr_c), dist = queue.pop(0)
            if grid[curr_r, curr_c]:
                return (curr_r, curr_c)
            if dist >= max_radius:
                continue
                
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dx, dy in neighbors:
                nr, nc = curr_r + dx, curr_c + dy
                if 0 <= nr < rows and 0 <= nc < cols:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), dist + 1))
                        
        print("[Warning] Could not find a valid node within search radius.")
        return (r, c)

    def plan_path(self, start_world, goal_world):
        start_px = self.world_to_map(start_world)
        goal_px = self.world_to_map(goal_world)

        start_node = (start_px[1], start_px[0])
        goal_node = (goal_px[1], goal_px[0])

        start_safe = self._get_valid_node(self.safe_map, start_node)
        goal_safe = self._get_valid_node(self.safe_map, goal_node)

        path = self._astar(self.safe_map, start_safe, goal_safe)
        
        if path is None:
            print("[Warning] No path on safe_map, trying original map...")
            start_orig = self._get_valid_node(self.top_down_map, start_node)
            goal_orig = self._get_valid_node(self.top_down_map, goal_node)
            path = self._astar(self.top_down_map, start_orig, goal_orig)

        if path is None:
            return None

        raw_waypoints = [self.map_to_world(p) for p in path]
        # Crucial Fix: Reduced step from 10 to 3 to prevent corner-cutting
        return self._simplify_waypoints(raw_waypoints, step=3)

    def _astar(self, grid, start, goal):
        rows, cols = grid.shape
        def heuristic(a, b):
            return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
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
                nr, nc = current[0] + dx, current[1] + dy
                if not (0 <= nr < rows and 0 <= nc < cols): continue
                if not grid[nr, nc]: continue

                # --- START OF MODIFICATION ---
                # Calculate base movement cost and add a penalty based on distance to obstacles
                base_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                
                dist_to_obs = self.distance_map[nr, nc]
                penalty = 0.0
                
                # Apply penalty if the node is within 15 pixels of an obstacle
                if dist_to_obs < 15:
                    penalty = 15.0 / (dist_to_obs + 0.1)
                    
                move_cost = base_cost + penalty
                # --- END OF MODIFICATION ---

                tentative_g = g_score[current] + move_cost
                neighbor = (nr, nc)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
        return None

    def _simplify_waypoints(self, waypoints, step=3):
        simplified = waypoints[::step]
        if len(simplified) == 0 or not np.allclose(simplified[-1], waypoints[-1]):
            simplified.append(waypoints[-1])
        return simplified

    def visualize_path(self, start_pos, goal_pos, waypoints, save_path="global_path_vis.png"):
        print("\n--- Visualizing Global Planner Path on Map ---")
        try:
            plt.figure(figsize=(8, 10))
            plt.imshow(self.top_down_map, cmap='gray', origin='upper')
            
            start_px = self.world_to_map(start_pos)
            goal_px = self.world_to_map(goal_pos)
            
            path_x_px = []
            path_y_px = []
            for wp in waypoints:
                px, py = self.world_to_map(wp)
                path_x_px.append(px)
                path_y_px.append(py)
                
            plt.plot(path_x_px, path_y_px, 'b-', label='A* Path', linewidth=2.5)
            plt.scatter(start_px[0], start_px[1], c='green', s=120, label='Start', zorder=5)
            plt.scatter(goal_px[0], goal_px[1], c='red', s=120, label='Goal', zorder=5)
            
            plt.title("Top-down Map with A* Path")
            plt.axis('off') 
            plt.legend(loc='upper left', framealpha=0.8)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=150)
            plt.close() 
            print(f"Saved map visualization to: {os.path.abspath(save_path)}")
        except Exception as e:
            print(f"Path visualization failed: {e}")