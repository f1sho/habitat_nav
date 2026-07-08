import numpy as np
import math
import quaternion

class DiscreteDWAPlanner:
    def __init__(self, safe_distance=0.4, semantic_safe_distance=0.8):
        self.safe_distance = safe_distance
        self.semantic_safe_distance = semantic_safe_distance
        self.last_turn_action = None 
        
        # Add memory variables for applying exponential moving average to sensor readings
        self.history_left = None
        self.history_center = None
        self.history_right = None

    def _normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def get_best_action(self, depth_frame, yolo_detections, agent_state, target_waypoint):
        height, width = depth_frame.shape
        obs_depth = depth_frame[:int(height * 0.7), :]
        
        def get_valid_min_depth(region):
            valid_pixels = region[region > 0.05]
            if len(valid_pixels) == 0:
                return 10.0  
            return np.min(valid_pixels)

        # Store raw depth values temporarily
        raw_left_dist = get_valid_min_depth(obs_depth[:, :int(width*0.4)])
        raw_center_dist = get_valid_min_depth(obs_depth[:, int(width*0.4) : int(width*0.6)])
        raw_right_dist = get_valid_min_depth(obs_depth[:, int(width*0.6):])

        # Apply data smoothing to mitigate severe perspective distortion during rotations
        alpha = 0.35 
        if self.history_left is None:
            self.history_left = raw_left_dist
            self.history_center = raw_center_dist
            self.history_right = raw_right_dist
        else:
            self.history_left = alpha * raw_left_dist + (1.0 - alpha) * self.history_left
            self.history_center = alpha * raw_center_dist + (1.0 - alpha) * self.history_center
            self.history_right = alpha * raw_right_dist + (1.0 - alpha) * self.history_right

        left_dist = self.history_left
        center_dist = self.history_center
        right_dist = self.history_right

        effective_base_safe_dist = self.safe_distance
        if center_dist < 0.2: 
            effective_base_safe_dist = 0.1
            
        current_safe_dist = effective_base_safe_dist
        sensitive_objects = ['chair', 'potted plant', 'tv', 'bed', 'sofa', 'vase'] 
        
        detected_sensitive = []
        
        for det in yolo_detections:
            if det['class_name'] in sensitive_objects:
                polygon = np.array(det['polygon']) 
                
                # Calculate the centroid of the segmentation mask
                box_center_x = np.mean(polygon[:, 0])
                box_center_y = np.mean(polygon[:, 1])
                
                if (width * 0.2) < box_center_x < (width * 0.8):
                    center_x_px = min(max(int(box_center_x), 0), width - 1)
                    center_y_px = min(max(int(box_center_y), 0), height - 1)
                    object_distance = depth_frame[center_y_px, center_x_px]

                    if object_distance < 2.5 and object_distance <= center_dist + 0.5:
                        current_safe_dist = self.semantic_safe_distance
                        detected_sensitive.append(f"{det['class_name']} at {object_distance:.2f}m")
                        break
                        
        if detected_sensitive:
            print(f"[Planner Log] Sensitive object detected: {', '.join(detected_sensitive)}. Safe distance set to {current_safe_dist}m.")
        else:
            print(f"[Planner Log] Path clear. Current safe distance is {current_safe_dist}m.")

        heading_error = self._calculate_heading_error(agent_state, target_waypoint)
        
        w_heading = 15.0
        w_clearance = 2.0
        w_velocity_penalty = 1.5

        actions = ['move_forward', 'turn_left', 'turn_right']
        best_action = 'turn_left'
        min_cost = float('inf')

        for action in actions:
            cost = 0.0
            
            if action == 'move_forward':
                if center_dist < current_safe_dist:
                    cost += 1000.0 
                    
                cost += w_heading * abs(heading_error)
                cost += w_clearance * (1.0 / (center_dist + 0.1))
                cost += w_clearance * (1.0 / (left_dist + 0.3)) * 0.3
                cost += w_clearance * (1.0 / (right_dist + 0.3)) * 0.3
                
                # Relax heading constraint to allow charging through narrow doors
                if abs(heading_error) < 0.8 and center_dist > current_safe_dist * 1.2:
                    cost -= 50.0

            elif action == 'turn_left':
                simulated_error = self._normalize_angle(heading_error - 0.26)
                cost += w_heading * abs(simulated_error)
                cost += w_velocity_penalty
                cost += w_clearance * (1.0 / (left_dist + 0.3)) * 0.1
                
                # Introduce a deadzone penalty to prevent overcorrection when nearly aligned
                if abs(heading_error) < 0.25:
                    cost += 30.0
                    
                if self.last_turn_action == 'turn_right':
                    cost += 15.0

            elif action == 'turn_right':
                simulated_error = self._normalize_angle(heading_error + 0.26)
                cost += w_heading * abs(simulated_error)
                cost += w_velocity_penalty
                cost += w_clearance * (1.0 / (right_dist + 0.3)) * 0.1
                
                # Introduce a deadzone penalty to prevent overcorrection when nearly aligned
                if abs(heading_error) < 0.25:
                    cost += 30.0
                    
                if self.last_turn_action == 'turn_left':
                    cost += 15.0

            if cost < min_cost:
                min_cost = cost
                best_action = action

        if min_cost >= 500.0:
            if self.last_turn_action in ['turn_left', 'turn_right']:
                best_action = self.last_turn_action
            else:
                best_action = 'turn_left' if left_dist >= right_dist else 'turn_right'

        if best_action in ['turn_left', 'turn_right']:
            self.last_turn_action = best_action
        else:
            # Always clear turn memory when moving forward to prevent death spirals in narrow spaces
            self.last_turn_action = None

        return best_action

    def _calculate_heading_error(self, agent_state, target_waypoint):
        pos = agent_state.position
        rot = agent_state.rotation
        
        forward_vec = quaternion.rotate_vectors(rot, np.array([0.0, 0.0, -1.0]))
        target_vec = target_waypoint - pos
        target_vec[1] = 0 
        
        forward_vec = forward_vec / (np.linalg.norm(forward_vec) + 1e-8)
        target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-8)
        
        cross_prod = np.cross(forward_vec, target_vec)
        dot_prod = np.dot(forward_vec, target_vec)
        
        return math.atan2(cross_prod[1], dot_prod)