import cv2
import numpy as np

class DemoVisualizer:
    def __init__(self, window_name="Semantic Navigation Demo", save_video=True):
        self.window_name = window_name
        self.save_video = save_video
        self.video_writer = None

    def show_frame(self, rgb_frame, detections, action, step, dist_to_wp):
        # Convert RGBA to BGR
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGBA2BGR)

        # CHANGE START: Draw segmentation polygons instead of bounding boxes
        for det in detections:
            polygon = det['polygon'] 
            conf = det['confidence']
            name = det['class_name']
            
            # Convert polygon coordinates to integers for OpenCV
            poly_pts = np.int32([polygon])
            
            # Draw the polygon outline
            cv2.polylines(bgr_frame, poly_pts, isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Find the top point of the polygon to place the text label
            text_x = int(np.min(polygon[:, 0]))
            text_y = int(np.min(polygon[:, 1])) - 10
            
            # Ensure text does not go out of the top boundary
            if text_y < 10:
                text_y = 20
                
            cv2.putText(bgr_frame, f"{name} {conf:.2f}", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # CHANGE END

        # Draw semi transparent HUD at the top left corner
        overlay = bgr_frame.copy()
        cv2.rectangle(overlay, (5, 5), (350, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, bgr_frame, 0.4, 0, bgr_frame)

        cv2.putText(bgr_frame, f"Step: {step} | Action: {action}", (15, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(bgr_frame, f"Dist to WP: {dist_to_wp:.2f}m", (15, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Save MP4 video
        if self.save_video:
            if self.video_writer is None:
                h, w = bgr_frame.shape[:2]
                # Use mp4v codec for video saving
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter('demo_output.mp4', fourcc, 8.0, (w, h))
            self.video_writer.write(bgr_frame)

        # Safe window display to catch cross thread errors
        try:
            cv2.imshow(self.window_name, bgr_frame)
            cv2.waitKey(10) 
        except Exception as e:
            pass 

    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
            print("Demo video saved as 'demo_output.mp4'!")
        try:
            cv2.destroyAllWindows()
        except:
            pass