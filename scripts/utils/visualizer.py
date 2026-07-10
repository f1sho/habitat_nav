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

        # Draw true segmentation outlines and ignore bounding boxes
        for det in detections:
            # Print keys to check the actual data structure from the perception module
            print(f"Debug check keys: {list(det.keys())}")
            
            text_x = None
            text_y = None
            
            if 'polygon' in det:
                polygon = np.array(det['polygon'])
                poly_pts = np.int32([polygon])
                cv2.polylines(bgr_frame, poly_pts, isClosed=True, color=(0, 255, 0), thickness=2)
                text_x = int(np.min(polygon[:, 0]))
                text_y = int(np.min(polygon[:, 1])) - 10
                
            elif 'mask' in det:
                mask_data = det['mask']
                if isinstance(mask_data, np.ndarray) and len(mask_data.shape) == 2:
                    mask_uint8 = (mask_data * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(bgr_frame, contours, -1, (0, 255, 0), 2)
                    
                    if len(contours) > 0:
                        all_pts = np.vstack(contours)
                        text_x = int(np.min(all_pts[:, 0, 0]))
                        text_y = int(np.min(all_pts[:, 0, 1])) - 10
                    else:
                        continue
                else:
                    polygon = np.array(mask_data)
                    poly_pts = np.int32([polygon])
                    cv2.polylines(bgr_frame, poly_pts, isClosed=True, color=(0, 255, 0), thickness=2)
                    text_x = int(np.min(polygon[:, 0]))
                    text_y = int(np.min(polygon[:, 1])) - 10

            elif 'segmentation' in det:
                # Catch another common key name for masks
                seg_data = np.array(det['segmentation'])
                poly_pts = np.int32([seg_data])
                cv2.polylines(bgr_frame, poly_pts, isClosed=True, color=(0, 255, 0), thickness=2)
                text_x = int(np.min(seg_data[:, 0]))
                text_y = int(np.min(seg_data[:, 1])) - 10

            else:
                # If only boxes are available the drawing is skipped
                continue

            conf = det.get('confidence', 0.0)
            name = det.get('class_name', 'Unknown')
            
            if text_y is None or text_x is None:
                continue
                
            if text_y < 10:
                text_y = 20
                
            cv2.putText(bgr_frame, f"{name} {conf:.2f}", (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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