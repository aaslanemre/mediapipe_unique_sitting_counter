import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
from datetime import datetime
import config as cfg
import utils

# --- GLOBAL TRACKING STORAGE ---
unique_sitting_ids = {} 
next_temp_id = 0 
temp_trackers = {} 

# --- Main Analysis Function ---
def analyze_video_for_sitting():
    global next_temp_id, temp_trackers
    
    # --- 1. INITIALIZATION ---
    if not os.path.exists(cfg.VIDEO_SOURCE):
        print(f"Error: Video file not found at {cfg.VIDEO_SOURCE}.")
        sys.exit(1)

    pose_model = YOLO(cfg.MODEL_NAME)
    cap = cv2.VideoCapture(cfg.VIDEO_SOURCE) 
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {cfg.VIDEO_SOURCE}")
        sys.exit(1)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # --- VIDEO WRITER SETUP (Unique Name) ---
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("_%Y%m%d_%H%M%S")
    
    base_name, ext = os.path.splitext(cfg.OUTPUT_VIDEO_NAME)
    unique_file_name = f"{base_name}{timestamp}{ext}"
    unique_save_path = os.path.join(cfg.OUTPUT_DIR, unique_file_name) 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(unique_save_path, fourcc, fps, (frame_width, frame_height))
    # ----------------------------------------------------

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # --- 2. YOLO INFERENCE ---
        results = pose_model(
            frame,
            conf=cfg.CONF_THRESHOLD,
            iou=cfg.IOU_THRESHOLD,
            classes=[0],
            device=cfg.DEVICE, 
            half=True,
            verbose=False
        )
        
        vis_frame = frame.copy()
        keypoints_to_draw = []
        
        # Draw the exclusion mask region
        mask_pixel_x = int(cfg.MONUMENT_MASK_X_MAX * frame_width)
        cv2.rectangle(vis_frame, (0, 0), (mask_pixel_x, frame_height), (0, 0, 100), -1)
        cv2.putText(vis_frame, "EXCLUSION ZONE", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # --- NEW: DRAW BENCH BOUNDARY BOX (Visualization) ---
        x_min_px = int(cfg.BENCH_X_MIN * frame_width)
        x_max_px = int(cfg.BENCH_X_MAX * frame_width)
        y_min_px = int(cfg.BENCH_Y_MIN * frame_height)
        y_max_px = int(cfg.BENCH_Y_MAX * frame_height)
        
        box_color = (255, 255, 0) 
        line_thickness = 2
        
        # Draw the rectangle with a simple dashed look (for clearer visualization)
        cv2.line(vis_frame, (x_min_px, y_min_px), (x_max_px, y_min_px), box_color, line_thickness)
        for x in range(x_min_px, x_max_px, 10): 
             cv2.line(vis_frame, (x, y_max_px), (x + 5, y_max_px), box_color, line_thickness)
        cv2.line(vis_frame, (x_min_px, y_min_px), (x_min_px, y_max_px), box_color, line_thickness)
        cv2.line(vis_frame, (x_max_px, y_min_px), (x_max_px, y_max_px), box_color, line_thickness)
        
        cv2.putText(vis_frame, "BENCH ZONE", (x_min_px + 5, y_min_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)
        # ---------------------------------------------------
        
        # --- 3. COUNTING & TRACKING LOOP ---
        current_feedback = "NO DETECTION"
        current_feedback_color = (128, 0, 0)
        
        for result in results:
            if result.keypoints is None:
                continue
                
            keypoints_pixel_xy = result.keypoints.xy.cpu().numpy()
            keypoints_normalized_xyn = result.keypoints.xyn.cpu().numpy()
            
            for kpts_pixel, kpts_normalized in zip(keypoints_pixel_xy, keypoints_normalized_xyn):
                
                # A. MASK CHECK
                if utils.is_inside_mask(kpts_normalized):
                    keypoints_to_draw.append((kpts_pixel, (100, 100, 100))) 
                    current_feedback = "MASKED"
                    current_feedback_color = (100, 100, 100)
                    continue
                
                # B. SITTING CHECK (Uses combined Posture AND Spatial checks from utils.py)
                is_person_sitting = utils.is_sitting_heuristic(kpts_normalized, frame_height)
                
                # C. SIMPLE TRACKING/ID ASSIGNMENT
                hip_x_normalized = kpts_normalized[cfg.LEFT_HIP_IDX][0]
                assigned_id = None
                
                for tid, pos in temp_trackers.items():
                    if abs(pos - hip_x_normalized) < cfg.TRACKING_PROXIMITY_THRESHOLD: 
                        assigned_id = tid
                        break
                
                if assigned_id is None:
                    assigned_id = next_temp_id
                    next_temp_id += 1
                
                temp_trackers[assigned_id] = hip_x_normalized
                
                # D. UNIQUE COUNTING LOGIC
                if is_person_sitting and assigned_id not in unique_sitting_ids:
                    unique_sitting_ids[assigned_id] = True
                    feedback = f"NEW COUNTED! (ID: {assigned_id})"
                    color = (0, 100, 255) # Blue-Orange
                    current_feedback = feedback
                    current_feedback_color = color
                elif is_person_sitting:
                    feedback = f"SITTING (ID: {assigned_id})"
                    color = (0, 255, 0) # Bright Green
                    current_feedback = feedback
                    current_feedback_color = color
                else:
                    feedback = "NOT COUNTED"
                    color = (0, 165, 255) # Orange

                keypoints_to_draw.append((kpts_pixel, color))

                if result.boxes is not None and len(result.boxes.xyxy) > 0:
                    box = result.boxes.xyxy.cpu().numpy()[0] 
                    x1, y1 = map(int, box[:2])
                    cv2.putText(vis_frame, feedback, (x1, y1 - 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


        # --- 4. FINAL DRAWING & UI ---
        for kpts, color in keypoints_to_draw:
            vis_frame = utils.draw_pose(vis_frame, [kpts], color=color)
        
        current_count = len(unique_sitting_ids)
        
        # Draw Status Boxes
        cv2.rectangle(vis_frame, (0, 0), (350, 73), (50, 50, 50), -1)
        cv2.putText(vis_frame, 'UNIQUE SITTING COUNT', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis_frame, str(current_count), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.rectangle(vis_frame, (350, 0), (frame_width, 73), current_feedback_color, -1)
        (text_width, _), _ = cv2.getTextSize(current_feedback, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = 350 + ((frame_width - 350) - text_width) // 2 
        cv2.putText(vis_frame, current_feedback, (text_x, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # --- 5. WRITE FRAME ---
        out.write(vis_frame)
        
        if frame_count % 100 == 0:
            print(f"Frames processed: {frame_count}. Current Count: {current_count}")

    # --- 6. CLEANUP ---
    cap.release()
    out.release()
    
    print(f"\n--- Analysis Complete ---")
    print(f"Output video saved to: {unique_save_path}")
    print(f"Total unique people seen sitting: {len(unique_sitting_ids)}")
    print("-------------------------\n")


if __name__ == "__main__":
    analyze_video_for_sitting()
