# main.py - Unique Sitting Person Counter with Region Masking

import cv2
import mediapipe as mp
import numpy as np
import os
import sys

# Initialize MediaPipe's Pose solution and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- CONFIGURATION ---
# Corrected path to step up from the current directory
VIDEO_SOURCE = '/Users/emrecanaslan/Desktop/copa_data/copavideo1.mkv'
OUTPUT_DIR = 'results' # New folder for output
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'sitting_count_output.mp4')
VISIBILITY_THRESHOLD = 0.8

# --- EXCLUSION MASK (NEW) ---
# Normalized X-coordinate (0.0 to 1.0) where the monument/bench is located.
# We will ignore any person whose center (mid-hip) is to the left of this line.
# Based on the image, setting this to 0.35 or 0.4 should exclude the far left bench/monument.
MONUMENT_MASK_X_MAX = 0.35 

# --- SITTING THRESHOLDS (Based on Positional Heuristics) ---
HIP_KNEE_Y_DIFF_MAX = 0.15 

# --- GLOBAL TRACKING STORAGE ---
unique_sitting_ids = {} 
next_temp_id = 0 

# --- HELPER FUNCTIONS ---

def is_inside_mask(landmarks, x_max):
    """
    Checks if the person's center (mid-hip) is inside the exclusion zone (left side).
    
    Returns: True if the person should be ignored (inside the monument region), False otherwise.
    """
    try:
        # Use the average horizontal position of the hips as the person's center
        l_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
        r_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x
        mid_hip_x = (l_hip_x + r_hip_x) / 2
        
        # If the person's center is to the left of the max X-coordinate
        if mid_hip_x < x_max:
            return True
        return False
    except Exception:
        # Default to False if landmarks are missing (i.e., don't mask if we can't confirm position)
        return False


def is_sitting_heuristic(landmarks, height):
    """
    Determines if a person is sitting by analyzing the vertical distance 
    between the Hip and Knee landmarks. (Side-view perspective)
    """
    try:
        # We check both left and right sides for robustness
        l_hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height
        l_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height
        r_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * height
        r_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * height
        
        # Hip-Knee Y-difference should be small if the leg is bent (sitting)
        l_diff = abs(l_hip_y - l_knee_y) / height
        r_diff = abs(r_hip_y - r_knee_y) / height
        
        # If the vertical difference is small (legs are horizontal/bent)
        if l_diff < HIP_KNEE_Y_DIFF_MAX or r_diff < HIP_KNEE_Y_DIFF_MAX:
            # Additionally check if the hips are near the bottom half of the screen (i.e., not standing up)
            if l_hip_y / height > 0.4 and r_hip_y / height > 0.4:
                return True
        
        return False
        
    except Exception:
        return False

# --- Main Program ---

def analyze_video_for_sitting():
    global next_temp_id
    
    if not os.path.exists(VIDEO_SOURCE):
        print(f"Error: Video file not found at {VIDEO_SOURCE}.")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_SOURCE) 
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}")
        sys.exit(1)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # --- VIDEO WRITER SETUP ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (frame_width, frame_height))
    print(f"Saving output video to: {OUTPUT_FILE}")
    # --------------------------

    temp_trackers = {} 

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            frame_height, frame_width, _ = image.shape
            
            # Draw the exclusion mask region for debug visualization
            mask_pixel_x = int(MONUMENT_MASK_X_MAX * frame_width)
            cv2.rectangle(image, (0, 0), (mask_pixel_x, frame_height), (0, 0, 100), -1) # Dark Blue overlay
            cv2.putText(image, "EXCLUSION ZONE", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # --- NEW MASK CHECK ---
                if is_inside_mask(landmarks, MONUMENT_MASK_X_MAX):
                    # Person is in the exclusion zone, skip all tracking and counting logic
                    feedback = "MASKED (Monument)"
                    feedback_box_color = (100, 100, 100) # Gray
                else:
                    # --- 1. SITTING CHECK ---
                    is_person_sitting = is_sitting_heuristic(landmarks, frame_height)
                    
                    # --- 2. SIMPLE TRACKING/ID ASSIGNMENT ---
                    hip_x_normalized = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x
                    
                    assigned_id = None
                    
                    for tid, pos in temp_trackers.items():
                        if abs(pos - hip_x_normalized) < 0.1:
                            assigned_id = tid
                            break
                    
                    if assigned_id is None:
                        assigned_id = next_temp_id
                        next_temp_id += 1
                    
                    temp_trackers[assigned_id] = hip_x_normalized
                    
                    # --- 3. UNIQUE COUNTING LOGIC ---
                    if is_person_sitting and assigned_id not in unique_sitting_ids:
                        unique_sitting_ids[assigned_id] = True
                        feedback = f"NEW PERSON {assigned_id} COUNTED!"
                        feedback_box_color = (255, 100, 0)
                    elif is_person_sitting:
                        feedback = f"SITTING (ID: {assigned_id})"
                        feedback_box_color = (0, 150, 0)
                    else:
                        feedback = "NOT SITTING"
                        feedback_box_color = (0, 165, 255)

                # --- 4. RENDER UI AND FEEDBACK ---
                
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            else:
                feedback = "NO BODY DETECTED"
                feedback_box_color = (128, 0, 0)

            # Status Boxes (Count Box)
            cv2.rectangle(image, (0, 0), (350, 73), (50, 50, 50), -1)
            cv2.putText(image, 'UNIQUE SITTING COUNT', (15, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, str(len(unique_sitting_ids)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Feedback Box
            cv2.rectangle(image, (350, 0), (frame_width, 73), feedback_box_color, -1)
            (text_width, _), _ = cv2.getTextSize(feedback, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            text_x = 350 + ((frame_width - 350) - text_width) // 2 
            cv2.putText(image, feedback, (text_x, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # --- WRITE FRAME TO OUTPUT FILE ---
            out.write(image)
            
            # Display result
            cv2.imshow('Unique Sitting Counter', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    out.release() # Release the VideoWriter
    cv2.destroyAllWindows()
    print(f"\n--- Analysis Complete ---")
    print(f"Output video saved to: {OUTPUT_FILE}")
    print(f"Total unique people seen sitting: {len(unique_sitting_ids)}")
    print("-------------------------\n")


if __name__ == "__main__":
    analyze_video_for_sitting()
    print("Analysis complete.")
