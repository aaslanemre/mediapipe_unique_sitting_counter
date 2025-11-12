import os
import torch


# --- PATHS AND VIDEO SETUP ---
VIDEO_SOURCE = '/Users/emrecanaslan/Desktop/copa_data/copavideo1.mkv'
OUTPUT_DIR = 'results'
OUTPUT_VIDEO_NAME = 'copavideo1_yolo_sitting_count.mp4'
# SAVE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_VIDEO_NAME)

# --- MODEL AND INFERENCE SETTINGS ---

MODEL_NAME = 'yolov8m-pose.pt' 
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

if torch.backends.mps.is_available():
    DEVICE = 'mps'
    print(f"INFO: Using Apple Silicon GPU acceleration: {DEVICE}")
else:
    DEVICE = 'cpu'
    print(f"INFO: MPS not available. Falling back to {DEVICE}.")

# --- COUNTING & FILTERING CONDITIONS ---
# Normalized X-coordinate (0.0 to 1.0) for exclusion mask.
MONUMENT_MASK_X_MAX = 0.35 
# Hip-Knee vertical difference maximum for sitting heuristic (normalized by image height).
HIP_KNEE_Y_DIFF_MAX = 0.7 
# Normalized horizontal distance for simple person tracking
TRACKING_PROXIMITY_THRESHOLD = 0.08

# --- YOLO KEYPOINT INDICES (COCO 17) ---
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12
LEFT_KNEE_IDX = 13
RIGHT_KNEE_IDX = 14