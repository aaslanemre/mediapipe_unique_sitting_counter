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
HIP_KNEE_Y_DIFF_MAX = 0.15
# Normalized horizontal distance for simple person tracking
TRACKING_PROXIMITY_THRESHOLD = 0.1

# --- NEW BENCH BOUNDARIES (Normalized 0.0 to 1.0) ---

# NOTE: YOU MUST TUNE THESE VALUES BASED ON YOUR FULL VIDEO FRAME
BENCH_X_MIN = MONUMENT_MASK_X_MAX   # Left edge of the bench box
BENCH_X_MAX = 0.65   # Right edge of the bench box
BENCH_Y_MIN = 0.50   # Top edge of the bench box (around the person's hips)
BENCH_Y_MAX = 0.80   # Bottom edge of the bench box

# --- YOLO KEYPOINT INDICES (COCO 17) ---
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12
LEFT_KNEE_IDX = 13
RIGHT_KNEE_IDX = 14
