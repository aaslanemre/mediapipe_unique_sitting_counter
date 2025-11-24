import torch

# --- PATHS AND VIDEO SETUP ---
# Update these paths to match your local environment
VIDEO_SOURCE = '/Users/emrecanaslan/Desktop/copa_data/copavideo1.mkv' 
OUTPUT_DIR = 'results'
OUTPUT_VIDEO_NAME = 'copavideo1_yolo11.1_sitting_count.mp4'

# --- MODEL AND INFERENCE SETTINGS ---
# UPDATED: Assuming 'yolo11m-pose.pt' is available (or use 'yolov8m-pose.pt' if 11 is not yet installed/available)
MODEL_NAME = 'yolov8m-pose.pt'

CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45

# --- HARDWARE ACCELERATION ---
if torch.backends.mps.is_available():
    DEVICE = 'mps'
    print(f"INFO: Using Apple Silicon GPU acceleration: {DEVICE}")
elif torch.cuda.is_available():
    DEVICE = 'cuda'
    print(f"INFO: Using NVIDIA CUDA acceleration: {DEVICE}")
else:
    DEVICE = 'cpu'
    print(f"INFO: GPU not available. Falling back to {DEVICE}.")

# --- COUNTING & FILTERING CONDITIONS (ANGLE-BASED) ---
MONUMENT_MASK_X_MAX = 0.0 
TRACKING_PROXIMITY_THRESHOLD = 0.1
# ANGLE-BASED HEURISTIC CONFIGURATION:
# We check the angle at the knee (Hip-Knee-Ankle). 
# Standing is ~180 degrees. Sitting is typically 90-120 degrees.
MIN_KNEE_ANGLE_FOR_SITTING = 75  # Minimum degrees for bent knee
MAX_KNEE_ANGLE_FOR_SITTING = 125 # Maximum degrees for bent knee

# --- TEMPORAL SMOOTHING ---
# How many consecutive frames a person must be "sitting" before the count increments.
# At 30 FPS, 30 frames = 1 second of continuous sitting.
FRAMES_TO_CONFIRM_SITTING = 30 

# --- BENCH BOUNDARIES (Normalized 0.0 to 1.0) ---
BENCH_X_MIN = 0.35   
BENCH_X_MAX = 0.65   
BENCH_Y_MIN = 0.40   
BENCH_Y_MAX = 0.80   

# --- YOLO KEYPOINT INDICES (COCO 17) ---
LEFT_HIP_IDX = 11
RIGHT_HIP_IDX = 12
LEFT_KNEE_IDX = 13
RIGHT_KNEE_IDX = 14
# Added for angle calculation:
LEFT_ANKLE_IDX = 15
RIGHT_ANKLE_IDX = 16
