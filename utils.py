import cv2
import numpy as np
import config as cfg

def is_inside_mask(kpts_normalized):
    """Checks if the person's center (mid-hip) is inside the exclusion zone."""
    try:
        l_hip_x = kpts_normalized[cfg.LEFT_HIP_IDX][0]
        r_hip_x = kpts_normalized[cfg.RIGHT_HIP_IDX][0]
        mid_hip_x = (l_hip_x + r_hip_x) / 2
        
        if mid_hip_x < cfg.MONUMENT_MASK_X_MAX:
            return True
        return False
    except IndexError:
        return False


def is_sitting_heuristic(kpts_normalized, height):
    """
    Determines if a person is sitting by checking two conditions:
    1. Posture Check: Hip-Knee proximity (legs are bent).
    2. Spatial Check: Both hips must be inside the defined BENCH rectangle.
    """
    try:
        # --- 1. POSTURE CHECK (Hip-Knee Proximity) ---
        l_hip_y_px = kpts_normalized[cfg.LEFT_HIP_IDX][1] * height
        l_knee_y_px = kpts_normalized[cfg.LEFT_KNEE_IDX][1] * height
        r_hip_y_px = kpts_normalized[cfg.RIGHT_HIP_IDX][1] * height
        r_knee_y_px = kpts_normalized[cfg.RIGHT_KNEE_IDX][1] * height
        
        l_diff = abs(l_hip_y_px - l_knee_y_px) / height
        r_diff = abs(r_hip_y_px - r_knee_y_px) / height
        
        is_posture_sitting = l_diff < cfg.HIP_KNEE_Y_DIFF_MAX or r_diff < cfg.HIP_KNEE_Y_DIFF_MAX


        # --- 2. SPATIAL CHECK (Hips on Bench Location) ---
        l_hip_x = kpts_normalized[cfg.LEFT_HIP_IDX][0]
        l_hip_y = kpts_normalized[cfg.LEFT_HIP_IDX][1]
        r_hip_x = kpts_normalized[cfg.RIGHT_HIP_IDX][0]
        r_hip_y = kpts_normalized[cfg.RIGHT_HIP_IDX][1]
        
        l_hip_on_bench = (
            (l_hip_x >= cfg.BENCH_X_MIN) and (l_hip_x <= cfg.BENCH_X_MAX) and
            (l_hip_y >= cfg.BENCH_Y_MIN) and (l_hip_y <= cfg.BENCH_Y_MAX)
        )
        
        r_hip_on_bench = (
            (r_hip_x >= cfg.BENCH_X_MIN) and (r_hip_x <= cfg.BENCH_X_MAX) and
            (r_hip_y >= cfg.BENCH_Y_MIN) and (r_hip_y <= cfg.BENCH_Y_MAX)
        )
        
        is_spatial_on_bench = l_hip_on_bench and r_hip_on_bench
        
        # --- FINAL RESULT: BOTH CONDITIONS MUST BE TRUE ---
        return is_posture_sitting and is_spatial_on_bench
        
    except IndexError:
        return False

def draw_pose(image, keypoints_xy, color=(255, 0, 0), thickness=3):
    """Draws skeleton and keypoints on the image from YOLO keypoints (x, y)."""
    # COCO 17-keypoint skeleton 
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    
    # Draw logic is simplified for clean separation
    for kpts in keypoints_xy:
        # Draw skeleton lines
        for (start, end) in skeleton:
            start_pt = (int(kpts[start][0]), int(kpts[start][1]))
            end_pt = (int(kpts[end][0]), int(kpts[end][1]))
            if start_pt != (0, 0) and end_pt != (0, 0):
                cv2.line(image, start_pt, end_pt, color, thickness)
        
        # Draw keypoints (Red circles, fixed color for visibility)
        for x, y in kpts:
            if x != 0 and y != 0:
                cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
    
    return image
