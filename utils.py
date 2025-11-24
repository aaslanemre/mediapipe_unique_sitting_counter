import cv2
import numpy as np
import config as cfg

def calculate_angle(p1, p2, p3):
    """
    Calculates the angle (in degrees) at keypoint p2 (the joint) using the cosine rule.
    p1, p2, p3 are normalized (x, y) coordinates.
    Returns 180 degrees if any points are missing, indicating a straight or non-detected limb.
    """
    # Check if any point data is missing (represented as (0, 0) in normalized space if not confident)
    if (np.sum(p1) == 0 or np.sum(p2) == 0 or np.sum(p3) == 0):
        return 180 
        
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    # Calculate vectors from the joint (p2)
    v1 = p1 - p2
    v2 = p3 - p2

    # Calculate cosine of angle
    dot_product = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    
    if mag_v1 == 0 or mag_v2 == 0:
        return 180 # Avoid division by zero

    cos_angle = dot_product / (mag_v1 * mag_v2)
    
    # Clamp the value to the range [-1, 1] for arccos stability
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def is_inside_mask(kpts_normalized):
    """Checks if the person's center (mid-hip) is inside the exclusion zone."""
    try:
        l_hip_x = kpts_normalized[cfg.LEFT_HIP_IDX][0]
        r_hip_x = kpts_normalized[cfg.RIGHT_HIP_IDX][0]
        
        # Handle cases where one hip is missing
        if l_hip_x > 0 and r_hip_x > 0:
            mid_hip_x = (l_hip_x + r_hip_x) / 2
        elif l_hip_x > 0:
            mid_hip_x = l_hip_x
        elif r_hip_x > 0:
            mid_hip_x = r_hip_x
        else:
            return False # Both hips missing

        if mid_hip_x < cfg.MONUMENT_MASK_X_MAX:
            return True
        return False
    except IndexError:
        return False


def is_sitting_heuristic(kpts_normalized, frame_height):
    """
    Determines if a person is sitting by checking two conditions:
    1. Posture Check (Angle-based): The knee angle (Hip-Knee-Ankle) is tight (legs are bent).
    2. Spatial Check: Both hips must be inside the defined BENCH rectangle.
    
    The 'frame_height' argument is now unused but kept for compatibility with the main script signature.
    """
    try:
        # --- 1. POSTURE CHECK (Knee Angle) ---
        
        # Left Leg: Hip (11), Knee (13), Ankle (15)
        l_hip = kpts_normalized[cfg.LEFT_HIP_IDX]
        l_knee = kpts_normalized[cfg.LEFT_KNEE_IDX]
        l_ankle = kpts_normalized[cfg.LEFT_ANKLE_IDX]

        # Right Leg: Hip (12), Knee (14), Ankle (16)
        r_hip = kpts_normalized[cfg.RIGHT_HIP_IDX]
        r_knee = kpts_normalized[cfg.RIGHT_KNEE_IDX]
        r_ankle = kpts_normalized[cfg.RIGHT_ANKLE_IDX]

        # Calculate angles at the knees
        l_angle = calculate_angle(l_hip, l_knee, l_ankle)
        r_angle = calculate_angle(r_hip, r_knee, r_ankle)

        # Check if EITHER knee angle falls within the sitting range
        is_l_sitting = (l_angle >= cfg.MIN_KNEE_ANGLE_FOR_SITTING and l_angle <= cfg.MAX_KNEE_ANGLE_FOR_SITTING)
        is_r_sitting = (r_angle >= cfg.MIN_KNEE_ANGLE_FOR_SITTING and r_angle <= cfg.MAX_KNEE_ANGLE_FOR_SITTING)
        
        is_posture_sitting = is_l_sitting or is_r_sitting

        # --- 2. SPATIAL CHECK (Hips on Bench Location) ---
        l_hip_x = l_hip[0]
        l_hip_y = l_hip[1]
        r_hip_x = r_hip[0]
        r_hip_y = r_hip[1]
        
        l_hip_on_bench = (
            (l_hip_x >= cfg.BENCH_X_MIN) and (l_hip_x <= cfg.BENCH_X_MAX) and
            (l_hip_y >= cfg.BENCH_Y_MIN) and (l_hip_y <= cfg.BENCH_Y_MAX)
        )
        
        r_hip_on_bench = (
            (r_hip_x >= cfg.BENCH_X_MIN) and (r_hip_x <= cfg.BENCH_X_MAX) and
            (r_hip_y >= cfg.BENCH_Y_MIN) and (r_hip_y <= cfg.BENCH_Y_MAX)
        )
        
        # Original logic: both hips must be on the bench
        is_spatial_on_bench = l_hip_on_bench and r_hip_on_bench
        
        # --- FINAL RESULT: BOTH CONDITIONS MUST BE TRUE ---
        return is_posture_sitting and is_spatial_on_bench
        
    except IndexError:
        return False
    except Exception:
        # Catch any unexpected array or math errors
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
        for i, (x, y) in enumerate(kpts):
            if x != 0 and y != 0:
                point_color = (0, 0, 255) # Blue for all points
                
                # Highlight key points used in the sitting heuristic
                if i in [cfg.LEFT_HIP_IDX, cfg.RIGHT_HIP_IDX]:
                    point_color = (0, 255, 255) # Yellow for Hips
                elif i in [cfg.LEFT_KNEE_IDX, cfg.RIGHT_KNEE_IDX, cfg.LEFT_ANKLE_IDX, cfg.RIGHT_ANKLE_IDX]:
                    point_color = (255, 0, 0) # Red for Leg Joints
                    
                cv2.circle(image, (int(x), int(y)), 5, point_color, -1)
    
    return image
