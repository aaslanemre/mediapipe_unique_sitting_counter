

# MediaPipe Unique Sitting Counter (`mediapipe_sit`)

This project implements a stable and focused solution using **MediaPipe Pose** to detect and count the number of **unique individuals** who exhibit a sitting posture throughout a video. It is designed to replace the problematic dependency stack from previous attempts.

The script processes a video file, applies pose estimation, uses positional heuristics for sitting detection, and assigns temporary IDs to accurately track and count each person only once.

-----
## 📸 Application Screenshot

Here is the live output of the analyzer, showing the pose estimation, tracking ID, and unique sitting count:

<img src="https://github.com/aaslanemre/mediapipe_unique_sitting_counter/blob/main/mask.png" alt="Screenshot of the MediaPipe sitting counter output with pose estimation overlay." width="700">

---
## Prerequisites and Setup

This project is built around the need for a specific, stable Python environment.

### 1\. Requirements

The project uses a minimal set of dependencies to maximize stability:

  * `opencv-python-headless`
  * `numpy`
  * `mediapipe`
  * `python` (Version **3.10** is strongly recommended for MediaPipe compatibility)

### 2\. Installation

Set up your virtual environment and install the dependencies.

```bash
# 1. Ensure Python 3.10 is accessible (e.g., using Homebrew: brew install python@3.10)
# 2. Create the virtual environment using Python 3.10
/opt/homebrew/bin/python3.10 -m venv venv

# 3. Activate the environment
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### 3\. Video Configuration

The `main.py` file uses a relative path to locate the source video. This path must be correct based on where you execute the script relative to the video file.

  * **File to Edit:** `main.py`
  * **Variable:** `VIDEO_SOURCE`


-----

## Logic Overview

### Sitting Detection

The script defines a person as "sitting" by checking two positional heuristics based on side-view analysis:

1.  **Bent Legs:** The vertical distance between the **Hip** and **Knee** landmarks is small (`HIP_KNEE_Y_DIFF_MAX`), indicating the legs are bent (as in sitting).
2.  **Low Position:** The hips are located in the lower half of the frame (`> 40% down`), distinguishing sitting from squatting or other activities.

### Unique Counting

The logic is designed for sequential person detection:

1.  **Temporary ID:** Each person detected in the frame is assigned a temporary ID based on their horizontal hip position.
2.  **Tracking Ledger:** When an ID is detected in a *sitting* state for the first time, it is permanently logged in the `unique_sitting_ids` set.
3.  **Persistence:** Once an ID is logged, it is never counted again, satisfying the "unique people" requirement, even if the person briefly stands up and sits back down.
