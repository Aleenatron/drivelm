import os
import json

BASE_PATH = "data/nuscenes"
CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]

EASY_QUESTIONS = [
    "How many vehicles are visible in the scene?",
    "What colors are the nearby vehicles?",
    "Is there a pedestrian in the scene?",
    "Is it safe for the ego vehicle to reverse?",
    "What is directly behind the ego vehicle?"
]

OUT_FILE = "dummyprompts.json"
NUM_PROMPTS = 5

# Step 1: Load sorted file lists
cam_files = {}
min_len = float("inf")

for cam in CAMERAS:
    folder = os.path.join(BASE_PATH, cam)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Camera folder missing: {folder}")
    files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])
    cam_files[cam] = files
    min_len = min(min_len, len(files))

if min_len == 0:
    raise ValueError("No image files found in one or more camera folders.")

# Step 2: Create prompts
prompts = []
for i in range(min(NUM_PROMPTS, min_len)):
    frame = {
        cam: os.path.join("data/nuscenes", cam, cam_files[cam][i])
        for cam in CAMERAS
    }
    prompt = [
        {
            "Q": EASY_QUESTIONS[i % len(EASY_QUESTIONS)],
            "A": "",
            "C": None,
            "con_up": None,
            "con_down": None,
            "cluster": None,
            "layer": None
        },
        frame
    ]
    prompts.append(prompt)

# Step 3: Save
with open(OUT_FILE, "w") as f:
    json.dump(prompts, f, indent=4)

print(f"\n✅ Successfully created {len(prompts)} dummy prompts in {OUT_FILE}")

