import os
import json
import random
from collections import defaultdict

DATA_FOLDER = "data/nuscenes/"
IMAGE_ID_JSON = "data/multi_frame/image_id_dummy.json"

# Step 1: Group by scene, pick latest timestamp per camera
scene_dict = defaultdict(dict)
for root, _, files in os.walk(DATA_FOLDER):
    for file in sorted(files):
        if not file.endswith(".jpg"):
            continue

        cam_name = os.path.basename(root)
        full_path = os.path.join(root, file)

        parts = file.split("__")
        if len(parts) < 3:
            continue

        scene_key = parts[0]
        timestamp = parts[2].split(".")[0]

        prev = scene_dict[scene_key].get(cam_name)
        if not prev or timestamp > prev.split("__")[-1].split(".")[0]:
            scene_dict[scene_key][cam_name] = full_path

# Step 2: List and select scene
scenes = {k: v for k, v in scene_dict.items() if len(v) == 6}
print("\n🧠 Available scene samples:")
for i, (scene_key, cams) in enumerate(scenes.items()):
    print(f"{i+1}. {scene_key} -> {list(cams.keys())}")
    if i == 4:
        break

scene_key = input("\n✍️ Enter a scene key from above: ").strip()
question = input("✍️ Enter a question (e.g., Summarize the scene): ").strip()

cams = scene_dict.get(scene_key, {})
if len(cams) < 6:
    print(f"❌ Scene '{scene_key}' does not have all 6 views.")
    exit()

# Step 3: Construct key and load JSON
img_key = cams["CAM_BACK"]
question_key = f"{img_key} {question.strip()} Answer:"

with open(IMAGE_ID_JSON, "r") as f:
    image_id_data = json.load(f)

existing_ids = [v[0] for v in image_id_data.values()]
new_id = max(existing_ids) + 1 if existing_ids else 0

# Step 4: Add new entry and save
image_id_data[question_key] = [new_id, cams]

with open(IMAGE_ID_JSON, "w") as f:
    json.dump(image_id_data, f, indent=2)

print(f"\n✅ Added new entry to image_id_dummy.json with ID {new_id}")

