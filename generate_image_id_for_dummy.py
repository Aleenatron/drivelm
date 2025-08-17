import json
import os

# Input + Output paths
DUMMY_PROMPTS_PATH = "data/multi_frame/dummyprompts.json"
OUTPUT_IMAGE_ID_PATH = "data/multi_frame/image_id_dummy.json"

with open(DUMMY_PROMPTS_PATH, "r") as f:
    prompts = json.load(f)

image_id_dict = {}

for idx, entry in enumerate(prompts):
    qa = entry[0]
    cams = entry[1]

    question = qa["Q"]
    cam_back_path = cams["CAM_BACK"]

    key = f"{cam_back_path} {question} Answer:"
    image_id_dict[key] = [idx, cams]

with open(OUTPUT_IMAGE_ID_PATH, "w") as f:
    json.dump(image_id_dict, f, indent=2)

print(f"✅ Dummy image_id.json written to: {OUTPUT_IMAGE_ID_PATH}")
