import json

DUMMY_PROMPTS_PATH = "data/multi_frame/dummyprompts.json"
IMAGE_ID_PATH = "data/multi_frame/image_id_dummy.json"

CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def create_key(cam_back_img, question):
    return f"{cam_back_img} {question.strip()} Answer:"

def merge_and_preview(dummy_prompts, image_id_data, write=False):
    existing_keys = set(image_id_data.keys())
    next_id = max([v[0] for v in image_id_data.values()], default=0) + 1
    new_entries = 0
    
    for item in dummy_prompts:
        if not isinstance(item, list) or len(item) != 2:
            print("⚠️ Skipping malformed item (not a 2-element list):", item)
            continue

        q_obj, cam_obj = item

        if not isinstance(q_obj, dict) or "Q" not in q_obj:
            print("⚠️ Skipping item without valid question dict:", q_obj)
            continue

        if not isinstance(cam_obj, dict):
            print("⚠️ Skipping item without valid camera dict:", cam_obj)
            continue

        missing_keys = [cam for cam in CAMERAS if cam not in cam_obj]
        if missing_keys:
            print(f"⚠️ Skipping due to missing cameras: {missing_keys}")
            continue

        question = q_obj["Q"].strip()
        cam_back_img = cam_obj["CAM_BACK"].strip()
        key = create_key(cam_back_img, question)

        print(f"\n🔍 Checking key: {key}")
        if key in existing_keys:
            print("✅ Skipping duplicate key")
            continue

        # Add the entry
        image_dict = {cam: cam_obj[cam] for cam in CAMERAS}
        image_id_data[key] = [next_id, image_dict]

        print(f"➕ Adding new entry (id {next_id}):")
        print(json.dumps({key: [next_id, image_dict]}, indent=2))

        next_id += 1
        new_entries += 1

def main():
    dummy_prompts = load_json(DUMMY_PROMPTS_PATH)
    image_id_data = load_json(IMAGE_ID_PATH)
    merge_and_preview(dummy_prompts, image_id_data, write=False)  # set write=True to write to disk

if __name__ == "__main__":
    main()
