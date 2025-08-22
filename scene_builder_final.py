import os, json, sys
from collections import defaultdict

# --- Config ---
DATA_FOLDER = "data/nuscenes"
IMAGE_ID_JSON = "data/multi_frame/final_image_id_dummy.json"
PROMPT_JSON   = "data/multi_frame/finaldummyprompts.json"
CAMERAS = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT",
           "CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]

# --- Your Finalized Question Bank (30 Qs) ---
ALL_QUESTIONS = [
    # Binary
    "Is there a pedestrian in the scene?",
    "Is it safe for the ego vehicle to reverse?",
    "Is it safe for the ego vehicle to turn right?",
    "Is the road free of traffic?",
    "Are there moving pedestrians to the back left of the ego car?",
    "Are there parked cars to the front of the ego car?",
    # Count
    "How many vehicles are visible in the scene?",
    "How many pedestrians are present in the scene?",
    "How many traffic cones are behind the ego car?",
    "How many lanes are visible on the road?",
    "How many traffic lights are visible in the front view?",
    # Perception
    "What is directly behind the ego vehicle?",
    "What are the important objects in the scene?",
    "What are objects to the back left of the ego car?",
    "What are objects to the front right of the ego car?",
    "What is the status of the pedestrians that are to the front of the ego car?",
    "What is the type of vehicle immediately in front of the ego car?",
    # Spatial
    "What is the relative position of important objects in the scene?",
    "What is the relative positioning of the important objects in the current scene?",
    "Where might the van, sedan, and pedestrian move in the future?",
    "Based on <CAM_FRONT, x, y>, what is the most possible action of the ego vehicle?",
    "Based on <CAM_BACK, x, y>, what is the most possible action of the ego vehicle?",
    # Planning
    "What are the safe actions of the ego car considering those objects?",
    "What actions taken by the ego vehicle can lead to a collision with object?",
    "Risks of ego vehicle based on its position now?",
    "In this scenario, what are dangerous actions to take for the ego vehicle?",
    # Summary
    "Summarize the scene.",
    "Please describe the current scene.",
    "What’s your comment on this scene?",
    "Comment on road and traffic condition."
]

def build_scene_index():
    scene_dict = defaultdict(dict)
    for cam in CAMERAS:
        folder = os.path.join(DATA_FOLDER, cam)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".jpg"):
                continue
            parts = fname.split("__")
            if len(parts) < 3:
                continue
            scene_key = parts[0]
            ts = parts[2].split(".")[0]
            full_path = os.path.join(folder, fname)
            prev = scene_dict[scene_key].get(cam)
            if not prev or ts > os.path.basename(prev).split("__")[-1].split(".")[0]:
                scene_dict[scene_key][cam] = full_path
    return {k:v for k,v in scene_dict.items() if len(v)==6}

def add_entry(question, cams, nid, image_id_data, prompts):
    img_key = cams["CAM_BACK"]
    question_key = f"{img_key} {question.strip()} Answer:"

    image_id_data[question_key] = [nid, cams]
    prompts.append([
        {"Q": question, "A": "", "C": None, "con_up": None, "con_down": None, "cluster": None, "layer": None},
        cams
    ])

def main():
    dry_run = "--dry-run" in sys.argv

    scenes = build_scene_index()
    if not scenes:
        print("❌ No complete 6-camera scenes found under", DATA_FOLDER)
        return

    # reset fresh each time
    image_id_data = {}
    prompts = []
    nid = 0

    print(f"\n🧠 Populating {len(scenes)} scenes × {len(ALL_QUESTIONS)} questions = {len(scenes)*len(ALL_QUESTIONS)} entries...")

    for scene_key, cams in scenes.items():
        print(f"➡️ Scene: {scene_key}")
        for q in ALL_QUESTIONS:
            add_entry(q, cams, nid, image_id_data, prompts)
            nid += 1

    if dry_run:
        print("\n✅ Dry run complete. Nothing was written.")
        print(json.dumps(list(image_id_data.items())[:2], indent=2))  # show sample
    else:
        os.makedirs(os.path.dirname(IMAGE_ID_JSON), exist_ok=True)
        with open(IMAGE_ID_JSON, "w") as f:
            json.dump(image_id_data, f, indent=2)
        with open(PROMPT_JSON, "w") as f:
            json.dump(prompts, f, indent=2)

        print(f"\n✅ Finished writing:\n  {IMAGE_ID_JSON}\n  {PROMPT_JSON}")

if __name__ == "__main__":
    main()
