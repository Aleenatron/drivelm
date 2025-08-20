import os, json, sys
from collections import defaultdict

# --- Config ---
DATA_FOLDER = "data/nuscenes"
IMAGE_ID_JSON = "data/multi_frame/image_id_dummy.json"
PROMPT_JSON   = "data/multi_frame/dummyprompts.json"
CAMERAS = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT",
           "CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]

EASY_QUESTIONS = [
    "How many vehicles are visible in the scene?",
    "What colors are the nearby vehicles?",
    "Is there a pedestrian in the scene?",
    "Is it safe for the ego vehicle to reverse?",
    "What is directly behind the ego vehicle?",
    "Summarize the scene",
    "What is the relative position of important objects in the scene?",
    "Is it safe for ego vehicle to right turn?",
    "Risks of ego vehicle based on its position now?",
    "What are the important objects in the scene?"
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

def next_image_id(image_id_json):
    if not os.path.exists(image_id_json):
        return 0
    with open(image_id_json, "r") as f:
        d = json.load(f)
    if not isinstance(d, dict) or not d:
        return 0
    try:
        return max(v[0] for v in d.values()) + 1
    except Exception:
        return 0

def add_entry(question, cams, nid, dry_run=False):
    """Either preview or write entries to both JSON files"""
    img_key = cams["CAM_BACK"]
    question_key = f"{img_key} {question.strip()} Answer:"

    # build both entries
    image_entry = {question_key: [nid, cams]}
    prompt_entry = [
        {
            "Q": question,
            "A": "",
            "C": None,
            "con_up": None,
            "con_down": None,
            "cluster": None,
            "layer": None
        },
        cams
    ]

    if dry_run:
        print(f"\n➡️ Would add to image_id_dummy.json:")
        print(json.dumps(image_entry, indent=2))
        print(f"\n➡️ Would append to dummyprompts.json:")
        print(json.dumps(prompt_entry, indent=2))
    else:
        # ---- image_id_dummy.json ----
        if os.path.exists(IMAGE_ID_JSON):
            with open(IMAGE_ID_JSON, "r") as f:
                image_id_data = json.load(f)
        else:
            image_id_data = {}
        image_id_data.update(image_entry)
        with open(IMAGE_ID_JSON, "w") as f:
            json.dump(image_id_data, f, indent=2)

        # ---- dummyprompts.json ----
        if os.path.exists(PROMPT_JSON):
            with open(PROMPT_JSON, "r") as f:
                prompts = json.load(f)
        else:
            prompts = []
        prompts.append(prompt_entry)
        with open(PROMPT_JSON, "w") as f:
            json.dump(prompts, f, indent=2)

        print(f"✅ Written: '{question}' with ID {nid}")

def main():
    dry_run = "--dry-run" in sys.argv

    scenes = build_scene_index()
    if not scenes:
        print("❌ No complete 6-camera scenes found under", DATA_FOLDER)
        return

    print("\n🧠 Available scene samples (complete 6 views):")
    scene_keys = sorted(scenes.keys())
    for i, sk in enumerate(scene_keys, 1):
        print(f"{i}. {sk} -> {list(scenes[sk].keys())}")

    sel = input("\n➡️ Pick a scene number or scene key: ").strip()
    if sel.isdigit() and 1 <= int(sel) <= len(scene_keys):
        scene_key = scene_keys[int(sel)-1]
    elif sel in scene_keys:
        scene_key = sel
    else:
        print("❌ Invalid selection.")
        return

    cams = scenes[scene_key]

    # select question
    print("\n📋 Available easy questions:")
    for i, q in enumerate(EASY_QUESTIONS, 1):
        print(f"{i}. {q}")
    qsel = input("➡️ Pick a question number (or type your own): ").strip()
    if qsel.isdigit() and 1 <= int(qsel) <= len(EASY_QUESTIONS):
        question = EASY_QUESTIONS[int(qsel)-1]
    else:
        question = qsel if qsel else EASY_QUESTIONS[0]

    nid = next_image_id(IMAGE_ID_JSON)

    print("\n🔎 Processing...\n")

    # 1. user-selected
    add_entry(question, cams, nid, dry_run=dry_run)

    # 2. auto summarize
    add_entry("Summarize the scene", cams, nid+1, dry_run=dry_run)

    if dry_run:
        print("\n✅ Dry run complete. Nothing was written.")
    else:
        print("\n✅ Both entries written successfully!")

if __name__ == "__main__":
    main()
