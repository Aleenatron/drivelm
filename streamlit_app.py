import streamlit as st
import json
import os
from PIL import Image
import subprocess

# ===== CONFIGURATION =====
CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
]
IMAGE_ROOT = "data/nuscenes"
PROMPT_FILE = "data/multi_frame/dummyprompts.json"
IMAGE_ID_FILE = "data/multi_frame/image_id_dummy.json"
PREDICTIONS_FILE = "multi_frame_results/T5-Medium/predictions.json"
EVAL_SCRIPT = "eval.py"

# ===== LOAD PROMPTS & IMAGE-ID MAPPING =====
with open(PROMPT_FILE, "r") as f:
    prompt_entries = json.load(f)

with open(IMAGE_ID_FILE, "r") as f:
    image_id_map = json.load(f)

# ====== Streamlit UI ======
st.set_page_config(layout="wide")
st.title("DriveVLM Multiview Q&A 🔍🚗")

# Build selection list
scene_keys = list(image_id_map.keys())
selected_key = st.selectbox("Select Scene + Question", scene_keys)

# Show Question
question = selected_key.split(" ", 1)[1]
st.markdown(f"### Question:\n**{question}**")

# Show 6 images
cols = st.columns(3)
img_paths = image_id_map[selected_key][1]

for idx, cam in enumerate(CAMERAS):
    img_path = img_paths[cam]
    try:
        image = Image.open(img_path)
        cols[idx % 3].image(image, caption=cam, use_column_width=True)
    except FileNotFoundError:
        cols[idx % 3].error(f"Missing image: {img_path}")

# Run Button
if st.button("Run DriveVLM"):
    with st.spinner("Running LLM model on images..."):
        subprocess.run(["python", EVAL_SCRIPT])

    # Load output
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "r") as f:
            preds = json.load(f)

        # Find match
        image_id = image_id_map[selected_key][0]
        matched = next((p["caption"] for p in preds if p["image_id"] == image_id), None)

        if matched:
            st.success(f"LLM Answer: {matched}")
        else:
            st.error("❌ No matching output found in predictions.json.")
    else:
        st.error("❌ predictions.json not found after inference.")
