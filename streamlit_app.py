import streamlit as st
import json
import os
from PIL import Image
import subprocess

# === CONFIG ===
CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
DATA_DIR = "data/nuscenes"
PROMPT_FILE = "data/multi_frame/dummyprompts.json"
IMAGE_MAP_FILE = "data/multi_frame/image_id_dummy.json"
PREDICTIONS_FILE = "multi_frame_results/predictions.json"

# === Load JSONs ===
with open(PROMPT_FILE, "r") as f:
    prompts = json.load(f)

with open(IMAGE_MAP_FILE, "r") as f:
    image_map = json.load(f)

# === UI ===
st.title("DriveVLM Multi-View QA 🚗📷")
scene_ids = list(prompts.keys())
selected_scene = st.selectbox("Select a Scene ID", scene_ids)

# === Display Images ===
st.subheader("Camera Views")
cols = st.columns(3)
image_ids = image_map[selected_scene]

for idx, cam in enumerate(CAMERAS):
    img_path = os.path.join(DATA_DIR, cam, image_ids[idx].split('/')[-1])
    try:
        img = Image.open(img_path)
        cols[idx // 3].image(img, caption=cam, use_column_width=True)
    except FileNotFoundError:
        cols[idx // 3].warning(f"Image not found: {img_path}")

# === Display Question ===
st.subheader("Question")
question = prompts[selected_scene]["Q"]
st.text(question)

# === Answer Section ===
if st.button("Get Answer"):
    # Call eval.py
    with st.spinner("Running model..."):
        subprocess.run(["python3", "modules/eval.py"])  # Update this if eval.py takes arguments

    # Load predictions
    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "r") as f:
            preds = json.load(f)

        answer = next((item["caption"] for item in preds if item["image_id"] == int(selected_scene)), None)
        if answer:
            st.success(f"Answer: {answer}")
        else:
            st.error("No answer found for this scene.")
    else:
        st.error("predictions.json not found.")
