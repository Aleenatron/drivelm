import streamlit as st
import json
import os
from PIL import Image
import subprocess

# ===== Constants =====
CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
PROMPT_FILE = "data/multi_frame/dummyprompts.json"
IMAGE_ID_FILE = "data/multi_frame/image_id_dummy.json"
PREDICTIONS_FILE = "multi_frame_results/T5-Medium/predictions.json"
EVAL_SCRIPT = "eval.py"

# ===== Load Data =====
with open(PROMPT_FILE, "r") as f:
    prompt_entries = json.load(f)

with open(IMAGE_ID_FILE, "r") as f:
    image_id_map = json.load(f)

# ===== Page Config & Styling =====
st.set_page_config(layout="centered", page_title="DriveVLM Q&A", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: "Segoe UI", sans-serif;
        font-size: 14px;
    }
    .stButton > button {
        width: 100%;
        padding: 0.5rem 1rem;
        font-size: 14px;
        border-radius: 6px;
        background-color: #005eb8;
        color: white;
        font-weight: 500;
    }
    .stImage img {
        border-radius: 6px;
        margin-bottom: 0.2rem;
    }
    .question-box, .answer-box {
        padding: 0.8rem;
        margin-top: 1rem;
        border-radius: 6px;
    }
    .question-box {
        background-color: #eef4fa;
        border-left: 4px solid #005eb8;
    }
    .answer-box {
        background-color: #e9fbe9;
        border-left: 4px solid #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚗 DriveVLM: Ask Questions based on 6 camera feeds")

# ===== Select a Question =====
questions = [entry[0]["Q"] for entry in prompt_entries]
question_to_index = {q: i for i, q in enumerate(questions)}
selected_question = st.selectbox("Select a question to run:", questions)

index = question_to_index[selected_question]
question_block = prompt_entries[index]
img_paths = question_block[1]

# ===== Buttons Row =====
col1, col2 = st.columns(2)
with col1:
    run_triggered = st.button("▶️ Run DriveVLM")
with col2:
    summarize_triggered = st.button("📝 Summarize Scene")

# ===== Display Question =====
st.markdown(f"<div class='question-box'><b>User:</b> {selected_question}</div>", unsafe_allow_html=True)

# ===== Helper to Load and Resize Images =====
def load_image(path, width=180):
    img = Image.open(path)
    w_percent = width / float(img.width)
    h_size = int(float(img.height) * w_percent)
    return img.resize((width, h_size))

# ===== Top-down Grid Layout =====
row1 = st.columns(3)
row2 = st.columns(3)

row1[0].image(load_image(img_paths["CAM_FRONT_LEFT"]), caption="Front Left", use_container_width=False)
row1[1].image(load_image(img_paths["CAM_FRONT"]), caption="Front", use_container_width=False)
row1[2].image(load_image(img_paths["CAM_FRONT_RIGHT"]), caption="Front Right", use_container_width=False)

row2[0].image(load_image(img_paths["CAM_BACK_LEFT"]), caption="Back Left", use_container_width=False)
row2[1].image(Image.open("car_top_down.png").resize((140, 140)), caption="Ego", use_container_width=False)
row2[2].image(load_image(img_paths["CAM_BACK_RIGHT"]), caption="Back Right", use_container_width=False)

# ===== Show CAM_BACK Below Layout =====
with row2[1]:
    st.image(load_image(img_paths["CAM_BACK"]), caption="Back View", use_container_width=False)

# ===== Run Inference and Show Answer =====
if run_triggered or summarize_triggered:
    with st.spinner("Running model inference..."):
        subprocess.run(["python", EVAL_SCRIPT])

    if os.path.exists(PREDICTIONS_FILE):
        with open(PREDICTIONS_FILE, "r") as f:
            preds = json.load(f)

        image_key = list(image_id_map.keys())[index]
        image_id = image_id_map[image_key][0]
        matched = next((p["caption"] for p in preds if p["image_id"] == image_id), None)

        if matched:
            st.markdown(f"<div class='answer-box'><b>DriveVLM:</b> {matched}</div>", unsafe_allow_html=True)
        else:
            st.error("❌ No prediction found.")
    else:
        st.error("❌ `predictions.json` not found.")




