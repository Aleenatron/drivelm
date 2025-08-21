import streamlit as st
import json
import os
from PIL import Image
import subprocess
from pathlib import Path

# ===== Constants =====
CAMERAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
PROMPT_FILE = "data/multi_frame/dummyprompts.json"
IMAGE_ID_FILE = "data/multi_frame/image_id_dummy.json"
PREDICTIONS_FILE = "multi_frame_results/T5-Medium/predictions.json"
EVAL_SCRIPT = "eval.py"
DUMMY_COCO_FILE = "data/multi_frame/dummy_multi_frame_test_coco.json"
METRICS_CSV = "multi_frame_results/T5-Medium/metrics.csv"   # where eval.py writes scores

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
        # NOTE: if you added --annotation-file support in eval.py, it is not needed here.
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

# =========================
# ==== COCO helpers (safe)
# =========================
COCO_TEMPLATE = {
    "info": {},
    "licenses": [],
    "type": "captions",
    "images": [],
    "annotations": []
}

def _load_json(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else {}
    with open(path, "r") as f:
        return json.load(f)

def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False)

def _init_or_load_coco(path):
    coco = _load_json(path, default=None)
    if not coco or not isinstance(coco, dict):
        coco = {**COCO_TEMPLATE}
        coco["images"] = []
        coco["annotations"] = []
    coco.setdefault("info", {})
    coco.setdefault("licenses", [])
    coco.setdefault("type", "captions")
    coco.setdefault("images", [])
    coco.setdefault("annotations", [])
    return coco

def _image_index_by_id(coco, image_id):
    for i, im in enumerate(coco["images"]):
        if im.get("id") == image_id:
            return i
    return -1

def _ensure_image_entry(coco, image_id, file_name, width=None, height=None):
    idx = _image_index_by_id(coco, image_id)
    if idx == -1:
        entry = {"id": image_id, "file_name": file_name}
        if isinstance(width, int) and width > 0: entry["width"] = width
        if isinstance(height, int) and height > 0: entry["height"] = height
        coco["images"].append(entry)
    else:
        # keep width/height if present; update file_name if changed
        coco["images"][idx]["file_name"] = file_name

def _next_ann_id(coco):
    return (max((a.get("id", 0) for a in coco["annotations"]), default=0) + 1)

def _add_annotation(coco, image_id, caption):
    coco["annotations"].append({
        "image_id": image_id,
        "id": _next_ann_id(coco),
        "caption": caption
    })

def _validate_coco_minimal(coco):
    assert coco.get("type") == "captions", "COCO[type] must be 'captions'"
    assert isinstance(coco.get("images"), list), "COCO[images] must be list"
    assert isinstance(coco.get("annotations"), list), "COCO[annotations] must be list"
    img_ids = {im.get("id") for im in coco["images"]}
    for ann in coco["annotations"]:
        assert "image_id" in ann and "caption" in ann and "id" in ann, "Annotation missing fields"
        assert ann["image_id"] in img_ids, f"Annotation image_id {ann['image_id']} missing in images"
    return True

# ===============================
# ==== Manual GT → COCO builder
# ===============================
st.divider()
st.subheader("✍️ Add manual ground-truth caption (COCO format)")

# current selection context
image_key = list(image_id_map.keys())[index]
image_id = image_id_map[image_key][0]
cam_back_path = img_paths["CAM_BACK"]

st.write(f"**Image ID:** `{image_id}`")
st.write(f"**COCO file_name:** `{cam_back_path}`")

# optional preview (not required by eval)
if os.path.exists(cam_back_path):
    try:
        st.image(Image.open(cam_back_path), caption=Path(cam_back_path).name, use_container_width=True)
    except Exception as e:
        st.warning(f"Preview failed: {e}")

# load or init dummy COCO file
coco = _init_or_load_coco(DUMMY_COCO_FILE)

# show existing refs (helps with style consistency)
existing_caps = [a["caption"] for a in coco["annotations"] if a["image_id"] == image_id]
if existing_caps:
    st.caption(f"Existing refs for image_id {image_id}:")
    st.code("\n".join(existing_caps[-3:]), language="text")

# input GT caption
gt_caption = st.text_area(
    "Ground-truth caption / answer (case & style matter for metrics)",
    placeholder="e.g., Two vehicles are visible behind the ego vehicle.",
    height=100
)

col_gt1, col_gt2, col_gt3 = st.columns(3)
with col_gt1:
    width_opt = st.number_input("width (optional)", min_value=0, value=0)
with col_gt2:
    height_opt = st.number_input("height (optional)", min_value=0, value=0)
with col_gt3:
    allow_multi = st.checkbox("Allow multiple refs per image_id", value=True)

if st.button("➕ Append to COCO GT", type="primary", disabled=not gt_caption.strip()):
    # 1) ensure image entry
    _ensure_image_entry(
        coco,
        image_id=image_id,
        file_name=cam_back_path,
        width=int(width_opt) if width_opt else None,
        height=int(height_opt) if height_opt else None
    )
    # 2) add annotation (COCO supports multiple per image)
    if (not allow_multi) and existing_caps:
        st.error("Annotation exists and 'allow multiple' is off. Enable it or edit the JSON manually.")
    else:
        _add_annotation(coco, image_id=image_id, caption=gt_caption.strip())
        try:
            _validate_coco_minimal(coco)
            _save_json(DUMMY_COCO_FILE, coco)
            st.success(f"Saved {DUMMY_COCO_FILE} • images={len(coco['images'])} • annotations={len(coco['annotations'])}")
        except AssertionError as e:
            st.error(f"COCO validation failed: {e}")

# schema quick check & preview
with st.expander("Schema check / preview"):
    try:
        _validate_coco_minimal(coco)
        st.success("COCO schema OK ✅")
    except AssertionError as e:
        st.error(f"Schema issue: {e}")
    st.caption("Last 5 annotations:")
    st.code(json.dumps(coco["annotations"][-5:], ensure_ascii=False, indent=2), language="json")

# ==========================================
# ==== Run COCOEvalCap on the manual GT
# ==========================================
st.subheader("📏 Evaluate predictions against manual COCO GT")
if st.button("🏁 Run COCOEvalCap on manual GT", use_container_width=True):
    with st.spinner("Running evaluation..."):
        # Requires you added --annotation-file support in eval.py (params + use in COCO block)
        subprocess.run(["python", EVAL_SCRIPT, "--annotation-file", DUMMY_COCO_FILE])
    if os.path.exists(METRICS_CSV):
        st.success("Evaluation complete. Showing metrics.csv below:")
        try:
            st.code(open(METRICS_CSV).read(), language="text")
        except Exception as e:
            st.warning(f"Could not read metrics.csv: {e}")
    else:
        st.warning("metrics.csv not found—check eval logs/paths.")
