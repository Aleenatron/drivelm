import streamlit as st
import json, os
from pathlib import Path
from PIL import Image
import subprocess

# =========================
# Constants (FINAL files)
# =========================
CAMERAS = ["CAM_FRONT","CAM_FRONT_LEFT","CAM_FRONT_RIGHT","CAM_BACK","CAM_BACK_LEFT","CAM_BACK_RIGHT"]
PROMPT_FILE = "data/multi_frame/finaldummyprompts.json"
IMAGE_ID_FILE = "data/multi_frame/final_image_id_dummy.json"
PREDICTIONS_FILE = "multi_frame_results/T5-Medium/finalpredictions.json"
EVAL_SCRIPT = "eval.py"

# Optional COCO helper outputs (unchanged behaviour)
FINAL_COCO_FILE = "data/multi_frame/final_multi_frame_test_coco.json"
METRICS_CSV = "multi_frame_results/T5-Medium/finalmetrics.csv"

# =========================
# Small helpers
# =========================
def _load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)

def _scene_key_from_cams(cams_dict):
    """Use CAM_BACK filename prefix as scene key."""
    back = cams_dict.get("CAM_BACK", "")
    stem = Path(back).name  # e.g., n015-2018-10-08-16-03-24+0800__CAM_BACK__...
    return stem.split("__")[0] if "__" in stem else stem

def _resize_image(path, width=180):
    img = Image.open(path)
    r = width / float(img.width)
    return img.resize((width, int(img.height * r)))

# =========================
# Load data (read-only)
# =========================
prompt_entries = _load_json(PROMPT_FILE, default=[])
image_id_map   = _load_json(IMAGE_ID_FILE, default={})

# Group prompts by scene: scene_key -> list of (index_in_prompts, question, cams_dict)
prompts_by_scene = {}
for idx, entry in enumerate(prompt_entries):
    if not isinstance(entry, list) or len(entry) != 2: 
        continue
    qd, cams = entry
    if not isinstance(qd, dict) or "Q" not in qd: 
        continue
    skey = _scene_key_from_cams(cams)
    prompts_by_scene.setdefault(skey, []).append((idx, qd["Q"], cams))

# =========================
# UI
# =========================
st.set_page_config(layout="centered", page_title="DriveVLM Q&A", initial_sidebar_state="collapsed")
st.title("🚗 DriveVLM: Ask Questions based on 6 camera feeds")

if not prompts_by_scene:
    st.error("No prompts found in finaldummyprompts.json")
    st.stop()

# 1) pick a scene (49 options)
scene_keys = sorted(prompts_by_scene.keys())
scene_sel = st.selectbox("Select a scene:", scene_keys, index=0)

# 2) pick any question that belongs to this scene (we list all for the scene)
scene_prompts = prompts_by_scene[scene_sel]
q_labels = [q for (_i, q, _c) in scene_prompts]
q_choice = st.selectbox("Select a question for this scene:", q_labels, index=0)

# find the selected (index, question, cams)
sel_idx, selected_question, img_paths = next(p for p in scene_prompts if p[1] == q_choice)

# --- show the chosen question
st.markdown(
    "<div style='background:#eef4fa;border-left:4px solid #005eb8;padding:.8rem;border-radius:6px;'>"
    f"<b>User:</b> {selected_question}</div>", unsafe_allow_html=True
)

# --- image grid
row1 = st.columns(3); row2 = st.columns(3)
row1[0].image(_resize_image(img_paths["CAM_FRONT_LEFT"]),  caption="Front Left")
row1[1].image(_resize_image(img_paths["CAM_FRONT"]),       caption="Front")
row1[2].image(_resize_image(img_paths["CAM_FRONT_RIGHT"]), caption="Front Right")
row2[0].image(_resize_image(img_paths["CAM_BACK_LEFT"]),   caption="Back Left")
row2[1].image(Image.open("car_top_down.png").resize((140, 140)), caption="Ego")
row2[2].image(_resize_image(img_paths["CAM_BACK_RIGHT"]),  caption="Back Right")
with row2[1]:
    st.image(_resize_image(img_paths["CAM_BACK"]), caption="Back View")

# --- controls
c1, c2 = st.columns(2)
with c1: run_triggered = st.button("▶️ Run DriveVLM")
with c2: summarize_triggered = st.button("📝 Summarize Scene")

# =========================
# Run inference & display
# =========================
if run_triggered or summarize_triggered:
    with st.spinner("Running model inference…"):
        # Your eval.py writes finalpredictions.json for all prompts; safe to reuse.
        subprocess.run(["python", EVAL_SCRIPT])

    preds = _load_json(PREDICTIONS_FILE, default=[])
    # Build the same key used in final_image_id_dummy.json
    key_string = f"{img_paths['CAM_BACK']} {selected_question} Answer:"
    if key_string not in image_id_map:
        st.error("Mapping for this (scene, question) not found in final_image_id_dummy.json.")
    else:
        image_id = image_id_map[key_string][0]
        ans = next((p["caption"] for p in preds if p["image_id"] == image_id), None)
        if ans:
            st.markdown(
                "<div style='background:#e9fbe9;border-left:4px solid #2e7d32;padding:.8rem;border-radius:6px;'>"
                f"<b>DriveVLM:</b> {ans}</div>", unsafe_allow_html=True
            )
        else:
            st.warning("Inference ran, but no prediction for this item yet.")

# =========================
# (Optional) COCO helpers
# =========================
COCO_TEMPLATE = {"info":{},"licenses":[],"type":"captions","images":[],"annotations":[]}
def _init_or_load_coco(path):
    coco = _load_json(path, None) or {**COCO_TEMPLATE}
    for k,v in COCO_TEMPLATE.items():
        coco.setdefault(k, v if not isinstance(v, list) else [])
    return coco
def _image_idx(coco, image_id):
    for i, im in enumerate(coco["images"]):
        if im.get("id")==image_id: return i
    return -1
def _ensure_image(coco, image_id, file_name, w=None, h=None):
    i = _image_idx(coco, image_id)
    if i==-1:
        e={"id":image_id,"file_name":file_name}
        if w: e["width"]=int(w)
        if h: e["height"]=int(h)
        coco["images"].append(e)
def _next_ann_id(coco): return max((a.get("id",0) for a in coco["annotations"]), default=0)+1
def _add_ann(coco, image_id, caption):
    coco["annotations"].append({"image_id":image_id,"id":_next_ann_id(coco),"caption":caption})
def _validate(coco):
    assert coco.get("type")=="captions"
    ids={im["id"] for im in coco["images"]}
    for a in coco["annotations"]:
        assert "image_id" in a and "caption" in a and "id" in a
        assert a["image_id"] in ids
    return True

st.divider()
st.subheader("✍️ Add manual ground-truth caption (COCO format)")

# Context based on current selection
key_string = f"{img_paths['CAM_BACK']} {selected_question} Answer:"
if key_string in image_id_map:
    image_id = image_id_map[key_string][0]
    cam_back_path = img_paths["CAM_BACK"]
    st.write(f"**Image ID:** `{image_id}`")
    st.write(f"**COCO file_name:** `{cam_back_path}`")

    if os.path.exists(cam_back_path):
        try: st.image(Image.open(cam_back_path), caption=Path(cam_back_path).name, use_container_width=True)
        except Exception as e: st.warning(f"Preview failed: {e}")

    coco = _init_or_load_coco(FINAL_COCO_FILE)
    existing = [a["caption"] for a in coco["annotations"] if a["image_id"]==image_id]
    if existing:
        st.caption(f"Existing refs for image_id {image_id}:")
        st.code("\n".join(existing[-3:]), language="text")

    gt = st.text_area("Ground-truth caption / answer", height=100)
    c1, c2, c3 = st.columns(3)
    with c1: w = st.number_input("width (optional)", 0, value=0)
    with c2: h = st.number_input("height (optional)", 0, value=0)
    with c3: allow_multi = st.checkbox("Allow multiple refs per image_id", value=True)

    if st.button("➕ Append to COCO GT", type="primary", disabled=not gt.strip()):
        _ensure_image(coco, image_id, cam_back_path, w or None, h or None)
        if (not allow_multi) and existing:
            st.error("Annotation exists and 'allow multiple' is off.")
        else:
            _add_ann(coco, image_id, gt.strip())
            try:
                _validate(coco)
                with open(FINAL_COCO_FILE, "w") as f: json.dump(coco, f, ensure_ascii=False)
                st.success(f"Saved {FINAL_COCO_FILE} • images={len(coco['images'])} • annotations={len(coco['annotations'])}")
            except AssertionError as e:
                st.error(f"COCO validation failed: {e}")
else:
    st.info("This (scene, question) is missing from image_id map.")
