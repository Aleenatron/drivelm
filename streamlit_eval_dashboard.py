# streamlit_eval_dashboard.py
import streamlit as st
import json, os, pandas as pd, pathlib
from datetime import datetime, timezone
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO  # <-- for downloadable PNG

# ---------- Paths ----------
PROMPT_FILE = "data/multi_frame/dummyprompts.json"
IMAGE_ID_FILE = "data/multi_frame/image_id_dummy.json"
PREDICTIONS_FILE = "multi_frame_results/T5-Medium/predictions.json"
RATINGS_FILE = "data/multi_frame/human_ratings.json"
OUT_DIR = pathlib.Path("multi_frame_results")

# ---------- Categories ----------
# canonical categories (color is merged into perception)
CATEGORIES = ["binary", "count", "spatial", "planning", "perception", "summary"]

# pretty labels shown in UI (map -> canonical key)
CATEGORY_LABELS = {
    "binary":      "binary",
    "count":       "count",
    "spatial":     "spatial (relative position / where)",
    "planning":    "planning",
    "perception":  "perception (what / which / how many)",
    "summary":     "summary",
}
LABEL_TO_CANON = {v: k for k, v in CATEGORY_LABELS.items()}

st.set_page_config(page_title="DriveVLM: Human Reliability & QA", layout="wide")
st.title("🧑‍⚖️ Human Reliability & Category Evaluation")

# ---------- helpers ----------
@st.cache_data
def jload(path, default=None):
    if not os.path.exists(path): return default
    with open(path) as f: return json.load(f)

@st.cache_data
def jloadl(path):
    if not os.path.exists(path): return []
    with open(path) as f: return [json.loads(l) for l in f if l.strip()]

@st.cache_data
def load_img(path, width=380):
    try:
        im = Image.open(path)
        w = width
        h = int(im.height * (w / im.width))
        return im.resize((w, h))
    except Exception:
        return None

# ---------- data ----------
prompts = jload(PROMPT_FILE, [])
image_id_map = jload(IMAGE_ID_FILE, {})
preds = jload(PREDICTIONS_FILE, [])

pred_by_id = {int(p["image_id"]): p["caption"] for p in (preds or [])}

# Build rows: (image_id, question, cams, pred)
rows = []
for entry in (prompts or []):
    if not isinstance(entry, list) or len(entry) < 2:
        continue
    q = entry[0].get("Q","")
    cams = entry[1]
    cam_back = cams.get("CAM_BACK","")
    key = f"{cam_back} {q} Answer:"
    if key in image_id_map:
        img_id = int(image_id_map[key][0])
        rows.append({
            "image_id": img_id,
            "question": q,
            "cams": cams,
            "pred": pred_by_id.get(img_id, "")
        })

# ---------- sidebar ----------
st.sidebar.header("Controls")
cat_label = st.sidebar.selectbox(
    "Category",
    [CATEGORY_LABELS[c] for c in CATEGORIES],
    index=0
)
cat = LABEL_TO_CANON[cat_label]  # store canonical category

if not rows:
    st.info("No prompts found.")
    st.stop()

options = [f"[{r['image_id']}] {r['question'][:80]}" for r in rows]
sel_idx = st.sidebar.selectbox("Pick item", list(range(len(rows))), format_func=lambda i: options[i], index=0)

item = rows[sel_idx]
img_id = item["image_id"]
cams = item["cams"]

# ---------- image grid ----------
st.subheader(f"Scene preview — image_id {img_id}")
c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)

def show(col, path, label):
    im = load_img(path)
    if im is not None:
        col.image(im, caption=label)
    else:
        col.error(f"Missing: {path}")

show(c1, cams["CAM_FRONT_LEFT"],  "Front Left")
show(c2, cams["CAM_FRONT"],       "Front")
show(c3, cams["CAM_FRONT_RIGHT"], "Front Right")
show(c4, cams["CAM_BACK_LEFT"],   "Back Left")
show(c5, cams["CAM_BACK"],        "Back")
show(c6, cams["CAM_BACK_RIGHT"],  "Back Right")

# ---------- question + model answer + rating ----------
st.markdown("### Question")
st.markdown(f"**{item['question']}**")

st.markdown("### Model answer")
st.info(item["pred"] or "(no prediction)")

with st.form("rate", clear_on_submit=False):
    rel = st.radio("Reliability", ["✅ reliable","❌ unreliable","⚠️ fuzzy"], horizontal=True, index=0)
    notes = st.text_input("Notes (optional)")
    save_btn = st.form_submit_button("💾 Save rating")

if save_btn:
    rel_map = {"✅ reliable": True, "❌ unreliable": False, "⚠️ fuzzy": None}
    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_id": int(img_id),
        "question": item["question"],
        "pred": item["pred"],
        "category": cat,           # canonical key saved
        "reliable": rel_map[rel],
        "notes": notes.strip()
    }
    pathlib.Path(RATINGS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(RATINGS_FILE, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # Clear caches so newly appended data shows up immediately
    jloadl.clear()
    st.success("Saved ✅")


# ---------- live summary ----------
ratings = jloadl(RATINGS_FILE)
if ratings:
    rdf = pd.DataFrame(ratings)

    # --- normalize legacy categories: map "color" -> "perception" ---
    if "category" in rdf.columns:
        rdf["category"] = rdf["category"].replace({"color": "perception"})
        rdf["category"] = rdf["category"].astype(str).str.strip().str.lower()

    # normalize dtypes to be safe for math/round
    if "reliable" in rdf.columns:
        rel_num = rdf["reliable"].map({True: 1.0, False: 0.0}).astype("float64")
        rdf["_rel_num"] = rel_num
        rdf["rel_label"] = rdf["reliable"].map({True:"reliable", False:"unreliable", None:"fuzzy"})
    else:
        rdf["_rel_num"] = pd.Series(dtype="float64")
        rdf["rel_label"] = "fuzzy"

    # category summary
    summ = (rdf.groupby("category", as_index=False)
              .agg(num=("question","count"),
                   acc=("_rel_num","mean")))
    if len(summ):
        # robust rounding (avoid dtype issues)
        summ["acc"] = pd.to_numeric(summ["acc"], errors="coerce").astype(float)
        summ["acc(%)"] = (summ["acc"] * 100.0).astype(float).round(1)
        st.markdown("### 📊 Category accuracy (live)")
        st.dataframe(summ[["category","num","acc(%)"]].sort_values("category"), use_container_width=True)

        # save and offer downloads for notebooks/archives
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = OUT_DIR / "human_summary.csv"
        json_path = OUT_DIR / "human_summary.json"
        summ[["category","num","acc(%)"]].to_csv(csv_path, index=False)
        summ[["category","num","acc(%)"]].to_json(json_path, orient="records", indent=2, force_ascii=False)

        with open(csv_path, "rb") as f:
            st.download_button("⬇️ Download summary CSV", f, file_name="human_summary.csv", mime="text/csv")

    # reliability counts per category (matplotlib stacked bars)
    st.markdown("### 🔥 Reliability distribution (stacked counts)")
    counts = (rdf.groupby(["category","rel_label"], as_index=False)
                .size()
                .pivot(index="category", columns="rel_label", values="size")
                .fillna(0)
                .astype(int))
    if len(counts):
        for col in ["reliable","unreliable","fuzzy"]:
            if col not in counts.columns:
                counts[col] = 0
        counts = counts[["reliable","unreliable","fuzzy"]]

        fig, ax = plt.subplots(figsize=(8, 3 + 0.25*len(counts)))
        bottom = None
        for col in counts.columns:
            ax.bar(counts.index, counts[col], label=col, bottom=bottom)
            bottom = (counts[col] if bottom is None else (bottom + counts[col]))
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.set_title("Reliability by category")
        ax.legend()
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig)

        # Download the chart as PNG
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)
        st.download_button(
            label="⬇️ Download reliability plot (PNG)",
            data=buf,
            file_name="reliability_by_category.png",
            mime="image/png"
        )
        plt.close(fig)
else:
    st.info("No ratings saved yet. Add a few above.")
