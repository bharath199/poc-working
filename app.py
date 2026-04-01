import streamlit as st
import glob
import os
import numpy as np
import cv2
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
import torchvision.transforms as T

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_ROOT = os.path.join(BASE_DIR, "output")
DATA_ROOT = os.path.join(BASE_DIR, "data", "demo")
JSON_LIST = os.path.join(BASE_DIR, "good_jsons.txt")
BASELINE_N = 5

st.set_page_config(layout="wide")

# -----------------------------
# MODEL (load once)
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
    return model

model = load_model()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
])

# -----------------------------
# DATA
# -----------------------------
@st.cache_data
def get_sequences():
    seqs = sorted(glob.glob(os.path.join(DATA_ROOT, "WIN_*")))
    return seqs

@st.cache_data
def load_image_paths(seq_dir):
    exts = ["*.jpg", "*.png", "*.jpeg"]
    paths = []

    for ext in exts:
        paths.extend(glob.glob(os.path.join(seq_dir, ext)))

    return sorted(paths)

def read_image(path):
    return cv2.imread(path)

@st.cache_data
def get_json_for_sequence(seq_dir):
    name = os.path.basename(seq_dir)
    json_name = name + ".json"

    if not os.path.exists(JSON_LIST):
        return None

    with open(JSON_LIST, "r") as f:
        paths = f.read().splitlines()

    for p in paths:
        if json_name in p:
            return p
    return None

# -----------------------------
# MASK
# -----------------------------
@st.cache_data
def load_mask_and_bbox(json_path, original_shape, new_shape):
    import json
    import numpy as np
    import cv2

    with open(json_path, "r") as f:
        data = json.load(f)

    poor = None
    for s in data["shapes"]:
        if s["label"] == "poor_solder":
            poor = s["points"]

    if poor is None:
        raise ValueError("No poor_solder label found")

    h0, w0 = original_shape[:2]
    h1, w1 = new_shape[:2]

    scale_x = w1 / w0
    scale_y = h1 / h0

    scaled_pts = [[int(x * scale_x), int(y * scale_y)] for x, y in poor]

    mask = np.zeros((h1, w1), dtype=np.uint8)
    pts = np.array(scaled_pts, dtype=np.int32)

    cv2.fillPoly(mask, [pts], 1)

    ys, xs = np.where(mask == 1)

    if len(ys) == 0:
        raise ValueError("Empty mask after scaling")

    return mask, (ys.min(), ys.max(), xs.min(), xs.max())

def overlay_mask(img, mask):
    overlay = img.copy()
    overlay[mask == 1] = (0, 255, 0)
    return cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

# -----------------------------
# DRIFT
# -----------------------------
@st.cache_data
def compute_scores(image_paths, mask, bbox):

    res_feats = []
    hand_feats = []

    for p in image_paths:
        img = cv2.imread(p)

        y1, y2, x1, x2 = bbox
        crop = img[y1:y2, x1:x2]
        crop_mask = mask[y1:y2, x1:x2]

        # ResNet
        masked = crop.copy()
        masked[crop_mask == 0] = 0
        resized = cv2.resize(masked, (224, 224))

        x = transform(resized).unsqueeze(0)
        with torch.no_grad():
            feat = model(x).numpy().flatten()

        res_feats.append(feat)

        # Handcrafted
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        region = gray[crop_mask == 1]

        hand_feats.append([
            np.sum(crop_mask),
            region.mean(),
            region.std(),
            cv2.Laplacian(gray, cv2.CV_64F).var()
        ])

    res_feats = np.array(res_feats)
    hand_feats = np.array(hand_feats)

    # normalize handcrafted
    hand_feats = (hand_feats - hand_feats.mean(0)) / (hand_feats.std(0) + 1e-6)

    res_base = res_feats[:BASELINE_N].mean(0)
    hand_base = hand_feats[:BASELINE_N].mean(0)

    def cosine(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8)

    res_scores = np.array([cosine(f, res_base) for f in res_feats])
    hand_scores = np.array([np.linalg.norm(f - hand_base) for f in hand_feats])

    # normalize
    res_scores = (res_scores - res_scores.min()) / (res_scores.max() - res_scores.min() + 1e-8)
    hand_scores = (hand_scores - hand_scores.min()) / (hand_scores.max() - hand_scores.min() + 1e-8)

    combined = 0.6 * res_scores + 0.4 * hand_scores

    # EMA smoothing
    smoothed = []
    s = combined[0]
    for v in combined:
        s = 0.3*v + 0.7*s
        smoothed.append(s)

    return np.array(smoothed)

# -----------------------------
# LOAD PRECOMPUTED (if exists)
# -----------------------------
def load_scores_if_available(seq_dir):
    path = os.path.join(seq_dir, "scores.npy")
    if os.path.exists(path):
        return np.load(path)
    return None

# -----------------------------
# PLOT
# -----------------------------
def plot(scores):

    n = len(scores)
    aoi = np.zeros(n)
    aoi[int(n*0.8):] = 1

    baseline = scores[:BASELINE_N].mean()
    alert = baseline + 0.2

    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(scores, label="Drift Score")
    ax.plot(aoi, '--', label="AOI")
    ax.axhline(alert, linestyle=':', label="Alert")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Score")
    ax.set_title("Drift vs AOI")

    ax.legend()
    ax.grid()

    return fig

# -----------------------------
# UI
# -----------------------------
st.title("SMT Process Drift Demo")

seqs = get_sequences()

if not seqs:
    st.error("No sequences found in /output")
    st.stop()

seq = st.selectbox("Select Sequence", seqs)
image_paths = load_image_paths(seq)

if not image_paths:
    st.error("No images in sequence")
    st.stop()

# mask
sample = read_image(image_paths[0])
json_path = get_json_for_sequence(seq)

if not json_path:
    st.error("JSON mapping missing")
    st.stop()

meta_path = os.path.join(seq, "meta.npy")
original_shape = np.load(meta_path)

mask, bbox = load_mask_and_bbox(
    json_path,
    original_shape,
    sample.shape
)

# -----------------------------
# SCORE LOADING (fast path)
# -----------------------------
scores = load_scores_if_available(seq)

if scores is None:
    st.info("Computing scores (first time only)...")
    scores = compute_scores(image_paths, mask, bbox)

# -----------------------------
# VIEW
# -----------------------------
col1, col2 = st.columns([1,1])

frame = st.slider("Frame", 0, len(image_paths)-1, 0)

img = read_image(image_paths[frame])
overlay = overlay_mask(img, mask)

with col1:
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
             caption=f"Frame {frame}")

with col2:
    st.pyplot(plot(scores))

# -----------------------------
# STATUS
# -----------------------------
current = scores[frame]
baseline = scores[:BASELINE_N].mean()

st.metric("Drift Score", f"{current:.3f}")

if current > baseline + 0.2:
    st.warning("⚠ Drift Detected")
else:
    st.success("✅ Normal")