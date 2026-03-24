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
OUTPUT_ROOT = "./output"
JSON_LIST = "good_jsons.txt"
BASELINE_N = 5

# -----------------------------
# MODEL (cached)
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
# HELPERS
# -----------------------------
def get_sequences():
    return sorted(glob.glob(os.path.join(OUTPUT_ROOT, "WIN_*")))

def load_images(seq_dir):
    paths = sorted(glob.glob(os.path.join(seq_dir, "*.png")))
    imgs = [cv2.imread(p) for p in paths]
    return imgs

def get_json_for_sequence(seq_dir):
    name = os.path.basename(seq_dir)
    json_name = name + ".json"

    with open(JSON_LIST, "r") as f:
        paths = f.read().splitlines()

    for p in paths:
        if json_name in p:
            return p
    return None

# -----------------------------
# MASK LOADING
# -----------------------------
@st.cache_data
def load_mask_and_bbox(json_path, image_shape):
    with open(json_path, "r") as f:
        data = json.load(f)

    poor_points = None
    for shape in data["shapes"]:
        if shape["label"] == "poor_solder":
            poor_points = shape["points"]

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(poor_points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)

    ys, xs = np.where(mask == 1)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    return mask, (y1, y2, x1, x2)

def extract_region(img, mask, bbox):
    y1, y2, x1, x2 = bbox
    crop = img[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    return crop, crop_mask

def overlay_mask(img, mask, color=(0,255,0), alpha=0.4):
    overlay = img.copy()
    overlay[mask == 1] = color
    return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

# -----------------------------
# DRIFT MODEL
# -----------------------------
def combined_drift_score(images, mask, bbox):

    res_feats = []
    hand_feats = []

    for img in images:
        crop, crop_mask = extract_region(img, mask, bbox)

        # ---- ResNet
        masked = crop.copy()
        masked[crop_mask == 0] = 0
        resized = cv2.resize(masked, (224,224))

        x = transform(resized).unsqueeze(0)
        with torch.no_grad():
            feat = model(x).numpy().flatten()

        res_feats.append(feat)

        # ---- Handcrafted
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        region = gray[crop_mask == 1]

        area = np.sum(crop_mask)
        mean = region.mean()
        std = region.std()
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()

        hand_feats.append([area, mean, std, sharp])

    res_feats = np.array(res_feats)
    hand_feats = np.array(hand_feats)

    # normalize handcrafted
    hand_feats = (hand_feats - hand_feats.mean(axis=0)) / (hand_feats.std(axis=0) + 1e-6)

    # baseline
    res_base = res_feats[:BASELINE_N].mean(axis=0)
    hand_base = hand_feats[:BASELINE_N].mean(axis=0)

    def cosine(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8)

    res_scores = np.array([cosine(f, res_base) for f in res_feats])
    hand_scores = np.array([np.linalg.norm(f - hand_base) for f in hand_feats])

    # normalize (for visualization)
    res_scores = (res_scores - res_scores.min()) / (res_scores.max() - res_scores.min() + 1e-8)
    hand_scores = (hand_scores - hand_scores.min()) / (hand_scores.max() - hand_scores.min() + 1e-8)

    combined = 0.6 * res_scores + 0.4 * hand_scores

    # smoothing
    def ema(x, alpha=0.3):
        s = x[0]
        out = []
        for v in x:
            s = alpha*v + (1-alpha)*s
            out.append(s)
        return np.array(out)

    return ema(combined)

# -----------------------------
# PLOT
# -----------------------------
def plot_drift(scores):

    n = len(scores)

    aoi = np.zeros(n)
    aoi[int(n*0.8):] = 1

    baseline = scores[:BASELINE_N].mean()
    alert_line = baseline + 0.2

    trend = np.gradient(scores)
    alerts = (scores > alert_line) & (trend > 0)

    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(scores, label="Drift Score")
    ax.plot(aoi, '--', label="AOI (0=PASS,1=NG)")
    ax.axhline(alert_line, linestyle=':', label="Alert Threshold")

    for i in range(n):
        if alerts[i]:
            ax.scatter(i, scores[i])

    ax.set_xlabel("Frame")
    ax.set_ylabel("Score")
    ax.set_title("Process Drift vs AOI")

    ax.legend()
    ax.grid()

    return fig

# -----------------------------
# UI
# -----------------------------
st.title("SMT Process Drift Demo")

seq_dirs = get_sequences()

if not seq_dirs:
    st.error("No sequences found")
    st.stop()

selected_seq = st.selectbox("Select Sequence", seq_dirs)

images = load_images(selected_seq)

json_path = get_json_for_sequence(selected_seq)

if json_path is None:
    st.error("No matching JSON found")
    st.stop()

mask, bbox = load_mask_and_bbox(json_path, images[0].shape)

scores = combined_drift_score(images, mask, bbox)

# -----------------------------
# FRAME VIEW
# -----------------------------
frame_idx = st.slider("Frame", 0, len(images)-1, 0)

img = images[frame_idx]
overlay = overlay_mask(img, mask)

st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
         caption=f"Frame {frame_idx}")

# -----------------------------
# PLOT
# -----------------------------
st.pyplot(plot_drift(scores))

# -----------------------------
# STATUS
# -----------------------------
current = scores[frame_idx]
baseline = scores[:BASELINE_N].mean()

st.metric("Drift Score", f"{current:.3f}")

if current > baseline + 0.2:
    st.warning("⚠ Drift Detected")
else:
    st.success("✅ Normal")