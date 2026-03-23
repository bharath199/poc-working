import torch
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import cv2
import glob
import json

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "./SolDef_AI/Filtered/c1/good/WIN_20220330_13_12_12_Pro.jpg"
JSON_PATH  = "./SolDef_AI/Filtered/c1/good/WIN_20220330_13_12_12_Pro.json"
SEQ_GLOB   = "./output_seq_blend/*.png"

BASELINE_N = 5

# -----------------------------
# LOAD MODEL
# -----------------------------
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model.eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
])

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def normalize_minmax(x):
    """Normalize array to [0,1] using min-max scaling"""
    return (x - x.min()) / (x.max() - x.min() + 1e-8)
def zscore(x):
    return (x - np.mean(x[:BASELINE_N])) / (np.std(x[:BASELINE_N]) + 1e-6)
def safe_z(x):
    mean = np.mean(x[:BASELINE_N])
    std = np.std(x[:BASELINE_N])
    std = max(std, 1e-3)   # clamp
    return (x - mean) / std
def percent_change(x):
    baseline_mean = np.mean(x[:BASELINE_N])
    return (x - baseline_mean) / (baseline_mean + 1e-6)
def relative_score(x):
    baseline_mean = np.mean(x[:BASELINE_N])
    return x / (baseline_mean + 1e-6)
def polygon_to_mask(img_shape, points):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def crop_to_mask(img, mask):
    ys, xs = np.where(mask == 1)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]

def apply_mask(img, mask):
    out = img.copy()
    out[mask == 0] = 0
    return out

# -----------------------------
# LOAD BASE IMAGE + MASK
# -----------------------------
img0 = cv2.imread(IMAGE_PATH)

with open(JSON_PATH, "r") as f:
    data = json.load(f)

# use poor_solder region (left pad) as tracking region
poor_points = None
for shape in data["shapes"]:
    if shape["label"] == "poor_solder":
        poor_points = shape["points"]

full_mask = polygon_to_mask(img0.shape, poor_points)

# crop once → reuse bbox for all frames
base_crop, base_mask = crop_to_mask(img0, full_mask)

h, w = base_crop.shape[:2]

# -----------------------------
# FEATURE FUNCTIONS
# -----------------------------
def extract_resnet(img, mask):
    crop = img.copy()
    crop[mask == 0] = 0
    crop = cv2.resize(crop, (224, 224))

    x = transform(crop).unsqueeze(0)
    with torch.no_grad():
        feat = model(x)

    return feat.numpy().flatten()

def handcrafted(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    region = gray[mask == 1]

    area = np.sum(mask)
    mean = region.mean()
    std = region.std()
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()

    return np.array([area, mean, std, sharp])


# -----------------------------
# LOAD SEQUENCE
# -----------------------------
paths = sorted(glob.glob(SEQ_GLOB))

res_feats = []
hand_feats = []

for p in paths:
    img = cv2.imread(p)

    # crop same region
    crop = img[
        np.where(full_mask)[0].min():np.where(full_mask)[0].max(),
        np.where(full_mask)[1].min():np.where(full_mask)[1].max()
    ]

    mask = base_mask.copy()

    res_feats.append(extract_resnet(crop, mask))
    hand_feats.append(handcrafted(crop, mask))

res_feats = np.array(res_feats)
hand_feats = np.array(hand_feats)

# normalize handcrafted features
hand_feats = (hand_feats - hand_feats.mean(axis=0)) / (hand_feats.std(axis=0) + 1e-6)

# -----------------------------
# BASELINE
# -----------------------------
res_base = res_feats[:BASELINE_N].mean(axis=0)
hand_base = hand_feats[:BASELINE_N].mean(axis=0)

# -----------------------------
# DISTANCE
# -----------------------------
def cosine(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8)

res_scores = []
hand_scores = []

for i in range(len(paths)):
    res_scores.append(cosine(res_feats[i], res_base))
    hand_scores.append(np.linalg.norm(hand_feats[i] - hand_base))

res_scores = np.array(res_scores)
hand_scores = np.array(hand_scores)

# normalize both
res_scores = res_scores-np.mean(res_scores[:BASELINE_N])
hand_scores = hand_scores-np.mean(hand_scores[:BASELINE_N])

# -----------------------------
# COMBINED SCORE
# -----------------------------
combined = 0.6 * normalize_minmax(res_scores) + 0.4 * normalize_minmax(hand_scores)

# -----------------------------
# SMOOTHING
# -----------------------------
def ema(x, alpha=0.3):
    out = []
    s = x[0]
    for v in x:
        s = alpha*v + (1-alpha)*s
        out.append(s)
    return np.array(out)

combined_s = ema(combined)

# -----------------------------
# TREND
# -----------------------------
trend = np.gradient(combined_s)

# -----------------------------
# ALERT
# -----------------------------
THRESH = np.mean(combined_s[:BASELINE_N]) * 3

alerts = (combined_s > THRESH) & (trend > 0)

from plot_drift_vs_aoi import plot_drift_vs_aoi
plot_drift_vs_aoi(combined_s)

# -----------------------------
# PRINT
# -----------------------------
for i in range(len(paths)):
    print(f"{i:02d} | res={res_scores[i]:.3f} | hand={hand_scores[i]:.3f} | comb={combined_s[i]:.3f} | alert={int(alerts[i])}")