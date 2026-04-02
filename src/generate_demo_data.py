import os
import cv2
import numpy as np
import shutil
import json

import torch
import torchvision.models as models
import torchvision.transforms as T

# -----------------------------
# CONFIG
# -----------------------------
SRC_ROOT = "./data_local/output"
DST_ROOT = "./data/demo"

NUM_SEQS = 3
SEQ_STRIDE = 3  # Select every nth sequence
NUM_FRAMES = 10
IMG_SIZE = (384, 384)
JPEG_QUALITY = 70

BASELINE_N = 5

# -----------------------------
# MODEL
# -----------------------------
print("Loading model...")
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model.eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
])

# -----------------------------
# HELPERS
# -----------------------------
import os

JSON_DIR = os.path.join(os.getcwd(), "data", "good_jsons")

def get_json_for_sequence(seq_name):
    if not os.path.exists(JSON_DIR):
        return None

    target = seq_name + ".json"

    for fname in os.listdir(JSON_DIR):
        if fname == target:  # exact match (safer than "in")
            return os.path.join(JSON_DIR, fname)

    return None


def load_mask_and_bbox(json_path, original_shape, new_shape):
    with open(json_path, "r") as f:
        data = json.load(f)

    poor = None
    for s in data["shapes"]:
        if s["label"] == "poor_solder":
            poor = s["points"]

    if poor is None:
        raise ValueError("No poor_solder found")

    h0, w0 = original_shape[:2]
    h1, w1 = new_shape[:2]

    scale_x = w1 / w0
    scale_y = h1 / h0

    scaled_pts = []
    for x, y in poor:
        scaled_pts.append([int(x * scale_x), int(y * scale_y)])

    mask = np.zeros((h1, w1), dtype=np.uint8)
    pts = np.array(scaled_pts, dtype=np.int32)

    cv2.fillPoly(mask, [pts], 1)

    ys, xs = np.where(mask == 1)

    if len(ys) == 0:
        raise ValueError("Empty mask after scaling")

    return mask, (ys.min(), ys.max(), xs.min(), xs.max())


def compute_scores(images, mask, bbox):
    res_feats = []
    hand_feats = []

    for img in images:
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

    hand_feats = (hand_feats - hand_feats.mean(0)) / (hand_feats.std(0) + 1e-6)

    res_base = res_feats[:BASELINE_N].mean(0)
    hand_base = hand_feats[:BASELINE_N].mean(0)

    def cosine(a, b):
        return 1 - np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8)

    res_scores = np.array([cosine(f, res_base) for f in res_feats])
    hand_scores = np.array([np.linalg.norm(f - hand_base) for f in hand_feats])

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
# MAIN
# -----------------------------
#delte old demo data
if os.path.exists(DST_ROOT):
    shutil.rmtree(DST_ROOT)
os.makedirs(DST_ROOT, exist_ok=True)

seqs = sorted([d for d in os.listdir(SRC_ROOT) if d.startswith("WIN_")])[::SEQ_STRIDE][:NUM_SEQS]

print(f"Processing {len(seqs)} sequences...")

for seq in seqs:
    print(f"\n--- {seq} ---")

    try:
        src_seq = os.path.join(SRC_ROOT, seq)
        dst_seq = os.path.join(DST_ROOT, seq)

        os.makedirs(dst_seq, exist_ok=True)

        frames = sorted([f for f in os.listdir(src_seq) if f.endswith(".png")])

        idxs = np.linspace(0, len(frames)-1, NUM_FRAMES).astype(int)

        images = []

        for j, i in enumerate(idxs):
            fname = frames[i]

            src_path = os.path.join(src_seq, fname)
            dst_path = os.path.join(dst_seq, f"{j:03d}.jpg")

            original = cv2.imread(src_path)
            if original is None:
                continue

            resized = cv2.resize(original, IMG_SIZE)

            cv2.imwrite(
                dst_path,
                resized,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            )

            images.append(resized)

        if len(images) == 0:
            raise ValueError("No valid images")

        # -----------------------------
        # MASK + SCORES
        # -----------------------------
        json_path = get_json_for_sequence(seq)

        if json_path is None:
            raise ValueError("No JSON mapping")

        mask, bbox = load_mask_and_bbox(
            json_path,
            original.shape,
            images[0].shape
        )
        np.save(os.path.join(dst_seq, "meta.npy"), original.shape)

        print("Computing scores...")
        scores = compute_scores(images, mask, bbox)

        np.save(os.path.join(dst_seq, "scores.npy"), scores)
        print("✓ Saved scores.npy")

    except Exception as e:
        print(f"⚠ Skipping {seq}: {e}")
        continue

print("\n✅ Demo dataset ready:", DST_ROOT)