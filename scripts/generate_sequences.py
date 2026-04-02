import cv2
import json
import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "./data_local/SolDef_AI/Filtered/c1/good/WIN_20220330_13_12_12_Pro.jpg"
JSON_PATH  = "./data_local/SolDef_AI/Filtered/c1/good/WIN_20220330_13_12_12_Pro.json"
OUTPUT_DIR = "./data_local/output/output_seq"
N_FRAMES   = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# HELPERS
# -----------------------------
def polygon_to_mask(img_shape, points):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def extract_patch(img, mask):
    coords = np.where(mask == 1)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    patch = img[y_min:y_max, x_min:x_max]
    patch_mask = mask[y_min:y_max, x_min:x_max]

    return patch, patch_mask, (y_min, y_max, x_min, x_max)


def compute_stats(patch, mask):
    region = patch[mask == 1]
    if len(region) == 0:
        return {}

    mean_intensity = np.mean(region)
    area = np.sum(mask)

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    return {
        "mean": mean_intensity,
        "area": area,
        "sharpness": sharpness
    }

def apply_drift(patch, mask, alpha):
    patch = patch.copy().astype(np.float32)
    mask = mask.astype(np.uint8)

    # ---- Blur (simulate spreading / loss of definition)
    k = int(1 + alpha * 5)
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(patch, (k, k), 0)

    # ---- Intensity decay (AFTER blur)
    decay = 1 - 0.4 * alpha
    region = blurred[mask == 1] * decay
    noise = 1 + 0.05 * np.random.randn(*region.shape)
    region = region * noise

    # put back
    patch[mask == 1] = region

    # ---- Area shrink (soft, not hard cut)
    kernel_size = int(1 + alpha * 2)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    new_mask = cv2.erode(mask, kernel, iterations=1)

    # soft blending instead of zeroing
    fade = alpha * 0.7
    patch[(mask == 1) & (new_mask == 0)] *= (1 - fade)

    return patch.astype(np.uint8), new_mask


# -----------------------------
# LOAD DATA
# -----------------------------
img = cv2.imread(IMAGE_PATH)

with open(JSON_PATH, "r") as f:
    data = json.load(f)

good_points = None
poor_points = None

for shape in data["shapes"]:
    if shape["label"] == "good":
        good_points = shape["points"]
    elif shape["label"] == "poor_solder":
        poor_points = shape["points"]

good_mask = polygon_to_mask(img.shape, good_points)
poor_mask = polygon_to_mask(img.shape, poor_points)

# -----------------------------
# EXTRACT PATCHES
# -----------------------------
good_patch, good_patch_mask, good_bbox = extract_patch(img, good_mask)
poor_patch, poor_patch_mask, poor_bbox = extract_patch(img, poor_mask)

# Flip good to align with poor
good_patch = cv2.flip(good_patch, 1)
good_patch_mask = cv2.flip(good_patch_mask, 1)

# Resize good → match poor size
h, w = poor_patch.shape[:2]
good_patch = cv2.resize(good_patch, (w, h))
good_patch_mask = cv2.resize(good_patch_mask, (w, h), interpolation=cv2.INTER_NEAREST)

# -----------------------------
# DEBUG: print stats
# -----------------------------
print("GOOD:", compute_stats(good_patch, good_patch_mask))
print("POOR:", compute_stats(poor_patch, poor_patch_mask))


# -----------------------------
# GENERATE SEQUENCE
# -----------------------------
for t in range(N_FRAMES):
    alpha = (t / (N_FRAMES - 1)) ** 1.5

    drift_patch, drift_mask = apply_drift(good_patch, good_patch_mask, alpha)

    # paste back into original image (use poor bbox location)
    frame = img.copy()

    y_min, y_max, x_min, x_max = poor_bbox

    region = frame[y_min:y_max, x_min:x_max]

    # blend region
    region[drift_mask == 1] = drift_patch[drift_mask == 1]

    frame[y_min:y_max, x_min:x_max] = region

    cv2.imwrite(f"{OUTPUT_DIR}/frame_{t:03d}.png", frame)

print("Done. Check ./data_local/output/output_seq/")