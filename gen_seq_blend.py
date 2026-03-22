import cv2
import json
import numpy as np
import os

# -----------------------------
# CONFIG
# -----------------------------
GOOD_JSONS_FILE = "./good_jsons.txt"
OUTPUT_DIR = "./output"
N_FRAMES   = 30
MAX_ALPHA  = 0.7   # don't go full defect

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


def normalize(p):
    p = p.astype(np.float32)
    return (p - p.mean()) / (p.std() + 1e-6)


def compute_stats(patch, mask):
    region = patch[mask == 1]
    if len(region) == 0:
        return {}

    mean_intensity = np.mean(region)
    area = np.sum(mask)

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    return {
        "mean": float(mean_intensity),
        "area": int(area),
        "sharpness": float(sharpness)
    }


def apply_drift_blend(good_patch, poor_patch, good_mask, poor_mask, alpha):
    good = good_patch.astype(np.float32)
    poor = poor_patch.astype(np.float32)

    # normalize both
    good_n = normalize(good)
    poor_n = normalize(poor)

    # intersection region only
    common = (good_mask == 1) & (poor_mask == 1)

    out = good_n.copy()

    # blend toward poor
    out[common] = (1 - alpha) * good_n[common] + alpha * poor_n[common]

    # rescale to image range
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)

    return out.astype(np.uint8)


# -----------------------------
# LOAD DATA
# -----------------------------
def generate_sequence_blend(
    image_path,
    json_path,
    output_dir="./output_seq_blend",
    n_frames=30,
    max_alpha=0.7,
):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    good_points = None
    poor_points = None

    for shape in data["shapes"]:
        if shape["label"] == "good":
            good_points = shape["points"]
        elif shape["label"] == "poor_solder":
            poor_points = shape["points"]

    if good_points is None or poor_points is None:
        raise ValueError("JSON must contain both 'good' and 'poor_solder' shapes")

    good_mask = polygon_to_mask(img.shape, good_points)
    poor_mask = polygon_to_mask(img.shape, poor_points)

    # -----------------------------
    # EXTRACT PATCHES
    good_patch, good_patch_mask, good_bbox = extract_patch(img, good_mask)
    poor_patch, poor_patch_mask, poor_bbox = extract_patch(img, poor_mask)

    # Flip good (right  left)
    good_patch = cv2.flip(good_patch, 1)
    good_patch_mask = cv2.flip(good_patch_mask, 1)

    # Resize to match poor patch
    h, w = poor_patch.shape[:2]
    good_patch = cv2.resize(good_patch, (w, h))
    good_patch_mask = cv2.resize(
        good_patch_mask, (w, h), interpolation=cv2.INTER_NEAREST
    )

    # -----------------------------
    # DEBUG STATS
    print("GOOD:", compute_stats(good_patch, good_patch_mask))
    print("POOR:", compute_stats(poor_patch, poor_patch_mask))

    # -----------------------------
    # GENERATE SEQUENCE
    for t in range(n_frames):
        alpha = (t / (n_frames - 1)) ** 1.5
        alpha = min(alpha, max_alpha)

        drift_patch = apply_drift_blend(
            good_patch,
            poor_patch,
            good_patch_mask,
            poor_patch_mask,
            alpha,
        )

        frame = img.copy()
        y_min, y_max, x_min, x_max = poor_bbox
        region = frame[y_min:y_max, x_min:x_max]

        mask = poor_patch_mask
        region[mask == 1] = drift_patch[mask == 1]

        frame[y_min:y_max, x_min:x_max] = region
        cv2.imwrite(f"{output_dir}/frame_{t:03d}.png", frame)

    print(f"Done. Check {output_dir}/")


if __name__ == "__main__":
    if not os.path.exists(GOOD_JSONS_FILE):
        raise FileNotFoundError(f"File not found: {GOOD_JSONS_FILE}")
    
    with open(GOOD_JSONS_FILE, "r") as f:
        json_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(json_paths)} JSON files to process\n")
    
    for idx, json_path in enumerate(json_paths, 1):
        # Derive image path from JSON path
        image_path = json_path.replace(".json", ".jpg")
        
        # Create output subdirectory for this sequence
        filename = os.path.basename(json_path).replace(".json", "")
        seq_output_dir = os.path.join(OUTPUT_DIR, filename)
        
        print(f"[{idx}/{len(json_paths)}] Processing: {filename}")
        
        try:
            generate_sequence_blend(
                image_path,
                json_path,
                seq_output_dir,
                N_FRAMES,
                MAX_ALPHA,
            )
            print(f"  [OK] Completed: {seq_output_dir}\n")
        except Exception as e:
            print(f"  [ERROR] {e}\n")
