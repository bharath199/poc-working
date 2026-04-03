import numpy as np
import matplotlib.pyplot as plt

# --- Load data ---
scores_path = "data/demo/WIN_20220330_13_12_12_Pro/scores.npy"
scores = np.load(scores_path)

frames = np.arange(len(scores))

# --- PARAMETERS ---
BASELINE_N = 5
crop_ratio = 0.5

# --- Synthetic AOI (Option A) ---
aoi = np.zeros(len(scores))
aoi[-2:] = 1  # last frames fail

# --- Threshold (your logic, slightly safer) ---
baseline = scores[:BASELINE_N].mean()
threshold = baseline + 0.2

# Optional clamp (prevents useless threshold)
threshold = min(threshold, scores.max() * 0.9)

# --- Crop ---
start = int(len(scores) * (1 - crop_ratio))

frames = frames[start:]
scores = scores[start:]
aoi = aoi[start:]

# --- Find events ---
threshold_cross = np.argmax(scores > threshold) if np.any(scores > threshold) else None
aoi_trigger = np.argmax(aoi == 1) if np.any(aoi == 1) else None

# --- Plot ---
plt.figure(figsize=(7,4))

# Drift curve
plt.plot(frames, scores, linewidth=3, label="Drift Score")

# AOI curve (lighter so it doesn't dominate)
plt.plot(frames, aoi, linestyle="--", linewidth=2, alpha=0.7, label="AOI")

# Threshold line
plt.axhline(threshold, linestyle="--", linewidth=3)
plt.text(frames[0], threshold + 0.02, "Threshold")

# --- Intervention window ---
if threshold_cross is not None and aoi_trigger is not None:
    plt.axvspan(frames[threshold_cross], frames[aoi_trigger], alpha=0.12)
    y_text = 1.1 * 0.95
    center_frame = (frames[threshold_cross] + frames[aoi_trigger]) / 2
    plt.text(center_frame, y_text, "Intervention Window", ha='center', va='top', fontsize=12)

# --- Annotations ---
if threshold_cross is not None:
    plt.scatter(frames[threshold_cross], scores[threshold_cross])
    plt.annotate(
        "Early Warning",
        (frames[threshold_cross], scores[threshold_cross]),
        textcoords="offset points",
        xytext=(10,10),
        arrowprops=dict(arrowstyle="->")
    )

if aoi_trigger is not None:
    plt.scatter(frames[aoi_trigger], scores[aoi_trigger])
    plt.annotate(
        "AOI detects (late)",
        (frames[aoi_trigger], scores[aoi_trigger]),
        textcoords="offset points",
        xytext=(10,-20),
        arrowprops=dict(arrowstyle="->")
    )

# --- AXIS FIX (your version) ---
y_max = max(1.1, scores.max() * 1.2, threshold * 1.2)
plt.ylim(0, y_max)

# X-axis integers
plt.xticks(frames.astype(int))

# Labels
plt.xlabel("Frame")
plt.ylabel("Drift Score")
plt.title("Drift Detected Before AOI Failure")

plt.legend()
plt.tight_layout()

# Save
plt.savefig("graph.png", dpi=200)

plt.show()