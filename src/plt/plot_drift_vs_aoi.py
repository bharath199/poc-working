import matplotlib.pyplot as plt
import numpy as np

def plot_drift_vs_aoi(combined_s, baseline_n=5, aoi_ratio=0.8):
    """
    combined_s : smoothed drift score (array)
    baseline_n : number of baseline frames
    aoi_ratio  : when AOI flips (e.g. 0.8 → 80% of sequence)
    """

    n = len(combined_s)

    # -----------------------------
    # AOI SIMULATION
    # -----------------------------
    aoi = np.zeros(n)
    aoi_start = int(n * aoi_ratio)
    aoi[aoi_start:] = 1  # 0=PASS, 1=NG

    # -----------------------------
    # ALERT LOGIC
    # -----------------------------
    baseline_mean = np.mean(combined_s[:baseline_n])
    alert_line = baseline_mean * 1.5

    trend = np.gradient(combined_s)
    alerts = (combined_s > alert_line) & (trend > 0)

    # -----------------------------
    # NORMALIZE FOR VISUAL ONLY
    # -----------------------------
    norm = (combined_s - combined_s.min()) / (combined_s.max() - combined_s.min() + 1e-8)

    # -----------------------------
    # PLOT
    # -----------------------------
    plt.figure(figsize=(10, 5))

    plt.plot(norm, label="Drift Score (normalized)", linewidth=2)
    plt.plot(aoi, '--', label="AOI (0=PASS, 1=NG)", linewidth=2)

    plt.axhline(
        (alert_line - combined_s.min()) / (combined_s.max() - combined_s.min() + 1e-8),
        linestyle=':',
        label="Drift Alert Threshold"
    )

    # mark alerts
    for i in range(n):
        if alerts[i]:
            plt.scatter(i, norm[i])

    plt.xlabel("Frame / Time")
    plt.ylabel("Score")
    plt.title("Process Drift vs AOI Detection")

    plt.legend()
    plt.grid()

    plt.show()