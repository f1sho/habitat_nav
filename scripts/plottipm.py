import csv
import math
import numpy as np
import matplotlib.pyplot as plt

csv_path = (
    "results/ipm_accuracy_pitchfix_final/"
    "yolo26n_seg_pt/ipm_accuracy_samples.csv"
)

gt_distances = []
pred_distances = []

contact_gt = []
contact_pred = []

with open(csv_path, "r", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        try:
            gt = float(row["gt_distance_m"])
            pred = float(row["predicted_distance_m"])
        except (ValueError, TypeError, KeyError):
            continue

        if not math.isfinite(gt) or not math.isfinite(pred):
            continue

        gt_distances.append(gt)
        pred_distances.append(pred)

        contact_valid = row["contact_valid"].strip().lower() == "true"

        if contact_valid:
            contact_gt.append(gt)
            contact_pred.append(pred)

gt_distances = np.array(gt_distances)
pred_distances = np.array(pred_distances)

contact_gt = np.array(contact_gt)
contact_pred = np.array(contact_pred)

print(f"All valid samples: {len(gt_distances)}")
print(f"Contact-valid samples: {len(contact_gt)}")

fig, ax = plt.subplots(figsize=(7.5, 6.5))

# Plot all valid samples.
ax.scatter(
    gt_distances,
    pred_distances,
    alpha=0.30,
    s=22,
    label="All valid samples",
)

# Overlay contact-valid samples.
ax.scatter(
    contact_gt,
    contact_pred,
    marker="x",
    s=40,
    linewidths=1.3,
    label="Contact-valid samples",
)

# Draw the ideal y = x reference line.
max_distance = max(
    np.max(gt_distances),
    np.max(pred_distances),
)

ax.plot(
    [0, max_distance],
    [0, max_distance],
    linestyle="--",
    linewidth=1.5,
    label="Ideal prediction (y = x)",
)

ax.set_xlabel("Depth Ground Truth Distance (m)")
ax.set_ylabel("IPM Predicted Distance (m)")
ax.set_title("IPM Predicted Distance vs Depth Ground Truth")

ax.legend()
ax.grid(alpha=0.2)

fig.tight_layout()

output_path = (
    "results/ipm_accuracy_pitchfix_final/"
    "ipm_predicted_vs_gt.png"
)

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)

print(f"Figure saved to: {output_path}")

plt.show()