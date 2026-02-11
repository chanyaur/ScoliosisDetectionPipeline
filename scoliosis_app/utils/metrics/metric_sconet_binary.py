import numpy as np

# === Paste your confusion matrix here ===
# Format: [[TN, FP],
#          [FN, TP]]
cm = np.array([
        [
      70,
      5
    ],
    [
      13,
      107
    ]

])

# --- Compute metrics ---
TN, FP, FN, TP = cm.ravel()

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0   # sensitivity
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (TP + TN) / cm.sum()

# --- Print summary ---
print("=== Metrics from Confusion Matrix ===")
print(f"Accuracy:            {accuracy:.4f}")
print(f"Precision:           {precision:.4f}")
print(f"Recall/Sensitivity:  {recall:.4f}")
print(f"F1-score:            {f1:.4f}")
