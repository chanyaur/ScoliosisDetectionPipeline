import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# === Paste your confusion matrix here ===
# Rows = true classes, Columns = predicted classes
cm = np.array([
    [
      66,
      1,
      8
    ],
    [
      8,
      7,
      15
    ],
    [
      12,
      0,
      108
    ]
])

# --- Convert confusion matrix to y_true, y_pred for sklearn ---
y_true, y_pred = [], []
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        y_true += [i] * cm[i, j]
        y_pred += [j] * cm[i, j]

# --- Compute metrics ---
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
macro_f1 = np.mean(f1)
weighted_f1 = np.average(f1, weights=support)
acc = accuracy_score(y_true, y_pred)

# --- Print summary ---
print("=== Per-class metrics ===")
for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
    print(f"Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}, Support={s}")

print("\n=== Overall metrics ===")
print(f"Accuracy:   {acc:.4f}")
print(f"Macro F1:   {macro_f1:.4f}")
print(f"Weighted F1:{weighted_f1:.4f}")
