import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

cm = np.array([    [
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

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Positive', 'Neutral', 'Negative'],
            yticklabels=['Positive', 'Neutral', 'Negative'])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix – ScoNet (Test Set)")
plt.tight_layout()
plt.show()
