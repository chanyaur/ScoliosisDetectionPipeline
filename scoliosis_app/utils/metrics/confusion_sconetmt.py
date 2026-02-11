import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

cm = np.array([        [
      70,
      0,
      5
    ],
    [
      10,
      12,
      8
    ],
    [
      20,
      2,
      98
    ]
])

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Positive', 'Neutral', 'Negative'],
            yticklabels=['Positive', 'Neutral', 'Negative'])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix – ScoNetMT (Test Set)")
plt.tight_layout()
plt.show()
