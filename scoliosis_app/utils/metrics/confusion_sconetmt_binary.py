import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

cm = np.array([            [
      64,
      11
    ],
    [
      9,
      111
    ]

])

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Positive', 'Negative'],
            yticklabels=['Positive', 'Negative'])
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix – ScoNet-MT-Binary (Test Set)")
plt.tight_layout()
plt.show()
