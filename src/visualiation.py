import matplotlib.pyplot as plt
from feature_extraction import (
    Normal_ref,
    testAnomaly,
    check_defect
)

import numpy as np

# Build reference and calculate scores
normal_reference = Normal_ref()

normal_scores = testAnomaly(normal_reference)

threshold = np.percentile(normal_scores, 95)

defect_results = check_defect(
    normal_reference,
    threshold
)

# Combine all defective scores
defect_scores = [
    score
    for category_scores in defect_results.values()
    for score in category_scores
]

# Plot distributions
plt.hist(
    normal_scores,
    bins=10,
    alpha=0.6,
    label="Normal bottles"
)

plt.hist(
    defect_scores,
    bins=10,
    alpha=0.6,
    label="Defective bottles"
)

plt.axvline(
    threshold,
    linestyle="--",
    label=f"Threshold: {threshold:.2f}"
)

plt.xlabel("Anomaly Score")
plt.ylabel("Number of Images")
plt.title("Normal vs Defective Bottle Anomaly Scores")

plt.legend()
plt.tight_layout()
plt.show()
