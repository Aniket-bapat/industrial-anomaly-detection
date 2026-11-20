
import os
import torch
import numpy as np

from feature_extraction import Normal_ref, testAnomaly

# Create model directory
os.makedirs("model", exist_ok=True)

# 1. Calculate normal reference
normal_reference = Normal_ref()

# 2. Calculate normal test scores
normal_scores = testAnomaly(normal_reference)

# 3. Calculate threshold
threshold = np.percentile(normal_scores, 95)

# 4. Save reference and threshold
torch.save(
    {
        "normal_reference": normal_reference,
        "threshold": float(threshold)
    },
    "model/normal_reference.pt"
)

print("Model reference saved successfully!")
print("Threshold:", threshold)
