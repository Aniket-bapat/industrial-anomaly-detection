import torch
from anomalib.models import Patchcore

model = Patchcore()

print("PatchCore created successfully!")
print("Model type:", type(model).__name__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\nModel architecture:")
print(model)