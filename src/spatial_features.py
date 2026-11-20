
import torch
from torchvision import models, transforms
from PIL import Image

# Load pretrained ResNet18
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])

# Load normal bottle
image_path = "../dataset/bottle/train/good/000.png"

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# Extract intermediate feature maps
with torch.no_grad():

    x = model.conv1(input_tensor)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    layer1 = model.layer1(x)
    layer2 = model.layer2(layer1)
    layer3 = model.layer3(layer2)
    layer4 = model.layer4(layer3)

print("Layer 1:", layer1.shape)
print("Layer 2:", layer2.shape)
print("Layer 3:", layer3.shape)
print("Layer 4:", layer4.shape)
