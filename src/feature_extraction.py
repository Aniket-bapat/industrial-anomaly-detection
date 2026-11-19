import inspect
import torch
from torchvision import models, transforms
from torchvision.models.resnet import BasicBlock
from PIL import Image



def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = feature_extractor(image_tensor)

    return features.flatten(1)

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

model.eval()

#print(model)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])


image_path = "../dataset/bottle/train/good/000.png"


image = Image.open(image_path).convert("RGB")

input_tensor = transform(image)

input_tensor = input_tensor.unsqueeze(0)
#print("Input shape:", input_tensor.shape)

feature_extractor = torch.nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor.eval()


with torch.no_grad():
    features = feature_extractor(input_tensor)

#print("Feature shape:", features.shape)

features = features.flatten(1)


# print("Flattened feature shape:", features.shape)



# print("First 10 features:")
# print(features[0, :10])

# print("Minimum:", features.min().item())
# print("Maximum:", features.max().item())
# print("Mean:", features.mean().item())


normal_path_2 = "../dataset/bottle/train/good/001.png"
normal_path = "../dataset/bottle/train/good/000.png"
defect_path = "../dataset/bottle/test/broken_large/000.png"

normal_features = extract_features(normal_path)
defect_features = extract_features(defect_path)
normal_features_2 = extract_features(normal_path_2)


print("Normal shape:", normal_features.shape)
print("Defect shape:", defect_features.shape)



distance = torch.norm(normal_features - defect_features, p=2)
distance2 = torch.norm(normal_features - normal_features_2, p=2)

print("Euclidean distance:", distance.item())
print("Euclidean distance:", distance2.item())