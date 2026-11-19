import inspect
import torch
from torchvision import models, transforms
from torchvision.models.resnet import BasicBlock
from PIL import Image
import os
import numpy as np

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


# print("Normal shape:", normal_features.shape)
# print("Defect shape:", defect_features.shape)



distance = torch.norm(normal_features - defect_features, p=2)
distance2 = torch.norm(normal_features - normal_features_2, p=2)

# print("Euclidean distance:", distance.item())
# print("Euclidean distance:", distance2.item())



def Normal_ref():
    normal_folder = "../dataset/bottle/train/good"

    image_files = sorted([
        f for f in os.listdir(normal_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    normal_features_list = []
    for filename in image_files:
        image_path = os.path.join(normal_folder, filename)

        features = extract_features(image_path)

    normal_features_list.append(features)


    normal_features_tensor = torch.cat(normal_features_list,dim=0)

    normal_reference = normal_features_tensor.mean(dim=0)
    # print("Normal reference shape:", normal_reference.shape)
    # print("Number of normal images:", len(image_files))
    # print("All normal features:", normal_features_tensor.shape)
    return normal_reference


normal_reference = Normal_ref()

defect_path = "../dataset/bottle/test/broken_large/000.png"

defect_features = extract_features(defect_path)

anomaly_score = torch.norm(defect_features.squeeze(0) - normal_reference,p=2)


# print("Anomaly score:", anomaly_score.item())



def testAnomaly(normal_refrence):
    normal_test_folder = "../dataset/bottle/test/good"
    normal_test_scores = []
    for filename in sorted(os.listdir(normal_test_folder)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(normal_test_folder, filename)

            features = extract_features(image_path)

            score = torch.norm(
                features.squeeze(0) - normal_reference,
                p=2
            )

            normal_test_scores.append(score.item())
    return normal_test_scores

# scores = testAnomaly(normal_refrence=normal_reference)

# print(" Sccores : ", type(scores))
# print("top 10 scores :")

# for i in range(10):
#     print(scores[i])





#scores_np = np.array(scores)

# print("Total images:", len(scores_np))
# print("Minimum:", scores_np.min())
# print("Maximum:", scores_np.max())
# print("Mean:", scores_np.mean())
# print("Median:", np.median(scores_np))
# print("Standard deviation:", scores_np.std())
# print("95th percentile:", np.percentile(scores_np, 95))


#threshold = np.percentile(scores_np,95)


#def check_defect(threshold):
    defect_categories = [
    "broken_large",
    "broken_small",
    "contamination"
    ]

    provisional_threshold = threshold


    for category in defect_categories:

        defect_folder = os.path.join(
        "../dataset/bottle/test",
        category
    )
        defect_scores = []

        for filename in sorted(os.listdir(defect_folder)):

            if filename.lower().endswith((".png", ".jpg", ".jpeg")):

                image_path = os.path.join(defect_folder, filename)

                features = extract_features(image_path)

                score = torch.norm(
                features.squeeze(0) - normal_reference,
                p=2
                )

                defect_scores.append(score.item())



    detected = sum(score > provisional_threshold for score in defect_scores)

    print("Flagged as anomalous:", detected, "/", len(defect_scores))

    return defect_scores


# defect_scores = check_defect(threshold=threshold)

# for i in range(10):
#     print(defect_scores[i])



def Normal_ref():
    normal_folder = "../dataset/bottle/train/good"

    image_files = sorted([
        f for f in os.listdir(normal_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    normal_features_list = []

    for filename in image_files:
        image_path = os.path.join(normal_folder, filename)

        features = extract_features(image_path)
        normal_features_list.append(features)

    normal_features_tensor = torch.cat(
        normal_features_list, dim=0
    )

    normal_reference = normal_features_tensor.mean(dim=0)

    print("Number of training images:", len(image_files))
    print("All normal features:", normal_features_tensor.shape)
    print("Normal reference:", normal_reference.shape)

    return normal_reference


def testAnomaly(normal_reference):
    normal_test_folder = "../dataset/bottle/test/good"

    normal_test_scores = []

    for filename in sorted(os.listdir(normal_test_folder)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):

            image_path = os.path.join(normal_test_folder, filename)
            features = extract_features(image_path)

            score = torch.norm(
                features.squeeze(0) - normal_reference,
                p=2
            )

            normal_test_scores.append(score.item())

    return normal_test_scores


def check_defect(normal_reference, threshold):

    defect_categories = [
        "broken_large",
        "broken_small",
        "contamination"
    ]

    results = {}

    for category in defect_categories:

        defect_folder = os.path.join(
            "../dataset/bottle/test",
            category
        )

        defect_scores = []

        for filename in sorted(os.listdir(defect_folder)):

            if filename.lower().endswith((".png", ".jpg", ".jpeg")):

                image_path = os.path.join(defect_folder, filename)
                features = extract_features(image_path)

                score = torch.norm(
                    features.squeeze(0) - normal_reference,
                    p=2
                )

                defect_scores.append(score.item())

        detected = sum(
            score > threshold for score in defect_scores
        )

        results[category] = defect_scores

        print("\nCategory:", category)
        print("Total images:", len(defect_scores))
        print("Mean score:", sum(defect_scores) / len(defect_scores))
        print("Flagged as anomalous:", detected, "/", len(defect_scores))

    return results

normal_reference = Normal_ref()

normal_scores = testAnomaly(normal_reference)

threshold = np.percentile(normal_scores, 95)

print("\nThreshold:", threshold)

defect_results = check_defect(
    normal_reference,
    threshold
)



