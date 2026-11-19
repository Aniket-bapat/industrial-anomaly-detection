
import os
import torch
import numpy as np

from PIL import Image
from torchvision import models, transforms


# ==========================================
# 1. LOAD PRETRAINED RESNET18
# ==========================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)
model.eval()

# Remove the final classification layer
feature_extractor = torch.nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor.eval()


# ==========================================
# 2. IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std
    )
])


# ==========================================
# 3. FEATURE EXTRACTION FUNCTION
# ==========================================

def extract_features(image_path):

    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = feature_extractor(image_tensor)

    return features.flatten(1)


# ==========================================
# 4. BUILD NORMAL REFERENCE VECTOR
# ==========================================

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
        normal_features_list,
        dim=0
    )

    normal_reference = normal_features_tensor.mean(dim=0)

    print("Number of training images:", len(image_files))
    print("All normal features:", normal_features_tensor.shape)
    print("Normal reference:", normal_reference.shape)

    return normal_reference


# ==========================================
# 5. TEST NORMAL IMAGES
# ==========================================

def testAnomaly(normal_reference):

    normal_test_folder = "../dataset/bottle/test/good"

    normal_test_scores = []

    for filename in sorted(os.listdir(normal_test_folder)):

        if filename.lower().endswith((".png", ".jpg", ".jpeg")):

            image_path = os.path.join(
                normal_test_folder,
                filename
            )

            features = extract_features(image_path)

            score = torch.norm(
                features.squeeze(0) - normal_reference,
                p=2
            )

            normal_test_scores.append(score.item())

    return normal_test_scores


# ==========================================
# 6. TEST DEFECTIVE IMAGES
# ==========================================

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

                image_path = os.path.join(
                    defect_folder,
                    filename
                )

                features = extract_features(image_path)

                score = torch.norm(
                    features.squeeze(0) - normal_reference,
                    p=2
                )

                defect_scores.append(score.item())

        detected = sum(
            score > threshold
            for score in defect_scores
        )

        results[category] = defect_scores

        print("\nCategory:", category)
        print("Total images:", len(defect_scores))
        print("Mean score:", sum(defect_scores) / len(defect_scores))
        print(
            "Flagged as anomalous:",
            detected,
            "/",
            len(defect_scores)
        )

    return results


# ==========================================
# 7. GENERATE ACTUAL AND PREDICTED LABELS
# ==========================================

def get_predictions(normal_scores, defect_results, threshold):

    actual_labels = []
    predicted_labels = []
    all_scores = []

    # Normal images: actual label = 0
    for score in normal_scores:

        actual_labels.append(0)

        predicted_labels.append(
            1 if score > threshold else 0
        )

        all_scores.append(score)

    # Defective images: actual label = 1
    for category_scores in defect_results.values():

        for score in category_scores:

            actual_labels.append(1)

            predicted_labels.append(
                1 if score > threshold else 0
            )

            all_scores.append(score)

    return actual_labels, predicted_labels, all_scores


# ==========================================
# 8. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    # Build normal reference
    normal_reference = Normal_ref()

    # Calculate normal test scores
    normal_scores = testAnomaly(normal_reference)

    # Calculate threshold
    threshold = np.percentile(normal_scores, 95)

    print("\nThreshold:", threshold)

    # Calculate defective image scores
    defect_results = check_defect(
        normal_reference,
        threshold
    )

    # Generate labels and scores
    actual_labels, predicted_labels, all_scores = get_predictions(
        normal_scores,
        defect_results,
        threshold
    )

    print("\nExperiment completed.")
    print("Total evaluated images:", len(actual_labels))
    print("Actual labels:", len(actual_labels))
    print("Predicted labels:", len(predicted_labels))
