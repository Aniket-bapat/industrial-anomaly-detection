import torch

from feature_extraction import extract_features


def load_detector():

    checkpoint = torch.load(
    "../model/normal_reference.pt",
    map_location="cpu",
    weights_only=True
)

    normal_reference = checkpoint["normal_reference"]
    threshold = checkpoint["threshold"]

    return normal_reference, threshold


def predict_image(image_path):

    normal_reference, threshold = load_detector()

    features = extract_features(image_path)

    # Calculate Euclidean distance
    score = torch.norm(
        features.squeeze(0) - normal_reference,
        p=2
    ).item()

    # Classify image
    if score > threshold:
        prediction = "Defective"
    else:
        prediction = "Normal"

    return prediction, score, threshold
