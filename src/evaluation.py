from feature_extraction import (
    Normal_ref,
    testAnomaly,
    check_defect,
    get_predictions
)
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

normal_reference = Normal_ref()


normal_scores = testAnomaly(normal_reference)


threshold = np.percentile(normal_scores, 95)


defect_results = check_defect(
    normal_reference,
    threshold
)


actual_labels, predicted_labels, all_scores = get_predictions(
    normal_scores,
    defect_results,
    threshold
)




from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    actual_labels,
    predicted_labels
)

print("\nConfusion Matrix:")
print(cm)


accuracy = accuracy_score(actual_labels, predicted_labels)

precision = precision_score(actual_labels, predicted_labels)

recall = recall_score(actual_labels, predicted_labels)

f1 = f1_score(actual_labels, predicted_labels)

print("\nEvaluation Metrics")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)