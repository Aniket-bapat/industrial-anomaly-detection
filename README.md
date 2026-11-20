# 🔍 Industrial Anomaly Detection using ResNet18

An image-based industrial anomaly detection system that identifies defective and defect-free bottles using a pretrained ResNet18 convolutional neural network and Euclidean distance-based anomaly scoring.

The project uses the **MVTec Anomaly Detection (MVTec AD) dataset**, specifically the Bottle category.

## 🚀 Live Demo

**Try the application:** [Open Live Demo](https://YOUR-LIVE-DEMO-URL-HERE)

> Replace the placeholder with your deployed Streamlit application URL once deployment is complete.

## 📌 Project Overview

Quality inspection is an important task in industrial manufacturing. Detecting defective products manually can be time-consuming and inconsistent.

This project implements a deep learning-based anomaly detection approach that learns the visual characteristics of normal bottles and identifies unusual samples based on their feature representations.

Instead of training a traditional binary classifier using normal and defective images, this approach uses only defect-free training images to construct a normal reference feature vector.

New images are classified by measuring their distance from this normal reference.

### Key Features

- Pretrained ResNet18 for deep feature extraction.
- Transfer learning using ImageNet weights.
- Normal reference vector generation.
- Euclidean distance-based anomaly scoring.
- Percentile-based anomaly threshold.
- Normal and defective image classification.
- Performance evaluation using standard classification metrics.
- Interactive image analysis through a Streamlit application.

## 🧠 Methodology

The system follows these steps:

1. **Image preprocessing:** Resize images to 224 × 224, convert them into tensors, and normalize them using ImageNet statistics.
2. **Feature extraction:** Use pretrained ResNet18 to extract 512-dimensional feature vectors.
3. **Normal reference generation:** Calculate the mean feature vector of all defect-free training images.
4. **Anomaly scoring:** Calculate the Euclidean distance between the extracted image features and the normal reference.
5. **Threshold calculation:** Determine an anomaly threshold using the 95th percentile of normal validation scores.
6. **Classification:** Classify an image as normal or defective based on its anomaly score.

### Mathematical Formulation

**Normal Reference Vector**

R = (1/N) Σ Fᵢ

Where:
- R represents the normal reference vector.
- N represents the number of normal training images.
- Fᵢ represents the feature vector of each normal image.

**Euclidean Distance**

D(x, R) = √Σ(xᵢ - Rᵢ)²

**Anomaly Classification**

- If D ≤ Threshold → Normal
- If D > Threshold → Defective

## 📂 Dataset

This project uses the official **MVTec Anomaly Detection (MVTec AD)** dataset.

- Dataset: MVTec AD
- Selected category: Bottle
- Training data: Defect-free bottle images.
- Testing data: Normal and defective bottle images.
- Defect categories: Broken Large, Broken Small, and Contamination.

**Official dataset website:**

https://www.mvtec.com/research-teaching/datasets/mvtec-ad

### Dataset Structure

After downloading and extracting the dataset, the relevant folder structure is:

```text
dataset/
└── bottle/
    ├── train/
    │   └── good/
    │
    ├── test/
    │   ├── good/
    │   ├── broken_large/
    │   ├── broken_small/
    │   └── contamination/
    │
    └── ground_truth/
        ├── broken_large/
        ├── broken_small/
        └── contamination/
```

The training directory contains only normal images, while the testing directory contains both normal and anomalous images.

## 🏗️ Project Structure

```text
Anomaly_Detection/
│
├── app.py
│
├── train.py
│
├── feature_extraction.py
│
├── anomaly_detection.py
│
├── evaluation.py
│
├── model/
│   └── normal_reference.pt
│
├── dataset/
│   └── bottle/
│       ├── train/
│       ├── test/
│       └── ground_truth/
│
├── requirements.txt
│
└── README.md
```

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Torchvision | ResNet18 pretrained model |
| ResNet18 | Feature extraction |
| NumPy | Numerical computations |
| Scikit-learn | Performance evaluation |
| Pillow | Image processing |
| Streamlit | Web application interface |

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR-GITHUB-REPOSITORY-URL
cd Anomaly_Detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download MVTec AD from the official website and extract the Bottle category into the `dataset/bottle/` directory.

### 5. Generate the normal reference

Run:

```bash
python train.py
```

This extracts normal image features, calculates the normal reference vector, determines the anomaly threshold, and saves the reference in:

```text
model/normal_reference.pt
```

### 6. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Model Evaluation

The proposed anomaly detection system is evaluated using the following metrics:

- **Accuracy:** Measures the overall proportion of correctly classified images.
- **Precision:** Measures how many images predicted as defective are actually defective.
- **Recall:** Measures how many actual defective images are successfully detected.
- **F1-score:** Represents the harmonic mean of precision and recall.
- **Confusion Matrix:** Shows correct and incorrect normal and defective classifications.

The evaluation considers both normal and defective test images.

Actual performance results should be added here after running the complete evaluation.

## 🔬 Experimental Approach

This project uses a pretrained ResNet18 model as a feature extractor.

The final classification layer is removed, and the resulting 512-dimensional feature representation is used for anomaly detection.

A single normal reference vector is calculated by averaging the feature vectors of normal training images.

Anomaly scores are calculated using Euclidean distance, and images exceeding the selected threshold are classified as anomalous.

This provides a simple baseline for unsupervised industrial anomaly detection.

## ⚠️ Limitations and Future Improvements

Current limitations:

- Uses a single mean reference vector to represent normal images.
- Uses global image-level features rather than localized defect features.
- May struggle with small or subtle defects.
- Classification performance depends on the selected threshold.

Future improvements:

- Implement anomaly localization using heatmaps.
- Explore PatchCore and other advanced anomaly detection approaches.
- Add multi-category industrial defect detection.
- Introduce batch image processing.
- Add an interactive performance dashboard.
- Store detection history and generate downloadable reports.

## 📚 References

1. Bergmann, P., Fauser, M., Sattlegger, D., and Steger, C. (2019). *MVTec AD – A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR 2019.

2. Bergmann, P., Batzner, K., Fauser, M., Sattlegger, D., and Steger, C. (2021). *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* International Journal of Computer Vision, 129, 1038–1059.

3. He, K., Zhang, X., Ren, S., and Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016.

## 📄 Dataset License

The MVTec AD dataset is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

Please refer to the official dataset website for licensing conditions. Commercial use of the dataset is not permitted under these terms.

## 👨‍💻 Author

**Your Name**

Industrial Anomaly Detection using Deep Learning and Computer Vision.

---

⭐ If you find this project useful, consider giving the repository a star!