# Oil Well Sensor Anomaly Detection
<img width="480" height="270" alt="image" src="https://github.com/user-attachments/assets/da58dc51-902b-4b13-89c0-e007ac37515a" />

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-yellow)
![Status](https://img.shields.io/badge/Status-Research%20%26%20Development-green)

## 📌 Project Overview

This project implements a robust **Semi-Supervised Anomaly Detection** system for oil well downhole sensors. By leveraging historical sensor data (Pressure and Temperature), the system detects abnormal behaviors that could indicate sensor failures, equipment malfunction, or irregular well operations.

The primary objective is to identify abnormal behaviors—such as sensor failures, equipment malfunctions, or irregular well operations—by analyzing historical pressure and temperature readings. Our approach is grounded in the philosophy of **Novelty Detection**, where models are trained exclusively on "Normal" operational data to establish a baseline of expected behavior. During inference, any significant deviation from this learned baseline is flagged as a potential anomaly. To address the inherent noise in high-frequency sensor data, we have implemented a **Sliding Window Smoothing** post-processing technique, which significantly reduces false positives and enhances the reliability of our alerts.

## 📊 The Dataset

The foundation of this analysis is high-frequency sensor data stored in the Parquet file format (WELL-00025_20200629194141.parquet), sourced from the comprehensive Petrobras 3W Dataset. This dataset is renowned for capturing the complex dynamics of offshore oil production.


### Selected Features
After correlation analysis and feature selection, the following key sensors were utilized:

| Feature Name | Description | Type |
| :--- | :--- | :--- |
| **P-PDG** | Permanent Downhole Gauge Pressure | Continuous |
| **P-TPT** | Temperature-Pressure Transducer Pressure | Continuous |
| **T-TPT** | Temperature-Pressure Transducer Temperature | Continuous |
| **P-MON-CKP** | Choke Pressure Monitor | Continuous |

### Labels
* **Class 0:** Normal Operation
* **Class 1-108:** Various Anomaly Types (treated collectively as "Anomaly" for binary classification)

## 🛠️ Methodology

### 1. Data Preprocessing
* **Cleaning:** Removal of columns with >40% missing values and rows with undefined states.
* **Splitting:**
    * **Training Set:** Composed **only** of normal data (first 60% of time-series).
    * **Testing Set:** Contains the remaining normal data mixed with all anomalous events.
* **Scaling:** `MinMaxScaler` is applied to normalize features between 0 and 1.
* **Noise Injection:** Gaussian noise is added to the training data. This acts as a regularizer, preventing the models from simply "memorizing" the data and helping them learn robust features (Denoising approach).

### 2. Models Implemented
We experimented with three distinct architectures to find the best performer:

#### A. Local Outlier Factor (LOF)
* **Type:** Density-based.
* **Configuration:** Used in `novelty=True` mode to perform semi-supervised learning.
* **Hyperparameters:** Grid search performed over `n_neighbors` and distance `metric` (Euclidean/Minkowski).

#### B. Isolation Forest (iForest)
* **Type:** Tree-based / Ensemble.
* **Mechanism:** Isolates observations by randomly selecting a feature and then randomly selecting a split value. Anomalies are easier to isolate (shorter path lengths).
* **Hyperparameters:** Grid search performed over `n_estimators`, `contamination`, and `max_samples`.

#### C. LSTM Autoencoder (Deep Learning)
* **Type:** Sequence-based Reconstruction.
* **Architecture:**
    * **Encoder:** Compression of time-series sequences into a latent vector using LSTM layers.
    * **Decoder:** Reconstruction of the original sequence from the latent vector.
* **Logic:** The model learns to reconstruct "Normal" patterns with low error. When presented with an "Anomaly", the reconstruction error (MSE) spikes.
* **Thresholding:** Dynamic thresholding based on the 99th percentile of training reconstruction errors.

### 3. Post-Processing (Smoothing)
Raw model predictions on time-series data are often noisy (flickering between normal/anomaly). We implemented a **Sliding Window Rule**:
* A window of size $N$ moves over the raw predictions.
* An anomaly is confirmed only if $>90\%$ of points in the window are flagged as anomalous.
* This significantly improves Precision and F1-Score.

## 🚀 Installation & Usage

### Prerequisites
* Python 3.8+
* Jupyter Notebook

### Installation
1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/oil-well-anomaly-detection.git](https://github.com/yourusername/oil-well-anomaly-detection.git)
    cd oil-well-anomaly-detection
    ```

2.  Install dependencies:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn tensorflow tqdm joblib
    ```

### Running the Analysis
1.  Place your `.parquet` data file in the root directory.
2.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook Anomaly.ipynb
    ```
3.  Run all cells to execute the training pipeline. The notebook will:
    * Visualize feature correlations.
    * Train models (LOF, IF, LSTM).
    * Perform Grid Search for hyperparameter optimization.
    * Save the best models to the `models/` directory.


## 📈 Results & Evaluation

Models are evaluated using the **F1-Score** on the test set, which balances Precision and Recall. This is critical as anomalies are rare events (class imbalance).

### Comparative Performance

We evaluated the models both with raw predictions and with the **Sliding Window Smoothing** post-processing applied. The results demonstrate that smoothing significantly reduces false positives and improves overall stability.

| Model | Configuration | F1-Score | Accuracy | Precision | TN | FP | FN | TP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LSTM Autoencoder** | **With Sliding Window** | **0.9986** | **0.9973** | **0.9972** | **32,496** | **1,787** | **0** | **632,813** |
| Isolation Forest | With Sliding Window | 0.9985 | 0.9971 | 0.9970 | 32,354 | 1,929 | 0 | 632,813 |
| LSTM Autoencoder | Raw Predictions | 0.9982 | 0.9966 | 0.9965 | 32,047 | 2,236 | 0 | 632,813 |
| Isolation Forest | Without Sliding Window | 0.9980 | 0.9962 | 0.9960 | 31,772 | 2,511 | 0 | 632,813 |
| Local Outlier Factor | With Sliding Window | 0.9978 | 0.9958 | 0.9956 | 31,490 | 2,793 | 0 | 632,813 |
| Local Outlier Factor | Without Sliding Window | 0.9965 | 0.9934 | 0.9931 | 29,856 | 4,427 | 0 | 632,813 |

**Note:**
* **TN (True Negative):** Correctly identified Normal data.
* **FP (False Positive):** Normal data incorrectly flagged as Anomaly.
* **FN (False Negative):** Anomaly incorrectly identified as Normal.
* **TP (True Positive):** Correctly identified Anomaly.

### Key Findings

1.  **Best Performer:** The **LSTM Autoencoder with Sliding Window** achieved the highest F1-score (**0.9986**). Its ability to capture temporal dependencies in the sensor data allows it to distinguish between transient noise and actual anomalies more effectively than density-based methods.
2.  **Impact of Smoothing:** Across all three architectures, the sliding window technique consistently improved F1-Scores and Accuracy. For example, LOF saw a significant jump from 0.9965 to 0.9978, highlighting the importance of temporal consistency in anomaly detection.
3.  **Model Comparison:** Isolation Forest proved to be a very strong runner-up, performing nearly as well as the deep learning approach while being computationally lighter to train.

## 🔮 Future Work
* Integration with real-time streaming API.
* Experimenting with Transformer-based Autoencoders.
* Adding classification to identify *specific* types of anomalies (Multiclass classification) once an anomaly is detected.

## 📝 License
[MIT](https://choosealicense.com/licenses/mit/)
