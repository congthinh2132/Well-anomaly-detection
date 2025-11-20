# Oil Well Anomaly Detection System
<img width="480" height="270" alt="image" src="https://github.com/user-attachments/assets/da58dc51-902b-4b13-89c0-e007ac37515a" />

This project implements an Anomaly Detection framework for oil well sensor data (Time-Series). It compares statistical, machine learning, and deep learning approaches to identify operational faults in real-time.

## 📌 Overview

The system processes pressure and temperature sensor data from oil wells to detect abnormal behaviors (e.g., flow instability, sensor failures, or occlusion). It utilizes a pipeline involving data preprocessing, feature selection, noise injection for robustness, and sliding-window post-processing to minimize false alarms.

## 🚀 Key Features

* **Data Preprocessing**: Handling missing values, timestamp alignment, and MinMax scaling.
* **Feature Selection**: Correlation heatmap analysis to select high-impact sensors.
* **Robust Training**: Implementation of **Denoising Autoencoders** by injecting Gaussian noise into training data.
* **Post-Processing**: A sliding window majority-voting mechanism to smooth predictions and improve precision.
* **Grid Search**: Automated hyperparameter tuning for LOF and Isolation Forest.

## 📂 Dataset

The project utilizes oil & gas time-series data (likely the **Petrobras 3W Dataset**).
**Selected Features:**
* `P-PDG`: Permanent Downhole Gauge Pressure
* `P-TPT`: Temperature Pressure Transducer (Pressure)
* `T-TPT`: Temperature Pressure Transducer (Temperature)
* `P-MON-CKP`: Upstream Pressure of Production Choke

## 🧠 Models Implemented

1.  **Local Outlier Factor (LOF)**: A density-based method that detects anomalies by comparing the local density of a point to its neighbors.
2.  **Isolation Forest (IF)**: An ensemble method that isolates anomalies by randomly partitioning the data space.
3.  **LSTM Autoencoder**: A Deep Learning model (Denoising) that learns to reconstruct normal time-series patterns. High reconstruction error indicates an anomaly.

## 🛠️ Installation & Requirements

### Prerequisites
* Python 3.8+
* Jupyter Notebook

### Libraries
Install the dependencies using pip:

    conda env create -f environment.yml

## ⚙️ Usage

1.  **Place Data**: Ensure your parquet file (e.g., `WELL-00025_20200629194141.parquet`) is in the project root.
2.  **Run Notebook**: Open `Anomaly.ipynb` and execute the cells.
3.  **Training**:
    * The notebook automatically performs Grid Search for LOF and IF.
    * It trains the LSTM Autoencoder and calculates the reconstruction threshold (99th percentile).
4.  **Artifacts**: Trained models and scalers are saved in the `models/` directory:
    * `models/my_scaler.joblib`
    * `models/my_lof_model.joblib`
    * `models/my_if_model.joblib`
    * `models/my_lstm_model.keras`
    * `models/my_lstm_threshold.json`

## 📊 Results

The models achieved high performance metrics on the test set, largely due to the denoising training strategy and sliding window smoothing.

| Model | F1 Score | Accuracy |
|-------|----------|----------|
| **LSTM Autoencoder** | ~0.99 | ~0.99 |
| **Isolation Forest** | ~0.99 | ~0.99 |
| **LOF** | ~0.99 | ~0.99 |

*Note: Metrics are based on binary classification (Normal=0, Anomaly=1) after post-processing.*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[MIT License](LICENSE)
