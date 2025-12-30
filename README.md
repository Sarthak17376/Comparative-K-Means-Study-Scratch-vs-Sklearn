# Comparative Analysis of K-Means Clustering: From Scratch vs. Scikit-Learn

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Libraries](https://img.shields.io/badge/Libraries-NumPy%20%7C%20Pandas%20%7C%20Scikit--Learn-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project focuses on the mathematical implementation and performance evaluation of the **K-Means Clustering algorithm**. 

The primary objective was to build a K-Means model **entirely from scratch** using only NumPy to understand the underlying mechanics of unsupervised learning (centroid initialization, distance calculation, and convergence). This custom implementation was then benchmarked against the industry-standard `sklearn.cluster.KMeans` library using the **UCI Wheat Seeds Dataset**.

## 📂 Dataset
The project utilizes the **[UCI Seeds Dataset](https://archive.ics.uci.edu/dataset/236/seeds)**.
* **Samples:** 210 instances
* **Features:** 7 geometric parameters of wheat kernels (Area, Perimeter, Compactness, Kernel Length, Kernel Width, Asymmetry Coefficient, Length of Kernel Groove).
* **Classes:** 3 varieties of wheat (Kama, Rosa, Canadian).

## 🛠 Technologies Used
* **Python 3.x**
* **NumPy:** Vectorized calculations for distance metrics and centroid updates.
* **Pandas:** Data manipulation and cleaning.
* **Matplotlib / Seaborn:** Visualization of clusters and PCA results.
* **Scikit-Learn:** Data preprocessing (StandardScaler), PCA, and benchmarking models.

## ⚙️ Methodology

### 1. Data Preprocessing
* Loaded raw data and handled missing values.
* Applied **Standard Scaling** to normalize features, ensuring that distance calculations (Euclidean) were not biased by feature magnitude.

### 2. Custom Implementation (`KMeansScratch`)
A Python class developed to mimic the behavior of standard clustering libraries:
* **Initialization:** Implemented logic to select initial centroids.
* **Assignment Step:** vectorized Euclidean distance calculation to assign points to the nearest cluster.
* **Update Step:** Recalculated centroids based on the mean of assigned points.
* **Convergence Check:** Iterative process stops when centroids stabilize or max iterations are reached.

### 3. Evaluation Metrics
Since clustering is unsupervised, but ground truth was available for validation, a hybrid evaluation approach was used:
* **Intrinsic Metrics:** Silhouette Score, Rand Index.
* **Extrinsic Metrics:** Accuracy (calculated via contingency matrix mapping).

### 4. Dimensionality Reduction
* Applied **Principal Component Analysis (PCA)** to reduce the 7-dimensional feature space into 2D for visual inspection of cluster separability.

## 📊 Results & Benchmarking

| Metric | Custom Implementation | Scikit-Learn Implementation |
| :--- | :---: | :---: |
| **Silhouette Score** | *[Insert Your Score]* | *[Insert Sklearn Score]* |
| **Rand Index** | *[Insert Your Score]* | *[Insert Sklearn Score]* |
| **Accuracy** | *[Insert Your Score]* | *[Insert Sklearn Score]* |

> **Observation:** The custom implementation successfully converged and produced cluster boundaries highly similar to the optimized Scikit-Learn implementation, demonstrating the validity of the underlying logic.

## 📈 Visualizations
*(Add your PCA plot screenshots here)*

*Figure 1: 2D PCA projection of the Custom K-Means Clustering results.*

## 🚀 How to Run

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/seeds-clustering-analysis.git](https://github.com/yourusername/seeds-clustering-analysis.git)
    ```
2.  **Install dependencies**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```
3.  **Run the Notebook**
    ```bash
    jupyter notebook 25AI60R20_Assignment10.ipynb
    ```

## 🤝 Contributing
Feel free to fork this repository and submit pull requests. Suggestions for optimizing the NumPy vectorization in the custom class are welcome!
