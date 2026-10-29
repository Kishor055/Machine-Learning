# K-Nearest Neighbors (KNN)

## 📌 Overview

K-Nearest Neighbors (KNN) is a supervised machine learning algorithm used for **classification and regression**. It predicts the output for a new data point by identifying the \(K\) closest training examples based on a distance metric.

For classification, KNN assigns the class that receives the most votes among the nearest neighbors. For regression, it typically predicts the average target value of those neighbors.

KNN is a non-parametric, instance-based learning algorithm because it stores training examples rather than learning an explicit model during training.

## 🎯 Learning Objectives

By completing this module, you will learn how to:

- Understand the fundamentals of K-Nearest Neighbors.
- Explain how KNN makes predictions.
- Calculate distances between data points.
- Implement KNN using Scikit-learn.
- Build a KNN classifier from scratch.
- Select an appropriate value of \(K\).
- Understand the importance of feature scaling.
- Evaluate classification performance using appropriate metrics.
- Visualize decision boundaries.
- Understand the advantages and limitations of KNN.

## 🧠 What Is K-Nearest Neighbors?

KNN predicts the class or value of a new observation using the closest examples in the training dataset.

For example, suppose a model classifies a fruit using its weight and size. If most of the nearest fruits belong to the apple class, the model predicts that the new fruit is an apple.

The algorithm follows three main steps:

1. Calculate the distance between the new observation and training observations.
2. Select the \(K\) nearest neighbors.
3. Predict the class using majority voting or predict a numerical value using aggregation.

## ⚙️ How KNN Works

1. Load and preprocess the dataset.
2. Separate features and target labels.
3. Split the dataset into training and testing sets.
4. Scale numerical features when appropriate.
5. Choose a value for \(K\).
6. Calculate distances between the query point and training samples.
7. Identify the nearest neighbors.
8. Generate a prediction based on those neighbors.
9. Evaluate the model using suitable metrics.

Unlike many other algorithms, KNN does not require a conventional parameter-fitting process. Most of its computational work happens during prediction.

## 📐 Distance Metrics

KNN uses a distance metric to determine which observations are nearest to a query point.

### 1. Euclidean Distance

Euclidean distance measures the straight-line distance between two points.

\[
d(x,y)=\sqrt{\sum_{i=1}^{n}(x_i-y_i)^2}
\]

**Example:** For points \(A=(2,3)\) and \(B=(5,7)\),

\[
d(A,B)=\sqrt{(5-2)^2+(7-3)^2}=5
\]

### 2. Manhattan Distance

Manhattan distance measures the sum of absolute differences across features.

\[
d(x,y)=\sum_{i=1}^{n}|x_i-y_i|
\]

### 3. Minkowski Distance

Minkowski distance generalizes Euclidean and Manhattan distance.

\[
d(x,y)=\left(\sum_{i=1}^{n}|x_i-y_i|^p\right)^{1/p}
\]

- \(p=1\): Manhattan distance.
- \(p=2\): Euclidean distance.

## 🔢 Choosing the Value of K

The value of \(K\) determines how many neighbors influence a prediction.

- **Small \(K\):** More sensitive to noise and local patterns; may overfit.
- **Large \(K\):** Produces smoother decision boundaries but may underfit.
- **Odd \(K\):** Can reduce tied votes in binary classification, although ties remain possible in multiclass problems.

A common starting point is approximately \(\sqrt{n}\), where \(n\) is the number of training samples. This is only a heuristic, not a universal rule.

The best value should be selected using cross-validation.

## 📊 Understanding KNN Classification

Consider a dataset with two classes: Class A and Class B.

If \(K=5\) and the nearest neighbors contain:

- 3 observations from Class A.
- 2 observations from Class B.

The majority-vote prediction is **Class A**.

The estimated probability for a class can be based on the proportion of neighbors belonging to that class, depending on the implementation.

## 📂 Suggested Folder Structure

```text
09-Classification/
└── 02-K-Nearest-Neighbors/
    ├── README.md
    ├── knn-classifier.py
    ├── from-scratch.py
    ├── visualization.py
    ├── practice.md
    └── requirements.txt
```

## 💻 Implementation Using Scikit-learn

This example uses the Iris dataset to classify flowers into three species.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create a pipeline to prevent data leakage
pipeline = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier()
)

# Tune K using cross-validation
param_grid = {
    "kneighborsclassifier__n_neighbors": [3, 5, 7, 9, 11, 13]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy"
)

# Train and tune the model
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("Best K:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=data.target_names))
```

**Important:** Feature scaling is included in the pipeline so that each fold learns its scaling parameters from its training data only. The test set remains separate from model selection.

## 🛠️ Implementation From Scratch

KNN can be implemented using NumPy without relying on Scikit-learn for the algorithm itself.

The main steps are:

1. Store the training features and labels.
2. Calculate the distance from a query point to every training point.
3. Sort the distances.
4. Select the \(K\) nearest observations.
5. Count the class labels among those neighbors.
6. Return the most frequent class.

Example distance calculation:

```python
import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

point_a = np.array([2, 3])
point_b = np.array([5, 7])

print("Distance:", euclidean_distance(point_a, point_b))
```

Expected output:

```text
Distance: 5.0
```

For a complete classifier, the distance function must be combined with nearest-neighbor selection and majority voting.

## 📈 Feature Scaling

Feature scaling is particularly important for KNN because distance calculations can be dominated by features with larger numerical ranges.

For example, if one feature represents age from 18 to 80 and another represents annual income from 20,000 to 200,000, income may dominate the distance calculation.

Common scaling methods include:

- **StandardScaler:** Transforms features using the training mean and standard deviation.
- **MinMaxScaler:** Scales features to a specified range, commonly 0 to 1.

Always fit the scaler on training data only. A Scikit-learn pipeline helps prevent data leakage during cross-validation.

## 📏 Model Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy | Proportion of correct predictions |
| Precision | Proportion of predicted positives that are correct |
| Recall | Proportion of actual positives correctly identified |
| F1-score | Harmonic mean of precision and recall |
| Confusion Matrix | Displays correct and incorrect class predictions |
| Cross-validation Score | Estimates performance across multiple training-validation splits |

For imbalanced datasets, consider precision, recall, F1-score, and balanced accuracy rather than relying only on accuracy.

## 📉 Advantages

- Simple and intuitive algorithm.
- Supports both classification and regression.
- Requires few assumptions about the data distribution.
- Can model nonlinear decision boundaries.
- Naturally supports multiclass classification.
- Easy to implement for small datasets.

## ⚠️ Limitations

- Prediction can be slow for large training datasets.
- Requires storing the training data.
- Sensitive to feature scaling and irrelevant features.
- Performance can degrade in high-dimensional spaces.
- Sensitive to noise and the choice of \(K\).
- Can be affected by imbalanced classes and uneven class densities.
- Distance calculations can become expensive as the dataset grows.

## 🌍 Real-World Applications

- **Healthcare:** Finding patients with similar measured characteristics.
- **Recommendation Systems:** Identifying similar users or items.
- **Image Recognition:** Classifying images using numerical feature representations.
- **Finance:** Identifying observations similar to known financial cases.
- **Pattern Recognition:** Classifying objects based on feature similarity.
- **Education:** Finding students with similar learning performance patterns.

## 🔍 Important Concepts to Remember

- KNN is a supervised, non-parametric, instance-based algorithm.
- It supports classification and regression.
- Distance metrics determine which observations are considered nearest.
- Feature scaling is important when features have different units or ranges.
- Small \(K\) values may overfit, while large values may underfit.
- Cross-validation helps choose an appropriate \(K\).
- KNN's prediction cost can be high because it compares new observations with stored training data.

## 📝 Practice Exercises

1. Implement a KNN classifier using Scikit-learn.
2. Build a KNN classifier from scratch using NumPy.
3. Compare Euclidean and Manhattan distance.
4. Train models using different values of \(K\).
5. Compare results with and without feature scaling.
6. Plot decision boundaries for a two-feature dataset.
7. Evaluate a multiclass classifier using a confusion matrix.
8. Tune \(K\) using cross-validation.
9. Compare uniform and distance-based neighbor weighting.
10. Compare KNN with Logistic Regression and Decision Trees on the same dataset.

## 📚 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## 🚀 How to Run

Navigate to this directory:

```bash
cd 09-Classification/02-K-Nearest-Neighbors
```

Install the required libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Run the Scikit-learn implementation:

```bash
python knn-classifier.py
```

Run the from-scratch implementation:

```bash
python from-scratch.py
```

Run the visualization script:

```bash
python visualization.py
```

## 📖 References

- [Scikit-learn: Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [Scikit-learn: KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [An Introduction to Statistical Learning](https://www.statlearning.com/)

---

**Next Topic:** Classification using Support Vector Machines (SVM).
