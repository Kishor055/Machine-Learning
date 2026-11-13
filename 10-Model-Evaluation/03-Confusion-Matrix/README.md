
# 03. Confusion Matrix

## 📌 Overview

A **Confusion Matrix** is a performance evaluation tool used in Machine Learning to measure the performance of a classification model.

It compares the actual class labels with the predicted class labels to identify how many predictions are correct and incorrect.

A confusion matrix helps us understand not only how accurate a model is but also what types of errors it makes.

It is commonly used to evaluate classification algorithms such as:

- Logistic Regression
- Decision Trees
- Random Forest
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machines (SVM)
- Artificial Neural Networks

---

## 🎯 Learning Objectives

After completing this topic, you will be able to:

- Understand the concept of a confusion matrix.
- Identify True Positive, True Negative, False Positive, and False Negative.
- Construct a confusion matrix.
- Calculate classification evaluation metrics.
- Understand Precision, Recall, F1-Score, and Accuracy.
- Implement a confusion matrix using Python.
- Visualize a confusion matrix using Seaborn.
- Understand binary and multiclass confusion matrices.
- Interpret classification model performance.

---

## 1. What is a Confusion Matrix?

A confusion matrix is a table that summarizes the performance of a classification model by comparing actual values with predicted values.

For binary classification, it consists of four important components:

1. True Positive (TP)
2. True Negative (TN)
3. False Positive (FP)
4. False Negative (FN)

### Confusion Matrix Structure

| Actual / Predicted | Positive | Negative |
|---|---|---|
| Positive | True Positive (TP) | False Negative (FN) |
| Negative | False Positive (FP) | True Negative (TN) |

### Visualization

```text
                    Predicted
                 Positive  Negative
              +---------------------+
Actual Positive|   TP    |    FN     |
              +---------------------+
Actual Negative|   FP    |    TN     |
              +---------------------+
```

---

## 2. Components of Confusion Matrix

### 2.1 True Positive (TP)

True Positive represents the number of positive observations correctly predicted as positive by the model.

**Example:**

A disease detection model correctly identifies 80 patients who actually have the disease.

TP = 80

### 2.2 True Negative (TN)

True Negative represents the number of negative observations correctly predicted as negative.

**Example:**

The model correctly identifies 90 healthy patients who do not have the disease.

TN = 90

### 2.3 False Positive (FP)

False Positive occurs when the model incorrectly predicts a negative observation as positive.

It is also called a **Type I Error**.

**Example:**

The model predicts that a healthy person has a disease.

FP = 10

### 2.4 False Negative (FN)

False Negative occurs when the model incorrectly predicts a positive observation as negative.

It is also called a **Type II Error**.

**Example:**

The model predicts that a person with a disease is healthy.

FN = 20

---

## 3. Example of Confusion Matrix

Consider a machine learning model designed to detect whether a patient has a disease.

Suppose the model produces the following results:

| Actual / Predicted | Disease | No Disease |
|---|---:|---:|
| Disease | 80 | 20 |
| No Disease | 10 | 90 |

From the matrix:

- True Positive (TP) = 80
- True Negative (TN) = 90
- False Positive (FP) = 10
- False Negative (FN) = 20

Total observations:

```text
Total = TP + TN + FP + FN

Total = 80 + 90 + 10 + 20

Total = 200
```

---

## 4. Evaluation Metrics Using Confusion Matrix

A confusion matrix is used to calculate several important classification evaluation metrics.

### 4.1 Accuracy

Accuracy measures the proportion of correctly classified observations out of the total observations.

**Formula:**

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Example:**

```text
Accuracy = (80 + 90) / 200

Accuracy = 170 / 200

Accuracy = 0.85

Accuracy = 85%
```

### 4.2 Precision

Precision measures how many observations predicted as positive are actually positive.

**Formula:**

```text
Precision = TP / (TP + FP)
```

**Example:**

```text
Precision = 80 / (80 + 10)

Precision = 80 / 90

Precision = 0.8889

Precision = 88.89%
```

### 4.3 Recall (Sensitivity)

Recall measures how many actual positive observations are correctly identified by the model.

**Formula:**

```text
Recall = TP / (TP + FN)
```

**Example:**

```text
Recall = 80 / (80 + 20)

Recall = 80 / 100

Recall = 0.80

Recall = 80%
```

### 4.4 Specificity

Specificity measures how many actual negative observations are correctly identified.

**Formula:**

```text
Specificity = TN / (TN + FP)
```

**Example:**

```text
Specificity = 90 / (90 + 10)

Specificity = 90 / 100

Specificity = 90%
```

### 4.5 F1-Score

F1-Score is the harmonic mean of Precision and Recall.

It balances both metrics and is particularly useful when dealing with imbalanced datasets.

**Formula:**

```text
F1-Score = 2 * (Precision * Recall) /
           (Precision + Recall)
```

**Example:**

```text
F1-Score = 2 * (0.8889 * 0.80) /
           (0.8889 + 0.80)

F1-Score = 0.8421

F1-Score = 84.21%
```

---

## 5. Python Implementation

### Step 1: Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
```

### Step 2: Define Actual and Predicted Values

```python
# Actual class labels
y_actual = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]

# Predicted class labels
y_predicted = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
```

Here:

- 1 represents Positive.
- 0 represents Negative.

### Step 3: Generate Confusion Matrix

```python
cm = confusion_matrix(y_actual, y_predicted)

print("Confusion Matrix:")
print(cm)
```

**Output:**

```text
Confusion Matrix:

[[4 1]
 [1 4]]
```

Interpretation:

- TN = 4
- FP = 1
- FN = 1
- TP = 4

### Step 4: Calculate Evaluation Metrics

```python
accuracy = accuracy_score(y_actual, y_predicted)

precision = precision_score(y_actual, y_predicted)

recall = recall_score(y_actual, y_predicted)

f1 = f1_score(y_actual, y_predicted)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
```

**Output:**

```text
Accuracy: 0.8
Precision: 0.8
Recall: 0.8
F1 Score: 0.8
```

---

## 6. Confusion Matrix Visualization Using Seaborn

Heatmaps make confusion matrices easier to understand.

```python
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.title("Confusion Matrix")

plt.show()
```

### Visualization Explanation

- X-axis represents predicted labels.
- Y-axis represents actual labels.
- Diagonal values represent correct predictions.
- Off-diagonal values represent incorrect predictions.
- Darker cells indicate higher observation counts.

---

## 7. Classification Report

Scikit-learn provides a classification report containing Precision, Recall, F1-Score, and Support.

```python
report = classification_report(
    y_actual,
    y_predicted,
    target_names=["Negative", "Positive"]
)

print(report)
```

**Output:**

```text
              precision    recall  f1-score   support

Negative         0.80      0.80      0.80         5
Positive         0.80      0.80      0.80         5

Accuracy                             0.80        10

Macro avg        0.80      0.80      0.80        10
Weighted avg     0.80      0.80      0.80        10
```

### Important Terms

- **Support:** Number of actual observations belonging to each class.
- **Macro Average:** Arithmetic mean of metric values across classes.
- **Weighted Average:** Average of metric values weighted by class support.

---

## 8. Multiclass Confusion Matrix

A confusion matrix can also be used for multiclass classification problems.

In multiclass classification, the model predicts one class among three or more possible classes.

**Example:** Classification of fruits into:

- Apple
- Banana
- Orange

### Example Matrix

| Actual / Predicted | Apple | Banana | Orange |
|---|---:|---:|---:|
| Apple | 8 | 1 | 1 |
| Banana | 2 | 7 | 1 |
| Orange | 0 | 1 | 9 |

The diagonal elements represent correct predictions.

The remaining elements represent incorrect predictions.

### Python Implementation

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_actual = [
    "Apple", "Apple", "Apple",
    "Banana", "Banana", "Banana",
    "Orange", "Orange", "Orange"
]

y_predicted = [
    "Apple", "Apple", "Orange",
    "Banana", "Apple", "Banana",
    "Orange", "Banana", "Orange"
]

cm = confusion_matrix(
    y_actual,
    y_predicted,
    labels=["Apple", "Banana", "Orange"]
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Apple", "Banana", "Orange"],
    yticklabels=["Apple", "Banana", "Orange"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Multiclass Confusion Matrix")

plt.show()
```

---

## 9. Advantages of Confusion Matrix

1. Provides a detailed evaluation of classification models.
2. Identifies different types of classification errors.
3. Helps calculate Accuracy, Precision, Recall, and F1-Score.
4. Useful for binary and multiclass classification.
5. Helps identify class imbalance problems.
6. Makes model performance easy to visualize.
7. Supports comparison between different classification models.

---

## 10. Limitations of Confusion Matrix

1. Does not directly evaluate regression models.
2. Can become difficult to interpret with a large number of classes.
3. Accuracy can be misleading for imbalanced datasets.
4. Does not independently indicate prediction confidence.
5. Performance depends on the classification threshold.
6. Does not directly measure computational efficiency.

---

## 11. Real-World Applications

| Application | Purpose |
|---|---|
| Healthcare | Disease detection |
| Email Systems | Spam detection |
| Banking | Fraud detection |
| Cybersecurity | Intrusion detection |
| Computer Vision | Image classification |
| NLP | Sentiment classification |
| Manufacturing | Defect detection |

---

## 12. Confusion Matrix vs Classification Report

| Feature | Confusion Matrix | Classification Report |
|---|---|---|
| Purpose | Shows prediction counts | Summarizes performance metrics |
| Output | Matrix | Metric table |
| True/False Predictions | Yes | Not directly |
| Precision | Calculated separately | Included |
| Recall | Calculated separately | Included |
| F1-Score | Calculated separately | Included |
| Visualization | Heatmap | Text/table |

---

## 13. Best Practices

- Never evaluate a model using accuracy alone.
- Check Precision and Recall according to the application.
- Use F1-Score when balancing Precision and Recall is important.
- Consider class imbalance before selecting evaluation metrics.
- Use a validation or test dataset for evaluation.
- Choose a suitable classification threshold.
- Compare confusion matrices across different models.
- Use normalized confusion matrices when comparing datasets of different sizes.

---

## 14. Important Scikit-Learn Functions

| Function | Description |
|---|---|
| `confusion_matrix()` | Generates confusion matrix |
| `accuracy_score()` | Calculates accuracy |
| `precision_score()` | Calculates precision |
| `recall_score()` | Calculates recall |
| `f1_score()` | Calculates F1-Score |
| `classification_report()` | Generates classification metrics |
| `ConfusionMatrixDisplay` | Visualizes confusion matrix |

---

## 15. Practice Exercises

### Beginner Level

1. Create a confusion matrix for a binary classification dataset.
2. Calculate TP, TN, FP, and FN manually.
3. Calculate Accuracy, Precision, and Recall.
4. Visualize the confusion matrix using Seaborn.

### Intermediate Level

1. Generate a confusion matrix using Logistic Regression.
2. Evaluate a Decision Tree classifier.
3. Compare two models using confusion matrices.
4. Generate a classification report.

### Advanced Level

1. Implement a multiclass confusion matrix.
2. Evaluate a model using an imbalanced dataset.
3. Compare macro and weighted F1-Scores.
4. Experiment with different classification thresholds.
5. Build a complete classification evaluation pipeline.

---

## 📚 Conclusion

A confusion matrix is one of the most important tools for evaluating classification models in Machine Learning.

It provides a detailed comparison between actual and predicted values and helps identify correct predictions and classification errors.

By using TP, TN, FP, and FN, we can calculate important performance metrics such as Accuracy, Precision, Recall, Specificity, and F1-Score.

Therefore, a confusion matrix helps developers understand model performance and select the most suitable classification model for real-world applications.

---

## 🔗 References

- [Scikit-Learn Confusion Matrix Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [Scikit-Learn Classification Report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
- [Scikit-Learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Seaborn Heatmap Documentation](https://seaborn.pydata.org/generated/seaborn.heatmap.html)
