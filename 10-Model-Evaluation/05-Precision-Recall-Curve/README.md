
# 05. Precision-Recall Curve

## 📌 Overview

The **Precision-Recall (PR) Curve** is an important evaluation technique used to measure the performance of classification models, especially when working with imbalanced datasets.

It represents the relationship between Precision and Recall at different classification thresholds.

A Precision-Recall Curve helps determine how effectively a classification model identifies positive observations while minimizing false positive predictions.

It is particularly useful in applications where identifying positive cases correctly is important, such as:

- Medical disease detection
- Fraud detection
- Spam email classification
- Cybersecurity threat detection
- Defect detection
- Rare event prediction

The Precision-Recall Curve is closely related to ROC-AUC but focuses specifically on the performance of the positive class.

---

## 🎯 Learning Objectives

After completing this topic, you will be able to:

- Understand Precision and Recall.
- Understand the Precision-Recall Curve.
- Calculate Precision and Recall.
- Understand classification thresholds.
- Interpret Precision-Recall Curves.
- Understand Average Precision (AP).
- Implement PR Curves using Scikit-Learn.
- Visualize PR Curves using Matplotlib.
- Compare multiple classification models.
- Evaluate models on imbalanced datasets.
- Understand the difference between ROC and PR curves.

---

## 1. What is a Precision-Recall Curve?

A Precision-Recall Curve is a graphical representation of the relationship between Precision and Recall at different classification thresholds.

It consists of:

- **X-axis:** Recall
- **Y-axis:** Precision

The curve shows how Precision changes as Recall increases or decreases.

A good classification model should achieve high Precision and high Recall.

### Precision-Recall Curve Structure

```text
Precision
   1.0 |-------------------*
       |              ****
       |          ****
       |       ***
       |     **
       |   **
       | **
   0.0 |________________________
       0.0                   1.0
                  Recall
```

**Interpretation:**

- A curve closer to the top-right region generally indicates better performance.
- High Precision means fewer false positive predictions.
- High Recall means fewer false negative predictions.
- A curve closer to the bottom indicates poor performance.

---

## 2. What is Precision?

Precision measures the proportion of positive predictions that are actually positive.

It answers the question:

**Out of all observations predicted as positive, how many were actually positive?**

### Formula

```text
Precision = TP / (TP + FP)
```

Where:

- TP = True Positive
- FP = False Positive

### Example

Suppose a spam detection model predicts 100 emails as spam.

Out of these:

- 80 are actually spam.
- 20 are legitimate emails.

Therefore:

```text
Precision = 80 / (80 + 20)

Precision = 80 / 100

Precision = 0.80

Precision = 80%
```

**Interpretation:** 80% of emails predicted as spam are actually spam.

---

## 3. What is Recall?

Recall is also known as Sensitivity or True Positive Rate.

It measures the proportion of actual positive observations correctly identified by the model.

It answers the question:

**Out of all actual positive observations, how many did the model correctly identify?**

### Formula

```text
Recall = TP / (TP + FN)
```

Where:

- TP = True Positive
- FN = False Negative

### Example

Suppose a disease detection model identifies:

- 80 patients correctly as having a disease.
- 20 patients with the disease are incorrectly classified as healthy.

Therefore:

```text
Recall = 80 / (80 + 20)

Recall = 80 / 100

Recall = 0.80

Recall = 80%
```

**Interpretation:** The model correctly identifies 80% of patients who actually have the disease.

---

## 4. Understanding the Relationship Between Precision and Recall

Precision and Recall are important metrics for evaluating classification models.

However, increasing one metric can sometimes decrease the other.

### Precision vs Recall

| Precision | Recall |
|---|---|
| Focuses on reducing False Positives | Focuses on reducing False Negatives |
| Measures correctness of positive predictions | Measures coverage of actual positive cases |
| Important when false alarms are costly | Important when missing positive cases is costly |
| Higher precision means fewer false alarms | Higher recall means fewer missed positive cases |

### Example

Consider a medical diagnosis system.

**High Precision:**
- The system predicts disease only when highly confident.
- Fewer healthy patients are incorrectly diagnosed.
- Some actual patients may be missed.

**High Recall:**
- The system identifies most patients who have the disease.
- Fewer actual cases are missed.
- More healthy patients may receive false alarms.

The appropriate balance depends on the application.

---

## 5. Classification Threshold

A classification threshold is a probability cutoff used to convert predicted probabilities into class labels.

For binary classification:

```text
If Probability >= Threshold:
    Predict Positive
Else:
    Predict Negative
```

### Example

Suppose a model predicts the following probabilities:

| Observation | Predicted Probability |
|---|---:|
| A | 0.95 |
| B | 0.85 |
| C | 0.65 |
| D | 0.45 |
| E | 0.25 |

Using a threshold of 0.50:

- A = Positive
- B = Positive
- C = Positive
- D = Negative
- E = Negative

Using a threshold of 0.80:

- A = Positive
- B = Positive
- C = Negative
- D = Negative
- E = Negative

### Effect of Threshold Changes

| Threshold | Precision | Recall |
|---|---|---|
| Lower Threshold | Generally Decreases | Generally Increases |
| Higher Threshold | Generally Increases | Generally Decreases |

Changing the classification threshold changes the number of positive predictions and therefore affects Precision and Recall.

The Precision-Recall Curve evaluates these changes across different thresholds.

---

## 6. What is Average Precision (AP)?

Average Precision (AP) summarizes the Precision-Recall Curve into a single numerical value.

It measures the quality of the precision-recall relationship across different recall levels.

AP values generally range between 0 and 1.

### Formula

Average Precision is calculated using:

```text
AP = Sum of [(Recall_n - Recall_(n-1)) * Precision_n]
```

Where:

- Recall_n = Current recall value.
- Recall_(n-1) = Previous recall value.
- Precision_n = Precision at the current recall level.

Scikit-Learn calculates Average Precision using a non-interpolated weighted sum over recall increments.

### AP Interpretation

| AP Score | General Interpretation |
|---|---|
| 0.90 - 1.00 | Excellent |
| 0.80 - 0.89 | Very Good |
| 0.70 - 0.79 | Good |
| 0.60 - 0.69 | Moderate |
| 0.50 - 0.59 | Weak |
| Below 0.50 | Depends on the problem and baseline |

**Note:** These are informal guidelines, not universal performance standards. AP should be compared against an appropriate baseline, especially on imbalanced datasets.

---

## 7. Python Implementation Using Scikit-Learn

### Step 1: Import Required Libraries

```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    classification_report
)
```

### Step 2: Create a Classification Dataset

```python
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    weights=[0.90, 0.10],
    random_state=42
)
```

Here:

- 90% observations belong to class 0.
- 10% observations belong to class 1.
- The dataset represents an imbalanced classification problem.

### Step 3: Split the Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

### Step 4: Train Logistic Regression Model

```python
model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)
```

### Step 5: Predict Probabilities

```python
y_prob = model.predict_proba(X_test)[:, 1]
```

**Important:** Precision-Recall Curves require prediction scores or probabilities rather than only predicted class labels.

### Step 6: Calculate Precision and Recall

```python
precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_prob
)

print("Precision:", precision)
print("Recall:", recall)
```

### Step 7: Calculate Average Precision

```python
ap_score = average_precision_score(
    y_test,
    y_prob
)

print("Average Precision:", ap_score)
```

### Step 8: Visualize Precision-Recall Curve

```python
plt.figure(figsize=(8, 6))

plt.plot(
    recall,
    precision,
    label=f"Logistic Regression (AP = {ap_score:.3f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve")

plt.legend(loc="lower left")
plt.grid(alpha=0.3)

plt.show()
```

---

## 8. Compare Multiple Classification Models

Precision-Recall Curves can be used to compare different classification algorithms.

### Python Implementation

```python
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42
    ),

    "SVM": SVC(
        probability=True,
        random_state=42
    )
}

plt.figure(figsize=(9, 7))

for name, model in models.items():

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_test,
        y_prob
    )

    ap = average_precision_score(
        y_test,
        y_prob
    )

    plt.plot(
        recall,
        precision,
        label=f"{name} (AP = {ap:.3f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title("Precision-Recall Curve Comparison")

plt.legend()
plt.grid(alpha=0.3)

plt.show()
```

### Interpretation

- A curve closer to the top-right region generally indicates better performance.
- Higher AP generally indicates better overall Precision-Recall performance.
- Models should be compared using the same test dataset.
- The best model depends on application requirements, not AP alone.

---

## 9. Precision-Recall Curve for Imbalanced Datasets

Imbalanced datasets contain significantly different numbers of observations in different classes.

Example:

| Class | Number of Observations |
|---|---:|
| Normal Transactions | 9,900 |
| Fraud Transactions | 100 |
| Total | 10,000 |

Here, only 1% of transactions are fraudulent.

A model predicting every transaction as normal would achieve 99% accuracy but fail to detect any fraudulent transaction.

Therefore, accuracy alone is misleading.

### Why Use Precision-Recall Curve?

1. Focuses on the positive class.
2. Highlights the trade-off between false positives and false negatives.
3. Helps identify models that detect rare positive cases.
4. Provides useful insights for imbalanced classification.
5. Helps select an appropriate classification threshold.

### Baseline Precision

For a random-ranking classifier, the expected baseline precision is approximately equal to the positive class prevalence.

For the example above:

```text
Positive Class Prevalence = 100 / 10000

Positive Class Prevalence = 0.01

Baseline Precision = 1%
```

A useful model should achieve performance meaningfully above the appropriate baseline.

---

## 10. Precision-Recall Curve vs ROC Curve

Both ROC and Precision-Recall Curves evaluate classification models across different thresholds.

However, they focus on different aspects of performance.

| Feature | ROC Curve | Precision-Recall Curve |
|---|---|---|
| Full Form | Receiver Operating Characteristic | Precision-Recall |
| X-axis | False Positive Rate | Recall |
| Y-axis | True Positive Rate | Precision |
| Focus | Overall class discrimination | Positive class performance |
| Metrics | ROC-AUC | Average Precision |
| Imbalanced Data | Can hide poor positive-class precision | Often more informative |
| Main Application | General classification evaluation | Rare positive class detection |

### When to Use ROC Curve?

- When evaluating ranking performance across thresholds.
- When both positive and negative class behavior matters.
- When comparing overall class discrimination.

### When to Use Precision-Recall Curve?

- When working with highly imbalanced datasets.
- When positive cases are rare.
- When false positive predictions are costly.
- When detecting fraud, diseases, or security threats.

---

## 11. Selecting the Best Classification Threshold

The best threshold depends on the application's requirements.

For example:

- Healthcare: High Recall may be important to reduce missed disease cases.
- Spam Detection: High Precision may be important to avoid hiding legitimate emails.
- Fraud Detection: A balance between Precision and Recall may be required.

### Python Implementation

```python
import numpy as np

from sklearn.metrics import (
    precision_recall_curve,
    f1_score
)

precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_prob
)

# Calculate F1 for thresholds
f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

best_index = np.argmax(f1_scores)

best_threshold = thresholds[best_index]

print("Best Threshold:", best_threshold)
print("Best F1 Score:", f1_scores[best_index])
```

### Apply the Best Threshold

```python
y_pred = (
    y_prob >= best_threshold
).astype(int)

print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
```

**Important:** Select thresholds using validation data or cross-validation, not the final test set. Evaluate the selected threshold once on unseen test data to avoid optimistic performance estimates.

---

## 12. Advantages of Precision-Recall Curve

1. Particularly useful for imbalanced datasets.
2. Focuses on positive class performance.
3. Evaluates models across different classification thresholds.
4. Helps understand the trade-off between Precision and Recall.
5. Supports classification model comparison.
6. Helps identify suitable classification thresholds.
7. Average Precision provides a single summary score.

---

## 13. Limitations of Precision-Recall Curve

1. Does not directly evaluate negative class performance.
2. Precision depends on the positive class prevalence.
3. Can be difficult to interpret for some multiclass problems.
4. Does not directly measure prediction calibration.
5. Does not automatically identify the best classification threshold.
6. AP scores can vary significantly between datasets with different class distributions.

---

## 14. Real-World Applications

| Application | Purpose |
|---|---|
| Healthcare | Detecting rare diseases |
| Banking | Credit card fraud detection |
| Cybersecurity | Detecting network attacks |
| Email Systems | Spam detection |
| Manufacturing | Identifying defective products |
| E-Commerce | Product recommendation |
| Marketing | Identifying potential customers |
| Computer Vision | Object detection |

---

## 15. Important Scikit-Learn Functions

| Function | Description |
|---|---|
| `precision_recall_curve()` | Calculates Precision, Recall, and thresholds |
| `average_precision_score()` | Calculates Average Precision |
| `precision_score()` | Calculates Precision |
| `recall_score()` | Calculates Recall |
| `f1_score()` | Calculates F1-Score |
| `PrecisionRecallDisplay` | Visualizes Precision-Recall Curve |
| `predict_proba()` | Returns class probabilities |
| `classification_report()` | Generates classification metrics |

---

## 16. Practice Exercises

### Beginner Level

1. Create a binary classification dataset.
2. Train a Logistic Regression model.
3. Calculate Precision and Recall.
4. Plot a Precision-Recall Curve.
5. Calculate Average Precision.

### Intermediate Level

1. Compare Logistic Regression and Random Forest.
2. Evaluate a classification model on an imbalanced dataset.
3. Experiment with different classification thresholds.
4. Compare Precision-Recall and ROC curves.
5. Calculate the F1-Score.

### Advanced Level

1. Implement Precision-Recall evaluation for fraud detection.
2. Compare different models using Average Precision.
3. Implement threshold optimization.
4. Evaluate the impact of class imbalance.
5. Compare macro, micro, and weighted averaging for multiclass classification.
6. Build a complete classification evaluation pipeline.

---

## 17. Best Practices

- Use predicted probabilities or decision scores.
- Evaluate models using unseen test data.
- Use stratified train-test splitting.
- Prefer Precision-Recall analysis for highly imbalanced classification tasks.
- Compare Average Precision against the positive class prevalence baseline.
- Consider the cost of False Positives and False Negatives.
- Select classification thresholds using validation data.
- Compare Precision, Recall, F1-Score, and Average Precision together.
- Avoid data leakage during model evaluation.

---

## 📚 Conclusion

The Precision-Recall Curve is an important evaluation technique used to analyze classification models, particularly when dealing with imbalanced datasets.

It represents the relationship between Precision and Recall at different classification thresholds.

Average Precision summarizes the Precision-Recall Curve into a single numerical value and helps compare different classification models.

Therefore, Precision-Recall analysis is especially valuable in applications such as medical diagnosis, fraud detection, spam filtering, and cybersecurity, where identifying positive cases accurately is important.

---

## 🔗 References

- [Scikit-Learn Precision-Recall Curve Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)
- [Scikit-Learn Average Precision Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [Scikit-Learn Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Scikit-Learn PrecisionRecallDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html)
- [Google Machine Learning - Classification](https://developers.google.com/machine-learning/crash-course/classification)
