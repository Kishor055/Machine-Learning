A comprehensive beginner-to-advanced tutorial on evaluating machine learning
models with Scikit-Learn metrics.

Topics covered
Why model evaluation matters
Regression metrics
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R-squared (R²)
Mean Absolute Percentage Error (MAPE)
Median Absolute Error
Max Error
Explained Variance
Classification metrics
Accuracy
Precision
Recall
F1-score
Confusion matrix
Classification report
Specificity
Balanced accuracy
Cohen's Kappa
Matthews Correlation Coefficient
ROC-AUC
Precision-Recall AUC
Log Loss
Multiclass metrics
Micro / macro / weighted averaging
Threshold-based evaluation
Cross-validation scoring
Comparing multiple models
Choosing appropriate metrics
Common metric mistakes
Complete evaluation workflows
Requirements

pip install numpy pandas scikit-learn matplotlib

Run

python metrics.py

Author

Kishor Patil
"""

=============================================================================
1. IMPORTS
=============================================================================

from future import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import (
load_diabetes,
load_iris,
make_classification,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
accuracy_score,
balanced_accuracy_score,
classification_report,
cohen_kappa_score,
confusion_matrix,
explained_variance_score,
f1_score,
log_loss,
matthews_corrcoef,
max_error,
mean_absolute_error,
mean_absolute_percentage_error,
mean_squared_error,
median_absolute_error,
precision_recall_curve,
precision_recall_fscore_support,
precision_score,
r2_score,
recall_score,
roc_auc_score,
roc_curve,
)
from sklearn.model_selection import (
StratifiedKFold,
cross_val_score,
train_test_split,
)

=============================================================================
2. HELPER
=============================================================================

def section(title: str) -> None:
"""Print a formatted section heading."""
print("\n" + "=" * 80)
print(title)
print("=" * 80)

=============================================================================
3. REGRESSION DATA
=============================================================================

def get_regression_data():
"""Load and split the built-in Diabetes regression dataset."""

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

return train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)
=============================================================================
4. REGRESSION METRICS
=============================================================================

def demonstrate_regression_metrics() -> None:
"""Demonstrate the most commonly used regression metrics."""

section("4. Regression Metrics")

X_train, X_test, y_train, y_test = get_regression_data()

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

print(
    """

Interpretation:

MAE
Average absolute prediction error.

MSE
Average squared prediction error.
Large errors receive greater penalties.

RMSE
Square root of MSE.
Expressed in the same units as the target.

R²
Measures how much target variance is explained by the model.
"""
)

=============================================================================
5. MAE
=============================================================================

def demonstrate_mae() -> None:
"""Explain Mean Absolute Error."""

section("5. Mean Absolute Error (MAE)")

y_true = np.array([100, 200, 300, 400])
y_pred = np.array([110, 190, 280, 420])

mae = mean_absolute_error(y_true, y_pred)

print("Actual:    ", y_true)
print("Predicted: ", y_pred)
print(f"MAE: {mae:.2f}")

print("\nFormula:")
print("MAE = mean(|y_true - y_pred|)")

print("\nLower MAE generally means smaller average absolute errors.")
=============================================================================
6. MSE
=============================================================================

def demonstrate_mse() -> None:
"""Explain Mean Squared Error."""

section("6. Mean Squared Error (MSE)")

y_true = np.array([100, 200, 300, 400])
y_pred = np.array([110, 190, 280, 420])

mse = mean_squared_error(y_true, y_pred)

print(f"MSE: {mse:.2f}")

print("\nFormula:")
print("MSE = mean((y_true - y_pred)²)")

print(
    "\nBecause errors are squared, large errors have a stronger "
    "effect on the metric."
)
=============================================================================
7. RMSE
=============================================================================

def demonstrate_rmse() -> None:
"""Explain Root Mean Squared Error."""

section("7. Root Mean Squared Error (RMSE)")

y_true = np.array([100, 200, 300, 400])
y_pred = np.array([110, 190, 280, 420])

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

print("\nFormula:")
print("RMSE = sqrt(MSE)")
=============================================================================
8. R-SQUARED
=============================================================================

def demonstrate_r2() -> None:
"""Explain R-squared."""

section("8. R-Squared (R²)")

y_true = np.array([10, 20, 30, 40, 50])
y_pred = np.array([12, 19, 31, 38, 48])

score = r2_score(y_true, y_pred)

print(f"R²: {score:.4f}")

print(
    """

R² compares model performance against a baseline that predicts
the mean target value.

A value closer to 1 generally indicates stronger explanatory performance.

Important:
R² is not simply "accuracy for regression", and a high R² does not
automatically imply that a model is useful in every application.
"""
)

=============================================================================
9. ADDITIONAL REGRESSION METRICS
=============================================================================

def demonstrate_additional_regression_metrics() -> None:
"""Demonstrate additional regression metrics."""

section("9. Additional Regression Metrics")

y_true = np.array([100, 200, 300, 400, 500], dtype=float)
y_pred = np.array([105, 180, 310, 395, 530], dtype=float)

metrics = {
    "MAE": mean_absolute_error(y_true, y_pred),
    "MSE": mean_squared_error(y_true, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
    "Median Absolute Error": median_absolute_error(y_true, y_pred),
    "Max Error": max_error(y_true, y_pred),
    "Explained Variance": explained_variance_score(y_true, y_pred),
    "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    "R²": r2_score(y_true, y_pred),
}

for name, value in metrics.items():
    print(f"{name:<25}: {value:.6f}")

print(
    """

Note:
MAPE is expressed as a ratio by Scikit-Learn. Multiply by 100 when
displaying it as a percentage.

MAPE can be problematic when true target values are zero or very close
to zero.
"""
)

=============================================================================
10. CLASSIFICATION DATA
=============================================================================

def get_binary_classification_data():
"""Create a reproducible binary classification dataset."""

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    class_sep=1.2,
    random_state=42,
)

return train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
=============================================================================
11. CLASSIFICATION MODEL
=============================================================================

def train_classification_model():
"""Train a logistic regression classifier."""

X_train, X_test, y_train, y_test = get_binary_classification_data()

model = LogisticRegression(
    max_iter=2000,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

return model, X_test, y_test, y_pred, y_proba
=============================================================================
12. ACCURACY
=============================================================================

def demonstrate_accuracy() -> None:
"""Demonstrate classification accuracy."""

section("12. Accuracy")

y_true = np.array([0, 1, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1])

accuracy = accuracy_score(y_true, y_pred)

print("Actual:    ", y_true)
print("Predicted: ", y_pred)
print(f"Accuracy: {accuracy:.4f}")

print(
    "\nAccuracy = correctly classified samples / total samples"
)

print(
    "\nWarning: Accuracy can be misleading when classes are highly "
    "imbalanced."
)
=============================================================================
13. PRECISION
=============================================================================

def demonstrate_precision() -> None:
"""Demonstrate precision."""

section("13. Precision")

y_true = np.array([0, 1, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1])

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0,
)

print(f"Precision: {precision:.4f}")

print(
    """

Precision answers:

"Of the samples predicted as positive, how many were actually positive?"

Precision = TP / (TP + FP)

High precision is important when false positives are particularly costly.
"""
)

=============================================================================
14. RECALL
=============================================================================

def demonstrate_recall() -> None:
"""Demonstrate recall."""

section("14. Recall")

y_true = np.array([0, 1, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1])

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0,
)

print(f"Recall: {recall:.4f}")

print(
    """

Recall answers:

"Of the actual positive samples, how many did the model identify?"

Recall = TP / (TP + FN)

High recall is important when false negatives are particularly costly.
"""
)

=============================================================================
15. F1 SCORE
=============================================================================

def demonstrate_f1() -> None:
"""Demonstrate F1-score."""

section("15. F1-Score")

y_true = np.array([0, 1, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1])

score = f1_score(
    y_true,
    y_pred,
    zero_division=0,
)

print(f"F1-score: {score:.4f}")

print(
    """

F1-score is the harmonic mean of precision and recall.

F1 = 2 × (Precision × Recall) / (Precision + Recall)

F1 can be useful when both false positives and false negatives matter.
"""
)

=============================================================================
16. CONFUSION MATRIX
=============================================================================

def demonstrate_confusion_matrix() -> None:
"""Demonstrate a binary confusion matrix."""

section("16. Confusion Matrix")

y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1])

cm = confusion_matrix(y_true, y_pred)

print("Confusion matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nComponents:")
print(f"True Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")

print(
    """

For binary classification:

            Predicted
            0       1

Actual 0 TN FP
1 FN TP
"""
)

=============================================================================
17. CONFUSION MATRIX PLOT
=============================================================================

def plot_confusion_matrix() -> None:
"""Visualize a confusion matrix."""

section("17. Confusion Matrix Visualization")

_, _, y_test, y_pred, _ = train_classification_model()

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))

image = ax.imshow(cm)

ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

for row in range(cm.shape[0]):
    for col in range(cm.shape[1]):
        ax.text(
            col,
            row,
            str(cm[row, col]),
            ha="center",
            va="center",
        )

fig.colorbar(image, ax=ax)
plt.tight_layout()
plt.show()
=============================================================================
18. CLASSIFICATION REPORT
=============================================================================

def demonstrate_classification_report() -> None:
"""Generate a complete classification report."""

section("18. Classification Report")

_, _, y_test, y_pred, _ = train_classification_model()

report = classification_report(
    y_test,
    y_pred,
    digits=4,
    zero_division=0,
)

print(report)

print(
    """

The classification report commonly includes:

precision
recall
f1-score
support

Support represents the number of true samples belonging to each class.
"""
)

=============================================================================
19. SPECIFICITY
=============================================================================

def demonstrate_specificity() -> None:
"""Calculate specificity for binary classification."""

section("19. Specificity")

y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])

cm = confusion_matrix(y_true, y_pred)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)

print(f"Specificity: {specificity:.4f}")

print("\nFormula:")
print("Specificity = TN / (TN + FP)")
=============================================================================
20. BALANCED ACCURACY
=============================================================================

def demonstrate_balanced_accuracy() -> None:
"""Demonstrate balanced accuracy."""

section("20. Balanced Accuracy")

y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
y_pred = np.array([0, 0, 0, 0, 0, 0, 1, 1])

score = balanced_accuracy_score(
    y_true,
    y_pred,
)

print(f"Balanced Accuracy: {score:.4f}")

print(
    "\nBalanced accuracy averages recall across classes and can be "
    "more informative than ordinary accuracy for imbalanced data."
)
=============================================================================
21. COHEN'S KAPPA
=============================================================================

def demonstrate_cohen_kappa() -> None:
"""Demonstrate Cohen's Kappa."""

section("21. Cohen's Kappa")

y_true = np.array([0, 1, 1, 0, 1, 0])
y_pred = np.array([0, 1, 0, 0, 1, 1])

score = cohen_kappa_score(
    y_true,
    y_pred,
)

print(f"Cohen's Kappa: {score:.4f}")

print(
    "\nKappa measures agreement while accounting for agreement "
    "that could occur by chance."
)
=============================================================================
22. MATTHEWS CORRELATION COEFFICIENT
=============================================================================

def demonstrate_matthews_corrcoef() -> None:
"""Demonstrate Matthews Correlation Coefficient."""

section("22. Matthews Correlation Coefficient")

y_true = np.array([0, 0, 0, 1, 1, 1])
y_pred = np.array([0, 0, 1, 1, 1, 0])

score = matthews_corrcoef(
    y_true,
    y_pred,
)

print(f"MCC: {score:.4f}")

print(
    """

MCC summarizes binary classification quality using all four
confusion-matrix components.

It can be useful for imbalanced classification problems.
"""
)

=============================================================================
23. PRECISION / RECALL / F1 TOGETHER
=============================================================================

def demonstrate_precision_recall_f1() -> None:
"""Calculate precision, recall, and F1 together."""

section("23. Precision, Recall and F1 Together")

_, _, y_test, y_pred, _ = train_classification_model()

precision, recall, f1, support = precision_recall_fscore_support(
    y_test,
    y_pred,
    average="binary",
    zero_division=0,
)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"Support:   {support}")
=============================================================================
24. ROC-AUC
=============================================================================

def demonstrate_roc_auc() -> None:
"""Demonstrate ROC-AUC using predicted probabilities."""

section("24. ROC-AUC")

_, _, y_test, _, y_proba = train_classification_model()

auc = roc_auc_score(
    y_test,
    y_proba,
)

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_proba,
)

print(f"ROC-AUC: {auc:.4f}")
print(f"Number of thresholds: {len(thresholds)}")

print(
    """

ROC-AUC evaluates how well the model ranks positive examples
above negative examples across classification thresholds.

It requires a continuous score such as:

predict_proba()
decision_function()

"""
)

=============================================================================
25. ROC CURVE
=============================================================================

def plot_roc_curve() -> None:
"""Plot the ROC curve."""

section("25. ROC Curve")

_, _, y_test, _, y_proba = train_classification_model()

fpr, tpr, _ = roc_curve(
    y_test,
    y_proba,
)

auc = roc_auc_score(
    y_test,
    y_proba,
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {auc:.3f}",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier",
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
=============================================================================
26. PRECISION-RECALL CURVE
=============================================================================

def demonstrate_precision_recall_curve() -> None:
"""Demonstrate the precision-recall curve and average precision."""

section("26. Precision-Recall Curve")

_, _, y_test, _, y_proba = train_classification_model()

precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_proba,
)

from sklearn.metrics import average_precision_score

average_precision = average_precision_score(
    y_test,
    y_proba,
)

print(f"Average Precision: {average_precision:.4f}")
print(f"Number of thresholds: {len(thresholds)}")

plt.figure(figsize=(7, 5))

plt.plot(
    recall,
    precision,
    label=f"Average Precision = {average_precision:.3f}",
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
=============================================================================
27. LOG LOSS
=============================================================================

def demonstrate_log_loss() -> None:
"""Demonstrate logarithmic loss."""

section("27. Log Loss")

y_true = np.array([0, 1, 1, 0])

y_probability = np.array(
    [
        [0.90, 0.10],
        [0.20, 0.80],
        [0.10, 0.90],
        [0.75, 0.25],
    ]
)

loss = log_loss(
    y_true,
    y_probability,
)

print(f"Log Loss: {loss:.4f}")

print(
    """

Log loss evaluates predicted probabilities, not only final class labels.

It strongly penalizes confident predictions that are incorrect.

Lower log loss is better.
"""
)

=============================================================================
28. MULTICLASS CLASSIFICATION
=============================================================================

def demonstrate_multiclass_metrics() -> None:
"""Demonstrate metrics for multiclass classification."""

section("28. Multiclass Classification Metrics")

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.20,
    random_state=42,
    stratify=iris.target,
)

model = LogisticRegression(
    max_iter=2000,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

for average in ("micro", "macro", "weighted"):
    score = f1_score(
        y_test,
        y_pred,
        average=average,
    )

    print(f"F1 ({average:<8}): {score:.4f}")

print(
    """

Micro:
Aggregate contributions from all classes before calculating the metric.

Macro:
Calculate the metric independently for each class and average equally.

Weighted:
Calculate the metric per class and weight by class support.
"""
)

=============================================================================
29. CROSS-VALIDATION METRICS
=============================================================================

def demonstrate_cross_validation_metrics() -> None:
"""Evaluate a classifier using stratified cross-validation."""

section("29. Cross-Validation Metrics")

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_classes=2,
    random_state=42,
)

model = LogisticRegression(
    max_iter=2000,
    random_state=42,
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

accuracy_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy",
)

f1_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="f1",
)

roc_auc_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="roc_auc",
)

print("Accuracy scores:", np.round(accuracy_scores, 4))
print("F1 scores:      ", np.round(f1_scores, 4))
print("ROC-AUC scores: ", np.round(roc_auc_scores, 4))

print("\nMean scores:")
print(f"Accuracy: {accuracy_scores.mean():.4f}")
print(f"F1:       {f1_scores.mean():.4f}")
print(f"ROC-AUC:  {roc_auc_scores.mean():.4f}")
=============================================================================
30. MODEL COMPARISON
=============================================================================

def compare_classification_models() -> None:
"""Compare several classifiers using the same cross-validation strategy."""

section("30. Comparing Classification Models")

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_classes=2,
    random_state=42,
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        random_state=42,
    ),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

results = []

for name, model in models.items():
    accuracy = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    f1 = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="f1",
    )

    results.append(
        {
            "Model": name,
            "Accuracy Mean": accuracy.mean(),
            "Accuracy Std": accuracy.std(),
            "F1 Mean": f1.mean(),
            "F1 Std": f1.std(),
        }
    )

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

print(
    "\nThe table demonstrates metric comparison; it should not be "
    "interpreted as a universal ranking of models."
)
=============================================================================
31. METRIC SELECTION
=============================================================================

def print_metric_selection_guide() -> None:
"""Print a practical guide for selecting evaluation metrics."""

section("31. Choosing Evaluation Metrics")

guide = [
    (
        "Regression",
        "MAE, RMSE, R²",
        "Choose based on error costs and interpretability.",
    ),
    (
        "Balanced classification",
        "Accuracy, F1, ROC-AUC",
        "Use multiple metrics when appropriate.",
    ),
    (
        "Imbalanced classification",
        "Precision, Recall, F1, PR-AUC",
        "Accuracy alone may hide poor minority-class performance.",
    ),
    (
        "False positives costly",
        "Precision",
        "Focus on the reliability of positive predictions.",
    ),
    (
        "False negatives costly",
        "Recall",
        "Focus on identifying actual positive cases.",
    ),
    (
        "Probability quality",
        "Log Loss",
        "Evaluate predicted probabilities rather than labels.",
    ),
    (
        "Ranking quality",
        "ROC-AUC / PR-AUC",
        "Useful when evaluating scores across thresholds.",
    ),
]

df = pd.DataFrame(
    guide,
    columns=[
        "Problem",
        "Possible Metrics",
        "Consideration",
    ],
)

print(df.to_string(index=False))
=============================================================================
32. THRESHOLD EFFECT
=============================================================================

def demonstrate_threshold_effect() -> None:
"""
Demonstrate how changing a classification threshold affects metrics.

The model's predicted probabilities remain unchanged; only the rule
converting probabilities into class labels changes.
"""

section("32. Classification Threshold")

_, _, y_test, _, y_proba = train_classification_model()

thresholds = [0.30, 0.50, 0.70]

rows = []

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)

    rows.append(
        {
            "Threshold": threshold,
            "Precision": precision_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            "Recall": recall_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
            "F1": f1_score(
                y_test,
                y_pred,
                zero_division=0,
            ),
        }
    )

results = pd.DataFrame(rows)

print(results.to_string(index=False))

print(
    """

Changing the threshold changes the trade-off between precision and recall.

The appropriate threshold depends on the application's error costs,
constraints, and validation strategy.
"""
)

=============================================================================
33. COMPLETE REGRESSION EVALUATION
=============================================================================

def complete_regression_evaluation() -> None:
"""Run a complete regression evaluation example."""

section("33. Complete Regression Evaluation")

X_train, X_test, y_train, y_test = get_regression_data()

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

metrics = {
    "MAE": mean_absolute_error(y_test, y_pred),
    "MSE": mean_squared_error(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "R²": r2_score(y_test, y_pred),
    "Explained Variance": explained_variance_score(
        y_test,
        y_pred,
    ),
    "Median Absolute Error": median_absolute_error(
        y_test,
        y_pred,
    ),
    "Max Error": max_error(
        y_test,
        y_pred,
    ),
}

results = pd.DataFrame(
    metrics.items(),
    columns=["Metric", "Value"],
)

print(results.to_string(index=False))
=============================================================================
34. COMPLETE CLASSIFICATION EVALUATION
=============================================================================

def complete_classification_evaluation() -> None:
"""Run a complete binary classification evaluation."""

section("34. Complete Classification Evaluation")

model, X_test, y_test, y_pred, y_proba = (
    train_classification_model()
)

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Balanced Accuracy": balanced_accuracy_score(
        y_test,
        y_pred,
    ),
    "Precision": precision_score(
        y_test,
        y_pred,
        zero_division=0,
    ),
    "Recall": recall_score(
        y_test,
        y_pred,
        zero_division=0,
    ),
    "F1": f1_score(
        y_test,
        y_pred,
        zero_division=0,
    ),
    "ROC-AUC": roc_auc_score(
        y_test,
        y_proba,
    ),
    "Log Loss": log_loss(
        y_test,
        model.predict_proba(X_test),
    ),
    "MCC": matthews_corrcoef(
        y_test,
        y_pred,
    ),
    "Cohen's Kappa": cohen_kappa_score(
        y_test,
        y_pred,
    ),
}

results = pd.DataFrame(
    metrics.items(),
    columns=["Metric", "Value"],
)

print(results.to_string(index=False))
=============================================================================
35. COMMON MISTAKES
=============================================================================

def print_common_mistakes() -> None:
"""Print common mistakes when using ML metrics."""

section("35. Common Metric Mistakes")

mistakes = [
    "Using accuracy as the only metric for imbalanced classification.",
    "Comparing models using different test sets.",
    "Selecting a model based only on training metrics.",
    "Using test data repeatedly for model tuning.",
    "Ignoring the difference between labels and probabilities.",
    "Using ROC-AUC without understanding class imbalance.",
    "Ignoring precision when false positives are expensive.",
    "Ignoring recall when false negatives are expensive.",
    "Reporting MAPE when target values can be zero.",
    "Interpreting R² as a percentage of prediction accuracy.",
    "Choosing a threshold on the test set.",
    "Using cross-validation incorrectly with time-dependent data.",
    "Reporting only a single metric when several are relevant.",
]

for number, mistake in enumerate(mistakes, start=1):
    print(f"{number:02d}. {mistake}")
=============================================================================
36. QUICK REFERENCE
=============================================================================

def print_quick_reference() -> None:
"""Print a quick reference table of common Scikit-Learn metrics."""

section("36. Metrics Quick Reference")

metrics = [
    ("mean_absolute_error", "Regression", "Lower is better"),
    ("mean_squared_error", "Regression", "Lower is better"),
    ("RMSE", "Regression", "Lower is better"),
    ("r2_score", "Regression", "Higher is generally better"),
    ("accuracy_score", "Classification", "Higher is better"),
    ("precision_score", "Classification", "Higher is better"),
    ("recall_score", "Classification", "Higher is better"),
    ("f1_score", "Classification", "Higher is better"),
    ("confusion_matrix", "Classification", "Error breakdown"),
    ("roc_auc_score", "Classification", "Higher is generally better"),
    ("log_loss", "Classification", "Lower is better"),
    ("balanced_accuracy_score", "Classification", "Higher is better"),
    ("matthews_corrcoef", "Classification", "Higher is better"),
]

df = pd.DataFrame(
    metrics,
    columns=["Metric", "Task", "Typical Direction"],
)

print(df.to_string(index=False))
=============================================================================
37. MAIN
=============================================================================

def main() -> None:
"""Run the complete metrics tutorial."""

print("=" * 80)
print("SCIKIT-LEARN METRICS TUTORIAL")
print("=" * 80)

# Regression metrics.
demonstrate_regression_metrics()
demonstrate_mae()
demonstrate_mse()
demonstrate_rmse()
demonstrate_r2()
demonstrate_additional_regression_metrics()

# Classification metrics.
demonstrate_accuracy()
demonstrate_precision()
demonstrate_recall()
demonstrate_f1()
demonstrate_confusion_matrix()
demonstrate_specificity()
demonstrate_balanced_accuracy()
demonstrate_cohen_kappa()
demonstrate_matthews_corrcoef()
demonstrate_precision_recall_f1()

# Probability-based metrics.
demonstrate_roc_auc()
demonstrate_precision_recall_curve()
demonstrate_log_loss()

# Multiclass and validation.
demonstrate_multiclass_metrics()
demonstrate_cross_validation_metrics()
compare_classification_models()

# Practical evaluation.
print_metric_selection_guide()
demonstrate_threshold_effect()
complete_regression_evaluation()
complete_classification_evaluation()

# Documentation helpers.
print_common_mistakes()
print_quick_reference()

print("\n" + "=" * 80)
print("METRICS TUTORIAL COMPLETED")
print("=" * 80)

if name == "main":
main()
