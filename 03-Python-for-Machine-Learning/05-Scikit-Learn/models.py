A comprehensive beginner-to-advanced tutorial covering commonly used
machine learning models available in Scikit-Learn.

Topics covered
Machine learning model workflow
Regression vs classification vs clustering
Linear Regression
Ridge Regression
Lasso Regression
Elastic Net
Logistic Regression
K-Nearest Neighbors
Decision Trees
Random Forest
Gradient Boosting
Support Vector Machines
Naive Bayes
Gaussian Process Regression
K-Means Clustering
DBSCAN
Agglomerative Clustering
Principal Component Analysis
Model prediction
Model evaluation
Cross-validation
Model comparison
Feature importance
Overfitting and underfitting
Reproducibility
Common model-selection mistakes
End-to-end model workflows
Requirements

pip install numpy pandas scikit-learn matplotlib

Run

python models.py

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

from sklearn.cluster import (
AgglomerativeClustering,
DBSCAN,
KMeans,
)
from sklearn.datasets import (
load_diabetes,
load_iris,
make_blobs,
make_classification,
)
from sklearn.ensemble import (
GradientBoostingClassifier,
GradientBoostingRegressor,
RandomForestClassifier,
RandomForestRegressor,
)
from sklearn.linear_model import (
ElasticNet,
Lasso,
LinearRegression,
LogisticRegression,
Ridge,
)
from sklearn.metrics import (
accuracy_score,
classification_report,
confusion_matrix,
f1_score,
mean_absolute_error,
mean_squared_error,
r2_score,
silhouette_score,
)
from sklearn.model_selection import (
StratifiedKFold,
cross_val_score,
train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import (
SVC,
SVR,
)
from sklearn.tree import (
DecisionTreeClassifier,
DecisionTreeRegressor,
plot_tree,
)

=============================================================================
2. HELPER FUNCTIONS
=============================================================================

def section(title: str) -> None:
"""Print a formatted section heading."""
print("\n" + "=" * 80)
print(title)
print("=" * 80)

def regression_rmse(
y_true: np.ndarray,
y_pred: np.ndarray,
) -> float:
"""Calculate Root Mean Squared Error."""
return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def print_regression_metrics(
y_true: np.ndarray,
y_pred: np.ndarray,
) -> None:
"""Print common regression evaluation metrics."""

print(f"MAE : {mean_absolute_error(y_true, y_pred):.4f}")
print(f"RMSE: {regression_rmse(y_true, y_pred):.4f}")
print(f"R²  : {r2_score(y_true, y_pred):.4f}")

def print_classification_metrics(
y_true: np.ndarray,
y_pred: np.ndarray,
) -> None:
"""Print common classification evaluation metrics."""

print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"F1-score: {f1_score(y_true, y_pred):.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )
)
=============================================================================
3. MACHINE LEARNING WORKFLOW
=============================================================================

def explain_model_workflow() -> None:
"""Explain the standard machine learning workflow."""

section("3. Machine Learning Model Workflow")

print(
    """

Typical workflow:

1. Collect data
        |
        v
2. Understand and clean data
        |
        v
3. Split into training and testing data
        |
        v
4. Preprocess features
        |
        v
5. Select a model
        |
        v
6. Train with fit()
        |
        v
7. Predict with predict()
        |
        v
8. Evaluate using appropriate metrics
        |
        v
9. Tune hyperparameters
        |
        v
10. Validate final model
        |
        v
11. Save and deploy

"""
)

print(
    "\nImportant: preprocessing should be fitted only on training data "
    "to avoid data leakage."
)
=============================================================================
4. REGRESSION DATASET
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
5. LINEAR REGRESSION
=============================================================================

def demonstrate_linear_regression() -> None:
"""Demonstrate ordinary least-squares linear regression."""

section("5. Linear Regression")

X_train, X_test, y_train, y_test = get_regression_data()

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model:")
print(model)

print("\nIntercept:")
print(model.intercept_)

print("\nCoefficients:")
print(model.coef_)

print("\nEvaluation:")
print_regression_metrics(y_test, y_pred)

print(
    """

Linear Regression models a continuous target as a linear combination
of input features.

Conceptually:

y = b0 + b1*x1 + b2*x2 + ... + bn*xn

"""
)

=============================================================================
6. RIDGE REGRESSION
=============================================================================

def demonstrate_ridge() -> None:
"""Demonstrate Ridge regression."""

section("6. Ridge Regression")

X_train, X_test, y_train, y_test = get_regression_data()

model = Ridge(alpha=1.0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Alpha: {model.alpha}")
print_regression_metrics(y_test, y_pred)

print(
    "\nRidge adds L2 regularization, which discourages excessively "
    "large coefficients."
)
=============================================================================
7. LASSO
=============================================================================

def demonstrate_lasso() -> None:
"""Demonstrate Lasso regression."""

section("7. Lasso Regression")

X_train, X_test, y_train, y_test = get_regression_data()

model = Lasso(
    alpha=0.01,
    max_iter=10000,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Alpha: {model.alpha}")
print_regression_metrics(y_test, y_pred)

zero_coefficients = np.sum(model.coef_ == 0)

print(f"\nZero coefficients: {zero_coefficients}")

print(
    "\nLasso uses L1 regularization and can drive some coefficients "
    "exactly to zero."
)
=============================================================================
8. ELASTIC NET
=============================================================================

def demonstrate_elastic_net() -> None:
"""Demonstrate Elastic Net regression."""

section("8. Elastic Net")

X_train, X_test, y_train, y_test = get_regression_data()

model = ElasticNet(
    alpha=0.01,
    l1_ratio=0.5,
    max_iter=10000,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Alpha:    {model.alpha}")
print(f"L1 ratio: {model.l1_ratio}")

print_regression_metrics(y_test, y_pred)

print(
    "\nElastic Net combines L1 and L2 regularization."
)
=============================================================================
9. CLASSIFICATION DATASET
=============================================================================

def get_classification_data():
"""Generate a reproducible binary classification dataset."""

X, y = make_classification(
    n_samples=1200,
    n_features=12,
    n_informative=7,
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
10. LOGISTIC REGRESSION
=============================================================================

def demonstrate_logistic_regression() -> None:
"""Demonstrate logistic regression classification."""

section("10. Logistic Regression")

X_train, X_test, y_train, y_test = get_classification_data()

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    max_iter=2000,
    random_state=42,
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print_classification_metrics(y_test, y_pred)

print(
    """

Logistic Regression predicts class probabilities using a logistic
function and is widely used as a baseline classification model.
"""
)

=============================================================================
11. K-NEAREST NEIGHBORS
=============================================================================

def demonstrate_knn() -> None:
"""Demonstrate K-Nearest Neighbors classification."""

section("11. K-Nearest Neighbors (KNN)")

X_train, X_test, y_train, y_test = get_classification_data()

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(
    n_neighbors=5,
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(f"Neighbors: {model.n_neighbors}")
print_classification_metrics(y_test, y_pred)

print(
    """

KNN predicts a sample based on nearby training samples.

Feature scaling is especially important because distance calculations
are central to KNN.
"""
)

=============================================================================
12. DECISION TREE CLASSIFIER
=============================================================================

def demonstrate_decision_tree_classifier() -> None:
"""Demonstrate a decision tree classifier."""

section("12. Decision Tree Classifier")

X_train, X_test, y_train, y_test = get_classification_data()

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Max depth: {model.max_depth}")
print_classification_metrics(y_test, y_pred)

print(
    "\nDecision trees learn a sequence of feature-based decision rules."
)
=============================================================================
13. RANDOM FOREST CLASSIFIER
=============================================================================

def demonstrate_random_forest_classifier() -> None:
"""Demonstrate Random Forest classification."""

section("13. Random Forest Classifier")

X_train, X_test, y_train, y_test = get_classification_data()

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Trees: {model.n_estimators}")
print_classification_metrics(y_test, y_pred)

print(
    "\nRandom Forest combines many decision trees to form an ensemble."
)
=============================================================================
14. GRADIENT BOOSTING CLASSIFIER
=============================================================================

def demonstrate_gradient_boosting_classifier() -> None:
"""Demonstrate Gradient Boosting classification."""

section("14. Gradient Boosting Classifier")

X_train, X_test, y_train, y_test = get_classification_data()

model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Trees:         {model.n_estimators}")
print(f"Learning rate: {model.learning_rate}")
print_classification_metrics(y_test, y_pred)

print(
    """

Gradient Boosting builds models sequentially, with later trees
attempting to improve the errors of earlier trees.
"""
)

=============================================================================
15. SUPPORT VECTOR MACHINE
=============================================================================

def demonstrate_svm_classifier() -> None:
"""Demonstrate Support Vector Classification."""

section("15. Support Vector Machine Classifier")

X_train, X_test, y_train, y_test = get_classification_data()

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(
    kernel="rbf",
    C=1.0,
    probability=True,
    random_state=42,
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(f"Kernel: {model.kernel}")
print(f"C:      {model.C}")

print_classification_metrics(y_test, y_pred)

print(
    "\nSVM models seek decision boundaries with useful margins between "
    "classes. Feature scaling is generally important."
)
=============================================================================
16. NAIVE BAYES
=============================================================================

def demonstrate_naive_bayes() -> None:
"""Demonstrate Gaussian Naive Bayes."""

section("16. Gaussian Naive Bayes")

X_train, X_test, y_train, y_test = get_classification_data()

model = GaussianNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print_classification_metrics(y_test, y_pred)

print(
    """

Naive Bayes uses Bayes' theorem with simplifying assumptions about
the relationship between features.

GaussianNB assumes feature likelihoods follow Gaussian distributions.
"""
)

=============================================================================
17. DECISION TREE REGRESSOR
=============================================================================

def demonstrate_decision_tree_regressor() -> None:
"""Demonstrate a decision tree regression model."""

section("17. Decision Tree Regressor")

X_train, X_test, y_train, y_test = get_regression_data()

model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Max depth: {model.max_depth}")
print_regression_metrics(y_test, y_pred)
=============================================================================
18. RANDOM FOREST REGRESSOR
=============================================================================

def demonstrate_random_forest_regressor() -> None:
"""Demonstrate Random Forest regression."""

section("18. Random Forest Regressor")

X_train, X_test, y_train, y_test = get_regression_data()

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Trees: {model.n_estimators}")
print_regression_metrics(y_test, y_pred)
=============================================================================
19. GRADIENT BOOSTING REGRESSOR
=============================================================================

def demonstrate_gradient_boosting_regressor() -> None:
"""Demonstrate Gradient Boosting regression."""

section("19. Gradient Boosting Regressor")

X_train, X_test, y_train, y_test = get_regression_data()

model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print_regression_metrics(y_test, y_pred)
=============================================================================
20. SUPPORT VECTOR REGRESSION
=============================================================================

def demonstrate_svr() -> None:
"""Demonstrate Support Vector Regression."""

section("20. Support Vector Regression")

X_train, X_test, y_train, y_test = get_regression_data()

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVR(
    kernel="rbf",
    C=10.0,
    epsilon=0.1,
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(f"Kernel:   {model.kernel}")
print(f"C:        {model.C}")
print(f"Epsilon:  {model.epsilon}")

print_regression_metrics(y_test, y_pred)
=============================================================================
21. K-MEANS
=============================================================================

def demonstrate_kmeans() -> None:
"""Demonstrate K-Means clustering."""

section("21. K-Means Clustering")

X, _ = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=1.2,
    random_state=42,
)

model = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42,
)

labels = model.fit_predict(X)

score = silhouette_score(
    X,
    labels,
)

print(f"Clusters: {model.n_clusters}")
print(f"Inertia:  {model.inertia_:.4f}")
print(f"Silhouette score: {score:.4f}")

print(
    """

K-Means attempts to partition observations into K clusters.

It is an unsupervised algorithm, so target labels are not required
during training.
"""
)

=============================================================================
22. DBSCAN
=============================================================================

def demonstrate_dbscan() -> None:
"""Demonstrate DBSCAN clustering."""

section("22. DBSCAN Clustering")

X, _ = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=0.8,
    random_state=42,
)

model = DBSCAN(
    eps=0.7,
    min_samples=5,
)

labels = model.fit_predict(X)

unique_labels = np.unique(labels)

print("Detected labels:")
print(unique_labels)

noise_count = np.sum(labels == -1)

print(f"\nNoise points: {noise_count}")

non_noise_mask = labels != -1

if len(np.unique(labels[non_noise_mask])) >= 2:
    score = silhouette_score(
        X[non_noise_mask],
        labels[non_noise_mask],
    )

    print(f"Silhouette score excluding noise: {score:.4f}")
=============================================================================
23. AGGLOMERATIVE CLUSTERING
=============================================================================

def demonstrate_agglomerative_clustering() -> None:
"""Demonstrate hierarchical agglomerative clustering."""

section("23. Agglomerative Clustering")

X, _ = make_blobs(
    n_samples=400,
    centers=3,
    cluster_std=1.1,
    random_state=42,
)

model = AgglomerativeClustering(
    n_clusters=3,
)

labels = model.fit_predict(X)

score = silhouette_score(
    X,
    labels,
)

print(f"Clusters: {model.n_clusters}")
print(f"Silhouette score: {score:.4f}")
=============================================================================
24. FEATURE IMPORTANCE
=============================================================================

def demonstrate_feature_importance() -> None:
"""Demonstrate tree-based feature importance."""

section("24. Feature Importance")

X_train, X_test, y_train, y_test = get_classification_data()

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
)

model.fit(X_train, y_train)

importance = model.feature_importances_

importance_df = pd.DataFrame(
    {
        "Feature": [
            f"feature_{i}"
            for i in range(X_train.shape[1])
        ],
        "Importance": importance,
    }
).sort_values(
    "Importance",
    ascending=False,
)

print(importance_df.to_string(index=False))

print(
    """

Feature importance indicates how much a tree-based model used features
according to its internal importance calculation.

It should not automatically be interpreted as causal importance.
"""
)

=============================================================================
25. DECISION TREE VISUALIZATION
=============================================================================

def visualize_decision_tree() -> None:
"""Visualize a small decision tree."""

section("25. Decision Tree Visualization")

X_train, X_test, y_train, y_test = get_classification_data()

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42,
)

model.fit(X_train, y_train)

plt.figure(figsize=(16, 8))

plot_tree(
    model,
    filled=False,
    feature_names=[
        f"feature_{i}"
        for i in range(X_train.shape[1])
    ],
    class_names=["class_0", "class_1"],
    rounded=True,
    fontsize=8,
)

plt.title("Decision Tree")
plt.tight_layout()
plt.show()
=============================================================================
26. MODEL COMPARISON - CLASSIFICATION
=============================================================================

def compare_classification_models() -> None:
"""Compare several classification models using cross-validation."""

section("26. Classification Model Comparison")

X, y = make_classification(
    n_samples=1200,
    n_features=12,
    n_informative=7,
    n_redundant=2,
    random_state=42,
)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42,
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=5,
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=150,
        random_state=42,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42,
    ),
    "Naive Bayes": GaussianNB(),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

results = []

for name, model in models.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy",
    )

    results.append(
        {
            "Model": name,
            "Mean Accuracy": scores.mean(),
            "Std": scores.std(),
        }
    )

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

print(
    "\nThese are cross-validation measurements on this particular "
    "synthetic dataset, not universal model rankings."
)
=============================================================================
27. MODEL COMPARISON - REGRESSION
=============================================================================

def compare_regression_models() -> None:
"""Compare several regression models using cross-validation."""

section("27. Regression Model Comparison")

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(
        alpha=0.01,
        max_iter=10000,
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        random_state=42,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42,
    ),
}

results = []

for name, model in models.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="r2",
    )

    results.append(
        {
            "Model": name,
            "Mean R²": scores.mean(),
            "Std": scores.std(),
        }
    )

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))
=============================================================================
28. OVERFITTING VS UNDERFITTING
=============================================================================

def demonstrate_overfitting_underfitting() -> None:
"""Demonstrate training/test performance for different tree depths."""

section("28. Overfitting and Underfitting")

X_train, X_test, y_train, y_test = get_classification_data()

depths = [1, 2, 3, 5, 10, None]

rows = []

for depth in depths:
    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42,
    )

    model.fit(X_train, y_train)

    train_accuracy = model.score(
        X_train,
        y_train,
    )

    test_accuracy = model.score(
        X_test,
        y_test,
    )

    rows.append(
        {
            "Max Depth": str(depth),
            "Train Accuracy": train_accuracy,
            "Test Accuracy": test_accuracy,
        }
    )

results = pd.DataFrame(rows)

print(results.to_string(index=False))

print(
    """

General pattern:

Underfitting:
Model is too simple to capture useful patterns.

Good fit:
Model generalizes reasonably well.

Overfitting:
Model fits training data extremely closely but generalizes poorly.

The exact interpretation depends on the dataset and evaluation design.
"""
)

=============================================================================
29. MODEL PARAMETERS VS HYPERPARAMETERS
=============================================================================

def explain_parameters_vs_hyperparameters() -> None:
"""Explain parameters learned during training vs hyperparameters."""

section("29. Parameters vs Hyperparameters")

print(
    """

Parameters:
Values learned by the model during fit().

Examples:
LinearRegression.coef_
LinearRegression.intercept_
RandomForestClassifier.feature_importances_

Hyperparameters:
Values configured before or during model selection.

Examples:
RandomForestClassifier.n_estimators
DecisionTreeClassifier.max_depth
KNeighborsClassifier.n_neighbors
LogisticRegression.C

Hyperparameters are commonly selected using validation or
cross-validation rather than the final test set.
"""
)

=============================================================================
30. MODEL PREDICTION API
=============================================================================

def demonstrate_model_api() -> None:
"""Demonstrate common Scikit-Learn estimator methods."""

section("30. Scikit-Learn Model API")

X_train, X_test, y_train, y_test = get_classification_data()

model = LogisticRegression(
    max_iter=2000,
    random_state=42,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("Common methods:")
print("fit()          -> train the model")
print("predict()      -> generate predictions")
print("predict_proba() -> generate class probabilities")
print("score()        -> model-specific default score")

print("\nFirst five predictions:")
print(predictions[:5])

print("\nFirst five probability rows:")
print(probabilities[:5])

print("\nModel score:")
print(model.score(X_test, y_test))
=============================================================================
31. RANDOM STATE
=============================================================================

def explain_reproducibility() -> None:
"""Explain reproducibility with random_state."""

section("31. Reproducibility")

print(
    """

For algorithms involving randomness, random_state can make experiments
reproducible.

Examples:

RandomForestClassifier(random_state=42)

train_test_split(..., random_state=42)

KMeans(..., random_state=42)

Reproducibility is important for:
- debugging
- tutorials
- experiments
- comparisons
- research
- production testing
"""
)

=============================================================================
32. MODEL SELECTION GUIDE
=============================================================================

def print_model_selection_guide() -> None:
"""Print a factual model-selection reference."""

section("32. Model Selection Reference")

rows = [
    (
        "Linear Regression",
        "Regression",
        "Simple linear relationships",
    ),
    (
        "Ridge",
        "Regression",
        "Linear regression with L2 regularization",
    ),
    (
        "Lasso",
        "Regression",
        "Linear regression with L1 regularization",
    ),
    (
        "Elastic Net",
        "Regression",
        "Combination of L1 and L2 regularization",
    ),
    (
        "Logistic Regression",
        "Classification",
        "Linear classification with probabilities",
    ),
    (
        "KNN",
        "Classification",
        "Distance-based classification",
    ),
    (
        "Decision Tree",
        "Classification / Regression",
        "Rule-based non-linear relationships",
    ),
    (
        "Random Forest",
        "Classification / Regression",
        "Tree ensemble",
    ),
    (
        "Gradient Boosting",
        "Classification / Regression",
        "Sequential tree ensemble",
    ),
    (
        "SVM / SVR",
        "Classification / Regression",
        "Margin-based learning",
    ),
    (
        "Naive Bayes",
        "Classification",
        "Probabilistic classification",
    ),
    (
        "K-Means",
        "Clustering",
        "Centroid-based clustering",
    ),
    (
        "DBSCAN",
        "Clustering",
        "Density-based clustering",
    ),
    (
        "Agglomerative",
        "Clustering",
        "Hierarchical clustering",
    ),
]

df = pd.DataFrame(
    rows,
    columns=[
        "Model",
        "Task",
        "General Description",
    ],
)

print(df.to_string(index=False))
=============================================================================
33. COMMON MISTAKES
=============================================================================

def print_common_mistakes() -> None:
"""Print common machine-learning model mistakes."""

section("33. Common Model Mistakes")

mistakes = [
    "Training and evaluating on exactly the same data.",
    "Using the test set repeatedly for hyperparameter tuning.",
    "Scaling the complete dataset before train/test splitting.",
    "Using KNN or SVM without considering feature scale.",
    "Allowing unrestricted decision trees to overfit.",
    "Assuming a more complex model is always better.",
    "Comparing models on different data splits.",
    "Using only one metric when the problem requires several.",
    "Ignoring class imbalance.",
    "Ignoring reproducibility.",
    "Interpreting feature importance as causation.",
    "Deploying a model without validating preprocessing consistency.",
    "Saving a model without documenting its dependencies and data assumptions.",
]

for number, mistake in enumerate(mistakes, start=1):
    print(f"{number:02d}. {mistake}")
=============================================================================
34. END-TO-END CLASSIFICATION
=============================================================================

def end_to_end_classification() -> None:
"""
Demonstrate a compact classification workflow.

This example intentionally keeps preprocessing simple. For production
workflows, Pipeline and ColumnTransformer are preferred.
"""

section("34. End-to-End Classification Workflow")

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = LogisticRegression(
    max_iter=2000,
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model trained successfully.")

print("\nEvaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
=============================================================================
35. END-TO-END REGRESSION
=============================================================================

def end_to_end_regression() -> None:
"""Demonstrate a compact regression workflow."""

section("35. End-to-End Regression Workflow")

diabetes = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data,
    diabetes.target,
    test_size=0.20,
    random_state=42,
)

model = Ridge(
    alpha=1.0,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model trained successfully.")

print("\nEvaluation:")
print_regression_metrics(y_test, y_pred)
=============================================================================
36. CLUSTERING WORKFLOW
=============================================================================

def end_to_end_clustering() -> None:
"""Demonstrate a compact unsupervised clustering workflow."""

section("36. End-to-End Clustering Workflow")

X, _ = make_blobs(
    n_samples=500,
    centers=4,
    cluster_std=1.1,
    random_state=42,
)

model = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42,
)

labels = model.fit_predict(X)

print("Clustering completed.")
print(f"Number of clusters: {len(np.unique(labels))}")
print(f"Inertia: {model.inertia_:.4f}")

print(
    "\nClustering evaluation can use internal measures such as "
    "silhouette score when their assumptions are appropriate."
)

print(
    f"Silhouette score: {silhouette_score(X, labels):.4f}"
)
=============================================================================
37. QUICK REFERENCE
=============================================================================

def print_quick_reference() -> None:
"""Print a quick Scikit-Learn model API reference."""

section("37. Model API Quick Reference")

reference = [
    ("fit(X, y)", "Train a supervised model"),
    ("fit(X)", "Fit an unsupervised estimator"),
    ("predict(X)", "Generate predictions"),
    ("predict_proba(X)", "Generate class probabilities"),
    ("transform(X)", "Transform features"),
    ("fit_transform(X)", "Fit and transform"),
    ("score(X, y)", "Estimator-specific default score"),
    ("get_params()", "Inspect hyperparameters"),
    ("set_params()", "Change hyperparameters"),
]

df = pd.DataFrame(
    reference,
    columns=[
        "Method",
        "Purpose",
    ],
)

print(df.to_string(index=False))
=============================================================================
38. MAIN
=============================================================================

def main() -> None:
"""Run the complete Scikit-Learn models tutorial."""

print("=" * 80)
print("SCIKIT-LEARN MODELS TUTORIAL")
print("=" * 80)

# Fundamentals.
explain_model_workflow()

# Regression.
demonstrate_linear_regression()
demonstrate_ridge()
demonstrate_lasso()
demonstrate_elastic_net()

# Classification.
demonstrate_logistic_regression()
demonstrate_knn()
demonstrate_decision_tree_classifier()
demonstrate_random_forest_classifier()
demonstrate_gradient_boosting_classifier()
demonstrate_svm_classifier()
demonstrate_naive_bayes()

# Regression tree/ensemble models.
demonstrate_decision_tree_regressor()
demonstrate_random_forest_regressor()
demonstrate_gradient_boosting_regressor()
demonstrate_svr()

# Unsupervised learning.
demonstrate_kmeans()
demonstrate_dbscan()
demonstrate_agglomerative_clustering()

# Model interpretation and evaluation.
demonstrate_feature_importance()

# Comparisons.
compare_classification_models()
compare_regression_models()

# Model behavior.
demonstrate_overfitting_underfitting()
explain_parameters_vs_hyperparameters()
demonstrate_model_api()
explain_reproducibility()

# Guides.
print_model_selection_guide()
print_common_mistakes()

# End-to-end examples.
end_to_end_classification()
end_to_end_regression()
end_to_end_clustering()

# Reference.
print_quick_reference()

print("\n" + "=" * 80)
print("MODELS TUTORIAL COMPLETED")
print("=" * 80)

if name == "main":
main()
