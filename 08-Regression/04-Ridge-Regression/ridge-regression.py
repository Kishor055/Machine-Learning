# File: `08-Regression/04-Ridge-Regression/ridge-regression.py`

```python
"""
Ridge Regression using Scikit-learn
===================================

Ridge Regression is a linear regression algorithm that applies
L2 regularization to reduce overfitting and shrink large coefficients.

Features:
    1. Dataset generation
    2. Train-test split
    3. Feature scaling
    4. Ridge Regression model training
    5. Hyperparameter tuning using GridSearchCV
    6. Evaluation using MAE, MSE, RMSE, and R²
    7. Coefficient analysis
    8. Model comparison with Linear Regression
    9. Actual vs predicted visualization
    10. Residual analysis

Install dependencies:
    pip install numpy pandas matplotlib scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ==================================================
# 1. Generate a Synthetic Dataset
# ==================================================

X, y = make_regression(
    n_samples=500,
    n_features=8,
    n_informative=6,
    noise=20,
    random_state=42,
)

feature_names = [f"Feature_{i + 1}" for i in range(X.shape[1])]

df = pd.DataFrame(X, columns=feature_names)
df["Target"] = y

print("=" * 60)
print("RIDGE REGRESSION USING SCIKIT-LEARN")
print("=" * 60)

print("\nDataset Preview:")
print(df.head())

print("\nDataset Shape:", df.shape)


# ==================================================
# 2. Separate Features and Target
# ==================================================

X = df.drop(columns=["Target"])
y = df["Target"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)


# ==================================================
# 3. Split Dataset
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ==================================================
# 4. Create a Pipeline
# ==================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge()),
])


# ==================================================
# 5. Tune Hyperparameter alpha
# ==================================================

param_grid = {
    "ridge__alpha": [
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
    ]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=5,
    n_jobs=-1,
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest Hyperparameter:")
print("Alpha:", grid_search.best_params_["ridge__alpha"])

print(
    "Best Cross-Validation RMSE:",
    np.sqrt(-grid_search.best_score_),
)


# ==================================================
# 6. Make Predictions
# ==================================================

y_pred = best_model.predict(X_test)

results = pd.DataFrame({
    "Actual": y_test.to_numpy(),
    "Predicted": y_pred,
})

results["Residual"] = results["Actual"] - results["Predicted"]

print("\nActual vs Predicted Values:")
print(results.head(10).round(2).to_string(index=False))


# ==================================================
# 7. Evaluate Model Performance
# ==================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ==================================================
# 8. Analyze Coefficients
# ==================================================

ridge_model = best_model.named_steps["ridge"]

coefficients = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": ridge_model.coef_,
})

coefficients["Absolute Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    by="Absolute Coefficient",
    ascending=False,
)

print("\nFeature Coefficients:")
print(coefficients.round(4).to_string(index=False))

print("\nIntercept:", round(ridge_model.intercept_, 4))


# ==================================================
# 9. Compare with Linear Regression
# ==================================================

linear_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("linear", LinearRegression()),
])

linear_pipeline.fit(X_train, y_train)

linear_predictions = linear_pipeline.predict(X_test)

linear_rmse = np.sqrt(
    mean_squared_error(y_test, linear_predictions)
)

ridge_rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression",
    ],
    "MAE": [
        mean_absolute_error(y_test, linear_predictions),
        mae,
    ],
    "RMSE": [
        linear_rmse,
        ridge_rmse,
    ],
    "R2": [
        r2_score(y_test, linear_predictions),
        r2,
    ],
})

print("\nModel Comparison:")
print(comparison.round(4).to_string(index=False))


# ==================================================
# 10. Visualize Actual vs Predicted Values
# ==================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    edgecolor="black",
    alpha=0.75,
    label="Ridge Predictions",
)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    color="red",
    label="Perfect Prediction",
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Ridge Regression: Actual vs Predicted")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==================================================
# 11. Residual Plot
# ==================================================

residuals = y_test.to_numpy() - y_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    y_pred,
    residuals,
    edgecolor="black",
    alpha=0.75,
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--",
)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Ridge Regression: Residual Analysis")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ==================================================
# 12. Compare Coefficients with Linear Regression
# ==================================================

linear_model = linear_pipeline.named_steps["linear"]

coefficient_comparison = pd.DataFrame({
    "Feature": feature_names,
    "Linear Regression": linear_model.coef_,
    "Ridge Regression": ridge_model.coef_,
})

print("\nCoefficient Comparison:")
print(coefficient_comparison.round(4).to_string(index=False))

coefficient_comparison.set_index("Feature").plot(
    kind="bar",
    figsize=(10, 6),
)

plt.title("Linear Regression vs Ridge Coefficients")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ==================================================
# 13. Predict New Samples
# ==================================================

new_samples = pd.DataFrame(
    [[0.5, 1.2, -0.3, 0.8, 1.0, -0.5, 0.2, 0.7]],
    columns=feature_names,
)

new_prediction = best_model.predict(new_samples)

print("\nNew Sample:")
print(new_samples)

print(f"\nPredicted Target: {new_prediction[0]:.4f}")


# ==================================================
# 14. Evaluate Different Alpha Values
# ==================================================

alpha_results = []

for alpha in [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]:
    candidate = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])

    candidate.fit(X_train, y_train)

    candidate_predictions = candidate.predict(X_test)

    alpha_results.append({
        "Alpha": alpha,
        "RMSE": np.sqrt(
            mean_squared_error(y_test, candidate_predictions)
        ),
        "R2": r2_score(y_test, candidate_predictions),
    })

alpha_df = pd.DataFrame(alpha_results)

print("\nPerformance for Different Alpha Values:")
print(alpha_df.round(4).to_string(index=False))

print("\nProgram Completed Successfully!")
```


