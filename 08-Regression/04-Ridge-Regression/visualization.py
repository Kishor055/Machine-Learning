# `08-Regression/04-Ridge-Regression/visualization.py`

```python
"""
Ridge Regression Visualization
------------------------------
Visualizes:
1. Synthetic dataset and feature relationships
2. Ridge regression curves for different alpha values
3. Effect of alpha on model coefficients
4. Actual vs. predicted values
5. Residual distribution
6. Model performance for different alpha values

Install dependencies:
    pip install numpy matplotlib scikit-learn

Run:
    python visualization.py
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. Generate a Synthetic Regression Dataset
# ============================================================

X, y = make_regression(
    n_samples=250,
    n_features=1,
    n_informative=1,
    noise=25,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


# ============================================================
# 2. Visualize the Dataset
# ============================================================

plt.figure(figsize=(9, 5))

plt.scatter(
    X_train[:, 0],
    y_train,
    color="steelblue",
    alpha=0.7,
    label="Training data",
)

plt.scatter(
    X_test[:, 0],
    y_test,
    color="orange",
    alpha=0.8,
    label="Testing data",
)

plt.title("Synthetic Regression Dataset")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 3. Compare Ridge Regression for Different Alpha Values
# ============================================================

alpha_values = [0.01, 1, 10, 100, 1000]

X_plot = np.linspace(
    X.min() - 0.5,
    X.max() + 0.5,
    300,
).reshape(-1, 1)

plt.figure(figsize=(10, 6))

plt.scatter(
    X_train[:, 0],
    y_train,
    color="gray",
    alpha=0.4,
    label="Training data",
)

# Standardize features inside each pipeline.
for alpha in alpha_values:
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha),
    )

    model.fit(X_train, y_train)
    y_plot = model.predict(X_plot)

    plt.plot(
        X_plot[:, 0],
        y_plot,
        linewidth=2,
        label=f"Ridge alpha={alpha}",
    )

# Ordinary linear regression baseline.
linear_model = make_pipeline(
    StandardScaler(),
    LinearRegression(),
)

linear_model.fit(X_train, y_train)

plt.plot(
    X_plot[:, 0],
    linear_model.predict(X_plot),
    linestyle="--",
    linewidth=2,
    label="Linear Regression",
)

plt.title("Effect of Alpha on Ridge Regression")
plt.xlabel("Feature")
plt.ylabel("Predicted Target")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 4. Visualize Coefficient Shrinkage
# ============================================================

# A multi-feature dataset helps demonstrate how Ridge
# shrinks several coefficients as alpha increases.

X_multi, y_multi = make_regression(
    n_samples=300,
    n_features=8,
    n_informative=6,
    noise=20,
    random_state=42,
)

X_train_multi, X_test_multi, y_train_multi, y_test_multi = (
    train_test_split(
        X_multi,
        y_multi,
        test_size=0.2,
        random_state=42,
    )
)

coefficient_values = {}

for alpha in alpha_values:
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha),
    )

    model.fit(X_train_multi, y_train_multi)

    ridge = model.named_steps["ridge"]
    coefficient_values[alpha] = ridge.coef_

feature_names = [
    f"Feature {i + 1}"
    for i in range(X_multi.shape[1])
]

x_positions = np.arange(len(feature_names))
bar_width = 0.15

plt.figure(figsize=(12, 6))

for index, alpha in enumerate(alpha_values):
    offset = (
        index - (len(alpha_values) - 1) / 2
    ) * bar_width

    plt.bar(
        x_positions + offset,
        coefficient_values[alpha],
        width=bar_width,
        label=f"alpha={alpha}",
    )

plt.axhline(0, color="black", linewidth=0.8)
plt.title("Ridge Regression: Coefficient Shrinkage")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.xticks(x_positions, feature_names)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 5. Actual vs. Predicted Values
# ============================================================

final_model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0),
)

final_model.fit(X_train_multi, y_train_multi)

y_pred = final_model.predict(X_test_multi)

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test_multi,
    y_pred,
    alpha=0.7,
)

minimum = min(y_test_multi.min(), y_pred.min())
maximum = max(y_test_multi.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    linewidth=2,
    label="Perfect predictions",
)

plt.title("Actual vs. Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 6. Residual Plot
# ============================================================

residuals = y_test_multi - y_pred

plt.figure(figsize=(8, 5))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.7,
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=2,
)

plt.title("Residual Plot")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 7. Residual Distribution
# ============================================================

plt.figure(figsize=(8, 5))

plt.hist(
    residuals,
    bins=20,
    edgecolor="black",
    alpha=0.8,
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=2,
)

plt.title("Distribution of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 8. Compare Model Performance Across Alpha Values
# ============================================================

results = {
    "alpha": [],
    "train_rmse": [],
    "test_rmse": [],
    "test_r2": [],
}

for alpha in alpha_values:
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha),
    )

    model.fit(X_train_multi, y_train_multi)

    train_predictions = model.predict(X_train_multi)
    test_predictions = model.predict(X_test_multi)

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train_multi,
            train_predictions,
        )
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test_multi,
            test_predictions,
        )
    )

    test_r2 = r2_score(
        y_test_multi,
        test_predictions,
    )

    results["alpha"].append(alpha)
    results["train_rmse"].append(train_rmse)
    results["test_rmse"].append(test_rmse)
    results["test_r2"].append(test_r2)

    print(
        f"Alpha: {alpha:>7} | "
        f"Train RMSE: {train_rmse:8.3f} | "
        f"Test RMSE: {test_rmse:8.3f} | "
        f"Test R²: {test_r2:7.3f}"
    )


# ============================================================
# 9. Plot RMSE vs. Alpha
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    results["alpha"],
    results["train_rmse"],
    marker="o",
    label="Training RMSE",
)

plt.plot(
    results["alpha"],
    results["test_rmse"],
    marker="s",
    label="Testing RMSE",
)

plt.xscale("log")
plt.title("Model Error vs. Regularization Strength")
plt.xlabel("Alpha (log scale)")
plt.ylabel("RMSE")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 10. Plot R² vs. Alpha
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    results["alpha"],
    results["test_r2"],
    marker="o",
    linewidth=2,
)

plt.xscale("log")
plt.title("R² Score vs. Regularization Strength")
plt.xlabel("Alpha (log scale)")
plt.ylabel("Test R² Score")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 11. Final Model Evaluation
# ============================================================

mae = np.mean(np.abs(y_test_multi - y_pred))
mse = mean_squared_error(y_test_multi, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_multi, y_pred)

print("\nFinal Ridge Regression Evaluation")
print("-" * 40)
print(f"MAE  : {mae:.3f}")
print(f"MSE  : {mse:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

print("\nVisualization completed successfully.")
```

## What this file demonstrates

| Visualization | Purpose |
|---|---|
| Dataset scatter plot | Understand the data |
| Regression lines | Observe how `alpha` affects predictions |
| Coefficient bar chart | See how Ridge shrinks coefficients |
| Actual vs. predicted | Evaluate prediction quality visually |
| Residual plot and histogram | Inspect prediction errors |
| RMSE and R² plots | Compare regularization strengths |

## Run the file

From the `08-Regression/04-Ridge-Regression/` directory:

```bash
pip install numpy matplotlib scikit-learn
python visualization.py
```

**Key takeaway:** Increasing `alpha` strengthens regularization and generally shrinks Ridge coefficients toward zero. An excessively large `alpha` can cause underfitting, so select it using validation or cross-validation rather than assuming a larger value is better.
