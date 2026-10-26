# visualization.py

```python
"""
Simple Linear Regression Visualization

Visualizations included:
1. Scatter plot with regression line
2. Residual plot
3. Actual vs. predicted plot
4. Residual distribution histogram
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def create_dataset():
    """Create a sample dataset of experience and salary."""
    X = np.array([
        [1], [2], [3], [4], [5],
        [6], [7], [8], [9], [10],
        [11], [12], [13], [14], [15],
        [16], [17], [18], [19], [20],
    ])

    y = np.array([
        25000, 28000, 32000, 35000, 39000,
        42000, 46000, 49000, 53000, 56000,
        60000, 63000, 67000, 70000, 74000,
        77000, 81000, 84000, 88000, 91000,
    ])

    return X, y


def train_model(X, y):
    """Split the data and train a Linear Regression model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("MODEL SUMMARY")
    print("-" * 40)
    print(f"Slope: {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(
        f"Equation: Salary = {model.intercept_:.2f} "
        f"+ {model.coef_[0]:.2f} * Experience"
    )
    print(f"Test MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"Test R²: {r2_score(y_test, y_pred):.4f}")

    return model, X_train, X_test, y_train, y_test, y_pred


def plot_regression_line(model, X_train, X_test, y_train, y_test):
    """Visualize training data, testing data, and the fitted line."""
    X_all = np.concatenate([X_train, X_test])
    x_line = np.linspace(
        X_all.min(),
        X_all.max(),
        200,
    ).reshape(-1, 1)

    y_line = model.predict(x_line)

    plt.figure(figsize=(10, 6))

    plt.scatter(
        X_train.ravel(),
        y_train,
        label="Training data",
        marker="o",
    )
    plt.scatter(
        X_test.ravel(),
        y_test,
        label="Testing data",
        marker="x",
        s=80,
    )
    plt.plot(
        x_line.ravel(),
        y_line,
        label="Regression line",
        linewidth=2,
    )

    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.title("Simple Linear Regression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_residuals(y_test, y_pred):
    """Plot prediction residuals against predicted values."""
    residuals = y_test - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, linestyle="--")

    plt.xlabel("Predicted Salary")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_actual_vs_predicted(y_test, y_pred):
    """Compare actual and predicted target values."""
    lower = min(y_test.min(), y_pred.min())
    upper = max(y_test.max(), y_pred.max())

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        label="Perfect predictions",
    )

    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Actual vs. Predicted Salary")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_residual_distribution(y_test, y_pred):
    """Show the distribution of residuals."""
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=8, edgecolor="black")

    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()


def main():
    # 1. Create dataset
    X, y = create_dataset()

    # 2. Train model and make predictions
    model, X_train, X_test, y_train, y_test, y_pred = train_model(
        X, y
    )

    # 3. Generate visualizations
    plot_regression_line(
        model, X_train, X_test, y_train, y_test
    )
    plot_residuals(y_test, y_pred)
    plot_actual_vs_predicted(y_test, y_pred)
    plot_residual_distribution(y_test, y_pred)

    # 4. Display all figures
    plt.show()


if __name__ == "__main__":
    main()
```

## Installation

```bash
pip install numpy matplotlib scikit-learn
```

## Run the script

From your repository root:

```bash
python 08-Regression/01-Linear-Regression/visualization.py
```

## What each visualization shows

1. **Regression line:** Shows how the fitted line relates to the observed training and testing data.
2. **Residual plot:** Helps identify systematic errors, changing error variance, and possible nonlinearity.
3. **Actual vs. predicted:** Points closer to the diagonal represent more accurate predictions.
4. **Residual histogram:** Shows the distribution of prediction errors; it can help assess their shape but does not establish normality by itself.

**Note:** The dataset is synthetic and intentionally small for learning. Residual plots and test metrics should be interpreted cautiously with such a small test set.
