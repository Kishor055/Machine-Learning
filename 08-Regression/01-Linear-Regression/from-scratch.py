```python
"""
Linear Regression From Scratch
------------------------------
Implement Simple Linear Regression using NumPy.

Topics covered:
1. Dataset creation
2. Mean calculation
3. Slope and intercept calculation
4. Prediction
5. Model evaluation
6. Visualization
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionScratch:
    """Simple Linear Regression using Ordinary Least Squares."""

    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X, y):
        """Learn the slope and intercept from training data."""
        X = np.asarray(X, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.size == 0 or X.size != y.size:
            raise ValueError(
                "X and y must contain the same nonzero number of samples."
            )

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("X and y must contain only finite values.")

        x_mean = np.mean(X)
        y_mean = np.mean(y)

        denominator = np.sum((X - x_mean) ** 2)

        if denominator == 0:
            raise ValueError("X must contain at least two distinct values.")

        # Calculate slope
        self.slope = np.sum(
            (X - x_mean) * (y - y_mean)
        ) / denominator

        # Calculate intercept
        self.intercept = y_mean - self.slope * x_mean

        return self

    def predict(self, X):
        """Predict target values for new inputs."""
        if self.slope is None or self.intercept is None:
            raise ValueError("Call fit() before predict().")

        X = np.asarray(X, dtype=float)

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        return self.intercept + self.slope * X

    def score(self, X, y):
        """Calculate the coefficient of determination (R²)."""
        y = np.asarray(y, dtype=float).reshape(-1)
        y_pred = self.predict(X).reshape(-1)

        if y.size != y_pred.size or y.size == 0:
            raise ValueError("X and y must have matching nonzero lengths.")

        if not np.all(np.isfinite(y)):
            raise ValueError("y must contain only finite values.")

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - (ss_res / ss_tot)


def mean_absolute_error(y_true, y_pred):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))


def mean_squared_error(y_true, y_pred):
    """Calculate Mean Squared Error."""
    return np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    # 1. Create a sample dataset
    X = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([25000, 30000, 40000, 45000, 55000], dtype=float)

    # 2. Create and train the model
    model = LinearRegressionScratch()
    model.fit(X, y)

    # 3. Generate predictions
    y_pred = model.predict(X)

    # 4. Display model parameters
    print("=" * 45)
    print("LINEAR REGRESSION FROM SCRATCH")
    print("=" * 45)

    print(f"Slope:      {model.slope:.2f}")
    print(f"Intercept:  {model.intercept:.2f}")
    print(f"Equation:   y = {model.intercept:.2f} + {model.slope:.2f}x")

    # 5. Evaluate the model
    print("\nMODEL EVALUATION")
    print("-" * 45)
    print(f"MAE:        {mean_absolute_error(y, y_pred):.2f}")
    print(f"MSE:        {mean_squared_error(y, y_pred):.2f}")
    print(f"RMSE:       {root_mean_squared_error(y, y_pred):.2f}")
    print(f"R² Score:   {model.score(X, y):.4f}")

    print("\nACTUAL VS PREDICTED")
    print("-" * 45)

    for actual, predicted in zip(y, y_pred):
        print(f"Actual: {actual:,.2f} | Predicted: {predicted:,.2f}")

    # 6. Predict for a new input
    experience = 6
    predicted_salary = model.predict([experience])

    print("\nNEW PREDICTION")
    print("-" * 45)
    print(f"Experience: {experience} years")
    print(f"Predicted salary: {predicted_salary[0]:,.2f}")

    # 7. Visualize the regression line
    plt.scatter(X, y, label="Actual values")
    plt.plot(X, y_pred, label="Regression line")

    plt.xlabel("Experience (Years)")
    plt.ylabel("Salary")
    plt.title("Simple Linear Regression From Scratch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```
