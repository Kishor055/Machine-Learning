# """\
 Introduction to Machine Learning — Examples

 Author: KISHOR KAKDE\
 Repository: Machine-Learning

 This file demonstrates the fundamental concepts of Machine Learning\
 using simple Python and Scikit-Learn examples.

 ## Topics Covered

 1. Traditional Programming
2. Features and Labels
3. Dataset Representation
4. Training and Testing Data
5. Model Training
6. Making Predictions
7. Model Evaluation
8. Generalization
9. A Complete ML Workflow

 Run:\
 python examples.py\
 """

 from **future** import annotations

 from sklearn.linear\_model import LinearRegression\
 from sklearn.metrics import mean\_absolute\_error, mean\_squared\_error, r2\_score\
 from sklearn.model\_selection import train\_test\_split\
 import numpy as np

 # =============================================================================

 # 1\. TRADITIONAL PROGRAMMING

 # =============================================================================

 def traditional\_programming\_example() -\> None:\
 """\
 Demonstrates traditional rule-based programming.

```
In traditional programming, the programmer explicitly defines
the rules that the computer should follow.
"""

temperature = 32

if temperature >= 30:
    result = "Hot"
elif temperature >= 20:
    result = "Warm"
else:
    result = "Cold"

print("\n1. Traditional Programming")
print("-" * 50)
print(f"Temperature: {temperature}°C")
print(f"Result: {result}")
```

 # =============================================================================

 # 2\. FEATURES AND LABELS

 # =============================================================================

 def features\_and\_labels\_example() -\> None:\
 """\
 Demonstrates the difference between features and labels.

```
Features:
    Input variables used by the model.

Label:
    The value that the model attempts to predict.
"""

features = [
    [1000, 2],
    [1200, 2],
    [1500, 3],
    [1800, 3],
    [2200, 4],
]

prices = [40, 48, 60, 72, 88]

print("\n2. Features and Labels")
print("-" * 50)

print("Features:")
for row in features:
    print(row)

print("\nLabels:")
print(prices)
```

 # =============================================================================

 # 3\. DATASET REPRESENTATION

 # =============================================================================

 def dataset\_representation\_example() -\> None:\
 """\
 Shows how a machine learning dataset can be represented.

```
Each row represents one observation.
Each column represents a feature.
"""

data = np.array(
    [
        [1000, 2, 40],
        [1200, 2, 48],
        [1500, 3, 60],
        [1800, 3, 72],
        [2200, 4, 88],
    ]
)

print("\n3. Dataset Representation")
print("-" * 50)

print("Dataset:")
print(data)

print(f"\nNumber of rows: {data.shape[0]}")
print(f"Number of columns: {data.shape[1]}")
```

 # =============================================================================

 # 4\. TRAINING AND TESTING DATA

 # =============================================================================

 def train\_test\_split\_example() -\> tuple:\
 """\
 Splits a dataset into training and testing data.

```
Training data:
    Used to teach the model.

Testing data:
    Used to evaluate how well the model performs on unseen data.
"""

X = np.array(
    [
        [800],
        [1000],
        [1200],
        [1400],
        [1600],
        [1800],
        [2000],
        [2200],
        [2400],
        [2600],
    ]
)

y = np.array(
    [
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("\n4. Training and Testing Data")
print("-" * 50)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

return X_train, X_test, y_train, y_test
```

 # =============================================================================

 # 5\. MODEL TRAINING

 # =============================================================================

 def model\_training\_example(\
 X\_train: np.ndarray,\
 y\_train: np.ndarray,\
 ) -\> LinearRegression:\
 """\
 Creates and trains a Linear Regression model.\
 """

```
model = LinearRegression()

model.fit(X_train, y_train)

print("\n5. Model Training")
print("-" * 50)

print("Model trained successfully.")
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

return model
```

 # =============================================================================

 # 6\. MAKING PREDICTIONS

 # =============================================================================

 def prediction\_example(\
 model: LinearRegression,\
 X\_test: np.ndarray,\
 y\_test: np.ndarray,\
 ) -\> np.ndarray:\
 """\
 Uses the trained model to make predictions.\
 """

```
predictions = model.predict(X_test)

print("\n6. Making Predictions")
print("-" * 50)

print(f"{'Area':>10} {'Actual':>15} {'Predicted':>15}")
print("-" * 45)

for area, actual, predicted in zip(
    X_test.flatten(),
    y_test,
    predictions,
):
    print(
        f"{area:>10} "
        f"{actual:>15.2f} "
        f"{predicted:>15.2f}"
    )

return predictions
```

 # =============================================================================

 # 7\. MODEL EVALUATION

 # =============================================================================

 def model\_evaluation\_example(\
 y\_test: np.ndarray,\
 predictions: np.ndarray,\
 ) -\> None:\
 """\
 Evaluates the regression model using common metrics.

```
Metrics:
    MAE  - Mean Absolute Error
    MSE  - Mean Squared Error
    RMSE - Root Mean Squared Error
    R²   - Coefficient of Determination
"""

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n7. Model Evaluation")
print("-" * 50)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")
```

 # =============================================================================

 # 8\. PREDICTING NEW DATA

 # =============================================================================

 def new\_prediction\_example(model: LinearRegression) -\> None:\
 """\
 Uses the trained model to predict the price of new houses.\
 """

```
new_houses = np.array(
    [
        [1100],
        [1700],
        [2100],
        [2800],
    ]
)

predictions = model.predict(new_houses)

print("\n8. Predictions for New Data")
print("-" * 50)

for area, price in zip(new_houses.flatten(), predictions):
    print(
        f"House Area: {area:>4} sq ft "
        f"→ Predicted Price: ₹{price:.2f} Lakhs"
    )
```

 # =============================================================================

 # 9\. GENERALIZATION

 # =============================================================================

 def generalization\_example(\
 model: LinearRegression,\
 X\_train: np.ndarray,\
 y\_train: np.ndarray,\
 X\_test: np.ndarray,\
 y\_test: np.ndarray,\
 ) -\> None:\
 """\
 Demonstrates the concept of generalization.

```
A good model should perform well on both training data
and previously unseen testing data.
"""

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("\n9. Generalization")
print("-" * 50)

print(f"Training R² score: {train_score:.4f}")
print(f"Testing R² score : {test_score:.4f}")

if abs(train_score - test_score) < 0.10:
    print("The model shows reasonably consistent performance.")
else:
    print("There may be a generalization problem.")
```

 # =============================================================================

 # 10\. COMPLETE MACHINE LEARNING WORKFLOW

 # =============================================================================

 def complete\_ml\_workflow() -\> None:\
 """\
 Demonstrates the complete beginner-level ML workflow.\
 """

```
print("\n" + "=" * 60)
print("COMPLETE MACHINE LEARNING WORKFLOW")
print("=" * 60)

# Step 1: Create / collect data
X = np.array(
    [
        [800],
        [1000],
        [1200],
        [1400],
        [1600],
        [1800],
        [2000],
        [2200],
        [2400],
        [2600],
    ]
)

y = np.array(
    [
        32,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
    ]
)

print("\nStep 1: Data collected")

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("Step 2: Data split")

# Step 3: Create model
model = LinearRegression()

print("Step 3: Model created")

# Step 4: Train
model.fit(X_train, y_train)

print("Step 4: Model trained")

# Step 5: Predict
predictions = model.predict(X_test)

print("Step 5: Predictions generated")

# Step 6: Evaluate
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Step 6: Model evaluated")

# Step 7: Results
print("\nFinal Results")
print("-" * 30)
print(f"MAE: {mae:.4f}")
print(f"R² : {r2:.4f}")
```

 # =============================================================================

 # MAIN PROGRAM

 # =============================================================================

 def main() -\> None:\
 """\
 Run all examples.\
 """

```
print("=" * 60)
print("INTRODUCTION TO MACHINE LEARNING")
print("Author: KISHOR KAKDE")
print("=" * 60)

traditional_programming_example()

features_and_labels_example()

dataset_representation_example()

X_train, X_test, y_train, y_test = train_test_split_example()

model = model_training_example(
    X_train,
    y_train,
)

predictions = prediction_example(
    model,
    X_test,
    y_test,
)

model_evaluation_example(
    y_test,
    predictions,
)

new_prediction_example(model)

generalization_example(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
)

complete_ml_workflow()

print("\n" + "=" * 60)
print("END OF EXAMPLES")
print("=" * 60)
```

 if **name** == "**main**":\
 main()

 Run it from your repository root:

```
python 01-ML-Fundamentals/01-Introduction-to-Machine-Learning/examples.py
```

 If Scikit-Learn/NumPy aren't installed yet:

```
pip install numpy scikit-learn
```

 This version is intentionally **educational rather than artificially complex**: someone opening the file as their first ML program can follow the progression from data → features → split → model → training → prediction → evaluation → generalization.
