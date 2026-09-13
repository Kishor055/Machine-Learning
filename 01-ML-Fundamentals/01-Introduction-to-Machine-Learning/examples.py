# Introduction to Machine Learning
# Author: KISHOR KAKDE

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# 1. Traditional Programming

temperature = 32

if temperature >= 30:
    result = "Hot"
elif temperature >= 20:
    result = "Warm"
else:
    result = "Cold"

print("1. Traditional Programming")
print("Temperature:", temperature)
print("Result:", result)


# 2. Features and Labels

X = [
    [1000, 2],
    [1200, 2],
    [1500, 3],
    [1800, 3],
    [2200, 4],
]

y = [40, 48, 60, 72, 88]

print("\n2. Features and Labels")
print("Features:", X)
print("Labels:", y)


# 3. Dataset using NumPy

X = np.array([
    [1000],
    [1200],
    [1500],
    [1800],
    [2200]
])

y = np.array([
    40,
    48,
    60,
    72,
    88
])

print("\n3. Dataset")
print("X:", X)
print("y:", y)
print("X Shape:", X.shape)
print("y Shape:", y.shape)


# 4. Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n4. Train Test Split")
print("X Train:", X_train)
print("X Test:", X_test)
print("y Train:", y_train)
print("y Test:", y_test)


# 5. Create Model

model = LinearRegression()


# 6. Train Model

model.fit(X_train, y_train)

print("\n5. Model Training")
print("Model trained successfully")


# 7. Model Parameters

print("\n6. Model Parameters")
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)


# 8. Make Predictions

predictions = model.predict(X_test)

print("\n7. Predictions")

for actual, predicted in zip(y_test, predictions):
    print(
        "Actual:",
        actual,
        "Predicted:",
        round(predicted, 2)
    )


# 9. Model Evaluation

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n8. Model Evaluation")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))


# 10. Predict New Data

new_house = np.array([
    [2000]
])

prediction = model.predict(new_house)

print("\n9. New Prediction")
print("Area:", new_house[0][0], "sq ft")
print("Predicted Price:", round(prediction[0], 2), "Lakhs")


# 11. Training and Testing Score

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("\n10. Model Scores")
print("Training Score:", round(train_score, 2))
print("Testing Score:", round(test_score, 2))


# 12. Machine Learning Workflow

print("\nMachine Learning Workflow Completed")
