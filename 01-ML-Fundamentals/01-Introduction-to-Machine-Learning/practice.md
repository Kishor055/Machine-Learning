# Machine Learning Fundamentals — Practice

 > **Author:** KISHOR KAKDE\
>  **Level:** Beginner\
>  **Module:** 01 — ML Fundamentals\
>  **Topic:** Introduction to Machine Learning

---

 ## 📌 Learning Objectives

 By completing these exercises, you should be able to:

 - Explain what Machine Learning is.
- Differentiate traditional programming from Machine Learning.
- Identify features and labels.
- Understand training and testing data.
- Explain how a Machine Learning model learns.
- Make predictions using a trained model.
- Understand the difference between training and testing.
- Explain generalization.
- Identify common Machine Learning applications.
- Build a basic Machine Learning workflow using Python.

---

 # 🟢 Part 1 — Basic Concepts

 ### Exercise 1 — What is Machine Learning?

 Answer the following questions in your own words:

 1. What is Machine Learning?
2. Why do we use Machine Learning?
3. How is Machine Learning different from traditional programming?
4. What does it mean for a machine to "learn" from data?
5. What is a Machine Learning model?

 **Expected outcome:**

 Write a short explanation of approximately 100–150 words.

---

 ### Exercise 2 — Traditional Programming vs Machine Learning

 Complete the table:

 | Concept | Traditional Programming | Machine Learning |
| --- | --- | --- |
| Input | ? | ? |
| Rules | ? | ? |
| Output | ? | ? |
| Learning from data | ? | ? |

 Then explain:

 > When would you prefer Machine Learning instead of traditional programming?

---

 ### Exercise 3 — Identify Machine Learning Applications

 Determine whether Machine Learning could reasonably be used for each problem.

 | Problem | Machine Learning? | Why? |
| --- | --- | --- |
| Predict house prices | ? | ? |
| Calculate 2 + 2 | ? | ? |
| Detect spam emails | ? | ? |
| Sort numbers | ? | ? |
| Recommend movies | ? | ? |
| Recognize faces | ? | ? |
| Calculate employee salary using a fixed formula | ? | ? |
| Predict customer churn | ? | ? |

---

 # 🟢 Part 2 — Features and Labels

 ### Exercise 4 — Identify Features and Labels

 Suppose you want to predict the price of a house.

 Your dataset contains:

 - Area
- Number of bedrooms
- Number of bathrooms
- Location
- Age of the house
- Parking spaces
- House price

 Answer:

 1. What are the features?
2. What is the label/target?
3. Which column is the model trying to predict?

---

 ### Exercise 5 — Student Performance

 Consider this dataset:

```
Hours_Studied | Attendance | Assignments | Exam_Score
------------------------------------------------------
2             | 70         | 5           | 45
4             | 80         | 7           | 60
6             | 90         | 9           | 75
8             | 95         | 10          | 88
```

 Identify:

```
Features:
?

Label:
?
```

 Then answer:

 > If `Exam_Score` is the target, how many features does the dataset contain?

---

 ### Exercise 6 — Create Your Own Dataset

 Create a small dataset for predicting **car prices**.

 Your dataset should contain at least:

 - 4 features
- 1 target
- 10 observations

 Example structure:

```
Engine_Size | Mileage | Age | Horsepower | Price
-------------------------------------------------
?           | ?       | ?   | ?          | ?
```

---

 # 🟡 Part 3 — Python Practice

 ### Exercise 7 — Represent Data Using NumPy

 Create the following dataset using NumPy:

```
Area    Bedrooms    Price
1000    2           40
1200    2           48
1500    3           60
1800    3           72
2200    4           88
```

 Requirements:

 - Store features in `X`.
- Store the target in `y`.
- Print `X`.
- Print `y`.
- Print their shapes.

 Expected concept:

```
X.shape
y.shape
```

---

 ### Exercise 8 — Inspect the Dataset

 Using Python, determine:

 1. Number of samples.
2. Number of features.
3. Shape of the feature matrix.
4. Shape of the target vector.

 Use:

```
X.shape
y.shape
```

---

 # 🟡 Part 4 — Training and Testing

 ### Exercise 9 — Train/Test Split

 Use `train_test_split()` from Scikit-Learn.

 Requirements:

 - Use `test_size=0.2`.
- Use `random_state=42`.
- Print the training data.
- Print the testing data.
- Print the number of training samples.
- Print the number of testing samples.

 Starter code:

```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

 ### Exercise 10 — Why Do We Split Data?

 Answer:

 1. Why shouldn't we train and evaluate a model using exactly the same data?
2. What is training data?
3. What is testing data?
4. What does "unseen data" mean?
5. What is the purpose of `random_state`?

 Write your answers in your own words.

---

 # 🟡 Part 5 — Your First Machine Learning Model

 ### Exercise 11 — Train Linear Regression

 Use the dataset:

```
Area    Price
800     32
1000    40
1200    48
1400    56
1600    64
1800    72
2000    80
2200    88
2400    96
2600    104
```

 Build a Linear Regression model.

 Requirements:

 1. Import `LinearRegression`.
2. Create the model.
3. Split the dataset.
4. Train the model.
5. Make predictions.
6. Display the predictions.

 Starter:

```
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions)
```

---

 ### Exercise 12 — Predict a New House

 Using your trained model, predict the price of a house with:

```
Area = 3000 sq ft
```

 Starter:

```
new_house = [[3000]]

prediction = model.predict(new_house)

print(prediction)
```

---

 # 🟠 Part 6 — Understanding the Model

 ### Exercise 13 — Model Parameters

 After training the Linear Regression model, print:

```
print(model.coef_)
print(model.intercept_)
```

 Answer:

 1. What is the coefficient?
2. What is the intercept?
3. What does the coefficient represent?
4. How does the model use these values to make predictions?

---

 ### Exercise 14 — Understand the Prediction Equation

 Linear Regression can be represented as:

```
y = mx + b
```

 Where:

```
y = prediction
m = coefficient
x = input feature
b = intercept
```

 Use your trained model to determine:

```
m = ?
b = ?
```

 Then manually calculate a prediction for:

```
x = 2000
```

 Compare your manual result with:

```
model.predict([[2000]])
```

---

 # 🟠 Part 7 — Model Evaluation

 ### Exercise 15 — Calculate MAE

 Use:

```
from sklearn.metrics import mean_absolute_error
```

 Calculate:

```
MAE
```

 Answer:

 > What does MAE tell us about the model?

---

 ### Exercise 16 — Calculate MSE

 Use:

```
from sklearn.metrics import mean_squared_error
```

 Calculate:

```
MSE
```

 Answer:

 > Why does MSE penalize large errors more heavily than MAE?

---

 ### Exercise 17 — Calculate RMSE

 Calculate:

```
RMSE = √MSE
```

 You may use:

```
import numpy as np
```

 Answer:

 > Why is RMSE often easier to interpret than MSE?

---

 ### Exercise 18 — Calculate R²

 Use:

```
from sklearn.metrics import r2_score
```

 Calculate the R² score.

 Answer:

 1. What does R² measure?
2. What does an R² close to `1` indicate?
3. Can R² be negative?
4. Why should you evaluate a model on unseen data?

---

 # 🔴 Part 8 — Think Like a Machine Learning Engineer

 ### Exercise 19 — Identify the Complete Workflow

 Put the following steps in the correct order:

```
Model Evaluation
Data Collection
Model Training
Data Cleaning
Prediction
Problem Definition
Data Splitting
Feature Engineering
```

 Write the correct workflow:

```
1. ?
2. ?
3. ?
4. ?
5. ?
6. ?
7. ?
8. ?
```

---

 ### Exercise 20 — Real-World Problem

 You work for an e-commerce company.

 The company wants to predict whether a customer will leave the platform.

 Available information:

 - Customer age
- Monthly spending
- Number of purchases
- Number of support tickets
- Account age
- Customer satisfaction score
- Churn status

 Answer:

 1. What is the Machine Learning problem?
2. What are the features?
3. What is the target?
4. Is this regression or classification?
5. What type of data would you use for training?
6. What type of data would you use for testing?
7. What metric could be useful for evaluating the model?

---

 # 🔴 Part 9 — Mini Project

 ## House Price Prediction

 Build a complete beginner-level Machine Learning project.

 ### Dataset

 Create a dataset containing:

```
Area
Bedrooms
Bathrooms
Age
Parking
Price
```

 Use at least **20 observations**.

 ### Requirements

 Your project should:

 1. Create or load the dataset.
2. Separate features and target.
3. Split the data.
4. Create a Machine Learning model.
5. Train the model.
6. Make predictions.
7. Evaluate the model.
8. Predict the price of a new house.
9. Explain the results.

 ### Recommended structure

```
house-price-practice/
│
├── README.md
├── data/
│   └── house_prices.csv
│
├── train.py
├── predict.py
└── evaluation.py
```

---

 # 🔵 Challenge Problems

 ## Challenge 1 — Compare Two Models

 Train two different regression models.

 Compare their:

 - MAE
- MSE
- RMSE
- R²

 Answer:

 > Which model performs better and why?

---

 ## Challenge 2 — Change the Test Size

 Train the same model using:

```
test_size = 0.1
test_size = 0.2
test_size = 0.3
test_size = 0.4
```

 Record the R² score for each experiment.

 Create a table:

 | Test Size | R² |
| --- | --- |
| 0.1 | ? |
| 0.2 | ? |
| 0.3 | ? |
| 0.4 | ? |

 Discuss what you observe.

---

 ## Challenge 3 — Change Random State

 Run the same experiment using:

```
random_state = 0
random_state = 10
random_state = 42
random_state = 100
```

 Compare the results.

 Answer:

 > Why can the evaluation score change when the random state changes?

---

 ## Challenge 4 — Detect Overfitting

 Train a model and compare:

```
Training Score
Testing Score
```

 Answer:

 > What would you suspect if the training score is extremely high but the testing score is much lower?

---

 # 🧠 Concept Check

 Before moving to the next topic, make sure you can explain these terms without looking at your notes:

 - Machine Learning
- Artificial Intelligence
- Deep Learning
- Dataset
- Sample
- Feature
- Label
- Target
- Model
- Training
- Testing
- Prediction
- Generalization
- Overfitting
- Underfitting
- Parameter
- Hyperparameter
- Regression
- Classification
- Model Evaluation

---

 # ✅ Completion Checklist

 Mark each item when you can do it independently:

 - [ ] Explain Machine Learning.
- [ ] Explain traditional programming.
- [ ] Compare traditional programming and ML.
- [ ] Identify features.
- [ ] Identify labels.
- [ ] Create a NumPy dataset.
- [ ] Split a dataset into training and testing sets.
- [ ] Create a Scikit-Learn model.
- [ ] Train a model.
- [ ] Make predictions.
- [ ] Predict new data.
- [ ] Calculate MAE.
- [ ] Calculate MSE.
- [ ] Calculate RMSE.
- [ ] Calculate R².
- [ ] Explain generalization.
- [ ] Explain overfitting.
- [ ] Build a complete beginner ML workflow.
- [ ] Complete the house-price mini project.

---

 # 🚀 Next Step

 After completing these exercises, continue to:

```
01-ML-Fundamentals/
└── 02-AI-vs-ML-vs-DL/
```

 The next topic focuses on understanding the relationship between:

```
Artificial Intelligence
        ↓
Machine Learning
        ↓
Deep Learning
        ↓
Neural Networks
```

---

 ## Author

 **KISHOR KAKDE**

 Machine Learning | Python | Data Science | AI
