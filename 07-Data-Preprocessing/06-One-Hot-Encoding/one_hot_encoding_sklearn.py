"""
One-Hot Encoding Using Scikit-learn
Machine Learning - Data Preprocessing

Topics Covered:
1. Creating a student dataset
2. Identifying categorical features
3. Applying OneHotEncoder
4. Encoding multiple categorical columns
5. Train-test split
6. Handling unknown categories
7. Transforming new student records
8. Saving encoded datasets
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 1. CREATE A SAMPLE STUDENT DATASET
# ============================================================

data = {
    "Student_ID": range(101, 121),
    "Name": [
        "Amit", "Priya", "Rahul", "Sneha", "Karan",
        "Neha", "Arjun", "Pooja", "Rohan", "Ananya",
        "Vikas", "Meera", "Akash", "Riya", "Sahil",
        "Isha", "Varun", "Kavya", "Nikhil", "Aarti"
    ],
    "City": [
        "Pune", "Mumbai", "Delhi", "Pune", "Mumbai",
        "Delhi", "Nagpur", "Pune", "Mumbai", "Delhi",
        "Pune", "Nagpur", "Mumbai", "Delhi", "Pune",
        "Mumbai", "Nagpur", "Pune", "Delhi", "Mumbai"
    ],
    "Department": [
        "Computer Science", "Mechanical", "Electrical",
        "Computer Science", "Civil", "Mechanical",
        "Electrical", "Computer Science", "Civil",
        "Mechanical", "Electrical", "Computer Science",
        "Civil", "Electrical", "Mechanical",
        "Computer Science", "Civil", "Mechanical",
        "Electrical", "Computer Science"
    ],
    "Gender": [
        "Male", "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male", "Female",
        "Male", "Female", "Male", "Female", "Male",
        "Female", "Male", "Female", "Male", "Female"
    ],
    "Marks": [
        85, 92, 78, 88, 76, 90, 82, 95, 81, 89,
        74, 93, 86, 79, 91, 84, 77, 96, 83, 87
    ],
    "Attendance": [
        90, 95, 80, 92, 75, 96, 85, 98, 88, 94,
        78, 97, 89, 82, 93, 86, 79, 99, 84, 91
    ]
}

df = pd.DataFrame(data)

print("=" * 70)
print("ORIGINAL STUDENT DATASET")
print("=" * 70)
print(df)

print("\nOriginal Dataset Shape:", df.shape)


# ============================================================
# 2. SELECT CATEGORICAL FEATURES AND TARGET
# ============================================================

categorical_columns = ["City", "Department", "Gender"]
target_column = "Marks"

X = df[categorical_columns]
y = df[target_column]

print("\nCategorical Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())


# ============================================================
# 3. SPLIT THE DATA INTO TRAINING AND TESTING SETS
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)

print("Training Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)


# ============================================================
# 4. CREATE THE ONE-HOT ENCODER
# ============================================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False,
    dtype=int
)


# ============================================================
# 5. FIT AND TRANSFORM TRAINING DATA
# ============================================================

X_train_encoded_array = encoder.fit_transform(X_train)

encoded_columns = encoder.get_feature_names_out(
    categorical_columns
)

X_train_encoded = pd.DataFrame(
    X_train_encoded_array,
    columns=encoded_columns,
    index=X_train.index
)

print("\n" + "=" * 70)
print("ENCODED TRAINING DATA")
print("=" * 70)

print(X_train_encoded.head())

print("\nEncoded Training Shape:", X_train_encoded.shape)


# ============================================================
# 6. TRANSFORM TESTING DATA
# ============================================================

# Reuse the encoder fitted on training data.
X_test_encoded_array = encoder.transform(X_test)

X_test_encoded = pd.DataFrame(
    X_test_encoded_array,
    columns=encoded_columns,
    index=X_test.index
)

print("\n" + "=" * 70)
print("ENCODED TESTING DATA")
print("=" * 70)

print(X_test_encoded.head())

print("\nEncoded Testing Shape:", X_test_encoded.shape)


# ============================================================
# 7. DISPLAY LEARNED CATEGORIES
# ============================================================

print("\n" + "=" * 70)
print("CATEGORIES LEARNED BY THE ENCODER")
print("=" * 70)

for column, categories in zip(
    categorical_columns,
    encoder.categories_
):
    print(f"{column}: {categories.tolist()}")

print("\nGenerated Feature Names:")
print(encoded_columns.tolist())


# ============================================================
# 8. COMBINE ENCODED FEATURES WITH TARGET
# ============================================================

train_processed = X_train_encoded.copy()
train_processed[target_column] = y_train

test_processed = X_test_encoded.copy()
test_processed[target_column] = y_test

print("\n" + "=" * 70)
print("PROCESSED TRAINING DATA")
print("=" * 70)
print(train_processed.head())

print("\n" + "=" * 70)
print("PROCESSED TESTING DATA")
print("=" * 70)
print(test_processed.head())


# ============================================================
# 9. SAVE ENCODED DATASETS
# ============================================================

train_processed.to_csv(
    "one_hot_encoded_train.csv",
    index=False
)

test_processed.to_csv(
    "one_hot_encoded_test.csv",
    index=False
)

print("\nTraining dataset saved: one_hot_encoded_train.csv")
print("Testing dataset saved: one_hot_encoded_test.csv")


# ============================================================
# 10. TRANSFORM NEW STUDENT RECORDS
# ============================================================

new_students = pd.DataFrame({
    "City": ["Pune", "Mumbai"],
    "Department": ["Computer Science", "Mechanical"],
    "Gender": ["Male", "Female"]
})

new_students_encoded_array = encoder.transform(new_students)

new_students_encoded = pd.DataFrame(
    new_students_encoded_array,
    columns=encoded_columns
)

print("\n" + "=" * 70)
print("NEW STUDENT RECORDS")
print("=" * 70)
print(new_students)

print("\nEncoded New Student Records:")
print(new_students_encoded)


# ============================================================
# 11. HANDLE AN UNKNOWN CATEGORY
# ============================================================

unknown_student = pd.DataFrame({
    "City": ["Chennai"],
    "Department": ["Computer Science"],
    "Gender": ["Male"]
})

unknown_encoded_array = encoder.transform(unknown_student)

unknown_encoded_df = pd.DataFrame(
    unknown_encoded_array,
    columns=encoded_columns
)

print("\n" + "=" * 70)
print("UNKNOWN CATEGORY HANDLING")
print("=" * 70)

print("Original Record:")
print(unknown_student)

print("\nEncoded Record:")
print(unknown_encoded_df)

print(
    "\nNote: Chennai was not necessarily present in the training data. "
    "With handle_unknown='ignore', its City features are all zeros."
)


# ============================================================
# 12. DATASET VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET VALIDATION")
print("=" * 70)

print("Original Dataset Shape:", df.shape)
print("Training Encoded Shape:", X_train_encoded.shape)
print("Testing Encoded Shape:", X_test_encoded.shape)

print("\nMissing Values in Training Data:")
print(X_train_encoded.isnull().sum())

print("\nMissing Values in Testing Data:")
print(X_test_encoded.isnull().sum())

print("\nAll training and testing feature columns match:")
print(X_train_encoded.columns.equals(X_test_encoded.columns))


# ============================================================
# END OF SCRIPT
# ============================================================

