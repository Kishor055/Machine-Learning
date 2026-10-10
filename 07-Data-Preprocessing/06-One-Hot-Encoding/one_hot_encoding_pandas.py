"""
One-Hot Encoding Using Pandas
Machine Learning - Data Preprocessing

This script demonstrates:
1. Creating a student dataset.
2. Identifying categorical features.
3. Applying One-Hot Encoding using pd.get_dummies().
4. Comparing original and encoded datasets.
5. Handling multiple categorical columns.
6. Saving the processed dataset as a CSV file.
"""

import pandas as pd


# ============================================================
# 1. CREATE A SAMPLE STUDENT DATASET
# ============================================================

data = {
    "Student_ID": [101, 102, 103, 104, 105, 106, 107, 108],
    "Name": [
        "Amit", "Priya", "Rahul", "Sneha",
        "Karan", "Neha", "Arjun", "Pooja"
    ],
    "City": [
        "Pune", "Mumbai", "Delhi", "Pune",
        "Mumbai", "Delhi", "Nagpur", "Pune"
    ],
    "Department": [
        "Computer Science", "Mechanical", "Electrical",
        "Computer Science", "Civil", "Mechanical",
        "Electrical", "Computer Science"
    ],
    "Gender": [
        "Male", "Female", "Male", "Female",
        "Male", "Female", "Male", "Female"
    ],
    "Marks": [85, 92, 78, 88, 76, 90, 82, 95],
    "Attendance": [90, 95, 80, 92, 75, 96, 85, 98]
}

df = pd.DataFrame(data)

print("=" * 70)
print("ORIGINAL STUDENT DATASET")
print("=" * 70)
print(df)

print("\nOriginal Dataset Shape:", df.shape)


# ============================================================
# 2. IDENTIFY CATEGORICAL COLUMNS
# ============================================================

categorical_columns = ["City", "Department", "Gender"]

print("\n" + "=" * 70)
print("CATEGORICAL COLUMNS")
print("=" * 70)

print(categorical_columns)

print("\nUnique Categories:")
for column in categorical_columns:
    print(f"{column}: {df[column].unique()}")


# ============================================================
# 3. ONE-HOT ENCODING A SINGLE COLUMN
# ============================================================

city_encoded = pd.get_dummies(
    df["City"],
    prefix="City",
    dtype=int
)

print("\n" + "=" * 70)
print("ONE-HOT ENCODING: CITY")
print("=" * 70)
print(city_encoded)


# ============================================================
# 4. ONE-HOT ENCODING MULTIPLE COLUMNS
# ============================================================

encoded_df = pd.get_dummies(
    df,
    columns=categorical_columns,
    dtype=int
)

print("\n" + "=" * 70)
print("DATASET AFTER ONE-HOT ENCODING")
print("=" * 70)
print(encoded_df)

print("\nEncoded Dataset Shape:", encoded_df.shape)


# ============================================================
# 5. ONE-HOT ENCODING WITH drop_first=True
# ============================================================

encoded_df_drop_first = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

print("\n" + "=" * 70)
print("ONE-HOT ENCODING WITH FIRST CATEGORY DROPPED")
print("=" * 70)
print(encoded_df_drop_first)

print("\nDataset Shape After Dropping First Categories:")
print(encoded_df_drop_first.shape)


# ============================================================
# 6. SELECT ONLY CATEGORICAL FEATURES
# ============================================================

categorical_df = pd.get_dummies(
    df[categorical_columns],
    dtype=int
)

print("\n" + "=" * 70)
print("ENCODED CATEGORICAL FEATURES ONLY")
print("=" * 70)
print(categorical_df)


# ============================================================
# 7. DISPLAY DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("Original Dataset Shape:", df.shape)
print("Encoded Dataset Shape:", encoded_df.shape)

print("\nEncoded Column Names:")
print(encoded_df.columns.tolist())

print("\nData Types:")
print(encoded_df.dtypes)

print("\nMissing Values:")
print(encoded_df.isnull().sum())


# ============================================================
# 8. SAVE THE ENCODED DATASET
# ============================================================

output_file = "one_hot_encoded_students_pandas.csv"

encoded_df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("DATASET EXPORT")
print("=" * 70)

print(f"Encoded dataset saved successfully: {output_file}")


# ============================================================
# 9. ENCODE NEW STUDENT DATA
# ============================================================

new_students = pd.DataFrame({
    "Student_ID": [109, 110],
    "Name": ["Rohan", "Ananya"],
    "City": ["Pune", "Mumbai"],
    "Department": ["Computer Science", "Mechanical"],
    "Gender": ["Male", "Female"],
    "Marks": [87, 91],
    "Attendance": [89, 94]
})

new_students_encoded = pd.get_dummies(
    new_students,
    columns=categorical_columns,
    dtype=int
)

# Align the new dataset with the original encoded columns.
# Any categories absent from the new batch receive zeros.
new_students_encoded = new_students_encoded.reindex(
    columns=encoded_df.columns,
    fill_value=0
)

print("\n" + "=" * 70)
print("NEW STUDENTS - ENCODED DATA")
print("=" * 70)
print(new_students_encoded)


# ============================================================
# 10. KEY TAKEAWAYS
# ============================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("1. pd.get_dummies() converts categorical values into binary columns.")
print("2. The columns parameter selects which features to encode.")
print("3. The dtype=int parameter produces integer indicator columns.")
print("4. drop_first=True removes the first category per encoded feature.")
print("5. reindex() aligns columns when processing new data.")
print("6. For ML pipelines, consider sklearn OneHotEncoder to learn and")
print("   reuse categories consistently during training and inference.")

# ============================================================
# END OF SCRIPT
# ============================================================
