"""
Conditional Probability — Probability for Machine Learning
==========================================================

Conditional probability measures the probability of an event occurring
given that another event is already known to have occurred.

## Core Formula

```
             P(A ∩ B)
P(A | B) = -----------
                P(B)
```

Where:

```
P(A | B)  = Probability of A given B
P(B | A)  = Probability of B given A
P(A ∩ B)  = Probability that A and B both occur
P(B)      = Probability of B
```

Conditional probability is one of the most important concepts in
probability and Machine Learning.

It forms the mathematical foundation for:

```
- Bayes' Theorem
- Naive Bayes
- Bayesian inference
- Classification
- Probabilistic graphical models
- Medical diagnosis
- Spam detection
- Risk estimation
- Decision making under uncertainty
```

## Learning Progression

```
1. Basic conditional probability
2. Probability validation
3. Intersection and multiplication rule
4. Conditional probability tables
5. Frequency/count-based probability
6. Independence
7. Dependence
8. Law of total probability
9. Bayes' theorem connection
10. Multiple events
11. Sequential conditional probability
12. Joint distributions
13. NumPy implementation
14. Machine Learning applications
15. Common mistakes
```

Author: Machine Learning Mathematics Repository
"""

# =============================================================================

# 1. IMPORTS

# =============================================================================

from **future** import annotations

import math
from typing import Dict, Mapping, Sequence

# =============================================================================

# 2. BASIC PROBABILITY VALIDATION

# =============================================================================

def validate_probability(
probability: float,
name: str = "probability",
) -> None:
"""
Validate that a value is a valid probability.

```
A probability must satisfy:

    0 <= P <= 1
"""

if not isinstance(probability, (int, float)):
    raise TypeError(f"{name} must be a number.")

if not math.isfinite(probability):
    raise ValueError(f"{name} must be finite.")

if not 0 <= probability <= 1:
    raise ValueError(
        f"{name} must be between 0 and 1. "
        f"Received: {probability}"
    )
```

# =============================================================================

# 3. CONDITIONAL PROBABILITY

# =============================================================================

def conditional_probability(
intersection: float,
condition: float,
) -> float:
"""
Calculate conditional probability.

```
Formula:

             P(A ∩ B)
    P(A | B) = ---------
                P(B)

Parameters
----------
intersection:
    P(A ∩ B), probability that both A and B occur.

condition:
    P(B), probability of the condition.

Returns
-------
float
    P(A | B)

Raises
------
ValueError
    If P(B) is zero or the supplied probabilities are invalid.
"""

validate_probability(intersection, "intersection")
validate_probability(condition, "condition")

if condition == 0:
    raise ValueError(
        "Conditional probability is undefined when "
        "P(B) = 0."
    )

if intersection > condition:
    raise ValueError(
        "For valid events, P(A ∩ B) cannot exceed P(B)."
    )

return intersection / condition
```

# =============================================================================

# 4. REVERSE CONDITIONAL PROBABILITY

# =============================================================================

def reverse_conditional_probability(
intersection: float,
event_a: float,
) -> float:
"""
Calculate P(B | A).

```
Formula:

             P(A ∩ B)
    P(B | A) = ---------
                P(A)
"""

validate_probability(intersection, "intersection")
validate_probability(event_a, "event_a")

if event_a == 0:
    raise ValueError(
        "Conditional probability is undefined when P(A) = 0."
    )

if intersection > event_a:
    raise ValueError(
        "P(A ∩ B) cannot exceed P(A)."
    )

return intersection / event_a
```

# =============================================================================

# 5. MULTIPLICATION RULE

# =============================================================================

def joint_probability(
probability_a: float,
probability_b_given_a: float,
) -> float:
"""
Calculate joint probability using the multiplication rule.

```
Formula:

    P(A ∩ B)
    =
    P(A)P(B | A)
"""

validate_probability(probability_a, "probability_a")
validate_probability(
    probability_b_given_a,
    "probability_b_given_a",
)

return probability_a * probability_b_given_a
```

# =============================================================================

# 6. INDEPENDENCE CHECK

# =============================================================================

def are_independent(
probability_a: float,
probability_b: float,
joint: float,
tolerance: float = 1e-9,
) -> bool:
"""
Check whether two events satisfy the independence condition.

```
Independent events satisfy:

    P(A ∩ B) = P(A)P(B)

Parameters
----------
probability_a:
    P(A)

probability_b:
    P(B)

joint:
    P(A ∩ B)

tolerance:
    Numerical comparison tolerance.
"""

validate_probability(probability_a, "probability_a")
validate_probability(probability_b, "probability_b")
validate_probability(joint, "joint")

if tolerance < 0:
    raise ValueError("Tolerance cannot be negative.")

return math.isclose(
    joint,
    probability_a * probability_b,
    rel_tol=tolerance,
    abs_tol=tolerance,
)
```

# =============================================================================

# 7. CONDITIONAL PROBABILITY FROM COUNTS

# =============================================================================

def conditional_probability_from_counts(
intersection_count: int,
condition_count: int,
) -> float:
"""
Calculate conditional probability using observed counts.

```
Formula:

                 Count(A ∩ B)
    P(A | B) = ----------------
                   Count(B)

Example
-------
If:

    30 customers purchased
    20 of those customers used a coupon

Then:

    P(Purchased | Coupon)

depends on the appropriate definition of the events and counts.
"""

if not isinstance(intersection_count, int):
    raise TypeError("intersection_count must be an integer.")

if not isinstance(condition_count, int):
    raise TypeError("condition_count must be an integer.")

if intersection_count < 0:
    raise ValueError(
        "intersection_count cannot be negative."
    )

if condition_count <= 0:
    raise ValueError(
        "condition_count must be greater than zero."
    )

if intersection_count > condition_count:
    raise ValueError(
        "intersection_count cannot exceed condition_count."
    )

return intersection_count / condition_count
```

# =============================================================================

# 8. CONDITIONAL PROBABILITY FROM A 2x2 TABLE

# =============================================================================

def conditional_probability_from_table(
table: Mapping[str, Mapping[str, int]],
event: str,
condition: str,
) -> float:
"""
Calculate conditional probability from a two-variable count table.

```
Expected structure:

    {
        "event_a": {
            "condition_b": count,
            "not_condition_b": count,
        },
        "not_event_a": {
            "condition_b": count,
            "not_condition_b": count,
        }
    }

The function calculates:

    P(event | condition)

Parameters
----------
table:
    Nested dictionary containing counts.

event:
    Name of the event row.

condition:
    Name of the condition column.
"""

if event not in table:
    raise KeyError(f"Event '{event}' not found in table.")

for row_name, row in table.items():
    if condition not in row:
        raise KeyError(
            f"Condition '{condition}' not found in row '{row_name}'."
        )

condition_count = sum(
    row[condition]
    for row in table.values()
)

if condition_count == 0:
    raise ValueError(
        "The condition has zero observations."
    )

intersection_count = table[event][condition]

return intersection_count / condition_count
```

# =============================================================================

# 9. CONDITIONAL PROBABILITY TABLE

# =============================================================================

def conditional_distribution(
joint_distribution: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
"""
Convert a joint probability table into conditional probabilities.

```
For every condition B:

    P(A | B)
    =
    P(A,B) / P(B)

Parameters
----------
joint_distribution:
    Nested mapping representing joint probabilities.

Returns
-------
dict
    Conditional distributions.
"""

if not joint_distribution:
    raise ValueError("Joint distribution cannot be empty.")

for row_name, row in joint_distribution.items():
    for column_name, probability in row.items():
        validate_probability(
            probability,
            f"P({row_name}, {column_name})",
        )

columns = set()

for row in joint_distribution.values():
    columns.update(row.keys())

result: Dict[str, Dict[str, float]] = {}

for column in columns:
    column_probability = sum(
        row.get(column, 0.0)
        for row in joint_distribution.values()
    )

    if column_probability == 0:
        raise ValueError(
            f"Condition '{column}' has probability zero."
        )

    result[column] = {
        row_name: row.get(column, 0.0) / column_probability
        for row_name, row in joint_distribution.items()
    }

return result
```

# =============================================================================

# 10. LAW OF TOTAL PROBABILITY

# =============================================================================

def total_probability(
probabilities_of_conditions: Sequence[float],
conditional_probabilities: Sequence[float],
) -> float:
"""
Calculate total probability.

```
Formula:

    P(B)
    =
    Σ P(B | Ai)P(Ai)

where the Ai form a partition of the sample space.

Parameters
----------
probabilities_of_conditions:
    P(A1), P(A2), ..., P(An)

conditional_probabilities:
    P(B|A1), P(B|A2), ..., P(B|An)
"""

if len(probabilities_of_conditions) == 0:
    raise ValueError("At least one condition is required.")

if len(probabilities_of_conditions) != len(
    conditional_probabilities
):
    raise ValueError(
        "Both sequences must have the same length."
    )

for index, probability in enumerate(
    probabilities_of_conditions
):
    validate_probability(
        probability,
        f"condition_probability[{index}]",
    )

for index, probability in enumerate(
    conditional_probabilities
):
    validate_probability(
        probability,
        f"conditional_probability[{index}]",
    )

total_conditions = sum(probabilities_of_conditions)

if not math.isclose(
    total_conditions,
    1.0,
    rel_tol=1e-9,
    abs_tol=1e-9,
):
    raise ValueError(
        "Condition probabilities must sum to 1."
    )

return sum(
    p_condition * p_event_given_condition
    for p_condition, p_event_given_condition
    in zip(
        probabilities_of_conditions,
        conditional_probabilities,
    )
)
```

# =============================================================================

# 11. BAYES' THEOREM USING CONDITIONAL PROBABILITY

# =============================================================================

def bayes_theorem(
probability_b_given_a: float,
probability_a: float,
probability_b: float,
) -> float:
"""
Calculate P(A | B) using Bayes' theorem.

```
Formula:

                 P(B | A)P(A)
    P(A | B) = -----------------
                     P(B)

This function demonstrates how Bayes' theorem is built directly
from conditional probability and the multiplication rule.
"""

validate_probability(
    probability_b_given_a,
    "probability_b_given_a",
)
validate_probability(probability_a, "probability_a")
validate_probability(probability_b, "probability_b")

if probability_b == 0:
    raise ValueError(
        "P(B) cannot be zero."
    )

return (
    probability_b_given_a
    * probability_a
    / probability_b
)
```

# =============================================================================

# 12. SEQUENTIAL CONDITIONAL PROBABILITY

# =============================================================================

def sequential_probability(
first_probability: float,
second_probability_given_first: float,
third_probability_given_first_two: float,
) -> float:
"""
Calculate:

```
    P(A ∩ B ∩ C)

using the chain rule:

    P(A,B,C)
    =
    P(A)
    P(B|A)
    P(C|A,B)
"""

validate_probability(
    first_probability,
    "first_probability",
)

validate_probability(
    second_probability_given_first,
    "second_probability_given_first",
)

validate_probability(
    third_probability_given_first_two,
    "third_probability_given_first_two",
)

return (
    first_probability
    * second_probability_given_first
    * third_probability_given_first_two
)
```

# =============================================================================

# 13. CHAIN RULE FOR MULTIPLE EVENTS

# =============================================================================

def chain_rule(
probabilities: Sequence[float],
) -> float:
"""
Calculate a joint probability from sequential conditional terms.

```
For example:

    P(A,B,C,D)

    =
    P(A)
    P(B|A)
    P(C|A,B)
    P(D|A,B,C)

The caller supplies these probability terms in order.
"""

if not probabilities:
    raise ValueError(
        "At least one probability is required."
    )

for index, probability in enumerate(probabilities):
    validate_probability(
        probability,
        f"probabilities[{index}]",
    )

result = 1.0

for probability in probabilities:
    result *= probability

return result
```

# =============================================================================

# 14. CONDITIONAL INDEPENDENCE

# =============================================================================

def check_conditional_independence(
joint_given_condition: float,
a_given_condition: float,
b_given_condition: float,
tolerance: float = 1e-9,
) -> bool:
"""
Check conditional independence of A and B given C.

```
A and B are conditionally independent given C if:

    P(A ∩ B | C)
    =
    P(A | C)P(B | C)

This concept is fundamental to Naive Bayes and probabilistic
graphical models.
"""

validate_probability(
    joint_given_condition,
    "joint_given_condition",
)

validate_probability(
    a_given_condition,
    "a_given_condition",
)

validate_probability(
    b_given_condition,
    "b_given_condition",
)

if tolerance < 0:
    raise ValueError(
        "Tolerance cannot be negative."
    )

expected_joint = (
    a_given_condition
    * b_given_condition
)

return math.isclose(
    joint_given_condition,
    expected_joint,
    rel_tol=tolerance,
    abs_tol=tolerance,
)
```

# =============================================================================

# 15. DATASET CONDITIONAL PROBABILITY

# =============================================================================

def categorical_conditional_probability(
data: Sequence[Mapping[str, object]],
target_column: str,
target_value: object,
condition_column: str,
condition_value: object,
) -> float:
"""
Calculate conditional probability from categorical records.

```
Calculates:

    P(target = target_value |
      condition = condition_value)

Example:

    P(Purchase = Yes | Gender = Female)

This is a simple educational implementation of conditional
probability from tabular data.
"""

matching_condition = [
    row
    for row in data
    if row.get(condition_column) == condition_value
]

if not matching_condition:
    raise ValueError(
        "No records satisfy the conditioning event."
    )

matching_both = [
    row
    for row in matching_condition
    if row.get(target_column) == target_value
]

return len(matching_both) / len(matching_condition)
```

# =============================================================================

# 16. CONDITIONAL PROBABILITY FOR CLASSIFICATION

# =============================================================================

def class_conditional_probabilities(
data: Sequence[Mapping[str, object]],
feature: str,
feature_value: object,
target: str,
) -> Dict[object, float]:
"""
Estimate:

```
    P(feature = value | class)

for each class in a dataset.

This is a simplified educational building block for
understanding Naive Bayes classification.
"""

if not data:
    raise ValueError("Dataset cannot be empty.")

classes = {
    row[target]
    for row in data
    if target in row
}

if not classes:
    raise ValueError(
        f"Target column '{target}' was not found."
    )

result: Dict[object, float] = {}

for class_value in classes:
    class_records = [
        row
        for row in data
        if row.get(target) == class_value
    ]

    if not class_records:
        continue

    matching_records = [
        row
        for row in class_records
        if row.get(feature) == feature_value
    ]

    result[class_value] = (
        len(matching_records)
        / len(class_records)
    )

return result
```

# =============================================================================

# 17. NUMPY IMPLEMENTATION

# =============================================================================

def numpy_conditional_probability_example() -> None:
"""
Demonstrate conditional probability with NumPy.

```
NumPy is optional.
"""

print("\n" + "=" * 72)
print("NUMPY CONDITIONAL PROBABILITY")
print("=" * 72)

try:
    import numpy as np
except ImportError:
    print("NumPy is not installed.")
    print("Install it with: pip install numpy")
    return

# Example observations:
#
# Feature:
#   1 = purchased
#   0 = did not purchase
#
# Condition:
#   1 = saw advertisement
#   0 = did not see advertisement

purchased = np.array(
    [1, 1, 0, 1, 0, 1, 0, 0, 1, 1]
)

saw_ad = np.array(
    [1, 1, 1, 1, 0, 0, 0, 1, 1, 0]
)

condition_mask = saw_ad == 1

purchase_given_ad = np.mean(
    purchased[condition_mask] == 1
)

print(
    "P(Purchase = 1 | Saw Ad = 1) = "
    f"{purchase_given_ad:.2%}"
)
```

# =============================================================================

# 18. PRACTICAL EXAMPLE — CUSTOMER PURCHASE

# =============================================================================

def customer_purchase_example() -> None:
"""
Demonstrate conditional probability using customer behavior.
"""

```
print("\n" + "=" * 72)
print("CUSTOMER PURCHASE EXAMPLE")
print("=" * 72)

data = [
    {"age_group": "young", "purchased": "yes"},
    {"age_group": "young", "purchased": "no"},
    {"age_group": "young", "purchased": "yes"},
    {"age_group": "adult", "purchased": "yes"},
    {"age_group": "adult", "purchased": "no"},
    {"age_group": "adult", "purchased": "yes"},
    {"age_group": "adult", "purchased": "no"},
    {"age_group": "senior", "purchased": "yes"},
    {"age_group": "senior", "purchased": "no"},
    {"age_group": "senior", "purchased": "no"},
]

probability = categorical_conditional_probability(
    data=data,
    target_column="purchased",
    target_value="yes",
    condition_column="age_group",
    condition_value="young",
)

print(
    "P(Purchased = Yes | Age Group = Young) = "
    f"{probability:.2%}"
)
```

# =============================================================================

# 19. MEDICAL EXAMPLE

# =============================================================================

def medical_conditional_probability_example() -> None:
"""
Demonstrate conditional probability using a hypothetical
medical screening scenario.

```
This is a mathematical example only and is not medical advice.
"""

print("\n" + "=" * 72)
print("MEDICAL CONDITIONAL PROBABILITY EXAMPLE")
print("=" * 72)

# Suppose:
#
# 100 people test positive.
# 8 actually have the condition.
#
# P(Condition | Positive)
# = 8 / 100

positive_tests = 100
condition_and_positive = 8

probability = conditional_probability_from_counts(
    intersection_count=condition_and_positive,
    condition_count=positive_tests,
)

print(
    "P(Condition | Positive Test) = "
    f"{probability:.2%}"
)

print(
    "\nNote: In real diagnostic interpretation, "
    "the population prevalence and test characteristics "
    "must also be considered."
)
```

# =============================================================================

# 20. COMMON MISTAKES

# =============================================================================

def print_common_mistakes() -> None:
"""Print common conditional probability mistakes."""

```
print("\n" + "=" * 72)
print("COMMON MISTAKES")
print("=" * 72)

mistakes = [
    (
        "1. Reversing the condition",
        "P(A|B) is generally not equal to P(B|A).",
    ),
    (
        "2. Using the wrong denominator",
        "P(A|B) divides by P(B), not P(A).",
    ),
    (
        "3. Ignoring zero-probability conditions",
        "P(A|B) is undefined when P(B) = 0.",
    ),
    (
        "4. Confusing independence with exclusivity",
        "Independent and mutually exclusive are different concepts.",
    ),
    (
        "5. Assuming independence without evidence",
        "Independence is a mathematical/modeling condition.",
    ),
    (
        "6. Ignoring the population",
        "Conditional probabilities depend on the conditioning group.",
    ),
    (
        "7. Confusing association with causation",
        "Conditional probability measures probability, not causality.",
    ),
]

for title, explanation in mistakes:
    print(f"\n{title}")
    print(f"   {explanation}")
```

# =============================================================================

# 21. COMPLETE ANALYSIS

# =============================================================================

def complete_conditional_analysis(
probability_a: float,
probability_b: float,
joint_probability_value: float,
) -> Dict[str, float | bool]:
"""
Perform a complete conditional probability analysis.

```
Calculates:

    P(A | B)
    P(B | A)
    P(A ∩ B)
    P(A)P(B)
    Independence
"""

validate_probability(probability_a, "probability_a")
validate_probability(probability_b, "probability_b")
validate_probability(
    joint_probability_value,
    "joint_probability",
)

if joint_probability_value > probability_a:
    raise ValueError(
        "Joint probability cannot exceed P(A)."
    )

if joint_probability_value > probability_b:
    raise ValueError(
        "Joint probability cannot exceed P(B)."
    )

p_a_given_b = conditional_probability(
    intersection=joint_probability_value,
    condition=probability_b,
)

p_b_given_a = reverse_conditional_probability(
    intersection=joint_probability_value,
    event_a=probability_a,
)

independent = are_independent(
    probability_a=probability_a,
    probability_b=probability_b,
    joint=joint_probability_value,
)

return {
    "P(A)": probability_a,
    "P(B)": probability_b,
    "P(A ∩ B)": joint_probability_value,
    "P(A | B)": p_a_given_b,
    "P(B | A)": p_b_given_a,
    "P(A)P(B)": probability_a * probability_b,
    "independent": independent,
}
```

# =============================================================================

# 22. MAIN DEMONSTRATION

# =============================================================================

def main() -> None:
"""
Run conditional probability demonstrations from beginner
to advanced concepts.
"""

```
print("=" * 72)
print("CONDITIONAL PROBABILITY — PROBABILITY FOR MACHINE LEARNING")
print("=" * 72)

# -------------------------------------------------------------------------
# Basic Example
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("1. BASIC CONDITIONAL PROBABILITY")
print("=" * 72)

p_a_and_b = 0.20
p_b = 0.50

p_a_given_b = conditional_probability(
    intersection=p_a_and_b,
    condition=p_b,
)

print(f"P(A ∩ B): {p_a_and_b:.2%}")
print(f"P(B):     {p_b:.2%}")
print(f"P(A | B): {p_a_given_b:.2%}")

# -------------------------------------------------------------------------
# Reverse Conditional Probability
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("2. REVERSE CONDITIONAL PROBABILITY")
print("=" * 72)

p_a = 0.40

p_b_given_a = reverse_conditional_probability(
    intersection=p_a_and_b,
    event_a=p_a,
)

print(f"P(A ∩ B): {p_a_and_b:.2%}")
print(f"P(A):     {p_a:.2%}")
print(f"P(B | A): {p_b_given_a:.2%}")

# -------------------------------------------------------------------------
# Multiplication Rule
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("3. MULTIPLICATION RULE")
print("=" * 72)

calculated_joint = joint_probability(
    probability_a=0.40,
    probability_b_given_a=0.50,
)

print(
    "P(A ∩ B) = P(A) × P(B | A)"
)
print(
    f"P(A ∩ B) = {calculated_joint:.2%}"
)

# -------------------------------------------------------------------------
# Independence
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("4. INDEPENDENCE")
print("=" * 72)

independent = are_independent(
    probability_a=0.50,
    probability_b=0.40,
    joint=0.20,
)

print(
    "P(A ∩ B) = P(A) × P(B)"
)
print(f"Independent: {independent}")

# -------------------------------------------------------------------------
# Count-Based Probability
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("5. COUNT-BASED CONDITIONAL PROBABILITY")
print("=" * 72)

count_probability = conditional_probability_from_counts(
    intersection_count=30,
    condition_count=50,
)

print(
    "P(A | B) = 30 / 50 = "
    f"{count_probability:.2%}"
)

# -------------------------------------------------------------------------
# 2x2 Table
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("6. CONDITIONAL PROBABILITY FROM A TABLE")
print("=" * 72)

table = {
    "Purchased": {
        "Saw_Ad": 30,
        "Did_Not_See_Ad": 10,
    },
    "Not_Purchased": {
        "Saw_Ad": 20,
        "Did_Not_See_Ad": 40,
    },
}

probability = conditional_probability_from_table(
    table=table,
    event="Purchased",
    condition="Saw_Ad",
)

print(
    "P(Purchased | Saw Ad) = "
    f"{probability:.2%}"
)

# -------------------------------------------------------------------------
# Law of Total Probability
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("7. LAW OF TOTAL PROBABILITY")
print("=" * 72)

p_b = total_probability(
    probabilities_of_conditions=[
        0.60,
        0.40,
    ],
    conditional_probabilities=[
        0.80,
        0.20,
    ],
)

print(
    "P(B) = P(B|A)P(A) + P(B|not A)P(not A)"
)

print(f"P(B) = {p_b:.2%}")

# -------------------------------------------------------------------------
# Bayes' Theorem
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("8. BAYES' THEOREM")
print("=" * 72)

posterior = bayes_theorem(
    probability_b_given_a=0.80,
    probability_a=0.60,
    probability_b=p_b,
)

print(
    "P(A | B) = "
    f"{posterior:.2%}"
)

# -------------------------------------------------------------------------
# Sequential Probability
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("9. SEQUENTIAL CONDITIONAL PROBABILITY")
print("=" * 72)

probability_abc = sequential_probability(
    first_probability=0.50,
    second_probability_given_first=0.60,
    third_probability_given_first_two=0.70,
)

print(
    "P(A,B,C) = "
    f"{probability_abc:.2%}"
)

# -------------------------------------------------------------------------
# Chain Rule
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("10. CHAIN RULE")
print("=" * 72)

probability_abcd = chain_rule(
    probabilities=[
        0.80,
        0.70,
        0.60,
        0.50,
    ]
)

print(
    "P(A,B,C,D) = "
    f"{probability_abcd:.2%}"
)

# -------------------------------------------------------------------------
# Conditional Independence
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("11. CONDITIONAL INDEPENDENCE")
print("=" * 72)

conditionally_independent = check_conditional_independence(
    joint_given_condition=0.24,
    a_given_condition=0.60,
    b_given_condition=0.40,
)

print(
    "P(A,B|C) = P(A|C)P(B|C)"
)

print(
    f"Conditionally independent: "
    f"{conditionally_independent}"
)

# -------------------------------------------------------------------------
# Customer Example
# -------------------------------------------------------------------------

customer_purchase_example()

# -------------------------------------------------------------------------
# Medical Example
# -------------------------------------------------------------------------

medical_conditional_probability_example()

# -------------------------------------------------------------------------
# NumPy Example
# -------------------------------------------------------------------------

numpy_conditional_probability_example()

# -------------------------------------------------------------------------
# Complete Analysis
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("12. COMPLETE CONDITIONAL ANALYSIS")
print("=" * 72)

analysis = complete_conditional_analysis(
    probability_a=0.40,
    probability_b=0.50,
    joint_probability_value=0.20,
)

for key, value in analysis.items():
    if isinstance(value, bool):
        print(f"{key:15s}: {value}")
    else:
        print(f"{key:15s}: {value:.4f}")

# -------------------------------------------------------------------------
# Common Mistakes
# -------------------------------------------------------------------------

print_common_mistakes()

# -------------------------------------------------------------------------
# Key Takeaway
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("KEY IDEA")
print("=" * 72)

print(
    "Conditional probability answers:\n"
    "'How does the probability of an event change when we know "
    "that another event has occurred?'\n\n"
    "It is the foundation for Bayes' theorem, probabilistic "
    "classification, Naive Bayes, and many probabilistic ML models."
)
```

# =============================================================================

# 23. SCRIPT ENTRY POINT

# =============================================================================

if **name** == "**main**":
main()
