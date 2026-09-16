"""
Probability Fundamentals
========================

File:
02-Mathematics-for-Machine-Learning/03-Probability/probability.py

Purpose:
A beginner-to-advanced implementation of the fundamental concepts
of probability required for Statistics and Machine Learning.

## Topics Covered

1. Probability fundamentals
2. Sample spaces
3. Events
4. Event validation
5. Probability from equally likely outcomes
6. Probability from counts
7. Complement rule
8. Addition rule
9. Multiplication rule
10. Mutually exclusive events
11. Independent events
12. Conditional probability
13. Total probability
14. Expected value
15. Variance
16. Standard deviation
17. Probability tables
18. Empirical probability
19. Simulation-based probability
20. Law of Large Numbers demonstration
21. Probability in Machine Learning
22. Common mistakes

## Mathematical Notation

Probability of event A:

```
P(A)
```

Complement:

```
P(A^c) = 1 - P(A)
```

Union:

```
P(A ∪ B)
```

Intersection:

```
P(A ∩ B)
```

Conditional probability:

```
P(A | B) = P(A ∩ B) / P(B)
```

Addition rule:

```
P(A ∪ B)
    = P(A) + P(B) - P(A ∩ B)
```

Multiplication rule:

```
P(A ∩ B)
    = P(A | B) P(B)
```

If A and B are independent:

```
P(A ∩ B)
    = P(A)P(B)
```

Expected value:

```
E[X] = Σ x P(X=x)
```

Variance:

```
Var(X) = E[(X - μ)^2]
```

Standard deviation:

```
σ = sqrt(Var(X))
```

## Why Probability Matters in Machine Learning

Probability provides a mathematical framework for reasoning about
uncertainty.

Machine Learning uses probability for:

```
- Classification
- Regression
- Bayesian inference
- Naive Bayes
- Probabilistic prediction
- Generative models
- Likelihood estimation
- Uncertainty estimation
- Anomaly detection
- Model evaluation
- Statistical learning
```

## Requirements

Python 3.9+

Optional:
NumPy

Install NumPy:

```
pip install numpy
```

"""

from **future** import annotations

import math
import random
from collections import Counter
from typing import Dict, Iterable, Sequence

# =============================================================================

# 01. VALIDATION UTILITIES

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

Parameters
----------
probability : float
    Probability to validate.
name : str
    Name used in error messages.

Raises
------
ValueError
    If probability is outside [0, 1].
"""

if not isinstance(probability, (int, float)):
    raise TypeError(
        f"{name} must be a number."
    )

if not 0 <= probability <= 1:
    raise ValueError(
        f"{name} must be between 0 and 1. "
        f"Got {probability}."
    )
```

def validate_non_empty(
values: Sequence,
name: str = "values",
) -> None:
"""
Validate that a sequence is not empty.
"""

```
if not values:
    raise ValueError(
        f"{name} cannot be empty."
    )
```

# =============================================================================

# 02. SAMPLE SPACE

# =============================================================================

def create_sample_space(
outcomes: Iterable,
) -> set:
"""
Create a sample space from possible outcomes.

```
The sample space is the set of all possible outcomes of a random
experiment.

Example
-------

Coin:

    S = {Heads, Tails}

Die:

    S = {1, 2, 3, 4, 5, 6}

Parameters
----------
outcomes : Iterable
    Possible outcomes.

Returns
-------
set
    Sample space.
"""

sample_space = set(outcomes)

if not sample_space:
    raise ValueError(
        "Sample space cannot be empty."
    )

return sample_space
```

def sample_space_size(
sample_space: Iterable,
) -> int:
"""
Return the number of outcomes in a sample space.
"""

```
return len(set(sample_space))
```

def explain_sample_space() -> None:
"""
Explain the concept of a sample space.
"""

```
print("\n" + "=" * 80)
print("SAMPLE SPACE")
print("=" * 80)

print(
    """
```

A sample space contains every possible outcome of a random experiment.

Examples:

Coin:

```
S = {H, T}
```

Six-sided die:

```
S = {1, 2, 3, 4, 5, 6}
```

Two coin flips:

```
S = {HH, HT, TH, TT}
```

For a valid finite sample space:

```
Every possible outcome should be represented.
```

Probability theory starts with defining the experiment and its
possible outcomes correctly.
"""
)

# =============================================================================

# 03. EVENTS

# =============================================================================

def create_event(
sample_space: Iterable,
outcomes: Iterable,
) -> set:
"""
Create an event as a subset of the sample space.

```
An event is a collection of possible outcomes.

Example:

    Sample space:
        {1, 2, 3, 4, 5, 6}

    Event "even number":
        {2, 4, 6}
"""

sample_space = set(sample_space)
event = set(outcomes)

if not event.issubset(sample_space):
    invalid = event - sample_space

    raise ValueError(
        f"Event contains outcomes not in sample space: {invalid}"
    )

return event
```

def complement_event(
sample_space: Iterable,
event: Iterable,
) -> set:
"""
Calculate the complement of an event.

```
Formula:

    A^c = S - A

where:

    S = sample space
    A = event
"""

sample_space = set(sample_space)
event = create_event(sample_space, event)

return sample_space - event
```

def union_event(
event_a: Iterable,
event_b: Iterable,
) -> set:
"""
Calculate the union of two events.

```
A ∪ B contains outcomes belonging to A, B, or both.
"""

return set(event_a) | set(event_b)
```

def intersection_event(
event_a: Iterable,
event_b: Iterable,
) -> set:
"""
Calculate the intersection of two events.

```
A ∩ B contains outcomes common to both events.
"""

return set(event_a) & set(event_b)
```

def difference_event(
event_a: Iterable,
event_b: Iterable,
) -> set:
"""
Calculate A - B.

```
These are outcomes in A that are not in B.
"""

return set(event_a) - set(event_b)
```

# =============================================================================

# 04. BASIC PROBABILITY

# =============================================================================

def probability_from_equally_likely_outcomes(
favorable_outcomes: int,
total_outcomes: int,
) -> float:
"""
Calculate probability when all outcomes are equally likely.

```
Formula:

    P(A) =
        number of favorable outcomes
        ---------------------------
          number of possible outcomes

Example:

    Probability of rolling a 6:

        1 / 6
"""

if total_outcomes <= 0:
    raise ValueError(
        "total_outcomes must be greater than zero."
    )

if favorable_outcomes < 0:
    raise ValueError(
        "favorable_outcomes cannot be negative."
    )

if favorable_outcomes > total_outcomes:
    raise ValueError(
        "favorable_outcomes cannot exceed total_outcomes."
    )

return favorable_outcomes / total_outcomes
```

def probability_from_event(
event: Iterable,
sample_space: Iterable,
) -> float:
"""
Calculate probability assuming all sample-space outcomes
are equally likely.

```
Formula:

    P(A) = |A| / |S|
"""

sample_space = set(sample_space)
event = create_event(sample_space, event)

if not sample_space:
    raise ValueError(
        "Sample space cannot be empty."
    )

return len(event) / len(sample_space)
```

# =============================================================================

# 05. WEIGHTED PROBABILITY

# =============================================================================

def probability_from_weights(
weights: Dict,
) -> Dict:
"""
Convert non-negative weights into probabilities.

```
Example:

    weights = {
        "A": 2,
        "B": 3,
        "C": 5
    }

probabilities:

    A = 0.2
    B = 0.3
    C = 0.5

This is useful when outcomes are not equally likely.
"""

if not weights:
    raise ValueError(
        "weights cannot be empty."
    )

if any(weight < 0 for weight in weights.values()):
    raise ValueError(
        "Weights cannot be negative."
    )

total = sum(weights.values())

if total <= 0:
    raise ValueError(
        "Total weight must be greater than zero."
    )

return {
    outcome: weight / total
    for outcome, weight in weights.items()
}
```

# =============================================================================

# 06. COMPLEMENT RULE

# =============================================================================

def probability_complement(
probability: float,
) -> float:
"""
Calculate the probability of the complement.

```
Formula:

    P(A^c) = 1 - P(A)
"""

validate_probability(probability)

return 1 - probability
```

# =============================================================================

# 07. ADDITION RULE

# =============================================================================

def probability_union(
probability_a: float,
probability_b: float,
probability_intersection: float = 0.0,
) -> float:
"""
Calculate P(A ∪ B).

```
General addition rule:

    P(A ∪ B)
        = P(A) + P(B) - P(A ∩ B)

The intersection is subtracted because overlapping outcomes
would otherwise be counted twice.

For mutually exclusive events:

    P(A ∩ B) = 0

therefore:

    P(A ∪ B) = P(A) + P(B)
"""

validate_probability(probability_a, "P(A)")
validate_probability(probability_b, "P(B)")
validate_probability(
    probability_intersection,
    "P(A ∩ B)",
)

result = (
    probability_a
    + probability_b
    - probability_intersection
)

if not 0 <= result <= 1:
    raise ValueError(
        "The supplied probabilities are inconsistent."
    )

return result
```

def probability_union_events(
event_a: Iterable,
event_b: Iterable,
sample_space: Iterable,
) -> float:
"""
Calculate union probability directly from events.
"""

```
sample_space = set(sample_space)

union = union_event(
    event_a,
    event_b,
)

return probability_from_event(
    union,
    sample_space,
)
```

# =============================================================================

# 08. MULTIPLICATION RULE

# =============================================================================

def probability_intersection(
probability_a: float,
probability_b_given_a: float,
) -> float:
"""
Calculate P(A ∩ B).

```
General multiplication rule:

    P(A ∩ B)
        = P(A) P(B | A)

This formula does not require independence.
"""

validate_probability(
    probability_a,
    "P(A)",
)

validate_probability(
    probability_b_given_a,
    "P(B|A)",
)

return probability_a * probability_b_given_a
```

def probability_intersection_independent(
probability_a: float,
probability_b: float,
) -> float:
"""
Calculate P(A ∩ B) for independent events.

```
If A and B are independent:

    P(A ∩ B) = P(A)P(B)
"""

validate_probability(
    probability_a,
    "P(A)",
)

validate_probability(
    probability_b,
    "P(B)",
)

return probability_a * probability_b
```

# =============================================================================

# 09. MUTUALLY EXCLUSIVE EVENTS

# =============================================================================

def are_mutually_exclusive(
event_a: Iterable,
event_b: Iterable,
) -> bool:
"""
Check whether two events are mutually exclusive.

```
Two events are mutually exclusive when:

    A ∩ B = ∅

They cannot occur simultaneously.
"""

return len(
    intersection_event(
        event_a,
        event_b,
    )
) == 0
```

def probability_exclusive_union(
probability_a: float,
probability_b: float,
) -> float:
"""
Calculate union probability for mutually exclusive events.

```
    P(A ∪ B) = P(A) + P(B)
"""

validate_probability(probability_a, "P(A)")
validate_probability(probability_b, "P(B)")

result = probability_a + probability_b

if result > 1:
    raise ValueError(
        "Probabilities of mutually exclusive events "
        "cannot sum to more than 1."
    )

return result
```

# =============================================================================

# 10. INDEPENDENCE

# =============================================================================

def are_independent(
probability_a: float,
probability_b: float,
probability_intersection: float,
tolerance: float = 1e-9,
) -> bool:
"""
Check whether two events are independent.

```
Independence condition:

    P(A ∩ B) = P(A)P(B)

A small numerical tolerance is used for floating-point calculations.
"""

validate_probability(probability_a, "P(A)")
validate_probability(probability_b, "P(B)")
validate_probability(
    probability_intersection,
    "P(A ∩ B)",
)

return math.isclose(
    probability_intersection,
    probability_a * probability_b,
    rel_tol=tolerance,
    abs_tol=tolerance,
)
```

def explain_independence() -> None:
"""
Explain independence vs mutual exclusivity.
"""

```
print("\n" + "=" * 80)
print("INDEPENDENCE VS MUTUAL EXCLUSIVITY")
print("=" * 80)

print(
    """
```

Independent events:

```
One event does not change the probability of the other.

P(A ∩ B) = P(A)P(B)
```

Mutually exclusive events:

```
They cannot happen together.

P(A ∩ B) = 0
```

Important:

```
For non-zero probability events, mutually exclusive events are
generally NOT independent.
```

Example:

```
A = rolling an even number
B = rolling an odd number
```

They are mutually exclusive.

But knowing that A happened completely determines that B did not happen.

Therefore, they are not independent.
"""
)

# =============================================================================

# 11. CONDITIONAL PROBABILITY

# =============================================================================

def conditional_probability(
probability_intersection: float,
probability_condition: float,
) -> float:
"""
Calculate conditional probability.

```
Formula:

    P(A | B)
        = P(A ∩ B) / P(B)

The denominator is P(B), because B is the condition being assumed.

This function is intentionally included here as a foundation.
A more extensive treatment belongs in conditional-probability.py.
"""

validate_probability(
    probability_intersection,
    "P(A ∩ B)",
)

validate_probability(
    probability_condition,
    "P(B)",
)

if probability_condition == 0:
    raise ValueError(
        "Conditional probability is undefined when P(B)=0."
    )

if probability_intersection > probability_condition:
    raise ValueError(
        "P(A ∩ B) cannot be greater than P(B)."
    )

return probability_intersection / probability_condition
```

# =============================================================================

# 12. TOTAL PROBABILITY

# =============================================================================

def total_probability(
probabilities_given_conditions: Sequence[float],
condition_probabilities: Sequence[float],
) -> float:
"""
Calculate probability using the Law of Total Probability.

```
If B1, B2, ..., Bn form a partition:

    P(A)
      = Σ P(A | Bi) P(Bi)

Parameters
----------
probabilities_given_conditions:
    [P(A|B1), P(A|B2), ...]

condition_probabilities:
    [P(B1), P(B2), ...]

Example
-------

    P(A|B1) = 0.8
    P(A|B2) = 0.2

    P(B1) = 0.3
    P(B2) = 0.7

    P(A) = 0.8(0.3) + 0.2(0.7)
"""

if len(probabilities_given_conditions) != len(
    condition_probabilities
):
    raise ValueError(
        "Both sequences must have the same length."
    )

if not probabilities_given_conditions:
    raise ValueError(
        "At least one condition is required."
    )

for probability in probabilities_given_conditions:
    validate_probability(
        probability,
        "conditional probability",
    )

for probability in condition_probabilities:
    validate_probability(
        probability,
        "condition probability",
    )

if not math.isclose(
    sum(condition_probabilities),
    1.0,
    abs_tol=1e-9,
):
    raise ValueError(
        "Condition probabilities must sum to 1."
    )

return sum(
    conditional * condition
    for conditional, condition in zip(
        probabilities_given_conditions,
        condition_probabilities,
    )
)
```

# =============================================================================

# 13. COUNT-BASED EMPIRICAL PROBABILITY

# =============================================================================

def empirical_probability(
outcomes: Sequence,
target,
) -> float:
"""
Estimate probability from observed data.

```
Formula:

    P_hat(A)
        = count(A)
          ---------
          n

This is an empirical estimate, not necessarily the true population
probability.
"""

validate_non_empty(
    outcomes,
    "outcomes",
)

count = sum(
    outcome == target
    for outcome in outcomes
)

return count / len(outcomes)
```

def empirical_distribution(
outcomes: Sequence,
) -> Dict:
"""
Estimate the complete empirical probability distribution.
"""

```
validate_non_empty(
    outcomes,
    "outcomes",
)

counts = Counter(outcomes)

total = len(outcomes)

return {
    outcome: count / total
    for outcome, count in counts.items()
}
```

# =============================================================================

# 14. EXPECTED VALUE

# =============================================================================

def expected_value(
values: Sequence[float],
probabilities: Sequence[float],
) -> float:
"""
Calculate expected value of a discrete random variable.

```
Formula:

    E[X] = Σ x P(X=x)

Expected value is the long-run average value of a random variable
under repeated sampling from the same probability model.
"""

if len(values) != len(probabilities):
    raise ValueError(
        "values and probabilities must have the same length."
    )

validate_non_empty(
    values,
    "values",
)

for probability in probabilities:
    validate_probability(probability)

if not math.isclose(
    sum(probabilities),
    1.0,
    abs_tol=1e-9,
):
    raise ValueError(
        "Probabilities must sum to 1."
    )

return sum(
    value * probability
    for value, probability in zip(
        values,
        probabilities,
    )
)
```

# =============================================================================

# 15. VARIANCE

# =============================================================================

def variance(
values: Sequence[float],
probabilities: Sequence[float],
) -> float:
"""
Calculate population variance of a discrete random variable.

```
Formula:

    Var(X)
        = Σ P(X=x)(x - μ)^2

Equivalent:

    Var(X) = E[X²] - E[X]²
"""

mean = expected_value(
    values,
    probabilities,
)

return sum(
    probability * (value - mean) ** 2
    for value, probability in zip(
        values,
        probabilities,
    )
)
```

# =============================================================================

# 16. STANDARD DEVIATION

# =============================================================================

def standard_deviation(
values: Sequence[float],
probabilities: Sequence[float],
) -> float:
"""
Calculate standard deviation.

```
Formula:

    σ = sqrt(Var(X))
"""

return math.sqrt(
    variance(
        values,
        probabilities,
    )
)
```

# =============================================================================

# 17. FAIR DIE EXAMPLE

# =============================================================================

def fair_die_example() -> None:
"""
Demonstrate basic probability using a fair six-sided die.
"""

```
print("\n" + "=" * 80)
print("FAIR DIE EXAMPLE")
print("=" * 80)

sample_space = create_sample_space(
    [1, 2, 3, 4, 5, 6]
)

even_numbers = create_event(
    sample_space,
    [2, 4, 6],
)

numbers_greater_than_4 = create_event(
    sample_space,
    [5, 6],
)

print("Sample space:")
print(sample_space)

print("\nEven numbers:")
print(even_numbers)

print(
    "\nP(Even) =",
    probability_from_event(
        even_numbers,
        sample_space,
    ),
)

print(
    "P(X > 4) =",
    probability_from_event(
        numbers_greater_than_4,
        sample_space,
    ),
)

union = union_event(
    even_numbers,
    numbers_greater_than_4,
)

intersection = intersection_event(
    even_numbers,
    numbers_greater_than_4,
)

print("\nUnion:")
print(union)

print("\nIntersection:")
print(intersection)

print(
    "\nP(Even ∪ X>4) =",
    probability_from_event(
        union,
        sample_space,
    ),
)

print(
    "P(Even ∩ X>4) =",
    probability_from_event(
        intersection,
        sample_space,
    ),
)
```

# =============================================================================

# 18. COIN FLIP EXAMPLE

# =============================================================================

def coin_flip_example() -> None:
"""
Demonstrate probability using coin flips.
"""

```
print("\n" + "=" * 80)
print("COIN FLIP EXAMPLE")
print("=" * 80)

sample_space = create_sample_space(
    ["H", "T"]
)

heads = {"H"}

probability_heads = probability_from_event(
    heads,
    sample_space,
)

probability_tails = probability_complement(
    probability_heads,
)

print("Sample space:")
print(sample_space)

print(f"\nP(Heads) = {probability_heads:.4f}")
print(f"P(Tails) = {probability_tails:.4f}")

print(
    "\nCheck:",
    probability_heads + probability_tails,
)
```

# =============================================================================

# 19. TWO-EVENT EXAMPLE

# =============================================================================

def two_event_example() -> None:
"""
Demonstrate addition and multiplication rules.
"""

```
print("\n" + "=" * 80)
print("TWO-EVENT PROBABILITY")
print("=" * 80)

probability_a = 0.4
probability_b = 0.5
probability_intersection = 0.2

union = probability_union(
    probability_a,
    probability_b,
    probability_intersection,
)

conditional_b_given_a = (
    probability_intersection / probability_a
)

intersection = probability_intersection(
    probability_a,
    conditional_b_given_a,
)

print(f"P(A)       = {probability_a:.4f}")
print(f"P(B)       = {probability_b:.4f}")
print(
    f"P(A ∩ B)   = "
    f"{probability_intersection:.4f}"
)
print(f"P(A ∪ B)   = {union:.4f}")
print(
    f"P(B | A)   = "
    f"{conditional_b_given_a:.4f}"
)
print(
    f"P(A ∩ B) calculated from multiplication rule = "
    f"{intersection:.4f}"
)
```

# =============================================================================

# 20. EXPECTED VALUE EXAMPLE

# =============================================================================

def expected_value_example() -> None:
"""
Demonstrate expected value using a fair die.
"""

```
print("\n" + "=" * 80)
print("EXPECTED VALUE EXAMPLE")
print("=" * 80)

values = [1, 2, 3, 4, 5, 6]
probabilities = [1 / 6] * 6

mean = expected_value(
    values,
    probabilities,
)

var = variance(
    values,
    probabilities,
)

std = standard_deviation(
    values,
    probabilities,
)

print(f"Expected value = {mean:.4f}")
print(f"Variance       = {var:.4f}")
print(f"Std deviation  = {std:.4f}")
```

# =============================================================================

# 21. EMPIRICAL PROBABILITY EXAMPLE

# =============================================================================

def empirical_probability_example() -> None:
"""
Estimate probabilities from observed coin-flip data.
"""

```
print("\n" + "=" * 80)
print("EMPIRICAL PROBABILITY")
print("=" * 80)

observations = [
    "H",
    "T",
    "H",
    "H",
    "T",
    "H",
    "T",
    "T",
    "H",
    "H",
]

probability_heads = empirical_probability(
    observations,
    "H",
)

distribution = empirical_distribution(
    observations,
)

print("Observations:")
print(observations)

print(
    f"\nEstimated P(Heads) = "
    f"{probability_heads:.4f}"
)

print("\nEmpirical distribution:")
for outcome, probability in distribution.items():
    print(
        f"    {outcome}: "
        f"{probability:.4f}"
    )
```

# =============================================================================

# 22. SIMULATION-BASED PROBABILITY

# =============================================================================

def simulate_coin_flips(
number_of_flips: int,
probability_heads: float = 0.5,
) -> float:
"""
Estimate P(Heads) through simulation.

```
This demonstrates the connection between theoretical probability
and empirical frequency.
"""

if number_of_flips <= 0:
    raise ValueError(
        "number_of_flips must be greater than zero."
    )

validate_probability(
    probability_heads,
    "probability_heads",
)

heads = 0

for _ in range(number_of_flips):
    if random.random() < probability_heads:
        heads += 1

return heads / number_of_flips
```

def simulation_demo() -> None:
"""
Show how empirical probability approaches theoretical probability
as the number of trials increases.
"""

```
print("\n" + "=" * 80)
print("SIMULATION-BASED PROBABILITY")
print("=" * 80)

random.seed(42)

theoretical_probability = 0.5

trial_sizes = [
    10,
    100,
    1_000,
    10_000,
    100_000,
]

print(
    f"Theoretical P(Heads) = "
    f"{theoretical_probability:.4f}\n"
)

for trials in trial_sizes:
    estimated = simulate_coin_flips(
        trials,
        theoretical_probability,
    )

    print(
        f"{trials:>7,} trials -> "
        f"estimated probability = "
        f"{estimated:.6f}"
    )
```

# =============================================================================

# 23. LAW OF LARGE NUMBERS

# =============================================================================

def law_of_large_numbers_demo() -> None:
"""
Demonstrate the Law of Large Numbers.

```
As the number of independent observations increases, the sample
average tends to approach the expected value under appropriate
assumptions.

This is a theoretical result, not a guarantee that every finite
sample will be close to the expected value.
"""

print("\n" + "=" * 80)
print("LAW OF LARGE NUMBERS")
print("=" * 80)

random.seed(42)

true_probability = 0.7

cumulative_successes = 0

for trial in range(1, 10_001):
    if random.random() < true_probability:
        cumulative_successes += 1

    if trial in (
        10,
        100,
        1_000,
        10_000,
    ):
        estimate = cumulative_successes / trial

        print(
            f"Trials = {trial:>5,} | "
            f"Estimated P(success) = "
            f"{estimate:.6f}"
        )

print(
    f"\nTheoretical probability = "
    f"{true_probability:.6f}"
)
```

# =============================================================================

# 24. PROBABILITY TABLE

# =============================================================================

def probability_table_example() -> None:
"""
Demonstrate a simple probability table.
"""

```
print("\n" + "=" * 80)
print("PROBABILITY TABLE")
print("=" * 80)

outcomes = [
    "A",
    "B",
    "C",
    "D",
]

probabilities = [
    0.10,
    0.20,
    0.30,
    0.40,
]

print(
    f"{'Outcome':<15}"
    f"{'Probability':<15}"
    f"{'Cumulative'}"
)

print("-" * 45)

cumulative = 0.0

for outcome, probability in zip(
    outcomes,
    probabilities,
):
    cumulative += probability

    print(
        f"{outcome:<15}"
        f"{probability:<15.4f}"
        f"{cumulative:.4f}"
    )
```

# =============================================================================

# 25. MACHINE LEARNING CONNECTION

# =============================================================================

def machine_learning_connection() -> None:
"""
Explain how basic probability connects to Machine Learning.
"""

```
print("\n" + "=" * 80)
print("PROBABILITY IN MACHINE LEARNING")
print("=" * 80)

print(
    """
```

1. Classification

---

A classifier can estimate:

```
P(Y = class | X)
```

Example:

```
P(spam | email_features) = 0.92
```

## 2. Naive Bayes

Naive Bayes uses:

```
P(Y | X)
    ∝
P(X | Y) P(Y)
```

The model combines prior and conditional probabilities.

3. Logistic Regression

---

Logistic regression predicts:

```
P(Y=1 | X)
```

using the sigmoid function:

```
             1
P(Y=1|X) = ---------
            1 + e^(-z)
```

## 4. Maximum Likelihood

Probability models define a likelihood:

```
L(theta | data)
```

MLE selects parameters that maximize this likelihood.

5. Cross-Entropy

---

For binary classification:

```
Loss =
    -[y log(p) + (1-y) log(1-p)]
```

This is closely connected to the negative log-likelihood
of the Bernoulli distribution.

6. Uncertainty

---

Instead of producing only:

```
prediction = 1
```

a probabilistic model can produce:

```
P(Y=1 | X) = 0.83
```

This provides information about predictive uncertainty.

7. Generative Models

---

Generative models attempt to model the probability distribution
of the data:

```
P(X)
```

or the joint distribution:

```
P(X, Y)
```

## 8. Anomaly Detection

A model may estimate:

```
P(X=x)
```

Very unusual observations can receive low probability.

However:

```
Low probability != automatically an error.
```

The model's assumptions and application context matter.
"""
)

# =============================================================================

# 26. COMMON MISTAKES

# =============================================================================

def common_mistakes() -> None:
"""
Explain common probability mistakes.
"""

```
print("\n" + "=" * 80)
print("COMMON PROBABILITY MISTAKES")
print("=" * 80)

mistakes = [
    (
        "Probability outside [0,1]",
        "A valid probability must satisfy 0 <= P <= 1.",
    ),
    (
        "Forgetting the sample space",
        "Always define what outcomes are possible.",
    ),
    (
        "Assuming all outcomes are equally likely",
        "Many real-world experiments have unequal outcome probabilities.",
    ),
    (
        "Confusing union and intersection",
        "Union means OR; intersection means AND.",
    ),
    (
        "Using P(A)+P(B) for every union",
        "The overlap P(A ∩ B) must be subtracted unless events are mutually exclusive.",
    ),
    (
        "Multiplying probabilities automatically",
        "P(A∩B)=P(A)P(B) requires independence.",
    ),
    (
        "Confusing independence and exclusivity",
        "Independent events can happen together; mutually exclusive events cannot.",
    ),
    (
        "Reversing conditional probability",
        "P(A|B) and P(B|A) are generally different.",
    ),
    (
        "Treating correlation as causation",
        "A statistical relationship does not establish a causal mechanism.",
    ),
    (
        "Ignoring model assumptions",
        "Probability calculations are meaningful only under appropriate assumptions.",
    ),
    (
        "Confusing theoretical and empirical probability",
        "Observed frequency is an estimate and can differ from the theoretical probability.",
    ),
]

for title, explanation in mistakes:
    print(f"\n{title}")
    print(f"    {explanation}")
```

# =============================================================================

# 27. QUICK REFERENCE

# =============================================================================

def quick_reference() -> None:
"""
Print a mathematical quick reference.
"""

```
print("\n" + "=" * 80)
print("PROBABILITY QUICK REFERENCE")
print("=" * 80)

formulas = [
    (
        "Basic probability",
        "P(A) = favorable outcomes / total outcomes",
    ),
    (
        "Complement",
        "P(A^c) = 1 - P(A)",
    ),
    (
        "Union",
        "P(A ∪ B) = P(A) + P(B) - P(A ∩ B)",
    ),
    (
        "Mutually exclusive union",
        "P(A ∪ B) = P(A) + P(B)",
    ),
    (
        "Intersection",
        "P(A ∩ B) = P(A)P(B|A)",
    ),
    (
        "Independent intersection",
        "P(A ∩ B) = P(A)P(B)",
    ),
    (
        "Conditional probability",
        "P(A|B) = P(A ∩ B) / P(B)",
    ),
    (
        "Total probability",
        "P(A) = Σ P(A|Bi)P(Bi)",
    ),
    (
        "Expected value",
        "E[X] = Σ xP(X=x)",
    ),
    (
        "Variance",
        "Var(X) = E[(X-μ)^2]",
    ),
    (
        "Standard deviation",
        "σ = sqrt(Var(X))",
    ),
]

for name, formula in formulas:
    print(
        f"{name:<30} : {formula}"
    )
```

# =============================================================================

# 28. MAIN

# =============================================================================

def main() -> None:
"""
Run the complete probability demonstration.
"""

```
print("=" * 80)
print("PROBABILITY FUNDAMENTALS FOR MACHINE LEARNING")
print("=" * 80)

explain_sample_space()

fair_die_example()

coin_flip_example()

two_event_example()

expected_value_example()

empirical_probability_example()

probability_table_example()

simulation_demo()

law_of_large_numbers_demo()

explain_independence()

machine_learning_connection()

common_mistakes()

quick_reference()

print("\n" + "=" * 80)
print("END OF PROBABILITY FUNDAMENTALS")
print("=" * 80)
```

# =============================================================================

# 29. SCRIPT ENTRY POINT

# =============================================================================

if **name** == "**main**":
main()
"""
"""

# =============================================================================

# LEARNING ROADMAP

# =============================================================================

#

# Recommended progression:

#

# Probability fundamentals

# |

# v

# Sample spaces + Events

# |

# v

# Probability rules

# |

# v

# Conditional Probability

# |

# v

# Bayes' Theorem

# |

# v

# Random Variables

# |

# v

# Probability Distributions

# |

# v

# Expected Value + Variance

# |

# v

# Likelihood

# |

# v

# Maximum Likelihood Estimation

# |

# v

# Bayesian Inference

# |

# v

# Probabilistic Machine Learning

#

# =============================================================================

# IMPORTANT DISTINCTIONS

# =============================================================================

#

# Probability:

# Models uncertainty about possible outcomes.

#

# Statistics:

# Uses observed data to learn about populations or processes.

#

# Random variable:

# A function assigning numerical values to outcomes.

#

# Event:

# A set of possible outcomes.

#

# PMF:

# Used for discrete random variables.

#

# PDF:

# Used for continuous random variables.

#

# CDF:

# P(X <= x)

#

# Conditional probability:

# Probability given some information.

#

# Independence:

# One event does not change the probability of another.

#

# Mutual exclusivity:

# Two events cannot occur simultaneously.

#

# These concepts are related, but they are NOT interchangeable.

#

# =============================================================================

# FINAL MACHINE LEARNING CONNECTION

# =============================================================================

#

# Probability

# |

# +----> Random Variables

# |

# +----> Distributions

# |

# +----> Conditional Probability

# |

# +----> Bayes' Theorem

# |

# +----> Likelihood

# |

# +----> Loss Functions

# |

# +----> Statistical Learning

# |

# +----> Machine Learning

# |

# +----> Probabilistic Machine Learning

#

# A strong understanding of probability makes later Machine Learning

# concepts significantly easier to understand mathematically.

#

# =============================================================================

# END OF FILE

# =============================================================================
