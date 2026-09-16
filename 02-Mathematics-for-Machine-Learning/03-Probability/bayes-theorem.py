"""
Bayes' Theorem — Probability for Machine Learning
=================================================

Bayes' Theorem provides a mathematical framework for updating the
probability of a hypothesis when new evidence becomes available.

## Core Formula

```
            P(B | A) * P(A)
P(A | B) = -------------------
                 P(B)
```

Where:

```
P(A | B) = Posterior probability
P(B | A) = Likelihood
P(A)     = Prior probability
P(B)     = Evidence / Marginal probability
```

This file progresses from basic implementations to practical
Machine Learning applications.

## Topics Covered

1. Bayes' Theorem fundamentals
2. Prior probability
3. Likelihood
4. Evidence
5. Posterior probability
6. Manual implementation
7. Law of Total Probability
8. Medical diagnosis example
9. Spam classification example
10. Multiple hypotheses
11. Naive Bayes intuition
12. Numerical stability
13. Odds form of Bayes' theorem
14. Likelihood ratios
15. Sequential Bayesian updating
16. NumPy implementation
17. Scikit-learn connection
18. Common mistakes
19. Complete Bayesian analysis
20. Practical ML applications

Author: Machine Learning Mathematics Repository
"""

# =============================================================================

# 1. IMPORTS

# =============================================================================

from **future** import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence

# =============================================================================

# 2. BAYES' THEOREM FUNDAMENTALS

# =============================================================================

def bayes_theorem(
prior: float,
likelihood: float,
evidence: float,
) -> float:
"""
Calculate posterior probability using Bayes' theorem.

```
Formula:

    P(A | B) = P(B | A) * P(A) / P(B)

Parameters
----------
prior:
    P(A), the prior probability of the hypothesis.

likelihood:
    P(B | A), the probability of observing the evidence
    if the hypothesis is true.

evidence:
    P(B), the overall probability of observing the evidence.

Returns
-------
float
    Posterior probability P(A | B).

Raises
------
ValueError
    If probabilities are outside [0, 1] or evidence is zero.
"""

_validate_probability(prior, "prior")
_validate_probability(likelihood, "likelihood")
_validate_probability(evidence, "evidence")

if evidence == 0:
    raise ValueError("Evidence probability P(B) cannot be zero.")

posterior = (likelihood * prior) / evidence

return posterior
```

# =============================================================================

# 3. PROBABILITY VALIDATION

# =============================================================================

def _validate_probability(value: float, name: str = "probability") -> None:
"""
Validate that a value represents a probability.

```
Valid probability:

    0 <= P <= 1
"""

if not isinstance(value, (int, float)):
    raise TypeError(f"{name} must be a number.")

if not math.isfinite(value):
    raise ValueError(f"{name} must be finite.")

if not 0 <= value <= 1:
    raise ValueError(
        f"{name} must be between 0 and 1. "
        f"Received: {value}"
    )
```

# =============================================================================

# 4. LAW OF TOTAL PROBABILITY

# =============================================================================

def total_probability(
prior_a: float,
probability_b_given_a: float,
probability_b_given_not_a: float,
) -> float:
"""
Calculate P(B) using the Law of Total Probability.

```
For hypothesis A and its complement A^c:

    P(B)
    =
    P(B | A)P(A)
    +
    P(B | A^c)P(A^c)

Since:

    P(A^c) = 1 - P(A)

Parameters
----------
prior_a:
    P(A)

probability_b_given_a:
    P(B | A)

probability_b_given_not_a:
    P(B | A^c)

Returns
-------
float
    P(B)
"""

_validate_probability(prior_a, "prior_a")
_validate_probability(
    probability_b_given_a,
    "probability_b_given_a",
)
_validate_probability(
    probability_b_given_not_a,
    "probability_b_given_not_a",
)

probability_not_a = 1 - prior_a

return (
    probability_b_given_a * prior_a
    + probability_b_given_not_a * probability_not_a
)
```

# =============================================================================

# 5. BAYES' THEOREM FROM PRIOR + LIKELIHOOD

# =============================================================================

def bayes_from_binary_hypothesis(
prior_a: float,
probability_b_given_a: float,
probability_b_given_not_a: float,
) -> float:
"""
Calculate P(A | B) when A has two possibilities:

```
    A
    not A

The evidence probability P(B) is calculated automatically.

Formula:

    P(A | B)
    =
    P(B | A)P(A)
    --------------------------------
    P(B | A)P(A)
    +
    P(B | not A)P(not A)
"""

evidence = total_probability(
    prior_a=prior_a,
    probability_b_given_a=probability_b_given_a,
    probability_b_given_not_a=probability_b_given_not_a,
)

return bayes_theorem(
    prior=prior_a,
    likelihood=probability_b_given_a,
    evidence=evidence,
)
```

# =============================================================================

# 6. STEP-BY-STEP BAYES CALCULATION

# =============================================================================

def explain_bayes_calculation(
prior_a: float,
probability_b_given_a: float,
probability_b_given_not_a: float,
) -> Dict[str, float]:
"""
Return every major component of a binary Bayes calculation.

```
This function is useful for learning and debugging.
"""

_validate_probability(prior_a, "prior_a")
_validate_probability(
    probability_b_given_a,
    "probability_b_given_a",
)
_validate_probability(
    probability_b_given_not_a,
    "probability_b_given_not_a",
)

probability_not_a = 1 - prior_a

probability_b = total_probability(
    prior_a,
    probability_b_given_a,
    probability_b_given_not_a,
)

numerator = probability_b_given_a * prior_a

posterior = numerator / probability_b

return {
    "prior_a": prior_a,
    "prior_not_a": probability_not_a,
    "likelihood_b_given_a": probability_b_given_a,
    "likelihood_b_given_not_a": probability_b_given_not_a,
    "evidence_b": probability_b,
    "numerator": numerator,
    "posterior_a_given_b": posterior,
}
```

# =============================================================================

# 7. MEDICAL DIAGNOSIS EXAMPLE

# =============================================================================

def medical_diagnosis_example() -> None:
"""
Demonstrate Bayes' theorem using a hypothetical diagnostic test.

```
Example assumptions:

    Disease prevalence = 1%
    Sensitivity       = 95%
    Specificity       = 90%

Therefore:

    P(Disease) = 0.01

    P(Positive | Disease) = 0.95

    P(Negative | No Disease) = 0.90

Therefore:

    P(Positive | No Disease) = 0.10
"""

print("\n" + "=" * 72)
print("MEDICAL DIAGNOSIS EXAMPLE")
print("=" * 72)

prevalence = 0.01
sensitivity = 0.95
specificity = 0.90

false_positive_rate = 1 - specificity

result = explain_bayes_calculation(
    prior_a=prevalence,
    probability_b_given_a=sensitivity,
    probability_b_given_not_a=false_positive_rate,
)

posterior = result["posterior_a_given_b"]

print(f"Prior probability:       {prevalence:.2%}")
print(f"Sensitivity:             {sensitivity:.2%}")
print(f"Specificity:             {specificity:.2%}")
print(f"False-positive rate:     {false_positive_rate:.2%}")
print(f"Probability of positive: {result['evidence_b']:.2%}")
print(f"Posterior probability:   {posterior:.2%}")

print("\nInterpretation:")
print(
    "The posterior probability answers:\n"
    "P(Disease | Positive Test)"
)
```

# =============================================================================

# 8. FREQUENCY / COUNT INTERPRETATION

# =============================================================================

def bayes_from_counts(
disease_and_positive: int,
disease_and_negative: int,
no_disease_and_positive: int,
no_disease_and_negative: int,
) -> float:
"""
Calculate P(Disease | Positive) directly from a 2x2 table.

```
Table:

                Positive    Negative
    Disease       TP          FN
    No Disease    FP          TN

Formula:

    P(Disease | Positive)
    =
    TP / (TP + FP)

This demonstrates that Bayes' theorem can also be understood
using population counts.
"""

counts = [
    disease_and_positive,
    disease_and_negative,
    no_disease_and_positive,
    no_disease_and_negative,
]

if any(
    not isinstance(value, int) or value < 0
    for value in counts
):
    raise ValueError("All counts must be non-negative integers.")

total_positive = (
    disease_and_positive
    + no_disease_and_positive
)

if total_positive == 0:
    raise ValueError(
        "There must be at least one positive observation."
    )

return disease_and_positive / total_positive
```

# =============================================================================

# 9. SPAM CLASSIFICATION EXAMPLE

# =============================================================================

def spam_classification_example() -> None:
"""
Demonstrate Bayesian reasoning for spam detection.

```
Suppose:

    P(Spam) = 0.30

    P("free" | Spam) = 0.80

    P("free" | Not Spam) = 0.05

Calculate:

    P(Spam | "free")
"""

print("\n" + "=" * 72)
print("SPAM CLASSIFICATION EXAMPLE")
print("=" * 72)

prior_spam = 0.30

word_given_spam = 0.80
word_given_not_spam = 0.05

posterior = bayes_from_binary_hypothesis(
    prior_a=prior_spam,
    probability_b_given_a=word_given_spam,
    probability_b_given_not_a=word_given_not_spam,
)

print(f"P(Spam):             {prior_spam:.2%}")
print(f'P("free" | Spam):    {word_given_spam:.2%}')
print(f'P("free" | NotSpam): {word_given_not_spam:.2%}')
print(f'P(Spam | "free"):    {posterior:.2%}')
```

# =============================================================================

# 10. MULTIPLE HYPOTHESES

# =============================================================================

def bayes_multiple_hypotheses(
priors: Mapping[str, float],
likelihoods: Mapping[str, float],
) -> Dict[str, float]:
"""
Calculate posterior probabilities for multiple hypotheses.

```
Given hypotheses H1, H2, ..., Hn:

    P(Hi | B)
    =
    P(B | Hi)P(Hi)
    -------------------------
    sum_j P(B | Hj)P(Hj)

Parameters
----------
priors:
    Mapping from hypothesis name to prior probability.

likelihoods:
    Mapping from hypothesis name to P(B | hypothesis).

Returns
-------
dict
    Posterior probabilities for every hypothesis.

Example
-------

priors = {
    "Model_A": 0.5,
    "Model_B": 0.3,
    "Model_C": 0.2,
}

likelihoods = {
    "Model_A": 0.7,
    "Model_B": 0.4,
    "Model_C": 0.2,
}
"""

if not priors:
    raise ValueError("At least one hypothesis is required.")

if set(priors) != set(likelihoods):
    raise ValueError(
        "Priors and likelihoods must contain the same hypotheses."
    )

for name, probability in priors.items():
    _validate_probability(probability, f"prior[{name}]")

for name, probability in likelihoods.items():
    _validate_probability(
        probability,
        f"likelihood[{name}]",
    )

prior_sum = sum(priors.values())

if not math.isclose(prior_sum, 1.0, rel_tol=1e-9):
    raise ValueError(
        f"Prior probabilities must sum to 1. "
        f"Current sum: {prior_sum}"
    )

unnormalized_posteriors = {
    hypothesis: priors[hypothesis] * likelihoods[hypothesis]
    for hypothesis in priors
}

evidence = sum(unnormalized_posteriors.values())

if evidence == 0:
    raise ValueError(
        "Evidence is zero. The supplied model assigns "
        "zero probability to the observed evidence."
    )

return {
    hypothesis: value / evidence
    for hypothesis, value in unnormalized_posteriors.items()
}
```

# =============================================================================

# 11. ODDS FORM OF BAYES' THEOREM

# =============================================================================

def posterior_odds(
prior_probability: float,
likelihood_ratio: float,
) -> float:
"""
Calculate posterior odds using:

```
    Posterior Odds
    =
    Prior Odds × Likelihood Ratio

Prior odds:

    P(A)
    ------
    1-P(A)

Likelihood ratio:

    P(B|A)
    -------
    P(B|not A)

Returns
-------
float
    Posterior odds.
"""

_validate_probability(
    prior_probability,
    "prior_probability",
)

if prior_probability in (0, 1):
    raise ValueError(
        "Prior probability must be strictly between 0 and 1."
    )

if likelihood_ratio < 0 or not math.isfinite(likelihood_ratio):
    raise ValueError(
        "Likelihood ratio must be a finite non-negative number."
    )

prior_odds = prior_probability / (1 - prior_probability)

return prior_odds * likelihood_ratio
```

def odds_to_probability(odds: float) -> float:
"""
Convert odds into probability.

```
    P = odds / (1 + odds)
"""

if odds < 0 or not math.isfinite(odds):
    raise ValueError(
        "Odds must be a finite non-negative number."
    )

return odds / (1 + odds)
```

# =============================================================================

# 12. LIKELIHOOD RATIO

# =============================================================================

def likelihood_ratio(
probability_evidence_given_a: float,
probability_evidence_given_not_a: float,
) -> float:
"""
Calculate the likelihood ratio:

```
    LR =
    P(B | A)
    --------
    P(B | not A)

A likelihood ratio greater than 1 means the evidence is
more likely under A than under not A.
"""

_validate_probability(
    probability_evidence_given_a,
    "probability_evidence_given_a",
)

_validate_probability(
    probability_evidence_given_not_a,
    "probability_evidence_given_not_a",
)

if probability_evidence_given_not_a == 0:
    if probability_evidence_given_a > 0:
        return math.inf

    return 1.0

return (
    probability_evidence_given_a
    / probability_evidence_given_not_a
)
```

# =============================================================================

# 13. SEQUENTIAL BAYESIAN UPDATING

# =============================================================================

def sequential_update(
prior: float,
evidence_probabilities_given_hypothesis: Sequence[float],
evidence_probabilities_given_not_hypothesis: Sequence[float],
) -> float:
"""
Sequentially update a binary hypothesis using independent evidence.

```
For each evidence item Ei:

    P(A | Ei)

becomes the prior for the next update.

This implementation treats the supplied evidence likelihoods
as conditionally independent given the hypothesis.

Parameters
----------
prior:
    Initial P(A).

evidence_probabilities_given_hypothesis:
    Values of P(Ei | A).

evidence_probabilities_given_not_hypothesis:
    Values of P(Ei | not A).

Returns
-------
float
    Final posterior probability.
"""

if len(evidence_probabilities_given_hypothesis) != len(
    evidence_probabilities_given_not_hypothesis
):
    raise ValueError(
        "Both evidence sequences must have the same length."
    )

current_probability = prior

for (
    probability_e_given_a,
    probability_e_given_not_a,
) in zip(
    evidence_probabilities_given_hypothesis,
    evidence_probabilities_given_not_hypothesis,
):
    current_probability = bayes_from_binary_hypothesis(
        prior_a=current_probability,
        probability_b_given_a=probability_e_given_a,
        probability_b_given_not_a=probability_e_given_not_a,
    )

return current_probability
```

# =============================================================================

# 14. BAYES' THEOREM WITH LOG-ODDS

# =============================================================================

def log_odds(probability: float) -> float:
"""
Convert probability into log-odds.

```
    log_odds(p) = log(p / (1-p))
"""

_validate_probability(probability, "probability")

if probability in (0, 1):
    raise ValueError(
        "Probability must be strictly between 0 and 1."
    )

return math.log(probability / (1 - probability))
```

def probability_from_log_odds(value: float) -> float:
"""
Convert log-odds back to probability.

```
    p = 1 / (1 + exp(-log_odds))
"""

if not math.isfinite(value):
    raise ValueError("Log-odds must be finite.")

# Numerically stable sigmoid implementation.
if value >= 0:
    exponent = math.exp(-value)
    return 1 / (1 + exponent)

exponent = math.exp(value)
return exponent / (1 + exponent)
```

# =============================================================================

# 15. NUMPY IMPLEMENTATION

# =============================================================================

def numpy_bayes_example() -> None:
"""
Demonstrate Bayesian calculations using NumPy.

```
NumPy is optional so that this educational file remains runnable
without external dependencies.
"""

print("\n" + "=" * 72)
print("NUMPY BAYES EXAMPLE")
print("=" * 72)

try:
    import numpy as np
except ImportError:
    print("NumPy is not installed.")
    print("Install it with: pip install numpy")
    return

prior = np.array([0.01, 0.99])

likelihood = np.array([0.95, 0.10])

unnormalized = prior * likelihood

posterior = unnormalized / np.sum(unnormalized)

print("Hypotheses:")
print("  Disease")
print("  No Disease")

print("\nPosterior probabilities:")
print(f"  Disease:    {posterior[0]:.2%}")
print(f"  No Disease: {posterior[1]:.2%}")
```

# =============================================================================

# 16. NUMERICAL STABILITY

# =============================================================================

def stable_log_bayes_update(
log_prior: float,
log_likelihood: float,
) -> float:
"""
Return an unnormalized log posterior.

```
Instead of multiplying very small probabilities:

    prior * likelihood

we can work in log-space:

    log(prior * likelihood)
    =
    log(prior) + log(likelihood)

This is important in probabilistic ML when many probabilities
are multiplied together.
"""

if not math.isfinite(log_prior):
    raise ValueError("log_prior must be finite.")

if not math.isfinite(log_likelihood):
    raise ValueError("log_likelihood must be finite.")

return log_prior + log_likelihood
```

# =============================================================================

# 17. BAYESIAN CLASSIFICATION CONCEPT

# =============================================================================

def bayesian_classification(
priors: Mapping[str, float],
likelihoods: Mapping[str, float],
) -> Dict[str, float]:
"""
Calculate posterior class probabilities.

```
This function demonstrates the mathematical core of Bayesian
classification.

Parameters
----------
priors:
    P(Class)

likelihoods:
    P(Evidence | Class)

Returns
-------
dict
    P(Class | Evidence) for each class.
"""

return bayes_multiple_hypotheses(
    priors=priors,
    likelihoods=likelihoods,
)
```

# =============================================================================

# 18. COMPLETE BAYES ANALYSIS

# =============================================================================

def complete_bayes_analysis(
prior_a: float,
probability_b_given_a: float,
probability_b_given_not_a: float,
) -> Dict[str, float]:
"""
Perform a complete binary Bayesian analysis.

```
Returns:

    - Prior
    - Complement prior
    - Likelihood
    - Evidence
    - Posterior
    - Likelihood ratio
    - Prior odds
    - Posterior odds
"""

probability_not_a = 1 - prior_a

evidence = total_probability(
    prior_a=prior_a,
    probability_b_given_a=probability_b_given_a,
    probability_b_given_not_a=probability_b_given_not_a,
)

posterior = bayes_theorem(
    prior=prior_a,
    likelihood=probability_b_given_a,
    evidence=evidence,
)

lr = likelihood_ratio(
    probability_evidence_given_a=probability_b_given_a,
    probability_evidence_given_not_a=probability_b_given_not_a,
)

if 0 < prior_a < 1:
    prior_odds_value = prior_a / probability_not_a

    if math.isinf(lr):
        posterior_odds_value = math.inf
    else:
        posterior_odds_value = posterior_odds(
            prior_probability=prior_a,
            likelihood_ratio=lr,
        )
else:
    prior_odds_value = math.inf if prior_a == 1 else 0.0
    posterior_odds_value = (
        math.inf if posterior == 1 else 0.0
    )

return {
    "prior": prior_a,
    "prior_complement": probability_not_a,
    "likelihood": probability_b_given_a,
    "evidence": evidence,
    "posterior": posterior,
    "likelihood_ratio": lr,
    "prior_odds": prior_odds_value,
    "posterior_odds": posterior_odds_value,
}
```

# =============================================================================

# 19. COMMON MISTAKES

# =============================================================================

def print_common_mistakes() -> None:
"""Print common mistakes when applying Bayes' theorem."""

```
print("\n" + "=" * 72)
print("COMMON MISTAKES")
print("=" * 72)

mistakes = [
    (
        "1. Confusing P(A|B) with P(B|A)",
        "Conditional probabilities are generally not interchangeable.",
    ),
    (
        "2. Ignoring the base rate",
        "The prior probability can strongly affect the posterior.",
    ),
    (
        "3. Forgetting false positives",
        "Evidence can occur even when the hypothesis is false.",
    ),
    (
        "4. Using an incorrect denominator",
        "Bayes requires the total probability of the observed evidence.",
    ),
    (
        "5. Assuming independence without justification",
        "Conditional independence is a modeling assumption.",
    ),
    (
        "6. Treating probabilities as certainty",
        "A posterior is a probability estimate under a model.",
    ),
    (
        "7. Multiplying many tiny probabilities directly",
        "Log probabilities are often preferred for numerical stability.",
    ),
]

for title, explanation in mistakes:
    print(f"\n{title}")
    print(f"   {explanation}")
```

# =============================================================================

# 20. MAIN DEMONSTRATION

# =============================================================================

def main() -> None:
"""
Run demonstrations of Bayes' theorem from beginner to advanced.
"""

```
print("=" * 72)
print("BAYES' THEOREM — PROBABILITY FOR MACHINE LEARNING")
print("=" * 72)

# -------------------------------------------------------------------------
# Basic Example
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("1. BASIC BAYES EXAMPLE")
print("=" * 72)

prior = 0.20
likelihood = 0.80
evidence = 0.50

posterior = bayes_theorem(
    prior=prior,
    likelihood=likelihood,
    evidence=evidence,
)

print(f"Prior P(A):          {prior:.2%}")
print(f"Likelihood P(B | A): {likelihood:.2%}")
print(f"Evidence P(B):       {evidence:.2%}")
print(f"Posterior P(A | B):  {posterior:.2%}")

# -------------------------------------------------------------------------
# Step-by-Step Calculation
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("2. STEP-BY-STEP CALCULATION")
print("=" * 72)

details = explain_bayes_calculation(
    prior_a=0.20,
    probability_b_given_a=0.80,
    probability_b_given_not_a=0.10,
)

for key, value in details.items():
    print(f"{key:30s}: {value:.6f}")

# -------------------------------------------------------------------------
# Medical Diagnosis
# -------------------------------------------------------------------------

medical_diagnosis_example()

# -------------------------------------------------------------------------
# Spam Detection
# -------------------------------------------------------------------------

spam_classification_example()

# -------------------------------------------------------------------------
# Count-Based Example
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("3. COUNT-BASED BAYES EXAMPLE")
print("=" * 72)

posterior_from_counts = bayes_from_counts(
    disease_and_positive=95,
    disease_and_negative=5,
    no_disease_and_positive=99,
    no_disease_and_negative=891,
)

print(
    "P(Disease | Positive) = "
    f"{posterior_from_counts:.2%}"
)

# -------------------------------------------------------------------------
# Multiple Hypotheses
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("4. MULTIPLE HYPOTHESES")
print("=" * 72)

priors = {
    "Class_A": 0.50,
    "Class_B": 0.30,
    "Class_C": 0.20,
}

likelihoods = {
    "Class_A": 0.70,
    "Class_B": 0.40,
    "Class_C": 0.20,
}

posteriors = bayes_multiple_hypotheses(
    priors=priors,
    likelihoods=likelihoods,
)

for hypothesis, probability in posteriors.items():
    print(f"{hypothesis:10s}: {probability:.2%}")

# -------------------------------------------------------------------------
# Likelihood Ratio
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("5. LIKELIHOOD RATIO")
print("=" * 72)

lr = likelihood_ratio(
    probability_evidence_given_a=0.95,
    probability_evidence_given_not_a=0.10,
)

print(f"Likelihood ratio = {lr:.2f}")

# -------------------------------------------------------------------------
# Odds Form
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("6. ODDS FORM OF BAYES")
print("=" * 72)

prior_probability = 0.01

posterior_odds_value = posterior_odds(
    prior_probability=prior_probability,
    likelihood_ratio=lr,
)

posterior_from_odds = odds_to_probability(
    posterior_odds_value
)

print(f"Prior probability:  {prior_probability:.2%}")
print(f"Likelihood ratio:   {lr:.2f}")
print(f"Posterior odds:     {posterior_odds_value:.4f}")
print(f"Posterior:          {posterior_from_odds:.2%}")

# -------------------------------------------------------------------------
# Sequential Updating
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("7. SEQUENTIAL BAYESIAN UPDATING")
print("=" * 72)

final_probability = sequential_update(
    prior=0.10,
    evidence_probabilities_given_hypothesis=[
        0.80,
        0.70,
        0.90,
    ],
    evidence_probabilities_given_not_hypothesis=[
        0.20,
        0.30,
        0.10,
    ],
)

print(
    "Final posterior after three evidence updates: "
    f"{final_probability:.2%}"
)

# -------------------------------------------------------------------------
# Log Odds
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("8. LOG-ODDS")
print("=" * 72)

probability = 0.75

log_odds_value = log_odds(probability)

recovered_probability = probability_from_log_odds(
    log_odds_value
)

print(f"Probability:         {probability:.4f}")
print(f"Log-odds:            {log_odds_value:.4f}")
print(f"Recovered probability: {recovered_probability:.4f}")

# -------------------------------------------------------------------------
# NumPy
# -------------------------------------------------------------------------

numpy_bayes_example()

# -------------------------------------------------------------------------
# Common Mistakes
# -------------------------------------------------------------------------

print_common_mistakes()

# -------------------------------------------------------------------------
# Final Message
# -------------------------------------------------------------------------

print("\n" + "=" * 72)
print("KEY IDEA")
print("=" * 72)

print(
    "Bayes' theorem updates a prior belief using observed evidence.\n"
    "It is a fundamental mathematical idea behind Bayesian inference,\n"
    "probabilistic classification, Naive Bayes, and many ML systems."
)
```

# =============================================================================

# 21. SCRIPT ENTRY POINT

# =============================================================================

if **name** == "**main**":
main()
