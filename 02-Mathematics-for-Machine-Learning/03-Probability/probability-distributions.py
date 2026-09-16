"""
Probability Distributions
=========================

File:
02-Mathematics-for-Machine-Learning/03-Probability/probability-distributions.py

Purpose:
A practical and theory-rich introduction to probability distributions
for Machine Learning.

This module covers:

```
1. What a probability distribution is
2. Discrete vs continuous distributions
3. Probability Mass Function (PMF)
4. Probability Density Function (PDF)
5. Cumulative Distribution Function (CDF)
6. Bernoulli distribution
7. Binomial distribution
8. Geometric distribution
9. Poisson distribution
10. Uniform distribution
11. Normal (Gaussian) distribution
12. Exponential distribution
13. Expected value
14. Variance
15. Standard deviation
16. Z-scores
17. Sampling from distributions
18. Distribution parameter estimation
19. NumPy implementations
20. Distribution selection for ML problems
21. Common mistakes
```

The goal is not only to calculate probabilities, but to understand the
mathematical assumptions behind each distribution and how those assumptions
connect to Machine Learning.

Requirements:
Python 3.9+

Optional:
NumPy
SciPy

Install optional dependencies:

```
pip install numpy scipy
```

## Mathematical Notation

Discrete random variable:

```
P(X = x)
```

Continuous random variable:

```
f(x)
```

CDF:

```
F(x) = P(X <= x)
```

Expected value:

```
E[X]
```

Variance:

```
Var(X)
```

Standard deviation:

```
sigma = sqrt(Var(X))
```

Important:
A probability distribution is a mathematical model of uncertainty.

```
Choosing an inappropriate distribution can lead to incorrect statistical
conclusions and poor Machine Learning models.
```

"""

from **future** import annotations

import math
import random
from typing import Iterable, Sequence

# =============================================================================

# 01. BASIC VALIDATION UTILITIES

# =============================================================================

def validate_probability(probability: float, name: str = "probability") -> None:
"""
Validate that a value is a valid probability.

```
A probability must satisfy:

    0 <= P <= 1

Parameters
----------
probability : float
    Probability value.
name : str
    Name used in the error message.

Raises
------
ValueError
    If the value is outside [0, 1].
"""

if not 0 <= probability <= 1:
    raise ValueError(
        f"{name} must be between 0 and 1. Got {probability}."
    )
```

def validate_positive(value: float, name: str = "value") -> None:
"""
Validate that a value is strictly positive.
"""

```
if value <= 0:
    raise ValueError(
        f"{name} must be greater than 0. Got {value}."
    )
```

# =============================================================================

# 02. WHAT IS A PROBABILITY DISTRIBUTION?

# =============================================================================

def explain_probability_distribution() -> None:
"""
Print the basic concept of probability distributions.

```
A probability distribution describes how probability is assigned
to possible values of a random variable.
"""

print("\n" + "=" * 80)
print("WHAT IS A PROBABILITY DISTRIBUTION?")
print("=" * 80)

print(
    """
```

A probability distribution tells us how likely different outcomes are.

Example:

```
Let X = number of heads when flipping a coin.
```

Possible values:

```
X = 0
X = 1
X = 2
...
```

The distribution tells us:

```
P(X = 0)
P(X = 1)
P(X = 2)
...
```

There are two major categories:

```
1. Discrete distributions
   - Outcomes are countable.
   - Example: number of customers.

2. Continuous distributions
   - Values can lie anywhere within an interval.
   - Example: height, weight, temperature.
```

Machine Learning uses probability distributions for:

```
- Classification
- Regression
- Generative models
- Bayesian inference
- Anomaly detection
- Uncertainty estimation
- Hypothesis testing
- Maximum Likelihood Estimation
- Naive Bayes
- Gaussian models
"""
)
```

# =============================================================================

# 03. PMF - PROBABILITY MASS FUNCTION

# =============================================================================

def pmf(values: Sequence[float], probabilities: Sequence[float]) -> dict:
"""
Construct a discrete Probability Mass Function (PMF).

```
For a discrete random variable:

    PMF(x) = P(X = x)

Requirements:

    P(X = x) >= 0

    sum(P(X = x)) = 1

Example
-------

A fair six-sided die:

    X = {1, 2, 3, 4, 5, 6}

    P(X=x) = 1/6
"""

if len(values) != len(probabilities):
    raise ValueError("values and probabilities must have the same length.")

if not probabilities:
    raise ValueError("PMF cannot be empty.")

for probability in probabilities:
    validate_probability(probability)

total = sum(probabilities)

if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
    raise ValueError(
        f"PMF probabilities must sum to 1. Got {total}."
    )

return dict(zip(values, probabilities))
```

def pmf_probability(
distribution: dict,
value: float,
) -> float:
"""
Return P(X = value) from a PMF.
"""

```
return distribution.get(value, 0.0)
```

# =============================================================================

# 04. PDF - PROBABILITY DENSITY FUNCTION

# =============================================================================

def explain_pdf() -> None:
"""
Explain the difference between PDF and probability.
"""

```
print("\n" + "=" * 80)
print("PDF - PROBABILITY DENSITY FUNCTION")
print("=" * 80)

print(
    """
```

For continuous random variables, we use a Probability Density Function:

```
f(x)
```

Important:

```
f(x) is a density, not the probability of an exact point.
```

For a continuous random variable:

```
P(X = x) = 0
```

under the usual continuous model.

Probability over an interval is calculated using area:

```
P(a <= X <= b)
    = integral from a to b of f(x) dx
```

Therefore:

```
Probability = AREA UNDER THE PDF CURVE
"""
)
```

# =============================================================================

# 05. CDF - CUMULATIVE DISTRIBUTION FUNCTION

# =============================================================================

def empirical_cdf(
data: Sequence[float],
x: float,
) -> float:
"""
Calculate an empirical CDF.

```
Definition:

    F(x) = P(X <= x)

For observed data, the empirical CDF is:

    F(x) =
        number of observations <= x
        --------------------------------
                total observations

Parameters
----------
data : Sequence[float]
    Observed values.
x : float
    Point at which the CDF is evaluated.
"""

if not data:
    raise ValueError("data cannot be empty.")

count = sum(value <= x for value in data)

return count / len(data)
```

def explain_cdf() -> None:
"""
Explain the Cumulative Distribution Function.
"""

```
print("\n" + "=" * 80)
print("CDF - CUMULATIVE DISTRIBUTION FUNCTION")
print("=" * 80)

print(
    """
```

The CDF answers:

```
"What is the probability that X is less than or equal to x?"
```

Mathematically:

```
F(x) = P(X <= x)
```

The CDF works for both:

```
- Discrete distributions
- Continuous distributions
```

Properties:

```
0 <= F(x) <= 1

F(x) is non-decreasing

F(-infinity) = 0

F(+infinity) = 1
```

CDFs are extremely useful for:

```
- Percentiles
- Quantiles
- Probability intervals
- Statistical testing
- Risk analysis
"""
)
```

# =============================================================================

# 06. BERNOULLI DISTRIBUTION

# =============================================================================

def bernoulli_pmf(x: int, p: float) -> float:
"""
Calculate the PMF of a Bernoulli random variable.

```
A Bernoulli random variable has exactly two outcomes:

    X = 1 -> success
    X = 0 -> failure

Formula:

    P(X=x) = p^x * (1-p)^(1-x)

where:

    0 <= p <= 1
"""

validate_probability(p, "p")

if x not in (0, 1):
    return 0.0

return (p ** x) * ((1 - p) ** (1 - x))
```

def bernoulli_mean(p: float) -> float:
"""
Expected value of Bernoulli distribution.

```
    E[X] = p
"""

validate_probability(p, "p")
return p
```

def bernoulli_variance(p: float) -> float:
"""
Variance of Bernoulli distribution.

```
    Var(X) = p(1-p)
"""

validate_probability(p, "p")
return p * (1 - p)
```

# =============================================================================

# 07. BINOMIAL DISTRIBUTION

# =============================================================================

def combination(n: int, k: int) -> int:
"""
Calculate n choose k:

```
    C(n, k) = n! / (k!(n-k)!)

Uses math.comb internally.
"""

if not isinstance(n, int) or not isinstance(k, int):
    raise TypeError("n and k must be integers.")

if n < 0:
    raise ValueError("n must be non-negative.")

if k < 0 or k > n:
    return 0

return math.comb(n, k)
```

def binomial_pmf(k: int, n: int, p: float) -> float:
"""
Calculate Binomial probability.

```
A Binomial random variable counts successes in n independent
Bernoulli trials with constant success probability p.

Formula:

    P(X=k) = C(n,k) p^k (1-p)^(n-k)

Assumptions:

    1. Fixed number of trials.
    2. Two outcomes per trial.
    3. Independent trials.
    4. Constant probability of success.
"""

validate_probability(p, "p")

if n < 0:
    raise ValueError("n must be non-negative.")

if k < 0 or k > n:
    return 0.0

return (
    combination(n, k)
    * (p ** k)
    * ((1 - p) ** (n - k))
)
```

def binomial_mean(n: int, p: float) -> float:
"""
Mean of Binomial distribution.

```
    E[X] = np
"""

validate_probability(p, "p")
return n * p
```

def binomial_variance(n: int, p: float) -> float:
"""
Variance of Binomial distribution.

```
    Var(X) = np(1-p)
"""

validate_probability(p, "p")
return n * p * (1 - p)
```

# =============================================================================

# 08. GEOMETRIC DISTRIBUTION

# =============================================================================

def geometric_pmf(k: int, p: float) -> float:
"""
Calculate the Geometric PMF.

```
Here k represents the trial number on which the first success occurs.

Formula:

    P(X=k) = (1-p)^(k-1) p

for:

    k = 1, 2, 3, ...

Assumptions:

    - Independent trials
    - Constant probability p
    - We stop at the first success
"""

validate_probability(p, "p")

if p == 0:
    return 0.0

if k < 1:
    return 0.0

return ((1 - p) ** (k - 1)) * p
```

def geometric_mean(p: float) -> float:
"""
Mean of Geometric distribution.

```
    E[X] = 1/p
"""

validate_probability(p, "p")

if p == 0:
    raise ValueError("Mean is undefined when p = 0.")

return 1 / p
```

def geometric_variance(p: float) -> float:
"""
Variance of Geometric distribution.

```
    Var(X) = (1-p) / p^2
"""

validate_probability(p, "p")

if p == 0:
    raise ValueError("Variance is undefined when p = 0.")

return (1 - p) / (p ** 2)
```

# =============================================================================

# 09. POISSON DISTRIBUTION

# =============================================================================

def poisson_pmf(k: int, lam: float) -> float:
"""
Calculate Poisson PMF.

```
The Poisson distribution models the number of events occurring
in a fixed interval when events occur independently at an
approximately constant average rate.

Formula:

    P(X=k) = (lambda^k * e^(-lambda)) / k!

where:

    lambda > 0

Examples:

    - Number of support requests per hour
    - Number of arrivals per minute
    - Number of defects per meter
"""

validate_positive(lam, "lambda")

if k < 0:
    return 0.0

return (
    (lam ** k)
    * math.exp(-lam)
    / math.factorial(k)
)
```

def poisson_mean(lam: float) -> float:
"""
Mean of Poisson distribution.

```
    E[X] = lambda
"""

validate_positive(lam, "lambda")
return lam
```

def poisson_variance(lam: float) -> float:
"""
Variance of Poisson distribution.

```
    Var(X) = lambda
"""

validate_positive(lam, "lambda")
return lam
```

# =============================================================================

# 10. UNIFORM DISTRIBUTION

# =============================================================================

def uniform_pdf(x: float, a: float, b: float) -> float:
"""
Calculate PDF of Continuous Uniform distribution.

```
Formula:

    f(x) = 1 / (b-a)

for:

    a <= x <= b

Otherwise:

    f(x) = 0
"""

if a >= b:
    raise ValueError("a must be less than b.")

if a <= x <= b:
    return 1 / (b - a)

return 0.0
```

def uniform_cdf(x: float, a: float, b: float) -> float:
"""
Calculate CDF of Continuous Uniform distribution.

```
    F(x) = 0                  x < a
    F(x) = (x-a)/(b-a)        a <= x <= b
    F(x) = 1                  x > b
"""

if a >= b:
    raise ValueError("a must be less than b.")

if x < a:
    return 0.0

if x > b:
    return 1.0

return (x - a) / (b - a)
```

def uniform_mean(a: float, b: float) -> float:
"""
Mean:

```
    E[X] = (a+b)/2
"""

if a >= b:
    raise ValueError("a must be less than b.")

return (a + b) / 2
```

def uniform_variance(a: float, b: float) -> float:
"""
Variance:

```
    Var(X) = (b-a)^2 / 12
"""

if a >= b:
    raise ValueError("a must be less than b.")

return ((b - a) ** 2) / 12
```

# =============================================================================

# 11. NORMAL / GAUSSIAN DISTRIBUTION

# =============================================================================

def normal_pdf(x: float, mean: float = 0.0, std: float = 1.0) -> float:
"""
Calculate the Normal distribution PDF.

```
Formula:

            1
    f(x) = ----------- exp(-(x-mu)^2 / (2 sigma^2))
           sigma sqrt(2pi)

Parameters:

    mean = mu
    std  = sigma

The standard normal distribution has:

    mu = 0
    sigma = 1
"""

validate_positive(std, "std")

coefficient = 1 / (std * math.sqrt(2 * math.pi))

exponent = -((x - mean) ** 2) / (2 * std ** 2)

return coefficient * math.exp(exponent)
```

def normal_cdf(
x: float,
mean: float = 0.0,
std: float = 1.0,
) -> float:
"""
Calculate Normal CDF using the error function.

```
Formula:

    F(x) =
        1/2 * [1 + erf((x-mu)/(sigma*sqrt(2)))]
"""

validate_positive(std, "std")

z = (x - mean) / (std * math.sqrt(2))

return 0.5 * (1 + math.erf(z))
```

def normal_z_score(
x: float,
mean: float,
std: float,
) -> float:
"""
Calculate the z-score.

```
Formula:

    z = (x - mu) / sigma

Interpretation:

    z = 0
        exactly at the mean

    z = 1
        one standard deviation above mean

    z = -2
        two standard deviations below mean
"""

validate_positive(std, "std")

return (x - mean) / std
```

def normal_mean(mean: float) -> float:
"""
Mean of Normal distribution.
"""

```
return mean
```

def normal_variance(std: float) -> float:
"""
Variance of Normal distribution.

```
    Var(X) = sigma^2
"""

validate_positive(std, "std")

return std ** 2
```

# =============================================================================

# 12. EXPONENTIAL DISTRIBUTION

# =============================================================================

def exponential_pdf(x: float, lam: float) -> float:
"""
Calculate Exponential PDF.

```
Formula:

    f(x) = lambda * exp(-lambda*x)

for:

    x >= 0

where:

    lambda > 0

Common applications:

    - Waiting times
    - Time until failure
    - Inter-arrival times
"""

validate_positive(lam, "lambda")

if x < 0:
    return 0.0

return lam * math.exp(-lam * x)
```

def exponential_cdf(x: float, lam: float) -> float:
"""
Calculate Exponential CDF.

```
Formula:

    F(x) = 1 - exp(-lambda*x)

for x >= 0.
"""

validate_positive(lam, "lambda")

if x < 0:
    return 0.0

return 1 - math.exp(-lam * x)
```

def exponential_mean(lam: float) -> float:
"""
Mean:

```
    E[X] = 1/lambda
"""

validate_positive(lam, "lambda")

return 1 / lam
```

def exponential_variance(lam: float) -> float:
"""
Variance:

```
    Var(X) = 1/lambda^2
"""

validate_positive(lam, "lambda")

return 1 / (lam ** 2)
```

# =============================================================================

# 13. EXPECTED VALUE

# =============================================================================

def discrete_expected_value(
values: Sequence[float],
probabilities: Sequence[float],
) -> float:
"""
Calculate expected value of a discrete random variable.

```
Formula:

    E[X] = sum(x * P(X=x))
"""

if len(values) != len(probabilities):
    raise ValueError("values and probabilities must have same length.")

for probability in probabilities:
    validate_probability(probability)

if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
    raise ValueError("Probabilities must sum to 1.")

return sum(
    value * probability
    for value, probability in zip(values, probabilities)
)
```

# =============================================================================

# 14. DISCRETE VARIANCE

# =============================================================================

def discrete_variance(
values: Sequence[float],
probabilities: Sequence[float],
) -> float:
"""
Calculate variance of a discrete random variable.

```
Formula:

    Var(X) = E[(X - mu)^2]

Equivalent formula:

    Var(X) = E[X^2] - E[X]^2
"""

mean = discrete_expected_value(values, probabilities)

return sum(
    probability * ((value - mean) ** 2)
    for value, probability in zip(values, probabilities)
)
```

# =============================================================================

# 15. SAMPLING

# =============================================================================

def sample_bernoulli(p: float) -> int:
"""
Generate one Bernoulli random sample.
"""

```
validate_probability(p, "p")

return 1 if random.random() < p else 0
```

def sample_binomial(n: int, p: float) -> int:
"""
Generate one Binomial sample by simulating n Bernoulli trials.
"""

```
if n < 0:
    raise ValueError("n must be non-negative.")

return sum(sample_bernoulli(p) for _ in range(n))
```

def sample_geometric(p: float) -> int:
"""
Generate one Geometric sample.

```
Returns the trial number on which the first success occurs.
"""

validate_probability(p, "p")

if p == 0:
    raise ValueError(
        "A geometric trial never succeeds when p=0."
    )

trials = 1

while sample_bernoulli(p) == 0:
    trials += 1

return trials
```

def sample_poisson(lam: float) -> int:
"""
Generate one Poisson sample using Knuth's algorithm.

```
This implementation is educational.

For production work with large lambda values, prefer a mature
scientific library such as NumPy/SciPy.
"""

validate_positive(lam, "lambda")

threshold = math.exp(-lam)

count = 0
product = 1.0

while product > threshold:
    count += 1
    product *= random.random()

return count - 1
```

def sample_uniform(a: float, b: float) -> float:
"""
Generate a Continuous Uniform sample.
"""

```
if a >= b:
    raise ValueError("a must be less than b.")

return random.uniform(a, b)
```

def sample_normal(mean: float = 0.0, std: float = 1.0) -> float:
"""
Generate a Normal sample using Python's random module.
"""

```
validate_positive(std, "std")

return random.gauss(mean, std)
```

def sample_exponential(lam: float) -> float:
"""
Generate an Exponential sample using inverse transform sampling.

```
If:

    U ~ Uniform(0,1)

then:

    X = -ln(1-U) / lambda

follows an Exponential distribution.
"""

validate_positive(lam, "lambda")

u = random.random()

return -math.log(1 - u) / lam
```

# =============================================================================

# 16. EMPIRICAL MEAN AND VARIANCE

# =============================================================================

def sample_mean(data: Sequence[float]) -> float:
"""
Calculate arithmetic mean.
"""

```
if not data:
    raise ValueError("data cannot be empty.")

return sum(data) / len(data)
```

def sample_variance(
data: Sequence[float],
ddof: int = 1,
) -> float:
"""
Calculate sample variance.

```
Formula:

    s^2 = sum((x_i - x_bar)^2) / (n - ddof)

Default:

    ddof = 1

gives the usual unbiased sample variance estimator under
standard assumptions.
"""

n = len(data)

if n <= ddof:
    raise ValueError(
        "Not enough observations for the requested degrees of freedom."
    )

mean = sample_mean(data)

return sum(
    (value - mean) ** 2
    for value in data
) / (n - ddof)
```

# =============================================================================

# 17. DISTRIBUTION SUMMARY

# =============================================================================

def distribution_summary() -> None:
"""
Print a quick-reference table of common distributions.
"""

```
print("\n" + "=" * 80)
print("COMMON PROBABILITY DISTRIBUTIONS")
print("=" * 80)

rows = [
    ("Bernoulli", "Discrete", "One binary trial", "p"),
    ("Binomial", "Discrete", "Number of successes", "n, p"),
    ("Geometric", "Discrete", "Trials until first success", "p"),
    ("Poisson", "Discrete", "Event counts", "lambda"),
    ("Uniform", "Continuous", "Equal density over interval", "a, b"),
    ("Normal", "Continuous", "Bell-shaped measurements", "mu, sigma"),
    ("Exponential", "Continuous", "Waiting time", "lambda"),
]

print(
    f"{'Distribution':<15}"
    f"{'Type':<15}"
    f"{'Use Case':<35}"
    f"{'Parameters'}"
)

print("-" * 80)

for name, kind, use_case, parameters in rows:
    print(
        f"{name:<15}"
        f"{kind:<15}"
        f"{use_case:<35}"
        f"{parameters}"
    )
```

# =============================================================================

# 18. REAL-WORLD EXAMPLES

# =============================================================================

def real_world_examples() -> None:
"""
Demonstrate how distributions map to real ML/statistical problems.
"""

```
print("\n" + "=" * 80)
print("REAL-WORLD EXAMPLES")
print("=" * 80)

examples = {
    "Bernoulli": "Did a customer click the advertisement? Yes/No.",
    "Binomial": "How many customers clicked among 100 visitors?",
    "Geometric": "How many attempts until the first successful conversion?",
    "Poisson": "How many support tickets arrive per hour?",
    "Uniform": "A random value generated equally between two bounds.",
    "Normal": "Measurement noise or approximately bell-shaped features.",
    "Exponential": "Time until the next event in a Poisson process.",
}

for distribution, example in examples.items():
    print(f"\n{distribution}:")
    print(f"    {example}")
```

# =============================================================================

# 19. MACHINE LEARNING APPLICATIONS

# =============================================================================

def machine_learning_applications() -> None:
"""
Explain how probability distributions appear in Machine Learning.
"""

```
print("\n" + "=" * 80)
print("PROBABILITY DISTRIBUTIONS IN MACHINE LEARNING")
print("=" * 80)

print(
    """
```

1. Naive Bayes

---

Naive Bayes models:

```
P(class | features)
```

using Bayes' theorem and conditional probability.

Different feature distributions can lead to different Naive Bayes
variants.

Example:

```
Gaussian Naive Bayes
    assumes continuous features follow Gaussian distributions.
```

## 2. Logistic Regression

Logistic regression models a Bernoulli probability:

```
P(Y=1 | X)
```

The sigmoid function converts a real-valued score into a probability:

```
             1
sigmoid(z) = ------
             1 + e^(-z)
```

## 3. Linear Regression

A common probabilistic interpretation assumes:

```
Y = Xw + epsilon
```

where:

```
epsilon ~ Normal(0, sigma^2)
```

This connects least-squares regression to maximum likelihood.

4. Gaussian Mixture Models

---

A GMM models data as a mixture of Gaussian distributions.

Useful for:

```
- Clustering
- Density estimation
- Anomaly detection
```

## 5. Maximum Likelihood Estimation

MLE chooses parameters that maximize:

```
L(theta | data)
```

or more commonly:

```
log L(theta | data)
```

Probability distributions define the likelihood function.

6. Bayesian Machine Learning

---

Bayesian methods combine:

```
Prior
+
Likelihood
=
Posterior
```

## 7. Anomaly Detection

A model can estimate:

```
P(x)
```

Very unlikely observations may be flagged as anomalies.

However, low probability does not automatically mean an observation
is erroneous. Model assumptions and the application context matter.
"""
)

# =============================================================================

# 20. NUMPY IMPLEMENTATION

# =============================================================================

def numpy_examples() -> None:
"""
Demonstrate common distributions using NumPy.

```
NumPy is optional. The script remains runnable without it.
"""

print("\n" + "=" * 80)
print("NUMPY DISTRIBUTION EXAMPLES")
print("=" * 80)

try:
    import numpy as np
except ImportError:
    print("NumPy is not installed.")
    print("Install it with:")
    print("    pip install numpy")
    return

rng = np.random.default_rng(42)

bernoulli_samples = rng.binomial(
    n=1,
    p=0.7,
    size=10,
)

binomial_samples = rng.binomial(
    n=10,
    p=0.5,
    size=10,
)

poisson_samples = rng.poisson(
    lam=4,
    size=10,
)

normal_samples = rng.normal(
    loc=100,
    scale=15,
    size=10,
)

exponential_samples = rng.exponential(
    scale=2,
    size=10,
)

uniform_samples = rng.uniform(
    low=0,
    high=1,
    size=10,
)

print("Bernoulli:")
print(bernoulli_samples)

print("\nBinomial:")
print(binomial_samples)

print("\nPoisson:")
print(poisson_samples)

print("\nNormal:")
print(normal_samples)

print("\nExponential:")
print(exponential_samples)

print("\nUniform:")
print(uniform_samples)
```

# =============================================================================

# 21. SCIPY EXAMPLES

# =============================================================================

def scipy_examples() -> None:
"""
Demonstrate SciPy's probability distribution interface.

```
SciPy provides production-quality implementations of many
probability distributions.

The common API includes:

    pmf()
    pdf()
    cdf()
    ppf()
    rvs()

where:

    PMF = probability mass
    PDF = probability density
    CDF = cumulative probability
    PPF = inverse CDF / quantile
    RVS = random variates
"""

print("\n" + "=" * 80)
print("SCIPY DISTRIBUTION EXAMPLES")
print("=" * 80)

try:
    from scipy import stats
except ImportError:
    print("SciPy is not installed.")
    print("Install it with:")
    print("    pip install scipy")
    return

print(
    "Binomial P(X=3), n=10, p=0.5:",
    stats.binom.pmf(3, n=10, p=0.5),
)

print(
    "Poisson P(X=4), lambda=3:",
    stats.poisson.pmf(4, mu=3),
)

print(
    "Normal PDF at x=0:",
    stats.norm.pdf(0),
)

print(
    "Normal CDF at x=1.96:",
    stats.norm.cdf(1.96),
)

print(
    "95th percentile of standard normal:",
    stats.norm.ppf(0.95),
)
```

# =============================================================================

# 22. DISTRIBUTION SELECTION GUIDE

# =============================================================================

def distribution_selection_guide() -> None:
"""
Explain how to select a distribution based on the problem.
"""

```
print("\n" + "=" * 80)
print("DISTRIBUTION SELECTION GUIDE")
print("=" * 80)

print(
    """
```

Ask these questions:

1. Is the variable discrete or continuous?

2. Are you counting events or measuring quantities?

3. Is there a fixed number of trials?

4. Are there exactly two outcomes?

5. Are you counting events within a fixed interval?

6. Are you modeling waiting time?

7. Does the data appear approximately bell-shaped?

Examples:

```
Binary outcome
    -> Bernoulli

Number of successes in n trials
    -> Binomial

Trials until first success
    -> Geometric

Event count in a fixed interval
    -> Poisson

Equal likelihood across an interval
    -> Uniform

Approximately bell-shaped continuous variable
    -> Normal

Waiting time until an event
    -> Exponential
```

Important:

```
Do not choose a distribution solely because it is convenient.

Check whether its assumptions are reasonable for your data-generating
process.
```

"""
)

# =============================================================================

# 23. COMPARING THE DISTRIBUTIONS

# =============================================================================

def compare_distributions() -> None:
"""
Print mathematical properties of common distributions.
"""

```
print("\n" + "=" * 80)
print("DISTRIBUTION FORMULA QUICK REFERENCE")
print("=" * 80)

formulas = {
    "Bernoulli":
        "P(X=x) = p^x (1-p)^(1-x)",

    "Binomial":
        "P(X=k) = C(n,k) p^k (1-p)^(n-k)",

    "Geometric":
        "P(X=k) = (1-p)^(k-1) p",

    "Poisson":
        "P(X=k) = lambda^k e^(-lambda) / k!",

    "Uniform":
        "f(x) = 1/(b-a)",

    "Normal":
        "f(x) = [1/(sigma sqrt(2pi))] exp(-(x-mu)^2/(2sigma^2))",

    "Exponential":
        "f(x) = lambda exp(-lambda x)",
}

for name, formula in formulas.items():
    print(f"\n{name}:")
    print(f"    {formula}")
```

# =============================================================================

# 24. PRACTICAL DEMONSTRATION

# =============================================================================

def practical_examples() -> None:
"""
Run practical examples for the major distributions.
"""

```
print("\n" + "=" * 80)
print("PRACTICAL DISTRIBUTION EXAMPLES")
print("=" * 80)

# -------------------------------------------------------------------------
# Bernoulli
# -------------------------------------------------------------------------

p = 0.7

print("\nBernoulli Distribution")
print("----------------------")
print(f"P(X=1) = {bernoulli_pmf(1, p):.4f}")
print(f"P(X=0) = {bernoulli_pmf(0, p):.4f}")
print(f"Mean    = {bernoulli_mean(p):.4f}")
print(f"Variance= {bernoulli_variance(p):.4f}")

# -------------------------------------------------------------------------
# Binomial
# -------------------------------------------------------------------------

n = 10
p = 0.5
k = 6

print("\nBinomial Distribution")
print("---------------------")
print(
    f"P(X={k}) when n={n}, p={p}: "
    f"{binomial_pmf(k, n, p):.6f}"
)
print(f"Mean     = {binomial_mean(n, p):.4f}")
print(f"Variance = {binomial_variance(n, p):.4f}")

# -------------------------------------------------------------------------
# Geometric
# -------------------------------------------------------------------------

p = 0.25
k = 4

print("\nGeometric Distribution")
print("----------------------")
print(
    f"P(first success on trial {k}) = "
    f"{geometric_pmf(k, p):.6f}"
)
print(f"Mean     = {geometric_mean(p):.4f}")
print(f"Variance = {geometric_variance(p):.4f}")

# -------------------------------------------------------------------------
# Poisson
# -------------------------------------------------------------------------

lam = 3
k = 4

print("\nPoisson Distribution")
print("--------------------")
print(
    f"P(X={k}) when lambda={lam}: "
    f"{poisson_pmf(k, lam):.6f}"
)
print(f"Mean     = {poisson_mean(lam):.4f}")
print(f"Variance = {poisson_variance(lam):.4f}")

# -------------------------------------------------------------------------
# Uniform
# -------------------------------------------------------------------------

a = 0
b = 10
x = 5

print("\nUniform Distribution")
print("--------------------")
print(f"PDF at x={x}: {uniform_pdf(x, a, b):.4f}")
print(f"CDF at x={x}: {uniform_cdf(x, a, b):.4f}")
print(f"Mean:          {uniform_mean(a, b):.4f}")
print(f"Variance:      {uniform_variance(a, b):.4f}")

# -------------------------------------------------------------------------
# Normal
# -------------------------------------------------------------------------

mean = 100
std = 15
x = 130

print("\nNormal Distribution")
print("-------------------")
print(f"PDF at x={x}: {normal_pdf(x, mean, std):.6f}")
print(f"CDF at x={x}: {normal_cdf(x, mean, std):.6f}")
print(f"Z-score:       {normal_z_score(x, mean, std):.4f}")
print(f"Variance:      {normal_variance(std):.4f}")

# -------------------------------------------------------------------------
# Exponential
# -------------------------------------------------------------------------

lam = 0.5
x = 2

print("\nExponential Distribution")
print("------------------------")
print(f"PDF at x={x}: {exponential_pdf(x, lam):.6f}")
print(f"CDF at x={x}: {exponential_cdf(x, lam):.6f}")
print(f"Mean:          {exponential_mean(lam):.4f}")
print(f"Variance:      {exponential_variance(lam):.4f}")
```

# =============================================================================

# 25. EXPECTED VALUE AND VARIANCE EXAMPLE

# =============================================================================

def expected_value_and_variance_example() -> None:
"""
Demonstrate expected value and variance using a fair die.
"""

```
print("\n" + "=" * 80)
print("EXPECTED VALUE AND VARIANCE")
print("=" * 80)

values = [1, 2, 3, 4, 5, 6]
probabilities = [1 / 6] * 6

mean = discrete_expected_value(values, probabilities)

variance = discrete_variance(
    values,
    probabilities,
)

print("Fair die:")
print(f"Expected value = {mean:.4f}")
print(f"Variance       = {variance:.4f}")
print(f"Std deviation  = {math.sqrt(variance):.4f}")
```

# =============================================================================

# 26. SAMPLING DEMONSTRATION

# =============================================================================

def sampling_demo() -> None:
"""
Demonstrate random sampling from common distributions.
"""

```
print("\n" + "=" * 80)
print("SAMPLING DEMONSTRATION")
print("=" * 80)

random.seed(42)

print("Bernoulli samples:")
print([sample_bernoulli(0.7) for _ in range(10)])

print("\nBinomial samples:")
print([sample_binomial(10, 0.5) for _ in range(10)])

print("\nPoisson samples:")
print([sample_poisson(3) for _ in range(10)])

print("\nUniform samples:")
print([round(sample_uniform(0, 1), 4) for _ in range(10)])

print("\nNormal samples:")
print([round(sample_normal(100, 15), 4) for _ in range(10)])

print("\nExponential samples:")
print([round(sample_exponential(0.5), 4) for _ in range(10)])
```

# =============================================================================

# 27. COMMON MISTAKES

# =============================================================================

def common_mistakes() -> None:
"""
Explain common probability distribution mistakes.
"""

```
print("\n" + "=" * 80)
print("COMMON MISTAKES")
print("=" * 80)

mistakes = [
    (
        "Confusing PMF and PDF",
        "PMF gives probability for discrete outcomes; PDF gives density."
    ),
    (
        "Treating PDF(x) as P(X=x)",
        "For continuous distributions, exact-point probability is generally 0."
    ),
    (
        "Ignoring distribution assumptions",
        "A mathematical formula is useful only when its assumptions are reasonable."
    ),
    (
        "Confusing variance and standard deviation",
        "Variance is measured in squared units; standard deviation is in original units."
    ),
    (
        "Using Binomial without independent trials",
        "Binomial assumes independent Bernoulli trials with constant p."
    ),
    (
        "Using Poisson automatically for every count",
        "Poisson assumes a particular event-counting model and rate structure."
    ),
    (
        "Assuming Normal distribution for all data",
        "Many real-world variables are not normally distributed."
    ),
    (
        "Ignoring parameter interpretation",
        "Always understand what p, n, lambda, mu, sigma, a, and b represent."
    ),
    (
        "Confusing probability with causation",
        "A probability relationship does not by itself prove a causal relationship."
    ),
    (
        "Multiplying probabilities without checking dependence",
        "P(A and B) = P(A)P(B) only when A and B are independent."
    ),
]

for mistake, explanation in mistakes:
    print(f"\n{mistake}")
    print(f"    {explanation}")
```

# =============================================================================

# 28. COMPLETE ANALYSIS

# =============================================================================

def complete_analysis() -> None:
"""
Demonstrate a complete probability-distribution workflow.
"""

```
print("\n" + "=" * 80)
print("COMPLETE DISTRIBUTION ANALYSIS")
print("=" * 80)

data = [
    82,
    85,
    88,
    90,
    91,
    93,
    95,
    97,
    100,
    102,
]

mean = sample_mean(data)
variance = sample_variance(data)
std = math.sqrt(variance)

print("Observed data:")
print(data)

print("\nDescriptive statistics:")
print(f"Mean               = {mean:.4f}")
print(f"Sample variance    = {variance:.4f}")
print(f"Sample std dev     = {std:.4f}")

value = 100

z = normal_z_score(
    value,
    mean,
    std,
)

print("\nNormal approximation:")
print(f"Value              = {value}")
print(f"Z-score            = {z:.4f}")

probability_below = normal_cdf(
    value,
    mean,
    std,
)

print(
    f"Approx. P(X <= {value}) "
    f"under fitted Normal model = "
    f"{probability_below:.4f}"
)

print(
    """
```

Important:

The final probability above is based on a Normal model fitted using
the sample mean and sample standard deviation.

It is a model-based approximation, not a guarantee that the underlying
population is actually Normal.
"""
)

# =============================================================================

# 29. MAIN

# =============================================================================

def main() -> None:
"""
Run all major demonstrations.
"""

```
print("=" * 80)
print("PROBABILITY DISTRIBUTIONS FOR MACHINE LEARNING")
print("=" * 80)

explain_probability_distribution()

explain_pdf()

explain_cdf()

distribution_summary()

compare_distributions()

practical_examples()

expected_value_and_variance_example()

real_world_examples()

machine_learning_applications()

distribution_selection_guide()

sampling_demo()

numpy_examples()

scipy_examples()

complete_analysis()

common_mistakes()

print("\n" + "=" * 80)
print("END OF PROBABILITY DISTRIBUTIONS")
print("=" * 80)
```

# =============================================================================

# 30. SCRIPT ENTRY POINT

# =============================================================================

if **name** == "**main**":
main()
"""

## Learning Roadmap

A recommended progression is:

```
1. Random variables
2. PMF
3. PDF
4. CDF
5. Expected value
6. Variance
7. Bernoulli
8. Binomial
9. Geometric
10. Poisson
11. Uniform
12. Normal
13. Exponential
14. Joint distributions
15. Conditional distributions
16. Bayes' theorem
17. Maximum Likelihood Estimation
18. Maximum A Posteriori estimation
19. Probabilistic Machine Learning
```

## Quick Reference

Bernoulli
Purpose:
One binary trial.

```
Parameters:
    p

Mean:
    p

Variance:
    p(1-p)
```

Binomial
Purpose:
Number of successes in n Bernoulli trials.

```
Parameters:
    n, p

Mean:
    np

Variance:
    np(1-p)
```

Geometric
Purpose:
Trial number of the first success.

```
Parameter:
    p

Mean:
    1/p

Variance:
    (1-p)/p^2
```

Poisson
Purpose:
Number of events in a fixed interval.

```
Parameter:
    lambda

Mean:
    lambda

Variance:
    lambda
```

Uniform
Purpose:
Equal density over an interval.

```
Parameters:
    a, b

Mean:
    (a+b)/2

Variance:
    (b-a)^2 / 12
```

Normal
Purpose:
Continuous bell-shaped distribution.

```
Parameters:
    mu, sigma

Mean:
    mu

Variance:
    sigma^2
```

Exponential
Purpose:
Waiting time.

```
Parameter:
    lambda

Mean:
    1/lambda

Variance:
    1/lambda^2
```

## Final ML Connection

Probability distributions provide the mathematical language for uncertainty.

A typical Machine Learning pipeline can be viewed as:

```
Data
  |
  v
Random Variables
  |
  v
Probability Distribution
  |
  v
Likelihood
  |
  v
Parameter Estimation
  |
  v
Statistical / ML Model
  |
  v
Prediction + Uncertainty
```

Understanding distributions deeply makes later topics much easier:

```
Probability
    ->
Statistics
    ->
Likelihood
    ->
Optimization
    ->
Machine Learning
    ->
Probabilistic Machine Learning
```

"""
