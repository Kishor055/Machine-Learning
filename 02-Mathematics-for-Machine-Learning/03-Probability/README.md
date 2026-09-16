# 🎲 Probability for Machine Learning

> **Probability is the mathematical language of uncertainty.**
> Machine Learning uses probability to model uncertainty, estimate likelihoods, make predictions, and reason about incomplete or noisy data.

---

## 📚 Table of Contents

* [1. What is Probability?](#1-what-is-probability)
* [2. Why Probability Matters in Machine Learning](#2-why-probability-matters-in-machine-learning)
* [3. Probability vs Statistics](#3-probability-vs-statistics)
* [4. Random Experiments](#4-random-experiments)
* [5. Sample Space](#5-sample-space)
* [6. Events](#6-events)
* [7. Types of Events](#7-types-of-events)
* [8. Probability of an Event](#8-probability-of-an-event)
* [9. Probability Axioms](#9-probability-axioms)
* [10. Complement Rule](#10-complement-rule)
* [11. Addition Rule](#11-addition-rule)
* [12. Multiplication Rule](#12-multiplication-rule)
* [13. Conditional Probability](#13-conditional-probability)
* [14. Independence](#14-independence)
* [15. Bayes' Theorem](#15-bayes-theorem)
* [16. Random Variables](#16-random-variables)
* [17. Discrete Random Variables](#17-discrete-random-variables)
* [18. Continuous Random Variables](#18-continuous-random-variables)
* [19. Probability Mass Function](#19-probability-mass-function)
* [20. Probability Density Function](#20-probability-density-function)
* [21. Cumulative Distribution Function](#21-cumulative-distribution-function)
* [22. Expected Value](#22-expected-value)
* [23. Variance and Standard Deviation](#23-variance-and-standard-deviation)
* [24. Bernoulli Distribution](#24-bernoulli-distribution)
* [25. Binomial Distribution](#25-binomial-distribution)
* [26. Geometric Distribution](#26-geometric-distribution)
* [27. Poisson Distribution](#27-poisson-distribution)
* [28. Uniform Distribution](#28-uniform-distribution)
* [29. Normal Distribution](#29-normal-distribution)
* [30. Exponential Distribution](#30-exponential-distribution)
* [31. Standard Normal Distribution](#31-standard-normal-distribution)
* [32. Z-Scores](#32-z-scores)
* [33. Central Limit Theorem](#33-central-limit-theorem)
* [34. Law of Large Numbers](#34-law-of-large-numbers)
* [35. Joint Probability](#35-joint-probability)
* [36. Marginal Probability](#36-marginal-probability)
* [37. Conditional Distributions](#37-conditional-distributions)
* [38. Independence vs Dependence](#38-independence-vs-dependence)
* [39. Covariance and Probability](#39-covariance-and-probability)
* [40. Probability in Machine Learning](#40-probability-in-machine-learning)
* [41. Probabilistic Classification](#41-probabilistic-classification)
* [42. Maximum Likelihood Estimation](#42-maximum-likelihood-estimation)
* [43. Maximum A Posteriori Estimation](#43-maximum-a-posteriori-estimation)
* [44. Probability and Loss Functions](#44-probability-and-loss-functions)
* [45. Common Mistakes](#45-common-mistakes)
* [46. Python Libraries](#46-python-libraries)
* [47. Practice Problems](#47-practice-problems)
* [48. Learning Roadmap](#48-learning-roadmap)
* [49. Quick Reference](#49-quick-reference)
* [50. Key Takeaways](#50-key-takeaways)

---

# 1. What is Probability?

**Probability** is a mathematical framework for measuring uncertainty.

It assigns a numerical value between `0` and `1` to an event:

$$
0 \leq P(A) \leq 1
$$

where:

* `0` → event is impossible
* `1` → event is certain
* values between `0` and `1` → different degrees of likelihood

For example:

```text
P(rain tomorrow) = 0.70
```

can be interpreted as a probability of `70%` under a particular probabilistic model.

Probability is fundamental to Machine Learning because real-world data is rarely perfectly deterministic.

---

# 2. Why Probability Matters in Machine Learning

Machine Learning deals with uncertainty everywhere.

Examples:

```text
Will this email be spam?
Will this customer purchase a product?
Will this patient have a disease?
What class does this image belong to?
What is the probability of default?
How confident is the model?
```

Instead of producing only:

```text
Spam
```

a model can produce:

```text
P(Spam) = 0.92
P(Not Spam) = 0.08
```

This probability can then be used for decision-making.

### Major ML applications

Probability is used in:

* Classification
* Regression
* Bayesian inference
* Naive Bayes
* Logistic Regression
* Hidden Markov Models
* Bayesian Networks
* Generative Models
* Probabilistic Graphical Models
* Uncertainty Estimation
* A/B Testing
* Recommendation Systems
* Anomaly Detection
* Risk Modeling
* Medical Diagnosis
* Natural Language Processing
* Computer Vision

---

# 3. Probability vs Statistics

Probability and Statistics are closely related but solve different problems.

### Probability

Starts with a known model and reasons about possible outcomes.

```text
Model → Data/Outcome
```

Example:

> If a fair coin is tossed, what is the probability of getting heads?

---

### Statistics

Starts with observed data and tries to understand the underlying process.

```text
Data → Model/Inference
```

Example:

> A coin was tossed 1,000 times and produced 530 heads. Is the coin likely to be fair?

### Simple distinction

| Probability         | Statistics              |
| ------------------- | ----------------------- |
| Model → outcomes    | Data → model            |
| Forward reasoning   | Inference               |
| Predict uncertainty | Learn from observations |
| Theoretical         | Data-driven             |

Machine Learning uses **both**.

---

# 4. Random Experiments

A **random experiment** is a process whose exact outcome cannot be known in advance.

Examples:

* Tossing a coin
* Rolling a die
* Drawing a card
* Selecting a random customer
* Measuring a random person's height

Although the individual outcome may be uncertain, the possible outcomes can be described mathematically.

---

# 5. Sample Space

The **sample space** is the set of all possible outcomes of a random experiment.

It is usually represented by:

$$
S
$$

### Example: Coin

```text
S = {Heads, Tails}
```

### Example: Die

```text
S = {1, 2, 3, 4, 5, 6}
```

### Example: Two coin tosses

```text
S = {
    HH,
    HT,
    TH,
    TT
}
```

The sample space is important because probabilities are defined over possible outcomes.

---

# 6. Events

An **event** is a subset of the sample space.

Suppose:

```text
S = {1, 2, 3, 4, 5, 6}
```

Event `A` = rolling an even number:

```text
A = {2, 4, 6}
```

Therefore:

$$
A \subseteq S
$$

---

# 7. Types of Events

## 7.1 Simple Event

Contains one outcome.

```text
A = {3}
```

---

## 7.2 Compound Event

Contains multiple outcomes.

```text
A = {2, 4, 6}
```

---

## 7.3 Certain Event

An event that always occurs.

$$
P(A)=1
$$

---

## 7.4 Impossible Event

An event that can never occur.

$$
P(A)=0
$$

---

## 7.5 Mutually Exclusive Events

Two events cannot occur simultaneously.

Example:

```text
Rolling 2
Rolling 5
```

in the same single die roll.

Therefore:

$$
P(A \cap B)=0
$$

---

## 7.6 Exhaustive Events

A collection of events that covers all possible outcomes.

---

# 8. Probability of an Event

For equally likely outcomes:

$$
P(A)=
\frac{\text{Number of favorable outcomes}}
{\text{Total number of possible outcomes}}
$$

### Example

Probability of rolling an even number:

```text
Favorable outcomes = {2,4,6}
Total outcomes = 6
```

Therefore:

$$
P(A)=\frac{3}{6}=0.5
$$

or:

```text
50%
```

---

# 9. Probability Axioms

Probability follows three fundamental axioms.

## Axiom 1 — Non-Negativity

$$
P(A)\geq0
$$

---

## Axiom 2 — Total Probability

$$
P(S)=1
$$

The probability of the entire sample space is `1`.

---

## Axiom 3 — Additivity

For mutually exclusive events:

$$
P(A\cup B)=P(A)+P(B)
$$

These axioms form the mathematical foundation of probability theory.

---

# 10. Complement Rule

The complement of event `A` represents the event that `A` does not occur.

It is written as:

$$
A^c
$$

The complement rule is:

$$
P(A^c)=1-P(A)
$$

### Example

If:

$$
P(A)=0.8
$$

then:

$$
P(A^c)=1-0.8=0.2
$$

---

# 11. Addition Rule

The general addition rule is:

$$
P(A\cup B)
=
P(A)+P(B)-P(A\cap B)
$$

The intersection is subtracted because it would otherwise be counted twice.

### For mutually exclusive events

If:

$$
P(A\cap B)=0
$$

then:

$$
P(A\cup B)=P(A)+P(B)
$$

---

# 12. Multiplication Rule

The general multiplication rule is:

$$
P(A\cap B)
=
P(A)P(B|A)
$$

where:

$$
P(B|A)
$$

means the probability of `B` given that `A` has occurred.

Similarly:

$$
P(A\cap B)
=
P(B)P(A|B)
$$

---

# 13. Conditional Probability

Conditional probability measures the probability of an event given that another event has occurred.

It is written as:

$$
P(A|B)
$$

and defined as:

$$
P(A|B)
=
\frac{P(A\cap B)}{P(B)}
$$

provided:

$$
P(B)>0
$$

### Example

Suppose:

```text
P(A ∩ B) = 0.2
P(B) = 0.5
```

Then:

$$
P(A|B)=\frac{0.2}{0.5}=0.4
$$

---

# 14. Independence

Two events are independent when knowing that one occurred does not change the probability of the other.

For independent events:

$$
P(A|B)=P(A)
$$

and:

$$
P(B|A)=P(B)
$$

Therefore:

$$
P(A\cap B)=P(A)P(B)
$$

### Example

For independent coin tosses:

```text
P(Heads on toss 2 | Heads on toss 1)
=
P(Heads on toss 2)
```

The first toss does not affect the second toss.

---

# 15. Bayes' Theorem

Bayes' theorem is one of the most important equations in probabilistic Machine Learning.

$$
P(A|B)
=
\frac{P(B|A)P(A)}
{P(B)}
$$

Where:

* `P(A|B)` → posterior
* `P(B|A)` → likelihood
* `P(A)` → prior
* `P(B)` → evidence

### Bayes' theorem structure

```text
Posterior
    =
Likelihood × Prior
------------------
Evidence
```

### Why it matters

Bayesian reasoning allows us to update beliefs when new evidence becomes available.

Applications include:

* Spam detection
* Medical diagnosis
* Bayesian classification
* Fault detection
* Fraud detection
* Risk analysis

---

# 16. Random Variables

A **random variable** maps outcomes of a random experiment to numerical values.

It is commonly represented by:

$$
X
$$

Example:

```text
Experiment:
Toss a coin

X = number of heads
```

Possible values:

```text
X ∈ {0, 1}
```

Random variables can be:

1. Discrete
2. Continuous

---

# 17. Discrete Random Variables

A discrete random variable takes countable values.

Examples:

* Number of customers
* Number of defective products
* Number of heads
* Number of website clicks
* Number of accidents

Example:

```text
X = number of heads in 3 coin tosses

X ∈ {0, 1, 2, 3}
```

---

# 18. Continuous Random Variables

A continuous random variable can take values from an interval.

Examples:

* Height
* Weight
* Temperature
* Time
* Blood pressure
* House price

For a continuous random variable:

$$
P(X=x)=0
$$

for an exact individual point under the usual continuous model.

Instead, probabilities are assigned to intervals:

$$
P(a<X<b)
$$

---

# 19. Probability Mass Function

A **Probability Mass Function (PMF)** describes probabilities for a discrete random variable.

$$
P(X=x)
$$

A valid PMF must satisfy:

$$
P(X=x)\geq0
$$

and:

$$
\sum_x P(X=x)=1
$$

### Example

Suppose:

```text
X = number of heads in one coin toss
```

For a fair coin:

```text
P(X=0) = 0.5
P(X=1) = 0.5
```

---

# 20. Probability Density Function

A **Probability Density Function (PDF)** describes the distribution of a continuous random variable.

It is represented by:

$$
f(x)
$$

A PDF satisfies:

$$
f(x)\geq0
$$

and:

$$
\int_{-\infty}^{\infty}f(x)\,dx=1
$$

For continuous variables, probability is the **area under the density curve**:

$$
P(a\leq X\leq b)
=
\int_a^b f(x)\,dx
$$

A PDF value itself is not generally the probability of an exact point.

---

# 21. Cumulative Distribution Function

The **CDF** gives the probability that a random variable is less than or equal to a particular value.

$$
F(x)=P(X\leq x)
$$

Properties:

$$
0\leq F(x)\leq1
$$

and:

$$
\lim_{x\rightarrow-\infty}F(x)=0
$$

$$
\lim_{x\rightarrow\infty}F(x)=1
$$

For continuous distributions:

$$
F(x)=\int_{-\infty}^{x}f(t)\,dt
$$

---

# 22. Expected Value

The **expected value** represents the long-run average of a random variable under a specified probability distribution.

For a discrete random variable:

$$
E[X]
=
\sum_x xP(X=x)
$$

For a continuous random variable:

$$
E[X]
=
\int_{-\infty}^{\infty}x f(x)\,dx
$$

### Example

Suppose:

```text
X = 1 with probability 0.5
X = 3 with probability 0.5
```

Then:

$$
E[X]
=
1(0.5)+3(0.5)
$$

$$
E[X]=2
$$

The expected value does not necessarily have to be one of the possible outcomes.

---

# 23. Variance and Standard Deviation

Variance measures the spread of a random variable around its expected value.

$$
Var(X)=E[(X-E[X])^2]
$$

An equivalent form is:

$$
Var(X)=E[X^2]-E[X]^2
$$

Standard deviation is:

$$
\sigma=\sqrt{Var(X)}
$$

Variance is expressed in squared units, while standard deviation is expressed in the original units.

See:

```text
02-Mathematics-for-Machine-Learning/02-Statistics/
```

for detailed implementations.

---

# 24. Bernoulli Distribution

The Bernoulli distribution models a single trial with exactly two possible outcomes.

Usually:

```text
Success = 1
Failure = 0
```

Let:

$$
P(X=1)=p
$$

Then:

$$
P(X=0)=1-p
$$

The PMF is:

$$
P(X=x)
=
p^x(1-p)^{1-x}
$$

for:

$$
x\in\{0,1\}
$$

### Mean

$$
E[X]=p
$$

### Variance

$$
Var(X)=p(1-p)
$$

Applications:

* Click/no-click
* Purchase/no-purchase
* Disease/no-disease
* Spam/not spam

---

# 25. Binomial Distribution

The Binomial distribution models the number of successes in `n` independent Bernoulli trials with the same success probability `p`.

$$
X\sim Binomial(n,p)
$$

PMF:

$$
P(X=k)
=
\binom{n}{k}
p^k(1-p)^{n-k}
$$

where:

$$
\binom{n}{k}
=
\frac{n!}{k!(n-k)!}
$$

### Mean

$$
E[X]=np
$$

### Variance

$$
Var(X)=np(1-p)
$$

Example:

> Number of customers who click an advertisement out of 100 independent opportunities.

---

# 26. Geometric Distribution

The geometric distribution models the number of trials until the first success, under a common convention.

For:

$$
X=1,2,3,\dots
$$

the PMF is:

$$
P(X=k)
=
(1-p)^{k-1}p
$$

### Mean

$$
E[X]=\frac{1}{p}
$$

### Example

Number of attempts required to obtain the first successful login.

> **Note:** Some textbooks define the geometric variable as the number of failures before the first success. Always check the convention being used.

---

# 27. Poisson Distribution

The Poisson distribution models the number of events occurring in a fixed interval when events are modeled using a constant average rate under the Poisson assumptions.

$$
X\sim Poisson(\lambda)
$$

PMF:

$$
P(X=k)
=
\frac{e^{-\lambda}\lambda^k}{k!}
$$

where:

* `λ` = average number of events
* `k` = observed number of events

### Mean

$$
E[X]=\lambda
$$

### Variance

$$
Var(X)=\lambda
$$

Applications:

* Number of calls per minute
* Website requests
* Server failures
* Arrivals
* Defects

The Poisson model has assumptions; it should not automatically be applied to every count dataset.

---

# 28. Uniform Distribution

A uniform distribution assigns equal density across an interval.

For:

$$
X\sim Uniform(a,b)
$$

the PDF is:

$$
f(x)=\frac{1}{b-a}
$$

for:

$$
a\leq x\leq b
$$

### Mean

$$
E[X]=\frac{a+b}{2}
$$

### Variance

$$
Var(X)=\frac{(b-a)^2}{12}
$$

---

# 29. Normal Distribution

The **Normal distribution** is one of the most important probability distributions in statistics and Machine Learning.

It is characterized by:

* Mean `μ`
* Standard deviation `σ`

Written as:

$$
X\sim N(\mu,\sigma^2)
$$

PDF:

$$
f(x)
=
\frac{1}{\sigma\sqrt{2\pi}}
e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

The distribution is:

* Symmetric around `μ`
* Bell-shaped
* Defined over all real numbers

Many statistical methods and ML techniques use normality assumptions, approximations, or transformations.

---

# 30. Exponential Distribution

The Exponential distribution is commonly used to model waiting times under a constant-rate Poisson process.

For rate parameter `λ`:

$$
f(x)=\lambda e^{-\lambda x}
$$

for:

$$
x\geq0
$$

### Mean

$$
E[X]=\frac{1}{\lambda}
$$

### Variance

$$
Var(X)=\frac{1}{\lambda^2}
$$

Applications can include modeled:

* Waiting times
* Time between events
* Reliability
* Service systems

The assumptions of the model matter when applying it to real data.

---

# 31. Standard Normal Distribution

The standard normal distribution has:

$$
\mu=0
$$

and:

$$
\sigma=1
$$

It is commonly written:

$$
Z\sim N(0,1)
$$

Many normal-distribution probabilities can be converted to this standard form.

---

# 32. Z-Scores

A **z-score** tells us how many standard deviations an observation is from the mean.

$$
z=
\frac{x-\mu}{\sigma}
$$

For a sample, the analogous standardized score may use the sample mean and sample standard deviation.

### Interpretation

```text
z = 0
```

means the observation is at the mean.

```text
z = +2
```

means it is two standard deviations above the mean.

```text
z = -1
```

means it is one standard deviation below the mean.

Z-scores are used in:

* Standardization
* Outlier analysis
* Statistical modeling
* Feature preprocessing
* Normal-distribution calculations

---

# 33. Central Limit Theorem

The **Central Limit Theorem (CLT)** is a foundational result in statistics.

Under suitable conditions, the distribution of the sample mean becomes approximately normal as sample size increases, even when the underlying population is not normal.

Conceptually:

```text
Population
     ↓
Repeated Samples
     ↓
Calculate Sample Means
     ↓
Distribution of Means
     ↓
Approximately Normal
```

The exact conditions and quality of the approximation depend on the underlying distribution and sample size.

The CLT supports many statistical procedures and approximations.

---

# 34. Law of Large Numbers

The **Law of Large Numbers** states, informally, that as the number of observations increases, an appropriate sample average tends to approach its expected value under suitable assumptions.

Example:

A fair coin has:

$$
P(Heads)=0.5
$$

A small number of tosses might produce:

```text
Heads = 70%
```

but over a sufficiently large number of independent tosses, the observed proportion tends to get closer to:

```text
50%
```

### Important distinction

```text
Law of Large Numbers
→ sample averages stabilize toward expected values

Central Limit Theorem
→ describes the distribution of appropriately normalized sample statistics
```

They are related but not the same theorem.

---

# 35. Joint Probability

Joint probability measures the probability that two or more events occur together.

$$
P(A\cap B)
$$

For random variables:

$$
P(X=x,Y=y)
$$

is a joint probability for discrete variables.

Joint distributions are fundamental to:

* Bayesian models
* Graphical models
* Multivariate statistics
* Probabilistic ML

---

# 36. Marginal Probability

A marginal probability describes one variable without conditioning on the specific value of another variable.

For discrete random variables:

$$
P(X=x)
=
\sum_y P(X=x,Y=y)
$$

This process is often called **marginalization**.

---

# 37. Conditional Distributions

A conditional distribution describes one random variable given information about another.

For example:

$$
P(Y|X)
$$

In Machine Learning, conditional probability is especially important because prediction can be expressed as:

$$
P(Y|X)
$$

where:

* `X` = observed features
* `Y` = target

A classification model may estimate:

$$
P(Y=k|X=x)
$$

for each class `k`.

---

# 38. Independence vs Dependence

### Independent

$$
P(A\cap B)=P(A)P(B)
$$

### Dependent

The occurrence of one event changes the probability of another.

For dependent events:

$$
P(A|B)\neq P(A)
$$

### Important ML distinction

Statistical dependence does not automatically imply a causal relationship.

Two variables can be associated because of:

* A common cause
* Selection effects
* Confounding
* Measurement processes
* Other dependencies

---

# 39. Covariance and Probability

For random variables `X` and `Y`:

$$
Cov(X,Y)
=
E[(X-E[X])(Y-E[Y])]
$$

Equivalent form:

$$
Cov(X,Y)
=
E[XY]-E[X]E[Y]
$$

Covariance measures how variables vary together.

Correlation standardizes covariance:

$$
\rho_{X,Y}
=
\frac{Cov(X,Y)}
{\sigma_X\sigma_Y}
$$

Correlation therefore has a standardized range:

$$
-1\leq\rho\leq1
$$

See:

```text
02-Mathematics-for-Machine-Learning/02-Statistics/
```

for detailed covariance and correlation implementations.

---

# 40. Probability in Machine Learning

Probability appears throughout the ML pipeline.

```text
Raw Data
   ↓
Probability Distributions
   ↓
Statistical Modeling
   ↓
Parameter Estimation
   ↓
Model Training
   ↓
Prediction
   ↓
Uncertainty
   ↓
Decision
```

Examples:

### Classification

$$
P(Y|X)
$$

### Generative modeling

$$
P(X,Y)
$$

### Bayesian inference

$$
P(\theta|D)
$$

### Anomaly detection

Estimate how probable an observation is under a model.

---

# 41. Probabilistic Classification

A classification model can output probabilities instead of only class labels.

Example:

```text
Class A = 0.10
Class B = 0.75
Class C = 0.15
```

The predicted class may be the class with the largest estimated probability.

However, a probability score should not automatically be interpreted as a perfectly calibrated real-world frequency unless the model is appropriately calibrated.

---

# 42. Maximum Likelihood Estimation

**Maximum Likelihood Estimation (MLE)** estimates model parameters by choosing values that maximize the likelihood of the observed data.

Given data:

$$
D=\{x_1,x_2,\dots,x_n\}
$$

and parameter:

$$
\theta
$$

the likelihood is:

$$
L(\theta|D)
=
P(D|\theta)
$$

MLE chooses:

$$
\hat{\theta}_{MLE}
=
\arg\max_{\theta}L(\theta|D)
$$

In practice, we often maximize the log-likelihood:

$$
\hat{\theta}_{MLE}
=
\arg\max_{\theta}\log L(\theta|D)
$$

because logarithms turn products into sums and are numerically easier to work with.

---

# 43. Maximum A Posteriori Estimation

**Maximum A Posteriori (MAP)** estimation combines likelihood with a prior distribution.

Using Bayes' theorem:

$$
P(\theta|D)
\propto
P(D|\theta)P(\theta)
$$

MAP chooses:

$$
\hat{\theta}_{MAP}
=
\arg\max_{\theta}
P(D|\theta)P(\theta)
$$

### MLE vs MAP

```text
MLE
→ uses likelihood

MAP
→ uses likelihood + prior
```

This relationship is important for understanding Bayesian methods and regularization.

---

# 44. Probability and Loss Functions

Many ML loss functions have probabilistic interpretations.

## Log Loss / Cross-Entropy

For binary classification:

$$
L
=
-\left[
y\log(p)
+
(1-y)\log(1-p)
\right]
$$

where:

* `y` = true label
* `p` = predicted probability

For multiple classes:

$$
L
=
-\sum_{k}y_k\log(p_k)
$$

Cross-entropy is closely connected to maximum likelihood for common classification models.

---

# 45. Common Mistakes

## Mistake 1 — Confusing Probability and Percentage

```text
0.25 = 25%
```

not:

```text
0.25%
```

---

## Mistake 2 — Forgetting the Sample Space

Probabilities must be defined relative to possible outcomes or a probabilistic model.

---

## Mistake 3 — Confusing Independence with Mutual Exclusivity

Independent events:

$$
P(A\cap B)=P(A)P(B)
$$

Mutually exclusive events:

$$
P(A\cap B)=0
$$

Two nontrivial mutually exclusive events generally cannot also be independent.

---

## Mistake 4 — Misusing Conditional Probability

Remember:

$$
P(A|B)
\neq
P(B|A)
$$

in general.

---

## Mistake 5 — Confusing PDF with Probability

For continuous distributions:

```text
PDF value ≠ probability at a point
```

Probability is obtained from area over an interval.

---

## Mistake 6 — Assuming Correlation Means Causation

Probability and statistics describe relationships, but association alone does not establish causation.

---

## Mistake 7 — Treating Model Probabilities as Automatically Calibrated

A model producing:

```text
0.90
```

does not automatically mean that exactly 90% of similar predictions will be correct.

Calibration must be evaluated.

---

## Mistake 8 — Ignoring Distribution Assumptions

A probability distribution is a model. Its assumptions should be checked before using it for inference or prediction.

---

# 46. Python Libraries

Python provides several tools for probability and statistical modeling.

### Standard Library

```python
import random
import math
import statistics
```

### NumPy

```python
import numpy as np
```

Useful for:

* Random sampling
* Arrays
* Distributions
* Numerical calculations

### SciPy

```python
from scipy import stats
```

Useful for:

* Probability distributions
* PDFs
* PMFs
* CDFs
* Statistical tests
* Random variables

### Pandas

```python
import pandas as pd
```

Useful for:

* Data analysis
* Grouped statistics
* Sampling
* Data preprocessing

### Scikit-learn

```python
from sklearn.model_selection import train_test_split
```

Useful for:

* Dataset splitting
* Probabilistic classifiers
* Model evaluation
* Preprocessing

---

# 47. Practice Problems

## Beginner

### Problem 1

A fair die is rolled.

What is:

$$
P(X=4)
$$

---

### Problem 2

A fair die is rolled.

What is the probability of obtaining an even number?

---

### Problem 3

A coin is tossed twice.

Find:

$$
P(\text{exactly one Head})
$$

---

## Intermediate

### Problem 4

Given:

$$
P(A)=0.6
$$

$$
P(B)=0.4
$$

and `A` and `B` are independent.

Calculate:

$$
P(A\cap B)
$$

---

### Problem 5

Given:

$$
P(A\cap B)=0.2
$$

and:

$$
P(B)=0.5
$$

Calculate:

$$
P(A|B)
$$

---

### Problem 6

A disease has prevalence `1%`.

A diagnostic test has:

```text
Sensitivity = 95%
Specificity = 90%
```

Use Bayes' theorem to calculate the probability that a person actually has the disease given a positive test.

---

## Advanced

### Problem 7

Derive the expected value and variance of a Bernoulli random variable.

---

### Problem 8

Derive the Binomial PMF.

---

### Problem 9

Explain why the sample mean tends to become approximately normally distributed under the conditions of the Central Limit Theorem.

---

### Problem 10

Explain the difference between:

$$
P(A|B)
$$

and:

$$
P(B|A)
$$

using a real-world example.

---

### Problem 11

Implement a discrete probability distribution from scratch using Python.

Requirements:

* Store outcomes
* Store probabilities
* Validate probabilities
* Calculate expected value
* Calculate variance
* Calculate CDF

---

### Problem 12

Simulate 10,000 coin tosses and visualize how the observed proportion of heads changes as the number of tosses increases.

Use this experiment to demonstrate the Law of Large Numbers.

---

# 48. Learning Roadmap

A recommended learning order:

```text
Probability
│
├── 01. Random Experiments
│
├── 02. Sample Space
│
├── 03. Events
│
├── 04. Probability Rules
│
├── 05. Conditional Probability
│
├── 06. Independence
│
├── 07. Bayes' Theorem
│
├── 08. Random Variables
│
├── 09. PMF
│
├── 10. PDF
│
├── 11. CDF
│
├── 12. Expected Value
│
├── 13. Variance
│
├── 14. Bernoulli
│
├── 15. Binomial
│
├── 16. Geometric
│
├── 17. Poisson
│
├── 18. Uniform
│
├── 19. Normal
│
├── 20. Exponential
│
├── 21. Z-Scores
│
├── 22. Joint Probability
│
├── 23. Marginal Probability
│
├── 24. Conditional Distributions
│
├── 25. CLT
│
├── 26. Law of Large Numbers
│
├── 27. MLE
│
├── 28. MAP
│
└── 29. Probabilistic Machine Learning
```

---

# 49. Quick Reference

## Basic Probability

$$
0\leq P(A)\leq1
$$

---

## Complement

$$
P(A^c)=1-P(A)
$$

---

## Addition Rule

$$
P(A\cup B)
=
P(A)+P(B)-P(A\cap B)
$$

---

## Conditional Probability

$$
P(A|B)
=
\frac{P(A\cap B)}{P(B)}
$$

---

## Multiplication Rule

$$
P(A\cap B)
=
P(A)P(B|A)
$$

---

## Independence

$$
P(A\cap B)=P(A)P(B)
$$

---

## Bayes' Theorem

$$
P(A|B)
=
\frac{P(B|A)P(A)}
{P(B)}
$$

---

## Expected Value

$$
E[X]
=
\sum_xxP(X=x)
$$

for discrete variables.

---

## Variance

$$
Var(X)
=
E[X^2]-E[X]^2
$$

---

## Standard Deviation

$$
\sigma=\sqrt{Var(X)}
$$

---

## Z-Score

$$
z=
\frac{x-\mu}{\sigma}
$$

---

## Bernoulli

$$
E[X]=p
$$

$$
Var(X)=p(1-p)
$$

---

## Binomial

$$
E[X]=np
$$

$$
Var(X)=np(1-p)
$$

---

## Poisson

$$
E[X]=\lambda
$$

$$
Var(X)=\lambda
$$

---

## Uniform

$$
E[X]=\frac{a+b}{2}
$$

$$
Var(X)=\frac{(b-a)^2}{12}
$$

---

## Normal

$$
X\sim N(\mu,\sigma^2)
$$

---

# 50. Key Takeaways

* Probability provides a mathematical framework for uncertainty.
* A sample space contains all possible outcomes.
* Events are subsets of the sample space.
* Probability values lie between `0` and `1`.
* Conditional probability measures probability given information.
* Independence means one event does not change the probability of another.
* Bayes' theorem updates probabilities using evidence.
* Random variables can be discrete or continuous.
* PMFs describe discrete probability distributions.
* PDFs describe continuous probability densities.
* CDFs describe cumulative probability.
* Expected value represents a probability-weighted average.
* Variance and standard deviation measure dispersion.
* Bernoulli models one binary trial.
* Binomial models repeated Bernoulli trials.
* Poisson models event counts under specific assumptions.
* Normal distributions are central to many statistical methods.
* Z-scores measure standardized distance from the mean.
* The Central Limit Theorem explains important behavior of sample means.
* The Law of Large Numbers explains long-run stabilization of averages.
* Joint and conditional probabilities are fundamental to probabilistic ML.
* MLE estimates parameters from likelihood.
* MAP combines likelihood with a prior.
* Cross-entropy has a probabilistic interpretation.
* Model probabilities should be interpreted carefully and, when needed, evaluated for calibration.
* Probability is one of the mathematical foundations of Machine Learning.

---

# 📂 Suggested Directory Structure

```text
03-Probability/
│
├── README.md
│
├── 01-basic-probability/
│   ├── sample-space.py
│   ├── events.py
│   ├── probability-rules.py
│   └── complement-rule.py
│
├── 02-conditional-probability/
│   ├── conditional-probability.py
│   ├── independence.py
│   └── bayes-theorem.py
│
├── 03-random-variables/
│   ├── random-variable.py
│   ├── discrete-random-variable.py
│   └── continuous-random-variable.py
│
├── 04-probability-distributions/
│   ├── pmf.py
│   ├── pdf.py
│   ├── cdf.py
│   ├── bernoulli.py
│   ├── binomial.py
│   ├── geometric.py
│   ├── poisson.py
│   ├── uniform.py
│   ├── normal.py
│   └── exponential.py
│
├── 05-probability-statistics/
│   ├── expected-value.py
│   ├── variance.py
│   ├── standard-deviation.py
│   └── z-score.py
│
├── 06-advanced-probability/
│   ├── joint-probability.py
│   ├── marginal-probability.py
│   ├── conditional-distribution.py
│   ├── central-limit-theorem.py
│   └── law-of-large-numbers.py
│
└── 07-probability-in-machine-learning/
    ├── maximum-likelihood.py
    ├── maximum-a-posteriori.py
    ├── probabilistic-classification.py
    └── cross-entropy.py
```

---

## 📖 Recommended Prerequisites

Before studying this section, understand:

```text
01-Mathematics-for-Machine-Learning
        │
        ├── Algebra
        ├── Functions
        └── Basic Mathematics
                 ↓
02-Statistics
        │
        ├── Mean
        ├── Median
        ├── Variance
        ├── Standard Deviation
        ├── Covariance
        └── Correlation
                 ↓
03-Probability
        │
        ├── Conditional Probability
        ├── Bayes' Theorem
        ├── Random Variables
        ├── Distributions
        └── Probabilistic ML
```

---

## 🚀 From Probability to Machine Learning

The ultimate goal is not simply to memorize probability formulas.

The important progression is:

```text
Probability
     ↓
Random Variables
     ↓
Probability Distributions
     ↓
Statistical Inference
     ↓
Parameter Estimation
     ↓
Uncertainty Modeling
     ↓
Probabilistic Models
     ↓
Machine Learning
     ↓
Prediction + Uncertainty
```

Once these concepts are comfortable, the next step is to connect probability with **Linear Algebra, Calculus, Optimization, Statistics, and Machine Learning algorithms**.

> **Probability teaches us how to reason under uncertainty — one of the core mathematical skills required to understand modern Machine Learning.**
