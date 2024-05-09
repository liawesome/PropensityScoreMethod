This repository contains code for implementing various methods to build propensity score models and test validations. Propensity scores are used to estimate the treatment effect, which is the causal impact of a treatment (e.g., receiving a loan) on an outcome (e.g., loan default). By creating a propensity score, we can account for selection bias, which occurs when the groups being compared (treated and untreated) are not comparable on observable characteristics.

Implemented Methods
This repository includes code for the following propensity score modeling methods:

Logistic Regression: The most common method for estimating propensity scores. It models the probability of receiving the treatment as a function of covariates.
Random Forest: A machine learning method that can capture complex relationships between the treatment and covariates.
Nearest Neighbors: A non-parametric method that can be useful when the relationship between the treatment and covariates is non-linear.
Note: Additional methods can be added to this repository in the future.

Dependencies
This code requires the following Python libraries:
pandas
numpy
scikit-learn
