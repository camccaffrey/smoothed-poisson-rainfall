# smoothed-poisson-rainfall

### Background

This repository presents a case study in statistical modeling developed during my time as a Graduate Student Researcher at UC Berkeley's [Institute of Transportation Studies](https://ucits.org/). While my primary research focused on large-scale processing and quality control of historical weather data from across California, this side project explores imputation techniques for missing rainfall data—aimed at better estimating wet-pavement exposure for traffic safety analyses.

### Overview

In this project, I fit smoothed Poisson models to hourly rainfall data from two California weather stations. The models apply regularization penalties to smooth the estimates over time and improve robustness in the presence of missing data.

**Goals**:

- Compare performance against baseline neighbor-average imputation methods
- Evaluate how imputation accuracy degrades with increasing gap size
- Assess the computational feasibility of incorporating uncertainty quantification

### Data

Rainfall observations were sourced from [Synoptic Data PBC's Weather API](https://synopticdata.com/weatherapi/), which granted my research team access to over a decade of hourly data from thousands of California weather stations.

### Modeling Approach

We model hourly rainfall as Poisson-distributed counts following

$$y_i \sim Poisson(\theta_i)$$

where

- $y_i$ is the observed rainfall at hour $i$, discretized as some non-negative integer
- $\theta_i$ is the Poisson rate parameter for hour $i$

#### Unregularized MLE
<!--
Below is the derivation for the (unregularized) maximum likelihood estimator for this data, denoted by $\hat{\theta}_{MLE} = \left(\hat{\theta}_1, ..., \hat{\theta}_n \right)$.

For independent observations from the Poisson distribution, the joint log-likelihood can be expressed as

$$
\begin{aligned}
L(\theta_1, ..., \theta_n) &= \prod^n_{i=1} p_{\theta_i}(y_i) \\
&= \prod^n_{i=1} \theta_i^{y_i} \exp(-\theta_i) (y_i!)^{-1}
\end{aligned}
$$

$$
\begin{aligned}
\ell(\theta_1, ..., \theta_n) &= \log L(\theta_1, ..., \theta_n) \\
&= \sum^n\_{i=1} \left[ y_i \log \theta_i - \theta_i - \log(y_i!) \right]
\end{aligned}
$$

Therefore, the estimator must satisfy

$$
\begin{aligned}
\hat{\theta}_{MLE} &= \text{arg max}\_{\theta_1, ..., \theta_n} \ell(\theta_1, ..., \theta_n) \\
&= \text{arg max}\_{\theta_1, ..., \theta_n} \sum^n\_{i=1} \left[ y_i \log \theta_i - \theta_i \right]
\end{aligned}
$$

To solve, we can maximize $\ell(\theta_1, ..., \theta_n)$ over all $\theta_i$ independently. That is, we can take derivatives w.r.t each $\theta_i$ such that

$$\frac{\partial \ell}{\partial \theta_i} = \frac{y_i}{\theta_i} - 1 = 0 \implies \hat{\theta}_i = y_i$$

Thus, $\hat{\theta}_{MLE}$ is just simply $y_i$ for all $i$.
-->

The maximum likelihood estimate (MLE) for this model is straightforward:

$$
\begin{aligned}
\hat{\theta} = \text{arg max } \ell(\theta_1, ..., \theta_n) = \text{arg max } \sum^n\_{i=1} \left[ y_i \log \theta_i - \theta_i \right]
\end{aligned}
$$

$$\frac{\partial \ell}{\partial \theta_i} = \frac{y_i}{\theta_i} - 1 = 0 \quad \implies \quad \hat{\theta}_i = y_i$$

This saturated model simply recovers the observed data—perfectly fitting available values, but offering no generalization, no smoothing, and no interpolation for missing values. It has:

- Zero bias, but high variance
- No temporal modeling (e.g., rainfall bursts or dry spells)
- No ability to impute missing values

#### Adding Regularization

To address this, I introduce regularization to encourage smoother estimates across time. Specifically, the estimator based on the penalized *negative* log-likelihood becomes:

<!--
$$
\begin{aligned}
\hat{\theta} &= \text{arg max}\_{\theta_1, ..., \theta_n} \left[\sum^n\_{i=1} (y_i \log \theta_i - \theta_i) - \lambda \cdot \text{Penalty}(\theta) \right] \\
&= \text{arg min}\_{\theta_1, ..., \theta_n} \left[\sum^n\_{i=1} (\theta_i - y_i \log \theta_i) + \lambda \cdot \text{Penalty}(\theta) \right]
\end{aligned}
$$
-->

$$
\begin{aligned}
\hat{\theta} = \text{arg min}\_{\theta} \left[\sum^n\_{i=1} (\theta_i - y_i \log \theta_i) + \lambda \cdot \text{Penalty}(\theta) \right]
\end{aligned}
$$

Here, $\lambda$ controls the regularization strength. I use an $L_2$ penalty on first-order differences in the **log-rate** parameters:

$$
\begin{aligned}
\text{Penalty}(\theta) = \sum^{n-1}\_{i=1} \left( \log \theta_{i+1} - \log \theta_i \right)^2
\end{aligned}
$$

This penalizes large relative jumps in rainfall rates, which is appropriate for data that varies across several orders of magnitude.

### Implementation

Because the regularized estimator lacks a closed-form solution, I solve it numerically using convex optimization (via `cvxpy`), with supporting code written in NumPu and SciPy.

<!--
$$Obj(\theta_1, ..., \theta_n) = \left[ \sum^n_{i=1} (\theta_i - y_i \eta_i) + \lambda \sum^{n-1}\_{i=1} ( \eta_{i+1} - \eta_i )^2 \right]$$
-->

The core model fitting function, `fit_poisson_smooth`, accepts:

- An array of rainfall observations
- A binary mask indicating valid (e.g., observed or training set) indices
- A regularization parameter $\lambda$

To select $\lambda$, I use $k$-fold cross-validation implemented in `cross_validate_lambda`, evaluating performance using Poisson deviance on held-out values.

#### Gap Simulation and Sequential Evaluation

To better understand model performance under different missing-data patterns, I built two helper functions:

- `sequential_train_test_split`: Simulates contiguous gaps of length $n$ by generating training/test masks.
- `sequential_fit`: Automates training, testing, and evaluation across a range of gap lengths.


### Results

Using 5-fold cross-validation, I identified $\lambda = 0.016$ and $\lambda = 0.113$ as top-performing values. Test performance was similar across both.

To simulate basic missingness, I randomly masked $10\\%$ of observed values as a test set, yielding the following performance metrics:

| Method           | Test RMSE (mm) | Test MAE (mm) |
|------------------|----------------|---------------|
| Smoothed Poisson |         $0.52$ |        $0.08$ |
| Neighbor-Average |         $0.75$ |        $0.12$ |

I also reframed the problem as binary classification: predict whether hourly rainfall exceeds $0.01$ inches ($0.254$ mm), the threshold for pavement wetness in my research.

| Method           | Error Rate     | True Positive Rate |
|------------------|----------------|--------------------|
| Smoothed Poisson |       $2.0\\%$ |            $75\\%$ |
| Neighbor-Average |       $4.6\\%$ |            $37\\%$ |

#### Robustness to Missing Gap Length

As gap size increases (from $1$ to $48$ hours), the smoothed Poisson model's performance deteriorates—especially for longer gaps—relative to the neighbor-average baseline. However, for **shorter gaps** ($n < 6$), the smoothed Poisson model consistently outperforms the baseline, demonstrating its usefulness in realistic imputation scenarios involving scattered or moderate missingness.
