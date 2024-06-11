---
title: Stochastic Gradient Descent. Finite-sum problems
author: Seminar
institute: Optimization for ML. Faculty of Computer Science. HSE University
format:
    beamer:
        pdf-engine: pdflatex
        aspectratio: 169
        fontsize: 9pt
        section-titles: false
        incremental: false
        include-in-header: ../../files/header.tex  # Custom LaTeX commands and preamble
---

# Lecture recap
## Finite-sum problem

We consider classic finite-sample average minimization:
$$
\min_{x \in \mathbb{R}^p} f(x) = \min_{x \in \mathbb{R}^p}\frac{1}{n} \sum_{i=1}^n f_i(x)
$$

The gradient descent acts like follows:
$$
\tag{GD}
x_{k+1} = x_k - \frac{\alpha_k}{n} \sum_{i=1}^n \nabla f_i(x)
$$

* Iteration cost is linear in $n$.
* Convergence with constant $\alpha$ or line search.

. . .

Let's switch from the full gradient calculation to its unbiased estimator, when we randomly choose $i_k$ index of point at each iteration uniformly:
$$
\tag{SGD}
x_{k+1} = x_k - \alpha_k  \nabla f_{i_k}(x_k)
$$
With $p(i_k = i) = \frac{1}{n}$, the stochastic gradient is an unbiased estimate of the gradient, given by:
$$
\mathbb{E}[\nabla f_{i_k}(x)] = \sum_{i=1}^{n} p(i_k = i) \nabla f_i(x) = \sum_{i=1}^{n} \frac{1}{n} \nabla f_i(x) = \frac{1}{n} \sum_{i=1}^{n} \nabla f_i(x) = \nabla f(x)
$$
This indicates that the expected value of the stochastic gradient is equal to the actual gradient of $f(x)$.

## Results for Gradient Descent

Stochastic iterations are $n$ times faster, but how many iterations are needed?

If $\nabla f$ is Lipschitz continuous then we have:

| Assumption   | Deterministic Gradient Descent | Stochastic Gradient Descent |
|:--:|:----:|:--:|
| PL     | $O(\log(1/\varepsilon))$  | $O(1/\varepsilon)$ |
| Convex | $O(1/\varepsilon)$   | $O(1/\varepsilon^2)$ |
| Non-Convex | $O(1/\varepsilon)$ | $O(1/\varepsilon^2)$ |

* Stochastic has low iteration cost but slow convergence rate. 
  * Sublinear rate even in strongly-convex case.
  * Bounds are unimprovable under standard assumptions.
  * Oracle returns an unbiased gradient approximation with bounded variance.
    
* Momentum and Quasi-Newton-like methods do not improve rates in stochastic case. Can only improve constant factors (bottleneck is variance, not condition number).

# Computational experiments

## Computational experiments

[Visualization of SGD](https://fa.bianp.net/teaching/2018/eecs227at/stochastic_gradient.html). 

Let's look at computational experiments for SGD [\faPython](https://colab.research.google.com/drive/1ITm0mHXq_1UyoIt18gULukLdBbtrCYt_?usp=sharing).