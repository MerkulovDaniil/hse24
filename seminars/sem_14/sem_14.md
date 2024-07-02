---
title: Proximal Gradient Method. Proximal operator
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

# Composite optimization

## Regularized / Composite Objectives

:::: {.columns}
::: {.column width="50%"}
Many nonsmooth problems take the form
$$
\min_{x \in \mathbb{R}^n} \varphi(x) = f(x) + r(x)
$$

* **Lasso, L1-LS, compressed sensing** 
    $$
    f(x) = \frac12 \|Ax - b\|_2^2, r(x) = \lambda \|x\|_1
    $$
* **L1-Logistic regression, sparse LR**
    $$
    f(x) = -y \log h(x) - (1-y)\log(1-h(x)), r(x) = \lambda \|x\|_1
    $$
:::

::: {.column width="50%"}
![](sem_14/composite_objective.pdf)
:::
::::

## Non-smooth convex optimization lower bounds

|convex (non-smooth) | strongly convex (non-smooth) |
|:-----:|:-----:|
| $f(x_k) - f^* \sim  \mathcal{O} \left( \dfrac{1}{\sqrt{k}} \right)$ | $f(x_k) - f^* \sim  \mathcal{O} \left( \dfrac{1}{k} \right)$ | 
| $k_\varepsilon \sim  \mathcal{O} \left( \dfrac{1}{\varepsilon^2} \right)$ | $k_\varepsilon \sim \mathcal{O} \left( \dfrac{1}{\varepsilon} \right)$ |

. . .

* Subgradient method is optimal for the problems above.
* One can use Mirror Descent (a generalization of the subgradient method to a possiby non-Euclidian distance) with the same convergence rate to better fit the geometry of the problem.
* However, we can achieve standard gradient descent rate $\mathcal{O}\left(\frac1k \right)$ (and even accelerated version $\mathcal{O}\left(\frac{1}{k^2} \right)$) if we will exploit the structure of the problem.


## Proximal operator

::: {.callout-note title="Proximal operator"}

For a convex set $E \in \mathbb{R}^n$ and a convex function $f: E \rightarrow \mathbb{R}$ operator $\text{prox}_f(x)$ s.t.
$$ \text{prox}_f(x) = \argmin_{y \in E} \left [ f(y) + \frac{1}{2} ||y - x||_2^2 \right] $$
is called **proximal operator** for function $f$ at point $x$

::: 

## From projections to proximity

Let $\mathbb{I}_S$ be the indicator function for closed, convex $S$. Recall orthogonal projection $\pi_S(y)$

. . .

$$
\pi_S(y) := \arg\min_{x \in S} \frac{1}{2}\|x-y\|_2^2.
$$

. . .

With the following notation of indicator function
$$
\mathbb{I}_S(x) = \begin{cases} 0, &x \in S, \\ \infty, &x \notin S, \end{cases}
$$

Rewrite orthogonal projection $\pi_S(y)$ as
$$
\pi_S(y) := \arg\min_{x \in \mathbb{R}^n} \frac{1}{2} \|x - y\|^2 + \mathbb{I}_S (x).
$$

. . .

Proximity: Replace $\mathbb{I}_S$ by some convex function!
$$
\text{prox}_{r} (y) = \text{prox}_{r, 1} (y) := \arg\min \frac{1}{2} \|x - y\|^2 + r(x)
$$


## Proximal Gradient Method

::: {.callout-tip title="Proximal Gradient Method Theorem"}

Consider the proximal gradient method 
$$ x_{k+1} = \text{prox}_{\alpha r}\left( x_k - \alpha \nabla f(x_k) \right) $$
for the criterion $\phi(x) = f(x) + r(x)$ s.t.:
1. $f$ is convex, differentiable with Lipschitz gradients;
1. $r$ is convex and prox-friendly.
Then Proximal Gradient Method with fixed step size $\alpha = \frac{1}{L}$ converges with rate $O(\frac{1}{k})$

::: 

## ISTA and FISTA

Methods for solving problems involving $L1$ regularization (e.g. Lasso).

:::: {.columns}

::: {.column width="60%"}
**ISTA** (Iterative Shrinkage-Thresholding Algorithm)

* Step:
$$ x_{k+1} = \text{prox}_{\alpha \lambda ||\cdot||_1}\left( x_k - \alpha \nabla f(x_k) \right) $$
* Convergence: $O(\frac{1}{k})$

![](sem_14/proximal_methods_comparison.pdf){width="50%"}
:::

::: {.column width="40%"}
**FISTA** (Fast Iterative Shrinkage-Thresholding Algorithm)

* Step:
$$ x_{k+1} = \text{prox}_{\alpha \lambda ||\cdot||_1}\left( y_k - \alpha \nabla f(y_k) \right), $$
$$ t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}, $$
$$ y_{k+1} = x_{x+1} + \frac{t_{k} - 1}{t_{k+1}}(x_{k+1} - x_{k}) $$

* Convergence: $O(\frac{1}{k^2})$

:::
::::

# Problems

## Problem 1. ReLU in prox

::: {.callout-question title="ReLU in criterion"}

Find the $\text{prox}_f(x)$ for $f(x) = \lambda \max(0, x)$:
$$ \text{prox}_{\lambda \max(0, \cdot)}(x) = \argmin_{y \in \mathbb{R}} \left[ \frac{1}{2}||y - x||^2 + \lambda \max(0, y) \right] $$

::: 

## Problem 2. Grouped $l_1$-regularizer

::: {.callout-question title="Grouped $l_1$-regularizer"}

Find the $\text{prox}_f(x)$ for $f(x) =  ||x||_{1/2} = \sum_{g = 0}^{G} ||x_g||_2$ where $x \in \mathbb{R}^n = [ \underbrace{x_1, x_2}_1, \ldots,  \underbrace{\ldots}_g, \ldots,  \underbrace{x_{n-2}, x_{n-1}, x_n}_G ]$:
$$ \text{prox}_{||x||_{1/2}}(x) = \argmin_{y \in \mathbb{R}} \left[ \frac{1}{2}||y - x||^2_2 + \sum_{g = 0}^{G} ||y_g||_2 \right] $$

::: 

# Colab examples
## Linear Least Squares with $L_1$-regularizer

Proximal Methods Comparison for Linear Least Squares with $L_1$-regularizer [\faPython Open in Colab](https://colab.research.google.com/drive/1Hx_sASRJsWR4XeACfT8sxFEdYxzMgzqm?usp=sharing).

