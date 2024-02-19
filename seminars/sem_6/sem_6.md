---
title: Optimality conditions. Optimization with equality / inequality conditions. KKT.
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

# Optimality Conditions

## Optimality Conditions. Important notions recap
$$
f(x) \to \min\limits_{x \in S}
$$

A set $S$ is usually called a budget set.

<!-- We say that the problem has a solution if the budget set is not empty: $x^* \in S$, in which the minimum or the infimum of the given function is achieved. -->


* A point $x^*$ is a global minimizer if $f(x^*) \leq f(x)$ for all $x$.
* A point $x^*$ is a local minimizer if there exists a neighborhood $N$ of $x^*$ such that $f(x^*) \leq f(x)$ for all $x \in N$.
* A point $x^*$ is a strict local minimizer (also called a strong local minimizer) if there exists a neighborhood $N$ of $x^*$ such that $f(x^*) < f(x)$ for all $x \in N$ with $x \neq x^*$.
* We call $x^*$ a stationary point (or critical) if $\nabla f(x^*) = 0$. Any local minimizer must be a stationary point.

![Illustration of different stationary (critical) points](Local minima.pdf){width=150}


## Unconstrained optimization recap

::: {.callout-tip title="First-Order Necessary Conditions"}
If $x^*$ is a local minimizer and $f$ is continuously differentiable in an open neighborhood, then
$$
\nabla f(x^*) = 0
\tag{1}$$
::: 

::: {.callout-tip title="Second-Order Sufficient Conditions"}
Suppose that $\nabla^2 f$ is continuous in an open neighborhood of $x^*$ and that
$$
\nabla f(x^*) = 0 \quad \nabla^2 f(x^*) \succ 0.
\tag{2}
$$
Then $x^*$ is a strict local minimizer of $f$.
::: 


# Optimization with equality conditions

## Optimization with equality conditions

Consider simple yet practical case of equality constraints:

$$
\begin{split}
    & f(x) \to \min\limits_{x \in \mathbb{R}^n} \\
    \text{s.t. } & h_i(x) = 0, i = 1, \ldots, p
\end{split}
$$

## Lagrange multipliers recap

The basic idea of Lagrange method implies the switch from conditional to unconditional optimization through increasing the dimensionality of the problem:

$$
L(x, \nu) = f(x) + \sum\limits_{i=1}^p \nu_i h_i(x) =  f(x) + \nu^T h(x) \to \min\limits_{x \in \mathbb{R}^n, \nu \in \mathbb{R}^p}
$$

\
\

. . .

:::: {.columns}

::: {.column width="50%"}
::: {.callout-tip icon="false" appearance="simple"}

Necessery conditions:

$$
\nabla_x L(x^{*}, \nu^{*}) = 0
$$

$$
\nabla_{\nu} L(x^{*}, \nu^{*}) = 0
$$


:::
:::

::: {.column width="50%"}
::: {.callout-important icon="false" appearance="simple"}

Sufficient conditions:


$$
\langle y , \nabla^{2}_{xx} L(x^{*}, \nu^{*}) y \rangle > 0, 
$$

$$
\forall y \neq 0 \in  \mathbb{R}^n: \nabla h_i(x^{*})^{T} y = 0
$$

:::
:::
::::

# Optimization with inequality conditions

## Optimization with inequality conditions

Consider simple yet practical case of inequality constraints:

$$
\begin{split}
    & f(x) \to \min\limits_{x \in \mathbb{R}^n} \\
    \text{s.t. } & g(x) \leq 0
\end{split}
$$

. . .

:::: {.columns}

::: {.column width="50%"}

::: {.callout-tip icon="false" appearance="simple"}
$g(x) \leq 0$ **is inactive**. $g(x^*) < 0$:

$$g(x^*) < 0$$
$$\nabla f(x^*) = 0$$
$$\nabla^2 f(x^*) > 0$$
$$ $$
:::
:::

::: {.column width="50%"}

::: {.callout-tip icon="false" appearance="simple"}
$g(x) \leq 0$ **is active**. $g(x^*) = 0$:

$$g(x^*) = 0$$
$$- \nabla f(x^*) = \lambda \nabla g(x^*), \lambda > 0$$ 
$$\langle y, \nabla^2_{xx} L(x^*, \lambda^*) y \rangle > 0,$$
$$\forall y \neq 0 \in \mathbb{R}^n : \nabla g(x^*)^\top y = 0$$

:::
:::
::::


# Karush-Kuhn-Tucker conditions

## General formulation

General problem of mathematical programming:

$$
\begin{split}
& f_0(x) \to \min\limits_{x \in \mathbb{R}^n}\\
\text{s.t. } & f_i(x) \leq 0, \; i = 1,\ldots,m\\
& h_i(x) = 0, \; i = 1,\ldots, p
\end{split}
$$

. . .

The solution involves constructing a Lagrange function:
$$
L(x, \lambda, \nu) = f_0(x) + \sum\limits_{i=1}^m \lambda_i f_i(x) + \sum\limits_{i=1}^p\nu_i h_i(x)
$$

## KKT Necessary conditions

Let $x^*$, $(\lambda^*, \nu^*)$ be a solution to a mathematical programming problem with zero duality gap (the optimal value for the primal problem $p^*$ is equal to the optimal value for the dual problem $d^*$). Let also the functions $f_0, f_i, h_i$ be differentiable.

. . .

::: {.callout-tip icon="false" appearance="simple"}
$$
\begin{split}
& (1) \nabla_x L(x^*, \lambda^*, \nu^*) = 0 \\
& (2) \nabla_\nu L(x^*, \lambda^*, \nu^*) = 0 \\
& (3) \lambda^*_i \geq 0, i = 1,\ldots,m \\
& (4) \lambda^*_i f_i(x^*) = 0, i = 1,\ldots,m \\
& (5) f_i(x^*) \leq 0, i = 1,\ldots,m \\
\end{split}
$$
:::

## KKT Some regularity conditions

These conditions are needed in order to make KKT solutions the necessary conditions. Some of them even turn necessary conditions into sufficient. For example, Slater’s condition:

\
\

. . .

::: {.callout-tip icon="false" appearance="simple"}

If for a convex problem (i.e., assuming minimization, $f_0,f_{i}$ are convex and $h_{i}$ are affine), there exists a point $x$ such that $h(x)=0$ and $f_{i}(x)<0$ (existance of a strictly feasible point), then we have a zero duality gap and KKT conditions become necessary and sufficient.

:::

## KKT Sufficient conditions

For smooth, non-linear optimization problems, a second order sufficient condition is given as follows. The solution $x^{*},\lambda ^{*},\nu ^{*}$, which satisfies the KKT conditions (above) is a constrained local minimum if for the Lagrangian,

$$
L(x, \lambda, \nu) = f_0(x) + \sum\limits_{i=1}^m \lambda_i f_i(x) + \sum\limits_{i=1}^p\nu_i h_i(x)
$$

the following conditions hold:

::: {.callout-tip icon="false" appearance="simple"}
$$
\begin{split}
& \langle y , \nabla^2_{xx} L(x^*, \lambda^*, \nu^*) y \rangle > 0 \\
& \forall y \neq 0 \in \mathbb{R}^n : \nabla h_i(x^*)^\top y = 0, \nabla f_0(x^*) ^\top y \leq 0,\nabla f_j(x^*)^\top y \leq 0 \\
& i = 1,\ldots, p \quad \forall j: f_j(x^*) = 0
\end{split}
$$
:::

# Problems

## Problem 1
::: {.callout-question title="Solve an optimality problem"}

Function $f: E \to \mathbb{R}$ is defined as 
$$f(x) = \ln \left( -Q(x) \right)$$ 
where $E = \{x \in \mathbb{R}^n : Q(x) < 0\}$ and 
$$Q(x) = \frac{1}{2} x^\top A x + b^\top x + c$$ 
with $A \in \mathbb{S}^n_{++}, \, b \in \mathbb{R}^n, \, c \in \mathbb{R}$.

Find the maximizer $x^*$ of the function $f$.
::: 

## Problem 2
::: {.callout-question title="Solve a Lagrange multipliers problem"}
Give an explicit solution of the following task.
$$
\begin{split}
& f(x, y) = x + y \to \min\\
\text{s.t. } & x^2 + y^2 = 1
\end{split}
$$

where $x, y \in \mathbb{R}$.
::: 


## Problem 3
::: {.callout-question title="Solve a Lagrange multipliers problem"}
Give an explicit solution of the following task.
$$
\begin{split}
& \langle c, x \rangle + \sum_{i=1}^n x_i \log x_i \to \min\limits_{x \in \mathbb{R}^n }\\
\text{s.t. } & \sum_{i=1}^n x_i = 1,
\end{split}
$$
where $x \in \mathbb{R}^n_{++}, c \neq 0$.
::: 


## Problem 4
::: {.callout-question title="Solve a Lagrange multipliers problem"}
Give an explicit solution of the following task.
$$
\begin{split}
& f(x, y) = (x - 2)^2 + 2 (y - 1)^2 \to \min\\
\text{s.t. } & x + 4 y \leq 3\\
& x \geq y
\end{split}
$$

where $x, y \in \mathbb{R}$.
::: 


## Problem 5
::: {.callout-question title="Solve a Lagrange multipliers problem"}
Given $y \in \{-1, 1\}$, and $X \in \mathbb{R}^{n \times p}$, the Support Vector Machine problem is:

$$
\begin{split}
& \dfrac{1}{2} ||w||_{2}^{2} + C \sum_{i=1}^{n} \xi_i \to \min_{w, w_0, \xi_i}\\
\text{s.t. } & \xi_i  \geq 0, i = 1,\ldots, n \\
& y_i (x_i^{T} w + w_0) \geq 1 - \xi_i, i = 1,\ldots, n 
\end{split}
$$

find the KKT stationarity condition.
::: 

<!-- $$
\begin{split}
& \rho(x, x+r) \to \min\limits_{r}\\
\text{s.t. } & y(x+r) = \text{target},
\end{split}
$$ -->


