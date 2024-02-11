---
title: Strongly convex functions. Optimality conditions.
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

# Function Convexity

## Convex Function

The function $f(x)$, **which is defined on the convex set** $S \subseteq \mathbb{R}^n$, is called **convex** on $S$, if:

$$
f(\lambda x_1 + (1 - \lambda)x_2) \le \lambda f(x_1) + (1 - \lambda)f(x_2)
$$

for any $x_1, x_2 \in S$ and $0 \le \lambda \le 1$.  
If the above inequality holds as strict inequality $x_1 \neq x_2$ and $0 < \lambda < 1$, then the function is called **strictly convex** on $S$.

![Difference between convex and non-convex function](convex_function.pdf){width=250}

## Strong Convexity

$f(x)$, **defined on the convex set** $S \subseteq \mathbb{R}^n$, is called $\mu$-strongly convex (strongly convex) on $S$, if:

$$
f(\lambda x_1 + (1 - \lambda)x_2) \le \lambda f(x_1) + (1 - \lambda)f(x_2) - \frac{\mu}{2} \lambda (1 - \lambda)\|x_1 - x_2\|^2
$$

for any $x_1, x_2 \in S$ and $0 \le \lambda \le 1$ for some $\mu > 0$.

![Strongly convex function is greater or equal than Taylor quadratic approximation at any point](strong_convexity.pdf){width=250}

# Differential Criteria of Convexity

## First-order differential criterion of convexity
The differentiable function $f(x)$ defined on the convex set $S \subseteq \mathbb{R}^n$ is convex if and only if $\forall x,y \in S$:

$$
f(y) \ge f(x) + \nabla f^T(x)(y-x)
$$

Let $y = x + \Delta x$, then the criterion will become more tractable:

$$
f(x + \Delta x) \ge f(x) + \nabla f^T(x)\Delta x
$$

![Convex function is greater or equal than Taylor linear approximation at any point](diff_conv.pdf){width=200}

## Second-order differential criterion of strong convexity
Twice differentiable function $f(x)$ defined on the convex set $S \subseteq \mathbb{R}^n$ is $\mu$-strongly convex if and only if $\forall x \in \mathbf{int}(S) \neq \emptyset$:

$$
\nabla^2 f(x) \succeq \mu I
$$

In other words:

$$
\langle y, \nabla^2f(x)y\rangle \geq \mu \|y\|^2
$$

## Motivational Experiment with JAX

Why convexity and strong convexity is important? Check the simple [\faPython code snippet](https://colab.research.google.com/drive/14qPF7fkCWAoKcmFbN0Up4V0LMR287Nch?usp=sharing).

# Problems

## Problem 1

::: {.callout-question}
Show, that $f(x) = \|x\|$ is convex on $\mathbb{R}^n$.
:::

::: {.callout-question}
Show, that $f(x) = x^\top Ax$, where $A\succeq 0$ - is convex on $\mathbb{R}^n$.
:::

## Problem 2

::: {.callout-question}
Show, that if $f(x)$ is convex on $\mathbb{R}^n$, then $\exp(f(x))$ is convex on $\mathbb{R}^n$.
:::

## Problem 3

::: {.callout-question}
If $f(x)$ is convex nonnegative function and $p \ge 1$. Show that $g(x)=f(x)^p$ is convex.
:::

## Problem 4

::: {.callout-question}
Show that, if $f(x)$ is concave positive function over convex S, then $g(x)=\frac{1}{f(x)}$ is convex.
:::

::: {.callout-question}
Show, that the following function is convex on the set of all positive denominators
$$
f(x) = \dfrac{1}{x_1 - \dfrac{1}{x_2 - \dfrac{1}{x_3 - \dfrac{1}{\ldots}}}}, x \in \mathbb{R}^n
$$
:::

## Problem 5

::: {.callout-question}
Let $S = \{x \in \mathbb{R}^n \; \vert \; x \succ 0, \Vert x \Vert_{\infty} \leq M \}$. Show that $f(x)=\sum_{i=1}^n x_i \log x_i$ is $\frac{1}{M}$-strongly convex.
:::

# PL-Condition

## Polyak-Lojasiewicz (PL) Condition


PL inequality holds if the following condition is satisfied for some  $\mu > 0$,
$$
\Vert \nabla f(x) \Vert^2 \geq \mu (f(x) - f^*) \forall x
$$
The example of a function, that satisfies the PL-condition, but is not convex.
$$
f(x,y) = \dfrac{(y - \sin x)^2}{2}
$$

Example of Pl non-convex function [\faPython Open in Colab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/PL_function.ipynb).


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


## Lagrange multipliers recap

Consider simple yet practical case of equality constraints:
$$
\begin{split}
    & f(x) \to \min\limits_{x \in \mathbb{R}^n} \\
    \text{s.t. } & h_i(x) = 0, i = 1, \ldots, p
\end{split}
$$

The basic idea of Lagrange method implies the switch from conditional to unconditional optimization through increasing the dimensionality of the problem:

$$
L(x, \nu) = f(x) + \sum\limits_{i=1}^p \nu_i h_i(x) \to \min\limits_{x \in \mathbb{R}^n, \nu \in \mathbb{R}^p}
$$

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
& \langle c, x \rangle + \sum_{i=1}^n x_i \log x_i \to \min\limits_{x \in \mathbb{R}^n }\\
\text{s.t. } & \sum_{i=1}^n x_i = 1,
\end{split}
$$
where $x \in \mathbb{R}^n_{++}, c \neq 0$.
::: 

## Adversarial Attacks as Constrained Optimization
![Any neural network can be fooled with invisible pertubation](adversarial_attacks_cat_ex.pdf){width=250}

:::: {.columns}
::: {.incremental}
::: {.column width="50%"}

* Targetted Adversarial Attack:
$$
\begin{split}
& \rho(x, x+r) \to \min\limits_{r \in \mathbb{R}^n }\\
\text{s.t. } & y(x+r) = \text{target\_class},
\end{split}
$$
:::

::: {.column width="50%"}
* Non-targetted Adversarial Attack:
$$
\begin{split}
& \rho(x, x+r) \to \min\limits_{r \in \mathbb{R}^n }\\
\text{s.t. } & y(x+r) = y(x),
\end{split}
$$
:::
:::
::::

## Solution from Szegedy et al, "Intriguing properties of neural networks"

<!-- 1. Solution uses Lagrange multipliers method
2. Solution uses L-BFGS optimizer -->

:::: {.columns}
::: {.incremental}
::: {.column width="50%"}
* Targetted Adversarial Attack Task:
$$
\begin{split}
& \rho(x, x+r) \to \min\limits_{r \in \mathbb{R}^n }\\
\text{s.t. } & y(x+r) = \text{target\_class},
\end{split}
$$

* Targetted Lagrange function $L(r, c \,|\, x)$:
$$
||r||^2 - c \log p(y=\text{target\_class} \,|\, x+r) \to \min\limits_{r \in \mathbb{R}^n }
$$

:::

::: {.column width="50%"}
* Non-targetted Adversarial Attack Task:
$$
\begin{split}
& \rho(x, x+r) \to \min\limits_{r \in \mathbb{R}^n }\\
\text{s.t. } & y(x+r) = y(x),
\end{split}
$$

* Non-targetted Lagrange function $L(r, c \,|\, x)$:
$$
||r||^2 + c \log p(y = y_{\text{origin}} \,|\, x+r) \to \min\limits_{r \in \mathbb{R}^n }
$$
:::
:::
::::

. . .

::: {.callout-important title="Method Problems"}
1. *Attack success or not* -- there is no guarantee the method will work;
2. Simple optimizers may not work due to nonconvexity of Neural Networks (authors use L-BFGS);
:::

. . .

::: {.callout-note title="More sophisticated methods"}
* Fast Gradient Sign Method (FGSM)
* Deep Fool
:::


<!-- $$
\begin{split}
& \rho(x, x+r) \to \min\limits_{r}\\
\text{s.t. } & y(x+r) = \text{target},
\end{split}
$$ -->


