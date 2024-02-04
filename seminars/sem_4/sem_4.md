---
title: Convexity. Strong convexity.
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

# Lecture reminder

# Convex Sets

## Line Segment

Suppose $x_1, x_2$ are two points in $\mathbb{R^n}$. Then the line segment between them is defined as follows:

$$
x = \theta x_1 + (1 - \theta)x_2, \; \theta \in [0,1]
$$

![Illustration of a line segment between points $x_1$, $x_2$](line_segment.pdf){width=200}

## Convex Set

The set $S$ is called **convex** if for any $x_1, x_2$ from $S$ the line segment between them also lies in $S$, i.e. 

$$
\forall \theta \in [0,1], \; \forall x_1, x_2 \in S: \theta x_1 + (1- \theta) x_2 \in S
$$

::: {.callout-example}
Any affine set, a ray, a line segment - they all are convex sets.
:::

![Top: examples of convex sets. Bottom: examples of non-convex sets.](convex_sets.pdf){width=200}

## Problem 1

::: {.callout-question}
Prove, that ball in $\mathbb{R}^n$ (i.e. the following set $\{ \mathbf{x} \mid \Vert \mathbf{x} - \mathbf{x}_c \Vert \leq r \}$) - is convex.
:::

## Problem 2

::: {.callout-question}
Is stripe - $\{x \in \mathbb{R}^n \mid \alpha \leq a^\top x \leq \beta \}$ - convex?
:::

## Problem 3

::: {.callout-question}
Let $S$ be such that $\forall x,y \in S \to \frac{1}{2}(x+y) \in S$. Is this set convex?
:::

## Problem 4

::: {.callout-question}
The set $S = \{x \; | \; x + S_2 \subseteq S_1\}$, where $S_1,S_2 \subseteq \mathbb{R}^n$ with $S_1$ convex. Is this set convex?
:::

# Functions

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

# Criteria of Convexity

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
Twice differentiable function $f(x)$ defined on the convex set $S \subseteq \mathbb{R}^n$ is called $\mu$-strongly convex if and only if $\forall x \in \mathbf{int}(S) \neq \emptyset$:

$$
\nabla^2 f(x) \succeq \mu I
$$

In other words:

$$
\langle y, \nabla^2f(x)y\rangle \geq \mu \|y\|^2
$$

## Motivational Experiment with JAX

Why convexity and strong convexity is important? Check the simple [\faPython code snippet](https://colab.research.google.com/drive/14qPF7fkCWAoKcmFbN0Up4V0LMR287Nch?usp=sharing).

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
