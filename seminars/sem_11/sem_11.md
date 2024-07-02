---
title: Subgradient and Subdifferencial
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

# Main notions recap
## Main notions recap

::: {.callout-note title="Main notions"}

For a domain set $E \in \mathbb{R}^n$ and a function $f: E \rightarrow \mathbb{R}$:

* A vector $g \in \mathbb{R}^n$ is called **subgradient** of the function $f$ at $x \in E$ if $\forall y \in E$
$$  f(y) \geq f(x) + g^T(y-x)$$ 
* A set $\partial f(x)$ is called **subdifferential** of the function $f$ at $x\in E$ if:
$$ \partial f(x)  = \{g \in \mathbb{R}^n \,|\, f(y) \geq f(x) + g^T(y-x)\} \forall y \in E$$
* $f(\cdot)$ is called **subdifferentiable** at point $x \in E$ if $\partial f(x) \neq \emptyset$ 

::: 

## Connection between subdifferentiation and convexity

::: {.callout-tip title="Connection between subdifferentiation and convexity"}

If $f: E \rightarrow \mathbb{R}$ is subdifferentiable on the **convex** subset $S \in E$ then $f$ is convex on $S$.

::: 

* The inverse is generally incorrect
* There is no sense to derive the subgradient of nonconvex function.

## Connection between subdifferentiation and differentiation

::: {.callout-tip title="CConnection between subdifferentiation and differentiation"}

1) If $f: E \rightarrow \mathbb{R}$ is convex and differentiable at $x \in \text{int } E$ then $\partial f(x) = \{ \Delta f(x) \}$

2) If $f: E \rightarrow \mathbb{R}$ is convex and for $x \in \text{int } E$ $\partial f(x) = \{ s \}$ then $f$ is differentiable at $x$ and $\Delta f(x) = s$

::: 

* Derive the subdifferencial of a differentiable function is overkill.

## Problem 1
::: {.callout-question title="Find the subgradient of the function $f$"}

Find the subgradient of the function
$$ f(x) = -\sqrt{x} $$

::: 

## Subdifferentiation rules
1) $f: E \rightarrow \mathbb{R}$, $x \in E$, $c > 0$
 $$\Rightarrow \partial (cf)(x) = c\partial f(x)$$

1) $f: F \rightarrow \mathbb{R}$, $g: G \rightarrow \mathbb{R}$, $x \in F \bigcap G$
$$\Rightarrow \partial (f+g)(x) \supseteq \partial f(x) + \partial g(x)$$
1) $T: V \rightarrow W = Ax + b$, $g: W \rightarrow \mathbb{R}$, $x_0 \in V$ 
$$\Rightarrow \partial (g \circ T)(x_0) \supseteq A^*\,\partial (g)(T(x_0))$$
1) $f(x) = \max(f_1(x), \ldots, f_m(x))$, $I(x) = \{ i \in 1\ldots m| f_i(x) = f(x) \}$ 
$$\Rightarrow  \partial f(x) \supseteq \text{Conv}(\bigcup_{i \in I(x)} \partial f_i(x))$$

::: {.callout-tip title="When is equality reached?"}

If abovementioned functions are convex and $x$ is inner point then all inequalities turn into equalities.

::: 


## Problem 2
::: {.callout-question title="Find the subgradient of the sum"}

Find the subgradient of the function $f(x) + g(x)$ if
$$ f(x) = -\sqrt{x} \text{ when } x \geq 0 $$
$$ g(x) = -\sqrt{-x} \text{ when } x \leq 0 $$

::: 


## Problem 3
::: {.callout-question title="$L_1$ regularizer"}


1) Find the subgradient of the function $f(x) = || Ax - b ||_1$;
1) For task $f(x) = \frac{1}{2} || Ax - b ||_{2}^{2} + \lambda ||x||_1  \rightarrow \min_x$ say which lambdas lead to $x_{opt} = 0$

::: 

## Problem 4
::: {.callout-question title="Differentiability checking"}

Check the differentiability of the function 
$$ f(A) = \sup_{||x||_2=1} x^TAx, \text{ where } A \in \mathbb{S}^n, \, x \in \mathbb{R}^n$$

::: 
