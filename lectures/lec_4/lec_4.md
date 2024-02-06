---
title: "Convexity: convex sets, convex functions. Polyak - Lojasiewicz Condition."
author: Daniil Merkulov
institute: Optimization for ML. Faculty of Computer Science. HSE University
# bibliography: ../../files/biblio.bib
# csl: ../../files/diabetologia.csl
format: 
    beamer:
        pdf-engine: pdflatex
        aspectratio: 169
        fontsize: 9pt
        section-titles: false
        incremental: true
        include-in-header: ../../files/header.tex  # Custom LaTeX commands and preamble
        header-includes: |
            \titlegraphic{\includegraphics[width=0.5\paperwidth]{back4.png}}
---

# Convex sets

## Affine set

:::: {.columns}
::: {.column width="40%"}
Suppose $x_1, x_2$ are two points in $\mathbb{R^n}$. Then the line passing through them is defined as follows:

$$
x = \theta x_1 + (1 - \theta)x_2, \theta \in \mathbb{R}
$$

The set $A$ is called **affine** if for any $x_1, x_2$ from $A$ the line passing through them also lies in $A$, i.e. 

$$
\forall \theta \in \mathbb{R}, \forall x_1, x_2 \in A: \theta x_1 + (1- \theta) x_2 \in A
$$

::: {.callout-example}
* $\mathbb{R}^n$ is an affine set.
* The set of solutions $\left\{x \mid \mathbf{A}x =  \mathbf{b} \right\}$ is also an affine set.
:::

:::
::: {.column width="60%"}
![Illustration of a line between two vectors $x_1$ and $x_2$](line.pdf)
:::
::::

## Cone
:::: {.columns}
::: {.column width="40%"}
A  non-empty set $S$ is called a cone, if:

$$
\forall x \in S, \; \theta \ge 0 \;\; \rightarrow \;\; \theta x \in S
$$
For any point in cone it also contains beam through this point.
:::
::: {.column width="60%"}
![Illustration of a cone](cone.pdf)
:::
::::

## Convex cone
:::: {.columns}
::: {.column width="40%"}
The set $S$ is called a convex cone, if:

$$
\forall x_1, x_2 \in S, \; \theta_1, \theta_2 \ge 0 \;\; \rightarrow \;\; \theta_1 x_1 + \theta_2 x_2 \in S
$$

Convex cone is just like cone, but it is also convex.

::: {.callout-example}

* $\mathbb{R}^n$
* Affine sets, containing $0$
* Ray
* $\mathbf{S}^n_+$ - the set of symmetric positive semi-definite matrices
:::
Convex cone: set that contains all conic combinations of points in the set
:::
::: {.column width="60%"}
![Illustration of a convex cone](convex_cone.pdf)
:::
::::


## Line segment

:::: {.columns}
::: {.column width="40%"}
Suppose $x_1, x_2$ are two points in $\mathbb{R}^n$. 

Then the line segment between them is defined as follows:

$$
x = \theta x_1 + (1 - \theta)x_2, \; \theta \in [0,1]
$$

Convex set contains line segment between any two points in the set.
:::
::: {.column width="60%"}
![Illustration of a line segment between points $x_1$, $x_2$](line_segment.pdf)
:::
::::


## Convex set

:::: {.columns}
::: {.column width="70%"}
The set $S$ is called **convex** if for any $x_1, x_2$ from $S$ the line segment between them also lies in $S$, i.e. 

$$
\forall \theta \in [0,1], \; \forall x_1, x_2 \in S: \theta x_1 + (1- \theta) x_2 \in S
$$

![Top: examples of convex sets. Bottom: examples of non-convex sets.](convex_sets.pdf)

:::
::: {.column width="30%"}

:::{.callout-example}
An empty set and a set from a single vector are convex by definition.
:::

:::{.callout-example}
Any affine set, a ray, a line segment - they all are convex sets.
:::
:::
::::


## Convex combination

Let $x_1, x_2, \ldots, x_k \in S$, then the point $\theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_k x_k$ is called the convex combination of points $x_1, x_2, \ldots, x_k$ if $\sum\limits_{i=1}^k\theta_i = 1, \; \theta_i \ge 0$.


## Convex hull
The set of all convex combinations of points from $S$ is called the convex hull of the set $S$.

$$
\mathbf{conv}(S) = \left\{ \sum\limits_{i=1}^k\theta_i x_i \mid x_i \in S, \sum\limits_{i=1}^k\theta_i = 1, \; \theta_i \ge 0\right\}
$$

* The set $\mathbf{conv}(S)$ is the smallest convex set containing $S$.
* The set $S$ is convex if and only if $S = \mathbf{conv}(S)$.

![Top: convex hulls of the convex sets. Bottom: convex hull of the non-convex sets.](convex_hulls.pdf)

## Minkowski addition
The Minkowski sum of two sets of vectors $S_1$ and $S_2$ in Euclidean space is formed by adding each vector in $S_1$ to each vector in $S_2$.

$$
S_1+S_2=\{\mathbf {s_1} +\mathbf {s_2} \,|\,\mathbf {s_1} \in S_1,\ \mathbf {s_2} \in S_2\}
$$

Similarly, one can define a linear combination of the sets.

::: {.callout-example}

:::: {.columns}
::: {.column width="60%"}
We will work in the $\mathbb{R}^2$ space. Let's define:

$$
S_1 := \{x \in \mathbb{R}^2 : x_1^2 + x_2^2 \leq 1\}
$$

This is a unit circle centered at the origin. And:

$$
S_2 := \{x \in \mathbb{R}^2 : -4 \leq x_1 \leq -1, -3 \leq x_2 \leq -1\}
$$

This represents a rectangle. The sum of the sets $S_1$ and $S_2$ will form an enlarged rectangle $S_2$ with rounded corners. The resulting set will be convex.
:::
::: {.column width="40%"}
![$S = S_1 + S_2$](minkowski.pdf)
:::
::::
:::

## Finding convexity

In practice, it is very important to understand whether a specific set is convex or not. Two approaches are used for this depending on the context.

* By definition.
* Show that $S$ is derived from simple convex sets using operations that preserve convexity.



## Finding convexity by definition

$$
x_1, x_2 \in S, \; 0 \le \theta \le 1 \;\; \rightarrow \;\; \theta x_1 + (1-\theta)x_2 \in S
$$

::: {.callout-example}
Prove, that ball in $\mathbb{R}^n$ (i.e. the following set $\{ \mathbf{x} \mid \Vert \mathbf{x} - \mathbf{x}_c \Vert \leq r \}$) - is convex.
:::

## Exercises

Which of the sets are convex:

* Stripe, $\{x \in \mathbb{R}^n \mid \alpha \leq a^\top x \leq \beta \}$
* Rectangle, $\{x \in \mathbb{R}^n \mid \alpha_i \leq x_i \leq \beta_i, i = \overline{1,n} \}$
* Kleen, $\{x \in \mathbb{R}^n \mid a_1^\top x \leq b_1, a_2^\top x \leq b_2 \}$
* A set of points closer to a given point than a given set that does not contain a point, $\{x \in \mathbb{R}^n \mid \Vert x - x_0\Vert _2 \leq \Vert x-y\Vert _2, \forall y \in S \subseteq \mathbb{R}^n \}$
* A set of points, which are closer to one set than another, $\{x \in \mathbb{R}^n \mid \mathbf{dist}(x,S) \leq \mathbf{dist}(x,T) , S,T \subseteq \mathbb{R}^n \}$
* A set of points, $\{x \in \mathbb{R}^{n} \mid x + X \subseteq S\}$, where $S \subseteq \mathbb{R}^{n}$ is convex and $X \subseteq \mathbb{R}^{n}$ is arbitrary.
* A set of points whose distance to a given point does not exceed a certain part of the distance to another given point is $\{x \in \mathbb{R}^n \mid \Vert x - a\Vert _2 \leq \theta\Vert x - b\Vert _2, a,b \in \mathbb{R}^n, 0 \leq 1 \}$

## Operations, that preserve convexity

The linear combination of convex sets is convex
Let there be 2 convex sets $S_x, S_y$, let the set 

$$
S = \left\{s \mid s = c_1 x + c_2 y, \; x \in S_x, \; y \in S_y, \; c_1, c_2 \in \mathbb{R}\right\}
$$

Take two points from $S$: $s_1 = c_1 x_1 + c_2 y_1, s_2 = c_1 x_2 + c_2 y_2$ and prove that the segment between them 
$\theta  s_1 + (1 - \theta)s_2, \theta \in [0,1]$ also belongs to $S$

$$
\theta s_1 + (1 - \theta)s_2
$$

$$
\theta (c_1 x_1 + c_2 y_1) + (1 - \theta)(c_1 x_2 + c_2 y_2)
$$

$$
c_1 (\theta x_1 + (1 - \theta)x_2) + c_2 (\theta y_1 + (1 - \theta)y_2)
$$

$$
c_1 x + c_2 y \in S
$$

## The intersection of any (!) number of convex sets is convex

If the desired intersection is empty or contains one point, the property is proved by definition. Otherwise, take 2 points and a segment between them. These points must lie in all intersecting sets, and since they are all convex, the segment between them lies in all sets and, therefore, in their intersection.

![Intersection of halfplanes](conv_inter.pdf){width=60%}

## The image of the convex set under affine mapping is convex

$$
S \subseteq \mathbb{R}^n \text{ convex}\;\; \rightarrow \;\; f(S) = \left\{ f(x) \mid x \in S \right\} \text{ convex} \;\;\;\; \left(f(x) = \mathbf{A}x + \mathbf{b}\right)
$$

Examples of affine functions: extension, projection, transposition, set of solutions of linear matrix inequality $\left\{ x \mid x_1 A_1 + \ldots + x_m A_m \preceq B\right\}$. Here $A_i, B \in \mathbf{S}^p$ are symmetric matrices $p \times p$. 

Note also that the prototype of the convex set under affine mapping is also convex.

$$
S \subseteq \mathbb{R}^m \text{ convex}\; \rightarrow \; f^{-1}(S) = \left\{ x \in \mathbb{R}^n \mid f(x) \in S \right\} \text{ convex} \;\; \left(f(x) = \mathbf{A}x + \mathbf{b}\right)
$$

## Example

Let $x \in \mathbb{R}$ is a random variable with a given probability distribution of $\mathbb{P}(x = a_i) = p_i$, where $i = 1, \ldots, n$, and $a_1 < \ldots < a_n$. It is said that the probability vector of outcomes of $p \in \mathbb{R}^n$ belongs to the probabilistic simplex, i.e. 

$$
P = \{ p \mid \mathbf{1}^Tp = 1, p \succeq 0 \} = \{ p \mid p_1 + \ldots + p_n = 1, p_i \ge 0 \}.
$$

Determine if the following sets of $p$ are convex:

* $\mathbb{P}(x > \alpha) \le \beta$
* $\mathbb{E} \vert x^{201}\vert \le \alpha \mathbb{E}\vert x \vert$
* $\mathbb{E} \vert x^{2}\vert \ge \alpha$$\mathbb{V} x \ge \alpha$

# Convex functions

## Jensen's inequality

:::: {.columns}
::: {.column width="30%"}
The function $f(x)$, **which is defined on the convex set** $S \subseteq \mathbb{R}^n$, is called **convex** on $S$, if:

$$
f(\lambda x_1 + (1 - \lambda)x_2) \le \lambda f(x_1) + (1 - \lambda)f(x_2)
$$

for any $x_1, x_2 \in S$ and $0 \le \lambda \le 1$.  
If the above inequality holds as strict inequality $x_1 \neq x_2$ and $0 < \lambda < 1$, then the function is called strictly convex on $S$.
:::
::: {.column width="70%"}
![Difference between convex and non-convex function](convex_function.pdf)
:::

::::

## Jensen's inequality

:::{.callout-theorem}
Let $f(x)$ be a convex function on a convex set $X \subseteq \mathbb{R}^n$ and let $x_i \in X, 1 \leq i \leq m$, be arbitrary points from $X$. Then

$$
f\left( \sum_{i=1}^{m} \lambda_i x_i \right) \leq \sum_{i=1}^{m} \lambda_i f(x_i)
$$

for any $\lambda = [\lambda_1, \ldots, \lambda_m] \in \Delta_m$ - probability simplex.
:::

**Proof**

1. First, note that the point $\sum_{i=1}^{m} \lambda_i x_i$ as a convex combination of points from the convex set $X$ belongs to $X$.
1. We will prove this by induction. For $m = 1$, the statement is obviously true, and for $m = 2$, it follows from the definition of a convex function.


## Jensen's inequality

3. Assume it is true for all $m$ up to $m = k$, and we will prove it for $m = k + 1$. Let $\lambda \in \Delta{k+1}$ and

    $$
    x = \sum_{i=1}^{k+1} \lambda_i x_i = \sum_{i=1}^{k} \lambda_i x_i + \lambda_{k+1} x_{k+1}.
    $$

    Assuming $0 < \lambda_{k+1} < 1$, as otherwise, it reduces to previously considered cases, we have

    $$
    x = \lambda_{k+1} x_{k+1} + (1 - \lambda_{k+1}) \bar{x}, 
    $$

    where $\bar{x} = \sum_{i=1}^{k} \gamma_i x_i$ and $\gamma_i = \frac{\lambda_i}{1-\lambda_{k+1}} \geq 0, 1 \leq i \leq k$.

4. Since $\lambda \in \Delta_{k+1}$, then $\gamma = [\gamma_1, \ldots, \gamma_k] \in \Delta_k$. Therefore $\bar{x} \in X$ and by the convexity of $f(x)$ and the induction hypothesis:

    $$
    f\left( \sum_{i=1}^{k+1} \lambda_i x_i \right) = f\left( \lambda_{k+1} x_{k+1} + (1 - \lambda_{k+1})\bar{x} \right) \leq
    \lambda_{k+1}f(x_{k+1}) + (1 - \lambda_{k+1})f(\bar{x}) \leq \sum_{i=1}^{k+1} \lambda_i f(x_i)
    $$

    Thus, initial inequality is satisfied for $m = k + 1$ as well.

## Examples of convex functions

* $f(x) = x^p, \;  p > 1,\;  x \in \mathbb{R}_+$
* $f(x) = \|x\|^p,\;  p > 1, x \in \mathbb{R}^n$
* $f(x) = e^{cx},\;  c \in \mathbb{R}, x \in \mathbb{R}$
* $f(x) = -\ln x,\;  x \in \mathbb{R}_{++}$
* $f(x) = x\ln x,\;  x \in \mathbb{R}_{++}$
* The sum of the largest $k$ coordinates $f(x) = x_{(1)} + \ldots + x_{(k)},\; x \in \mathbb{R}^n$
* $f(X) = \lambda_{max}(X),\;  X = X^T$
* $f(X) = - \log \det X, \;  X \in S^n_{++}$

## Epigraph

:::: {.columns}
::: {.column width="50%"}
For the function $f(x)$, defined on $S \subseteq \mathbb{R}^n$, the following set:

$$
\text{epi } f = \left\{[x,\mu] \in S \times \mathbb{R}: f(x) \le \mu\right\}
$$

is called **epigraph** of the function $f(x)$. 

:::{.callout-theorem}

## Convexity of the epigraph is the convexity of the function

For a function $f(x)$, defined on a convex set $X$, to be convex on $X$, it is necessary and sufficient that the epigraph of $f$ is a convex set.
:::
:::
::: {.column width="50%"}
![Epigraph of a function](epigraph.pdf)
:::
::::

## Convexity of the epigraph is the convexity of the function

1. **Necessity**: Assume $f(x)$ is convex on $X$. Take any two arbitrary points $[x_1, \mu_1] \in \text{epi}f$ and $[x_2, \mu_2] \in \text{epi}f$. Also take $0 \leq \lambda \leq 1$ and denote $x_{\lambda} = \lambda x_1 + (1 - \lambda) x_2, \mu_{\lambda} = \lambda \mu_1 + (1 - \lambda) \mu_2$. Then,

    $$
    \lambda\begin{bmatrix}x_1\\ \mu_1\end{bmatrix} + (1 - \lambda)\begin{bmatrix}x_2\\ \mu_2\end{bmatrix} = \begin{bmatrix}x_{\lambda}\\ \mu_{\lambda}\end{bmatrix}.
    $$

    From the convexity of the set $X$, it follows that $x_{\lambda} \in X$. Moreover, since $f(x)$ is a convex function,

    $$
    f(x_{\lambda}) \leq \lambda f(x_1) + (1 - \lambda) f(x_2) \leq \lambda \mu_1 + (1 - \lambda) \mu_2 = \mu_{\lambda}
    $$

    Inequality above indicates that $\begin{bmatrix}x_{\lambda}\\ \mu_{\lambda}\end{bmatrix} \in \text{epi}f$. Thus, the epigraph of $f$ is a convex set.

1. **Sufficiency**: Assume the epigraph of $f$, $\text{epi}f$, is a convex set. Then, from the membership of the points $[x_1, \mu_1]$ and $[x_2, \mu_2]$ in the epigraph of $f$, it follows that

    $$
     \begin{bmatrix}x_{\lambda}\\ \mu_{\lambda}\end{bmatrix} =  \lambda\begin{bmatrix}x_1\\ \mu_1\end{bmatrix} + (1 - \lambda)\begin{bmatrix}x_2\\ \mu_2\end{bmatrix} \in \text{epi}f
    $$

    for any $0 \leq \lambda \leq 1$, i.e., $f(x_{\lambda}) \leq \mu_{\lambda} = \lambda \mu_1 + (1 - \lambda) \mu_2$. But this is true for all $\mu_1 \geq f(x_1)$ and $\mu_2 \geq f(x_2)$, particularly when $\mu_1 = f(x_1)$ and $\mu_2 = f(x_2)$. Hence we arrive at the inequality

    $$
    f(x_{\lambda}) = f (\lambda x_1 + (1 - \lambda) x_2) \leq \lambda f(x_1) + (1 - \lambda) f(x_2).
    $$

    Since points $x_1 \in X$ and $x_2 \in X$ can be arbitrarily chosen, $f(x)$ is a convex function on $X$.




## Sublevel set

:::: {.columns}
::: {.column width="50%"}
For the function $f(x)$, defined on $S \subseteq \mathbb{R}^n$, the following set:

$$
\mathcal{L}_\beta = \left\{ x\in S : f(x) \le \beta\right\}
$$

is called **sublevel set** or Lebesgue set of the function $f(x)$. 
:::
::: {.column width="50%"}
![Sublevel set of a function with respect to level $\beta$](sublevel_set.pdf)
:::
::::

## First-order differential criterion of convexity

:::: {.columns}
::: {.column width="55%"}
The differentiable function $f(x)$ defined on the convex set $S \subseteq \mathbb{R}^n$ is convex if and only if $\forall x,y \in S$:

$$
f(y) \ge f(x) + \nabla f^T(x)(y-x)
$$

Let $y = x + \Delta x$, then the criterion will become more tractable:

$$
f(x + \Delta x) \ge f(x) + \nabla f^T(x)\Delta x
$$

:::
::: {.column width="45%"}
![Convex function is greater or equal than Taylor linear approximation at any point](global_linear_lower_bound.pdf)
:::
::::

## Second-order differential criterion of convexity

Twice differentiable function $f(x)$ defined on the convex set $S \subseteq \mathbb{R}^n$ is convex if and only if $\forall x \in \mathbf{int}(S) \neq \emptyset$:

$$
\nabla^2 f(x) \succeq 0
$$

In other words, $\forall y \in \mathbb{R}^n$:

$$
\langle y, \nabla^2f(x)y\rangle \geq 0
$$

## Connection with epigraph
The function is convex if and only if its epigraph is a convex set.

::: {.callout-example}
Let a norm $\Vert \cdot \Vert$ be defined in the space $U$. Consider the set:

$$
K := \{(x,t) \in U \times \mathbb{R}^+ : \Vert x \Vert \leq t \}
$$

which represents the epigraph of the function $x \mapsto \Vert x \Vert$. This set is called the cone norm. According to the statement above, the set $K$ is convex.

In the case where $U = \mathbb{R}^n$ and $\Vert x \Vert = \Vert x \Vert_2$ (Euclidean norm), the abstract set $K$ transitions into the set:

$$
\{(x,t) \in \mathbb{R}^n \times \mathbb{R}^+ : \Vert x \Vert_2 \leq t \}
$$
:::

## Connection with sublevel set
If $f(x)$ - is a convex function defined on the convex set $S \subseteq \mathbb{R}^n$, then for any $\beta$ sublevel set $\mathcal{L}_\beta$ is convex.

The function $f(x)$ defined on the convex set $S \subseteq \mathbb{R}^n$ is closed if and only if for any $\beta$ sublevel set $\mathcal{L}_\beta$ is closed.

## Reduction to a line
$f: S \to \mathbb{R}$ is convex if and only if $S$ is a convex set and the function $g(t) = f(x + tv)$ defined on $\left\{ t \mid x + tv \in S \right\}$  is convex for any $x \in S, v \in \mathbb{R}^n$, which allows checking convexity of the scalar function to establish convexity of the vector function.

## Strong convexity
:::: {.columns}
::: {.column width="55%"}
$f(x)$, **defined on the convex set** $S \subseteq \mathbb{R}^n$, is called $\mu$-strongly convex (strongly convex) on $S$, if:

$$
f(\lambda x_1 + (1 - \lambda)x_2) \le \lambda f(x_1) + (1 - \lambda)f(x_2) - \frac{\mu}{2} \lambda (1 - \lambda)\|x_1 - x_2\|^2
$$

for any $x_1, x_2 \in S$ and $0 \le \lambda \le 1$ for some $\mu > 0$.
:::
::: {.column width="45%"}
![Strongly convex function is greater or equal than Taylor quadratic approximation at any point](global_quad_lower_bound.pdf)
:::
::::

## First-order differential criterion of strong convexity

Differentiable $f(x)$ defined on the convex set $S \subseteq \mathbb{R}^n$ is $\mu$-strongly convex if and only if $\forall x,y \in S$:

$$
f(y) \ge f(x) + \nabla f^T(x)(y-x) + \dfrac{\mu}{2}\|y-x\|^2
$$

Let $y = x + \Delta x$, then the criterion will become more tractable:

$$
f(x + \Delta x) \ge f(x) + \nabla f^T(x)\Delta x + \dfrac{\mu}{2}\|\Delta x\|^2
$$

:::{.callout-theorem}

Let $f(x)$ be a differentiable function on a convex set $X \subseteq \mathbb{R}^n$. Then $f(x)$ is strongly convex on $X$ with a constant $\mu > 0$ if and only if

$$ 
f(x) - f(x_0) \geq \langle \nabla f(x_0), x - x_0 \rangle + \frac{\mu}{2} \| x - x_0 \|^2
$$

for all $x, x_0 \in X$.
::::

## Proof of first-order differential criterion of strong convexity

**Necessity**: Let $0 < \lambda \leq 1$. According to the definition of a strongly convex function,

$$ 
f(\lambda x + (1 - \lambda) x_0) \leq \lambda f(x) + (1 - \lambda) f(x_0) - \frac{\mu}{2} \lambda (1 - \lambda) \| x - x_0 \|^2 
$$

or equivalently,

$$ 
f(x) - f(x_0) - \frac{\mu}{2} (1 - \lambda) \| x - x_0 \|^2 \geq \frac{1}{\lambda} [f(\lambda x + (1 - \lambda) x_0) - f(x_0)] = 
$$

$$ 
= \frac{1}{\lambda} [f(x_0 + \lambda(x - x_0)) - f(x_0)] = \frac{1}{\lambda} [\lambda \langle \nabla f(x_0), x - x_0 \rangle + o(\lambda)] = 
$$

$$ 
= \langle \nabla f(x_0), x - x_0 \rangle + \frac{o(\lambda)}{\lambda}. 
$$

Thus, taking the limit as $\lambda \downarrow 0$, we arrive at the initial statement.

## Proof of first-order differential criterion of strong convexity
**Sufficiency**: Assume the inequality in the theorem is satisfied for all $x, x_0 \in X$. Take $x_0 = \lambda x_1 + (1 - \lambda) x_2$, where $x_1, x_2 \in X$, $0 \leq \lambda \leq 1$. According to the inequality, the following inequalities hold:

$$ 
f(x_1) - f(x_0) \geq \langle \nabla f(x_0), x_1 - x_0 \rangle + \frac{\mu}{2} \| x_1 - x_0 \|^2, 
$$

$$ 
f(x_2) - f(x_0) \geq \langle \nabla f(x_0), x_2 - x_0 \rangle + \frac{\mu}{2} \| x_2 - x_0 \|^2. 
$$

Multiplying the first inequality by $\lambda$ and the second by $1 - \lambda$ and adding them, considering that

$$ 
x_1 - x_0 = (1 - \lambda)(x_1 - x_2), \quad x_2 - x_0 = \lambda(x_2 - x_1), 
$$

and $\lambda(1 - \lambda)^2 + \lambda^2(1 - \lambda) = \lambda(1 - \lambda)$, we get

$$ 
\begin{split}
\lambda f(x_1) + (1 - \lambda) f(x_2) - f(x_0) - \frac{\mu}{2} \lambda (1 - \lambda) \| x_1 - x_2 \|^2 \geq \\
\langle \nabla f(x_0), \lambda x_1 + (1 - \lambda) x_2 - x_0 \rangle = 0. 
\end{split}
$$

Thus, inequality from the definition of a strongly convex function is satisfied. It is important to mention, that $\mu = 0$ stands for the convex case and corresponding differential criterion.

## Second-order differential criterion of strong convexity
Twice differentiable function $f(x)$ defined on the convex set $S \subseteq \mathbb{R}^n$ is called $\mu$-strongly convex if and only if $\forall x \in \mathbf{int}(S) \neq \emptyset$:

$$
\nabla^2 f(x) \succeq \mu I
$$

In other words:

$$
\langle y, \nabla^2f(x)y\rangle \geq \mu \|y\|^2
$$

:::{.callout-theorem}

Let $X \subseteq \mathbb{R}^n$ be a convex set, with $\text{int}X \neq \emptyset$. Furthermore, let $f(x)$ be a twice continuously differentiable function on $X$. Then $f(x)$ is strongly convex on $X$ with a constant $\mu > 0$ if and only if

$$
\langle y, \nabla^2 f(x) y \rangle \geq \mu \| y \|^2 \quad 
$$

for all $x \in X$ and $y \in \mathbb{R}^n$.
:::

## Proof of second-order differential criterion of strong convexity

The target inequality is trivial when $y = \mathbf{0}_n$, hence we assume $y \neq \mathbf{0}_n$.

**Necessity**: Assume initially that $x$ is an interior point of $X$. Then $x + \alpha y \in X$ for all $y \in \mathbb{R}^n$ and sufficiently small $\alpha$. Since $f(x)$ is twice differentiable,

$$
f(x + \alpha y) = f(x) + \alpha \langle \nabla f(x), y \rangle + \frac{\alpha^2}{2} \langle y, \nabla^2 f(x) y \rangle + o(\alpha^2).
$$

Based on the first order criterion of strong convexity, we have

$$
\frac{\alpha^2}{2} \langle y, \nabla^2 f(x) y \rangle + o(\alpha^2) = f(x + \alpha y) - f(x) - \alpha \langle \nabla f(x), y \rangle \geq \frac{\mu}{2} \alpha^2 \| y \|^2.
$$

This inequality reduces to the target inequality after dividing both sides by $\alpha^2$ and taking the limit as $\alpha \downarrow 0$.

If $x \in X$ but $x \notin \text{int}X$, consider a sequence $\{x_k\}$ such that $x_k \in \text{int}X$ and $x_k \rightarrow x$ as $k \rightarrow \infty$. Then, we arrive at the target inequality after taking the limit.

## Proof of second-order differential criterion of strong convexity

**Sufficiency**: Using Taylor's formula with the Lagrange remainder and the target inequality, we obtain for $x + y \in X$:

$$
f(x + y) - f(x) - \langle \nabla f(x), y \rangle = \frac{1}{2} \langle y, \nabla^2 f(x + \alpha y) y \rangle \geq \frac{\mu}{2} \| y \|^2, 
$$

where $0 \leq \alpha \leq 1$. Therefore,

$$
f(x + y) - f(x) \geq \langle \nabla f(x), y \rangle + \frac{\mu}{2} \| y \|^2.
$$

Consequently, by the first order criterion of strong convexity, the function $f(x)$ is strongly convex with a constant $\mu$. It is important to mention, that $\mu = 0$ stands for the convex case and corresponding differential criterion.

## Facts

:::: {.columns}
::: {.column width="50%"}
* $f(x)$ is called (strictly) concave, if the function $-f(x)$ - is (strictly) convex.
* Jensen's inequality for the convex functions:

    $$
    f \left( \sum\limits_{i=1}^n \alpha_i x_i \right) \leq \sum\limits_{i=1}^n \alpha_i f(x_i)
    $$

    for $\alpha_i \geq 0; \quad \sum\limits_{i=1}^n \alpha_i = 1$ (probability simplex)  
    For the infinite dimension case:
    
    $$
    f \left( \int\limits_{S} x p(x)dx \right) \leq \int\limits_{S} f(x)p(x)dx
    $$

    If the integrals exist and $p(x) \geq 0, \quad \int\limits_{S} p(x)dx = 1$.
:::
:::{.column width="50%"}
* If the function $f(x)$ and the set $S$ are convex, then any local minimum $x^* = \text{arg}\min\limits_{x \in S} f(x)$ will be the global one. Strong convexity guarantees the uniqueness of the solution.
* Let $f(x)$ - be a convex function on a convex set $S \subseteq \mathbb{R}^n$. Then $f(x)$ is continuous $\forall x \in \textbf{ri}(S)$.
:::
::::

## Operations that preserve convexity  

* Non-negative sum of the convex functions: $\alpha f(x) + \beta g(x), (\alpha \geq 0 , \beta \geq 0)$.
* Composition with affine function $f(Ax + b)$ is convex, if $f(x)$ is convex.
* Pointwise maximum (supremum) of any number of functions: If $f_1(x), \ldots, f_m(x)$ are convex, then $f(x) = \max \{f_1(x), \ldots, f_m(x)\}$ is convex.
* If $f(x,y)$ is convex on $x$ for any $y \in Y$: $g(x) = \underset{y \in Y}{\text{sup}}f(x,y)$ is convex.
* If $f(x)$ is convex on $S$, then $g(x,t) = t f(x/t)$ - is convex with $x/t \in S, t > 0$. 
* Let $f_1: S_1 \to \mathbb{R}$ and $f_2: S_2 \to \mathbb{R}$, where $\text{range}(f_1) \subseteq S_2$. If $f_1$ and $f_2$ are convex, and $f_2$ is increasing, then $f_2 \circ f_1$ is convex on $S_1$.

## Other forms of convexity
* Log-convexity: $\log f$ is convex; Log convexity implies convexity.
* Log-concavity: $\log f$ concave; **not** closed under addition!
* Exponential convexity: $[f(x_i + x_j)] \succeq 0$, for $x_1, \ldots, x_n$ 
* Operator convexity: $f(\lambda X + (1 - \lambda )Y)$ 
* Quasiconvexity: $f(\lambda x + (1 - \lambda) y) \leq \max \{f(x), f(y)\}$
* Pseudoconvexity: $\langle \nabla f(y), x - y \rangle \geq 0 \longrightarrow f(x) \geq f(y)$
* Discrete convexity: $f : \mathbb{Z}^n \to \mathbb{Z}$; “convexity + matroid theory.”

## Examples
::: {.callout-example}
Show, that $f(x) = c^\top x + b$ is convex and concave.
:::

## Examples
::: {.callout-example}
Show, that $f(x) = x^\top Ax$, where $A\succeq 0$ - is convex on $\mathbb{R}^n$.
:::

## Examples
::: {.callout-example}
Show, that $f(A) = \lambda_{max}(A)$ - is convex, if $A \in S^n$.
:::

PL inequality holds if the following condition is satisfied for some $\mu > 0$,
$$
\Vert \nabla f(x) \Vert^2 \geq \mu (f(x) - f^*) \forall x
$$
The example of a function, that satisfies the PL-condition, but is not convex.
$$
f(x,y) = \dfrac{(y - \sin x)^2}{2}
$$

# Convexity in ML

## Convex optimization problem

:::: {.columns}
::: {.column width="50%"}
![The idea behind the definition of a convex optimization problem](cop.pdf)
:::
::: {.column width="50%"}

Note, that there is an agreement in notation of mathematical programming. The problems of the following type are called **Convex optimization problem**:

$$
\begin{split}
& f_0(x) \to \min\limits_{x \in \mathbb{R}^n}\\
\text{s.t. } & f_i(x) \leq 0, \; i = 1,\ldots,m\\
& Ax = b,
\end{split}
\tag{COP}
$$

where all the functions $f_0(x), f_1(x), \ldots, f_m(x)$ are convex and all the equality constraints are affine. It sounds a bit strange, but not all convex problems are convex optimization problems. 

$$
\tag{CP}
f_0(x) \to \min\limits_{x \in S},
$$

where $f_0(x)$ is a convex function, defined on the convex set $S$. The necessity of affine equality constraint is essential.
:::
::::

## Linear Least Squares

![Illustration](lls_idea.pdf)


## Neural networks?