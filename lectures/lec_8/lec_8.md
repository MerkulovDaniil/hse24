---
title: "Linear Programming. Simplex Algorithm. Applications."
author: Daniil Merkulov
institute: Optimization for ML. Faculty of Computer Science. HSE University
format: 
    beamer:
        pdf-engine: pdflatex
        aspectratio: 169
        fontsize: 9pt
        section-titles: false
        incremental: true
        include-in-header: ../../files/header.tex  # Custom LaTeX commands and preamble
        header-includes: |
            \titlegraphic{\includegraphics[width=0.5\paperwidth]{back8.png}}
---

# Linear Programming

## What is Linear Programming?

:::: {.columns}

::: {.column width="40%"}
![](LP.pdf)
:::

::: {.column width="60%"}
Generally speaking, all problems with linear objective and linear equalities/inequalities constraints could be considered as Linear Programming. However, there are some formulations.

$$
\tag{LP.Basic}
\begin{split}
&\min_{x \in \mathbb{R}^n} c^{\top}x \\
\text{s.t. } & Ax \leq b\\
\end{split}
$$

for some vectors $c \in \mathbb{R}^n$, $b \in \mathbb{R}^m$ and matrix $A \in \mathbb{R}^{m \times n}$. Where the inequalities are interpreted component-wise.

. . .

**Standard form.** This form seems to be the most intuitive and geometric in terms of visualization. Let us have vectors $c \in \mathbb{R}^n$, $b \in \mathbb{R}^m$ and matrix $A \in \mathbb{R}^{m \times n}$.

$$
\tag{LP.Standard}
\begin{split}
&\min_{x \in \mathbb{R}^n} c^{\top}x \\
\text{s.t. } & Ax = b\\
& x_i \geq 0, \; i = 1,\dots, n
\end{split}
$$
:::

::::


## Example: Diet problem

:::: {.columns}

::: {.column width="50%"}
![](diet_LP.pdf)
:::

::: {.column width="50%"}
Imagine, that you have to construct a diet plan from some set of products: bananas, cakes, chicken, eggs, fish. Each of the products has its vector of nutrients. Thus, all the food information could be processed through the matrix $W$. Let us also assume, that we have the vector of requirements for each of nutrients $r \in \mathbb{R}^n$. We need to find the cheapest configuration of the diet, which meets all the requirements:

$$
\begin{split}
&\min_{x \in \mathbb{R}^p} c^{\top}x \\
\text{s.t. } & Wx \succeq r\\
& x_i \geq 0, \; i = 1,\dots, n
\end{split}
$$

[\faPython Open In Colab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/LP.ipynb#scrollTo=fpT9Ywy5obfu)
:::

::::

## Basic transformations

* Max-min
    $$
    \begin{split}
    &\min_{x \in \mathbb{R}^n} c^{\top}x \\
    \text{s.t. } & Ax \leq b\\
    \end{split} \quad \leftrightarrow \quad
    \begin{split}
    &\max_{x \in \mathbb{R}^n} -c^{\top}x \\
    \text{s.t. } & Ax \leq b\\
    \end{split} 
    $$

* Equality to inequality
    $$
    Ax = b \leftrightarrow 
    \begin{cases}
    Ax \leq  b\\
    Ax \geq b
    \end{cases}
    $$

* Inequality to equality by increasing the dimension of the problem by $m$.
    $$
    Ax \leq b \leftrightarrow 
    \begin{cases}
    Ax + z =  b\\
    z \geq 0
    \end{cases}
    $$

* Unsigned variables to nonnegative variables.
    $$
    x \leftrightarrow 
    \begin{cases}
    x = x_+ - x_-\\
    x_+ \geq 0 \\
    x_- \geq 0
    \end{cases}
    $$

## Example: Chebyshev approximation problem

$$
\min_{x \in \mathbb{R}^n} \|Ax - b\|_\infty \leftrightarrow \min_{x \in \mathbb{R}^n} \max_{i} |a_i^\top x - b_i|
$$

Could be equivalently written as an LP with the replacement of the maximum coordinate of a vector:

. . .

$$
\begin{split}
&\min_{t \in \mathbb{R}, x \in \mathbb{R}^n} t \\
\text{s.t. } & a_i^\top x - b_i \leq t, \; i = 1,\dots, n\\
& -a_i^\top x + b_i \leq t, \; i = 1,\dots, n
\end{split}
$$

## $\ell_1$ approximation problem

$$
\min_{x \in \mathbb{R}^n} \|Ax - b\|_1 \leftrightarrow \min_{x \in \mathbb{R}^n} \sum_{i=1}^n |a_i^\top x - b_i|
$$

Could be equivalently written as an LP with the replacement of the sum of coordinates of a vector:

. . .

$$
\begin{split}
&\min_{t \in \mathbb{R}^n, x \in \mathbb{R}^n} \mathbf{1}^\top t \\
\text{s.t. } & a_i^\top x - b_i \leq t_i, \; i = 1,\dots, n\\
& -a_i^\top x + b_i \leq t_i, \; i = 1,\dots, n
\end{split}
$$

# Duality in Linear Programming

## Duality

:::: {.columns}

::: {.column width="50%"}
Primal problem:

$$
\begin{split}
&\min_{x \in \mathbb{R}^n} c^{\top}x \\
\text{s.t. } & Ax = b\\
& x_i \geq 0, \; i = 1,\dots, n
\end{split}
$$ {#eq-lp_primal}

. . .

KKT for optimal $x^*, \nu^*, \lambda^*$:
$$
\begin{split}
&L(x, \nu, \lambda) = c^Tx + \nu^T(Ax-b) - \lambda^T x \\
&-A^T \nu^* + \lambda^* = c \\
&Ax^* = b\\
&x^*\succeq 0\\
&\lambda^*\succeq 0\\
&\lambda^*_i x_i^* = 0\\
\end{split}
$$

:::

. . .

::: {.column width="50%"}
Has the following dual:

$$
\begin{split}
&\max_{\nu \in \mathbb{R}^m} -b^{\top}\nu \\
\text{s.t. } & -A^T\nu \preceq c\\
\end{split}
$$ {#eq-lp_dual}

Find the dual problem to the problem above (it should be the original LP). Also, write down KKT for the dual problem, to ensure, they are identical to the primal KKT.
:::

::::

## Strong duality in linear programming

:::{callout-theorem}
(i) If either problem @eq-lp_primal or @eq-lp_dual has a (finite) solution, then so does the other, and the objective values are equal.

(ii) If either problem @eq-lp_primal or @eq-lp_dual is unbounded, then the other problem is infeasible.
:::

. . .

**PROOF.**  For (i), suppose that @eq-lp_primal has a finite optimal solution $x^*$. It follows from KKT that there are optimal vectors $\lambda^*$ and $\nu^*$ such that $(x^*, \nu^*, \lambda^*)$ satisfies KKT. We noted above that KKT for @eq-lp_primal and @eq-lp_dual are equivalent. Moreover, $c^Tx^* = (-A^T \nu^* + \lambda^*)^T x^* = - (\nu^*)^T A x^* =  - b^T\nu^*$, as claimed.

A symmetric argument holds if we start by assuming that the dual problem @eq-lp_dual has a solution.

. . .

To prove (ii), suppose that the primal is unbounded, that is, there is a sequence of points $x_k$, $k = 1,2,3,\ldots$ such that
$$
c^Tx_k \downarrow -\infty, \quad Ax_k = b, \quad x_k \geq 0.
$$

. . .

Suppose too that the dual @eq-lp_dual is feasible, that is, there exists a vector $\bar{\nu}$ such that $-A^T \bar{\nu} \leq c$. From the latter inequality together with $x_k \geq 0$, we have that $-\bar{\nu}^T Ax_k \leq c^T x_k$, and therefore 
$$
-\bar{\nu}^T b = -\bar{\nu}^T Ax_k \leq c^T x_k \downarrow -\infty,
$$
yielding a contradiction. Hence, the dual must be infeasible. A similar argument can be used to show that the unboundedness of the dual implies the infeasibility of the primal.


## Example: [Transportation problem](https://jckantor.github.io/ND-Pyomo-Cookbook/notebooks/03.01-Transportation-Networks.html)

The prototypical transportation problem deals with the distribution of a commodity from a set of sources to a set of destinations. The object is to minimize total transportation costs while satisfying constraints on the supplies available at each of the sources, and satisfying demand requirements at each of the destinations.

![Western Europe Map. [\faPython Open In Colab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/LP_transport.ipynb)](LP_west_europe.png)

## Example: [Transportation problem](https://jckantor.github.io/ND-Pyomo-Cookbook/notebooks/03.01-Transportation-Networks.html)

:::: {.columns}

::: {.column width="69%"}

| Customer / Source | Arnhem [\faEuroSign/ton] |  Gouda [\faEuroSign/ton] | Demand [tons] |
| :---: | :--: | :--: | :--: |
| London | n/a | 2.5 | 125 |
| Berlin | 2.5 | n/a | 175 |
| Maastricht | 1.6 | 2.0 | 225 |
| Amsterdam | 1.4 | 1.0 | 250 |
| Utrecht | 0.8 | 1.0 | 225 |
| The Hague | 1.4 | 0.8 | 200 |
| **Supply [tons]** | 550 tons | 700 tons | |

$$
\text{minimize:}\quad \text{Cost} = \sum_{c \in \text{Customers}}\sum_{s \in \text{Sources}} T[c,s] x[c,s]
$$

. . .

$$
\sum_{c \in \text{Customers}} x[c,s] \leq \text{Supply}[s] \qquad \forall s \in \text{Sources}
$$

. . .

$$
\sum_{s\in \text{Sources}} x[c,s] = \text{Demand}[c] \qquad \forall c \in \text{Customers}
$$

:::

::: {.column width="31%"}
This can be represented in the following graph:

![Graph associated with the problem](LP_transport_graph.pdf)
:::

::::

# Simplex Algorithm

## Geometry of simplex algorithm

:::: {.columns}

::: {.column width="50%"}
![](LP_basis.pdf)
:::

::: {.column width="50%"}
We will consider the following simple formulation of LP, which is, in fact, dual to the Standard form:

$$
\tag{LP.Inequality}
\begin{split}
&\min_{x \in \mathbb{R}^n} c^{\top}x \\
\text{s.t. } & Ax \leq b
\end{split}
$$

* Definition: a **basis** $\mathcal{B}$ is a subset of $n$ (integer) numbers between $1$ and $m$, so that $\text{rank} A_B = n$. 
* Note, that we can associate submatrix $A_B$ and corresponding right-hand side $b_B$ with the basis $\mathcal{B}$. 
* Also, we can derive a point of intersection of all these hyperplanes from the basis: $x_B = A^{-1}_B b_B$. 
* If $A x_B \leq b$, then basis $\mathcal{B}$ is **feasible**. 
* A basis $\mathcal{B}$ is optimal if $x_B$ is an optimum of the $\text{LP.Inequality}$.
:::

::::

## The solution of LP if exists lies in the corner

:::: {.columns}

::: {.column width="40%"}
![](LP.pdf)
:::

::: {.column width="60%"}

:::{.callout-theorem}
1. If Standard LP has a nonempty feasible region, then there is at least one basic feasible point
1. If Standard LP has solutions, then at least one such solution is a basic optimal point.
1. If Standard LP is feasible and bounded, then it has an optimal solution.

. . .

For proof see [Numerical Optimization by Jorge Nocedal and Stephen J. Wright](https://fmin.xyz/assets/files/NumericalOptimization.pdf) theorem 13.2
:::

The high-level idea of the simplex method is following: 

* Ensure, that you are in the corner. 
* Check optimality.
* If necessary, switch the corner (change the basis).
* Repeat until converge.
:::
::::

## Optimal basis

:::: {.columns}

::: {.column width="40%"}
![](LP_basis.pdf)
:::

::: {.column width="60%"}
Since we have a basis, we can decompose our objective vector $c$ in this basis and find the scalar coefficients $\lambda_B$:

$$
\lambda^\top_B A_B = c^\top \leftrightarrow \lambda^\top_B = c^\top A_B^{-1}
$$

:::{.callout-theorem}
If all components of $\lambda_B$ are non-positive and $B$ is feasible, then $B$ is optimal.
:::

**Proof**
$$
\begin{split}
\uncover<+->{\exists x^*: Ax^* &\leq b, c^\top x^* < c^\top x_B \\}
\uncover<+->{A_B x^* &\leq b_B \\}
\uncover<+->{\lambda_B^\top A_B x^* &\geq \lambda_B^\top b_B \\}
\uncover<+->{c^\top x^* & \geq \lambda_B^\top A_B x_B \\}
\uncover<+->{c^\top x^* & \geq c^\top  x_B \\}
\end{split}
$$
:::
::::

## Changing basis

:::: {.columns}

::: {.column width="50%"}
![](LP_change.pdf)
:::

::: {.column width="50%"}

Suppose, some of the coefficients of $\lambda_B$ are positive. Then we need to go through the edge of the polytope to the new vertex (i.e., switch the basis)

* Suppose, we have a basis $\mathcal{B}$: $\lambda^\top_B = c^\top A_B^{-1}$
* Let's assume, that $\lambda^k_B > 0$. We'd like to drop $k$ from the basis and form a new one:
    $$
    \begin{cases}
    A_{B \textbackslash \{k\}} d = 0 \\
    a^T_k d = -1
    \end{cases}
    $$

* For all $j \notin \mathcal{B}$ calculate the projection stepsize:
    $$
    \mu_j = \frac{b_j - a_j^T x_B}{a_j^T d}
    $$
* Define the new vertex, that you will add to the new basis:
    $$
    \begin{split}
    t = \text{arg}\min_j \{\mu_j \mid \mu_j > 0\} \\
    \mathcal{B}' = \mathcal{B}\textbackslash \{k\} \cup \{t\} \\ 
    x_{B'} = x_B + \mu_t d = A^{-1}_{B'} b_{B'}
    \end{split}
    $$

:::
::::

## Finding an initial basic feasible solution

<!-- Let us consider $\text{LP.Canonical}$.

$$
\begin{split}
&\min_{x \in \mathbb{R}^n} c^{\top}x \\
\text{s.t. } & Ax = b\\
& x_i \geq 0, \; i = 1,\dots, n
\end{split}
$$

The proposed algorithm requires an initial basic feasible solution and corresponding basis. To compute this solution and basis, we start by multiplying by $-1$ any row $i$ of $Ax = b$ such that $b_i < 0$. This ensures that $b \geq 0$. We then introduce artificial variables $z \in \mathbb{R}^m$ and consider the following LP:

$$
\tag{LP.Phase 1}
\begin{split}
&\min_{x \in \mathbb{R}^n, z \in \mathbb{R}^m} 1^{\top}z \\
\text{s.t. } & Ax + Iz = b\\
& x_i, z_j \geq 0, \; i = 1,\dots, n \; j = 1,\dots, m
\end{split}
$$

which can be written in canonical form $\min\{\tilde{c}^\top \tilde{x} \mid \tilde{A}\tilde{x} = \tilde{b}, \tilde{x} \geq 0\}$ by setting

$$
\tilde{x} = \begin{bmatrix}x\\z\end{bmatrix}, \quad \tilde{A} = [A \; I], \quad \tilde{b} = b, \quad \tilde{c} = \begin{bmatrix}0_n\\1_m\end{bmatrix}
$$ -->

## Finding an initial basic feasible solution

<!-- An initial basis for $\text{LP.Phase 1}$ is $\tilde{A}_B = I, \tilde{A}_N = A$ with corresponding basic feasible solution $\tilde{x}_N = 0, \tilde{x}_B = \tilde{A}^{-1}_B \tilde{b} = \tilde{b} \geq 0$. We can therefore run the simplex method on $\text{LP. Phase 1}$, which will converge to an optimum $\tilde{x}^*$. $\tilde{x} = (\tilde{x}_N \; \tilde{x}_B)$. There are several possible outcomes:

* $\tilde{c}^\top \tilde{x} > 0$. Original primal is infeasible.
* $\tilde{c}^\top \tilde{x} = 0 \to 1^\top z^* = 0$. The obtained solution is a starting point for the original problem (probably with slight modification). -->

# Convergence of the Simplex Algorithm

## Unbounded budget set

:::: {.columns}

::: {.column width="50%"}
![](LP_unbounded.pdf)
:::

::: {.column width="50%"}
In this case, all $\mu_j$ will be negative.
:::
::::

## Degeneracy

:::: {.columns}

::: {.column width="50%"}
![](LP_degenerate.pdf)
:::

::: {.column width="50%"}
One needs to handle degenerate corners carefully. If no degeneracy exists, one can guarantee a monotonic decrease of the objective function on each iteration.
:::
::::

## Exponential convergence

:::: {.columns}

::: {.column width="50%"}
![](LP_IPM.pdf)
:::

::: {.column width="50%"}

* A wide variety of applications could be formulated as linear programming.
* Simplex algorithm is simple but could work exponentially long.
* Khachiyan’s ellipsoid method is the first to be proven to run at polynomial complexity for LPs. However, it is usually slower than simplex in real problems.
* Interior point methods are the last word in this area. However, good implementations of simplex-based methods and interior point methods are similar for routine applications of linear programming.
:::
::::

## [Klee Minty](https://en.wikipedia.org/wiki/Klee%E2%80%93Minty_cube) example

Since the number of edge points is finite, the algorithm should converge (except for some degenerate cases, which are not covered here). However, the convergence could be exponentially slow, due to the high number of edges. There is the following iconic example when the simplex algorithm should perform exactly all vertexes.

:::: {.columns}

::: {.column width="60%"}
In the following problem, the simplex algorithm needs to check $2^n - 1$ vertexes with $x_0 = 0$.

$$
\begin{split} 
& \max_{x \in \mathbb{R}^n} 2^{n-1}x_1 + 2^{n-2}x_2 + \dots + 2x_{n-1} + x_n \\
\text{s.t. } & x_1 \leq 5 \\
& 4x_1 + x_2 \leq 25 \\
& 8x_1 + 4x_2 + x_3 \leq 125 \\
& \ldots \\
& 2^n x_1 + 2^{n-1}x_2 + 2^{n-2}x_3 + \ldots + x_n \leq 5^n\\ 
& x \geq 0  
\end{split}
$$
:::

::: {.column width="40%"}
![](LP_KM.pdf)
:::
::::

# Other

## Minimization of convex function as LP

![How LP can help with general convex problem](convex_via_LP.pdf){width=75%}

* The function is convex iff it can be represented as a pointwise maximum of linear functions.
* In high dimensions, the approximation may require too many functions.
* More efficient convex optimizers (not reducing to LP) exist.

## Hardware progress vs Software progress


# Mixed Integer Programming

## Mixed Integer Programming