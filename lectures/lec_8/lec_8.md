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

. . .

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
\min_{x \in \mathbb{R}^n} \|Ax - b\|_\infty \leftrightarrow \min_{x \in \mathbb{R}^n} \max_{i} |a_i^T x - b_i|
$$

Could be equivalently written as an LP with the replacement of the maximum coordinate of a vector:

. . .

$$
\begin{split}
&\min_{t \in \mathbb{R}, x \in \mathbb{R}^n} t \\
\text{s.t. } & a_i^T x - b_i \leq t, \; i = 1,\dots, n\\
& -a_i^T x + b_i \leq t, \; i = 1,\dots, n
\end{split}
$$

## $\ell_1$ approximation problem

$$
\min_{x \in \mathbb{R}^n} \|Ax - b\|_1 \leftrightarrow \min_{x \in \mathbb{R}^n} \sum_{i=1}^n |a_i^T x - b_i|
$$

Could be equivalently written as an LP with the replacement of the sum of coordinates of a vector:

. . .

$$
\begin{split}
&\min_{t \in \mathbb{R}^n, x \in \mathbb{R}^n} \mathbf{1}^T t \\
\text{s.t. } & a_i^T x - b_i \leq t_i, \; i = 1,\dots, n\\
& -a_i^T x + b_i \leq t_i, \; i = 1,\dots, n
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

* Definition: a **basis** $\mathcal{B}$ is a subset of $n$ (integer) numbers between $1$ and $m$, so that $\text{rank} A_\mathcal{B} = n$. 
* Note, that we can associate submatrix $A_\mathcal{B}$ and corresponding right-hand side $b_\mathcal{B}$ with the basis $\mathcal{B}$. 
* Also, we can derive a point of intersection of all these hyperplanes from the basis: $x_\mathcal{B} = A^{-1}_\mathcal{B} b_\mathcal{B}$. 
* If $A x_\mathcal{B} \leq b$, then basis $\mathcal{B}$ is **feasible**. 
* A basis $\mathcal{B}$ is optimal if $x_\mathcal{B}$ is an optimum of the $\text{LP.Inequality}$.
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
Since we have a basis, we can decompose our objective vector $c$ in this basis and find the scalar coefficients $\lambda_\mathcal{B}$:

$$
\lambda^T_\mathcal{B} A_\mathcal{B} = c^T \leftrightarrow \lambda^T_\mathcal{B} = c^T A_\mathcal{B}^{-1}
$$

:::{.callout-theorem}
If all components of $\lambda_\mathcal{B}$ are non-positive and $\mathcal{B}$ is feasible, then $\mathcal{B}$ is optimal.
:::

**Proof**
$$
\begin{split}
\uncover<+->{\exists x^*: Ax^* &\leq b, c^T x^* < c^T x_\mathcal{B} \\}
\uncover<+->{A_\mathcal{B} x^* &\leq b_\mathcal{B} \\}
\uncover<+->{\lambda_\mathcal{B}^T A_\mathcal{B} x^* &\geq \lambda_\mathcal{B}^T b_\mathcal{B} \\}
\uncover<+->{c^T x^* & \geq \lambda_\mathcal{B}^T A_\mathcal{B} x_\mathcal{B} \\}
\uncover<+->{c^T x^* & \geq c^T  x_\mathcal{B} \\}
\end{split}
$$
:::
::::

## Changing basis

:::: {.columns}

::: {.column width="35%"}
![](LP_change.pdf)
Suppose, some of the coefficients of $\lambda_\mathcal{B}$ are positive. Then we need to go through the edge of the polytope to the new vertex (i.e., switch the basis)
:::

::: {.column width="65%"}
* Suppose, we have a basis $\mathcal{B}$: $\lambda^T_\mathcal{B} = c^T A_\mathcal{B}^{-1}$
* Let's assume, that $\lambda^k_\mathcal{B} > 0$. We'd like to drop $k$ from the basis and form a new one:
    $$
    \uncover<+->{ \begin{cases}
    A_{\mathcal{B} \textbackslash \{k\}} d = 0 \\
    a^T_k d = -1
    \end{cases}} \qquad \uncover<+->{ c^Td } \uncover<+->{ = \lambda^T_\mathcal{B} A_\mathcal{B} d } \uncover<+->{ = \sum\limits_{i=1}^n \lambda^i_\mathcal{B}  (A_\mathcal{B} d)^i  } \uncover<+->{ = -\lambda^k_\mathcal{B} < 0 }
    $$

* For all $j \notin \mathcal{B}$ calculate the projection stepsize:
    $$
    \mu_j = \frac{b_j - a_j^T x_\mathcal{B}}{a_j^T d}
    $$
* Define the new vertex, that you will add to the new basis:
    $$
    \begin{split}
    t = \text{arg}\min_j \{\mu_j \mid \mu_j > 0\} \\
    \mathcal{B}' = \mathcal{B}\textbackslash \{k\} \cup \{t\} \\ 
    x_{\mathcal{B'}} = x_\mathcal{B} + \mu_t d = A^{-1}_{\mathcal{B'}} b_{\mathcal{B'}}
    \end{split}
    $$
* Note, that changing basis implies objective function decreasing
    $$
    c^Tx_{\mathcal{B'}} = c^T(x_\mathcal{B} + \mu_t d) = c^Tx_\mathcal{B} + \mu_t c^Td
    $$

:::
::::

## Finding an initial basic feasible solution

:::: {.columns}

::: {.column width="50%"}
We aim to solve the following problem:
$$
\begin{split}
&\min_{x \in \mathbb{R}^n} c^{\top}x \\
\text{s.t. } & Ax \leq b
\end{split}
$$ {#eq-LP_ineq}
The proposed algorithm requires an initial basic feasible solution and corresponding basis.
:::

. . .

::: {.column width="50%"}
We start by reformulating the problem:
$$
\begin{split}
&\min_{y \in \mathbb{R}^n, z \in \mathbb{R}^n} c^{\top}(y-z) \\
\text{s.t. } & Ay - Az \leq b \\ 
& y \geq 0, z \geq 0 \\ 
\end{split}
$$ {#eq-LP_ineq_new}
:::
::::

. . .

Given the solution of [Problem @eq-LP_ineq_new] the solution of [Problem @eq-LP_ineq] can be recovered and vice versa

$$
x = y-z \qquad \Leftrightarrow \qquad y_i = \max(x_i, 0), \quad z_i = \max(-x_i, 0)
$$

Now we will try to formulate new LP problem, which solution will be basic feasible point for [Problem @eq-LP_ineq_new]. Which means, that we firstly run Simplex algorithm for Phase-1 problem and run Phase-2 problem with known starting point. Note, that basic feasible solution for Phase-1 should be somehow easily established.

## Finding an initial basic feasible solution


:::: {.columns}

::: {.column width="50%"}
$$
\tag{Phase-2 (Main LP)}
\begin{split}
&\min_{y \in \mathbb{R}^n, z \in \mathbb{R}^n} c^{\top}(y-z) \\
\text{s.t. } & Ay - Az \leq b \\ 
& y \geq 0, z \geq 0 \\ 
\end{split}
$$

. . .

$$
\tag{Phase-1}
\begin{split}
&\min_{\xi \in \mathbb{R}^m, y \in \mathbb{R}^n, z \in \mathbb{R}^n} \sum\limits_{i=1}^m \xi_i \\
\text{s.t. } & Ay - Az \leq b + \xi \\ 
& y \geq 0, z \geq 0, \xi \geq 0 \\ 
\end{split}
$$
:::

. . .

::: {.column width="50%"}
* If Phase-2 (Main LP) problem has a feasible solution, then Phase-1 optimum is zero (i.e. all slacks $\xi_i$ are zero).
    
    **Proof:** trivial check.
* If Phase-1 optimum is zero (i.e. all slacks $\xi_i$ are zero), then we get a feasible basis for Phase-2.
    
    **Proof:** trivial check.

:::
::::

. . .

* Now we know, that if we can solve a Phase-1 problem then we will either find a starting point for the simplex method in the original method (if slacks are zero) or verify that the original problem was infeasible (if slacks are non-zero).
* But how to solve Phase-1? It has basic feasible solution (the problem has $2n + m$ variables and the point below ensures $2n + m$ inequalities are satisfied as equalities (active).)
    $$
    z = 0 \quad y = 0 \quad \xi_i = \max(0, -b_i)
    $$

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
* Khachiyan’s ellipsoid method (1979) is the first to be proven to run at polynomial complexity for LPs. However, it is usually slower than simplex in real problems.
* Major breakthrough - Narendra Karmarkar’s method for solving LP (1984) using interior point method.
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



# Mixed Integer Programming

## Complexity of MIP

:::: {.columns}

::: {.column width="50%"}
Consider the following Mixed Integer Programming (MIP):
$$
\begin{split}
z = 8x_1 + 11x_2 + 6x_3 + 4x_4 &\to \max_{x_1, x_2, x_3, x_4} \\
\text{s.t. }  5x_1+7x_2+4x_3+3x_4 &\leq 14 \\
x_i \in \{0,1\} \quad \forall i &
\end{split}
$$ {#eq-mip}
:::

. . .

::: {.column width="50%"}
Relax it to:
$$
\begin{split}
z = 8x_1 + 11x_2 + 6x_3 + 4x_4 &\to \max_{x_1, x_2, x_3, x_4} \\
\text{s.t. }  5x_1+7x_2+4x_3+3x_4 &\leq 14 \\
x_i \in [0,1] \quad \forall i &
\end{split}
$${#eq-mip_lp_relax}
:::
::::

. . .

:::: {.columns}

::: {.column width="50%"}
Optimal solution
$$
x_1=0, x_2=x_3=x_4=1, \text{ and } z=21.
$$
:::

. . .

::: {.column width="50%"}
Optimal solution
$$
x_1 = x_2 = 1, x_3 = 0.5, x_4 = 0, \text{ and } z = 22.
$$

. . .

* Rounding $x_3 = 0$: gives $z = 19$.
* Rounding $x_3 = 1$: Infeasible.
:::
::::

. . .

:::{.callout-important}

### MIP is much harder, than LP

* Naive rounding of LP relaxation of the initial MIP problem might lead to infeasible or suboptimal solution.
* General MIP is NP-hard.
* However, if the coefficient matrix of an MIP is a [*totally unimodular matrix*](https://en.wikipedia.org/wiki/Integer_programming), then it can be solved in polynomial time.
:::





## Unpredictable complexity of MIP

:::: {.columns}

::: {.column width="35%"}
* It is hard to predict what will be solved quickly and what will take a long time
* [\faLink Dataset](https://miplib.zib.de/index.html)
* [\faPython Source code](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/miplib.ipynb)
:::
::: {.column width="65%"}
![](miplib.pdf)
:::
::::

## Hardware progress vs Software progress

What would you choose, assuming, that the question posed correctly (you can compile software for any hardware and the problem is the same for both options)? We will consider the time period from 1992 to 2023.

:::: {.columns}

::: {.column width="50%"}
:::{.callout-caution}

### Hardware

Solving MIP with an old software on the modern hardware

:::
:::
::: {.column width="50%"}
:::{.callout-caution}

### Software

Solving MIP with a modern software on the old hardware
:::
:::
::::

. . .

:::: {.columns}

::: {.column width="50%"}
$$
\approx 1.664.510\text{ x speedup}
$$
Moore's law states, that computational power doubles every 18 monthes.

:::
::: {.column width="50%"}
$$
\approx 2.349.000\text{ x speedup}
$$
R. Bixby conducted an intensive experiment with benchmarking all CPLEX software version starting from 1992 to 2007 and measured overall software progress ($29 000$ times), later (in 2009) he was a cofounder of Gurobi optimization software, which gives additional $\approx 81$ [speedup](https://www.gurobi.com/features/gurobi-optimizer-delivers-unmatched-performance/) on MILP.
:::
::::

. . .

It turns out that if you need to solve a MILP, it is better to use an old computer and modern methods than vice versa, the newest computer and methods of the early 1990s!^[[\beamerbutton{R. Bixby report}](https://www.math.uni-bielefeld.de/documenta/vol-ismp/25_bixby-robert.pdf) [\beamerbutton{Recent study}](https://plato.asu.edu/talks/japan23.pdf)]



