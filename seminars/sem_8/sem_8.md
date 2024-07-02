---
title: Linear Programming and simplex algorithm.
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

# Linear Programming 

## Linear Programming Recap. Common Forms

:::: {.columns}

::: {.column width="50%"}

For some vectors $c \in \mathbb{R}^n$, $b \in \mathbb{R}^m$ and matrix $A \in \mathbb{R}^{m \times n}$


* Basic form of Linear Programming Problem is:
\begin{align*}
\tag{LP.Basic}
&\min_{x \in \mathbb{R}^n} c^{\top}x \\
\text{s.t. } & Ax \leq b\\
\end{align*}


* Standard Form of Linear Programming Problem is:
\begin{align*}
\tag{LP.Standard}
&\min_{x \in \mathbb{R}^n} c^{\top}x \\
\text{s.t. } & Ax = b\\
& x_i \geq 0, \; i = 1,\dots, n
\end{align*}


:::

::: {.column width="50%"}

![Illustration of the LP Problem.](linear_programming_problem.pdf){width=250}

:::

::::



## Linear Programming Recap. Primal and Dual Problems
There are four possibilities:

1. Both the primal and the dual are infeasible.
1. The primal is infeasible and the dual is unbounded.
1. The primal is unbounded and the dual is infeasible.
1. Both the primal and the dual are feasible and their optimal values are equal.


# Simplex Algorithm

## Simplex Algorithm Foundations
:::: {.columns}

::: {.column width="50%"}

![Simplex Algorithm main notions.](LP_basis.pdf){width=150}

:::

::: {.column width="50%"}

![Simplex Algorithm basis change.](LP_change.pdf){width=150}

:::

::::

::: {.callout-note title="Simplex Algorithm main notions"}

* A **basis** $B$ is a subset of $n$ (integer) numbers between $1$ and $m$, so that $\text{rank} A_B = n$. Note, that we can associate submatrix $A_B$ and corresponding right-hand side $b_B$ with the basis $B$. Also, we can derive a point of intersection of all these hyperplanes from basis: $x_B = A^{-1}_B b_B$.
* If $A x_B \leq b$, then basis $B$ is **feasible**.
* A basis $B$ is **optimal** if $x_B$ is an optimum of the $\texttt{LP.Basic}$.

::: 



## Simplex Algorithm Foundations
:::: {.columns}

::: {.column width="50%"}

![Simplex Algorithm main notions.](LP_basis.pdf){width=150}

:::

::: {.column width="50%"}

![Simplex Algorithm basis change.](LP_change.pdf){width=150}

:::

::::

::: {.callout-note title="Simplex Algorithm Intuition"}

* The Simplex Algorithm walks along the edges of the polytope, at every corner choosing the edge that decreases $c^\top x$ most
* This either terminates at a corner, or leads to an unconstrained edge ($-\infty$ optimum)

::: 


## Simplex Algorithm Foundations
:::: {.columns}

::: {.column width="50%"}

![Simplex Algorithm main notions.](LP_basis.pdf){width=150}

:::

::: {.column width="50%"}

![Simplex Algorithm basis change.](LP_change.pdf){width=150}

:::

::::


::: {.callout-tip title="Existence of the Standard LP Problem Solution"}

1. If Standartd LP has a nonempty feasible region, then there is at least one basic feasible point
1. If Standartd LP has solutions, then at least one such solution is a basic optimal point.
1. If Standartd LP is feasible and bounded, then it has an optimal solution.

::: 



## Simplex Algorithm Foundations
:::: {.columns}

::: {.column width="50%"}

![Simplex Algorithm main notions.](LP_basis.pdf){width=150}

:::

::: {.column width="50%"}

![Simplex Algorithm basis change.](LP_change.pdf){width=150}

:::

::::

::: {.callout-tip title="Corner Optimality Theorem"}
Let $\lambda_B$ be the coordinates of our objective vector $c$ in the basis $B$:
$$
\lambda^\top_B A_B = c^\top \leftrightarrow \lambda^\top_B = c^\top A_B^{-1}
$$

If all components of $\lambda_B$ are non-positive and $B$ is feasible, then $B$ is optimal. 

::: 



<!-- # LP Problems Examples

## LP Problems Examples. Klee Minty example
In the following problem simplex algorithm needs to check $2^n - 1$ vertexes with $x_0 = 0$:

:::: {.columns}

::: {.column width="50%"}

\begin{align*} & \max_{x \in \mathbb{R}^n} 2^{n-1}x_1 + 2^{n-2}x_2 + \dots + 2x_{n-1} + x_n\\
\text{s.t. } & x_1 \leq 5\\
& 4x_1 + x_2 \leq 25\\
& 8x_1 + 4x_2 + x_3 \leq 125\\
& \ldots\\
& 2^n x_1 + 2^{n-1}x_2 + 2^{n-2}x_3 + \ldots + x_n \leq 5^n\ & x \geq 0
\tag{Klee-Minty Problem}
\end{align*}

:::

::: {.column width="50%"}

![Convergense of Simplex Algorithm for the Klee-Minty Problem.](linear_programming_examples_klee_minty.pdf){width=250}

Open [\faPython Relevant Collab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/LP.ipynb#scrollTo=GrKuLNXYik8V)

:::

:::: -->

# LP Problems Examples

## LP Problems Examples. Production Plans

Suppose you are thinking about starting up a business to produce a *Product X*. 

Let's find the maximum weekly profit for your business in the [\faPython Production Plan Problem](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/LP.ipynb#scrollTo=-VNwdz5RDiYu).



## LP Problems Examples. Max Flow Min Cut Problem

See Outer Presentation.



## LP Problems Examples. Different Applications

Look at different practical applications of LP Problems and Simplex Algorithm in the [\faPython Related Collab Notebook ](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/LP.ipynb#scrollTo=GrKuLNXYik8V).

