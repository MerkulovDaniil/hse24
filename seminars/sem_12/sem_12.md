---
title: Conjugate gradient method
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
## Strongly convex quadratics

:::: {.columns}

::: {.column width="60%"}

Consider the following quadratic optimization problem:

$$
\min\limits_{x \in \mathbb{R}^d} f(x) =  \min\limits_{x \in \mathbb{R}^d} \dfrac{1}{2} x^\top  A x - b^\top  x + c, \text{ where }A \in \mathbb{S}^d_{++}.
$$
:::

::: {.column width="40%"}
Optimality conditions:
$$
\Delta f(x^{*}) = Ax^* - b = 0 \iff Ax^* = b
$$
:::
::::

. . .

![](sem_12/SD_vs_CG.pdf)


## Overview of the CG method for the quadratic problem
 

1) **Initialization.** $k = 0$ and $x_k = x_0$, $d_k = d_0 = -\nabla f(x_0)$.

. . .

2) **Optimal Step Length.** By the procedure of *line search* we find the optimal length of step. This involves calculate $\alpha_{k}$ minimizing $f(x_k + \alpha_k d_k)$:

$$
\alpha_k = -\frac{d_k^\top (A x_k - b)}{d_k^\top A d_k}
$$

. . .

3) **Algorithm Iteration.** Update the position of $x_k$ by moving in the direction $d_k$, with a step size $\alpha_k$:

$$
x_{k+1} = x_k + \alpha_k d_k
$$

. . .

4) **Direction Update.** Update the $d_{k+1} = -\nabla f(x_{k+1}) + \beta_k d_k$, where $\beta_k$ is calculated by the formula:

$$
\beta_k = \frac{\nabla f(x_{k+1})^\top A d_k}{d_k^\top A d_k}.
$$

. . .

5) **Convergence Loop.** Repeat steps 2-4 until $n$ directions are built, where $n$ is the dimension of space (dimension of $x$).

## Optimal Step Length

Exact line search:
$$
\alpha_k=\arg \min _{\alpha \in \mathbb{R}^{+}} f\left(x_{k+1}\right)=\arg \min _{\alpha \in \mathbb{R}^{+}} f\left(x_k + \alpha d_k\right)
$$

. . .

Let's find an analytical expression for the step $\alpha_k$:

$$
f\left(x_k + \alpha d_k\right) = \frac{1}{2} \left(x_k + \alpha d_k\right)^{\top} A\left(x_k+\alpha d_k\right) - b^{\top}\left(x_k + \alpha d_k\right) + c
$$
$$
= \frac{1}{2} \alpha^2 d_k^{\top} A d_k+d_k^{\top}\left(A x_k-b\right) \alpha+\left(\frac{1}{2} x_k^{\top} A x_k+x_k^{\top} d_k+c\right)
$$

. . .

We consider $A \in \mathbb{S}^d_{++}$, so the point with zero derivative on this parabola is a minimum:

$$
\left(d_k^{\top} A d_k\right) \alpha_k+d_k^{\top}\left(A x_k-b\right)=0 \iff \alpha_k=-\frac{d_k^{\top}\left(A x_k-b\right)}{d_k^{\top} A d_k}
$$

## Direction Update

We update the direction in such a way that the next direction is $A$ - orthogonal to the previous one:

$$
d_{k+1} \perp_A d_{k} \iff d_{k+1}^{\top} A d_{k} = 0
$$

. . .

Since $d_{k+1} = -\nabla f(x_{k+1}) + \beta_k d_k$, we choose $\beta_k$ so that there is $A$ - orthogonality:

$$
d_{k+1}^{\top} A d_{k} = -\nabla f\left(x_{k+1}\right)^{\top} A d_k + \beta_k d_k^{\top} A d_k = 0 \iff \beta_k = \frac{\nabla f\left(x_{k+1}\right)^{\top} A d_k}{d_k^{\top} A d_k}
$$

. . .

::: {.callout-tip title="Lemma 1"}
All directions of construction using the procedure described above are orthogonal to each other:
$$
d_{i}^{\top} A d_{j} = 0, \text{ if } i \neq j
$$
$$
d_{i}^{\top} A d_{j} > 0, \text{ if } i = j
$$
:::


## $A$-orthogonality

[![](A_orthogonality.pdf){#fig-aorth}](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/CG.ipynb)


## Convergence of the CG method


::: {.callout-tip title="Lemma 2"}
Suppose, we solve $n$-dimensional quadratic convex optimization problem. The conjugate directions method:
$$
x_{k+1}=x_0+\sum_{i=0}^k \alpha_i d_i,
$$

where $\alpha_i = -\dfrac{d_i^\top (A x_i - b)}{d_i^\top A d_i}$ taken from the line search, converges for at most $n$ steps of the algorithm.

:::

## CG method in practice

In practice, the following formulas are usually used for the step $\alpha_k$ and the coefficient $\beta_{k}$:

$$
\alpha_k = \dfrac{r^{\top}_k r_k}{d^{\top}_{k}A d_{k}} \qquad \beta_k = \dfrac{r^{\top}_k r_k}{r^{\top}_{k-1} r_{k-1}},
$$

where $r_k = b - Ax_k$, since $x_{k+1} = x_{k} + \alpha_k d_k$ then $r_{k+1} = r_k - \alpha_k A d_k$. Also, $r_i^T r_k = 0, \forall i \neq k$ (**Lemma 5** from the lecture).

. . .

Let's get an expression for $\beta_k$:
$$
\beta_k = \frac{\nabla f\left(x_{k+1}\right)^{\top} A d_k}{d_k^{\top} A d_k} =  -\frac{r_{k+1}^{\top} A d_k}{d_k^{\top} A d_k}
$$

. . .

$\text{Numerator: }r_{k+1}^{\top} A d_k=\frac{1}{\alpha_k} r_{k+1}^{\top}\left(r_k-r_{k+1}\right)= [r_{k+1}^{\top} r_{k} = 0] = -\frac{1}{\alpha_k} r_{k+1}^{\top} r_{k+1}$
$\text{Denominator: }d_k^{\top} A d_k=\left(r_k+\beta_{k-1} p_{k-1}\right)^{\top} {A} {p}_k=\frac{1}{\alpha_k} {r}_k^{\top}\left({r}_k-{r}_{k+1}\right)=\frac{1}{\alpha_k} {r}_k^{\top} {r}_k$

. . .

::: {.callout-question}
Why is this modification better than the standard version?
:::

## CG method in practice. Pseudocode

$$
\begin{aligned}
& \mathbf{r}_0 := \mathbf{b} - \mathbf{A x}_0 \\
& \hbox{if } \mathbf{r}_{0} \text{ is sufficiently small, then return } \mathbf{x}_{0} \text{ as the result}\\
& \mathbf{d}_0 := \mathbf{r}_0 \\
& k := 0 \\
& \text{repeat} \\
& \qquad \alpha_k := \frac{\mathbf{r}_k^\mathsf{T} \mathbf{r}_k}{\mathbf{d}_k^\mathsf{T} \mathbf{A d}_k}  \\
& \qquad \mathbf{x}_{k+1} := \mathbf{x}_k + \alpha_k \mathbf{d}_k \\
& \qquad \mathbf{r}_{k+1} := \mathbf{r}_k - \alpha_k \mathbf{A d}_k \\
& \qquad \hbox{if } \mathbf{r}_{k+1} \text{ is sufficiently small, then exit loop} \\
& \qquad \beta_k := \frac{\mathbf{r}_{k+1}^\mathsf{T} \mathbf{r}_{k+1}}{\mathbf{r}_k^\mathsf{T} \mathbf{r}_k} \\
& \qquad \mathbf{d}_{k+1} := \mathbf{r}_{k+1} + \beta_k \mathbf{d}_k \\
& \qquad k := k + 1 \\
& \text{end repeat} \\
& \text{return } \mathbf{x}_{k+1} \text{ as the result}
\end{aligned}
$$


## Non-linear conjugate gradient method

In case we do not have an analytic expression for a function or its gradient, we will most likely not be able to solve the one-dimensional minimization problem analytically. Therefore, step 2 of the algorithm is replaced by the usual line search procedure. But there is the following mathematical trick for the fourth point:

For two iterations, it is fair:

$$
x_{k+1} - x_k = c d_k,
$$

where $c$ is some kind of constant. Then for the quadratic case, we have:

$$ 
\nabla f(x_{k+1}) - \nabla f(x_k) = (A x_{k+1} - b) - (A x_k - b) = A(x_{k+1}-x_k) = cA d_k
$$

Expressing from this equation the work $Ad_k = \dfrac{1}{c} \left( \nabla f(x_{k+1}) - \nabla f(x_k)\right)$, we get rid of the "knowledge" of the function in step definition $\beta_k$, then point 4 will be rewritten as:

$$
\beta_k = \frac{\nabla f(x_{k+1})^\top (\nabla f(x_{k+1}) - \nabla f(x_k))}{d_k^\top (\nabla f(x_{k+1}) - \nabla f(x_k))}.
$$

This method is called the Polack - Ribier method.

# Computational experiments
## Computational experiments


Run code in [\faPython Colab](https://colab.research.google.com/drive/1N_PH8h8corIpVZSsXDzJ9Utpv7vVp6f6?usp=sharing). 
The code taken from [\faGithub](https://github.com/amkatrutsa/optimization_course/blob/master/Spring2022/cg.ipynb).



