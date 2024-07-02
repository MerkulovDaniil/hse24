---
title: "Conjugate gradients method"
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
header-includes:
  - \newcommand{\bgimage}{../../files/back12.jpeg}
---
# Quadratic optimization problem

## Strongly convex quadratics

:::: {.columns}

::: {.column width="60%"}
Consider the following quadratic optimization problem:
$$
\min\limits_{x \in \mathbb{R}^n} f(x) =  \min\limits_{x \in \mathbb{R}^n} \dfrac{1}{2} x^\top  A x - b^\top  x + c, \text{ where }A \in \mathbb{S}^n_{++}.
$$ {#eq-main_problem}
:::
::: {.column width="40%"}
Optimality conditions
$$
Ax^* = b
$$
:::
::::
![](SD_vs_CG.pdf)

## Exact line search aka steepest descent

:::: {.columns}
::: {.column width="80%"}
$$
\alpha_k = \text{arg}\min_{\alpha \in \mathbb{R^+}} f(x_{k+1}) = \text{arg}\min_{\alpha \in \mathbb{R^+}} f(x_k - \alpha \nabla f(x_k))
$$
More theoretical than practical approach. It also allows you to analyze the convergence, but often exact line search can be difficult if the function calculation takes too long or costs a lot.

An interesting theoretical property of this method is that each following iteration is orthogonal to the previous one:
$$
\alpha_k = \text{arg}\min_{\alpha \in \mathbb{R^+}} f(x_k - \alpha \nabla f(x_k))
$$

. . .

Optimality conditions:

. . .

$$
\nabla f(x_k)^T\nabla f(x_{k+1})  = 0
$$

:::{.callout-caution}

### Optimal value for quadratics

$$
\nabla f(x_k)^\top A (x_k - \alpha \nabla f(x_k)) - \nabla f(x_k)^\top b = 0 \qquad \alpha_k = \frac{\nabla f(x_k)^T \nabla f(x_k)}{\nabla f(x_k)^T A \nabla f(x_k)}
$$
:::
:::
::: {.column width="20%"}

![Steepest Descent](GD_vs_Steepest.pdf)

[Open In Colab $\clubsuit$](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Steepest_descent.ipynb)
:::
::::

# Orthogonality

## Conjugate directions. $A$-orthogonality.

[![](A_orthogonality.pdf){#fig-aorth}](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/CG.ipynb)

## Conjugate directions. $A$-orthogonality.

Suppose, we have two coordinate systems and some quadratic function $f(x) = \frac12 x^T I x$ looks just like on the left part of @fig-aorth, while in other coordinates it looks like $f(\hat{x}) = \frac12 \hat{x}^T A \hat{x}$, where $A \in \mathbb{S}^n_{++}$.

:::: {.columns}

::: {.column width="50%"}
$$
\frac12 x^T I x
$$
:::
::: {.column width="50%"}
$$
\frac12 \hat{x}^T A \hat{x}
$$
:::
::::
Since $A = Q \Lambda Q^T$:
$$
\uncover<+->{ \frac12 \hat{x}^T A \hat{x} }\uncover<+->{ = \frac12 \hat{x}^T Q \Lambda Q^T \hat{x} }\uncover<+->{  = \frac12 \hat{x}^T Q \Lambda^{\frac12}\Lambda^{\frac12} Q^T \hat{x} }\uncover<+->{ = \frac12 x^T I x} \uncover<+->{  \qquad \text{if } x  = \Lambda^{\frac12} Q^T \hat{x} } \uncover<+->{\text{ and }  \hat{x} = Q \Lambda^{-\frac12} x}
$$

. . .

:::{.callout-caution}

### $A$-orthogonal vectors

Vectors $x \in \mathbb{R}^n$ and $y \in \mathbb{R}^n$ are called $A$-orthogonal (or $A$-conjugate) if
$$
x^T A y = 0 \qquad \Leftrightarrow \qquad x \perp_A y 
$$
When $A = I$, $A$-orthogonality becomes orthogonality.
:::

## Gram–Schmidt process

**Input:** $n$ linearly independent vectors $u_0, \ldots, u_{n-1}$.

**Output:** $n$ linearly independent vectors, which are pairwise orthogonal $d_0, \ldots, d_{n-1}$.

![Illustration of Gram-Schmidt orthogonalization process](GS1.pdf)

## Gram–Schmidt process {.noframenumbering} 

**Input:** $n$ linearly independent vectors $u_0, \ldots, u_{n-1}$.

**Output:** $n$ linearly independent vectors, which are pairwise orthogonal $d_0, \ldots, d_{n-1}$.

![Illustration of Gram-Schmidt orthogonalization process](GS2.pdf)

## Gram–Schmidt process {.noframenumbering}

**Input:** $n$ linearly independent vectors $u_0, \ldots, u_{n-1}$.

**Output:** $n$ linearly independent vectors, which are pairwise orthogonal $d_0, \ldots, d_{n-1}$.

![Illustration of Gram-Schmidt orthogonalization process](GS3.pdf)

## Gram–Schmidt process {.noframenumbering}

**Input:** $n$ linearly independent vectors $u_0, \ldots, u_{n-1}$.

**Output:** $n$ linearly independent vectors, which are pairwise orthogonal $d_0, \ldots, d_{n-1}$.

![Illustration of Gram-Schmidt orthogonalization process](GS4.pdf)

## Gram–Schmidt process {.noframenumbering}

**Input:** $n$ linearly independent vectors $u_0, \ldots, u_{n-1}$.

**Output:** $n$ linearly independent vectors, which are pairwise orthogonal $d_0, \ldots, d_{n-1}$.

![Illustration of Gram-Schmidt orthogonalization process](GS5.pdf)

## Gram–Schmidt process

:::: {.columns}
::: {.column width="20%"}

![](GS5.pdf)

![](Projection.pdf)

:::

::: {.column width="80%"}

**Input:** $n$ linearly independent vectors $u_0, \ldots, u_{n-1}$.

. . .

**Output:** $n$ linearly independent vectors, which are pairwise orthogonal $d_0, \ldots, d_{n-1}$.
$$
\begin{aligned}
\uncover<+->{ d_0 &= u_0 \\ }
\uncover<+->{ d_1 &= u_1 - \pi_{d_0}(u_1) \\ }
\uncover<+->{ d_2 &= u_2 - \pi_{d_0}(u_2) - \pi_{d_1}(u_2) \\ }
\uncover<+->{ &\vdots \\ }
\uncover<+->{ d_k &= u_k - \sum\limits_{i=0}^{k-1}\pi_{d_i}(u_k) }
\end{aligned}
$$

. . .

$$
d_k = u_k + \sum\limits_{i=0}^{k-1}\beta_{ik} d_i \qquad \beta_{ik} = - \dfrac{\langle d_i, u_k \rangle}{\langle d_i, d_i \rangle}
$$ {#eq-GS}
:::
::::


# Conjugate Directions (CD) method

## General idea

* In an isotropic $A=I$ world, the steepest descent starting from an arbitrary point in any $n$ orthogonal linearly independent directions will converge in $n$ steps in exact arithmetic. We attempt to construct the same procedure in the case $A \neq I$ using the concept of $A$-orthogonality.
* Suppose, we have a set of $n$ linearly independent $A$-orthogonal directions $d_0, \ldots, d_{n-1}$ (which will be computed with Gram-Schmidt process). 
* We would like to build a method, that goes from $x_0$ to the $x^*$ for the quadratic problem with stepsizes $\alpha_i$, which is, in fact, just the decomposition of $x^* - x_0$ to some basis:
    $$
    x^* = x_0 + \sum\limits_{i=0}^{n-1} \alpha_i d_i \qquad x^* - x_0 = \sum\limits_{i=0}^{n-1} \alpha_i d_i
    $$
* We will prove, that $\alpha_i$ and $d_i$ could be selected in a very efficient way (Conjugate Gradient method).

## Idea of Conjugate Directions (CD) method

Thus, we formulate an algorithm:

1. Let $k = 0$ and $x_k = x_0$, count $d_k = d_0 = -\nabla f(x_0)$.
2. By the procedure of line search we find the optimal length of step. Calculate $\alpha$ minimizing $f(x_k + \alpha_k d_k)$ by the formula
    $$
    \alpha_k = -\frac{d_k^\top (A x_k - b)}{d_k^\top A d_k}
    $$ {#eq-line_search}
3. We're doing an algorithm step:
    $$
    x_{k+1} = x_k + \alpha_k d_k
    $$
4. Update the direction: $d_{k+1} = -\nabla f(x_{k+1}) + \beta_k d_k$ in order to make $d_{k+1} \perp_A d_k$, where $\beta_k$ is calculated by the formula:
    $$
    \beta_k = \frac{\nabla f(x_{k+1})^\top A d_k}{d_k^\top A d_k}.
    $$
5. Repeat steps 2-4 until $n$ directions are built, where $n$ is the dimension of space (dimension of $x$).

## Conjugate Directions (CD) method

::: {.callout-theorem}

## Lemma 1. Linear independence of $A$-conjugate vectors.

If a set of vectors $d_1, \ldots, d_n$ - are $A$-conjugate (each pair of vectors is $A$-conjugate), these vectors are linearly independent. $A \in \mathbb{S}^n_{++}$.

:::

. . .

**Proof**

We'll show, that if $\sum\limits_{i=1}^n\alpha_i d_i = 0$, than all coefficients should be equal to zero:

. . .

$$
\begin{aligned}
\uncover<+->{ 0 &= \sum\limits_{i=1}^n\alpha_i d_i \\ }
\uncover<+->{ \text{Multiply by } d_j^T A \cdot \qquad &= d_j^\top A \left( \sum\limits_{i=1}^n\alpha_i d_i \right) }
\uncover<+->{ =  \sum\limits_{i=1}^n \alpha_i d_j^\top A d_i  \\ }
\uncover<+->{ &=  \alpha_j d_j^\top A d_j  + 0 + \ldots + 0 }
\end{aligned}
$$

. . .

Thus, $\alpha_j = 0$, for all other indices one has to perform the same process

## Proof of convergence

We will introduce the following notation:

* $r_k = b - Ax_k$ - residual,
* $e_k = x_k - x^*$ - error.
* Since $Ax^* = b$, we have $r_k = b - Ax_k = Ax^* - Ax_k = -A (x_k - x^*)$
    $$
    r_k = -Ae_k.
    $$ {#eq-res_error}
* Note also, that since $x_{k+1} = x_0 + \sum\limits_{i=1}^k\alpha_i d_i$, we have 
    $$
    e_{k+1} = e_0 + \sum\limits_{i=1}^k\alpha_i d_i.
    $$ {#eq-err_decomposition}

## Proof of convergence

::: {.callout-theorem}

## Lemma 2. Convergence of conjugate direction method.

Suppose, we solve $n$-dimensional quadratic convex optimization problem ([-@eq-main_problem]). The conjugate directions method
$$
x_{k+1} = x_0 + \sum\limits_{i=0}^k\alpha_i d_i
$$
with $\alpha_i = \frac{\langle d_i, r_i \rangle}{\langle d_i, Ad_i \rangle}$ taken from the line search, converges for at most $n$ steps of the algorithm.
:::

. . .

:::: {.columns}
::: {.column width="33%"}
**Proof**

1. We need to prove, that $\delta_i = - \alpha_i$:
    $$
    e_0 = x_0 - x^* =  \sum\limits_{i=0}^{n-1}\delta_i d_i
    $$
:::

. . .

::: {.column width="66%"}
2. We multiply both hand sides from the left by $d_k^T A$:
    $$
    \begin{aligned}
    \uncover<+->{ d_k^T Ae_0 &= \sum\limits_{i=0}^{n-1}\delta_i d_k^T A d_i}\uncover<+->{  = \delta_k d_k^T A d_k \\}
    \uncover<+->{ d_k^T A\left(e_0 + \sum\limits_{i=0}^{k-1}\alpha_i d_i \right)}\uncover<+->{ = d_k^T A e_k }\uncover<+->{  &= \delta_k d_k^T A d_k \quad \left(A-\text{ orthogonality}\right)\\}
    \uncover<+->{ \delta_k = \frac{ d_k^T A e_k}{d_k^T A d_k }}\uncover<+->{ = -\frac{ d_k^T r_k}{d_k^T A d_k } }\uncover<+->{  &\Leftrightarrow \delta_k = - \alpha_k }
    \end{aligned}
    $$
:::
::::

## Lemms for convergence

::: {.callout-theorem}

## Lemma 3. Error decomposition

$$
e_i = \sum\limits_{j=i}^{n-1}-\alpha_j d_j 
$$ {#eq-err_decomposition}

:::

. . .

**Proof**

By definition
$$
\uncover<+->{ e_{i} = e_0 + \sum\limits_{j=0}^{i-1}\alpha_j d_j }\uncover<+->{ = x_0 - x^* + \sum\limits_{j=0}^{i-1}\alpha_j d_j }\uncover<+->{  = -\sum\limits_{j=0}^{n-1}\alpha_j d_j + \sum\limits_{j=0}^{i-1}\alpha_j d_j}\uncover<+->{  = \sum\limits_{j=i}^{n-1}-\alpha_j d_j }
$$

## Lemms for convergence

::: {.callout-theorem}

## Lemma 4. Residual is orthogonal to all previous directions for CD

Consider residual of the CD method at $k$ iteration $r_k$, then for any $i < k$:

$$
d_i^T r_k = 0
$$ {#eq-res_orth_dir}

:::

. . .

**Proof**

:::: {.columns}
::: {.column width="40%"}
Let's write down ([-@eq-err_decomposition]) for some fixed index $k$:

. . .

$$
e_k = \sum\limits_{j=k}^{n-1}-\alpha_j d_j 
$$

. . .

Multiply both sides by $-d_i^TA \cdot$
$$
-d_i^TA e_k = \sum\limits_{j=k}^{n-1}\alpha_j d_i^TA d_j  = 0
$$
:::

::: {.column width="60%"}
![](CG_lem1.pdf)
Thus, $d_i^T r_k = 0$ and residual $r_k$ is orthogonal to all previous directions $d_i$ for the CD method.
:::
::::

# Conjugate Gradients (CG) method

## The idea of the Conjugate Gradients (CG) method

* It is literally the Conjugate Direction method, where we have a special (effective) choice of $d_0, \ldots, d_{n-1}$.
* In fact, we use the Gram-Schmidt process with $A$-orthogonality instead of Euclidian orthogonality to get them from a set of starting vectors.
* The residuals on each iteration $r_0, \ldots, r_{n-1}$ are used as starting vectors for Gram-Schmidt process.
* The main idea is that for an arbitrary CD method, the Gramm-Schmidt process is quite computationally expensive and requires a quadratic number of vector addition and scalar product operations $\mathcal{O}\left( n^2\right)$, while in the case of CG, we will show that the complexity of this procedure can be reduced to linear $\mathcal{O}\left( n\right)$.

. . .

:::{.callout-caution appearance="simple"}
$$
\text{CG} = \text{CD} + r_0, \ldots, r_{n-1} \text{ as starting vectors for Gram–Schmidt} + A\text{-orthogonality.}
$$
:::

# Conjugate gradients (CG) method

## Lemms for convergence

::: {.callout-theorem}

## Lemma 5. Residuals are orthogonal to each other in the CG method

All residuals are pairwise orthogonal to each other in the CG method:
$$
r_i^T r_k = 0 \qquad \forall i \neq k
$$ {#eq-res_orth_cg}

:::

. . .

:::: {.columns}
::: {.column width="40%"}

**Proof**

Let's write down Gram-Schmidt process ([-@eq-GS]) with $\langle \cdot, \cdot \rangle$ replaced with $\langle \cdot, \cdot \rangle_A = x^T A y$

. . .

$$
d_i = u_i + \sum\limits_{j=0}^{k-1}\beta_{ji} d_j \;\; \beta_{ji} = - \dfrac{\langle d_j, u_i \rangle_A}{\langle d_j, d_j \rangle_A}
$$ {#eq-gs_cg1}

. . .

Then, we use residuals as starting vectors for the process and $u_i = r_i$.

. . .

$$ 
d_i = r_i + \sum\limits_{j=0}^{k-1}\beta_{ji} d_j \;\; \beta_{ji} = - \dfrac{\langle d_j, r_i \rangle_A}{\langle d_j, d_j \rangle_A}
$$ {#eq-gs_cg2}
:::

::: {.column width="60%"}
![](CG_lem1.pdf)
Multiply both sides of ([-@eq-gs_cg1]) by $r_k^T \cdot$ for some index $k$:
$$
r_k^Td_i = r_k^Tu_i + \sum\limits_{j=0}^{k-1}\beta_{ji} r_k^Td_j 
$$

. . .

If $j < i < k$, we have the lemma 4 with $d_i^T r_k = 0$ and $d_j^T r_k = 0$. We have:
$$
r_k^Tu_i= 0 \;\text{ for CD} \;\; r_k^Tr_i = 0 \;\text{ for CG}
$$
:::
::::

## Lemms for convergence

Moreover, if $k=i$:
$$
\uncover<+->{ r_k^Td_k = r_k^Tu_k + \sum\limits_{j=0}^{k-1}\beta_{jk} r_k^Td_j}\uncover<+->{  = r_k^Tu_k + 0,}
$$

. . .

and we have for any $k$ (due to arbitrary choice of $i$):
$$
r_k^Td_k = r_k^Tu_k.
$$ {#eq-lemma5}

. . .

::: {.callout-theorem}

## Lemma 6. Residual recalculation

$$
r_{k+1} = r_k - \alpha_k A d_k 
$$ {#eq-res_recalculation}

:::

. . .

$$
r_{k+1} = -A e_{k+1} = -A \left( e_{k} + \alpha_k d_k \right) = -A e_{k} - \alpha_k A d_k = r_k - \alpha_k A d_k 
$$

Finally, all these above lemmas are enough to prove, that $\beta_{ji} = 0$ for all $i,j$, except the neighboring ones.

## Gram-Schmidt process in CG method

Consider the Gram-Schmidt process in the CG method
$$
\uncover<+->{ \beta_{ji} = - \dfrac{\langle d_j, u_i \rangle_A}{\langle d_j, d_j \rangle_A} }\uncover<+->{  = - \dfrac{ d_j^T A u_i }{ d_j^T A d_j }}\uncover<+->{  = - \dfrac{ d_j^T A r_i }{ d_j^T A d_j }}\uncover<+->{  = - \dfrac{r_i^T A d_j}{ d_j^T A d_j }.}
$$

. . .

Consider the scalar product $\langle r_i, r_{j+1} \rangle$ using ([-@eq-res_recalculation]):
$$
\begin{aligned}
\uncover<+->{ \langle r_i, r_{j+1} \rangle}\uncover<+->{  &= \langle r_i, r_j - \alpha_j A d_j  \rangle }\uncover<+->{ = \langle r_i, r_j \rangle - \alpha_j\langle r_i, A d_j  \rangle \\}
\uncover<+->{ \alpha_j\langle r_i, A d_j  \rangle }\uncover<+->{  &= \langle r_i, r_j \rangle - \langle r_i, r_{j+1} \rangle }
\end{aligned}
$$

1. If $i=j$: $\alpha_i\langle r_i, A d_i  \rangle = \langle r_i, r_i \rangle - \langle r_i, r_{i+1} \rangle = \langle r_i, r_i \rangle$. This case is not of interest due to the GS process.
2. Neighboring case $i=j + 1$: $\alpha_j\langle r_i, A d_j \rangle = \langle r_i, r_{i-1} \rangle - \langle r_i, r_{i} \rangle = - \langle r_i, r_i \rangle$
3. For any other case: $\alpha_j\langle r_i, A d_j \rangle = 0$, because all residuals are orthogonal to each other.

. . .

Finally, we have a formula for $i=j + 1$:
$$
\uncover<+->{ \beta_{ji} = - \dfrac{r_i^T A d_j}{ d_j^T A d_j}}\uncover<+->{  = \dfrac{1}{\alpha_j}\dfrac{\langle r_i, r_i \rangle}{ d_j^T A d_j} }\uncover<+->{  =  \dfrac{d_j^T A d_j}{d_j^T r_j}\dfrac{\langle r_i, r_i \rangle}{ d_j^T A d_j} }\uncover<+->{ = \dfrac{\langle r_i, r_i \rangle}{\langle r_j, r_j \rangle} }\uncover<+->{ = \dfrac{\langle r_i, r_i \rangle}{\langle r_{i-1}, r_{i-1} \rangle}}
$$

. . .

And for the direction
$$
d_{k+1} = r_{k+1} + \beta_{k,k+1} d_k, \qquad  \beta_{k,k+1} = \beta_k = \dfrac{\langle r_{k+1}, r_{k+1} \rangle}{\langle r_{k}, r_{k} \rangle}.
$$

## Conjugate gradients method

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

## Convergence

**Theorem 1.** If matrix $A$ has only $r$ different eigenvalues, then the conjugate gradient method converges in $r$ iterations.

**Theorem 2.** The following convergence bound holds

$$
\| x_{k} - x^* \|_A \leq 2\left( \dfrac{\sqrt{\kappa(A)} - 1}{\sqrt{\kappa(A)} + 1} \right)^k \|x_0 - x^*\|_A,
$$

where $\|x\|^2_A = x^{\top}Ax$ and $\kappa(A) = \frac{\lambda_1(A)}{\lambda_n(A)}$ is the conditioning number of matrix $A$, $\lambda_1(A) \geq ... \geq \lambda_n(A)$ are the eigenvalues of matrix $A$

**Note:** Compare the coefficient of the geometric progression with its analog in gradient descent.

## Numerical results

$$
f(x) = \frac{1}{2} x^T A x - b^T x \to \min_{x \in \mathbb{R}^n}
$$

![](cg_random_0.001_100_60.pdf)

## Numerical results

$$
f(x) = \frac{1}{2} x^T A x - b^T x \to \min_{x \in \mathbb{R}^n}
$$

![](cg_random_10_100_60.pdf)

## Numerical results

$$
f(x) = \frac{1}{2} x^T A x - b^T x \to \min_{x \in \mathbb{R}^n}
$$

![](cg_random_10_1000_60.pdf)

## Numerical results

$$
f(x) = \frac{1}{2} x^T A x - b^T x \to \min_{x \in \mathbb{R}^n}
$$

![](cg_clustered_10_1000_60.pdf)

## Numerical results

$$
f(x) = \frac{1}{2} x^T A x - b^T x \to \min_{x \in \mathbb{R}^n}
$$

![](cg_clustered_10_1000_600.pdf)

## Numerical results

$$
f(x) = \frac{1}{2} x^T A x - b^T x \to \min_{x \in \mathbb{R}^n}
$$

![](cg_uniform spectrum_1_100_60.pdf)

## Numerical results

$$
f(x) = \frac{1}{2} x^T A x - b^T x \to \min_{x \in \mathbb{R}^n}
$$

![](cg_Hilbert_1_10_60.pdf)

# Non-linear CG

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

This method is called the Polack-Ribier method.

## Numerical results

$$
f(x) = \frac{\mu}{2} \|x\|_2^2 + \frac1m \sum_{i=1}^m \log (1 + \exp(- y_i \langle a_i, x \rangle)) \to \min_{x \in \mathbb{R}^n}
$$

![](cg_non_linear_1000_300_0_None.pdf)

## Numerical results

$$
f(x) = \frac{\mu}{2} \|x\|_2^2 + \frac1m \sum_{i=1}^m \log (1 + \exp(- y_i \langle a_i, x \rangle)) \to \min_{x \in \mathbb{R}^n}
$$

![](cg_non_linear_1000_300_1_None.pdf)

## Numerical results

$$
f(x) = \frac{\mu}{2} \|x\|_2^2 + \frac1m \sum_{i=1}^m \log (1 + \exp(- y_i \langle a_i, x \rangle)) \to \min_{x \in \mathbb{R}^n}
$$

![](cg_non_linear_1000_300_1_20.pdf)

## Numerical results

$$
f(x) = \frac{\mu}{2} \|x\|_2^2 + \frac1m \sum_{i=1}^m \log (1 + \exp(- y_i \langle a_i, x \rangle)) \to \min_{x \in \mathbb{R}^n}
$$

![](cg_non_linear_1000_300_1_50.pdf)

## Numerical results

$$
f(x) = \frac{\mu}{2} \|x\|_2^2 + \frac1m \sum_{i=1}^m \log (1 + \exp(- y_i \langle a_i, x \rangle)) \to \min_{x \in \mathbb{R}^n}
$$

![](cg_non_linear_1000_300_10_None.pdf)

## Numerical results

$$
f(x) = \frac{\mu}{2} \|x\|_2^2 + \frac1m \sum_{i=1}^m \log (1 + \exp(- y_i \langle a_i, x \rangle)) \to \min_{x \in \mathbb{R}^n}
$$

![](cg_non_linear_1000_300_10_20.pdf)