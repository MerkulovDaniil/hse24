---
title: "Gradient methods for conditional problems. Projected Gradient Descent. Frank-Wolfe method. Idea of Mirror Descent algorithm."
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
            \titlegraphic{\includegraphics[width=0.5\paperwidth]{back13.jpeg}}
---
# Conditional methods

## Constrained optimization

:::: {.columns}

::: {.column width="50%"}

### Unconstrained optimization

$$
\min_{x \in \mathbb{R}^n} f(x)
$$

* Any point $x_0 \in \mathbb{R}^n$ is feasible and could be a solution.

:::

. . .

::: {.column width="50%"}

### Constrained optimization

$$
\min_{x \in S} f(x)
$$

* Not all $x \in \mathbb{R}^n$ are feasible and could be a solution.
* The solution has to be inside the set $S$.
* Example: 
    $$
    \frac12\|Ax - b\|_2^2 \to \min_{\|x\|_2^2 \leq 1}
    $$

:::

::::

. . .

Gradient Descent is a great way to solve unconstrained problem 
$$
\tag{GD}
x_{k+1} = x_k - \alpha_k \nabla f(x_k)
$$
Is it possible to tune GD to fit constrained problem? 

. . .

**Yes**. We need to use projections to ensure feasibility on every iteration.

## Example: White-box Adversarial Attacks 

:::: {.columns}

::: {.column width="55%"}

![[Source](https://arxiv.org/abs/1811.07018)](adversarial.jpeg)

:::

::: {.column width="45%"}

* Mathematically, a neural network is a function $f(w; x)$
* Typically, input $x$ is given and network weights $w$ optimized
* Could also freeze weights $w$ and optimize $x$, adversarially!
$$ 
\min_{\delta} \text{size}(\delta) \quad \text{s.t.} \quad \text{pred}[f(w;x+\delta)] \neq y
$$
or 
$$
\max_{\delta} l(w; x+\delta, y) \; \text{s.t.} \; \text{size}(\delta) \leq \epsilon, \; 0 \leq x+\delta \leq 1
$$
:::
::::


## Idea of Projected Gradient Descent

![Suppose, we start from a point $x_k$.](PGD1.pdf)

## Idea of Projected Gradient Descent{.noframenumbering}

![And go in the direction of $-\nabla f(x_k)$.](PGD2.pdf)

## Idea of Projected Gradient Descent{.noframenumbering}

![Occasionally, we can end up outside the feasible set.](PGD3.pdf)

## Idea of Projected Gradient Descent{.noframenumbering}

![Solve this little problem with projection!](PGD4.pdf)

## Idea of Projected Gradient Descent

$$
x_{k+1} = \text{proj}_S\left(x_k - \alpha_k \nabla f(x_k) \right)  \qquad \Leftrightarrow \qquad \begin{aligned}
y_k &= x_k - \alpha_k \nabla f(x_k) \\
x_{k+1} &= \text{proj}_S\left( y_k\right)
\end{aligned}
$$

![Illustration of Projected Gradient Descent algorithm](PGD.pdf)

# Projection

## Projection

The distance $d$ from point $\mathbf{y} \in \mathbb{R}^n$ to closed set $S \subset \mathbb{R}^n$:
$$
d(\mathbf{y}, S, \| \cdot \|) = \inf\{\|x - y\| \mid x \in S \}
$$

. . .

We will focus on Euclidean projection (other options are possible) of a point $\mathbf{y} \in \mathbb{R}^n$ on set $S \subseteq \mathbb{R}^n$ is a point $\text{proj}_S(\mathbf{y}) \in S$: 
$$
\text{proj}_S(\mathbf{y}) = \frac12 \underset{\mathbf{x} \in S}{\operatorname{argmin}} \|x - y\|_2^2
$$

. . .

* **Sufficient conditions of existence of a projection**. If $S \subseteq \mathbb{R}^n$ - closed set, then the projection on set $S$ exists for any point.
* **Sufficient conditions of uniqueness of a projection**. If $S \subseteq \mathbb{R}^n$ - closed convex set, then the projection on set $S$ is unique for any point.
* If a set is open, and a point is beyond this set, then its projection on this set does not exist.
* If a point is in set, then its projection is the point itself.

## Projection criterion (Bourbaki-Cheney-Goldstein inequality)

:::: {.columns}

::: {.column width="65%"}

:::{.callout-theorem}
\small
Let $S \subseteq \mathbb{R}^n$ be closed and convex, $\forall x \in S, y \in \mathbb{R}^n$. Then
$$
\langle y - \text{proj}_S(y), \mathbf{x} - \text{proj}_S(y)\rangle \leq 0
$$ {#eq-proj1}
$$
\|x - \text{proj}_S(y)\|^2 + \|y - \text{proj}_S(y)\|^2 \leq \|x-y\|^2
$$ {#eq-proj2}

**Proof**

1. $\text{proj}_S(y)$ is minimizer of differentiable convex function $d(y, S, \| \cdot \|) = \|x - y\|^2$ over $S$. By first-order characterization of optimality.
    $$
    \begin{aligned}
    \uncover<+->{\nabla d(\text{proj}_S(y))^T(x - \text{proj}_S(y))&\geq 0 \\ }
    \uncover<+->{2\left(\text{proj}_S(y) - y \right)^T(x - \text{proj}_S(y))&\geq 0 \\ }
    \uncover<+->{\left(y - \text{proj}_S(y) \right)^T(x - \text{proj}_S(y))&\leq 0}
    \end{aligned}
    $$
2. Use cosine rule $2x^Ty = \|x\|^2 + \|y\|^2 - \|x-y\|^2$ with $x = x - \text{proj}_S(y)$ and $y = y - \text{proj}_S(y)$. By the first property of the theorem:
    $$
    \begin{aligned}
    \uncover<+->{ 0 \geq 2x^Ty = \|x - \text{proj}_S(y)\|^2 + \|y + \text{proj}_S(y)\|^2 - \|x-y\|^2 \\ }
    \uncover<+->{ \|x - \text{proj}_S(y)\|^2 + \|y + \text{proj}_S(y)\|^2 \leq \|x-y\|^2 }
    \end{aligned}
    $$
:::
:::

::: {.column width="35%"}
![Obtuse or straight angle should be for any point $x \in S$](proj_crit.pdf)
:::
::::

## Projection operator is non-expansive

* A function $f$ is called non-expansive if $f$ is $L$-Lipschitz with $L \leq 1$ ^[Non-expansive becomes contractive if $L < 1$.]. That is, for any two points $x,y \in \text{dom} f$,
    $$
    \|f(x)-f(y)\| \leq L\|x-y\|, \text{ where } L \leq 1.
    $$
    It means the distance between the mapped points is possibly smaller than that of the unmapped points.

* Projection operator is non-expansive:
    $$
    \| \text{proj}(x) - \text{proj}(y) \|_2 \leq \| x - y \|_2.
    $$

* Next: variational characterization implies non-expansiveness. i.e.,
    $$
    \langle y - \text{proj}(y), x - \text{proj}(y) \rangle \leq 0 \quad \forall x \in S \qquad \Rightarrow \qquad \| \text{proj}(x) - \text{proj}(y) \|_2 \leq \| x - y \|_2.
    $$ 

## Projection operator is non-expansive

Shorthand notation: let $\pi = \text{proj}$ and $\pi(x)$ denotes $\text{proj}(x)$.

. . .

Begins with the variational characterization / obtuse angle inequality
$$
\langle y-\pi(y) , x-\pi(y) \rangle \leq 0 \quad \forall x \in S.
$$ {#eq-proj1}

. . .

:::: {.columns}

::: {.column width="50%"}
Replace $x$ by $\pi(x)$ in @eq-proj1
$$
\langle y-\pi(y), \pi(x)-\pi(y) \rangle \leq 0.
$$ {#eq-proj2}
:::

. . .

::: {.column width="50%"}
Replace $y$ by $x$ and $x$ by $\pi(y)$ in @eq-proj1

$$
\langle x-\pi(x), \pi(y)-\pi(x) \rangle \leq 0.
$$ {#eq-proj3}
:::

::::

. . .

(@eq-proj2)+(@eq-proj3) will cancel $\pi(y) - \pi(x)$, not good. So flip the sign of (@eq-proj3) gives
$$
\langle \pi(x)-x, \pi(x)-\pi(y) \rangle \leq 0.
$$ {#eq-proj4}

. . .

:::: {.columns}

::: {.column width="60%"}
$$
\begin{split}
\langle y-\pi(y)+\pi(x)-x , \pi(x)-\pi(y) \rangle & \leq 0 \\
\langle y-x+\pi(x)-\pi(y), \pi(x)-\pi(y) \rangle & \leq 0 \\
\langle y - x, \pi(x) - \pi(y) \rangle & \leq -\langle \pi(x)-\pi(y), \pi(x)-\pi(y) \rangle \\
\langle y - x, \pi(y) - \pi(x) \rangle & \geq \lVert \pi(x) - \pi(y) \rVert^2_2 \\
\lVert (y - x)^\top (\pi(y) - \pi(x)) \rVert_2 & \geq \lVert \pi(x) - \pi(y) \rVert^2_2
\end{split}
$$
:::

. . .

::: {.column width="40%"}
By Cauchy-Schwarz inequality, the left-hand-side is upper bounded by $\lVert y - x \rVert_2 \lVert \pi(y) - \pi(x) \rVert_2$, we get $\lVert y - x \rVert_2 \lVert \pi(y) - \pi(x) \rVert_2 \geq \lVert \pi(x) - \pi(y) \rVert^2_2$. Cancels $\lVert \pi(x) - \pi(y) \rVert_2$ finishes the proof.
:::

::::

## Example: projection on the ball

Find $\pi_S (y) = \pi$, if $S = \{x \in \mathbb{R}^n \mid \|x - x_0\| \le R \}$, $y \notin S$ 

. . .

Build a hypothesis from the figure: $\pi = x_0 + R \cdot \frac{y - x_0}{\|y - x_0\|}$ 

. . .

:::: {.columns}

::: {.column width="60%"}
Check the inequality for a convex closed set: $(\pi - y)^T(x - \pi) \ge 0$ 

. . .

$$
\begin{split}
\left( x_0 - y + R \frac{y - x_0}{\|y - x_0\|} \right)^T\left( x - x_0 - R \frac{y - x_0}{\|y - x_0\|} \right) &= \\
\left( \frac{(y - x_0)(R - \|y - x_0\|)}{\|y - x_0\|} \right)^T\left( \frac{(x-x_0)\|y-x_0\|-R(y - x_0)}{\|y - x_0\|} \right) &= \\
\frac{R - \|y - x_0\|}{\|y - x_0\|^2} \left(y - x_0 \right)^T\left( \left(x-x_0\right)\|y-x_0\|-R\left(y - x_0\right) \right) &= \\
\frac{R - \|y - x_0\|}{\|y - x_0\|} \left( \left(y - x_0 \right)^T\left( x-x_0\right)-R\|y - x_0\| \right) &= \\
\left(R - \|y - x_0\| \right) \left( \frac{(y - x_0 )^T( x-x_0)}{\|y - x_0\|}-R \right) & \\
\end{split}
$$
:::

. . .

::: {.column width="40%"}
The first factor is negative for point selection $y$. The second factor is also negative, which follows from the Cauchy-Bunyakovsky inequality: 

. . .

$$
\begin{split}
(y - x_0 )^T( x-x_0) &\le \|y - x_0\|\|x-x_0\| \\
\frac{(y - x_0 )^T( x-x_0)}{\|y - x_0\|} - R &\le \frac{\|y - x_0\|\|x-x_0\|}{\|y - x_0\|} - R = \|x - x_0\| - R \le 0
\end{split}
$$

![Ball](proj_ball.pdf){width=60%}

:::
::::

## Example: projection on the halfspace

Find $\pi_S (y) = \pi$, if $S = \{x \in \mathbb{R}^n \mid c^T x = b \}$, $y \notin S$. Build a hypothesis from the figure: $\pi = y + \alpha c$. Coefficient $\alpha$ is chosen so that $\pi \in S$: $c^T \pi = b$, so:

. . .

:::: {.columns}

::: {.column width="50%"}
![Hyperplane](proj_half.pdf)
:::

. . .

::: {.column width="50%"}
$$
\begin{split}
c^T (y + \alpha c) &= b \\
c^Ty + \alpha c^T c &= b \\
c^Ty &= b - \alpha c^T c \\
\end{split}
$$
Check the inequality for a convex closed set: $(\pi - y)^T(x - \pi) \ge 0$ 
$$
\begin{split}
(y + \alpha c - y)^T(x - y - \alpha c) =& \\
\alpha c^T(x - y - \alpha c) =& \\
\alpha (c^Tx) - \alpha (c^T y) - \alpha^2 (c^Tc) =& \\
\alpha b - \alpha (b - \alpha c^T c) - \alpha^2 c^Tc =& \\
\alpha b - \alpha b + \alpha^2 c^T c - \alpha^2 c^Tc =& 0 \ge 0
\end{split}
$$
:::

::::
 
# Projected Gradient Descent (PGD)

## Idea

$$
x_{k+1} = \text{proj}_S\left(x_k - \alpha_k \nabla f(x_k) \right)  \qquad \Leftrightarrow \qquad \begin{aligned}
y_k &= x_k - \alpha_k \nabla f(x_k) \\
x_{k+1} &= \text{proj}_S\left( y_k\right)
\end{aligned}
$$

![Illustration of Projected Gradient Descent algorithm](PGD.pdf)

## Convergence rate for smooth and convex case

:::{.callout-theorem}
Let $f: \mathbb{R}^n \to \mathbb{R}$ be convex and differentiable. Let $S \subseteq  \mathbb{R}^n$d be a closed convex set, and assume that there is a minimizer $x^*$ of $f$ over $S$; furthermore, suppose that $f$ is smooth over $S$ with parameter $L$. The Projected Gradient Descent algorithm with stepsize $\frac1L$ achieves the following convergence after iteration $k > 0$:
$$
f(x_k) - f^* \leq \frac{L\|x_0 - x^*\|_2^2}{2k}
$$
:::

. . .

**Proof**

1. Let's prove sufficient decrease lemma, assuming, that $y_{k} = x_k - \frac1L\nabla f(x_k)$ and cosine rule $2x^Ty = \|x\|^2 + \|y\|^2 - \|x-y\|^2$:
    $$
    \begin{aligned}
    \uncover<+->{ &\text{Smoothness:} &f(x_{k+1})& \leq f(x_{k}) + \langle \nabla f(x_{k}), x_{k+1}-x_{k} \rangle +\frac{L}{2} \| x_{k+1}-x_{k}\|^2\\ }
    \uncover<+->{ &\text{Method:} & &= f(x_{k})-L\langle y_{k} - x_k , x_{k+1}-x_{k} \rangle +\frac{L}{2} \| x_{k+1}-x_{k}\|^2\\ }
    \uncover<+->{ &\text{Cosine rule:} & &= f(x_{k})-\frac{L}{2}\left( \|y_{k} - x_k\|^2 + \|x_{k+1}-x_{k}\|^2 - \|y_{k} - x_{k+1}\|^2\right) +\frac{L}{2} \| x_{k+1}-x_{k}\|^2\\ }
    \uncover<+->{ & & &= f(x_{k})-\frac{1}{2L}\|\nabla f(x_k)\|^2 + \frac{L}{2} \|y_{k} - x_{k+1}\|^2 \\ }
    \end{aligned}
    $$ {#eq-suff_dec}

## Convergence rate for smooth and convex case

2. Now we do not immediately have progress at each step. Let's use again cosine rule:
    $$
    \begin{aligned}
    \left\langle\frac1L \nabla f(x_k), x_k - x^* \right\rangle &=  \frac12\left(\frac{1}{L^2}\|\nabla f(x_k)\|^2 + \|x_k - x^*\|^2 -  \|x_k - x^* - \frac1L \nabla f(x_k)\|^2 \right) \\
    \langle \nabla f(x_k), x_k - x^* \rangle &=  \frac{L}{2}\left(\frac{1}{L^2}\|\nabla f(x_k)\|^2 + \|x_k - x^*\|^2 -  \|y_k - x^*\|^2 \right) \\
    \end{aligned}
    $$
3. We will use now projection property: $\|x - \text{proj}_S(y)\|^2 + \|y - \text{proj}_S(y)\|^2 \leq \|x-y\|^2$ with $x = x^*, y = y_k$:
    $$
    \begin{aligned}
    \|x^* - \text{proj}_S(y_k)\|^2 + \|y_k - \text{proj}_S(y_k)\|^2 \leq \|x^*-y_k\|^2 \\
    \|y_k - x^*\|^2 \geq \|x^* - x_{k+1}\|^2 + \|y_k - x_{k+1}\|^2
    \end{aligned}
    $$
4. Now, using convexity and previous part:
    $$
    \begin{aligned}
    &\text{Convexity:} &f(x_k) - f^* &\leq  \langle \nabla f(x_k), x_k - x^* \rangle \\
    & & &\leq  \frac{L}{2}\left(\frac{1}{L^2}\|\nabla f(x_k)\|^2 + \|x_k - x^*\|^2 -  \|x_{k+1} - x^*\|^2 - \|y_k - x_{k+1}\|^2 \right) \\
    &\text{Sum for } i=0,k-1 &\sum\limits_{i=0}^{k-1} \left[f(x_i) - f^*\right]&\leq\sum\limits_{i=0}^{k-1} \frac{1}{2L}\|\nabla f(x_i)\|^2 + \frac{L}{2}\|x_0 - x^*\|^2  - \frac{L}{2} \sum\limits_{i=0}^{i-1} \|y_i - x_{i+1}\|^2 
    \end{aligned}
    $$

## Convergence rate for smooth and convex case

5. Bound gradients with [sufficient decrease lemma @eq-suff_dec]:
    $$
    \begin{aligned}
    \sum\limits_{i=0}^{k-1} \left[f(x_i) - f^*\right]&\leq \sum\limits_{i=0}^{k-1}\left[ f(x_{i}) - f(x_{i+1}) + \frac{L}{2} \|y_{i} - x_{i+1}\|^2 \right] + \frac{L}{2}\|x_0 - x^*\|^2  - \frac{L}{2} \sum\limits_{i=0}^{i-1} \|y_i - x_{i+1}\|^2  \\
    &\leq f(x_0) - f(x_k) + \frac{L}{2} \sum\limits_{i=0}^{i-1} \|y_i - x_{i+1}\|^2 + \frac{L}{2}\|x_0 - x^*\|^2  - \frac{L}{2} \sum\limits_{i=0}^{i-1} \|y_i - x_{i+1}\|^2 \\
    &\leq f(x_0) - f(x_k) + \frac{L}{2}\|x_0 - x^*\|^2 \\ 
    \sum\limits_{i=0}^{k-1} f(x_i) - k f^* &\leq f(x_0) - f(x_k) + \frac{L}{2}\|x_0 - x^*\|^2\\ 
    \sum\limits_{i=1}^{k} \left[ f(x_i) - f^*\right] &\leq \frac{L}{2}\|x_0 - x^*\|^2\\ 
    \end{aligned}
    $$

## Convergence rate for smooth and convex case
6. Let's show monotonic decrease of the iteration of the method.

7. And finalize the convergence bound.

# Frank-Wolfe Method

## Idea

![Illustration of Frank-Wolfe (conditional gradient) algorithm](FW1.pdf)

## Idea {.noframenumbering}

![Illustration of Frank-Wolfe (conditional gradient) algorithm](FW2.pdf)

## Idea {.noframenumbering}

![Illustration of Frank-Wolfe (conditional gradient) algorithm](FW3.pdf)

## Idea {.noframenumbering}

![Illustration of Frank-Wolfe (conditional gradient) algorithm](FW4.pdf)

## Idea {.noframenumbering}

![Illustration of Frank-Wolfe (conditional gradient) algorithm](FW5.pdf)

## Idea {.noframenumbering}

![Illustration of Frank-Wolfe (conditional gradient) algorithm](FW6.pdf)

## Idea {.noframenumbering}

![Illustration of Frank-Wolfe (conditional gradient) algorithm](FW7.pdf)

## Idea

$$
\begin{split}
y_k &= \text{arg}\min_{x \in S} f^I_{x_k}(x) = \text{arg}\min_{x \in S} \langle\nabla f(x_k), x \rangle \\
x_{k+1} &= \gamma_k x_k + (1-\gamma_k)y_k
\end{split}
$$

![Illustration of Frank-Wolfe (conditional gradient) algorithm](FW.pdf)

## Convergence (1/2)

Consider the problem
$$
f(x) \rightarrow \min\limits_{x \in S},
$$
where $f$ is convex and $L$-smooth. The Frank-Wolfe method is given by:
$$
\begin{cases}
x_{k + 1} = \gamma_k x_{k} + (1 - \gamma_k)s_k \\
s_k = \arg \min\limits_{x \in S} f^{I}_{x_k}(x) = \arg \min\limits_{x \in S} \left\langle \nabla f(x_k), x \right\rangle
\end{cases},
$$
where $f^{I}_{x_k}(x)$ is the first-order Taylor approximation at the point $x_k$. For $\gamma_k = \frac{k - 1}{k + 1}$, it holds that
$$
f(x_k) - f(x^*) \leqslant \frac{2 L R^2}{k + 1},
$$
where $R = \max\limits_{x, y \in S} \|x - y\|$. Thus, we have sublinear convergence.

## Convergence (2/2)
$L$-smoothness:
$$
f(x) - f(y) - \langle \nabla f(y), x - y \rangle \leqslant \frac{L}{2} \|x - y\|^2, \quad \forall x, y \in S
$$
$$
\begin{aligned}
f\left(x_{k+1}\right) - f\left(x_k\right) &\leqslant \left\langle \nabla f\left(x_k\right), x_{k+1} - x_k \right\rangle + \frac{L}{2} \left\|x_{k+1} - x_k\right\|^2 \\
&= (1 - \gamma_k) \left\langle \nabla f\left(x_k\right), s_k - x_k \right\rangle + \frac{L (1 - \gamma_k)^2}{2} \left\|s_k - x_k\right\|^2 
\end{aligned}
$$

Convexity:
$$ f(x) - f(y) - \langle \nabla f(y), x - y \rangle \geqslant 0 \quad \forall x, y \in S \Rightarrow \quad x := x^*, y := x_k \Rightarrow \quad \langle \nabla f(x_k), x^* - x_k \rangle \leqslant f(x^*) - f(x_k) $$
$$ f\left(x_{k+1}\right) - f\left(x_k\right) \leqslant (1 - \gamma_k) \left\langle \nabla f\left(x_k\right), x^* - x_k \right\rangle + \frac{L (1 - \gamma_k)^2}{2} R^2 \leqslant (1 - \gamma_k) \left( f(x^*) - f(x_k) \right) + (1 - \gamma_k)^2 \frac{L R^2}{2} $$
$$ f\left(x_{k+1}\right) - f(x^*) \leqslant \gamma_k \left( f(x_k) - f(x^*) \right) + (1 - \gamma_k)^2 \frac{L R^2}{2} $$

Denote $\delta_k = \frac{f\left(x_k\right) - f\left(x^*\right)}{L R^2}$. Then the inequality can be rewritten as
$$
\delta_{k+1} \leqslant \gamma_k \delta_k + \frac{(1 - \gamma_k)^2}{2} = \frac{k - 1}{k + 1} \delta_k + \frac{2}{(k + 1)^2}.
$$
Starting from the inequality $\delta_2 \leqslant \frac{1}{2}$, applying induction on $k$ yields the desired result.