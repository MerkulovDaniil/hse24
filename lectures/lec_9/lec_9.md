---
title: "Gradient Descent. Convergence for quadratics; smooth convex case; PL case. Lower bounds."
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
            \titlegraphic{\includegraphics[width=0.5\paperwidth]{back9.jpeg}}
---

# Gradient Descent

## Direction of local steepest descent

:::: {.columns}
::: {.column width="40%"}

Let's consider a linear approximation of the differentiable function $f$ along some direction $h, \|h\|_2 = 1$:

. . .

$$
f(x + \alpha h) = f(x) + \alpha \langle f'(x), h \rangle + o(\alpha)
$$

. . .

We want $h$ to be a decreasing direction:

$$
f(x + \alpha h) < f(x)
$$

$$
f(x) + \alpha \langle f'(x), h \rangle + o(\alpha) < f(x)
$$

. . .

and going to the limit at $\alpha \rightarrow 0$:

$$
\langle f'(x), h \rangle \leq 0
$$

:::

. . .

::: {.column width="60%"}

Also from Cauchy–Bunyakovsky–Schwarz inequality:

$$
\begin{split}
|\langle f'(x), h \rangle | &\leq \| f'(x) \|_2 \| h \|_2 \\
\langle f'(x), h \rangle &\geq -\| f'(x) \|_2 \| h \|_2 = -\| f'(x) \|_2
\end{split}
$$

. . .

Thus, the direction of the antigradient

$$
h = -\dfrac{f'(x)}{\|f'(x)\|_2}
$$

gives the direction of the **steepest local** decreasing of the function $f$.

. . .

The result of this method is

$$
x_{k+1} = x_k - \alpha f'(x_k)
$$

:::
::::

## Gradient flow ODE

:::: {.columns}
::: {.column width="78%"}

Let's consider the following ODE, which is referred to as the Gradient Flow equation.

$$
\tag{GF}
\frac{dx}{dt} = -f'(x(t))
$$

. . .

and discretize it on a uniform grid with $\alpha$ step:

$$
\frac{x_{k+1} - x_k}{\alpha} = -f'(x_k),
$$

. . .

where $x_k \equiv x(t_k)$ and $\alpha = t_{k+1} - t_k$ - is the grid step.

From here we get the expression for $x_{k+1}$

$$
x_{k+1} = x_k - \alpha f'(x_k),
$$

which is exactly gradient descent.

[Open In Colab $\clubsuit$](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/GD_vs_GF.ipynb)
:::

. . .

::: {.column width="22%"}
![Gradient flow trajectory](GD_vs_GF.pdf)
:::
::::

## Convergence of Gradient Descent algorithm

Heavily depends on the choice of the learning rate $\alpha$:

[![](gd_2d.png)](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/GD_2d_visualization.ipynb)

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
\nabla f(x_{k+1})^\top \nabla f(x_k) = 0
$$
:::
::: {.column width="20%"}

![Steepest Descent](GD_vs_Steepest.pdf)

[Open In Colab $\clubsuit$](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Steepest_descent.ipynb)
:::
::::

# Strongly convex quadratics

## Coordinate shift

:::: {.columns}

::: {.column width="70%"}

Consider the following quadratic optimization problem:

$$
\label{problem}
\min\limits_{x \in \mathbb{R}^d} f(x) =  \min\limits_{x \in \mathbb{R}^d} \dfrac{1}{2} x^\top  A x - b^\top  x + c, \text{ where }A \in \mathbb{S}^d_{++}.
$$

. . .

* Firstly, without loss of generality we can set $c = 0$, which will or affect optimization process.
* Secondly, we have a spectral decomposition of the matrix $A$: 
    $$
    A = Q \Lambda Q^T
    $$
* Let's show, that we can switch coordinates to make an analysis a little bit easier. Let $\hat{x} = Q^T(x - x^*)$, where $x^*$ is the minimum point of initial function, defined by $Ax^* = b$. At the same time $x = Q\hat{x} + x^*$.
    $$
    \begin{split}
    \uncover<+->{ f(\hat{x}) &= \frac12  (Q\hat{x} + x^*)^\top  A (Q\hat{x} + x^*) - b^\top  (Q\hat{x} + x^*) \\}
    \uncover<+->{ &= \frac12 \hat{x}^T Q^TAQ\hat{x} + (x^*)^TAQ\hat{x} + \frac12 (x^*)^T A (x^*)^T - b^T Q\hat{x} - b^T x^*\\}
    \uncover<+->{ &=  \frac12 \hat{x}^T \Lambda \hat{x}}
    \end{split}
    $$

:::
::: {.column width="30%"}
![](coordinate_shift.pdf)
:::
::::

## Convergence analysis

Now we can work with the function $f(x) = \frac12 x^T \Lambda x$ with $x^* = 0$ without loss of generality (drop the hat from the $\hat{x}$)

:::: {.columns}
::: {.column width="50%"}
$$
\begin{split}
\uncover<+->{x^{k+1} &= x^k - \alpha^k \nabla f(x^k)} 
\uncover<+->{= x^k - \alpha^k \Lambda x^k \\ } 
\uncover<+->{&= (I - \alpha^k \Lambda) x^k \\ }
\uncover<+->{ x^{k+1}_{(i)} &= (1 - \alpha^k \lambda_{(i)}) x^k_{(i)} \text{ For $i$-th coordinate} \\ }
\uncover<+->{  x^{k+1}_{(i)} &= (1 - \alpha^k \lambda_{(i)})^k x^0_{(i)}}
\end{split}
$$
\uncover<+->{
Let's use constant stepsize $\alpha^k = \alpha$. Convergence condition:
$$
\rho(\alpha) = \max_{i} |1 - \alpha \lambda_{(i)}| < 1
$$
Remember, that $\lambda_{\text{min}} = \mu > 0, \lambda_{\text{max}} = L \geq \mu$.}

:::: {.columns}
::: {.column width="50%"}
$$
\begin{split}
\uncover<+->{ |1 - \alpha \mu| &< 1 \\ }
\uncover<+->{ -1 < 1 &- \alpha \mu < 1 \\ }
\uncover<+->{ \alpha < \frac{2}{\mu} \quad & \quad \alpha\mu > 0}
\end{split}
$$
:::
::: {.column width="50%"}
$$
\begin{split}
\uncover<+->{ |1 - \alpha L| &< 1 \\ }
\uncover<+->{ -1 < 1 &- \alpha L < 1 \\ }
\uncover<+->{ \alpha < \frac{2}{L} \quad & \quad \alpha L > 0}
\end{split}
$$
:::
::::

. . .

$\alpha < \frac{2}{L}$ is needed for convergence.

:::

. . .

::: {.column width="50%"}
Now we would like to tune $\alpha$ to choose the best (lowest) convergence rate

$$
\begin{split}
\uncover<+->{ \rho^* &=  \min_{\alpha} \rho(\alpha) } \uncover<+->{  = \min_{\alpha} \max_{i} |1 - \alpha \lambda_{(i)}| \\ }
\uncover<+->{ &=  \min_{\alpha} \left\{|1 - \alpha \mu|, |1 - \alpha L| \right\} \\ }
\uncover<+->{ \alpha^* &: \quad  1 - \alpha^* \mu = \alpha^* L - 1 \\ }
\uncover<+->{ & \alpha^* = \frac{2}{\mu + L} } \uncover<+->{ \quad \rho^* = \frac{L - \mu}{L + \mu} \\ }
\uncover<+->{ x^{k+1} &= \left( \frac{L - \mu}{L + \mu} \right)^k x^0 } \uncover<+->{ \quad f(x^{k+1}) = \left( \frac{L - \mu}{L + \mu} \right)^{2k} f(x^0)}
\end{split}
$$
:::
::::

## Convergence analysis

So, we have a linear convergence in the domain with rate $\frac{\kappa - 1}{\kappa + 1} = 1 - \frac{2}{\kappa + 1}$, where $\kappa = \frac{L}{\mu}$ is sometimes called *condition number* of the quadratic problem.

| $\kappa$ | $\rho$ | Iterations to decrease domain gap $10$ times | Iterations to decrease function gap $10$ times |
|:-:|:-:|:-----------:|:-----------:|
| $1.1$ | $0.05$ | $1$ | $1$ |
| $2$ | $0.33$ | $3$ | $2$ |
| $5$ | $0.67$ | $6$ | $3$ |
| $10$ | $0.82$ | $12$ | $6$ |
| $50$ | $0.96$ | $58$ | $29$ |
| $100$ | $0.98$ | $116$ | $58$ |
| $500$ | $0.996$ | $576$ | $288$ |
| $1000$ | $0.998$ | $1152$ | $576$ |

# Polyak-Lojasiewicz smooth case

## Polyak-Lojasiewicz condition. Linear convergence of gradient descent without convexity

PL inequality holds if the following condition is satisfied for some $\mu > 0$,
$$
\Vert \nabla f(x) \Vert^2 \geq 2 \mu (f(x) - f^*) \quad \forall x
$$
It is interesting, that the Gradient Descent algorithm might converge linearly even without convexity.

The following functions satisfy the PL condition but are not convex. [\faPython Link to the code](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/PL_function.ipynb)

:::: {.columns}

::: {.column width="50%"}

$$
f(x) = x^2 + 3\sin^2(x)
$$

![PL function](pl_2d.pdf){width=65%}

:::

. . .

::: {.column width="50%"}

$$
f(x,y) = \dfrac{(y - \sin x)^2}{2}
$$

![PL function](pl_3d.pdf){width=80%}

:::
::::

## Convergence analysis

:::{.callout-theorem}
Consider the Problem 

$$
f(x) \to \min_{x \in \mathbb{R}^d}
$$

and assume that $f$ is $\mu$-Polyak-Lojasiewicz and $L$-smooth, for some $L\geq \mu >0$.

Consider $(x^k)_{k \in \mathbb{N}}$ a sequence generated by the gradient descent constant stepsize algorithm, with a stepsize satisfying $0<\alpha \leq \frac{1}{L}$. Then:

$$
f(x^{k})-f^* \leq (1-\alpha \mu)^k (f(x^0)-f^*).
$$
:::

## Convergence analysis

We can use $L$-smoothness, together with the update rule of the algorithm, to write
$$
\begin{split}
\uncover<+->{ f(x^{k+1})& \leq f(x^{k}) + \langle \nabla f(x^{k}), x^{k+1}-x^{k} \rangle +\frac{L}{2} \| x^{k+1}-x^{k}\|^2\\ }
\uncover<+->{ &= f(x^{k})-\alpha\Vert \nabla f(x^{k}) \Vert^2 +\frac{L \alpha^2}{2} \| \nabla f(x^{k})\|^2 \\ }
\uncover<+->{ &= f(x^{k}) - \frac{\alpha}{2} \left(2 - L \alpha \right)\Vert \nabla f(x^{k}) \Vert^2 \\ }
\uncover<+->{ & \leq f(x^{k}) - \frac{\alpha}{2}\Vert \nabla f(x^{k})\Vert^2,}
\end{split}
$$

. . .

where in the last inequality we used our hypothesis on the stepsize that $\alpha L \leq 1$.

. . .

We can now use the Polyak-Lojasiewicz property to write:

$$
f(x^{k+1}) \leq f(x^{k}) - \alpha \mu (f(x^{k}) - f^*).
$$

The conclusion follows after subtracting $f^*$ on both sides of this inequality and using recursion.

## Any $\mu$-strongly convex differentiable function is a PL-function

:::{.callout-theorem}
If a function $f(x)$ is differentiable and $\mu$-strongly convex, then it is a PL function.
:::

**Proof**

:::: {.columns}

::: {.column width="60%"}

By first order strong convexity criterion:
$$
f(y) \geq f(x) + \nabla f(x)^T(y-x) + \dfrac{\mu}{2}\|y-x\|_2^2
$$
Putting $y = x^*$:
$$
\begin{split}
\uncover<+->{ f(x^*) &\geq f(x) + \nabla f(x)^T(x^*-x) + \dfrac{\mu}{2}\|x^*-x\|_2^2 \\ }
\uncover<+->{ f(x) - f(x^*) &\leq \nabla f(x)^T(x-x^*) - \dfrac{\mu}{2}\|x^*-x\|_2^2 = \\ }
\uncover<+->{ &= \left(\nabla f(x)^T - \dfrac{\mu}{2}(x^*-x)\right)^T (x-x^*) = \\ }
\uncover<+->{ &= \frac12 \left(\frac{2}{\sqrt{\mu}}\nabla f(x)^T - \sqrt{\mu}(x^*-x)\right)^T \sqrt{\mu}(x-x^*) = \\ }
\end{split}
$$
:::

. . .

::: {.column width="40%"}

Let $a = \frac{1}{\sqrt{\mu}}\nabla f(x)$ and $b =\sqrt{\mu}(x-x^*) -\frac{1}{\sqrt{\mu}}\nabla f(x)$ 

. . .

Then $a+b = \sqrt{\mu}(x-x^*)$ and $a-b=\frac{2}{\sqrt{\mu}}\nabla f(x)-\sqrt{\mu}(x-x^*)$
:::
::::

## Any $\mu$-strongly convex differentiable function is a PL-function

$$
\begin{split}
\uncover<+->{ f(x) - f(x^*) &\leq \frac12 \left(\frac{1}{\mu}\|\nabla f(x)\|^2_2 - \left\|\sqrt{\mu}(x-x^*) -\frac{1}{\sqrt{\mu}}\nabla f(x)\right\|_2^2\right) \\ }
\uncover<+->{ f(x) - f(x^*) &\leq \frac{1}{2\mu}\|\nabla f(x)\|^2_2, \\ }
\end{split}
$$

. . .

which is exactly the PL condition. It means, that we already have linear convergence proof for any strongly convex function.

# Smooth convex case

## Smooth convex case

:::{.callout-theorem}
Consider the Problem 

$$
f(x) \to \min_{x \in \mathbb{R}^d}
$$

and assume that $f$ is convex and $L$-smooth, for some $L>0$.

Let $(x^{k})_{k \in \mathbb{N}}$ be the sequence of iterates generated by the gradient descent constant stepsize algorithm, with a stepsize satisfying $0 < \alpha\leq \frac{1}{L}$. Then, for all $x^* \in {\rm{argmin}}~f$, for all $k \in \mathbb{N}$ we have that

$$
f(x^{k})-f^* \leq \frac{\|x^0-x^*\| ^2}{2 \alpha k}.
$$
:::

## Convergence analysis

* As it was before, we first use smoothness:
    $$
    \begin{split}
    f(x^{k+1})& \leq f(x^{k}) + \langle \nabla f(x^{k}), x^{k+1}-x^{k} \rangle +\frac{L}{2} \| x^{k+1}-x^{k}\|^2\\
    &= f(x^{k})-\alpha\Vert \nabla f(x^{k}) \Vert^2 +\frac{L \alpha^2}{2} \| \nabla f(x^{k})\|^2 \\
    &= f(x^{k}) - \frac{\alpha}{2} \left(2 - L \alpha \right)\Vert \nabla f(x^{k}) \Vert^2 \\
    & \leq f(x^{k}) - \frac{\alpha}{2}\Vert \nabla f(x^{k})\Vert^2, \\
    f(x^{k}) - f(x^{k+1}) & \geq \dfrac{1}{2L} \Vert \nabla f(x^{k})\Vert^2 \text{ if } \alpha \leq \frac1L
    \end{split}
    $$ {#eq-gd-cs-smoothness}
    Typically, for the convergent gradient descent algorithm the higher the learning rate the faster the convergence. That is why we often will use $\alpha = \frac1L$.
 * After that we add convexity:
    $$
    \begin{split}
    \uncover<+->{ f(y) &\geq f(x) + \langle \nabla f(x), y-x\rangle} \uncover<+->{ \text{ with } y = x^*, x = x^k} \\
    \uncover<+->{f(x^k) - f^* &\leq \langle \nabla f(x^k), x^k-x^*\rangle }
    \end{split}
    $$ {#eq-gd-cs-convexity}

## Convergence analysis

* Now we put @eq-gd-cs-convexity to @eq-gd-cs-smoothness:
    $$
    \begin{split}
    \uncover<+->{ f(x^{k+1}) &\leq f(x^{k}) -\frac{\alpha}{2} \Vert \nabla f(x^{k})\Vert^2 \leq f^* + \langle \nabla f(x^k), x^k-x^*\rangle - \frac{\alpha}{2} \Vert \nabla f(x^{k})\Vert^2 \\ }
    \uncover<+->{ &= f^* + \langle \nabla f(x^k), x^k-x^* - \frac{\alpha}{2} \nabla f(x^{k})\rangle \\ }
    \uncover<+->{ &= f^* + \frac{1}{2 \alpha}\left\langle \alpha \nabla f(x^k), 2\left(x^k-x^* - \frac{\alpha}{2} \nabla f(x^{k})\right)\right\rangle }
    \end{split}
    $$
    \uncover<+->{ Let $a = x^k-x^*$ and $b =x^k-x^* - \alpha\nabla f(x^k)$.} \uncover<+->{Then $a+b = \alpha \nabla f(x^k)$ and $a-b=2\left(x^k-x^* - \frac{\alpha}{2} \nabla f(x^{k})\right)$.}
    $$
    \begin{split}
    \uncover<+->{ f(x^{k+1}) &\leq f^* + \frac{1}{2 \alpha}\left[ \|x^k-x^*\|_2^2 - \|x^k-x^* - \alpha\nabla f(x^k)\|_2^2\right] \\ }
    \uncover<+->{ &\leq f^* + \frac{1}{2 \alpha}\left[ \|x^k-x^*\|_2^2 - \|x^{k+1}-x^*\|_2^2\right] \\ }
    \uncover<+->{ 2\alpha \left(f(x^{k+1}) - f^*\right) &\leq \|x^k-x^*\|_2^2 - \|x^{k+1}-x^*\|_2^2 }
    \end{split}
    $$
* Now suppose, that the last line is defined for some index $i$ and we sum over $i \in [0, k-1]$. Almost all summands will vanish due to the telescopic nature of the sum:
    $$
    \begin{split}
    \uncover<+->{2\alpha \sum\limits_{i=0}^{k-1} \left(f(x^{i+1}) - f^*\right) &\leq \|x^0-x^*\|_2^2 - \|x^{k}-x^*\|_2^2} \uncover<+->{ \leq \|x^0-x^*\|_2^2 }
    \end{split}
    $$ {#eq-gd-sc-telescopic}

## Convergence analysis

* Due to the monotonic decrease at each iteration $f(x^{i+1}) < f(x^i)$:
    $$
    kf(x^k) \leq \sum\limits_{i=0}^{k-1}f(x^{i+1})
    $$
* Now putting it to @eq-gd-sc-telescopic:
    $$
    \begin{split}
    \uncover<+->{ 2\alpha kf(x^k) - 2\alpha kf^* &\leq 2\alpha \sum\limits_{i=0}^{k-1} \left(f(x^{i+1}) - f^*\right)  \leq \|x^0-x^*\|_2^2 \\ }
    \uncover<+->{ f(x^k) - f^* &\leq \frac{\|x^0-x^*\|_2^2}{2 \alpha k} } \uncover<+->{ \leq  \frac{L \|x^0-x^*\|_2^2}{2 k} }
    \end{split}
    $$

# Lower bounds

## How optimal is $\mathcal{O}\left(\frac1k\right)$?

* Is it somehow possible to understand, that the obtained convergence is the fastest possible with this class of problem and this class of algorithms?
* The iteration of gradient descent:
    $$
    \begin{aligned}
    x^{k+1} &= x^k - \alpha^k \nabla f(x^k)\\
    &= x^{k-1} - \alpha^{k-1} \nabla f(x^{k-1}) - \alpha^k \nabla f(x^k) \\
    & \;\;\vdots \\
    &= x^0 - \sum\limits_{i=0}^k \alpha^{k-i} \nabla f(x^{k-i})
    \end{aligned}
    $$
* Consider a family of first-order methods, where
    $$
    x^{k+1} \in x^0 + \text{span} \left\{\nabla f(x^{0}), \nabla f(x^{1}), \ldots, \nabla f(x^{k})\right\}
    $$ {#eq-fom}

## Smooth convex case

:::{.callout-theorem}
There exists a function $f$ that is $L$-smooth and convex such that any [method @eq-fom] satisfies
$$
\min_{i \in [1, k]} f(x^i) - f^* \geq \frac{3L \|x^0 - x^*\|_2^2}{32(1+k)^2}
$$
:::

. . .

* No matter what gradient method you provide, there is always a function $f$ that, when you apply your gradient method on minimizing such $f$, the convergence rate is lower bounded as $\mathcal{O}\left(\frac{1}{k^2}\right)$.
* The key to the proof is to explicitly build a special function $f$.

## Nesterov’s worst function

:::: {.columns}

::: {.column width="50%"}
* Let $d=2k+1$ and $A \in \mathbb{R}^{d \times d}$.
    $$
    \begin{bmatrix}
        2 & -1 & 0 & 0 & \cdots & 0 \\
        -1 & 2 & -1 & 0 & \cdots & 0 \\
        0 & -1 & 2 & -1  & \cdots & 0 \\
        0 & 0 & -1 & 2  & \cdots & 0 \\
        \vdots & \vdots & \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & 0 & 0 & \cdots & 2  \\
    \end{bmatrix}
    $$
* Notice, that
    $$
    x^T A x = x[1]^2 + x[d]^2 + \sum_{i=1}^{d-1} (x[i] - x[i+1])^2,
    $$
    and, from this expression, it's simple to check $0 \preceq A \preceq 4I$.
* Define the following $L$-smooth convex function
    $$
    f(x) = \frac{L}{8} x^T A x - \frac{L}{4}\langle x, e_1 \rangle.
    $$
:::
::: {.column width="50%"}
* The optimal solution $x^*$ satisfies $Ax^* = e_1$, and solving this system of equations gives
    $$
    x^*[i] = 1 - \frac{i}{d+1},
    $$
* And the objective value is
    $$
    \begin{split}
    f(x^*) &=  \frac{L}{8} {x^*}^T A x^* - \frac{L}{4}\langle x^*, e_1 \rangle \\
    &= -\frac{L}{8} \langle x^*, e_1 \rangle = -\frac{L}{8} \left(1 - \frac{1}{d+1}\right).
    \end{split}
    $$
:::
::::