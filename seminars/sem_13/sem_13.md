---
title: Conditional gradient methods. Projected Gradient Descent. Frank-Wolfe Method. Mirror Descent Algorithm Idea.
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

# Lecture recap. Projection
## Projection

The **distance** $d$ from point $\mathbf{y} \in \mathbb{R}^n$ to closed set $S \subset \mathbb{R}^n$:
$$
d(\mathbf{y}, S, \| \cdot \|) = \inf\{\|x - y\| \mid x \in S \}
$$
We will focus on **Euclidean projection** (other options are possible) of a point $\mathbf{y} \in \mathbb{R}^n$ on set $S \subseteq \mathbb{R}^n$ is a point $\text{proj}_S(\mathbf{y}) \in S$: 
$$
\text{proj}_S(\mathbf{y}) = \frac12 \underset{\mathbf{x} \in S}{\operatorname{argmin}} \|x - y\|_2^2
$$

* **Sufficient conditions of existence of a projection**. If $S \subseteq \mathbb{R}^n$ - closed set, then the projection on set $S$ exists for any point.
* **Sufficient conditions of uniqueness of a projection**. If $S \subseteq \mathbb{R}^n$ - closed convex set, then the projection on set $S$ is unique for any point.
* If a set is open, and a point is beyond this set, then its projection on this set does not exist.
* If a point is in set, then its projection is the point itself.

## Projection

:::: {.columns}

::: {.column width="65%"}

:::{.callout-tip title="Bourbaki-Cheney-Goldstein inequality theorem"}
\small
Let $S \subseteq \mathbb{R}^n$ be closed and convex, $\forall x \in S, y \in \mathbb{R}^n$. Then
$$
\langle y - \text{proj}_S(y), \mathbf{x} - \text{proj}_S(y)\rangle \leq 0
$$ {#eq-proj1}
$$
\|x - \text{proj}_S(y)\|^2 + \|y - \text{proj}_S(y)\|^2 \leq \|x-y\|^2
$$ {#eq-proj2}
:::

::: {.callout-tip title="Non-expansive function"}
A function $f$ is called **non-expansive** if $f$ is $L$-Lipschitz with $L \leq 1$ ^[Non-expansive becomes contractive if $L < 1$.]. That is, for any two points $x,y \in \text{dom} f$,
    $$
    \|f(x)-f(y)\| \leq L\|x-y\|, \text{ where } L \leq 1.
    $$
    It means the distance between the mapped points is possibly smaller than that of the unmapped points.
:::

:::

::: {.column width="35%"}
![Obtuse or straight angle should be for any point $x \in S$](proj_crit.pdf)
:::

::::

## Problems

::: {.callout-question}
Is projection operator non-expansive?
:::

::: {.callout-question}
Find projection $\text{proj}_S(\mathbf{y})$ onto $S$, where $S$:

* $l_2$-ball with center 0 and radius 1:
$$
S = \{x\in\mathbb{R}^d \vert \;\Vert x \Vert^2_2=\sum_{i=1}^d x_i^2 \leq 1\}
$$
* $\mathbb{R}^d$-cube:
$$
S = \{x\in\mathbb{R}^d \vert \;a_i\leq x_i \leq b_i\}
$$
* Affine constraints:
$$
S = \{x\in\mathbb{R}^d \vert \;Ax=b\}
$$
:::

# Lecture recap. PGD
## Projected Gradient Descent (PGD). Idea

$$
x_{k+1} = \text{proj}_S\left(x_k - \alpha_k \nabla f(x_k) \right)  \qquad \Leftrightarrow \qquad \begin{aligned}
y_k &= x_k - \alpha_k \nabla f(x_k) \\
x_{k+1} &= \text{proj}_S\left( y_k\right)
\end{aligned}
$$

![Illustration of Projected Gradient Descent algorithm](PGD.pdf)

# Lecture recap. Frank-Wolfe Method
## Frank-Wolfe Method (FWM). Idea

![Illustration of Frank-Wolfe (conditional gradient) algorithm](FW1.pdf)

## Frank-Wolfe Method (FWM). Idea {.noframenumbering}

![Illustration of Frank-Wolfe (conditional gradient) algorithm](FW2.pdf)

## Frank-Wolfe Method (FWM). Idea {.noframenumbering}

![Illustration of Frank-Wolfe (conditional gradient) algorithm](FW3.pdf)

## Frank-Wolfe Method (FWM). Idea {.noframenumbering}

![Illustration of Frank-Wolfe (conditional gradient) algorithm](FW4.pdf)

## Frank-Wolfe Method (FWM). Idea {.noframenumbering}

![Illustration of Frank-Wolfe (conditional gradient) algorithm](FW5.pdf)

## Frank-Wolfe Method (FWM). Idea {.noframenumbering}

![Illustration of Frank-Wolfe (conditional gradient) algorithm](FW6.pdf)

## Frank-Wolfe Method (FWM). Idea {.noframenumbering}

![Illustration of Frank-Wolfe (conditional gradient) algorithm](FW7.pdf)

## Frank-Wolfe Method (FWM). Idea

$$
\begin{split}
y_k &= \text{arg}\min_{x \in S} f^I_{x_k}(x) = \text{arg}\min_{x \in S} \langle\nabla f(x_k), x \rangle \\
x_{k+1} &= \gamma_k x_k + (1-\gamma_k)y_k
\end{split}
$$

![Illustration of Frank-Wolfe (conditional gradient) algorithm](FW.pdf)

# Convergence rates
## Convergence rate for smooth and convex case

:::{.callout-theorem}
Let $f: \mathbb{R}^n \to \mathbb{R}$ be convex and differentiable. Let $S \subseteq  \mathbb{R}^n$d be a closed convex set, and assume that there is a minimizer $x^*$ of $f$ over $S$; furthermore, suppose that $f$ is smooth over $S$ with parameter $L$. 

* The **Projected Gradient Descent** algorithm with stepsize $\frac1L$ achieves the following convergence after iteration $k > 0$:
$$
f(x_k) - f^* \leq \frac{L\|x_0 - x^*\|_2^2}{2k}
$$

* The **Frank-Wolfe Method** achieves the following convergence after iteration $k > 0$:
$$
f(x_k) - f^* \leq \frac{2L\|x_0 - x^*\|_2^2}{k+1}
$$

:::

:::{.callout-tip title="FWM specificity"}

* FWM convergence rate for the $\mu$-strongly convex functions is $\mathcal{O} \left( \dfrac{1}{k} \right)$

* FWM doesn't work for non-smooth functions. But modifications do.
  
* FWM works for any norm.

:::

# Mirror Descent
## Subgradient method: linear approximation + proximity

Recall SubGD step with sub-gradient $g_k$:
$$
x_{k+1} = x_k - \alpha_k g_k  \qquad \Leftrightarrow \qquad \begin{aligned}
x_{k+1} &= \underset{x}{\operatorname{argmin}} \; \underbrace{f(x_k) + g_k^\top(x-x_k)}_{\text{linear approximation to f}}+\underbrace{\dfrac{1}{2\alpha} \|x - x_k\|_2^2}_{\text{proximity term}} \\
 &= \underset{x}{\operatorname{argmin}} \; \alpha g_k^\top x+\dfrac{1}{2} \|x - x_k\|_2^2
\end{aligned}
$$

![$\Vert\cdot\Vert_1$ is not spherical symmetrical](EuclideanGeometry.pdf){width=160}

## Example. Poor condition

Consider $f(x_1, x_2)=x_1^2\cdot\dfrac{1}{100}+x_2^2\cdot 100$.

![Poorly conditioned problem in $\Vert\cdot\Vert_2$ norm](BadCondition.pdf)

## Example. Poor condition
Suppose we are at the point: $x_k=(-10 \quad -0.1)^\top$.
SubGD method: $x_{k+1} = x_k - \alpha\nabla f(x_k)$
$$
\nabla f(x_k) = (\dfrac{2x_1}{100} \quad 2x_2\cdot 100)^\top\Big\vert_{(-10 \; -0.1)^\top}=\left(-\dfrac{1}{5} \quad -20\right)^\top
$$

**The problem:** due to elongation of the level sets the direction of movement $(x_{k+1}-x_k)$ is $\sim\perp$ $(x^*-x_k)$.

**The solution:** Change proximity term
$$
x_{k+1} = \underset{x}{\operatorname{argmin}} \; \underbrace{f(x_k) + g_k^\top(x-x_k)}_{\text{linear approximation to f}}+\underbrace{\dfrac{1}{2\alpha} (x - x_k)^\top \textcolor{orange}{I} (x-x_k)}_{\text{proximity term}}
$$
to another
$$
x_{k+1} = \underset{x}{\operatorname{argmin}} \; \underbrace{f(x_k) + g_k^\top(x-x_k)}_{\text{linear approximation to f}}+\underbrace{\dfrac{1}{2\alpha} (x - x_k)^\top \textcolor{orange}{Q} (x-x_k)}_{\text{proximity term}},
$$
where $Q=\begin{pmatrix}
    \frac{1}{50} & 0\\
    0 & 200
\end{pmatrix}$ for this example. And more generally to another function $\color{orange}{B_\phi(x, y)}$ that measures proximity.

## Example. Poor condition
Let's find $x_{k+1}$ for this **new** algorithm
$$
\alpha\nabla f(x_k) + \begin{pmatrix}
    \frac{1}{50} & 0\\
    0 & 200
\end{pmatrix}(x-x_k) = 0.
$$
Solving for $x$, we get
$$
x_{k+1} = x_k - \alpha \begin{pmatrix}
    50 & 0\\
    0 & \frac{1}{200}
\end{pmatrix} \nabla f(x_k) = (-10 \; -0.1)^\top - \alpha (-10 \; -0.1)^\top
$$
**Observation:** Changing the proximity term, we **change the direction** $x_{k+1}-x_k$. In other words, if we measure distance using this **new** way, we also **change Lipschitzness**.

::: {.callout-question}
What is the Lipshitz constant of $f$ at the point $(1 \; 1)^\top$ for the norm:
$$
\Vert z \Vert^2 = z^\top \begin{pmatrix}
    50 & 0\\
    0 & \frac{1}{200}
\end{pmatrix} z?
$$
:::

## Example. Robust Regression

Square loss $\Vert Ax-b\Vert^2_2$ is very sensitive to outliers.

**Instead:** $\min \Vert Ax-b\Vert_1$. This problem also **convex**.

Let's compute $L$-Lipshitz constant for $f(x)=\Vert Ax-b\Vert_1$:
$$
\vert \Vert Ax-b\Vert_1 - \Vert Ay-b\Vert_1 \vert \leq L \Vert x-y \Vert_2.
$$
To simplify calculation: $A=I$, $b=0$, i.e. $f(x)=\Vert x \Vert_1$.

If we take $x=\bold{1}_d$, $y=(1+\varepsilon)\bold{1}_d$:
$$
\vert n - (1+\varepsilon)n\vert = \varepsilon n \leq L \Vert x - y \Vert_2=\Vert -\varepsilon \Vert_2=\sqrt(n\varepsilon^2)=\varepsilon\sqrt{n}.
$$
Finally, we get $\color{orange}{L=\sqrt{n}}$. As we can see, $L$ is **dimension dependent**.


::: {.callout-question}
Show that if $\Vert\nabla f(x)\Vert_\infty \leq 1$, then $\Vert\nabla f(x)\Vert_2 \leq \sqrt{d}$.
:::

# References
## References

Examples for the Mirror Descent was taken from the [\faYoutube](https://www.youtube.com/watch?v=m_SJafYedbQ&list=PLXsmhnDvpjORzPelSDs0LSDrfJcqyLlZc&index=21) Lecture.