---
title: "Duality"
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
            \titlegraphic{\includegraphics[width=0.5\paperwidth]{back7.png}}
---

# Introduction

---

:::: {.columns}
::: {.column width="65%"}
> The reader will find no figures in this work. The methods which I set forth do not require either constructions or geometrical or mechanical reasonings: but only algebraic operations, subject to a regular and uniform rule of procedure.

Preface to Mécanique analytique
:::

::: {.column width="35%"}
![Joseph-Louis Lagrange](lagrange.jpg)
:::

::::

---

## Motivation

Duality lets us associate to any constrained optimization problem a concave maximization problem, whose solutions lower bound the optimal value of the original problem. What is interesting is that there are cases, when one can solve the primal problem by first solving the dual one. Now, consider a general constrained optimization problem:

. . .

$$
\text{ Primal: }f(x) \to \min\limits_{x \in S}  \qquad \text{ Dual: } g(y) \to \max\limits_{y \in \Omega} 
$$


. . .


We'll build $g(y)$, that preserves the uniform bound:

$$
g(y) \leq f(x) \qquad \forall x \in S, \forall y \in \Omega
$$


. . .


As a consequence:

$$
\max\limits_{y \in \Omega} g(y) \leq \min\limits_{x \in S} f(x)  
$$

## Lagrange duality

We'll consider one of many possible ways to construct $g(y)$ in case, when we have a general mathematical programming problem with functional constraints:


. . .


$$
\begin{split}
& f_0(x) \to \min\limits_{x \in \mathbb{R}^n}\\
\text{s.t. } & f_i(x) \leq 0, \; i = 1,\ldots,m\\
& h_i(x) = 0, \; i = 1,\ldots, p
\end{split}
$$

. . .

And the Lagrangian, associated with this problem:

$$
L(x, \lambda, \nu) = f_0(x) + \sum\limits_{i=1}^m \lambda_i f_i(x) + \sum\limits_{i=1}^p\nu_i h_i(x) = f_0(x) + \lambda^\top f(x) + \nu^\top h(x)
$$

## Dual function

We assume $\mathcal{D} = \bigcap\limits_{i=0}^m\textbf{dom } f_i \cap \bigcap\limits_{i=1}^p\textbf{dom } h_i$ is nonempty. We define the Lagrange dual function (or just dual function) $g: \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$ as the minimum value of the Lagrangian over $x$: for $\lambda \in \mathbb{R}^m, \nu \in \mathbb{R}^p$

. . .

$$
g(\lambda, \nu) = \inf_{x \in \mathcal{D}} L(x, \lambda, \nu) = \inf_{x \in \mathcal{D}} \left( f_0(x) +\sum\limits_{i=1}^m \lambda_i f_i(x) + \sum\limits_{i=1}^p\nu_i h_i(x) \right)
$$

. . .


When the Lagrangian is unbounded below in $x$, the dual function takes on the value $-\infty$. Since the dual function is the pointwise infimum of a family of affine functions of $(\lambda, \nu)$, it is concave, even when the original problem is not convex.

## Dual function as a lower bound

:::: {.columns}

::: {.column width="50%"}
Let us show, that the dual function yields lower bounds on the optimal value $p^*$ of the original problem for any $\lambda \succeq 0, \nu$. Suppose some $\hat{x}$ is a feasible point for the original problem, i.e., $f_i(\hat{x}) \leq 0$ and $h_i(\hat{x}) = 0, \; \lambda \succeq 0$. Then we have:

. . .


$$
L(\hat{x}, \lambda, \nu) = f_0(\hat{x}) + \underbrace{\lambda^\top f(\hat{x})}_{\leq 0} + \underbrace{\nu^\top h(\hat{x})}_{= 0} \leq f_0(\hat{x})
$$

. . .


Hence

$$
g(\lambda, \nu) = \inf_{x \in \mathcal{D}} L(x, \lambda, \nu) \leq L(\hat{x}, \lambda, \nu)  \leq f_0(\hat{x})
$$

. . .


$$
g(\lambda, \nu) \leq p^*
$$

:::

. . .

::: {.column width="50%"}

A natural question is: what is the *best* lower bound that can be obtained from the Lagrange dual function? 
This leads to the following optimization problem:

. . .

$$
\begin{split}
& g(\lambda, \nu) \to \max\limits_{\lambda \in \mathbb{R}^m, \; \nu \in \mathbb{R}^p }\\
\text{s.t. } & \lambda \succeq 0
\end{split}
$$

. . .

The term "dual feasible", to describe a pair $(\lambda, \nu)$ with $\lambda \succeq 0$ and $g(\lambda, \nu) > -\infty$, now makes sense. It means, as the name implies, that $(\lambda, \nu)$ is feasible for the dual problem. We refer to $(\lambda^*, \nu^*)$ as dual optimal or optimal Lagrange multipliers if they are optimal for the above problem.

:::
::::

## Summary

|  | Primal | Dual |
|:--:|:--:|:--:|
| Function | $f_0(x)$ | $g(\lambda, \nu) = \min\limits_{x \in \mathcal{D}} L(x, \lambda, \nu)$ |
| | | |
| Variables | $x \in S \subseteq \mathbb{R^n}$ | $\lambda \in \mathbb{R}^m_{+}, \nu \in \mathbb{R}^p$ |
| | | |
| Constraints | $f_i(x) \leq 0$, $i = 1,\ldots,m$ $h_i(x) = 0, \; i = 1,\ldots, p$ | $\lambda_i \geq 0, \forall i \in \overline{1,m}$ |
| | | |
| Problem | $\begin{matrix}& f_0(x) \to \min\limits_{x \in \mathbb{R}^n}\\ \text{s.t. } & f_i(x) \leq 0, \; i = 1,\ldots,m\\ & h_i(x) = 0, \; i = 1,\ldots, p \end{matrix}$ | $\begin{matrix}  g(\lambda, \nu) &\to \max\limits_{\lambda \in \mathbb{R}^m, \nu \in \mathbb{R}^p }\\ \text{s.t. } & \lambda \succeq 0 \end{matrix}$ | 
| | | |
| Optimal | $\begin{matrix} &x^* \text{ if feasible},  \\ &p^* = f_0(x^*)\end{matrix}$ | $\begin{matrix} &\lambda^*, \nu^* \text{ if } \max \text{ is achieved},  \\ &d^* = g(\lambda^*, \nu^*)\end{matrix}$ |

## Example. Linear Least Squares

We are addressing a problem within a non-empty budget set, defined as follows:

. . .

$$
\begin{aligned}
    & \text{min} \quad x^T x \\
    & \text{s.t.} \quad Ax = b,
\end{aligned}
$$

with the matrix $A \in \mathbb{R}^{m \times n}$. 

. . .

This problem is devoid of inequality constraints, presenting $m$ linear equality constraints instead. The Lagrangian is expressed as $L(x, \nu) = x^T x + \nu^T (Ax - b)$, spanning the domain $\mathbb{R}^n \times \mathbb{R}^m$. The dual function is denoted by $g(\nu) = \inf_x L(x, \nu)$. Given that $L(x, \nu)$ manifests as a convex quadratic function in terms of $x$, the minimizing $x$ can be derived from the optimality condition

. . .

:::: {.columns}

::: {.column width="50%"}

$$
\nabla_x L(x, \nu) = 2x + A^T \nu = 0,
$$

. . .

leading to $x = -(1/2)A^T \nu$. As a result, the dual function is articulated as

. . .

$$
g(\nu) = L(-(1/2)A^T \nu, \nu) = -(1/4)\nu^T A A^T \nu - b^T \nu,
$$

:::

. . .

::: {.column width="50%"}

emerging as a concave quadratic function within the domain $\mathbb{R}^p$. According to the lower bound property, for any $\nu \in \mathbb{R}^p$, the following holds true:

. . .

$$
-(1/4)\nu^T A A^T \nu - b^T \nu \leq \inf\{x^T x \,|\, Ax = b\}.
$$

Which is a simple non-trivial lower bound without any problem solving.
:::
::::

## Example. Two-way partitioning problem

:::: {.columns}

::: {.column width="60%"}

We are examining a (nonconvex) problem:
$$
\begin{aligned}
    & \text{minimize} \quad x^T W x \\
    & \text{subject to} \quad x_i^2 =1, \quad i=1,\ldots,n,
\end{aligned}
$$

. . .

![Illustration of two-way partitioning problem](partition.pdf){width=100%}
:::

. . .

::: {.column width="40%"}

This problem can be construed as a two-way partitioning problem over a set of $n$ elements, denoted as $\{1, \ldots , n\}$: A viable $x$ corresponds to the partition
$$
\{1,\ldots,n\} = \{i|x_i =-1\} \cup \{i|x_i =1\}.
$$

. . .

The coefficient $W_{ij}$ in the matrix represents the expense associated with placing elements $i$ and $j$ in the same partition, while $-W_{ij}$ signifies the cost of segregating them. The objective encapsulates the aggregate cost across all pairs of elements, and the challenge posed by problem is to find the partition that minimizes the total cost.

:::

::::

## Example. Two-way partitioning problem

We now derive the dual function for this problem. The Lagrangian is expressed as
$$
L(x,\nu) = x^T W x + \sum_{i=1}^n \nu_i (x_i^2 -1) = x^T (W + \text{diag}(\nu)) x - \mathbf{1}^T \nu.
$$
. . .

By minimizing over $x$, we procure the Lagrange dual function: 
$$
g(\nu) = \inf_x x^T (W + \text{diag}(\nu)) x - \mathbf{1}^T \nu
= \begin{cases}\begin{array}{ll}
    -\mathbf{1}^T\nu & \text{if } W+\text{diag}(\nu) \succeq 0 \\
    -\infty & \text{otherwise},
\end{array} \end{cases}
$$

. . .

exploiting the realization that the infimum of a quadratic form is either zero (when the form is positive semidefinite) or $-\infty$ (when it's not).

. . .

This dual function furnishes lower bounds on the optimal value of the problem. For instance, we can adopt the particular value of the dual variable
$$
\nu = -\lambda_{\text{min}}(W) \mathbf{1}
$$

. . .

which is dual feasible, since $W +\text{diag}(\nu)=W -\lambda_{\text{min}}(W) I \succeq 0.$

. . .

This renders a simple bound on the optimal value $p^*$: $p^* \geq -\mathbf{1}^T\nu = n \lambda_{\text{min}}(W).$

. . .

The code for the problem is available here [\faPython Open in Colab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Partitioning.ipynb)

# Strong duality

## Strong duality

It is common to name this relation between optimals of primal and dual problems as **weak duality**. For problem, we have: 

$$
p^* \geq d^*
$$

. . .

While the difference between them is often called **duality gap:** 

$$
p^* - d^* \geq 0
$$

. . .

Note, that we always have weak duality, if we've formulated primal and dual problem. It means, that if we have managed to solve the dual problem (which is always concave, no matter whether the initial problem was or not), then we have some lower bound. Surprisingly, there are some notable cases, when these solutions are equal.

. . .

**Strong duality** happens if duality gap is zero: 

$$
p^* = d^*
$$

. . .

Notice: both $p^*$ and $d^*$ may be $\infty$. 

* Several sufficient conditions known!
* “Easy” necessary and sufficient conditions: unknown.

## Strong duality in linear least squares

:::{.callout-exercise}
In the Least-squares solution of linear equations example above calculate the primal optimum $p^*$ and the dual optimum $d^*$ and check whether this problem has strong duality or not.
:::

## Useful features of duality

* **Construction of lower bound on solution of the primal problem.**

	It could be very complicated to solve the initial problem. But if we have the dual problem, we can take an arbitrary $y \in \Omega$ and substitute it in $g(y)$ - we'll immediately obtain some lower bound.

* **Checking for the problem's solvability and attainability of the solution.** 

	From the inequality $\max\limits_{y \in \Omega} g(y) \leq \min\limits_{x \in S} f_0(x)$ follows: if $\min\limits_{x \in S} f_0(x) = -\infty$, then $\Omega = \varnothing$ and vice versa.

* **Sometimes it is easier to solve a dual problem than a primal one.** 

	In this case, if the strong duality holds: $g(y^*) = f_0(x^*)$ we lose nothing.

* **Obtaining a lower bound on the function's residual.** 

	$f_0(x) - f_0^* \leq f_0(x) - g(y)$ for an arbitrary $y \in \Omega$ (suboptimality certificate). Moreover, $p^* \in [g(y), f_0(x)], d^* \in [g(y), f_0(x)]$

* **Dual function is always concave**

	As a pointwise minimum of affine functions.

## Slater's condition 

:::{.callout-theorem}
If for a convex optimization problem (i.e., assuming minimization, $f_0,f_{i}$ are convex and $h_{i}$ are affine), there exists a point $x$ such that $h(x)=0$ and $f_{i}(x)<0$ (existance of a strictly feasible point), then we have a zero duality gap and KKT conditions become necessary and sufficient.
:::

## An example of convex problem, when Slater's condition does not hold

:::{.callout-example}

$$
\min \{ f_0(x) = x \mid f_1(x) = \frac{x^2}{2} \leq 0 \}, 
$$

. . .

The only point in the budget set is: $x^* = 0$. However, it is impossible to find a non-negative $\lambda^* \geq 0$, such that 
$$
\nabla f_0(0) + \lambda^* \nabla f_1(0) = 1 + \lambda^* x = 0.
$$

:::

## A nonconvex quadratic problem with strong duality

:::: {.columns}

::: {.column width="35%"}

On rare occasions strong duality obtains for a nonconvex problem. As an important example, we consider the problem of minimizing a nonconvex quadratic function over the unit ball

. . .

$$
\begin{split}
& x^\top A x  + 2b^\top x\to \min\limits_{x \in \mathbb{R}^{n} }\\
\text{s.t. } & x^\top x \leq 1 
\end{split}
$$

. . .

where $A \in \mathbb{S}^n, A \nsucceq 0$ and $b \in \mathbb{R}^n$. Since $A \nsucceq 0$, this is not a convex problem. This problem is sometimes called the trust region problem, and arises in minimizing a second-order approximation of a function over the unit ball, which is the region in which the approximation is assumed to be approximately valid.

:::

. . .

::: {.column width="65%"}

**Solution**

. . .

Lagrangian and dual function

$$
L(x, \lambda) = x^\top A x + 2 b^\top x + \lambda (x^\top x - 1) = x^\top( A + \lambda I)x + 2 b^\top x - \lambda
$$

. . .

$$
g(\lambda) = \begin{cases} -b^\top(A + \lambda I)^{\dagger}b - \lambda &\text{ if } A + \lambda I \succeq 0 \\ -\infty, &\text{ otherwise}  \end{cases}
$$

. . .

Dual problem:

$$
\begin{split}
& -b^\top(A + \lambda I)^{\dagger}b - \lambda \to \max\limits_{\lambda \in \mathbb{R}}\\
\text{s.t. } & A + \lambda I \succeq 0
\end{split}
$$

. . .

$$
\begin{split}
& -\sum\limits_{i=1}^n \dfrac{(q_i^\top b)^2}{\lambda_i + \lambda} - \lambda  \to \max\limits_{\lambda \in \mathbb{R}}\\
\text{s.t. } & \lambda \geq - \lambda_{min}(A)
\end{split}
$$
:::
::::

# Applications

## Sensitivity analysis

:::: {.columns}

::: {.column width="50%"}

Let us switch from the original optimization problem

$$
\begin{split}
& f_0(x) \to \min\limits_{x \in \mathbb{R}^n}\\
\text{s.t. } & f_i(x) \leq 0, \; i = 1,\ldots,m\\
& h_i(x) = 0, \; i = 1,\ldots, p
\end{split}
\tag{P}
$$

:::

. . .

::: {.column width="50%"}

To the perturbed version of it:

$$
\begin{split}
& f_0(x) \to \min\limits_{x \in \mathbb{R}^n}\\
\text{s.t. } & f_i(x) \leq u_i, \; i = 1,\ldots,m\\
& h_i(x) = v_i, \; i = 1,\ldots, p
\end{split}
\tag{Per}
$$

:::
::::

. . .

Note, that we still have the only variable $x \in \mathbb{R}^n$, while treating $u \in \mathbb{R}^m, v \in \mathbb{R}^p$ as parameters. It is obvious, that $\text{Per}(u,v) \to \text{P}$ if $u = 0, v = 0$. We will denote the optimal value of $\text{Per}$ as $p^*(u, v)$, while the optimal value of the original problem $\text{P}$ is just $p^*$. One can immediately say, that $p^*(u, v) = p^*$.

. . .

Speaking of the value of some $i$-th constraint we can say, that

* $u_i = 0$ leaves the original problem
* $u_i > 0$ means that we have relaxed the inequality
* $u_i < 0$ means that we have tightened the constraint

. . .

One can even show, that when $\text{P}$ is convex optimization problem, $p^*(u,v)$ is a convex function.

## Sensitivity analysis

Suppose, that strong duality holds for the orriginal problem and suppose, that $x$ is any feasible point for the perturbed problem:
$$
\begin{split}
\uncover<+->{p^*(0,0) &= p^* = d^* = g(\lambda^*, \nu^*) \leq \\}
\uncover<+->{& \leq L(x, \lambda^*, \nu^*) = \\}
\uncover<+->{& = f_0(x) + \sum\limits_{i=1}^m \lambda_i^* f_i(x) + \sum\limits_{i=1}^p\nu_i^* h_i(x) \leq \\}
\uncover<+->{& \leq f_0(x) + \sum\limits_{i=1}^m \lambda_i^* u_i + \sum\limits_{i=1}^p\nu_i^* v_i }
\end{split}
$$

. . .

Which means
$$
\begin{split}
f_0(x) \geq p^*(0,0) - {\lambda^*}^T u - {\nu^*}^T v 
\end{split}
$$

. . .

And taking the optimal $x$ for the perturbed problem, we have:
$$
p^*(u,v) \geq p^*(0,0) - {\lambda^*}^T u - {\nu^*}^T v 
$$ {#eq-sensitivity}

## Sensitivity analysis

In scenarios where strong duality holds, we can draw several insights about the sensitivity of optimal solutions in relation to the Lagrange multipliers. These insights are derived from the inequality expressed in equation above:

* **Impact of Tightening a Constraint (Large $\lambda_i^\star$)**:  
   When the $i$th constraint's Lagrange multiplier, $\lambda_i^\star$, holds a substantial value, and if this constraint is tightened (choosing $u_i < 0$), there is a guarantee that the optimal value, denoted by $p^\star(u, v)$, will significantly increase.

* **Effect of Adjusting Constraints with Large Positive or Negative $\nu_i^\star$**:  
   - If $\nu_i^\star$ is large and positive and $v_i < 0$ is chosen, or  
   - If $\nu_i^\star$ is large and negative and $v_i > 0$ is selected,  
   then in either scenario, the optimal value $p^\star(u, v)$ is expected to increase greatly.

* **Consequences of Loosening a Constraint (Small $\lambda_i^\star$)**:  
   If the Lagrange multiplier $\lambda_i^\star$ for the $i$th constraint is relatively small, and the constraint is loosened (choosing $u_i > 0$), it is anticipated that the optimal value $p^\star(u, v)$ will not significantly decrease.

* **Outcomes of Tiny Adjustments in Constraints with Small $\nu_i^\star$**:  
   - When $\nu_i^\star$ is small and positive, and $v_i > 0$ is chosen, or  
   - When $\nu_i^\star$ is small and negative, and $v_i < 0$ is opted for,  
   in both cases, the optimal value $p^\star(u, v)$ will not significantly decrease.

. . .

These interpretations provide a framework for understanding how changes in constraints, reflected through their corresponding Lagrange multipliers, impact the optimal solution in problems where strong duality holds.

## Local sensitivity 

:::: {.columns}

::: {.column width="50%"}

Suppose now that $p^*(u, v)$ is differentiable at $u = 0, v = 0$. 

. . .

$$
\lambda_i^* = -\dfrac{\partial p^*(0,0)}{\partial u_i} \quad \nu_i^* = -\dfrac{\partial p^*(0,0)}{\partial v_i}
$$ {#eq-local-sensitivity}

. . .

To show this result we consider the directional derivative of $p^*(u,v)$ along the direction of some $i$-th basis vector $e_i$:

. . .

$$
\lim_{t \to 0} \dfrac{p^*(t e_i,0) - p^*(0,0)}{t} = \dfrac{\partial p^*(0,0)}{\partial u_i}
$$

. . .

From the inequality @eq-sensitivity and taking the limit $t \to0$ with $t>0$ we have

. . .

$$
\dfrac{p^*(t e_i,0) - p^*}{t} \geq -\lambda_i^* \to  \dfrac{\partial p^*(0,0)}{\partial u_i} \geq -\lambda_i^*
$$

. . .

For the negative $t < 0$ we have:

. . .

$$
\dfrac{p^*(t e_i,0) - p^*}{t} \leq -\lambda_i^* \to  \dfrac{\partial p^*(0,0)}{\partial u_i} \leq -\lambda_i^*
$$

:::

. . .

::: {.column width="50%"}

The same idea can be used to establish the fact about $v_i$. 

. . .

The local sensitivity result @eq-local-sensitivity provides a way to understand the impact of constraints on the optimal solution $x^*$ of an optimization problem. If a constraint $f_i(x^*)$ is negative at $x^*$, it's not affecting the optimal solution, meaning small changes to this constraint won't alter the optimal value. In this case, the corresponding optimal Lagrange multiplier will be zero, as per the principle of complementary slackness. 

. . .

However, if $f_i(x^*) = 0$, meaning the constraint is precisely met at the optimum, then the situation is different. The value of the $i$-th optimal Lagrange multiplier, $\lambda^*_i$, gives us insight into how 'sensitive' or 'active' this constraint is. A small $\lambda^*_i$ indicates that slight adjustments to the constraint won't significantly affect the optimal value. Conversely, a large $\lambda^*_i$ implies that even minor changes to the constraint can have a significant impact on the optimal solution.

:::

::::

## Mixed strategies for matrix games

:::: {.columns}

::: {.column width="65%"}

![The scheme of a mixed strategy matrix game](msmg.pdf)

:::

. . .

::: {.column width="35%"}

In zero-sum matrix games, players 1 and 2 choose actions from sets $\{1,...,n\}$ and $\{1,...,m\}$, respectively. The outcome is a payment from player 1 to player 2, determined by a payoff matrix $P \in \mathbb{R}^{n \times m}$. Each player aims to use mixed strategies, choosing actions according to a probability distribution: player 1 uses probabilities $u_k$ for each action $i$, and player 2 uses $v_l$.

. . .

The expected payoff from player 1 to player 2 is given by $\sum_{k=1}^{n} \sum_{l=1}^{m} u_k v_l P_{kl} = u^T P v$. Player 1 seeks to minimize this expected payoff, while player 2 aims to maximize it.

:::
::::

## Mixed strategies for matrix games. Player 1's Perspective

:::: {.columns}

::: {.column width="30%"}
![](msmg_1.pdf)
:::
::: {.column width="70%"}
Assuming player 2 knows player 1's strategy $u$, player 2 will choose $v$ to maximize $u^T P v$. The worst-case expected payoff is thus:

$$
\max_{v \geq 0, 1^T v = 1} u^T P v = \max_{i=1,...,m} (P^T u)_i
$$

. . .

Player 1's optimal strategy minimizes this worst-case payoff, leading to the optimization problem:

$$
\begin{split}
& \min \max_{i=1,...,m} (P^T u)_i\\
& \text{s.t. } u \geq 0 \\
& 1^T u = 1
\end{split}
$$ {#eq-player1-problem}

This forms a convex optimization problem with the optimal value denoted as $p^*_1$.

:::
::::

## Mixed strategies for matrix games. Player 2's Perspective

:::: {.columns}

::: {.column width="30%"}
![](msmg_2.pdf)
:::
::: {.column width="70%"}

Conversely, if player 1 knows player 2's strategy $v$, the goal is to minimize $u^T P v$. This leads to:

$$
\min_{u \geq 0, 1^T u = 1} u^T P v = \min_{i=1,...,n} (P v)_i
$$

. . .

Player 2 then maximizes this to get the largest guaranteed payoff, solving the optimization problem:

$$
\begin{split}
& \max \min_{i=1,...,n} (P v)_i \\
& \text{s.t. }  v \geq 0 \\
& 1^T v = 1
\end{split}
$$ {#eq-player2-problem}

The optimal value here is $p^*_2$.
:::
::::

## Mixed strategies for matrix games

### Duality and Equivalence

It's generally advantageous to know the opponent's strategy, but surprisingly, in mixed strategy matrix games, this advantage disappears. The key lies in duality: the problems above are Lagrange duals. By formulating player 1's problem as a linear program and introducing Lagrange multipliers, we find that the dual problem matches player 2's problem. Due to strong duality in feasible linear programs, $p^*_1 = p^*_2$, showing no advantage in knowing the opponent’s strategy.

. . .

### Formulating and Solving the Lagrange Dual

We approach problem @eq-player1-problem by setting it up as a linear programming (LP) problem. The goal is to minimize a variable $t$, subject to certain constraints:

. . .

1. $u \geq 0$,
2. The sum of elements in $u$ equals 1 ($1^T u = 1$),
3. $P^T u$ is less than or equal to $t$ times a vector of ones ($P^T u \leq t \mathbf{1}$).

. . .

Here, $t$ is an additional variable in the real numbers ($t \in \mathbb{R}$).

. . .

### Constructing the Lagrangian

. . .

We introduce multipliers for the constraints: $\lambda$ for $P^T u \leq t \mathbf{1}$, $\mu$ for $u \geq 0$, and $\nu$ for $1^T u = 1$. The Lagrangian is then formed as:

. . .

$$
L = t + \lambda^T (P^T u - t \mathbf{1}) - \mu^T u + \nu (1 - 1^T u) = \nu + (1 - 1^T \lambda)t + (P\lambda - \nu \mathbf{1} - \mu)^T u
$$

. . .

## Mixed strategies for matrix games

:::: {.columns}

::: {.column width="70%"}
### Defining the Dual Function

. . .

The dual function $g(\lambda, \mu, \nu)$ is defined as:

. . .

$$
g(\lambda, \mu, \nu) = 
\begin{cases} 
\nu & \text{if } 1^T\lambda=1 \text{ and } P\lambda - \nu \mathbf{1} = \mu \\
-\infty & \text{otherwise} 
\end{cases}
$$

. . .

### Solving the Dual Problem

The dual problem seeks to maximize $\nu$ under the following conditions:

1. $\lambda \geq 0$,
2. The sum of elements in $\lambda$ equals 1 ($1^T \lambda = 1$),
3. $\mu \geq 0$,
4. $P\lambda - \nu \mathbf{1} = \mu$.

. . .

Upon eliminating $\mu$, we obtain the Lagrange dual of @eq-player1-problem:

. . .

$$
\begin{split}
& \max \nu \\
& \text{s.t. }   \lambda \geq 0 \\
&  1^T \lambda = 1 \\
& P\lambda \geq \nu \mathbf{1}
\end{split}
$$ 

:::

. . .

::: {.column width="30%"}

### Conclusion

This formulation shows that the Lagrange dual problem is equivalent to problem @eq-player2-problem. Given the feasibility of these linear programs, strong duality holds, meaning the optimal values of @eq-player1-problem and @eq-player2-problem are equal.

:::
::::