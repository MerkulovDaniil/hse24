---
title: Discover acceleration of gradient descent
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
## GD. Convergence rates

$$
\min_{x \in \mathbb{R}^n} f(x) \qquad \qquad x_{k+1} = x_k - \alpha_k \nabla f(x_k) \qquad \kappa =\dfrac{L}{\mu}
$$

| | smooth & convex | smooth & strongly convex (or PL) |
|:-----:|:-----:|:--------:|
|Upper bound | $f(x_k) - f^* \approx  \mathcal{O} \left( \dfrac{1}{k} \right)$ | $\|x_k - x^*\|^2 \approx \mathcal{O} \left( \left(\dfrac{\kappa - 1}{\kappa + 1}\right)^k \right)$ |
|Lower bound | $f(x_k) - f^* \approx  \Omega \left( \dfrac{1}{k^2} \right)$ | $\|x_k - x^*\|^2 \approx \Omega \left( \left(\dfrac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^k \right)$ |



## Three update schemes

* **Normal gradient**
  $$\boldsymbol{x}_k - \alpha_k\nabla f(\boldsymbol{x}_k)$$
  Move the point $\boldsymbol{x}_k$ in the direction $-\nabla f(\boldsymbol{x}_k)$ for $\alpha_k\Vert\nabla f(\boldsymbol{x}_k)\Vert$ amount.

. . . 

* **Polyak’s Heavy Ball Method**
  $$
  \boldsymbol{x}_k - \alpha_k\nabla f(\boldsymbol{x}_k) + \color{blue}{\beta_k(\boldsymbol{x}_k - \boldsymbol{x}_{k-1})}
  $$
  Perform a GD, $\color{blue}{\text{move the updated-}\boldsymbol{x}\text{ in the direction of the previous step for }\beta_k\Vert \boldsymbol{x}_k - \boldsymbol{x}_{k-1}\Vert\text{ amount}}$.

. . . 

* **Nesterov’s acceleration**
  $$
  \color{red}{\boldsymbol{x}_k - \alpha_k\nabla f(}\color{blue}{\boldsymbol{x}_k+\beta_k(\boldsymbol{x}_k - \boldsymbol{x}_{k-1})}\color{red}{)}\color{black}{ + }\color{orange}{\beta_k(\boldsymbol{x}_k - \boldsymbol{x}_{k-1})}
  $$
  $\color{blue}{\text{Move the not-yet-updated-}\boldsymbol{x}\text{ in the direction of the previous step for }\beta_k\Vert \boldsymbol{x}_k - \boldsymbol{x}_{k-1}\Vert \text{ amount}}$, $\color{red}\text{perform a GD on the }\color{blue}{\text{shifted-}\boldsymbol{x}}\color{black}{\text{, then }}\color{orange}{\text{move the updated-}\boldsymbol{x}}\text{ in the direction of the previous step for}$ $\color{orange}{\beta_k\Vert \boldsymbol{x}_k-\boldsymbol{x}_{k-1}\Vert}$.



## HBM for a quadratic problem

:::: {.columns}

::: {.column width="20%"}
:::{.callout-question}
Which step size strategy is used for $\color{red}{\text{GD}}$?
:::
:::

::: {.column width="80%"}
[![$\color{red}{\text{GD}}$ vs. $\color{blue}{\text{HBM with fixed }\beta}.$](HBM_exact.pdf){width="90%"}](https://angms.science/doc/teaching/GDLS.pdf)
:::

::::
**Observation:** for nice f (with spherical level sets), GD is already good enough and HBM adds a little effect. However, for bad f (with elliptic level sets), HBM is better in some cases.



## HBM for a quadratic problem

[![$\color{red}{\text{GD with }\alpha=\dfrac{1}{L}}$ vs. $\color{blue}{\text{HBM with fixed }\beta}.$](HBM_L.pdf){width="70%"}](https://angms.science/doc/teaching/GDLS.pdf)

**Observation:** same. If nice f (spherical lv. sets), GD is already good enough. If bad f (with elliptic lv. sets), HBM is better in some cases.

# NAG for DL
## NAG as a Momentum Method

* Start by setting $k=0, a_0 = 1, \boldsymbol{x}_{-1}=\boldsymbol{y}_0, \boldsymbol{y}_0$ to an arbitrary parameter setting, iterates
  $$
  \text{Gradient update } \boldsymbol{x}_k = \boldsymbol{y}_k - \alpha_k\nabla f(\boldsymbol{y}_k)
  $${#eq-grad-update}
  $$
  \text{Extrapolation weight } a_{k+1} = \dfrac{1+\sqrt{1+4a^2_k}}{2}
  $${#eq-extrap-weight}
  $$
  \text{Extrapolation } \boldsymbol{y}_{k+1} = \boldsymbol{x}_k + \dfrac{a_k - 1}{a_{k+1}}(\boldsymbol{x}_k-\boldsymbol{x}_{k+1})
  $${#eq-extrap}
  Note that here fix step-size is used: $\alpha_k = \dfrac{1}{L} \; \forall k$.
* **Theorem.** If f is $L$-smooth and convex, the sequence $\{f(\boldsymbol{x}_k)\}_k$ produced by NAG convergences to the
optimal value $f^*$ as the rate $\mathcal{O}(\dfrac{1}{k^2})$ as
  $$
  f(\boldsymbol{x}_k)-f^*\leq\dfrac{4L\Vert \boldsymbol{x}_k-\boldsymbol{x}^*\Vert^2}{(k+2)^2}
  $$
* The above representation is difficult to understand, so we will rewrite these equations in a more intuitive manner. 



## NAG as a Momentum Method

If we define
$$
\boldsymbol{v}_k \equiv \boldsymbol{x}_k - \boldsymbol{x}_{k-1}
$${#eq-v}
$$
\beta_k \equiv \dfrac{a_k-1}{a_{k+1}}
$${#eq-mu}
then the combination of @eq-extrap and @eq-mu implies:
$$
\boldsymbol{y}_k = \boldsymbol{x}_{k-1} + \beta_{k-1}\boldsymbol{v}_{k-1}
$$
which can be used to rewrite @eq-grad-update as follows using $\alpha_k = \alpha_{k-1}$:
$$
\boldsymbol{x}_k = \boldsymbol{x}_{k-1} + \beta_{k-1}\boldsymbol{v}_{k-1}-\alpha_{k-1}\nabla f(\boldsymbol{x}_{k-1} + \beta_{k-1}\boldsymbol{v}_{k-1})
$${#eq-one}
$$
\boldsymbol{v}_k = \beta_{k-1}\boldsymbol{v}_{k-1}-\alpha_{k-1}\nabla f(\boldsymbol{x}_{k-1} + \beta_{k-1}\boldsymbol{v}_{k-1})
$${#eq-two}
where @eq-two is a consequence of @eq-v. Alternatively:
$$
\color{blue}{\boldsymbol{v}_{k+1} = \beta_{k}\boldsymbol{v}_{k}-\alpha_{k}\nabla f(\boldsymbol{x}_{k}}\color{red}{+ \beta_{k}\boldsymbol{v}_{k}}\color{blue}{)}
$$
$$
\color{blue}{\boldsymbol{x}_{k+1} = \boldsymbol{x}_k + \boldsymbol{v}_{k+1}}
$$
where $\alpha_k > 0$ is the **learning rate**, $\beta_k$ is the **momentum coefficient**. Compare $\color{blue}{\text{HBM}}$ with $\color{red}{\text{NAG}}$.



## NAG for a quadratic problem

Consider the following quadratic optimization problem:
$$
\label{problem}
\min\limits_{x \in \mathbb{R}^d} q(x) =  \min\limits_{x \in \mathbb{R}^d} \dfrac{1}{2} x^\top  A x - b^\top  x, \text{ where }A \in \mathbb{S}^d_{++}.
$$
Every symmetric matrix $A$ has an eigenvalue decomposition
$$
A=Q\text{diag}\,(\lambda_1, \dots, \lambda_n)\,Q^T = Q\Lambda Q^T, \quad Q=[q_1, \dots, q_n].
$$
and, as per convention, we will assume that the $\lambda_i$'s are sorted, from smallest $\lambda_1$ to biggest $\lambda_n$. It is clear, that $\lambda_i$ correspond to the **curvature** along the associated eigenvector directions.

We can reparameterize $q(x)$ by the matrix transform $Q$ and optimize $y=Qx$ using the objective
$$
p(y) \equiv q(x) = q(Q^\top y) = y^\top Q (Q^\top \Lambda Q) Q^\top y/2 - b^\top Q^\top y = y^\top\Lambda y/2 - c^\top y,
$$
where $c=Qb$. 

We can further rewrite $p$ as 
$$
p(y) = \sum_{i=1}^n[p]_i([y]_i),
$$ 
where $[p]_i(t) = \lambda_i t^2/2-[c]_i t$.


## NAG for a quadratic problem

:::{.callout-tip title="Theorem 2.1 from [[1]](https://www.cs.toronto.edu/~gdahl/papers/momentumNesterovDeepLearning.pdf)."}
Let $p(y) = \sum_{i=1}^n[p]_i([y]_i)$ such that $[p]_i(t) = \lambda_i t^2/2-[c]_i t$. Let $\alpha$ be arbitrary and fixed. Denote by $\text{HBM}_x(\beta, p, y,v)$ and $\text{HBM}_v(\beta,p,y,v)$ the parameter vector and the velocity vector respectively, obtained by applying one step of HBM (i.e., Eq. 1 and then Eq. 2) to the function $p$ at point $y$, with velocity $v$, momentum coefficient $\beta$, and learning rate $\alpha$. Define $\text{NAG}_x$ and $\text{NAG}_v$ analogously. Then the following holds for $z \in  \{x, v\}$:
$$
\text{HBM}_z (\beta,p,y,v) = 
\left[\begin{gathered}
      \text{HBM}_z (\beta,[p]_1,[y]_1, [v]_1)\\
      \vdots\\
      \text{HBM}_z (\beta,[p]_n,[y]_n, [v]_n)\\
    \end{gathered}
\right]
$$
$$
\text{NAG}_z (\beta,p,y,v) = 
\left[\begin{gathered}
      \text{NAG}_z (\beta(1-\alpha\lambda_1),[p]_1,[y]_1, [v]_1)\\
      \vdots\\
      \text{NAG}_z (\beta(1-\alpha\lambda_n),[p]_n,[y]_n, [v]_n)\\
    \end{gathered}
\right]
$$
:::



## NAG for a quadratic problem. Proof (1/2)
**Proof:**

It's easy to show that if
$$x_{i+1} = \text{HBM}_x (\beta_i,[q]_i,[x]_i, [v]_i)$$
$$v_{i+1} = \text{HBM}_v (\beta_i,[q]_i,[x]_i, [v]_i)$$
then for $y_i=Qx_i, w_i=Qv_i$
$$y_{i+1} = \text{HBM}_x (\beta_i,[p]_i,[y]_i, [w]_i)$$
$$w_{i+1} = \text{HBM}_v (\beta_i,[p]_i,[y]_i, [w]_i)$$.
Then, consider one step of $\text{HBM}_v$:
$$\begin{aligned}
& \text{HBM}_v (\beta,p,y,v) = \beta v - \alpha\nabla p(y)\\
& =(\beta[v]_1-\alpha\nabla_{[y]_1}p(y), \dots, \beta[v]_n-\alpha\nabla_{[y]_n}p(y)) \\
& =(\beta[v]_1-\alpha\nabla{[p]_1}([y]_1), \dots, \beta[v]_n-\alpha\nabla{[p]_n}([y]_n)) \\
& = (\text{HBM}_v (\beta_1,[p]_1,[y]_1, [v]_1), \dots, \text{HBM}_v (\beta_i,[p]_i,[y]_i, [v]_i))
\end{aligned}$$
This shows that one step of $\text{HBM}_v$ on $p$ is precisely equivalent to $n$ simultaneous applications of $\text{HBM}_v$ to the one-dimensional quadratics $[p]_i$, all with the same $\beta$ and $\alpha$. Similarly, for $\text{HBM}_x$.

## NAG for a quadratic problem. Proof (2/2)

Next we show that NAG, applied to a one-dimensional quadratic with a momentum coefficient $\beta$, is equivalent to $\text{HBM}$ applied to the same quadratic and with the same learning rate, but with a momentum coefficient $\beta(1 -\alpha\lambda)$. We show this by expanding $\text{NAG}_v(\beta, [p]_i, y, v)$ (where $y$ and $v$ are scalars):
$$\begin{aligned}
\text{NAG}_v (\beta, [p]_i,y,v)
& =\beta v-\alpha\nabla [p]_i(y+\beta v) \\
& =\beta v-\alpha(\lambda_i(y+\beta v) -c_i) \\
& =\beta v-\alpha\lambda_i\beta v - \alpha(\lambda_i y-c_i) \\
& =\beta (1 - \alpha\lambda_i) v - \alpha\nabla[p]_i(y) \\
& =\text{HBM}_v (\beta(1-\alpha\lambda_i), [p]_i,y,v). \\
\end{aligned}$$
QED.

**Observations**:

* HBM and NAG become **equivalent** when $\alpha$ is small (when $\alpha\lambda \ll 1$ for every eigenvalue $\lambda$ of A), so NAG and HBM are distinct only when $\alpha$ is reasonably large.
* When $\alpha$ is relatively large, NAG uses smaller effective momentum for the high-curvature eigen-directions, which **prevents oscillations** (or divergence) and thus allows the use of a larger $\beta$ than is possible with CM for a given $\alpha$.


## NAG for DL

[![The table reports the squared errors on the problems for each combination of $\beta_{max}$ and a momentum type (NAG, CM). When $\beta_{max}$ is $0$ the choice of NAG vs CM is of no consequence so the training errors are presented in a single column. For each choice of $\beta_{max}$, the highest-performing learning rate is used. The column $\text{SGD}_{C}$ lists the results of Chapelle & Erhan (2011) who used 1.7M SGD steps and tanh networks. The column $\text{HF}^\dagger$ lists the results of HF without L2 regularization; and the column $\text{HF}^*$ lists the results of Martens (2010).](NAG_conv.pdf){width="80%"}](https://www.cs.toronto.edu/~gdahl/papers/momentumNesterovDeepLearning.pdf)


## References and Python Examples

* Figures for HBM was taken from the [presentation](https://angms.science/doc/teaching/GDLS.pdf). Visit [site](https://angms.science) for more tutorials.

* Why Momentum Really Works. [Link](https://distill.pub/2017/momentum/).

* Run code in [\faPython Colab](https://drive.google.com/file/d/1Qtnazye0_wz47Q0tRyaV0w0_EhCMouME/view?usp=sharing). The code taken from [\faGithub](https://github.com/amkatrutsa/optimization_course/blob/master/Spring2022/hb_acc_grad.ipynb).

* On the importance of initialization and momentum in deep learning. [Link](https://www.cs.toronto.edu/~gdahl/papers/momentumNesterovDeepLearning.pdf).

