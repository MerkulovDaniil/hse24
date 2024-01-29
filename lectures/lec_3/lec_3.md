---
title: Automatic differentiation.
author: Daniil Merkulov
institute: Optimization for ML. Faculty of Computer Science. HSE University
bibliography: ../../files/biblio.bib
csl: ../../files/diabetologia.csl
format: 
    beamer:
        pdf-engine: pdflatex
        aspectratio: 169
        fontsize: 9pt
        section-titles: false
        incremental: true
        include-in-header: ../../files/header.tex  # Custom LaTeX commands and preamble
        header-includes: |
            \titlegraphic{\includegraphics[width=0.5\paperwidth]{back3.png}}
---

# Automatic differentiation

## {.plain}
![When you got the idea](autograd_expectations.jpeg)

## {.plain}
![This is not autograd](avtograd.jpeg){width=65%}

## Problem

Suppose we need to solve the following problem:

$$
L(w) \to \min_{w \in \mathbb{R}^d}
$$

. . .

* Such problems typically arise in machine learning, when you need to find optimal hyperparameters $w$ of an ML model (i.e. train a neural network). 
* You may use a lot of algorithms to approach this problem, but given the modern size of the problem, where $d$ could be dozens of billions it is very challenging to solve this problem without information about the gradients using zero-order optimization algorithms. 
* That is why it would be beneficial to be able to calculate the gradient vector $\nabla_w L = \left( \frac{\partial L}{\partial w_1}, \ldots, \frac{\partial L}{\partial w_d}\right)^T$. 
* Typically, first-order methods perform much better in huge-scale optimization, while second-order methods require too much memory.

## Example: multidimensional scaling

Suppose, we have a pairwise distance matrix for $N$ $d$-dimensional objects $D \in \mathbb{R}^{N \times N}$. Given this matrix, our goal is to recover the initial coordinates $W_i \in \mathbb{R}^d, \; i = 1, \ldots, N$.

. . .

$$
L(W) = \sum_{i, j = 1}^N \left(\|W_i - W_j\|^2_2 - D_{i,j}\right)^2 \to \min_{W \in \mathbb{R}^{N \times d}}
$$

. . .

Link to a nice visualization [$\clubsuit$](http://www.benfrederickson.com/numerical-optimization/), where one can see, that gradient free methods handle this problem much slower, especially in higher dimensions.

:::{.callout-question}
Is it somehow connected with PCA?
:::

## Example: Gradient Descent without gradient

:::: {.columns}
::: {.column width="50%"}
Suppose we need to solve the following problem:

$$
L(w) \to \min_{w \in \mathbb{R}^d}
$$

. . .

with the Gradient Descent (GD) algorithm:

$$
w_{k+1} = w_k - \alpha_k \nabla_w L(w_k)
$$

. . .

Is it possible to replace $\nabla_w L(w_k)$ using only zero order information? 

. . .

Yes, but at a cost.

. . .

One can consider 2-point gradient estimator[^1] $G$:

$$
G = d\dfrac{L(w + \varepsilon v)- L(w - \varepsilon v)}{2 \varepsilon}v, 
$$

where $v$ is spherically symmetric.
:::

. . .

::: {.column width="50%"}
!["Illustration of two-point estimator of Gradient Descent"](zgd_2p.pdf)
:::

::::

[^1]: I suggest a [nice](https://scholar.harvard.edu/files/yujietang/files/slides_2019_zero-order_opt_tutorial.pdf) presentation about gradient-free methods


## Example: Gradient Descent without gradient

:::: {.columns}
::: {.column width="50%"}

$$
w_{k+1} = w_k - \alpha_k G
$$

. . .
 
One can also consider the idea of finite differences:

$$
G =  \sum\limits_{i=1}^d\dfrac{L(w+\varepsilon e_i) - L(w-\varepsilon e_i)}{2\varepsilon} e_i
$$

[Open In Colab $\clubsuit$](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Zero_order_GD.ipynb)

:::

::: {.column width="50%"}
!["Illustration of finite differences estimator of Gradient Descent"](zgd_fd.pdf)
:::

::::

## The curse of dimensionality for zero-order methods

$$
\min_{x \in \mathbb{R}^n} f(x)
$$

. . .

$$
\text{GD: } x_{k+1} = x_k - \alpha_k \nabla f(x_k) \qquad \qquad \text{Zero order GD: } x_{k+1} = x_k - \alpha_k G,
$$

where $G$ is 2-point or multi-point estimator of the gradient.

. . .

|  | $f(x)$ - smooth | $f(x)$ - smooth and convex | $f(x)$ - smooth and strongly convex |
|:-:|:---:|:---:|:------:|
| GD | $\|\nabla f(x_k)\|^2 \approx \mathcal{O} \left( \dfrac{1}{k} \right)$ | $f(x_k) - f^* \approx  \mathcal{O} \left( \dfrac{1}{k} \right)$ | $\|x_k - x^*\|^2 \approx \mathcal{O} \left( \left(1 - \dfrac{\mu}{L}\right)^k \right)$ |
| Zero order GD | $\|\nabla f(x_k)\|^2 \approx \mathcal{O} \left( \dfrac{n}{k} \right)$ | $f(x_k) - f^* \approx  \mathcal{O} \left( \dfrac{n}{k} \right)$ | $\|x_k - x^*\|^2 \approx \mathcal{O} \left( \left(1 - \dfrac{\mu}{n L}\right)^k \right)$ |


## Finite differences

The naive approach to get approximate values of gradients is **Finite differences** approach. For each coordinate, one can calculate the partial derivative approximation:

$$
\dfrac{\partial L}{\partial w_k} (w) \approx \dfrac{L(w+\varepsilon e_k) - L(w)}{\varepsilon}, \quad e_k = (0, \ldots, \underset{{\tiny k}}{1}, \ldots, 0)
$$

. . .

:::{.callout-question}
If the time needed for one calculation of $L(w)$ is $T$, what is the time needed for calculating $\nabla_w L$ with this approach?

. . .

**Answer** $2dT$, which is extremely long for the huge scale optimization. Moreover, this exact scheme is unstable, which means that you will have to choose between accuracy and stability.

. . .

**Theorem**

There is an algorithm to compute $\nabla_w L$ in $\mathcal{O}(T)$ operations. [^2]

:::

[^2]: Linnainmaa S. The representation of the cumulative rounding error of an algorithm as a Taylor expansion of the local rounding errors.  Master’s Thesis (in Finnish), Univ. Helsinki, 1970.

## Forward mode automatic differentiation

To dive deep into the idea of automatic differentiation we will consider a simple function for calculating derivatives: 

$$
L(w_1, w_2) = w_2 \log w_1 + \sqrt{w_2 \log w_1}
$$

. . .

Let's draw a *computational graph* of this function:

![Illustration of computation graph of primitive arithmetic operations for the function $L(w_1, w_2)$](comp_graph.pdf)

. . .

Let's go from the beginning of the graph to the end and calculate the derivative $\dfrac{\partial L}{\partial w_1}$.

## Forward mode automatic differentiation{.noframenumbering}

![Illustration of forward mode automatic differentiation](comp_graph1.pdf)

:::: {.columns}

::: {.column width="50%"}
### Function 

$w_1 = w_1, w_2 = w_2$
:::

. . .

::: {.column width="50%"}
### Derivative

$\dfrac{\partial w_1}{\partial w_1} = 1, \dfrac{\partial w_2}{\partial w_1} = 0$ 
:::

::::




## Forward mode automatic differentiation{.noframenumbering}

![Illustration of forward mode automatic differentiation](comp_graph2.pdf)

. . .


:::: {.columns}

::: {.column width="50%"}
### Function 

$v_1 = \log w_1$ 
:::

. . .


::: {.column width="50%"}
### Derivative

$\frac{\partial v_1}{\partial w_1} = \frac{\partial v_1}{\partial w_1} \frac{\partial w_1}{\partial w_1} = \frac{1}{w_1} 1$
:::

::::

## Forward mode automatic differentiation{.noframenumbering}

![Illustration of forward mode automatic differentiation](comp_graph3.pdf)

. . .


:::: {.columns}

::: {.column width="50%"}
### Function 

$v_2 = w_2 v_1$
:::

. . .


::: {.column width="50%"}
### Derivative

$\frac{\partial v_2}{\partial w_1} = \frac{\partial v_2}{\partial v_1}\frac{\partial v_1}{\partial w_1} + \frac{\partial v_2}{\partial w_2}\frac{\partial w_2}{\partial w_1} = w_2\frac{\partial v_1}{\partial w_1} + v_1\frac{\partial w_2}{\partial w_1}$
:::

::::

## Forward mode automatic differentiation{.noframenumbering}

![Illustration of forward mode automatic differentiation](comp_graph4.pdf)

. . .


:::: {.columns}

::: {.column width="50%"}
### Function 

$v_3 = \sqrt{v_2}$
:::

. . .


::: {.column width="50%"}
### Derivative

$\frac{\partial v_3}{\partial w_1} = \frac{\partial v_3}{\partial v_2}\frac{\partial v_2}{\partial w_1} = \frac{1}{2\sqrt{v_2}}\frac{\partial v_2}{\partial w_1}$
:::

::::

## Forward mode automatic differentiation{.noframenumbering}

![Illustration of forward mode automatic differentiation](comp_graph5.pdf)

. . .


:::: {.columns}

::: {.column width="50%"}
### Function 

$L = v_2 + v_3$ 
:::

. . .


::: {.column width="50%"}
### Derivative

$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial v_2}\frac{\partial v_2}{\partial w_1} + \frac{\partial L}{\partial v_3}\frac{\partial v_3}{\partial w_1} = 1\frac{\partial v_2}{\partial w_1} + 1\frac{\partial v_3}{\partial w_1}$
:::

::::

## Make the similar computations for $\dfrac{\partial L}{\partial w_2}$

![Illustration of computation graph of primitive arithmetic operations for the function $L(w_1, w_2)$](comp_graph.pdf)


## Forward mode automatic differentiation example {.noframenumbering}

![Illustration of forward mode automatic differentiation](cgraph_ex_1.pdf)

:::: {.columns}

::: {.column width="50%"}
### Function 

$w_1 = w_1, w_2 = w_2$
:::

::: {.column width="50%"}
### Derivative

$\dfrac{\partial w_1}{\partial w_2} = 0, \dfrac{\partial w_2}{\partial w_2} = 1$
:::

::::

## Forward mode automatic differentiation example {.noframenumbering}

![Illustration of forward mode automatic differentiation](cgraph_ex_2.pdf)

:::: {.columns}

::: {.column width="50%"}
### Function 

$v_1 = \log w_1$
:::

::: {.column width="50%"}
### Derivative

$\frac{\partial v_1}{\partial w_2} = \frac{\partial v_1}{\partial w_2} \frac{\partial w_2}{\partial w_2}= 0 \cdot 1$
:::

::::

## Forward mode automatic differentiation example {.noframenumbering}

![Illustration of forward mode automatic differentiation](cgraph_ex_3.pdf)

:::: {.columns}

::: {.column width="50%"}
### Function 

$v_2 = w_2 v_1$
:::

::: {.column width="50%"}
### Derivative

$\frac{\partial v_2}{\partial w_2} = \frac{\partial v_2}{\partial v_1}\frac{\partial v_1}{\partial w_2} + \frac{\partial v_2}{\partial w_2}\frac{\partial w_2}{\partial w_2} = w_2\frac{\partial v_1}{\partial w_2} + v_1\frac{\partial w_2}{\partial w_2}$ 
:::

::::

## Forward mode automatic differentiation example {.noframenumbering}

![Illustration of forward mode automatic differentiation](cgraph_ex_4.pdf)

:::: {.columns}

::: {.column width="50%"}
### Function 

$v_3 = \sqrt{v_2}$
:::

::: {.column width="50%"}
### Derivative

$\frac{\partial v_3}{\partial w_2} = \frac{\partial v_3}{\partial v_2}\frac{\partial v_2}{\partial w_2} = \frac{1}{2\sqrt{v_2}}\frac{\partial v_2}{\partial w_2}$
:::

::::

## Forward mode automatic differentiation example {.noframenumbering}

![Illustration of forward mode automatic differentiation](cgraph_ex_5.pdf)

:::: {.columns}

::: {.column width="50%"}
### Function 

$L = v_2 + v_3$
:::

::: {.column width="50%"}
### Derivative

$\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial v_2}\frac{\partial v_2}{\partial w_2} + \frac{\partial L}{\partial v_3}\frac{\partial v_3}{\partial w_2} = 1\frac{\partial v_2}{\partial w_2} + 1\frac{\partial v_3}{\partial w_2}$
:::

::::

## Forward mode automatic differentiation algorithm


:::: {.columns}

::: {.column width="50%"}

Suppose, we have a computational graph $v_i, i \in [1; N]$. Our goal is to calculate the derivative of the output of this graph with respect to some input variable $w_k$, i.e. $\dfrac{\partial v_N}{\partial w_k}$. This idea implies propagation of the gradient with respect to the input variable from start to end, that is why we can introduce the notation: 

. . .

$$
\overline{v_i} = \dfrac{\partial v_i}{\partial w_k}
$$

![Illustration of forward chain rule to calculate the derivative of the function $L$ with respect to $w_k$.](auto_diff_forward.pdf){width=80%}

:::

. . .

::: {.column width="50%"}

* For $i = 1, \ldots, N$:
    * Compute $v_i$ as a function of its parents (inputs) $x_1, \ldots, x_{t_i}$:
        $$
        v_i = v_i(x_1, \ldots, x_{t_i})
        $$
    * Compute the derivative $\overline{v_i}$ using the forward chain rule:
        $$
        \overline{v_i} = \sum_{j = 1}^{t_i}\dfrac{\partial v_i}{\partial x_j}\dfrac{\partial x_j}{\partial w_k}
        $$

. . .

Note, that this approach does not require storing all intermediate computations, but one can see, that for calculating the derivative $\dfrac{\partial L}{\partial w_k}$ we need $\mathcal{O}(T)$ operations. This means, that for the whole gradient, we need $d\mathcal{O}(T)$ operations, which is the same as for finite differences, but we do not have stability issues, or inaccuracies now (the formulas above are exact).

:::

::::

## {.plain}
![](yoda.jpg)


## Backward mode automatic differentiation

We will consider the same function with a computational graph:

![Illustration of computation graph of primitive arithmetic operations for the function $L(w_1, w_2)$](comp_graph.pdf)

. . .


Assume, that we have some values of the parameters $w_1, w_2$ and we have already performed a forward pass (i.e. single propagation through the computational graph from left to right). Suppose, also, that we somehow saved all intermediate values of $v_i$. Let's go from the end of the graph to the beginning and calculate the derivatives $\dfrac{\partial L}{\partial w_1}, \dfrac{\partial L}{\partial w_2}$:

## Backward mode automatic differentiation example {.noframenumbering}

![Illustration of backward mode automatic differentiation](revad1.pdf)

. . .

### Derivatives

. . .

$$
\dfrac{\partial L}{\partial L} = 1
$$

## Backward mode automatic differentiation example {.noframenumbering}

![Illustration of backward mode automatic differentiation](revad2.pdf)

. . .

### Derivatives

. . .

$$
\begin{aligned}\frac{\partial L}{\partial v_3} &= \frac{\partial L}{\partial L} \frac{\partial L}{\partial v_3}\\ &= \frac{\partial L}{\partial L} 1\end{aligned}
$$ 

## Backward mode automatic differentiation example {.noframenumbering}

![Illustration of backward mode automatic differentiation](revad3.pdf)

. . .

### Derivatives

. . .

$$
\begin{aligned}\frac{\partial L}{\partial v_2} &= \frac{\partial L}{\partial v_3}\frac{\partial v_3}{\partial v_2} + \frac{\partial L}{\partial L}\frac{\partial L}{\partial v_2} \\&= \frac{\partial L}{\partial v_3}\frac{1}{2\sqrt{v_2}} +  \frac{\partial L}{\partial L}1\end{aligned}
$$

## Backward mode automatic differentiation example {.noframenumbering}

![Illustration of backward mode automatic differentiation](revad4.pdf)

. . .

### Derivatives

. . .

$$
\begin{aligned}\frac{\partial L}{\partial v_1} &=\frac{\partial L}{\partial v_2}\frac{\partial v_2}{\partial v_1} \\ &= \frac{\partial L}{\partial v_2}w_2\end{aligned}
$$

## Backward mode automatic differentiation example {.noframenumbering}

![Illustration of backward mode automatic differentiation](revad5.pdf)

. . .

### Derivatives

. . .

$$
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial v_1}\frac{\partial v_1}{\partial w_1} = \frac{\partial L}{\partial v_1}\frac{1}{w_1} \qquad \qquad \frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial v_2}\frac{\partial v_2}{\partial w_2} = \frac{\partial L}{\partial v_1}v_1
$$


## Backward (reverse) mode automatic differentiation

:::{.callout-question}
Note, that for the same price of computations as it was in the forward mode we have the full vector of gradient $\nabla_w L$. Is it a free lunch? What is the cost of acceleration?

. . .

**Answer** Note, that for using the reverse mode AD you need to store all intermediate computations from the forward pass. This problem could be somehow mitigated with the gradient checkpointing approach, which involves necessary recomputations of some intermediate values. This could significantly reduce the memory footprint of the large machine-learning model.
:::


## Reverse mode automatic differentiation algorithm


:::: {.columns}

::: {.column width="50%"}

Suppose, we have a computational graph $v_i, i \in [1; N]$. Our goal is to calculate the derivative of the output of this graph with respect to all inputs variable $w$, i.e. $\nabla_w v_N =  \left( \frac{\partial v_N}{\partial w_1}, \ldots, \frac{\partial v_N}{\partial w_d}\right)^T$. This idea implies propagation of the gradient of the function with respect to the intermediate variables from the end to the origin, that is why we can introduce the notation: 

$$
\overline{v_i}  = \dfrac{\partial L}{\partial v_i} = \dfrac{\partial v_N}{\partial v_i}
$$

![Illustration of reverse chain rule to calculate the derivative of the function $L$ with respect to the node $v_i$.](auto_diff_reverse.pdf){width=60%}

:::

::: {.column width="50%"}

* **FORWARD PASS** 
    
    For $i = 1, \ldots, N$:

    * Compute and store the values of $v_i$ as a function of its parents (inputs) 

* **BACKWARD PASS**
    
    For $i = N, \ldots, 1$:

    * Compute the derivative $\overline{v_i}$ using the backward chain rule and information from all of its children (outputs) ($x_1, \ldots, x_{t_i}$):
        $$
        \overline{v_i} = \dfrac{\partial L}{\partial v_i} = \sum_{j = 1}^{t_i} \dfrac{\partial L}{\partial x_j} \dfrac{\partial x_j}{\partial v_i}
        $$


:::

::::


## Choose your fighter


:::: {.columns}

::: {.column width="40%"}

![Which mode would you choose for calculating gradients there?](ad_choose.pdf)
:::

::: {.column width="60%"}
:::{.callout-question}
Which of the AD modes would you choose (forward/ reverse) for the following computational graph of primitive arithmetic operations? Suppose, you are needed to compute the jacobian $J = \left\{ \dfrac{\partial L_i}{\partial w_j} \right\}_{i,j}$
:::

. . .

**Answer** Note, that the reverse mode computational time is proportional to the number of outputs here, while the forward mode works proportionally to the number of inputs there. This is why it would be a good idea to consider the forward mode AD. 

:::



::::

## Choose your fighter

![ [$\clubsuit$](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Autograd_and_Jax.ipynb) This graph nicely illustrates the idea of choice between the modes. The $n = 100$ dimension is fixed and the graph presents the time needed for Jacobian calculation w.r.t. $x$ for $f(x) = Ax$](forward_vs_reverse_ad.pdf){width=88%}

## Choose your fighter

:::: {.columns}

::: {.column width="40%"}

![Which mode would you choose for calculating gradients there?](ad_mixed.pdf)

:::

::: {.column width="60%"}
:::{.callout-question}
Which of the AD modes would you choose (forward/ reverse) for the following computational graph of primitive arithmetic operations? Suppose, you are needed to compute the jacobian $J = \left\{ \dfrac{\partial L_i}{\partial w_j} \right\}_{i,j}$. Note, that $G$ is an arbitrary computational graph
:::
. . .

**Answer** It is generally impossible to say it without some knowledge about the specific structure of the graph $G$. Note, that there are also plenty of advanced approaches to mix forward and reverse mode AD, based on the specific $G$ structure.

:::


::::

## Feedforward Architecture

:::: {.columns}

::: {.column width="40%"}

**FORWARD**

* $v_0 = x$ typically we have a batch of data $x$ here as an input.
* For $k = 1, \ldots, t-1, t$: 
    
    * $v_k = \sigma(v_{k-1}w_k)$. Note, that practically speaking the data has dimension $x  \in \mathbb{R}^{b \times d}$, where $b$ is the batch size (for the single data point $b=1$). While the weight matrix $w_k$ of a $k$ layer has a shape $n_{k-1} \times n_k$, where $n_k$ is the dimension of an inner representation of the data. 

* $L = L(v_t)$ - calculate the loss function.

**BACKWARD**

* $v_{t+1} = L, \dfrac{\partial L}{\partial L} = 1$
* For $k = t, t-1, \ldots, 1$: 
    
    * $\underset{b \times n_k}{\dfrac{\partial L}{\partial v_k}} = \underset{b \times n_{k+1}}{\dfrac{\partial L}{\partial v_{k+1}}} \underset{n_{k+1} \times n_k}{\dfrac{\partial v_{k+1}}{\partial v_{k}}}$
    * $\underset{b \times n_{k-1} \cdot n_k}{\dfrac{\partial L}{\partial w_k}} = \underset{b \times n_{k+1}}{\dfrac{\partial L}{\partial v_{k+1}}} \cdot  \underset{n_{k+1} \times n_{k-1} \cdot n_k}{\dfrac{\partial v_{k+1}}{\partial w_{k}}}$


:::

::: {.column width="60%"}

![Feedforward neural network architecture](feedforward.pdf)

:::

::::

## Gradient propagation through the linear least squares

:::: {.columns}

::: {.column width="40%"}

![$x$ could be found as a solution of linear system](linear_least_squares_layer.pdf)

:::

::: {.column width="60%"}

Suppose, we have an invertible matrix $A$ and a vector $b$, the vector $x$ is the solution of the linear system $Ax = b$, namely one can write down an analytical solution $x = A^{-1}b$, in this example we will show, that computing all derivatives $\dfrac{\partial L}{\partial A}, \dfrac{\partial L}{\partial b}, \dfrac{\partial L}{\partial x}$, i.e. the backward pass, costs approximately the same as the forward pass.

. . .

It is known, that the differential of the function does not depend on the parametrization:

$$
dL = \left\langle\dfrac{\partial L}{\partial x}, dx \right\rangle = \left\langle\dfrac{\partial L}{\partial A}, dA \right\rangle + \left\langle\dfrac{\partial L}{\partial b}, db \right\rangle
$$

. . .

Given the linear system, we have:

$$
\begin{split}
Ax &= b \\
dAx + Adx = db &\to dx = A^{-1}(db - dAx)
\end{split}
$$

:::

::::

## Gradient propagation through the linear least squares

:::: {.columns}

::: {.column width="40%"}

![$x$ could be found as a solution of linear system](linear_least_squares_layer.pdf)

:::

::: {.column width="60%"}

The straightforward substitution gives us:

$$
\left\langle\dfrac{\partial L}{\partial x}, A^{-1}(db - dAx) \right\rangle = \left\langle\dfrac{\partial L}{\partial A}, dA \right\rangle + \left\langle\dfrac{\partial L}{\partial b}, db \right\rangle
$$

. . .

$$
\left\langle -A^{-T}\dfrac{\partial L}{\partial x} x^T, dA \right\rangle + \left\langle A^{-T}\dfrac{\partial L}{\partial x},db \right\rangle = \left\langle\dfrac{\partial L}{\partial A}, dA \right\rangle + \left\langle\dfrac{\partial L}{\partial b}, db \right\rangle
$$

. . .

Therefore:

$$
\dfrac{\partial L}{\partial A} = -A^{-T}\dfrac{\partial L}{\partial x} x^T \quad \dfrac{\partial L}{\partial b} =  A^{-T}\dfrac{\partial L}{\partial x}
$$

. . .

It is interesting, that the most computationally intensive part here is the matrix inverse, which is the same as for the forward pass. Sometimes it is even possible to store the result itself, which makes the backward pass even cheaper.

:::

::::

## Gradient propagation through the SVD

:::: {.columns}

::: {.column width="30%"}

![](svd_singular_regularizer_comp_graph.pdf)

:::

::: {.column width="70%"}

Suppose, we have the rectangular matrix $W \in \mathbb{R}^{m \times n}$, which has a singular value decomposition:

$$
W = U \Sigma V^T, \quad U^TU = I, \quad V^TV = I, \quad \Sigma = \text{diag}(\sigma_1, \ldots, \sigma_{\min(m,n)})
$$

1. Similarly to the previous example:

    $$
    \begin{split}
    W &= U \Sigma V^T \\
    dW &= dU \Sigma V^T + U d\Sigma V^T + U \Sigma dV^T \\
    U^T dW V &= U^TdU \Sigma V^TV + U^TU d\Sigma V^TV + U^TU \Sigma dV^TV \\
    U^T dW V &= U^TdU \Sigma + d\Sigma + \Sigma dV^TV
    \end{split}
    $$

:::

::::

## Gradient propagation through the SVD

:::: {.columns}

::: {.column width="30%"}

![](svd_singular_regularizer_comp_graph.pdf)

:::

::: {.column width="70%"}

2. Note, that $U^T U = I \to dU^TU + U^T dU = 0$. But also $dU^TU = (U^T dU)^T$, which actually involves, that the matrix $U^TdU$ is antisymmetric:

    $$
    (U^T dU)^T +  U^T dU = 0 \quad \to \quad \text{diag}( U^T dU) = (0, \ldots, 0)
    $$

    The same logic could be applied to the matrix $V$ and

    $$
    \text{diag}(dV^T V) = (0, \ldots, 0)
    $$

3. At the same time, the matrix $d \Sigma$ is diagonal, which means (look at the 1.) that

    $$
    \text{diag}(U^T dW V) = d \Sigma 
    $$

    Here on both sides, we have diagonal matrices.

:::

::::

## Gradient propagation through the SVD

:::: {.columns}

::: {.column width="30%"}

![](svd_singular_regularizer_comp_graph.pdf)

:::

::: {.column width="70%"}

4. Now, we can decompose the differential of the loss function as a function of $\Sigma$ - such problems arise in ML problems, where we need to restrict the matrix rank:

    $$
    \begin{split}
    dL &= \left\langle\dfrac{\partial L}{\partial \Sigma}, d\Sigma \right\rangle \\
    &= \left\langle\dfrac{\partial L}{\partial \Sigma}, \text{diag}(U^T dW V)\right\rangle \\
    &= \text{tr}\left(\dfrac{\partial L}{\partial \Sigma}^T \text{diag}(U^T dW V) \right)
    \end{split}
    $$

:::

::::

## Gradient propagation through the SVD

:::: {.columns}

::: {.column width="30%"}

![](svd_singular_regularizer_comp_graph.pdf)

:::

::: {.column width="70%"}

5. As soon as we have diagonal matrices inside the product, the trace of the diagonal part of the matrix will be equal to the trace of the whole matrix:

    $$
    \begin{split}
    dL &= \text{tr}\left(\dfrac{\partial L}{\partial \Sigma}^T \text{diag}(U^T dW V) \right) \\
    &= \text{tr}\left(\dfrac{\partial L}{\partial \Sigma}^T U^T dW V \right)  \\
    &= \left\langle\dfrac{\partial L}{\partial \Sigma}, U^T dW V \right\rangle \\
    &= \left\langle U \dfrac{\partial L}{\partial \Sigma} V^T, dW \right\rangle 
    \end{split}
    $$

:::

::::

## Gradient propagation through the SVD

:::: {.columns}

::: {.column width="30%"}

![](svd_singular_regularizer_comp_graph.pdf)

:::

::: {.column width="70%"}

6. Finally, using another parametrization of the differential

    $$
    \left\langle U \dfrac{\partial L}{\partial \Sigma} V^T, dW \right\rangle = \left\langle\dfrac{\partial L}{\partial W}, dW \right\rangle
    $$

    $$
    \dfrac{\partial L}{\partial W} =  U \dfrac{\partial L}{\partial \Sigma} V^T,
    $$

    This nice result allows us to connect the gradients $\dfrac{\partial L}{\partial W}$ and $\dfrac{\partial L}{\partial \Sigma}$.
:::

::::

## Hessian vector product without the hessian

When you need some information about the curvature of the function you usually need to work with the hessian. However, when the dimension of the problem is large it is challenging. For a scalar-valued function $f : \mathbb{R}^n \to \mathbb{R}$, the Hessian at a point $x \in \mathbb{R}^n$ is written as $\nabla^2 f(x)$. A Hessian-vector product function is then able to evaluate
$$
v \mapsto \nabla^2 f(x) \cdot v
$$
for any vector $v \in \mathbb{R}^n$. We have to use the identity
$$
\nabla^2 f (x) v = \nabla [x \mapsto \nabla f(x) \cdot v] = \nabla g(x),
$$
where $g(x) = \nabla f(x)^T \cdot v$ is a new vector-valued function that dots the gradient of $f$ at $x$ with the vector $v$.

```python
import jax.numpy as jnp

def hvp(f, x, v):
    return grad(lambda x: jnp.vdot(grad(f)(x), v))(x)
```

## Hutchinson Trace Estimation [@article_hut]

This example illustrates the estimation the Hessian trace of a neural network using Hutchinson’s method, which is an algorithm to obtain such an estimate from matrix-vector products:

Let $X \in \mathbb{R}^{d \times d}$  and $v \in \mathbb{R}^d$ be a random vector such that $\mathbb{E}[vv^T] = I$. Then,

$$
\mathrm{Tr}(X) = \mathbb{E}[v^TXv] = \frac{1}{V}\sum_{i=1}^{V}v_i^TXv_i.
$$

![[Source](https://docs.backpack.pt/en/master/use_cases/example_trace_estimation.html)](https://docs.backpack.pt/en/master/_images/sphx_glr_example_trace_estimation_001.png){width=55%}

## Activation checkpointing

The animated visualization of the above approaches [\faGithub](https://github.com/cybertronai/gradient-checkpointing)

An example of using a gradient checkpointing [\faGithub](https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/gradient-checkpointing-nin.ipynb)


## What automatic differentiation (AD) is NOT:

:::: {.columns}

::: {.column width="40%"}

* AD is not a finite differences
* AD is not a symbolic derivative
* AD is not just the chain rule
* AD is not just backpropagation
* AD (reverse mode) is time-efficient and numerically stable
* AD (reverse mode) is memory inefficient (you need to store all intermediate computations from the forward pass). 

:::

::: {.column width="60%"}

![Different approaches for taking derivatives](differentiation_scheme.pdf)

:::

::::

## Code

[Open In Colab $\clubsuit$](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Autograd_and_Jax.ipynb)