---
title: Automatic Differentiation
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

# Lecture reminder

# Automatic Differentiation

## Forward mode
![Illustration of forward chain rule to calculate the derivative of the function $v_i$ with respect to $w_k$.](auto_diff_forward.pdf){width=300}

* Uses the forward chain rule
* Has complexity $d \times \mathcal{O}(T)$ operations

## Reverse mode
![Illustration of reverse chain rule to calculate the derivative of the function $L$ with respect to the node $v_i$.](auto_diff_reverse.pdf){width=300}


* Uses the backward chain rule
* Stores the information from the forward pass
* Has complexity $\mathcal{O}(T)$ operations

# Automatic Differentiation Problems

## Toy example

::: {.callout-example}
$$
f(x_1, x_2) = x_1 * x_2 + \sin x_1
$$

Let's calculate the derivatives $\dfrac{\partial f}{\partial x_i}$ using forward and reverse modes.
:::

. . .

![Illustration of computation graph of $f(x_1, x_2)$.](sem_3/autograd_example.pdf){width=300}


## Automatic Differentiation with JAX



:::: {.columns}

::: {.column width="50%"}
::: {.callout-tip icon="false" title="Example №1"}
$$f(X) = tr(AX^{-1}B)$$

$$\nabla f = - X^{-T} A^T B^T X^{-T}$$

:::

. . .

:::

::: {.column width="50%"}
::: {.callout-tip icon="false" title="Example №2"}
$$g(x) = 1/3 \cdot ||x||_2^3$$

$$\nabla^2 g = ||x||_2^{-1} x x^T + ||x||_2 I_n$$
:::
:::

::::

. . .

\
\

Let's calculate the gradients and hessians of $f$ and $g$ in python [\faPython](https://colab.research.google.com/drive/14FXSFirBR7OI76p1z72n353Ve9LmwL90#scrollTo=61Ryf-1eWeZP&line=1&uniqifier=1)


## Problem 1

::: {.callout-question}
Which of the AD modes would you choose (forward/ reverse) for the following computational graph of primitive arithmetic operations?
:::

![Which mode would you choose for calculating gradients there?](ad_choose.pdf){width=175}

## Problem 2

:::: {.columns}

::: {.column width="50%"}

Suppose, we have an invertible matrix $A$ and a vector $b$, the vector $x$ is the solution of the linear system $Ax = b$, namely one can write down an analytical solution $x = A^{-1}b$.


\
\

::: {.callout-question}

Find the derivatives $\dfrac{\partial L}{\partial A}, \dfrac{\partial L}{\partial b}$.

:::

:::

::: {.column width="50%"}

![$x$ could be found as a solution of linear system](linear_least_squares_layer.pdf){width=200}


:::
::::

## Problem 3

:::: {.columns}

::: {.column width="50%"}

Suppose, we have the rectangular matrix $W \in \mathbb{R}^{m \times n}$, which has a singular value decomposition:

\
\

$$
W = U \Sigma V^T, \quad U^TU = I, \quad V^TV = I,
$$ 
$$
\Sigma = \text{diag}(\sigma_1, \ldots, \sigma_{\min(m,n)})
$$

\
\
The regularizer $R(W) = \text{tr}(\Sigma)$ in any loss function encourages low rank solutions. 

::: {.callout-question}

Find the derivative $\dfrac{\partial R}{\partial W}$.

:::
:::

::: {.column width="50%"}

![Computation graph for singular regularizer](svd_singular_regularizer_comp_graph.pdf){width=200}

:::
::::

## Computation experiment with JAX

Let's make sure numerically that we have correctly calculated the derivatives in problems 2-3 [\faPython](https://colab.research.google.com/drive/14FXSFirBR7OI76p1z72n353Ve9LmwL90#scrollTo=LlqwKMaPR0Sf) 


# Gradient checkpointing 

## Feedforward Architecture

![Computation graph for obtaining gradients for a simple feed-forward neural network with n layers. The activations marked with an $f$. The gradient of the loss with respect to the activations and parameters marked with $b$.](sem_3/backprop.pdf){width=350}

. . .

::: {.callout-important}

The results obtained for the $f$ nodes are needed to compute the $b$ nodes.

:::

## Vanilla backpropagation

![Computation graph for obtaining gradients for a simple feed-forward neural network with n layers. The purple color indicates nodes that are stored in memory.](sem_3/vanilla_backprop.pdf){width=350}

. . .

* All activations $f$ are kept in memory after the forward pass.

. . .


::: {.callout-tip icon="false" appearance="simple"}

* Optimal in terms of computation: it only computes each node once. 

:::

. . .

::: {.callout-important icon="false" appearance="simple"}

* High memory usage. The memory usage grows linearly with the number of layers in the neural network. 

:::


## Memory poor backpropagation

![Computation graph for obtaining gradients for a simple feed-forward neural network with n layers. The purple color indicates nodes that are stored in memory.](sem_3/poor_mem_backprop.pdf){width=350}

. . .

* Each activation $f$  is recalculated as needed.

. . .


::: {.callout-tip icon="false" appearance="simple"}

* Optimal in terms of memory: there is no need to store all activations in memory.

:::

. . .

::: {.callout-important icon="false" appearance="simple"}

* Computationally inefficient. The number of node evaluations scales with $n^2$, whereas it vanilla backprop scaled as $n$: each of the n nodes is recomputed on the order of $n$ times.

:::

## Checkpointed backpropagation

![Computation graph for obtaining gradients for a simple feed-forward neural network with n layers. The purple color indicates nodes that are stored in memory.](sem_3/checkpoint_backprop.pdf){width=350}

. . .

* Trade-off between the **vanilla** and **memory poor** approaches. The strategy is to mark a subset of the neural net activations as checkpoint nodes, that will be stored in memory.

. . .


::: {.callout-tip icon="false" appearance="simple"}

* Faster recalculation of activations $f$. We only need to recompute the nodes between a $b$ node and the last checkpoint preceding it when computing that $b$ node during backprop. 

:::

. . .

::: {.callout-tip icon="false" appearance="simple"}

* Memory consumption depends on the number of checkpoints. More effective then **vanilla** approach.

:::

## Gradient checkpointing visualization


The animated visualization of the above approaches [\faGithub](https://github.com/cybertronai/gradient-checkpointing)


An example of using a gradient checkpointing [\faGithub](https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/mechanics/gradient-checkpointing-nin.ipynb)