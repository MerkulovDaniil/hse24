---
title: "Stories from modern Machine Learning from the optimization perspective"
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
  - \newcommand{\bgimage}{../../files/back18.jpeg}
---

# General introduction

## Optimization for Neural Network training

Neural network is a function, that takes an input $x$ and current set of weights (parameters) $\mathbf{w}$ and predicts some vector as an output. Note, that a variety of feed-forward neural networks could be represented as a series of linear transformations, followed by some nonlinear function (say, $\text{ReLU }(x)$ or sigmoid):

. . .

$$
\mathcal{NN}(\mathbf{w}, x) = \sigma_L \circ w_L \circ \ldots \circ \sigma_1 \circ w_1 \circ x \qquad \mathbf{w} = \left(W_1, b_1, \ldots W_L, b_L\right),
$$ 

. . .

where $L$ is the number of layers, $\sigma_i$ - non-linear activation function, $w_i = W_i x + b_i$ - linear layer.

. . .

Typically, we aim to find $\mathbf{w}$ in order to solve some problem (let say to be $\mathcal{NN}(\mathbf{w}, x_i) \sim y_i$ for some training data $x_i, y_i$). In order to do it, we solve the optimization problem:

. . .

$$
L(\mathbf{w}, X, y) \to \min_{\mathbf{w}} \qquad \dfrac{1}{N} \sum\limits_{i=1}^N l(\mathbf{w}, x_i, y_i) \to \min_{\mathbf{w}}
$$

## Loss functions

In the context of training neural networks, the loss function, denoted by $l(\mathbf{w}, x_i, y_i)$, measures the discrepancy between the predicted output $\mathcal{NN}(\mathbf{w}, x_i)$ and the true output $y_i$. The choice of the loss function can significantly influence the training process. Common loss functions include:

### Mean Squared Error (MSE)

Used primarily for regression tasks. It computes the square of the difference between predicted and true values, averaged over all samples.
$$
\text{MSE}(\mathbf{w}, X, y) = \frac{1}{N} \sum_{i=1}^N (\mathcal{NN}(\mathbf{w}, x_i) - y_i)^2
$$

### Cross-Entropy Loss

Typically used for classification tasks. It measures the dissimilarity between the true label distribution and the predictions, providing a probabilistic interpretation of classification.
$$
\text{Cross-Entropy}(\mathbf{w}, X, y) = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log (\mathcal{NN}(\mathbf{w}, x_i)_c)
$$
where $y_{i,c}$ is a binary indicator (0 or 1) if class label $c$ is the correct classification for observation $i$, and $C$ is the number of classes.

## Simple example: Fashion MNIST classification problem

![](fashion_mnist.png)

![[\faPython Open in colab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/NN_optimization.ipynb)](Fashion_MNIST training.pdf){width=70%}

# Loss surface of Neural Networks

## Visualizing loss surface of neural network via line projection

We denote the initial point as $w_0$, representing the weights of the neural network at initialization. The weights after training are denoted as $\hat{w}$.

Initially, we generate a random Gaussian direction $w_1 \in \mathbb{R}^p$, which inherits the magnitude of the original neural network weights for each parameter group. Subsequently, we sample the training and testing loss surfaces at points along the direction $w_1$, situated close to either $w_0$ or $\hat{w}$.

Mathematically, this involves evaluating:
$$
L (\alpha) = L (w_0 + \alpha w_1), \text{ where } \alpha \in [-b, b].
$$
Here, $\alpha$ plays the role of a coordinate along the $w_1$ direction, and $b$ stands for the bounds of interpolation. Visualizing $L (\alpha)$ enables us to project the $p$-dimensional surface onto a one-dimensional axis.

It is important to note that the characteristics of the resulting graph heavily rely on the chosen projection direction. It's not feasible to maintain the entirety of the information when transforming a space with 100,000 dimensions into a one-dimensional line through projection. However, certain properties can still be established. For instance, if $L (\alpha) \mid_{\alpha=0}$ is decreasing, this indicates that the point lies on a slope. Additionally, if the projection is non-convex, it implies that the original surface was not convex.

## Visualizing loss surface of neural network

![[\faPython Open in colab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/NN_Surface_Visualization.ipynb)](Line_projection_No Dropout.pdf)

## Visualizing loss surface of neural network {.noframenumbering}

![[\faPython Open in colab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/NN_Surface_Visualization.ipynb)](Line_projection_Dropout 0.pdf)

## Plane projection

We can explore this idea further and draw the projection of the loss surface to the plane, which is defined by 2 random vectors. Note, that with 2 random gaussian vectors in the huge dimensional space are almost certainly orthogonal. So, as previously, we generate random normalized gaussian vectors $w_1, w_2 \in \mathbb{R}^p$ and evaluate the loss function
$$
L (\alpha, \beta) = L (w_0 + \alpha w_1 + \beta w_2), \text{ where } \alpha, \beta \in [-b, b]^2.
$$

![[\faPython Open in colab](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/NN_Surface_Visualization.ipynb)](plane_projection.jpeg){width=70%}

## Can plane projections be useful? ^[[Visualizing the Loss Landscape of Neural Nets, Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein](https://arxiv.org/abs/1712.09913)]

:::: {.columns}
::: {.column width="35%"}
![The loss surface of ResNet-56 without skip connections](noshortLog.png)
:::

::: {.column width="65%"}
![The loss surface of ResNet-56 with skip connections](shortHighResLog.png)
:::

::::

## Can plane projections be useful, really? ^[[Loss Landscape Sightseeing with Multi-Point Optimization, Ivan Skorokhodov, Mikhail Burtsev](https://arxiv.org/abs/1910.03867)]

![Examples of a loss landscape of a typical CNN model on FashionMNIST and CIFAR10 datasets found with MPO. Loss values are color-coded according to a logarithmic scale](icons-grid.png)

## Impact of initialization ^[[On the importance of initialization and momentum in deep learning Ilya Sutskever, James Martens, George Dahl, Geoffrey Hinton](https://proceedings.mlr.press/v28/sutskever13.html)]

:::{.callout-tip appearance="simple"}
Properly initializing a NN important. NN loss is highly nonconvex; optimizing it to attain a “good” solution hard, requires careful tuning. 
:::

* Don’t initialize all weights to be the same — why?
* Random: Initialize randomly, e.g., via the Gaussian $N(0, \sigma^2)$, where std $\sigma$ depends on the number of neurons in a given layer. *Symmetry breaking*.
* One can find more useful advices [here](https://cs231n.github.io/neural-networks-2/)

## Impact of initialization ^[[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](https://arxiv.org/abs/1502.01852)]

:::: {.columns}
::: {.column width="50%"}
![22-layer ReLU net: good init converges faster](converge_22layers.pdf)
:::

::: {.column width="50%"}
![30-layer ReLU net: good init is able to converge](converge_30layers.pdf)
:::

::::

## Grokking ^[[Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets,   Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra](https://arxiv.org/abs/2201.02177)]

![Training transformer with 2 layers, width 128, and 4 attention heads, with a total of about $4 \cdot 10^5$ non-embedding parameters. Reproduction of experiments (~ half an hour) is available [here](https://colab.research.google.com/drive/1r3Wg84XECq57fT2B1dvHLSJrJ2sjIDCJ?usp=sharing)](grokking.png){width=55%}

## Double Descent ^[[Reconciling modern machine learning practice and the bias-variance trade-off, Mikhail Belkin, Daniel Hsu, Siyuan Ma, Soumik Mandal](https://arxiv.org/abs/1812.11118)]

![](doubledescent.pdf){width=100%}

## Exponential learning rate

* [Exponential Learning Rate Schedules for Deep Learning](http://www.offconvex.org/2020/04/24/ExpLR1/)

# Modern problems

## Wide vs narrow local minima

![](sam_a.pdf)

## Wide vs narrow local minima {.noframenumbering}

![](sam_b.pdf)

## Wide vs narrow local minima {.noframenumbering}

![](sam_c.pdf)

## Stochasticity allows to escape local minima

![](sgd_escape.pdf)

## Local divergence can also be benefitial

![](sgd_local_divergence.pdf)

# Automatic Differentiation stories

## Gradient Vanishing/Exploding

* Multiplication of a chain of matrices in backprop
* If several of these matrices are “small” (i.e., norms < 1), when we multiply them, the gradient will decrease exponentially fast and tend to vanish (hurting learning in lower layers much more)
* Conversely, if several matrices have large norm, the gradient will tend to explode. In both cases, the gradients are unstable.
* Coping with unstable gradients poses several challenges, and must be
dealt with to achieve good results.

## Feedforward Architecture

![Computation graph for obtaining gradients for a simple feed-forward neural network with n layers. The activations marked with an $f$. The gradient of the loss with respect to the activations and parameters marked with $b$.](backprop.pdf){width=350}

. . .

::: {.callout-important}

The results obtained for the $f$ nodes are needed to compute the $b$ nodes.

:::

## Vanilla backpropagation

![Computation graph for obtaining gradients for a simple feed-forward neural network with n layers. The purple color indicates nodes that are stored in memory.](vanilla_backprop.pdf){width=350}

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

![Computation graph for obtaining gradients for a simple feed-forward neural network with n layers. The purple color indicates nodes that are stored in memory.](poor_mem_backprop.pdf){width=350}

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

![Computation graph for obtaining gradients for a simple feed-forward neural network with n layers. The purple color indicates nodes that are stored in memory.](checkpoint_backprop.pdf){width=350}

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

# Large batch training

## Large batch training

![](time.pdf)

## Large batch training

![](batchsize.pdf)