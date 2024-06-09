---
title: Conjugate functions. Dual (sub)gradient method. Augmented Lagrangian method. ADMM.
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

# Large batch distributed training

## Accurate, Large Minibatch SGD. Motivation

::: {.callout-tip title="Main Pros of Big Data"}
The increasing data and model scale is rapidly improving accuracy
:::

. . .

::: {.callout-important title="Main Cons of Big Data"}
As model and data scale grow, so does training time
:::

. . .

**Solution:** Use distributed SGD with large batch size to make **more efficient** iterations

## Accurate, Large Minibatch SGD. Problem
Loss function
$$ L(w) = \frac{1}{|X|} \sum_{x \in X} l(x, w) $$
One Iteration of Minibatch SGD (batch size is $n$)
$$ w_{t+1} = w_t - \eta \frac{1}{n} \sum_{x \in \mathcal{B}} \nabla l(x, w_t) $$
$k$ Iterations of Minibatch SGD (batch size is $n$)
$$ w_{t+k} = w_t - \eta \frac{1}{n} \sum_{j < k}\sum_{x \in \mathcal{B_j}} \nabla l(x, w_{t+j}) $$
One Large Batch Iteration of Minibatch SGD (batch size is $kn$)
$$ \hat{w}_{t+1} = w_t - \hat{\eta} \frac{1}{kn} \sum_{j < k}\sum_{x \in \mathcal{B_j}} \nabla l(x, w_{t}) $$

**Desired due to multi-GPU training**: $\hat{w}_{t+1} \sim w_{t+k}$

## Accurate, Large Minibatch SGD. Main idea
**Desired due to multi-GPU training**: $\hat{w}_{t+1} \sim w_{t+k}$

::: {.callout-tip title="Main Paper Asumption"}
If we could assume $\nabla l(x, w_t) \sim \nabla l(x, w_{t+j})$ for $j < k$, then setting $\hat \eta = k\eta$ would yield $\hat{w}_{t+1} \sim w_{t+k}$
:::

. . .

::: {.callout-question title="What is wrong with assumption"}
When is condition $\nabla l(x, w_t) \sim \nabla l(x, w_{t+j})$ clearly not hold?
:::

. . .

1. The network changes rapidly in initial training
1. Very large $k$ causes very large $\hat \eta$ and makes training too unstable

## Accurate, Large Minibatch SGD. Solving assumption problems
::: {.callout-question title="What to do with assumption problems"}
How would you struggle with assumption problems?
:::

. . .

**Gradual warmup.** Iteration-wise linear scheduler for start value $\hat \eta = \eta$ and finish value $\hat \eta  = k \eta$ after $\sim 5$ epochs.

* avoids a sudden increase of the learning rate

**Constant per-worker sample size.** For global batch size $kn$ we keep the *per-worker* sample size $n$ constant when changing the number of workers $k$. 

. . .

* extremly important for Batch Normalization!

## Accurate, Large Minibatch SGD. Results on ImageNet

:::: {.columns}
::: {.column width="80%"}
![](sem_19/sem_19_large_batch_results.pdf){width="85%"}

:::

::: {.column width="20%"}
The training curves closely match the baseline (aside from the warmup period) up through $8k$ minibatches.
:::
::::

## Accurate, Large Minibatch SGD. Results on ImageNet
:::: {.columns}
::: {.column width="50%"}
![](sem_19/sem_19_large_batch_train_val.pdf){width="100%"}

:::

::: {.column width="50%"}
Both sets
of curves match closely after training for sufficient epochs. 

Note that the BN statistics (for inference only) are computed using running average, which is updated less frequently with a large
minibatch and thus is noisier in early training (this explains the
larger variation of the validation error in early epochs).
:::
::::

# Reduce memory usage

## Reduce memory usage. CPU Offloading

* Offloading the weights to the CPU and only loading them on the GPU when performing the forward pass

* CPU offloading works on submodules rather than whole models. 

* Inference is much slower due to the iterative uploading and offloading. 

* Colab Example [\faPython Open in Colab](https://colab.research.google.com/drive/1Ugx3dUl_MHsYCAgz12qbuz_couJajpb6?usp=sharing).


## Reduce memory usage. Model Offloading

* CPU Offloading makes inference slower because *submodules* are moved to GPU as needed, and they’re immediately returned to the CPU when a new module runs.

* Full-model offloading is an alternative that moves whole models to the GPU, instead of handling each model’s constituent *submodules*.

* During model offloading, only one of the main components of the pipeline (typically the text encoder, UNet or VAE) is placed on the GPU while the others wait on the CPU.

* Colab Example [\faPython Open in Colab](https://colab.research.google.com/drive/1Ugx3dUl_MHsYCAgz12qbuz_couJajpb6?usp=sharing).


## Reduce memory usage. Quantization
* Quantization maps a floating point value $x \in [\alpha, \beta]$ to a 
$b$-bit integer $x_q \in [\alpha_q, \beta_q]$.

* The quantization process is defined as
$$x_q = \text{clip}\Big( \text{round}\big(\frac{1}{s} x + z\big), \alpha_q, \beta_q \Big)$$
 And the de-quantization process is defined as
 $$x = s (x_q - z)$$
 The value of scale $s$ and zero point $z$ are
 \begin{align}
s &= \frac{\beta - \alpha}{\beta_q - \alpha_q} \\
z &= \text{round}\big(\frac{\beta \alpha_q - \alpha \beta_q}{\beta - \alpha}\big) \\
\end{align}
**Note** that $z$ is an integer and $s$ is a positive floating point number.

* Quantization allows to perform a lot of heavy DL-operations (e.g. matrix maltiplication) in integer scope using efficient integer hardware (`NVIDIA Tensor Core` or `Tensor Core IMMA operations`) and algorithms.

## Reduce memory usage. Quantization
* For more theory look at *Quantization for Neural Networks
*, Lei Mao: [\faGit](https://leimao.github.io/article/Neural-Networks-Quantization/#Quantization).

* Colab Example [\faPython Open in Colab](https://colab.research.google.com/drive/1Ugx3dUl_MHsYCAgz12qbuz_couJajpb6?usp=sharing).
