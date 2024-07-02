---
title: "Large models training"
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
  - \newcommand{\bgimage}{../../files/back19.jpeg}
---

# GPT-2 training Memory footprint

## GPT-2 training Memory footprint

:::: {.columns}
::: {.column width="25%"}
![](gpt2_memory.pdf)
:::

::: {.column width="75%"}

Example: 1.5B parameter GPT-2 model needs 3GB for weights in 16-bit precision but can't be trained on a 32GB GPU using Tensorflow or PyTorch. Major memory usage during training includes optimizer states, gradients, parameters, activations, temporary buffers, and fragmented memory.

**Model States:**

* Optimizer states (e.g., Adam) require memory for time-averaged momentum and gradient variance.
* Mixed-precision training (fp16/32) necessitates storing parameters and activations as fp16, but keeps fp32 copies for updates.

**Memory Requirements Example:**

* Training with Adam in mixed precision for a model with $\Psi$ parameters: 2$\Psi$ bytes for fp16 parameters and gradients, 12$\Psi$ bytes for optimizer states (parameters, momentum, variance).
* Total: 16$\Psi$ bytes; for GPT-2 with 1.5B parameters, this equals 24GB.

**Residual Memory Consumption:**

* Activations: Significant memory usage, e.g., 1.5B parameter GPT-2 model with sequence length 1K and batch size 32 requires ~60GB.
* Activation checkpointing can reduce activation memory by about 50%, with a 33% recomputation overhead.

:::

::::

## GPT-2 training Memory footprint{.noframenumbering}

:::: {.columns}
::: {.column width="25%"}
![](gpt2_memory.pdf)
:::

::: {.column width="75%"}

Example: 1.5B parameter GPT-2 model needs 3GB for weights in 16-bit precision but can't be trained on a 32GB GPU using Tensorflow or PyTorch. Major memory usage during training includes optimizer states, gradients, parameters, activations, temporary buffers, and fragmented memory.

**Temporary Buffers:**

* Store intermediate results; e.g., gradient all-reduce operations fuse gradients into a single buffer.
* For large models, temporary buffers can consume substantial memory (e.g., 6GB for 1.5B parameter model with fp32 buffer).

**Memory Fragmentation:**

* Memory fragmentation can cause out-of-memory issues despite available memory, as contiguous blocks are required.
* In some cases, over 30% of memory remains unusable due to fragmentation.

:::

::::



# Large batch training

## Large batch training ^[[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)]

![](time.pdf){width=90%}

## Large batch training ^[[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)]

![](batchsize.pdf){width=85%}

## Large batch training ^[[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)]

| Effective batch size ($kn$)  | $\alpha$ | top-1 error (%)  |
|:-------:|:-----------------------:|:------------------------:|
| 256   | $0.05$                | 23.92 ± 0.10           |
| 256   | $0.10$                | 23.60 ± 0.12           |
| 256   | $0.20$                | 23.68 ± 0.09           |
| 8k    | $0.05 \cdot 32$       | 24.27 ± 0.08           |
| 8k    | $0.10 \cdot 32$       | 23.74 ± 0.09           |
| 8k    | $0.20 \cdot 32$       | 24.05 ± 0.18           |
| 8k    | $0.10$                | 41.67 ± 0.10           |
| 8k    | $0.10 \cdot \sqrt{32}$| 26.22 ± 0.03           |

Comparison of learning rate scaling rules. ResNet-50 trained on ImageNet. A reference learning rate of $\alpha=0.1$ works best for $kn=256$ (23.68% error). The linear scaling rule suggests $\alpha=0.1\cdot32$ when $kn=8$k, which again gives best performance (23.74\% error). Other ways of scaling $\alpha$ give worse results.

## Linear and square root scaling rules

When training with large batches, the learning rate must be adjusted to maintain convergence speed and stability. The **linear scaling rule**^[[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)] suggests multiplying the learning rate by the same factor as the increase in batch size:
$$
\alpha_{\text{new}} = \alpha_{\text{base}} \cdot \frac{\text{Batch Size}_{\text{new}}}{\text{Batch Size}_{\text{base}}}
$$
The **square root scaling rule**^[[Learning Rates as a Function of Batch Size: A Random Matrix Theory Approach to Neural Network Training](https://arxiv.org/abs/2006.09092)] proposes scaling the learning rate with the square root of the batch size increase:
$$
\alpha_{\text{new}} = \alpha_{\text{base}} \cdot \sqrt{\frac{\text{Batch Size}_{\text{new}}}{\text{Batch Size}_{\text{base}}}}
$$
Authors claimed, that it suits for adaptive optimizers like Adam, RMSProp and etc. while linear scaling rule serves well for SGD.

## Gradual warmup ^[[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)]

Gradual warmup helps to avoid instability when starting with large learning rates by slowly increasing the learning rate from a small value to the target value over a few epochs. This is defined as:
$$
\alpha_t = \alpha_{\text{max}} \cdot \frac{t}{T_w}
$$
where $t$ is the current iteration and $T_w$ is the warmup duration in iterations. In the original paper, authors used first 5 epochs for gradual warmup.

:::: {.columns}
::: {.column width="36%"}
![no warmup](distr-warmup-none.pdf)
:::

::: {.column width="32%"}
![constant warmup](distr-warmup-constant.pdf)
:::

::: {.column width="32%"}
![gradual warmup](distr-warmup-gradual.pdf)
:::

::::

## Gradient accumulation

Gradient accumulation allows the effective batch size to be increased without requiring larger memory by accumulating gradients over several mini-batches:

:::: {.columns}
::: {.column width="50%"}

### Without gradient accumulation

```python
for i, (inputs, targets) in enumerate(data):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
```
:::

. . .

::: {.column width="50%"}

### With gradient accumulation

```python
for i, (inputs, targets) in enumerate(data):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    if (i+1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

:::

::::



# MultiGPU training

## Data Parallel training

1. Parameter server sends the full copy of the model to each device
2. Each device makes forward and backward passes
3. Parameter server gathers gradients
4. Parameter server updates the model

. . .

Per device batch size: $b$. Overall batchsize: $Db$. Data parallelism involves splitting the data across multiple GPUs, each with a copy of the model. Gradients are averaged and weights updated synchronously:

![Scheme of Data Parallel training](DP.pdf){width=80%}

## Distributed Data Parallel training

Distributed Data Parallel (DDP) ^[[Getting Started with Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)] extends data parallelism across multiple nodes. Each node computes gradients locally, then synchronizes with others. Below one can find differences from the PyTorch [site](https://pytorch.org/tutorials/beginner/ddp_series_theory.html). This is used by default in [ \faPython Accelerate library](https://huggingface.co/docs/transformers/accelerate).

|    DataParallel   | DistributedDataParallel  |
|:----------------:|:----------------:|
| More overhead; model is replicated and destroyed at each forward pass | Model is replicated only once                                    |
| Only supports single-node parallelism                            | Supports scaling to multiple machines                            |
| Slower; uses multithreading on a single process and runs into Global Interpreter Lock (GIL) contention | Faster (no GIL contention) because it uses multiprocessing |


## Naive model parallelism 

Model parallelism divides the model across multiple GPUs. Each GPU handles a subset of the model layers, reducing memory load per GPU. Allows to work with the models, that won’t fit in the single GPU
Poor resource utilization. 

![Model parallelism](MP.png)

## Pipeline model parallelism (GPipe) ^[[GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)]

GPipe splits the model into stages, each processed sequentially. Micro-batches are passed through the pipeline, allowing for overlapping computation and communication:
![](gpipe.png)


## Pipeline model parallelism (PipeDream) ^[[PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377)]

PipeDream uses asynchronous pipeline parallelism, balancing forward and backward passes across the pipeline stages to maximize utilization and reduce idle time:
![](pipedream.png)

## ZeRO ^[[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)]

![](zero.png)

## LoRA ^[[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)]

:::: {.columns}
::: {.column width="50%"}
![](lora.pdf)
:::

::: {.column width="50%"}

LoRA reduces the number of parameters by approximating weight matrices with low-rank factorization:
$$
W_{\text{new}} = W + \Delta W
$$
where $\Delta W = A B^T$, with $A$ and $B$ being low-rank matrices. This reduces computational and memory overhead while maintaining model performance.

* $A$ is initialized as usual, while $B$ is initialized with zeroes in order to start from identity mapping
* $r$ is typically selected between 2 and 64
* Usually applied to attention modules

. . .

$$
h = W_{\text{new}}x = Wx + \Delta Wx = Wx + AB^T x
$$
:::

::::

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

# Quantization

## Split the weight matrix into 2 well clustered factors ^[[Quantization of Large Language Models with an Overdetermined Basis](https://arxiv.org/abs/2404.09737)]

![Scheme of post-training quantization approach.](quantization_scheme.pdf)