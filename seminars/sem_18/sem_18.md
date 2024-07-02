---
title: Sharpness-Aware Minimization. Mode Connectivity. Grokking. Double Descent.
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

# SAM
## Flat Minimum vs Sharp Minimum
![](sem_18/flat_vs_sharp_minima.png){width=80% fig-align="center"}

. . .

::: {.callout-question title="Sharp Minimum"}
What's wrong with Sharp Minimum?
:::



## Sharpness-Aware Minimization^[[Foret, Pierre, et al. "Sharpness-aware minimization for efficiently improving generalization." (2020).](https://arxiv.org/pdf/2010.01412)]

:::: {.columns}

::: {.column width="50%"}
![A sharp minimum to which a ResNet trained with SGD converged.](sem_18/no_sam.png){width=60% fig-align="center"}
:::

. . .

::: {.column width="50%"}
![A wide minimum to which the same ResNet trained with SAM converged.](sem_18/sam_wide.png){width=60% fig-align="center"}
:::

::::

. . .

::: {.callout-tip icon="false" appearance="simple"}
Sharpness-Aware Minimization (SAM) is a procedure that aims to improve model generalization by simultaneously minimizing loss value and **loss sharpness**.
:::


## Learning setup

The training dataset drawn $i.i.d.$ from a distribution $D$:
$$S = \{(x_i, y_i)\}_{i=1}^{n},$$
where $x_i$ -- feature vector and $y_i$ -- label.

. . .

The training set loss:
$$L_{S} = \dfrac{1}{n} \sum_{i=1}^{n} l(\boldsymbol{w}, x_i, y_i),$$ 
where $l$ -- per-data-point loss function, $\boldsymbol{w}$ -- parameters. 

. . .

The population loss:
$$L_{D} = \mathbb{E}_{(x, y)} [l(\boldsymbol{w}, \boldsymbol{x}, \boldsymbol{y})]$$


## What is sharpness?

:::{.callout-theorem}
For any $\rho>0$, with high probability over training set $S$ generated from distribution $D$,
$$
L_{D}(\boldsymbol{w}) \leq \max _{\|\boldsymbol{\epsilon}\|_2 \leq \rho} L_{S}(\boldsymbol{w}+\boldsymbol{\epsilon})+h\left(\|\boldsymbol{w}\|_2^2 / \rho^2\right),
$$
where $h: \mathbb{R}_{+} \rightarrow \mathbb{R}_{+}$is a strictly increasing function (under some technical conditions on $L_{D}(\boldsymbol{w})$ ).
:::

. . .

Adding and subtracting $L_{S}(\boldsymbol{w})$:
$$
\left[\max _{\|\in\|_2 \leq \rho} L_{{S}}(\boldsymbol{w}+\boldsymbol{\epsilon})-L_{{S}}(\boldsymbol{w})\right]+L_{\mathcal{S}}(\boldsymbol{w})+h\left(\|\boldsymbol{w}\|_2^2 / \rho^2\right)
$$

The term in square brackets captures the **sharpness** of $L_S$ at $\boldsymbol{w}$ by measuring how quickly the training loss can be increased by moving from $\boldsymbol{w}$ to a nearby parameter value.


## Sharpness-Aware Minimization

The function $h$ is removed in favor of a simpler constant $\lambda$. The authors propose selecting parameter values by solving  the following Sharpness-Aware Minimization (SAM) problem:

$$
\min _{\boldsymbol{w}} L_{{S}}^{S A M}(\boldsymbol{w})+\lambda\|\boldsymbol{w}\|_2^2 \quad \text { where } \quad L_{{S}}^{S A M}(\boldsymbol{w}) \triangleq \max _{\|\epsilon\|_p \leq \rho} L_S(\boldsymbol{w}+\boldsymbol{\epsilon}),
$$

with $\rho \geq 0$ as hyperparameter and $p$ in $[1, \infty]$ (a little generalization, though $p=2$ is empirically the best choice).

## How to minimize $L_{{S}}^{S A M}$?


In order to minimize $L_{{S}}^{S A M}$ an efficient approximation of its gradient is used. A first step is to consider the first-order Taylor expansion of $L_{{S}}(\boldsymbol{w}+\boldsymbol{\epsilon})$:
$$
\boldsymbol{\epsilon}^*(\boldsymbol{w}) \triangleq \underset{\|\epsilon\|_p \leq \rho}{\arg \max } L_{{S}}(\boldsymbol{w}+\boldsymbol{\epsilon}) \approx \underset{\|\epsilon\|_p \leq \rho}{\arg \max } L_{{S}}(\boldsymbol{w})+\boldsymbol{\epsilon}^T \nabla_{\boldsymbol{w}} L_{{S}}(\boldsymbol{w})=\underset{\|\epsilon\|_p \leq \rho}{\arg \max } \boldsymbol{\epsilon}^T \nabla_{\boldsymbol{w}} L_{{S}}(\boldsymbol{w}) .
$$

. . .

The last expression is just the argmax of the dot product of the vectors $\boldsymbol{\epsilon}$ and $\nabla_{\boldsymbol{w}} L_{{S}}(\boldsymbol{w})$, and it is well known which is the argument that maximizes it:

$$
\hat{\boldsymbol{\epsilon}}(\boldsymbol{w})=\rho \operatorname{sign}\left(\nabla_{\boldsymbol{w}} L_{{S}}(\boldsymbol{w})\right)\left|\nabla_{\boldsymbol{w}} L_{{S}}(\boldsymbol{w})\right|^{q-1} /\left(\left\|\nabla_{\boldsymbol{w}} L_{{S}}(\boldsymbol{w})\right\|_q^q\right)^{1 / p},
$$
where $1 / p+1 / q=1$.

. . .

Thus
$$
\begin{aligned}
\nabla_{\boldsymbol{w}} L_{{S}}^{S A M}(\boldsymbol{w}) & \approx \nabla_{\boldsymbol{w}} L_{{S}}(\boldsymbol{w}+\hat{\boldsymbol{\epsilon}}(\boldsymbol{w}))=\left.\frac{d(\boldsymbol{w}+\hat{\boldsymbol{\epsilon}}(\boldsymbol{w}))}{d \boldsymbol{w}} \nabla_{\boldsymbol{w}} L_{{S}}(\boldsymbol{w})\right|_{\boldsymbol{w}+\hat{\boldsymbol{\epsilon}}(w)} \\
& =\left.\nabla_w L_{{S}}(\boldsymbol{w})\right|_{\boldsymbol{w}+\hat{\boldsymbol{\epsilon}}(\boldsymbol{w})}+\left.\frac{d \hat{\boldsymbol{\epsilon}}(\boldsymbol{w})}{d \boldsymbol{w}} \nabla_{\boldsymbol{w}} L_{{S}}(\boldsymbol{w})\right|_{\boldsymbol{w}+\hat{\boldsymbol{\epsilon}}(\boldsymbol{w})}
\end{aligned}
$$


## Sharpness-Aware Minimization

Modern frameworks can easily compute the preceding approximation. However, to speed up the computation, second-order terms can be dropped obtaining:
$$
\left.\nabla_{\boldsymbol{w}} L_{{S}}^{S A M}(\boldsymbol{w}) \approx \nabla_{\boldsymbol{w}} L_{{S}}(w)\right|_{\boldsymbol{w}+\hat{\boldsymbol{\epsilon}}(\boldsymbol{w})}
$$

. . .

![SAM pseudo-code](sem_18/sam_code.png){width=100% fig-align="center"}

## SAM results

![Error rate reduction obtained by switching to SAM. Each point is a different dataset / model / data augmentation.](sem_18/summary_plot.png){width=50% fig-align="center"}


# Mode Connectivity
## Mode Connectivity^[[Garipov, Timur, et al. "Loss surfaces, mode connectivity, and fast ensembling of dnns." Advances in neural information processing systems 31 (2018).](https://arxiv.org/pdf/1802.10026)]
![The $l_2$-regularized cross-entropy train loss surface of a ResNet-164 on CIFAR-100, as a function of network weights in a two-dimensional subspace. In each panel, the horizontal axis is fixed and is attached to the optima of two independently trained networks. The vertical axis changes between panels as we change planes (defined in the main text). Left: Three optima for independently trained networks. Middle and Right: A quadratic Bezier curve, and a polygonal chain with one bend, connecting the lower two optima on the left panel along a path of near-constant loss. Notice that in each panel a direct linear path between each mode would incur high loss.](sem_18/mode_connectivity.png){width=80% fig-align="center"}



## Curve-Finding Procedure

:::: {.columns}

::: {.column width="50%"}

- Weights of pretrained networks: 

$$\widehat{w}_1, \widehat{w}_2 \in \mathbb{R}^{\mid \text {net} \mid}$$


- Define parametric curve: $\phi_\theta(\cdot):[0,1] \rightarrow \mathbb{R}^{\mid \text {net} \mid}$

$$
\phi_\theta(0)=\widehat{w}_1, \quad \phi_\theta(1)=\widehat{w}_2
$$

- DNN loss function: 

$$\mathcal{L}(w)$$

- Minimize averaged loss w.r.t. $\theta$:


$$
\underset{\theta}{\operatorname{minimize}} \;\; \ell(\theta)=\int_0^1 \mathcal{L}\left(\phi_\theta(t)\right) d t=\mathbb{E}_{t \sim U(0,1)} \mathcal{L}\left(\phi_\theta(t)\right)
$$

:::

::: {.column width="50%"}
![](sem_18/mc_ep4.png){width=70% fig-align="center"}
:::

::::

## Curve-Finding Procedure

:::: {.columns}

::: {.column width="50%"}

- Weights of pretrained networks: 

$$\widehat{w}_1, \widehat{w}_2 \in \mathbb{R}^{\mid \text {net} \mid}$$


- Define parametric curve: $\phi_\theta(\cdot):[0,1] \rightarrow \mathbb{R}^{\mid \text {net} \mid}$

$$
\phi_\theta(0)=\widehat{w}_1, \quad \phi_\theta(1)=\widehat{w}_2
$$

- DNN loss function: 

$$\mathcal{L}(w)$$

- Minimize averaged loss w.r.t. $\theta$:


$$
\underset{\theta}{\operatorname{minimize}} \;\; \ell(\theta)=\int_0^1 \mathcal{L}\left(\phi_\theta(t)\right) d t=\mathbb{E}_{t \sim U(0,1)} \mathcal{L}\left(\phi_\theta(t)\right)
$$

:::

::: {.column width="50%"}
![](sem_18/mc_ep50.png){width=70% fig-align="center"}
:::

::::

## Curve-Finding Procedure

:::: {.columns}

::: {.column width="50%"}

- Weights of pretrained networks: 

$$\widehat{w}_1, \widehat{w}_2 \in \mathbb{R}^{\mid \text {net} \mid}$$


- Define parametric curve: $\phi_\theta(\cdot):[0,1] \rightarrow \mathbb{R}^{\mid \text {net} \mid}$

$$
\phi_\theta(0)=\widehat{w}_1, \quad \phi_\theta(1)=\widehat{w}_2
$$

- DNN loss function: 

$$\mathcal{L}(w)$$

- Minimize averaged loss w.r.t. $\theta$:


$$
\underset{\theta}{\operatorname{minimize}} \;\; \ell(\theta)=\int_0^1 \mathcal{L}\left(\phi_\theta(t)\right) d t=\mathbb{E}_{t \sim U(0,1)} \mathcal{L}\left(\phi_\theta(t)\right)
$$

:::

::: {.column width="50%"}
![](sem_18/mc_ep500.png){width=70% fig-align="center"}
:::

::::

# Grokking
## Grokking^[[Power, Alethea, et al. "Grokking: Generalization beyond overfitting on small algorithmic datasets." (2022).](https://arxiv.org/pdf/2201.02177)]

:::: {.columns}

::: {.column width="40%"}

- After achieving zero train loss the weights continue evolving in a kind of random walk manner    


- It is possible that they slowly drift to a wider minima


- Recently discovered grokking effect confirms this hypo

:::

::: {.column width="60%"}
![Grokking: A dramatic example of generalization far after overfitting on an algorithmic dataset.](sem_18/grokking.png){width=70% fig-align="center"}

:::

::::

# Double Descent
## Double Descent^[[Belkin, Mikhail, et al. "Reconciling modern machine-learning practice and the classical bias–variance trade-off." (2019)](https://arxiv.org/pdf/1812.11118)]

![Curves for training risk (dashed line) and test risk (solid line). (a) The classical U-shaped risk curve arising from the bias-variance trade-off. (b) The double descent risk curve, which incorporates the U-shaped risk curve (i.e., the “classical” regime) together with the observed behavior from using high capacity function classes (i.e., the “modern” interpolating regime), separated by the interpolation threshold. The predictors to the right of the interpolation threshold have zero training risk.](sem_18/double_descent.png){width=100% fig-align="center"}