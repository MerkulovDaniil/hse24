---
title: Advanced stochastic methods. Adaptivity and variance reduction
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
## Main problem of SGD

$$
f(x) = \frac{\mu}{2} \|x\|_2^2 + \frac1m \sum_{i=1}^m \log (1 + \exp(- y_i \langle a_i, x \rangle)) \to \min_{x \in \mathbb{R}^n}
$$

![](sem_17/sgd_problems.pdf)

## Key idea of variance reduction

**Principle:** reducing variance of a sample of $X$ by using a sample from another random variable $Y$ with known expectation:
$$
Z_\alpha = \alpha (X - Y) + \mathbb{E}[Y]
$$

* $\mathbb{E}[Z_\alpha] = \alpha \mathbb{E}[X] + (1-\alpha)\mathbb{E}[Y]$
* $\text{var}(Z_\alpha) = \alpha^2 \left(\text{var}(X) + \text{var}(Y) - 2\text{cov}(X, Y)\right)$
  * If $\alpha = 1$: no bias
  * If $\alpha < 1$: potential bias (but reduced variance).
* Useful if $Y$ is positively correlated with $X$.

**Application to gradient estimation ?**

* SVRG: Let $X = \nabla f_{i_k}(x^{(k-1)})$ and $Y = \nabla f_{i_k}(\tilde{x})$, with $\alpha = 1$ and $\tilde{x}$ stored.
* $\mathbb{E}[Y] = \frac{1}{n} \sum_{i=1}^n \nabla f_i(\tilde{x})$ full gradient at $\tilde{x}$;
* $X - Y = \nabla f_{i_k}(x^{(k-1)}) - \nabla f_{i_k}(\tilde{x})$

## SVRG (Stochastic Variance Reduced Gradient; Johnson, and Zhang, 2013)

* **Initialize:** $\tilde{x} \in \mathbb{R}^d$
* **For** $i_{epoch} = 1$ **to** `# of epochs`
  * Compute all gradients $\nabla f_i(\tilde{x})$; store $\nabla f(\tilde{x}) = \frac{1}{n} \sum_{i=1}^n \nabla f_i(\tilde{x})$
  * Initialize $x_0 = \tilde{x}$
  * **For** `t = 1` **to** `length of epochs (m)`
    * $x_t = x_{t-1} - \alpha \left[\nabla f(\tilde{x}) + \left(\nabla f_{i_t}(x_{t-1}) - \nabla f_{i_t}(\tilde{x})\right)\right]$
    * Update $\tilde{x} = x_t$


**Notes:**

* Two gradient evaluations per inner step.
* Two parameters: length of epochs + step-size $\gamma$.
* Linear convergence rate, simple proof.

## SAG (Stochastic average gradient, Schmidt, Le Roux, and Bach 2013)

* Maintain table, containing gradient $g_i$ of $f_i$, $i = 1, \dots, n$
* Initialize $x^{(0)}$, and $g^{(0)}_i = \nabla f_i(x^{(0)})$, $i = 1, \dots, n$
* At steps $k = 1,2,3,\dots$, pick random $i_k \in \{1,\dots,n\}$, then let
  $$
  g^{(k)}_{i_k} = \nabla f_{i_k}(x^{(k-1)}) \quad \text{(most recent gradient of $f_{i_k}$)}
  $$
  Set all other $g_i^{(k)} = g_i^{(k-1)}$, $i \neq i_k$, i.e., these stay the same
* Update
  $$
  x^{(k)} = x^{(k-1)} - \alpha_k \frac{1}{n} \sum_{i=1}^n g_i^{(k)}
  $$
* SAG gradient estimates are no longer unbiased, but they have greatly reduced variance
* Isn't it expensive to average all these gradients? Basically just as efficient as SGD, as long we're clever:
  $$
  x^{(k)} = x^{(k-1)} - \alpha_k \underbrace{ \left(\frac1n g_i^{(k)} - \frac1ng_i^{(k-1)} + \underbrace{\frac{1}{n} \sum_{i=1}^n g_i^{(k-1)}}_{\text{old table average}}\right)}_{\text{new table average}}
  $$

## SAG convergence

Assume that $f(x) = \frac{1}{n} \sum_{i=1}^n f_i(x)$, where each $f_i$ is differentiable, and $\nabla f_i$ is Lipschitz with constant $L$.

Denote $\bar{x}^{(k)} = \frac{1}{k} \sum_{l=0}^{k-1} x^{(l)}$, the average iterate after $k - 1$ steps.

:::{.callout-theorem}
SAG, with a fixed step size $\alpha = \frac{1}{16L}$, and the initialization
$$g^{(0)}_i = \nabla f_i(x^{(0)}) - \nabla f(x^{(0)}), \quad i=1,\dots,n$$
satisfies
$$\mathbb{E}[f(\bar{x}^{(k)})] - f^\star \leq \frac{48n}{k}[f(x^{(0)}) - f^\star] + \frac{128L}{k} \|x^{(0)} - x^\star\|^2$$
where the expectation is taken over random choices of indices.
:::

## SAG convergence

* Result stated in terms of the average iterate $\bar{x}^{(k)}$, but also can be shown to hold for the best iterate $x^{(k)}_{best}$ seen so far.
* This is $\mathcal{O}\left(\frac{1}{k}\right)$ convergence rate for SAG. Compare to $\mathcal{O}\left(\frac{1}{k}\right)$ rate for GD, and $\mathcal{O}\left(\frac{1}{\sqrt{k}}\right)$ rate for SGD.
* But, the constants are different! Bounds after k steps:
  * GD: $\frac{L \|x^{(0)} - x^\star\|^2}{2k}$
  * SAG: $\frac{48n [f(x^{(0)}) - f^\star] + 128L \|x^{(0)} - x^\star\|^2}{k}$
* So the first term in SAG bound suffers from a factor of $n$; authors suggest smarter initialization to make $f(x^{(0)}) - f^\star$ small (e.g., they suggest using the result of $n$ SGD steps).

## SAG convergence

Assume further that each $f_i$ is strongly convex with parameter $\mu$.

:::{.callout-theorem} 
SAG, with a step size $\alpha = \frac{1}{16L}$ and the same initialization as before, satisfies
$$
\mathbb{E}[f(x^{(k)})] - f^\star \leq \left(1 - \min\left(\frac{\mu}{16L}, \frac{1}{8n}\right)\right)^k \left(\frac32\left(f(x^{(0)}) - f^\star\right) + \frac{4L}{n} \|x^{(0)} - x^\star\|^2\right)
$$
:::

**Notes:**

* This is linear convergence rate $\mathcal{O}(\gamma^k)$ for SAG. Compare this to $\mathcal{O}(\gamma^k)$ for GD, and only $\mathcal{O}\left(\frac{1}{k}\right)$ for SGD.
* Like GD, we say SAG is adaptive to strong convexity.
* Proofs of these results not easy: 15 pages, computed-aided!

# Adaptivity or scaling
## Adagrad (Duchi, Hazan, and Singer 2010)

Very popular adaptive method. Let $g^{(k)} = \nabla f_{i_k}(x^{(k-1)})$, and update for $j = 1, \dots, p$:

$$
v^{(k)}_j = v^{k-1}_j + (g_j^{(k)})^2
$$
$$
x_j^{(k)} = x_j^{(k-1)} - \alpha \frac{g_j^{(k)}}{\sqrt{v^{(k)}_j  + \epsilon}}
$$

**Notes:**

* AdaGrad does not require tuning the learning rate: $\alpha > 0$ is a fixed constant, and the learning rate decreases naturally over iterations.
* The learning rate of rare informative features diminishes slowly.
* Can drastically improve over SGD in sparse problems.
* Main weakness is the monotonic accumulation of gradients in the denominator. AdaDelta, Adam, AMSGrad, etc. improve on this, popular in training deep neural networks.
* The constant $\epsilon$ is typically set to $10^{-6}$ to ensure that we do not suffer from division by zero or overly large step sizes.

## RMSProp (Tieleman and Hinton, 2012)

An enhancement of AdaGrad that addresses its aggressive, monotonically decreasing learning rate. Uses a moving average of squared gradients to adjust the learning rate for each weight. Let $g^{(k)} = \nabla f_{i_k}(x^{(k-1)})$ and update rule for $j = 1, \dots, p$:
$$
v^{(k)}_j = \gamma v^{(k-1)}_j + (1-\gamma) (g_j^{(k)})^2
$$
$$
x_j^{(k)} = x_j^{(k-1)} - \alpha \frac{g_j^{(k)}}{\sqrt{v^{(k)}_j + \epsilon}}
$$

**Notes:**

* RMSProp divides the learning rate for a weight by a running average of the magnitudes of recent gradients for that weight.
* Allows for a more nuanced adjustment of learning rates than AdaGrad, making it suitable for non-stationary problems.
* Commonly used in training neural networks, particularly in recurrent neural networks.

## Adadelta (Zeiler, 2012)

An extension of RMSProp that seeks to reduce its dependence on a manually set global learning rate. Instead of accumulating all past squared gradients, Adadelta limits the window of accumulated past gradients to some fixed size $w$. Update mechanism does not require learning rate $\alpha$:

$$
v^{(k)}_j = \gamma v^{(k-1)}_j + (1-\gamma) (g_j^{(k)})^2
$$
$$
\tilde{g}_j^{(k)} = \frac{\sqrt{{\Delta x_j^{(k-1)}} + \epsilon}}{\sqrt{v^{(k)}_j+ \epsilon}} g_j^{(k)}
$$
$$
x_j^{(k)} = x_j^{(k-1)} - \tilde{g}_j^{(k)}
$$
$$
\Delta x_j^{(k)} = \rho \Delta x_j^{(k-1)} + (1-\rho) (\tilde{g}_j^{(k)})^2
$$

**Notes:**

* Adadelta adapts learning rates based on a moving window of gradient updates, rather than accumulating all past gradients. This way, learning rates adjusted are more robust to changes in model's dynamics.
* The method does not require an initial learning rate setting, making it easier to configure.
* Often used in deep learning where parameter scales differ significantly across layers.

## Adam (Kingma and Ba, 2014)

Combines elements from both AdaGrad and RMSProp. It considers an exponentially decaying average of past gradients and squared gradients. Update rule:

$$
m_j^{(k)} = \beta_1 m_j^{(k-1)} + (1-\beta_1) g_j^{(k)}
$$

$$
v_j^{(k)} = \beta_2 v_j^{(k-1)} + (1-\beta_2) (g_j^{(k)})^2
$$

$$
\hat{m}_j = \frac{m_j^{(k)}}{1-\beta_1^k}, \quad \hat{v}_j = \frac{v_j^{(k)} }{1-\beta_2^k}
$$

$$
x_j^{(k)} = x_j^{(k-1)} - \alpha \frac{\hat{m}_j}{\sqrt{\hat{v}_j} + \epsilon}
$$

**Notes:**

* Adam is suitable for large datasets and high-dimensional optimization problems.
* It corrects the bias towards zero in the initial moments seen in other methods like RMSProp, making the estimates more accurate.
* Highly popular in training deep learning models, owing to its efficiency and straightforward implementation.
* However, the proposed algorithm in initial version does not converge even in convex setting (later fixes appeared)

# Computational experiments
## Computational experiments

Let's look at computational experiments for

- SGD, SAG and SVRG in JAX [\faPython](https://colab.research.google.com/drive/1K2tVGX_GHXCSdP1VJsidUL4xRTWzOZ4q?usp=sharing).

- SVRG in JAX for VAE [\faPython](https://colab.research.google.com/drive/1K_FeWban_eXDq-dJnoU1k-zWoMjjgg31?usp=sharing).



