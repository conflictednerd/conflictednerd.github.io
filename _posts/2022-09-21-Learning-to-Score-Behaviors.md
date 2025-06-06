---
layout: post
title: Learning to Score Behaviors
date: 2022-09-21 14:52:46
description: An extended summary of "Learning to Score Behaviors for Guided Policy Optimization".
tags: RL, paper-summary
categories: blog
related_posts: true
toc: true
math: true
---

(This is a note based on [Learning to Score Behaviors for Guided Policy Optimization](https://arxiv.org/abs/1906.04349). I am trying to expand and clarify some of the algorithms that were presented there. <span style="color:red">More content may be added to this note in the future!</span>)

The core question:

> What is the right measure of similarity between two policies acting on the same underlaying MDP and how can we devise algorithms to leverage this information for RL?

## Concepts

### Behavioral Embedding Map (BEM)

A function $\Phi:\Gamma \to \mathcal{E}$ that maps trajectories to embeddings. $\mathcal{E}$ can be seen as a behavioral manifold. Examples of BEMs include:

1. final state $\Phi (\tau) = s_H$,
2. actions vector $\Phi(\tau) = [a_0, \cdots, a_H]$,
3. total reward $\Phi(\tau) = \sum_{t=0}^{H}r_t$.

### (On-Policy) Policy Embedding

Given a policy $\pi$, let $\mathbb{P}_\pi$ be the distribution it induces over the *space of trajectories, $\Gamma$*. For a behavioral embedding map $\Phi$, let $\mathbb{P}_\pi^\Phi$ be the corresponding pushforward distribution on $\mathcal{E}$ induced by $\Phi$. $P_\pi^\Phi$ is called the *policy embedding* of a policy $\pi$.

### Wasserstein Distance

Let $\mu, \nu$ be probability measures over $\mathcal{X} \subseteq \mathbb{R}^m$, $\mathcal{Y} \subseteq \mathbb{R}^n$ and let $C: \mathcal{X}\times\mathcal{Y} \to \mathbb{R}$ be a cost function. For $\gamma > 0$, a smoothed Wasserstein distance is defined as
$$
\mathrm{WD}_\gamma (\mu, \nu) := \min_{\pi \in \Pi(\mu, \nu)} \int_{\mathcal{X}\times\mathcal{Y}} C(\mathbf{x}, \mathbf{y})d_\pi(\mathbf{x}, \mathbf{y}) + \Sigma,
$$
where
$$
\Sigma = \gamma \mathrm{KL}(\pi \| \xi),
$$
$\Pi(\mu, \nu)$ is the space of couplings (joint distributions) over $\mathcal{X}\times\mathcal{Y}$ with marginal distributions $\mu$ and $\nu$. When the cost is an $\mathscr{l}_p$ distance and $\gamma = 0$, $\mathrm{WD}_\gamma$ is known as the Earth mover's distance and the corresponding optimization is known as the *optimal transport problem*.

Notice that this is a hard optimization problem to solve: the search space is the space of couplings of $\mu, \nu$ and the objective involves an integral.

### Wasserstein Distance: Dual Formulation

Let $\mathcal{C}(\mathcal{X})$ and $\mathcal{C}(\mathcal{Y})$ denote the space of continuous functions over $\mathcal{X}$ and $\mathcal{Y}$ respectively. We can view the cost function $C: \mathcal{X}\times\mathcal{Y} \to \mathbb{R}$ as the *"ground cost"* of moving a unit of mass from $x$ to $y$. Using Fenchel duality, we can obtain the following dual formulation of the optimization problem that defines WD:
$$
\mathrm{WD}(\mu, \nu) = \max_{\lambda_\mu\in \mathcal{C}(\mathcal{X}), \lambda_\nu \in \mathcal{C}(\mathcal{Y})} \Psi(\lambda_\mu, \lambda_\nu),
$$
where
$$
\Psi(\lambda_\mu, \lambda_\nu) = \int_\mathcal{X} \lambda_\mu(\mathbf{x})d_\mu(\mathbf{x}) - \int_\mathcal{Y} \lambda_\nu(\mathbf{y})d_\nu(\mathbf{y}) - E_C(\lambda_\mu, \lambda_\nu).
$$
The last term, $E_C$ is known as the damping term and is defined as 
$$
E_C (\lambda_\mu, \lambda_\nu) = \mathbb{I}(\gamma > 0) \int_{\mathcal{X}\times\mathcal{Y}} \rho(\mathbf{x}, \mathbf{y}) d_\xi(\mathbf{x}, \mathbf{y}) + \mathbb{I}(\gamma = 0) \mathbb{I}(\mathcal{A}),
$$
for 
$$
\rho(\mathbf{x}, \mathbf{y}) = \gamma \exp (\frac{\lambda_\mu(\mathbf{x}) - \lambda_\nu (\mathbf{y}) - C(\mathbf{x}, \mathbf{y})}{\gamma})
$$
and 
$$
\mathcal{A} = \left[(\lambda_\mu, \lambda_\nu) \in \{(u, v) \mathrm{\\;s.t.\\;} \forall (\mathbf{x}, \mathbf{y}) \in \mathcal{X}\times\mathcal{Y}: u(\mathbf{x}) - v(\mathbf{y}) \leqslant C(\mathbf{x}, \mathbf{y})\}\right].
$$
We can set the damping distribution $d_\xi (\mathbf{x}, \mathbf{y})\propto 1$ for discrete domains and $d_\xi (\mathbf{x}, \mathbf{y}) = d_\mu(\mathbf{x})d_\nu(\mathbf{y})$ for continuous domains.

We have transformed the original optimization problem, but this new formulation now seems even more complicated! But note that if $\lambda_\mu^\*, \lambda_\nu^\*$ are the functions achieving the maximum of $\Psi$, and $\gamma$ is sufficiently small, then $\mathrm{WD}_\gamma(\mu, \nu) \approx \mathbb{E}_\mu[\lambda_\mu^\*(\mathbf{x})] - \mathbb{E}_\nu[\lambda_\nu^\*(\mathbf{y})]$, with equality when $\gamma = 0$.

For our purposes, we generally have $\mathcal{X} = \mathcal{Y}$ and $C(x, x) = 0$ for all $x\in \mathcal{X}$. Now if we let $\gamma = 0$, then one can see that $\lambda_\mu^\*(x) = \lambda_\nu^\*(y) = \lambda^\*(x)$ for all $x\in\mathcal{X}$. In this case, $\mathrm{WD}(\mu, \nu) = \mathbb{E}_\mu[\lambda^\*(x)] - \mathbb{E}_\nu[\lambda^\*(x)]$, which means that $\lambda^\*$ is a function that assigns high scores to regions where $\mu$ has more mass and low scores to those where $\nu$ has more mass. This means that we can interpret $\lambda^\*$ as a function that can distinguish policy embeddings of two policies. So, **if we have $\lambda^\*$**, we can estimate $\mathrm{WD}(\mu, \nu)$ by sampling from $\mu$ and $\nu$.

### Computing $\lambda_\mu^\*$ and $\lambda_\nu^\*$

To make the optimization tractable, the paper uses RBF kernels and approximates them using random Fourier feature maps. As a result, the functions $\lambda$ that are learned have the following form
$$
\lambda(\mathbf x) = (\mathbf{p}^\lambda)^T\phi(\mathbf{x}),
$$
where $\phi$ is a random feature map with $m$ random features and $\mathbf{p}^\lambda \in \mathbb{R}^m$ are the parameters that we optimize. For RBF kernels, $\phi$ is defined as 
$$
\phi(\mathbf{z}) = \frac{1}{\sqrt{m}}\cos(\mathbf{Gz}+\mathbf{b})
$$
for $\mathbf{z} \in \mathbb{R}^d$, where $\mathbf{G} \in \mathbb{R}^{m\times d}$ is a Gaussian with iid entries taken from $\mathcal{N}(0, 1)$ and $\mathbf{b} \in \mathbb{R}^m$ has iid entries taken from $\mathrm{Uniform}(0, 2\pi)$. The $\cos$ function is applies elementwise.

From hereon, the optimization of $\Psi$ over $\lambda$ is synonymous with optimization over $\mathbf{p}^\lambda$. Having said these, we can optimize $\Psi$ by running SGD to find the optimal vectors $\mathbf{p}^{\lambda_\mu}, \mathbf{p}^{\lambda_\nu}$. Given kernels $\kappa, \mathscr{l}$ and a fresh sample $(x_t, y_t) \sim \mu \otimes \nu$, the SGD step w.r.t. the current iterates $\mathbf{p}_{t-1}^\mu, \mathbf{p}_{t-1}^\nu$ satisfies:
$$
F(\mathbf{p}_1, \mathbf{p}_2, x, y) = \exp \left(\frac{\mathbf{p}_1^T \phi_\kappa(x) - \mathbf{p}_2^T \phi_\mathscr{l}(y) - C(x, y)}{\gamma}\right), \\\\
v_t = \frac{\alpha}{\sqrt{t}}\left(\phi_\kappa(x_t), -\phi_\mathscr{l} (y_t)\right)^T \\\\
\begin{pmatrix}
\mathbf{p}_{t+1}^\mu \\\\
\mathbf{p}_{t+1}^\nu
\end{pmatrix} =
\begin{pmatrix}
\mathbf{p}_{t}^\mu \\\\
\mathbf{p}_{t}^\nu
\end{pmatrix} + (1 - F(\mathbf{p}_t^\mu, \mathbf{p}_t^\nu, x_t, y_t))v_t.
$$
($\alpha$ is the learning rate)

In what comes next, these update equations are regarded as the **BEM update step**.

If $\mathbf{p}^\mu_\*, \mathbf{p}^\nu_\*$ are the optimal dual vectors, $\mathbf{p}_\* = (\mathbf{p}^\mu_\*, \mathbf{p}^\nu_\*)$, $(x_1, y_1), \cdots, (x_k, y_k) \stackrel{\mathrm{i.i.d.}}{\sim}\mu\otimes\nu$, $\mathbf{v}_i^{\kappa, \mathscr{l}} = (\phi_\kappa(x_i), -\phi_\mathscr{l} (y_i))$ for all $i$, and $\hat{\mathbb{E}}$ denotes the empirical expectation over the $k$ samples, then we can get an estimate of $\mathrm{WD}_\gamma (\mu, \nu)$ as
$$
\widehat{\mathrm{WD}}_\gamma(\mu, \nu) = \hat{\mathbb{E}} \left[\left<\mathbf{p}_\*, \mathbf{v}_i^{\kappa, \mathscr{l}} \right> - \frac{F(\mathbf{p}^\mu_\*, \mathbf{p}^\nu_\*, x_i, y_i)}{\gamma}\right]
$$
To put things into perspective, if $\pi_1, \pi_2$ are two policies and $\Phi$ is a behavioral embedding map, we can write
$$
\mathrm{WD}_\gamma (\mathbb{P}_{\pi_1}^\Phi, \mathbb{P}_{\pi_2}^\Phi) \approx \mathbb{E}_{\tau\sim\mathbb{P}_{\pi_1}}[\lambda_1^\*(\Phi(\tau))] - \mathbb{E}_{\tau\sim\mathbb{P}_{\pi_2}}[\lambda_2^\*(\Phi(\tau))]
$$
with $\lambda_1^\*, \lambda_2^\*$ being the optimal dual functions. The maps $s_i := \lambda_i^\* \circ \Phi : \Gamma \to \mathbb{R}$ define score functions over the space of trajectories, and if $\gamma$ is close to zero, they assign higher scores tot trajectories from $\pi_i$ whose behavior embedding is common under $\pi_i$ but uncommon under $\pi_{j\neq i}$.

## Algorithms

### Random Features Wasserstein SGD (Algorithm 1)

1. **Input:** kernels $\kappa, \mathscr{l}$ over $\mathcal{X}, \mathcal{Y}$ respectively, with corresponding random feature maps $\phi_\kappa, \phi_\mathscr{l}$, smoothing parameter $\gamma$, step size $\alpha$, number of optimization rounds $M$, initial dual vectors $\mathbf{p}_0^\mu, \mathbf{p}_0^\nu$.
2. for $t = 0, \cdots, M$ :
   1. Sample $(x_t, y_t) \sim \mu \otimes \nu$.
   2. Update $\begin{pmatrix}\mathbf{p}_{t}^\mu \\\\\\mathbf{p}_{t}^\nu\end{pmatrix}$ using **BEM update step**.
3. **Return:** $\mathbf{p}_M^\mu, \mathbf{p}_M^\nu$.

**How does this work in practice?** Let's say we have a base policy $b$ and another policy $\pi$ and we want to estimate $\mathrm{WD}_\gamma (\mathbb{P}_b^\Phi, \mathbb{P}_\pi^\Phi)$. Note that these can be random variables; for instance, $b$ might be chosen uniformly from a set of policies $B = \{b_1, \cdots, b_n\}$. First, we need to choose a behavior embedding map $\Phi: \Gamma \to \mathbb{R}^d$ that maps trajectories into $d$ dimensional vectors. Then, we must initialize random feature maps $\phi_b, \phi_\pi: \mathbb{R}^d \to \mathbb{R}^m$ that map the outputs of $\Phi$ to $\mathbb{R}^m$. From what I understood from the code, we can use a single random feature map for both $b$ and $\pi$. Assuming this is the case, we essentially need to initialize $\phi(\mathbf{z}) = \frac{1}{\sqrt m} \cos (\mathbf{Gz} + \mathbf{b})$. To do so, we sample $\mathbf{G} \in \mathbb{R}^{m\times d}$ from $\mathcal{N}(0, 1)$ and $\mathbf{b} \in \mathbb{R}^m$ from $\mathrm{Uniform}(0, 2\pi)$. Next, we initialize dual parameter vectors $\mathbf{p}^\pi, \mathbf{p}^b$ that will be optimized. Finally, we need to run $\pi$ and $b$ to obtain $k$ sample trajectories. These trajectories are saved in buffers $B_\pi, B_b$. (Actually, we only need the behavior embedding of the trajectories, i.e., $\Phi(\tau)$s). The last step is to run SGD to find optimal parameters $\mathbf{p}_\*^\pi, \mathbf{p}_\*^b$. This is done using the algorithm presented above, where sampling $(x_t, y_t) \sim \mu \otimes\nu$ is analogues to sampling from the buffers.

When all is set and done, we have $\mathbf{p}_\*^\pi, \mathbf{p}_\*^b$ and we can write
$$
\widehat{\mathrm{WD}}_\gamma(\mathbb{P}_b^\Phi, \mathbb{P}_\pi^\Phi) = \frac{1}{k}\sum_{i=1}^{k} \left<\mathbf{p}_\*, \mathbf{v}_i\right> - \frac{F(\mathbf{p}^b_\*, \mathbf{p}^\pi_\*, x_i, y_i)}{\gamma}
$$
where $x_i \in B_b$, $y_i \in B_\pi$ and $F$, $\mathbf{v}_i$ are as were defined in the previous section.

Additionally, we we can get the following score function that assigns high scores to trajectories that are similar to $\pi$ but different from $b$:
$$
s_\pi (\tau) = \lambda_\pi^\* \circ \Phi (\tau) = \frac{1}{\sqrt{m}}(\mathbf{p}_\*^\pi)^T\cos(\mathbf{G}(\Phi(\tau))+ \mathbf{b})
$$
**Implementation detail:** In the released code, $\gamma = 1$ is chosen.

### A Simple Behavior-Guided Algorithm (Algorithm 2)

Now, let's see how we can use the behavior functions to devise a simple algorithm.

Imagine a simple scenario in which we have a policy $b$ for doing some task $T_1$ and we want to find a policy $\pi_\theta$ for solving another task $T_2$ that exhibits similar behavior to $b$. The idea is to use the regular reward for task $T_2$ to optimize $\theta$, but also use the scoring functions that were found in the previous section to encourage trajectories with similar behavior to $b$.

#### Learnable parameters

1. parameters of our policy $\theta_0$.
2. parameters of behavioral test function $\mathbf{p}^b, \mathbf{p}^{\pi_{\theta_0}} \in \mathbb{R}^m$

#### Initializations

1. behavioral embedding map $\Phi:\Gamma \to \mathbb{R}^d$ to encode trajectories
2. random kernels $\mathbf{G} \in \mathbb{R}^{m\times d}$ from normal distribution and $\mathbf{b} \in \mathbb{R}^m$ from uniform distribution
3. buffers $B_b, B_\pi$ for storing trajectories

#### Hyperparameters

1. weighting coefficient $\beta \geqslant 0$ and smoothing coefficient $\gamma > 0$
2. number of iterations $T$ and rollouts per iteration $M$
3. learning rates $\alpha$ for behavioral test functions and $\eta$ for policy optimization

#### Algorithm

1. for $t = 1, \cdots, T$:
   1. Collect trajectories $\{\tau_i^\pi\}_{i=1}^M$ using policy $\pi_{t-1} = \pi_{\theta_{t-1}}$ and add them to the buffer $B_\pi$
   2. Collect trajectories $\{\tau_i^b\}_{i=1}^M$ using policy $b$ and add them to the buffer $B_b$
   3. Update behavioral test functions $\lambda_\pi, \lambda_b$ with *Random Features Wasserstein SGD*
   4. Optimize the policy by taking SGA step $\theta_t = \theta_{t-1} + \eta \hat\nabla_\theta \hat F(\theta_{t-1})$
   5. Clear the buffers

#### Implementation details

1. The objective function $F(\theta)$ that policies perform SGA over is of the form
   $$
   F(\theta) = \mathbb{E}_{\tau_1, \tau_2 \sim \mathbb{P}_{\pi_{t-1}} \otimes \mathbb{P}_b} \left[\hat R(\tau_1, \tau_2)\right]
   $$
   where $\hat R (\tau_1, \tau_2)$ is the surrogate reward defined as 
   $$
   \hat R (\tau_1, \tau_2) = \sum_i \frac{\pi_{\theta}(a_i \mid s_i)}{\pi_{t-1}(a_i \mid s_i)}A^{\pi_{t-1}} (s_i, a_i) - \beta\widehat{\mathrm{WD}}_\gamma (\mathbb{P}_{\pi_\theta}^{\Phi}, \mathbb{P}_b^\Phi).
   $$
   The first summation term is the regular advantage weighted by importance sampling ratios that is used in regular policy optimization algorithms. The second term, involving Wasserstein distance is what encourages behaviors similar to $b$. Using the estimates of WD we can write
   $$
   F(\theta) \approx \mathbb {E}_{\tau \sim \mathbb{P}_{\pi_\theta}}\left[\mathcal{R}(\tau) - \beta \lambda_{\pi_\theta}^\*(\Phi(\tau))\right] + \beta \mathbb{E}_{\tau \sim \mathbb{P}_b} \left[\lambda_b^\*(\Phi(\tau))\right].
   $$
   Consequently,
   $$
   \nabla_\theta F(\theta) \approx \nabla_\theta \mathbb{E}_{\tau\sim \mathbb{P}_{\pi_\theta}}[\mathcal{R}(\tau)] - \beta \nabla_\theta\mathbb{E}_{\tau\sim \mathbb{P}_{\pi_\theta}}[\lambda_{\pi_\theta}^\*(\Phi(\tau))]
   $$
   The first term is the regular policy gradient and *the second term can be considered as a constant reward* that is added to each trajectory based on its similarity to the the behavior of $\pi_\theta$ and its dissimilarity to the behavior of $b$.

   **So, to optimize $F$ all we need to do is to add a reward of $-\beta \lambda_{\pi_\theta}^\*(\Phi(\tau))$ to each transition in the trajectory $\tau$.**

2. Because the base policy $b$ is fixed, we can append trajectories into its buffer $B_b$ without resetting it.

3. In the third step where we update behavioral test function parameters $\mathbf{p}$, we can start the optimization from the previously found parameters which may help speed up the convergence.

4. In the paper, the authors suggest that we first optimize the policy ($F$) and then optimize behavioral test functions ($\lambda$), which is strange to me.

