---
layout: post
title: Variational Inference
date: 2023-03-07 01:19:15
description: A note on variational inference and VAEs.
tags: VI, VAE, Probability
categories: blog
related_posts: true
toc: true
math: true
---

## Preliminaries
It is usually the case that we have a dataset $\mathcal{D} = \\{x_1, \cdots, x_N \\}$ and a parametrized family of distributions $p_\theta (x)$. We would like to find the parameters that best describe the data. This is typically done using maximum likelihood estimation (MLE). In this method, the optimal parameters are those that maximize the log likelihood of the data. Mathematically speaking,

$$
\hat{\theta}_\mathrm{MLE} = \arg\max_\theta \frac{1}{N}\sum_{i=1}^{N}\log p_{\theta}(x_i).
$$

If $p_\theta(x)$ is a simple distribution like a Gaussian for which the likelihood can be written in an analytic form, there are no problems and we can find the MLE using an optimization technique of choice. But in many cases, the data cannot be approximated with a simple distribution like a Gaussian. For instance, consider the distribution of images in dataset like MNIST. We often say that they reside on a **data manifold** that can be very complicated. No simple distribution can capture the complex nature of such data.

**Latent variable models** provide one way of modelling complex data distributions. In a latent variable model we assume that there are a set of hidden variables $\mathbf{z}$ that influence the data generation process. For the MNIST dataset, the digit or the style of handwriting could be considered as latent variables. In the simplest case, we could consider the following latent variable model:

$$
p(x) = \int p(x\mid z)p(z)\mathrm{d}z
$$

But more complicated/structured models can also be considered (They are usually shown as graphical models).

**An Aside:** Inferring latent variables from the a data sample (i.e., computing the posterior $p(z\mid x)$) can also be very useful. In a deep learning setting, they could provide a compact and useful representation of data. For example, in **self-supervised learning** it is common to learn to infer a latent representation from a large unlabeled dataset, and then train a very simple linear classifier on top of it to predict a desired label, using very few labeled samples.

In a latent variable model, the MLE will take the following form:

$$
\hat{\theta}_\mathrm{MLE} = \arg\max_\theta \frac{1}{N}\sum_{i=1}^{N}\log \int p_{\theta}(x_i\mid z)p_\theta(z)\mathrm{d}z.
$$

Analytically computing the above integral is in most cases very difficult or even impossible (There are some exceptions; for instance, see Gaussian mixture models). This is where variational inference comes to rescue. It circumvents the intractability issue allows us to compute the MLE of parameters.

Before we dive deep into the details of VI, it is useful to clearly sort out our assumptions. So in the next sections, we will see which distributions we assume have a simple tractable form and which ones are interactable/hard.

### The Knowns
#### $p(z)$
This distribution, known as the *prior over the latent variables*, is usually assumed to be a simple distribution such as $\mathcal{N}(\mathbf{0}, \mathbf{I})$. We don't even parametrize it (no optimizing it!) and assume that it is a fixed distribution known a priori.
#### $p_\theta(x\mid z)$
This is usually known as the *likelihood*, and it is the probability that our model generates a data point $x$, **given the latent $z$**. We typically assume that this distribution belongs to a simple (parametrized) family of distributions, such as the Gaussians. To give an example, we may choose to represent them as $\mathcal{N}(\mu_\theta(z), \sigma_\theta(z))$.
#### $p_\theta(x, z)$
The *joint distribution* can be obtained using the Bayes rule by simply multiplying the densities of the prior and the likelihood, which are both known, tractable distributions. So we can say

$$
p_\theta (x, z) = p_\theta(x\mid z)p(z).
$$

Note that we usually use a model (e.g., a probabilistic graphical model) of this joint distribution that tells us about the relation between the latent $z$ and the observed variable $x$.
### The Unknowns
#### $p_\theta(x)$
Also known as the *marginal* or the *evidence*, this is the probability that our model has generated a data point $x$. It is called so because it can be obtained from marginalizing the joint distribution:

$$
p_\theta (x) = \int p(x, z)\mathrm{d}z = \int p_\theta(x\mid z)p(z)\mathrm{d}z.
$$

Notice that although the joint distribution is known, integrating it with respect to $z$ is interactable in all but very few cases. So effectively, the marginal distribution is not available to us.
#### $p_\theta(z\mid x)$
The probability of the latent given a data point is called the posterior distribution over the latent variables (or simply the *posterior*). We can think of it as our belief about the latent given an observation $x$. Using the Bayes rule we can write it as

$$
p_\theta(z\mid x) = \frac{p_\theta(x, z)}{p_\theta(x)}.
$$

Although the nominator is known, due to the intractability of the marginal in the denominator, we can't compute the posterior either.
### The Goal
In VI, we want to approximate the marginal and the posterior. Having tractable approximates is very useful. For instance, we could use them to find good parameters for our model by approximating $\hat{\theta}_\mathrm{MLE}$.
## Evidence Lower Bound
### Overview
Consider the maximum likelihood estimation of parameters $\theta$ for modelling a dataset $\mathcal{D} = \\{ x_1, \cdots, x_N \\}$ of samples using a latent variable model $p_\theta(x)$. But, as we mentioned earlier, the marginal distribution is interactable, so we can't even directly compute the likelihoods $\log p_\theta(x_i)$ of the samples in our dataset given a particular set of parameters. To circumvent this issue we will try to approximate the posterior $p_\theta(z\mid x_i)$ with a simple (tractable) distribution $q_i(z)$ which can be, for instance, $\mathcal{N}(\mu_i, \sigma_i)$. We will then use $q_i$'s to derive a tractable lower bound (ELBO) on $\log p_\theta(x_i)$, which is what we actually want. Finally, instead of maximizing $\log p_\theta(x_i)$s, we will maximize these tractable lower bounds. If the bounds are sufficiently tight, then pushing up the lower bounds will also push up the actual likelihoods. Furthermore, we will see that as $q_i$ approximates the posterior $p_\theta(z\mid x_i)$ better (in the sense of KL-divergence), the ELBO bound will get tighter; to the extent that if $q_i = p_\theta(z\mid x_i)$, then ELBO will be exact.

There is one more trick that we will use. Instead of considering a different $q_i$ to approximate the posterior given each sample in the dataset, we will use a parametrized family $q_\phi(z\mid x)$ to represent all of them at once. We can think of it as using a neural net with parameters $\phi$ that will output the mean and standard deviation of the approximate posterior given each sample, so that $q_i = q_\phi(x_i) = \mathcal{N}(\mu_\phi(x_i), \sigma_\phi(x_i))$. This trick, known as **amortized variational inference**, will help us optimize  $q_i$s (to tighten our bound) in a much more efficient manner.
### Derivation
#### Approach 1: Bounding the Log-Likelihood
As we said in the overview, we ultimately want to give a lower bound for $\log p(x_i)$ (the subscript of $\theta$ is dropped for convenience). So. let's start from this quantity and try to introduce $q_i(z)$ along the way.

$$
\begin{align*}
\log p(x_i) &= \log \int_z p(x_i\mid z)p(z) \\
&= \log \int_z p(x_i\mid z)p(z)\frac{q_i(z)}{q_i(z)} \\
&= \log \mathbb{E}_{z\sim q_i} \left[\frac{p(x_i\mid z)p(z)}{q_i(z)}\right] &&\text{Definition of expected value} \\
&\geqslant \mathbb{E}_{z\sim q_i}\left[\log\frac{p(x_i\mid z)p(z)}{q_i(z)}\right] &&\text{Jensen inequality} \\
&= \mathbb{E}_{z\sim q_i}\left[\log p(x_i\mid z) +\log p(z) \right] - \mathbb{E}_{z\sim q_i}[q_i(z)] \\
&= \mathbb{E}_{z\sim q_i}\left[\log p(x_i\mid z) +\log p(z) \right] + \mathcal{H}(q_i)
\end{align*}
$$

($\mathcal{H}(\cdot)$ denotes the entropy)

Take a look at the last expression in the derivation above. Every term used there is tractable: $p(x_i\mid z)$ is the likelihood given the latent, $p(z)$ is the prior over the latent, and $q_i$ is any distribution of our choice. This expression is called the **evidence lower bound (ELBO)** and is usually denoted as $\mathcal{L}_i(p, q_i)$.

> Note that we can get a Monte Carlo estimate of the expected value terms by sampling different $z$'s from $q_i$ and averaging the respective $\log p(x_i\mid z)p(z)$ values.

So, we defined the ELBO as

$$
\mathcal{L}_i(p, q_i) = \mathbb{E}_{z\sim q_i}\left[\log p(x_i\mid z) +\log p(z) \right] + \mathcal{H}(q_i),
$$

and established that

$$
\log p(x_i) \geqslant \mathcal{L}_i(p, q_i).
$$

This result, in itself, lets us use any distribution $q_i(z)$ to get a tractable lower bound for $\log p(x_i)$. But how tight is this bound? and how should we choose $q_i$? The basic intuition is that $q_i(z)$ should approximate the posterior $p(z\mid x_i)$, in the sense of KL-divergence. So, the ideal $q_i$ would minimize $D_\mathrm{KL} (q_i(z)\| p(z\mid x_i))$. This intuition will be made rigorous in the next section, where we take a different approach to derive ELBO.

#### Approach 2: KL minimization
Following the intuition given above, let's examine $D_\mathrm{KL} (q_i(z)\|p(z\mid x_i))$ more closely.

$$
\begin{align*}
D_\mathrm{KL} (q_i(z)\|p(z\mid x_i)) &= \mathbb{E}_{z\sim q_i}\left[\log\frac{q_i(z)}{p(z\mid x_i)}\right] \\
&= \mathbb{E}_{z\sim q_i} \left[\log\frac{q_i(z)p(x_i)}{p(x_i, z)}\right] &&\text{Bayes' Rule} \\
&= \mathbb{E}_{z\sim q_i}\left[\log\frac{q_i(z)p(x_i)}{p(x_i\mid z)p(z)}\right] &&\text{Bayes' Rule} \\
&= \mathbb{E}_{z\sim q_i}[\underbrace{-\log p(x_i\mid z) - \log p(z) + \log q_i(z)}_{-\mathcal{L_i(p, q_i)}} + \underbrace{\log p(x_i)}_{\text{constant}}] \\
&= -\mathcal{L}_i(p, q_i) + \log p(x_i) \\
&\implies \log p(x_i) = \mathcal{L}_i(p, q_i) + D_\mathrm{KL}(q_i(z) \| p(z\mid x_i)).
\end{align*}
$$

Therefore $D_\mathrm{KL} (q_i(z) \| p(z \mid x_i))$ is actually the approximation error when we use $\mathcal{L}_i(p, q_i)$ instead of $\log p(x_i)$.

This suggests a natural optimization scheme to push up the value of $p_\theta (x_i)$: we can alternate between maximizing 
$$\mathcal{L}_{i} (p_\theta, q_i)$$
w.r.t. $q_i$ to tighten the bound (which is equivalent to minimizing the KL), and maximizing
$$\mathcal{L}_{i} (p_\theta, q_i)$$
w.r.t. $\theta$ to push up the lower bound.

<div class="text-center">
  <img src="/assets/img/blog/vi1.png" class="img-fluid" style="max-width: 70%;" />
</div>

To sum it all up, take a look at the algorithm below.
+ for each $x_i$ (or minibatch)
	+ calculate $\nabla_\theta\mathcal{L}_{i} (p, q_i)$ by
		+ sample $z \sim q_i$
		+ let $$\nabla_{\theta} \mathcal{L}_{i} (p, q_{i}) \approx \nabla_{\theta} \log p_{\theta} (x_i \mid z)$$
	+ let $\theta \leftarrow \theta + \alpha \nabla_\theta \mathcal{L}_i(p, q_i)$
	+ update $q_i$ to tighten the bound by
		+ let $q_i \leftarrow \arg \max_{q_i} \mathcal{L}_i(p, q_i)$

This algorithm could have been fully practical if not for the last step. We have not specified how one should update $q_i$ to maximize $\mathrm{ELBO}$ (or to minimize the KL). In the special case when $q_i \sim \mathcal{N} (\mu_{i}, \sigma_{i})$, we can analytically compute
$$\nabla_{\mu_{i}} \mathcal{L}_{i} (p, q_{i})$$
and
$$\nabla_{\sigma_{i}} \mathcal{L}_{i} (p, q_{i})$$
and use them to update parameters (here, mean and variance) of $$q_i$$ using gradient ascent. But even this requires us to store one set of parameters for each $q_i$, resulting in a total of $$N \times (\mid \mu_{z} \mid  + \mid \sigma_{z} \mid )$$. This means that the number of parameters grows with the size of the dataset, which is impractical. In the next section, we will maintain exactly how $q_i$'s should be updated and use amortized inference to manage the number of parameters.

#### Amortized VI
The idea of amortized variational inference is to use a network parametrized by $\phi$ to represent the approximate posterior for all data points. This would break the dependence of the number of parameters to the size of the dataset. This network, denoted by $q_\phi(z\mid x)$ would take as input a data point $x$ and output the distribution $q_i(z)$. A common choice, used in VAEs is to have

$$
q_\phi(z\mid x) = \mathcal{N} (\mu_\phi(x), \sigma_\phi(x)).
$$

Using amortized VI would changing the last step of the algorithm presented above to $\phi \leftarrow \arg\max_\phi \mathcal{L}(p_\theta(x_i\mid z), q_\phi(z\mid x_i))$. Similar to how we updated $\theta$, we use gradient ascent to optimize $\phi$. For this, we would need to compute

$$
\nabla_\phi \mathcal{L}(p_\theta(x_i\mid z), q_\phi(z\mid x_i)).
$$

The final missing piece to complete our algorithm is to calculate this gradient. So let's examine it more closely.

$$
\begin{align*}
&\nabla_\phi \mathcal{L}(p_\theta(x_i\mid z), q_\phi(z\mid x_i)) \\
&= \nabla_\phi \underbrace{\mathbb{E}_{z\sim q_\phi(z\mid x_i)}\left[\log p_\theta(x_i\mid z) + \log p(z)\right]}_{J(\phi)} + \nabla_\phi\underbrace{\mathcal{H}(q_\phi(z\mid x_i))}_{\text{entropy of Gaussian}}.
\end{align*}
$$

Notice that the second term is just the gradient of the entropy of a Gaussian distribution which has a closed analytical form (If we are using automatic differentiation tools, computing this gradient would be very easy). Therefore we focus on the first term. To compute the first term we can use **policy gradient theorem** which would yield

$$
\nabla_\phi J(\phi) = \mathbb{E}_{z\sim q_\phi(z\mid x_i)} \left[\left(\log p_\theta(x_i\mid z) + \log p(z)\right)\nabla_\phi q_\phi(z\mid x_i)\right].
$$

The expected value on the right hand side can be estimated by sampling from $q_\phi(z\mid x_i)$, which is easy to do as it is a normal distribution with mean $\mu_\phi(x_i)$ and standard deviation $\sigma_\phi(x_i)$. So the policy gradient theorem would give us the following estimate of the gradient

$$
\nabla_\phi J(\phi) \approx \frac{1}{M}\sum_{j=1}^{M} \left(\log p_\theta(x_i\mid z_j) + \log p(z_j)\right)\nabla_\phi q_\phi(z\mid x_i)
$$

where $z_j$'s are sampled from $q_\phi(z\mid x_i)$. 

The policy gradient estimator is known to have a high variance. For our purposes, a better estimator can be obtained through **reparameterization trick**.  
The main intuition behind reparameterization trick is that we can view a sample $z$ from $q_\phi(z\mid x_i) = \mathcal{N}(\mu_\phi(x_i), \sigma_\phi(x_i))$ as $z = \mu_\phi(x_i) + \varepsilon \sigma_\phi(x_i)$ where $\varepsilon \sim \mathcal{N}(0, 1)$.  
Substituting this, we can rewrite $J(\phi)$ as

$$
J(\phi) = \mathbb{E}_{\varepsilon \sim \mathcal{N}(0, 1)}\left[\log p_\theta(x_i\mid \mu_\phi(x_i) + \varepsilon \sigma_\phi(x_i)) + \log p(\mu_\phi(x_i) + \varepsilon \sigma_\phi(x_i))\right].
$$

Because the distribution over which the expected value is defined does not depend on $\phi$ anymore, we can push the gradient operator inside it and have

$$
\nabla_\phi J(\phi) = \mathbb{E}_{\varepsilon \sim \mathcal{N}(0, 1)}\left[\nabla_\phi\log p_\theta(x_i\mid \mu_\phi(x_i) + \varepsilon \sigma_\phi(x_i)) + \nabla_\phi\log p(\mu_\phi(x_i) + \varepsilon \sigma_\phi(x_i))\right].
$$

To get a estimate of the gradient we can sample $\varepsilon_1, \cdots, \varepsilon_M \sim \mathcal{N}(0, 1)$ and write

$$
\nabla_\phi J(\phi) \approx \frac{1}{M}\sum_{j=1}^{M}\nabla_\phi\log p_\theta(x_i\mid \mu_\phi(x_i) + \varepsilon_j \sigma_\phi(x_i)) + \nabla_\phi\log p(\mu_\phi(x_i) + \varepsilon_j \sigma_\phi(x_i)).
$$

This estimator has a much lower variance and even using $M=1$ would give us a good approximation of the gradient.

> When using the reparameterization trick in the VAEs, the ELBO is usually written in the form of the reconstruction loss and a KL term (See the next section). With this formulation, the KL term, $D_\mathrm{KL}(q_\phi(z\mid x_i)\|p(z))$, would be the divergence between two Gaussians and would again have a closed form that we could differentiate. This would simplify $J(\phi)$ as $$J(\phi) = \mathbb{E}_{\varepsilon \sim \mathcal{N}(0, 1)} [\log p_\theta(x_i\mid \mu_\phi(x_i) + \varepsilon\sigma_\phi(x_i))].$$


### Summary
As we saw ELBO, $$\mathcal{L}_{i} (p_{\theta}, q_{\phi})$$ gives a lower bound on the per-sample evidence, $p(x_{i})$.
We can write the ELBO in several different ways

$$
\begin{align*}
\mathcal{L} &= \mathbb{E}_{z\sim q_\phi(z\mid x)} \left[\log\frac{p_\theta(x, z)}{q_\phi(z\mid x)}\right] \\
&= \mathbb{E}_{z\sim q_\phi(z\mid x)} [\log p_\theta(x\mid z) + \log p(z)] + \mathcal{H}(q_\phi(z\mid x)) \\
&= \log p_\theta(x) - D_\mathrm{KL}(q_\phi(z\mid x) \| p_\theta(z\mid x)) && \text{Evidence minus } \mathbf{posterior} \text{ KL} \\
&= \mathbb{E}_{z \sim q_\phi(z\mid x)}[\log p_\theta(x, z)] + \mathcal{H}(q_\phi(z\mid x)) && \text{Avg negative energy plus entropy} \\
&= \mathbb{E}_{z\sim q_\phi(z\mid x)}[\log p_\theta(x\mid z)] - D_\mathrm{KL}(q_\phi(z\mid x)\|p(z)) && \text{Avg reconstruction minus } \mathbf{\text{prior}} \text{ KL}
\end{align*}
$$

In variational inference, maximizing ELBO with respect to $\phi$ would encourage the encoder $q_\phi$ to be like the posterior $p_\theta (z\mid x)$. Maximizing it with respect to $\theta$ could push up the evidence (used in maximizing the likelihood). To compute the gradient of ELBO with respect to $\theta$ and $\phi$, consider the average reconstruction minus prior KL formulation of it. We have

$$
\begin{align*}
&\nabla_\theta \mathcal{L} (p_\theta, q_\phi) = \mathbb{E}_{z\sim q_\phi(z\mid x)}[\nabla_\theta\log p_\theta(x\mid z)] \\
&\implies \nabla_\theta \mathcal{L} (p_\theta, q_\phi) \approx \frac{1}{M}\sum_{j=1}^{M}\nabla_\theta\log p_\theta(x\mid z_j) &&\text{where $z_1, \cdots, z_M \sim q_\phi(z\mid x)$}
\end{align*}
$$

To compute $\nabla_\phi \mathcal{L}$, taking the gradient with respect to the KL term is easy as there is a closed form expression for the KL between Gaussians. As for the gradient with respect to the first term we can either use the policy gradient theorem:

$$
\begin{align*}
&\nabla_\phi \mathbb{E}_{z\sim q_\phi(z\mid x)}[\log p_\theta(x\mid z)] = \mathbb{E}_{z\sim q_\phi(z\mid x)}[\log p_\theta(x\mid z) \nabla_\phi \log q_\phi(z\mid x)] \\
&\nabla_\phi \mathbb{E}_{z\sim q_\phi(z\mid x)}[\log p_\theta(x\mid z)] \approx \frac{1}{M} \sum_{j=1}^{M}\log p_\theta(x\mid z_j) \nabla_\phi \log q_\phi(z_j\mid x) && \text{where $z_1, \cdots, z_M \sim q_\phi(z\mid x)$}
\end{align*}
$$

or the reparameterization trick:

$$
\begin{align*}
&\nabla_\phi \mathbb{E}_{z\sim q_\phi(z\mid x)}[\log p_\theta(x\mid z)] = \mathbb{E}_{\varepsilon\sim \mathcal{N}(0, 1)}[\nabla_\phi\log p_\theta(x\mid \mu_\phi(x) + \varepsilon\sigma_\phi(x))] \\
&\nabla_\phi \mathbb{E}_{z\sim q_\phi(z\mid x)}[\log p_\theta(x\mid z)] \approx \frac{1}{M} \sum_{j=1}^{M} \nabla_\phi\log p_\theta(x\mid \mu_\phi(x) + \varepsilon_j\sigma_\phi(x)) && \text{where $\varepsilon_1, \cdots, \varepsilon_M \sim \mathcal{N}(0, 1)$}
\end{align*}
$$

## Variational Autoencoder
<div class="text-center">
  <img src="/assets/img/blog/vae.png" class="img-fluid" style="max-width: 70%;" />
</div>

For a straight forward implementation in PyTorch, check out [this tutorial](https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed).
I have implemented a simple [VAE for MNIST data using JAX and Flax](https://colab.research.google.com/drive/1YiKhOB6FsyBIZvAf1en9nbOHRvvlOwwt?usp=sharing), which I think is a neat implementation that is close to the actual math. (though admittedly, it could have been a bit cleaner) It can serve as a baseline implementation if you want to design a more complex VAE in JAX.

**Implementation Detail:** I think in implementations we usually consider the conditional to have unit variance: $p_\theta(x\mid z) = \mathcal{N}(\mu_\theta(z), I)$ so that the decoder only outputs the reconstructed version of $x$, which would effectively result in a MSE loss between $x$, $\hat x$. Also the KL does not need $\sigma_\theta$, so we don't really lose anything by setting it to a constant. On the other hand, if we let a learnable parameter $\sigma_x$ control the variance, it will always go down as the model trains (makes sense, the smaller $\sigma_x$ is, the higher the log likelihood could get). This would mean that as training progresses, we would get larger values for the likelihood. Effectively, this is equivalent to gradually increasing the contribution of the likelihood term (vs the KL term) so that the model focuses more on reconstruction rather than being close to prior. In my experiments, this helped the VAE more accurately reconstruct the images, at the cost of a very high KL divergence. I have to examine this more, but I suspect this could actually result in less useful latents due to overfitting.

## Other Resources
Here are a few pointers to some material you can use to study variational inference.

+ [Lecture 18 of Berkeley's Deep RL course](https://youtu.be/UTMpM4orS30): Sergey Levine is a great teacher and the material he presents in this ~2 hour lecture covers most of what I discussed above with an RL flavor. I highly recommend this.
+ [An Introduction to Variational Autoencoder](https://arxiv.org/abs/1906.02691) By Max Welling and Diederik Kingma is a great reference on VAEs. Additionally, it has a full chapter on going beyond Gaussian posteriors which is very interesting.
+ [ELBO surgery: yet another way to carve up the variational evidence lower bound](http://approximateinference.org/accepted/HoffmanJohnson2016.pdf) This is a brief (and very nicely written!) paper that examines multiple different ways of writing the evidence lower bound.
+ [Variational Inference: A Review for Statisticians](https://arxiv.org/abs/1601.00670) This is a more theoretical note about variational inference. David Blei is great at explaining VI and I also recommend watching some of his talks online.