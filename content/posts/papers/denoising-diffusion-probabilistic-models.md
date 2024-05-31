---
title: "Notes on Denoising Diffusion Probabilistic Models"
date: 2023-12-19T17:51:09+01:00
draft: false
math: true

# cover:
#     image: "<image path/url>"
#     # can also paste direct link from external site
#     # ex. https://i.ibb.co/K0HVPBd/paper-mod-profilemode.png
#     alt: "<alt text>"
#     caption: "<text>"
#     relative: false # To use relative path for cover image, used in hugo Page-bundles

tags: ["DPM", "Diffusion Probabilistic Models", "AI", "Research", "Paper", "Machine Learning", "Deep Learning", "Probabilistic Models", "Probabilistic Graphical Models", "PGM", "Directed Models", "Diffusion", "Markov Chain", "Statistical Physics", "Diffusion Models", "DDPM", "Denoising", "Denoising Diffusion Probabilistic Models"]

ShowToc: true
---

***Disclaimer:*** *This is part of my notes on AI research papers. I do this to learn and communicate what I understand. Feel free to comment if you have any suggestion, that would be very much appreciated.*

The following post is a summary of the paper [Denoising Diffusion Probabilistic Models](#1) by [Jonathan Ho](https://arxiv.org/search/cs?searchtype=author&query=Ho,+J), [Ajay Jain](https://arxiv.org/search/cs?searchtype=author&query=Jain,+A) and [Pieter Abbeel](https://arxiv.org/search/cs?searchtype=author&query=Abbeel,+P), from [University of California, Berkeley](https://www.berkeley.edu/). The paper was published in 2020 and is a follow up on [**Diffusion Probabilistic Models**](#2) (DPM). Authors show that a simple reparameterization trick allows Diffusion Models to efficiently sample high quality images and close the gap with [Denoising Score Matching](#3) models. 

As already seen in the [previous post](https://alex-pv01.github.io/posts/papers/deep-unsupervised-learning-using-nonequilibrium-thermodynamics/), a DPM (of Diffusion Model) is a parameterized Markov chain trained using variational inference to produce samples matching the data after a finite time. It consists of a forward trajectory that adds small perturbations to the complex data distribution until reaching a tractable distribution, and a reverse trajectory where a function learns the reverse perturbation step. Once the model is trained we obtain an approximation of the data distribution where we can sample. 

The method proposed by the authors is build on top of Gaussian DPM including the following modifications: 

#### Forward trajectory
They ignore the fact that $\beta\_t$ are learnable and fix them to constants. Thus, the posterior $q$ has no learnable parameters.

#### Reverse trajectory
In the case of the reverse distribution $p(\textbf{x}^{(t-1)} | \textbf{x}^{(t)}) = \mathcal{N}(\textbf{x}^{(t-1)};\textbf{f}\_\mu(\textbf{x}^{(t)},t),\textbf{f}\_\Sigma(\textbf{x}^{(t)},t))$:

1. First, they set $\textbf{f}\_\Sigma(\textbf{x}^{(t)},t)) = \sigma\_t^2\textbf{I}$. Experimentally they use both $\sigma\_t^2 = \beta\_t$ and $\sigma\_t^2 = \tilde{\beta}\_t = \frac{1 - \bar{\alpha}\_{t-1}}{1 - \bar{\alpha}\_t} \beta\_t$, with $\alpha\_t := 1 - \beta\_t$ and $\bar{\alpha}\_t := \prod\_{s=1}^t \alpha\_s$. 

2. Second, they propose the reparameterization 
$$ \textbf{x}\_{t} (\textbf{x}\_{0}, \epsilon) = \sqrt{\bar{\alpha}\_t} \textbf{x}\_0 + \sqrt{1 - \bar{\alpha}\_{t}} \epsilon $$
where, $\epsilon \sim \mathcal{N}(\textbf{0}, \textbf{I})$. This leads to parameterizing the mean as
$$ \textbf{f}\_\mu(\textbf{x}^{(t)},t) = \frac{1}{\sqrt{\alpha\_t}}\Big( \textbf{x}^{(ts)} - \frac{\beta\_t}{\sqrt{1-\bar{\alpha}\_t}} \textbf{f}\_{\epsilon}(\textbf{x}^{(t)}, t) \Big). $$

With these, each of the summands of the bound of the log likelihood can be simplified to 
$$ L\_{\text{simple}} := \mathbb{E}\_{t, \textbf{x}^{(0)}, \epsilon}\Big[ || \epsilon - \textbf{f}\_{\epsilon}(\sqrt{\bar{\alpha}\_t} \textbf{x}^{(0)} + \sqrt{1 - \bar{\alpha}\_{t}} \epsilon, t) ||^2 \Big] $$

This resembles denoising score matching over multiple scales indexed by $t$ [[3]](#3), and further simplifies the diffusion model's variational bound.




### References
<a id="1">[1]</a> Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).

<a id="2">[2]</a> Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., & Ganguli, S. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. [arXiv:1503.03585](https://arxiv.org/abs/1503.03585).

<a id="3">[3]</a> Song, Y., & Ermon, S. (2019). *Generative Modeling by Estimating Gradients of the Data Distribution*. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600v3).