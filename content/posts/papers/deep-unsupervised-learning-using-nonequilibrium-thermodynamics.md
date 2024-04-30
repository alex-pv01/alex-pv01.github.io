---
title: "Notes on Deep Unsupervised Learning Using Nonequilibrium Thermodynamics"
date: 2023-12-18T18:06:41+01:00
draft: false
math: true

# cover:
#     image: "<image path/url>"
#     # can also paste direct link from external site
#     # ex. https://i.ibb.co/K0HVPBd/paper-mod-profilemode.png
#     alt: "<alt text>"
#     caption: "<text>"
#     relative: false # To use relative path for cover image, used in hugo Page-bundles

tags: ["DPM", "Diffusion Probabilistic Models", "AI", "Research", "Paper", "Machine Learning", "Deep Learning", "Probabilistic Models", "Probabilistic Graphical Models", "PGM", "Directed Models", "Thermodynamics", "Diffusion", "Markov Chain", "Statistical Physics"]

ShowToc: true
---

***Disclaimer:*** *This is part of my notes on AI research papers. I do this to learn and communicate what I understand. Feel free to comment if you have any suggestion, that would be very much appreciated.*

The following post is a summary of the paper [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](#1) by [Jascha Sohl-Dickstein](https://arxiv.org/search/cs?searchtype=author&query=Sohl-Dickstein,+J), [Eric A. Weiss](https://arxiv.org/search/cs?searchtype=author&query=Weiss,+E+A), [Niru Maheswaranathan](https://arxiv.org/search/cs?searchtype=author&query=Maheswaranathan,+N) and [Surya Ganguli](https://arxiv.org/search/cs?searchtype=author&query=Ganguli,+S), from [Stanford University](https://www.stanford.edu/) and [University of California, Berkeley](https://www.berkeley.edu/). The paper was published in 2015 and it is the first one to introduce the concept of **Diffusion Probabilistic Models (DPMs)** or, in short, **Diffusion Models**. 

By that time, probabilistic models suffered from a conflicting trade-off between *tractability* and *flexibility*. 
- Tractable models are those that can be analytically evaluated and easily fit to data. However, they are not flexible enough to capture complex distributions.
- Flexible models can fit the structure of any distribution but they are hard to train, evaluate and sample from, often requiring a costly Monte Carlo process.

They proposed **Diffusion Probabilistic Models** as a new family of probabilistic models that tackle this dichotomy and claim  that DPMs allow:
1. Extreme flexibility in model structure.
2. Exact sampling.
3. Easy multiplication with other distributions, suitable for conditioning to other priors.
4. Cheap evaluation of the log likelihood and probability of individual states.

## Diffusion Probabilistic Models
<!-- The method is based on an idea from non-equilibrium statistical physics ([Jarzynski, 1997]()(see https://www.youtube.com/watch?v=LXcQx6Bu3OQ)) and sequential Monte Carlo ([Neal 2001]) -->
The method is based on the idea of using a Markov chain to *gradualy* transform one distribution into another. In particular, they build a Markov chain which converts a simple known distribution, for instance a Gaussian, into the target complex data distribution using a *diffusion process*. Thus, learning involves estimating small perturbations within this process, which are more tractable than explicitly modeling the target distribution with a single intricate function.

<!-- ----- add image here -----
caption: animation of a diffusion process, from a complex distribution to a simple one. -->


### Algorithm
The algorithm is defined as a two-step process, the **forward trajectory** and the **reverse trajectory**. Let us assume that the distribution of our data is $q(\textbf{x}^{(0)})$. 

#### Forward trajectory
The algorithm will gradually transform $q(\textbf{x}^{(0)})$ into a simple (analytically tractable) distribution $\pi(\textbf{y})$ by repeatingly applying a *Markov diffusion kernel* $T\_\pi(\textbf{y}|\textbf{y}';\beta)$, where $\beta$ is the *diffusion rate*. Thus,
\begin{align}
\pi(\textbf{y}) &= \int \pi(\textbf{y}') T_{\pi}(\textbf{y}|\textbf{y}';\beta) d\textbf{y}' \\\
q(\textbf{x}^{(t)}|\textbf{x}^{(t-1)}) &= T_{\pi}(\textbf{x}^{(t)}|\textbf{x}^{(t-1)};\beta)
\end{align}

Therefore, the forward trajectory starting at $\textbf{x}^{(0)}$ and performing $T$ steps of diffusion is:
$$q(\textbf{x}^{(0\cdots T)} = q(\textbf{x}^{(0)})\prod_{t=1}^T T_{\pi}(\textbf{x}^{(t)}|\textbf{x}^{(t-1)};\beta) = q(\textbf{x}^{(0)}) \prod_{t=1}^T q(\textbf{x}^{(t)}|\textbf{x}^{(t-1)})$$

For the sake of tractability, authors choose the Markov diffusion kernel to be either a Gaussian distribution or a Binomial distribution, leading to **Gaussian diffusion** (3) and **Binomial diffusion** (4), respectively:

\begin{align}
q(\textbf{x}^{(t)}|\textbf{x}^{(t-1)}) &= \mathcal{N}(\textbf{x}^{(t)};\textbf{x}^{(t-1)}\sqrt{1-\beta\_t},\beta\_t\textbf{I})
\\\
q(\textbf{x}^{(t)}|\textbf{x}^{(t-1)}) &= \mathcal{B}(\textbf{x}^{(t)};\textbf{x}^{(t-1)}(1-\beta\_t)+0.5\beta\_t)
\end{align}

Arriving at the final distribution $\pi(\textbf{x}^{(T)})$, which is either a simple Gaussian $\mathcal{N}(\textbf{x}^{(T)};\textbf{0}, \textbf{I})$, or a Binomial distribution $\mathcal{B}(\textbf{x}^{(T)};0.5)$.

#### Reverse trajectory

The reverse trajectory is the same as the forward but in the opposite direction. It starts at the simple distribution and gradually transforms it into the target distribution: 

\begin{align}
p(\textbf{x}^{(T)}) &= \pi(\textbf{x}^{(T)})
\\\
p(\textbf{x}^{(0\cdots T)}) &= p(\textbf{x}^{(T)})\prod_{t=1}^T p(\textbf{x}^{(t)}|\textbf{x}^{(t-1)})
\end{align}

For both Gaussian and Binomial diffusion, when $\beta\_t$ are *small enough*, the reverse trajectory has the same functional form as the forward process [[2]](#2). The longer the trajectory, the smaller the step sizes can be made. Thus, $p(\textbf{x}^{(t-1)} | \textbf{x}^{(t)})$ will also be a **Gaussian** (7) or **Binomial** (8) distribution:

\begin{align}
p(\textbf{x}^{(t-1)} | \textbf{x}^{(t)}) &= \mathcal{N}(\textbf{x}^{(t-1)};\textbf{f}\_\mu(\textbf{x}^{(t)},t),\textbf{f}\_\Sigma(\textbf{x}^{(t)},t)),
\\\
p(\textbf{x}^{(t-1)} | \textbf{x}^{(t)}) &= \mathcal{B}(\textbf{x}^{(t-1)};\textbf{f}\_b(\textbf{x}^{(t)},t)).
\end{align}

Where $\textbf{f}\_\mu(\textbf{x}^{(t)},t)$ and $\textbf{f}\_\Sigma(\textbf{x}^{(t)},t)$ are functions defining the mean and variance of the reverse Markov transitions for the Gaussian case, and $\textbf{f}\_b(\textbf{x}^{(t)},t)$ is the function providing the bit flip probability in the Binomial case. Authors use *Multilayer Perceptrons (MLPs)* to define these functions.


#### Model Probability
The reverse process lead to the data generative model 

\begin{align} 
p(\textbf{x}^{(0)}) &= \int d\textbf{x}^{(1\cdots T)} p(\textbf{x}^{(0\cdots T)}) 
\\\
&= \int d\textbf{x}^{(1\cdots T)} p(\textbf{x}^{(0\cdots T)}) \frac{q(\textbf{x}^{(1\cdots T)}|\textbf{x}^{(0)})}{q(\textbf{x}^{(1\cdots T)}|\textbf{x}^{(0)})}
\\\
&= \int d\textbf{x}^{(1\cdots T)} q(\textbf{x}^{(1\cdots T)}|\textbf{x}^{(0)}) \frac{p(\textbf{x}^{(0\cdots T)})}{q(\textbf{x}^{(1\cdots T)}|\textbf{x}^{(0)})}
\\\
&= \int d\textbf{x}^{(1\cdots T)} q(\textbf{x}^{(1\cdots T)}|\textbf{x}^{(0)}) \cdot p(\textbf{x}^{(T)})\prod_{t=1}^T \frac{p(\textbf{x}^{(t-1)}|\textbf{x}^{(t)})}{q(\textbf{x}^{(1\cdots T)}|\textbf{x}^{(0)})}
\end{align}

This can be evaluated rapidly by averaging over samples from the forward trajectory $q(\textbf{x}^{(1\cdots T)}|\textbf{x}^{(0)})$. 


#### Training
Training consists on maximizing the model log likelihood, 

\begin{align}
L &= \int d\textbf{x}^{(0)} q(\textbf{x}^{(0)})\log p(\textbf{x}^{(0)})
\\\
&= \int d\textbf{x}^{(0)} q(\textbf{x}^{(0)}) \cdot \log \Bigg\[\int d\textbf{x}^{(1\cdots T)} q(\textbf{x}^{(1\cdots T)}|\textbf{x}^{(0)}) \cdot p(\textbf{x}^{(T)})\prod_{t=1}^T \frac{p(\textbf{x}^{(t-1)}|\textbf{x}^{(t)})}{q(\textbf{x}^{(1\cdots T)}|\textbf{x}^{(0)})}\Bigg\]
\end{align}



### References
<a id="1">[1]</a> Sohl-Dickstein, J., Weiss, E. A., Maheswaranathan, N., & Ganguli, S. (2015). *Deep Unsupervised Learning using Nonequilibrium Thermodynamics*. [arXiv:1503.03585](https://arxiv.org/abs/1503.03585).

<a id="2">[2]</a> Feller, W. (1949). *On the Theory of Stochastic Processes, with Particular Reference to Applications*. DOI: [10.1007/978-3-319-16859-3_42](https://doi.org/10.1007/978-3-319-16859-3_42)
