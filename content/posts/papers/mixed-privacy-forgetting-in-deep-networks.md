---
title: "Notes on Mixed-Privacy Forgetting in Deep Networks"
date: 2024-04-09T10:13:13+02:00
draft: false
math: true

cover:
    image: "<image path/url>"
    # can also paste direct link from external site
    # ex. https://i.ibb.co/K0HVPBd/paper-mod-profilemode.png
    alt: "<alt text>"
    caption: "<text>"
    relative: false # To use relative path for cover image, used in hugo Page-bundles

tags: [Mixed-Privacy Forgetting, DNN, AI, Research, Paper, Machine Unlearning]

ShowToc: false
---

***Disclaimer:*** *This is part of my notes on AI research papers. I do this to learn and communicate what I understand. Feel free to comment if you have any suggestion, that would be very much appreciated.*

The following post is a comment on the paper [Mixed-Privacy Forgetting in Deep Networks](#1) by [Aditya Golatkar](https://arxiv.org/search/cs?searchtype=author&query=Golatkar,+A), [Alessandro Achille](https://arxiv.org/search/cs?searchtype=author&query=Achille,+A), [Avinash Ravichandran](https://arxiv.org/search/cs?searchtype=author&query=Ravichandran,+A), [Marzia Polito](https://arxiv.org/search/cs?searchtype=author&query=Polito,+M), and [Stefano Soatto](https://arxiv.org/search/cs?searchtype=author&query=Soatto,+S).

Golatkar et. al. introduce a novel method for **forgetting** in a mixed-privacy setting, where a **core** subset of the training samples will not be forgotten. Their method allow efficient removal of all non-core (a.k.a. **user**) data by simply setting to zero a subset of the weights of the model, with minimal loss in performance. To do so, they introduce Mixed-Linear Forgetting (ML-Forgetting), which they claim to be the first algorithm to achieve forgettiing for deep networks trained on large-scale computer vision problems without compromising the accuracy. 

### Contributions
1. Introduce the problem of mixed-privacy forgetting in deep networks.
2. Propose ML-Forgetting, that trains a set of non-linear core weights and a set of linear user weights.
3. By-design, ML-Forgetting allows to forget user all data by setting the user weights to zero.
4. First algorithm to achieve forgetting for deep networks trained on large-scale computer vision problems without compromising the accuracy.
5. Can handle multiple sequential forgetting requests, as well as class removal.

### Mixed-Linear Forgetting
The main idea behind the method lies in the concept of **quadratic forgetting**, which comes from forgetting from a linear regression model, that has a quadratic loss function. User data is learned using such loss function, taking advantage of its convexity. First, they introduce the Mixed-Linear model, and then discuss the forgetting mechanism.

#### Mixed-Linear Model
Two separate minimization problems are solved, one for the core data ($\mathcal{D}_c$) and one for the user data ($\mathcal{D}_u$). If $f\_{\textbf{w}}$ is the model with parameters $\textbf{w}$, we have:

$$\textbf{w}\_c^* = \arg\min\_{\textbf{w}\_c}  L\_{\mathcal{D_c}}(f\_{\textbf{w}_c})$$
$$\textbf{w}\_u^* = \arg\min\_{\textbf{w}\_u}  L\_{\mathcal{D_u}}(f\_{\textbf{w}\_c^*+\textbf{w}_u})$$

where $L\_{\mathcal{D}}$ is the loss function for the dataset $\mathcal{D}$. Since the deep network $f\_{\textbf{w}}$ is non-linear, the loss function $L\_{\mathcal{D}\_u}(f\_{\textbf{w}\_c^\*+\textbf{w}\_u})$ can be highly non-convex. In light of [[2]](#2), if $\textbf{w}\_u$ is a small perturbation, we can hope for a linear approximation $f\_{\textbf{w}}$ around $f\_{\textbf{w}\_c^\*}$, to have a similar performance to fine-tuning the entire model. Thus, the Mixed-Linear model is defined as the first-order Taylor expansion:

$$f^{\text{ML}}\_{\textbf{w}\_c^\*+\textbf{w}\_u} (\textbf{x}) = f\_{\textbf{w}\_c^\*}(\textbf{x}) + \nabla\_w f\_{\textbf{w}\_c^\*}(\textbf{x}) \cdot \textbf{w}\_u$$

Furthermore, they use Cross-Entropy loss and Mean Squared Error loss, leading to the following minimization problem:

$$\textbf{w}\_c^* = \arg\min\_{\textbf{w}\_c}  L^{\text{CE}}\_{\mathcal{D_c}}(f\_{\textbf{w}_c})$$
$$\textbf{w}\_u^* = \arg\min\_{\textbf{w}\_u}  L^{\text{MSE}}\_{\mathcal{D_u}}(f^{\text{ML}}\_{\textbf{w}\_c^*+\textbf{w}_u})$$

The MSE loss has the advantage that the weights $\textbf{w}\_u$ are the solution of a quadratic minimization problem, which can be solved in closed form.

#### Forgetting Mechanism
As seen in [[3]](#3) and [[4]](#4), in the case of the quadratic training loss, the optimal forgetting step to delete $\mathcal{D}\_f \subset \mathcal{D}$ is given by: 
$$\textbf{w}\_u \mapsto \textbf{w}\_u - \Delta\textbf{w}\_u + \sigma^2 \epsilon$$
where $\Delta\textbf{w}\_u = H^{-1}\_{\mathcal{D}_r}(\textbf{w}\_c)\nabla\_\textbf{w}L\_{\mathcal{D}_r}(f\_{\textbf{w}\_u})$ is the optimal forgetting step, $H\_{\mathcal{D}_r}(\textbf{w}\_c)$ is the Hessian of the loss function $L\_{\mathcal{D}_r}$, $\mathcal{D}_r=\mathcal{D}-\mathcal{D}_f$ is the retained data, and $\epsilon \sim N(0,I)$ is a random noise vector. As $\Delta\textbf{w}\_u$ is only an approximation of the optimal forgetting step, by adding noise, they can destroy the information that may leak. In practice is not feasible to compute the Hessian, so they use the Jacobian-Vector Product (JVP) instead (see [[2]](#2)).


### Personal considerations:

- Although the method is interesting, I am not sure how practical it is. The theoretical framework heavily relies on the assumption that the perturbation $\textbf{w}\_u$ is small, which may not be the case in practice. I find useful the fact of using core data to train a "foundational" (or core) model and then fine-tune it with user data (actually, this is the trend in SOTA models e.g., for generative AI). However, if the user data is far from being "small enough" and because of the linear approximation, the method may not work as expected.


### References
<a id="1">[1]</a> Golatkar, A., Achille, A., Ravichandran, A., Polito, M., & Soatto, S. (2021). Mixed-Privacy Forgetting in Deep Networks [arXiv:2012.13431](https://arxiv.org/abs/2012.13431).

<a id="2">[2]</a> Mu, F., Liang, Y., & Li, Y. (2020). Gradients as features for deep representation learning. [arXiv:2004.05529](https://arxiv.org/abs/2004.05529)

<a id="3">[3]</a> Guo, C., Goldstein, T., Hannun, A., & Van Der Maaten, L. (2020). Certified data removal from machine learning models [arXiv:911.03030](https://arxiv.org/abs/1911.03030)

<a id="4">[4]</a> Golatkar, A., Achille, A., & Soatto, S. (2020). Eternal sunshine of the spotless net: Selective forgetting in deep networks [arXiv:1911.04933](https://arxiv.org/abs/1911.04933)
