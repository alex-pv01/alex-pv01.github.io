---
title: "Notes on SSSE: Efficiently Erasing Samples From Trained Machine Learning Models"
date: 2024-04-10T12:20:47+02:00
draft: false
math: true

tags: [Machine Unlearning, Data Removal, Data Erasure, Fisher Information Matrix, Convex Models, Machine Learning, Deep Learning, AI, Paper, Research]

ShowToc: false
---

***Disclaimer:*** *This is part of my notes on AI research papers. I do this to learn and communicate what I understand. Feel free to comment if you have any suggestion, that would be very much appreciated.*

The following post is a comment on the paper [SSSE: Efficiently Erasing Samples From Trained Machine Learning Models](#1) by [Alexandra Peste](https://arxiv.org/search/cs?searchtype=author&query=Peste,+A), [Dan Alistarh](https://arxiv.org/search/cs?searchtype=author&query=Alistarh,+D), and [Christoph H. Lampert](https://arxiv.org/search/cs?searchtype=author&query=Lampert,+C+H).

Peste et. al. propose **Single-Step Sample Erasure (SSSE)**, a method to efficiently and effectively erase samples from trained machine learning models. The removal step **only requires access to the data to be erased**, not the entire original training set. 

Although SSSE can be used for both convex and non-convex models, the main idea of the method comes from the optimal forgetting step in convex models. Let $f\_{\textbf{w}^\*}$ is the pre-trained model to be erased, and given a dataset $\mathcal{D}$ together with $\mathcal{D}\_f \subset \mathcal{D}$ and $\mathcal{D}\_r = \mathcal{D} \setminus \mathcal{D}\_f$, the data to forget and retain respectively. If we assume that the loss function $L$ is strictly convex, the optimal forgetting step is to find the optimal parameters $\textbf{w}^\*_r$ that maximize the loss is given by:
$$ \textbf{w}^\*\_r \approx \textbf{w}^\* + \frac{1}{n-k}H^{-1}\_{\mathcal{D}_r}(\textbf{w}^\*) \nabla\_{\textbf{w}} L\_{\mathcal{D}\_f}(f\_{\textbf{w^\*}}) $$
where $H\_{\mathcal{D}\_r}(\textbf{w}^\*)$ is the Hessian of the loss function $L$ at $\textbf{w}^\*$ with respect to the data in $\mathcal{D}\_r$, and $n = |\mathcal{D}|$ and $k = |\mathcal{D}\_f|$ are the total number of samples and the number of samples to be erased respectively, and $L\_{\mathcal{D}\_f}(f\_{\textbf{w^\*}})$ is the loss of the model $f\_{\textbf{w\*}}$ on the data $\mathcal{D}\_f$.
 
Since the data to forget is much smaller than the entire dataset, the authors assume that the Hessian $H\_{\mathcal{D}\_r}(\textbf{w}^\*)$ is approximately the same as $H\_{\mathcal{D}}(\textbf{w}^\*)$. Computing the inverse of the Hessian is computationally expensive. Assuming that $L\_{\mathcal{D}} = \sum_{i=1}^n \ell(f\_{\textbf{w}^\*}(\textbf{x}^{(i)}), \textbf{y}^{(i)}) = \sum_{i=1}^n -\log p(y\_i|x\_i; \textbf{w}^\*)$, Peste et. al. propose to approximate it by the **empirical Fisher Information Matrix (FIM)**, which is defined by: 
$$ F\_\mathcal{D}(\textbf{w}^\*) = \frac{1}{n} \sum\_{i=1}^n \nabla\_{\textbf{w}} \log p(\textbf{y}^{(i)}|\textbf{x}^{(i)};\textbf{w}^\*) \cdot \nabla\_{\textbf{w}}^T \log p(\textbf{y}^{(i)}|\textbf{x}^{(i)};\textbf{w}^*) $$
where $\textbf{x}^{(i)}$ and $\textbf{y}^{(i)}$ are the input and output of the model $f\_{\textbf{w}^\*}$ respectively. In practice this is computed efficiently with rank-1 updates by means of Sherman-Morrison lemma (#2). These defines the **SSSE** which is an approximation to the optimal forgetting step:
$$ \textbf{w}^\*\_r \approx \textbf{w}^\* + \frac{\epsilon}{n-k}F^{-1}\_{\mathcal{D}}(\textbf{w}^\*) \nabla\_{\textbf{w}} L\_{\mathcal{D}_f}(f\_{\textbf{w\*}}) $$
where $\epsilon$ is a hyperparameter that controls the step size.

To properly choose $\epsilon$, they define a **similarity ratio** to measure if the updated model is closer to the original model or to the retrained-from-scratch model. It is defined comparing the Area-Under-the-Curve (AUC) score corresponding to the Receiver Operating Characteristic (ROC) curve. In this setting they cannot conclude that SSSE converges precisely to the optimal forgetting step, but they show that it is effective in practice, not only for convex models but also for non-convex models.

### Personal Thoughts
- As far as I understand, they are using the entire dataset to compute the empirical FIM, hence contradicting their claim that SSSE only requires access to the data to be erased. This is a bit confusing.

- The theoretical analysis is quite interesting, and definitely studying the optimal forgetting step in convex models is a good direction. However, they assume need to assume that the loss is not only strictly convex but also that it is the negative log-likelihood of the data. This is a strong assumption that justifies their reasoning but they translate it to a more general setting only providing empirical evidence. 

- Is nice that they introduce the similarity ratio to measure the performance of the method. There is a need for a metric to evaluate the performance of unlearning methods, and this is a good start.



### Reference
<a id="1">[1]</a> Peste, A., Alistarh, D., & Lampert, C. H. (2021). SSSE: Efficiently Erasing Samples From Trained Machine Learning Models. [arXiv:2107.03860](https://arxiv.org/abs/2107.03860).

<a id="2">[2]</a> Sherman, J., & Morrison, W. J. (1950). [Adjustment of an Inverse Matrix Corresponding to Changes in the Elements of a Given Column or a Given Row of the Original Matrix](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-21/issue-1/Adjustment-of-an-Inverse-Matrix-Corresponding-to-a-Change-in/10.1214/aoms/1177729893.full). *The Annals of Mathematical Statistics*, 21(1), 124-127.