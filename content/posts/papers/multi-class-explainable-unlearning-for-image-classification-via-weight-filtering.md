---
title: "Notes on Multi Class Explainable Unlearning for Image Classification via Weight Filtering"
date: 2024-04-08T09:36:22+02:00
draft: false
math: true

tags: [Machine Unlearning, Explainable AI, Image Classification, Weight Filtering, Multi-Class, AI, Research, Paper]
---

***Disclaimer:*** *This is part of my notes on AI research papers. I do this to learn and communicate what I understand. Feel free to comment if you have any suggestion, that would be very much appreciated.*

The following post is a comment on the paper [Multi Class Explainable Unlearning for Image Classification via Weight Filtering](#1) by [Samuele Poppi](https://arxiv.org/search/cs?searchtype=author&query=Poppi,+S), [Sara Sarto](https://arxiv.org/search/cs?searchtype=author&query=Sarto,+S), [Marcella Cornia](https://arxiv.org/search/cs?searchtype=author&query=Cornia,+M), [Lorenzo Baraldi](https://arxiv.org/search/cs?searchtype=author&query=Baraldi,+L) and [Rita Cucchiara](https://arxiv.org/search/cs?searchtype=author&query=Cucchiara,+R).

Samuele P., et. al. propose a novel approach to unlearn a **multiple** classes from a pre-trained image classification model in a **single untraining round**. The technique learns a "Weight Filtering Network" (WF-Net) that is able to modulate the inner components of the model to remove the class of interest. The method discovers the weights that are responsible for the classification of the target class and then filters them out. This approach implicitly discovers the underlying relationships between network inner components and output classes and therefore allows to obtain a representation that can be employed for **explainability** purposes.

In comparison with single-class unlearning, WF-Net avoids the need of storing multiple models and performing multiple untraining rounds. This allows for a significant reduction in computational costs and memory usage, both at untraining and testing stages and provides increased flexibility.  

The key observation is that there is a **mapping** between the inner components of a network and the output classes, as stated in [[2]](#2). Once trained, WF-Net is able to turn on and off those inner components to accomplish the desired unlearning behaviour on a class of choice. In practice, each layer $l$ of the pre-trained model is point-wise multiplied by a Weight Filtering (WF) layer $\alpha_l$, which allows to modulate the weights of the model. The WF-Net, which is the sequence of all WF layers $\alpha:=\\{\alpha_l\\}_l$, is trained to remove a number of classes $N_c$ from the model. Note that $\alpha_l$ has shape $N_c \times K$, where $K$ is the length of $w_l$. After a single untraining round we end up with a single checkpoint of the WF-Net that can be used to instruct the model to behave as if any of the $N_c$ classes were never learned. It is possible to **forget all classes at same time** by setting $N_c$ to the total number of classes in the model. There are three key aspects regarding the training of the WF-Net:
 
1. **Loss function**: Is composed of two terms, an **unlearning loss** $L_f$ and a **retaining loss** $L_r$. Both are implemented as cross-entropy losses. The total loss should be minimized zeroing $L_r(\cdot)$ while maximizing $L_f(\cdot)$:
$$ L = \lambda_0 \sum_{(x,y)\in\mathcal{D_r}} L_r(M(x),y) + \lambda_1 \sum_{(x,y)\in\mathcal{D_f}} \frac{1}{L_f(M(x),y)}$$
where $\mathcal{D_r}$ and $\mathcal{D_f}$ are the datasets of the classes to be retained and forgotten, respectively, $\lambda_0$ and $\lambda_1$ are hyperparameters that control the importance of the two terms, and $M$ is the WF model, i.e. the pre-trained model with fixed weights together with the WF-Net. 

2. **Regularization**: Adding a regularizer $R(\cdot)$ to ensure only few parameters of $\alpha_l$ are dropped to zero. $R(\hat\alpha)$ is computed as the average of inverted alphas $\hat\alpha_l:=1-\alpha_l$:
$$ L = \lambda_0 \sum_{(x,y)\in\mathcal{D_r}} L_r(M(x),y) + \lambda_1 \sum_{(x,y)\in\mathcal{D_f}} \frac{1}{L_f(M(x),y)} + \lambda_2 R(\hat\alpha)$$

3. **Label expansion**: To realize untraining of all classes simultaneously, during the training process, each mini-batch of size $B$ is divided into two halves, obtaining $B/2$ samples from the classes to be **unlearned** and $B/2$ samples from the classes to be **retained**. Samples from the first half are labeled with the **original** labels, while samples from the second half are **randomly** labeled. The random strategy is used to randomly retain one of the rows of each $\alpha_l$. This retain step is performed $T$ times, pairing each time the samples with a different random label, and expanding the size of the retaining loss to $(T, B/2)$. The last step is averaging both losses.


### Personal considerations:

- Note that **this approach is not really about unlearning**, but about **modulating** the weights of the network using an additional network that is able to filter out the weights that are responsible for the classification of one (or more) of the $N_c$ classes to be forgotten. Despite that, it is an interesting method that can be used to improve the explainability models while acting **"as if"** the model has been untrained from a certain class. 

- I am sceptical about the computational efficiency of the model. As far as I understand, the WF-Net has as many parameters as the pre-trained model, and expands the size of the mini-batches by a factor of $T$. This could lead to a significant increase in the computational cost of the training process of the WF-Net.



### References
<a id="1">[1]</a> Poppi, S., Sarto, S., Cornia, M., Baraldi, L., & Cucchiara, R. (2023). Multi Class Explainable Unlearning for Image Classification via Weight Filtering. [arXiv:2304.02049](https://arxiv.org/abs/2304.02049).

<a id="2">[2]</a> Wang, A., Lee, W., & Qi, X. (2022). HINT: Hierarchical Neuron Concept Explainer. [arXiv:2203.14196](https://arxiv.org/abs/2203.14196).