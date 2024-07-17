---
title: 'Notes on Rethinking the Representation in Federated Unsupervised Learning With Non-IID Data'
date: 2024-06-24T10:58:50+02:00
draft: false
math: true

cover:
    image: "<image path/url>"
    # can also paste direct link from external site
    # ex. https://i.ibb.co/K0HVPBd/paper-mod-profilemode.png
    alt: "<alt text>"
    caption: "<text>"
    relative: false # To use relative path for cover image, used in hugo Page-bundles

tags: []

ShowToc: false
---

***Disclaimer:*** *This is part of my notes on AI research papers. I do this to learn and communicate what I understand. Feel free to comment if you have any suggestion, that would be very much appreciated.*

The following post is a comment on the paper [Rethinking the Representation in Federated Unsupervised Learning With Non-IID Data](#1) by [Xinting Liao](https://arxiv.org/search/cs?searchtype=author&query=Liao,+X), [Weiming Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+W), [Chaochao Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen,+C), [Pengyang Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou,+P), [Fengyuan Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu,+F), [Huabin Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu,+H), [Binhui Yao](https://arxiv.org/search/cs?searchtype=author&query=Yao,+B), [Tao Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+T), [Xiaolin Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng,+X) and [Yanchao Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan,+Y), from [Zhejiang University](https://www.zju.edu.cn/english/), [Midea Group](https://www.midea-group.com/), and [Fuzhou University](https://en.fzu.edu.cn/).


In this paper authors propose *FedU²*, a **Federated Unsupervised Learning (FUSL)** framework that enhances generating uniform and unified representations with **non-IID data**. 

Research on this topic is motivated by the fact that **Federated Learning (FL)** is becoming increasingly popular in the AI community, as it allows multiple parties (or clients) to **collaboratively** train a model without sharing their data. However, in practice, the data from each party is substantially different to one another, which leads to the **non-IID data distribution problem**. Therefore, in this work, they consider the problem of Federated Unsupervised Learning (FUSL) with non-IID data, where the goal is to learn a unified representation among **imbalanced**, **unlabeled**, and **decentralized data**.

Such scenario pose two main challenges:
1. **Representation collapse entanglement:** Representation collapse in a client model subsequently exacerbates the representation of global and other local models. 

> ***Representation collapse*** is a phenomenon in self-supervised learning, where the model learns to output the same representation for all inputs, which is a trivial solution to the task. It can be *local* or *global*, if it happens for a subset of the data or for all the data, respectively. Furthermore, it can be distinguished by two types according to the severity of the collpase, either **complete collapse**, where each input is mapped to a single constant point (or very close to it), or **dimensional collapse**, where the inputs are mapped to a low-dimensional manifold. See [[2]](#2) for more details.

2. **Representation inconsistency:** Inconsistent client model parameters lead to discrepant parameter spaces, bringing less unified representations among clients.




### References
<a id="1">[1]</a> Liao, X., Liu, W., Chen, C., Zhou, P., Yu, F., Zhu, H., Yao, B., Wang, T., Zheng, X., & Tan, Y. (2024). Rethinking the Representation in Federated Unsupervised Learning With Non-IID Data. [arXiv:2403.16398](https://arxiv.org/abs/2403.16398).

<a id="2">[2]</a> Hua T., Wang W., Xue Z., Ren S., Wang Y., Zhao H. (2021) On Feature Decorrelation in Self-Supervised Learning. [arXiv:2105.00470](https://arxiv.org/abs/2105.00470).