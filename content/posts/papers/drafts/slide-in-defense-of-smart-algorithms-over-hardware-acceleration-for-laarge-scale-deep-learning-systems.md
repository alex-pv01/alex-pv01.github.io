---
title: "Slide in Defense of Smart Algorithms Over Hardware Acceleration for Laarge Scale Deep Learning Systems"
date: 2024-04-21T18:45:21+02:00
draft: true
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

The following post is a comment on the paper [SLIDE: In Defense of Smart Algorithms over Hardware Acceleration for Large-Scale Deep Learning Systems](#1) by [Beidi Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen,+B), [Tharun Medini](https://arxiv.org/search/cs?searchtype=author&query=Medini,+T), [James Farwell](https://arxiv.org/search/cs?searchtype=author&query=Farwell,+J), [Sameh Gobriel](https://arxiv.org/search/cs?searchtype=author&query=Gobriel,+S), [Charlie Tai](https://arxiv.org/search/cs?searchtype=author&query=Tai,+C) and [Anshumali Shrivastava](https://arxiv.org/search/cs?searchtype=author&query=Shrivastava,+A), from [Rice University](https://www.rice.edu/) and [Intel Corporation](https://www.intel.com/content/www/us/en/homepage.html).

To get around the costly computations associated with large models and data, the community is increasingly investing in specialized hardware for model training. However, such machines are expensive and hard to generalize to a multitude of tasks. In this paper, the authors propose **SLIDE** (**S**ub-**Li**near **D**eep learning **E**ngine) that uniquely blends smart randomized algorithms, with multi-core parallelism and workload optimization. The authors show that SLIDE on a 44-core CPU, drastically reduces the computations during both training (3.5 times faster) and inference outperforming an optimized implementation of Tensorflow on a Tesla V100 GPU. 

They explore the idea of **adaptative sparzity**. The idea stems from the fact that we can accurately train neural networks by selectively sparsifying most of the neurons, based on their activation, during every gradient update. However, this technique does not directly lead to computational savings. To achieve that, they employ **Locality Sensitive Hash** (LSH) tables to identify a sparse neurons efficiently during each step.

### Locality Sensivite Hashing
LSH is a family of functions with the property that similar input objects in the domain of these functions have a higher **probability of colliding** in the range space than non-similar ones. 

It is shown that having an LSH family for a given similarity measure, is sufficient for efficiently solving nearest-neighbor search in sub-linear time. 

#### LSH Algorithm
The LSH algorithm uses two parameters $(K, L)$. Authors contruct $L$ independent hash tables, each of which has a meta-hash function $H$ that is formed by concatenating $K$ random independent hash functions. Given a query, we collect one bucket of each table and return the union of all $L$ buckets, which reduces the number of false negatives. Only valid nearest-neighbor items are likely to match all $K$ hash values for a given query, thus $H$ reduces the number of false positives:
1. **Pre-processing phase:** Construct $L$ hash tables, storing the pointers to the data elements.
2. **Query phase:** Given a query $Q$, search for its nearest-neighbors by querying all $L$ hash tables and returning the union of all $L$ buckets.

#### LSH for Estimation and Sampling
It turns out that taking a few hash buckets (as low as 1) is sufficient for adaptive sampling. Given a collection of vectors $\mathcal{C}$ and a query $Q$, we get a candidate set $S$ from a $(K,L)$-LSH algorithm. Every element $x_i \in \mathcal{C}$ gets sampled into $S$ with probability $p_i$, where $p_i$ is a monotonically inicreasing function of $Q \codt x_i$. Thus, we can pay on-time linear cost of preprocessing $\mathcal{C}$ into hash tables, and adaptive sampling for quary $Q$ only requires few hash lookups.

### SLIDE Algorithm
1. **Initialization:** 

### References
<a id="1">[1]</a> Chen, B., Medini, T., Farwell, J., Gobriel, S., Tai, C., & Shrivastava, A. (2024). *SLIDE: In Defense of Smart Algorithms over Hardware Acceleration for Large-Scale Deep Learning Systems*. [arXiv:1903.03129](https://arxiv.org/abs/1903.03129).