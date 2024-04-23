---
title: "Notes on LLaVA-Gemma: Accelerating Multimodal Foundation Models With a Compact Language Model"
date: 2024-04-11T15:46:55+02:00
draft: false
math: true

cover:
    image: "<image path/url>"
    # can also paste direct link from external site
    # ex. https://i.ibb.co/K0HVPBd/paper-mod-profilemode.png
    alt: "<alt text>"
    caption: "<text>"
    relative: false # To use relative path for cover image, used in hugo Page-bundles

tags: ["AI", "LLM", "Multimodal", "Vision", "Language", "Model", "LLaVA", "Gemma", "Intel Labs", Paper, Research]

ShowToc: false
---

***Disclaimer:*** *This is part of my notes on AI research papers. I do this to learn and communicate what I understand. Feel free to comment if you have any suggestion, that would be very much appreciated.*

The following post is a comment on the paper [LlaVA-Gemma: Accelerating Multimodal Foundation Models With a Compact Language Model](#1) by [Musashi Hinck](https://arxiv.org/search/cs?searchtype=author&query=Hinck,+M), [Matthew L. Olson](https://arxiv.org/search/cs?searchtype=author&query=Olson,+M+L), [David Cobbley](https://arxiv.org/search/cs?searchtype=author&query=Cobbley,+D), [Shao-Yen Tseng](https://arxiv.org/search/cs?searchtype=author&query=Tseng,+S), and [Vasudev Lal](https://arxiv.org/search/cs?searchtype=author&query=Lal,+V).

Hinck et. al. who are part of the Cognitive AI research area at [Intel Labs](https://www.intel.com/content/www/us/en/research/overview.html), train a suite of Multimodal Foundation Models (MFM) using the [LLaVA](https://llava-vl.github.io/) framework with the recently released [Gemma](https://blog.google/technology/developers/gemma-open-models/) Large Language Models (LLM). There is an increasing interest in small and capable **Vision Language Models (VLM)** and their proposed model, **LLaVA-Gemma**, is a step towards this direction. They provide an ablation study on three design features of the model:

1. Utilizing a more powerful **image backbone**: The original LLaVA model uses [CLIP](https://openai.com/research/clip) as a pretrained vision encoder. They propose to use a larger and more powerful vision encoder, the [DINOv2](https://dinov2.metademolab.com/). Both models are compared in terms of GQA, MME, MM-Vet, POPE, VQAv2, and, MMVP, which are common benchmarking tools to other LLM works. The DINOv2 model shows **better performance** in general.

2. Increasing the size of the **language backbone**: In the case of the language encoder, LLaVA uses [Llama-2](https://llama.meta.com/llama2/), whereas Hinck et. al. use the recently released Gemma model. It comes in a variety of sizes, of which they use the smaller ones, Gemma-2B and Gemma-7B. The main difference between Llama and Gemma is that the latter uses a larger token set than any other LLM, that is 256k tokens (vs 50k tokens in Llama). However, their experiments show that the larger token set does not improve the performance of the model, in fact, it shows a **decrease in performance** (in terms of the above mentioned benchmarks). Since there is no *a priori* reason to expect this behaviour, they suggest that understanding the reasons behind this could be a fruitful area for future research.

3. Pretraining the connector: [Other studies](#2) have shown that pretraining the connector, which is a MLP that maps the image features to the language features, can downstream performance. Contrary to this, their experiments show that pretraining the connector do improve the performance of the model.

A deeper analysis between Gemma-2B and Gemma-7B shows that the larger model does 

Overall, the proposed LLaVA-Gemma falls short of performance compared to the SOTA models, but it is a step towards the democratization of VLMs. The good part is that authors have made the code and models available in the [Hugging Face](https://huggingface.co/Intel/llava-gemma-2b) repository!


### References
<a id="1">[1]</a> Hinck, M., Olson, M. L., Cobbley, D., Tseng, S.-Y., & Lal, V. (2024). LlaVA-Gemma: Accelerating Multimodal Foundation Models With a Compact Language Model. [arXiv:2404.01331](https://arxiv.org/abs/2404.01331).

<a id="2">[2]</a> Karamcheti, S., Nair, S., Balakrishna, A., Liang, P., Kollar, T., & Sadigh, D. (2024). Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models. [arXiv:2402.07865](https://arxiv.org/abs/2402.07865).
