---
title: "Notes on The Era of 1-Bit LLMs: All Large Language Models Are in 1.58 Bits"
date: 2024-04-13T15:46:04+02:00
draft: false
math: true

# cover:
#     image: "<image path/url>"
#     # can also paste direct link from external site
#     # ex. https://i.ibb.co/K0HVPBd/paper-mod-profilemode.png
#     alt: "<alt text>"
#     caption: "<text>"
#     relative: false # To use relative path for cover image, used in hugo Page-bundles

tags: []

ShowToc: false
---

***Disclaimer:*** *This is part of my notes on AI research papers. I do this to learn and communicate what I understand. Feel free to comment if you have any suggestion, that would be very much appreciated.*

The following post is a comment on the paper [The Era of 1-Bit LLMs: All Large Language Models Are in 1.58 Bits](#1) by [Shuming Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma,+S), [Hongyu Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+H), [Lingxiao Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma,+L), [Lei Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+L), [Wenhui Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+W), [Shaohan Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang,+S), [Li Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong,+L), [Ruiping Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+R), [Jilong Xue](https://arxiv.org/search/cs?searchtype=author&query=Xue,+J), and [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei,+F).


Ma et. al. from [Microsoft Research Asia](https://www.msra.cn/) and [University of Chinese Academy of Sciences](https://english.ucas.ac.cn/), are developing a new research line on 1-bit Large Language Models (LLMs) under the frame of [General AI](https://thegenerality.com/agi/), whose mission is to advance AI for humanity. This is the second paper they have published in this regard.

This work is a follow up on ther previous publication where they introduced [BitNet](#2). The project is motivated by the **increasing size of LLMs** that pose challenges for deployment and raise concerns about the environmental impact due to the high energy coonsumption.

Authors claim that the new version introduced in this paper, **BitNet b1.58** provide a Pareto solution to reduce inference cost of LLMs while maintaining the model performance. Thus, BitNet b1.58 enbales a **new paradigm** of LLM complexity and calls for actions to design new hardware optimized for 1-bit LLMs.

### BitNet b1.58
The main difference is in regard of the binarization step. There is no binarization anymore, now they adopt **absmean quantization** where the weights are set to be $+1, -1$ or $0$. This step is formalized as:

\begin{align}
\tilde{W} &= \text{RoundClip}(\frac{W}{\gamma + \epsilon}, -1, +1), \\\
\text{RoundClip}(x,a,b) &= \max(a,\min(b,\text{round}(x))), \\\
\gamma &= \frac{1}{nm}\sum_{i,j}\|W_{i,j}\|.
\end{align}

The rest of the BitLinear layer remains the same as in the original BitNet.

A good thing that they mention is that the method incorporate LLaMA-alike components so it is easier to implement them into the open-source frameworks.

### Results

There is a dramatic improvement in terms of energy consumption and memory usage, while maintaining state-of-the-art performance. The following figures ilustrate this.

![latency-and-memory](/figures/the-era-of-1-bit-llms-all-large-language-models-are-in-1.58bits/figure-2.png "Decoding latency (left) and memory consumption (right) of BitNet b1.58 varyiing the model size.")

![llama-vs-bitnet](/figures/the-era-of-1-bit-llms-all-large-language-models-are-in-1.58bits/table-3.png "Comparison of the throughput between BitNet b1.58 70B and LLaMA LLM 70B.")

![energy-consumption](/figures/the-era-of-1-bit-llms-all-large-language-models-are-in-1.58bits/figure-3.png "Energy consumption of BitNet b1.58 compared to LLaMA LLM at 7nm process nodes. The components of arithmetic operations (left) and the end-to-end energy cost across different model sizes (right)")

![zero-shot-accuracy](/figures/the-era-of-1-bit-llms-all-large-language-models-are-in-1.58bits/table-4.png "Zero-shot accuracy comparison of BitNet b1.58 with StableLM-3B (SOTA open-source 3B model) with 2T tokens. BitNet achieves superior performance on all tasks, indication to have strong generalization capabilities.")



### Personal thoughts

- I believe this paper is a game changer. They have unveiled a new paradigm, this is not only a new era of LLMs, but a new era of DNNs. I want to see 1-bit everything and everywhere! 1b-ResNet, 1b-YoLo, 1b-CNNs, etc.

- Whenever I see binary something I automatically think about sparsity. Is it possible to make the active weights sparse so that the model can benefit from sparsity to make it even more efficient? 

- Another thing that I already commented in the previous [post]() about BitNet, is whether it would be possible to also binarize the data. Encoding the inputs into a binary space then performing binary operations with binary weights and finally debinarizing to get the sampling probabilities. 


### References
<a id="1">[1]</a> Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., Dong, L., Wang, R., Xue, J., & Wei, F. (2024). The Era of 1-Bit LLMs: All Large Language Models Are in 1.58 Bits. [arXiv:2402.17764](https://arxiv.org/abs/2402.17764).

<a id="1">[2]</a> Wang, H., Ma, S., Dong, L., Huang, S., Wang, H., Ma, L., Yang, F., Wang, R., Wu, Y., & Wei, F. (2023). Bitnet: Scaling 1 Bit Transformers for Large Language Models. [arXiv:2310.11453](https://arxiv.org/abs/2310.11453).