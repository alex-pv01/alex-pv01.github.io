---
title: "What are Binary Latent Diffusion Models?"
date: 2023-10-12T18:34:11+02:00
draft: true

tags: ["diffusion", "ai", "binary", "model", "text", "image", "text-to-image", "Bernoulli"]

ShowToc: true
---

I believe many of the ones reading this post have already heard about the [Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (LDMs) and their applications in the field of text-to-image generation. Some of the best examples out there are the [SDXL](https://arxiv.org/abs/2307.01952) models, just published by [Stability AI](https://stability-ai.com/). Recently such models have extended to other domains to solve different tasks such as [video generation](https://arxiv.org/abs/2110.01124), [Audio generation](https://arxiv.org/abs/2110.01124), [3D object generation](https://arxiv.org/abs/2110.01124), and many more. However, Latent Diffusion Models still suffer from some limitations, e.g. the need for a large number of steps to generate high-quality samples, and for a large number of parameters to achieve good results.

In order to democratize the use of LMDs and make them more accessible to the community, several research groups are working towards simplifying the models and making them more efficient. One of the most promising approaches is the use of binary latent variables instead of continuous ones [Ze Wang et al, 2023](https://arxiv.org/abs/2304.04820). They propose a new model called Binary Latent Diffusion Models (BLDMs) that uses binary latent variables to generate high-quality samples by redefining the diffusion process.


## Recap on Diffusion Models
## Recap on Latent Diffusion Models
## What are Binary Latent Diffusion Models?
### Binary Image Representations
#### Parametrization of the Autoencoder Loss
#### Similarities with the VQ-VAE
### Bernoulli Diffusion Process
#### Forward Diffusion Process
#### Reverse Diffusion Process
#### Parameterization of the Diffusion step Loss
#### Similarities with the Gaussian Diffusion Process
## Conditioned Generation
## Comparison with existing methods
## Code and Pretrained Models
## Quick Summary
## Citation
## References