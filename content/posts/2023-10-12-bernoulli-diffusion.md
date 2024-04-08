---
title: "What are Binary Latent Diffusion Models?"
date: 2023-10-12T18:34:11+02:00
draft: true

tags: ["diffusion", "ai", "binary", "model", "text", "image", "text-to-image", "Bernoulli"]

ShowToc: true
---

I believe many of the ones reading this post have already heard about the [Latent Diffusion Models](https://arxiv.org/abs/2112.10752) (LDMs) and their applications in the field of text-to-image generation. Some of the best examples out there are the [SDXL](https://arxiv.org/abs/2307.01952) models, by [Stability AI](https://stability-ai.com/). Recently such models have extended to other domains to solve different tasks such as [video generation](https://arxiv.org/abs/2110.01124), [Audio generation](https://arxiv.org/abs/2110.01124), [3D object generation](https://arxiv.org/abs/2110.01124), and many more. However, Latent Diffusion Models still suffer from some limitations, e.g. the need for a large number of steps to generate high-quality samples, and for a large number of parameters to achieve good results.

In order to democratize the use of LMDs and make them more accessible to the community, several research groups are working towards simplifying the models and making them more efficient. One approach is the use of binary latent variables instead of continuous ones [Ze Wang et al, 2023](https://arxiv.org/abs/2304.04820). They propose a new model called Binary Latent Diffusion Models (BLDMs) that uses binary latent variables to generate high-quality samples by redefining the diffusion process.


## Recap on Diffusion Models
[comment]: <> (This section can provide a brief overview of diffusion models in general, setting the stage for the specific focus on binary latent diffusion.)
The outstanding results of diffusion models have attracted the attention of many researchers and practitioners. This skyrocketed the number of diffusion models papers published in the last few years and the number of applications of diffusion models in different domains. One can find well-documented articles like [Lilian`s](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) blogpost or [Yannic's](https://www.youtube.com/watch?v=W-O7AZNzbzQ) talk that deeply explain the theory behind diffusion models. However I find pertinent to give a brief overview of the main concepts behind diffusion models to set the stage for the specific focus on binary latent diffusion.

Diffusion models are a family of generative models that use a diffusion process to generate samples. The diffusion process is a stochastic process that starts from a simple distribution and gradually transforms it into a complex distribution.

--------------add image here-----------------
caption: afegir la típica imatge de la distribució simple transformant-se en la complexa

The diffusion process is defined by a sequence of steps, where each step is a stochastic transformation that takes the current distribution and transforms it into the next distribution. Such process is reversible, meaning that steps can be done in the opposite direction and recover the previous distribution. Hence, a diffusion generative model is composed of two parts: a **forward diffusion process** and a **reverse diffusion process**. The forward diffusion process is used to transform a complex distribution into a simple one, while the reverse diffusion process will reconstruct the complex distribution from the simple.

In orther to picture the following concepts, take any generative problem you would like to solve. To keep it simple, let's say you want to generate images of cats. Any other images, videos, audios, and any other type of data can be represented as a vector of numbers, so the same reasoning can be applied. In our case, clearly cat images have an easy particularity, the pixels of which the image if made of contian certain patterns that make them look like cats. Suppose that we would like to generate colored high quality images, that is, our images are represented by a matrix of size 1024x1024 times 3 for the RGB channels, which lead to 3,145,728 pixels, and the value in each pixel ranges from 0 to 255. That flagrant amount of $$256^{1024\times 1024 \times 3} = \text{too many times bigger than the nuber of atoms in the observable universe}.$$ Imagine you want to get an image of a cat by randomly picking one of the pixel space, you would be trying for a while.

--------------add an interactive random image sampler-----------------
caption: add an interactive random image sampler that allows you to sample images of this space. a veure si algú treu una imatge de gat :P

The idea behind diffusion models is to define a distribution over the huge space of pixels, or any other type of data, that allows you to sample meaningful elements of this space.

### Forward Diffusion Process

Assume that we have the cat distribution of images, that is $q(\mathbb{x})$ and one can use it to sample cat images $\mathbb{x} \sim q(\mathbb{x})$. The *forward diffusion* is defined as a [Markov Chain](https://setosa.io/ev/markov-chains/) that starts from the cat distribution and gradually transforms it into a simple distribution. 

### Reverse Diffusion Process



## Recap on Latent Diffusion Models
[comment]: <> (Similarly, this section can offer a summary of latent diffusion models to provide context for the subsequent discussion of binary latent diffusion.)
## What are Binary Latent Diffusion Models?
[comment]: <> (This is the core of your article, where you can delve into the details of binary latent diffusion models, including binary image representations, the Bernoulli diffusion process, and conditioned generation.)
### Binary Image Representations
[comment]: <> (Here, you can discuss the parametrization of the autoencoder loss and draw similarities with the VQ-VAE, a related model for discrete image representations.)
#### Parametrization of the Autoencoder Loss
#### Similarities with the VQ-VAE
### Bernoulli Diffusion Process
[comment]: <> (This section can cover the forward and reverse diffusion processes, the parameterization of the diffusion step loss, and similarities with the Gaussian diffusion process.)
#### Forward Diffusion Process
#### Reverse Diffusion Process
#### Parameterization of the Diffusion step Loss
#### Similarities with the Gaussian Diffusion Process
## Conditioned 
[comment]: <> (This part can focus on how binary latent diffusion models facilitate conditioned generation of images.)
## Comparison with existing methods
[comment]: <> (You can compare binary latent diffusion models with other existing methods, highlighting their advantages and potential areas of improvement.)
## Code and Pretrained Models
[comment]: <> (This section can provide practical resources for your readers, such as code implementations and pretrained models, to encourage further exploration and experimentation.)
## Quick Summary
[comment]: <> (A concise summary of the key points discussed in the article.)
## Citation
[comment]: <> (Guidelines for citing the article, if applicable.)
## References
[comment]: <> (A list of references to acknowledge the sources of information used in the article.)