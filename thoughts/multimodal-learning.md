---
title: Multimodal Contrastive Learning - Aligning Vision and Language
date: August 5, 2024
tags: [multimodal, contrastive learning, vision-language, embeddings]
---

<p class="frst">
The ability to learn from multiple modalities simultaneously—vision, language, audio—has emerged as one of the most powerful paradigms in modern AI. Contrastive learning provides an elegant framework for aligning these different modalities in a shared semantic space.
</p>

## The Alignment Problem

How do we learn representations that capture the correspondence between images and text? This is fundamentally a problem of alignment: finding a mapping that brings semantically similar concepts close together, regardless of modality.

### Why Contrastive Learning?

Contrastive methods learn by comparing positive pairs (matching image-text) against negative pairs (mismatched):

$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(i, t_i) / \tau)}{\sum_{j} \exp(\text{sim}(i, t_j) / \tau)}$$

where $\text{sim}(\cdot, \cdot)$ measures similarity and $\tau$ is a temperature parameter.

<p class="note">
CLIP trained on 400 million image-text pairs achieves remarkable zero-shot transfer by learning a joint embedding space through contrastive learning.
</p>

## Geometric Considerations

The choice of embedding space geometry profoundly affects what relationships the model can capture.

### Euclidean Embeddings

Most models use Euclidean space with cosine similarity:

$$\text{sim}(i, t) = \frac{i \cdot t}{\|i\| \|t\|}$$

This works well for many tasks but struggles with hierarchical relationships.

### Hyperbolic Alternatives

My research explores hyperbolic embeddings for vision-language models. The key insight: visual and linguistic concepts often form hierarchies.

Consider the hierarchy:
- "animal" → "mammal" → "dog" → "golden retriever"

In Euclidean space, representing such trees requires dimensions exponential in depth. Hyperbolic space represents them efficiently with linear dimension growth.

## The MERU Model

Recent work on MERU (Hyperbolic Image-Text Representations) shows promising results:

\begin{equation} \label{eq:meru-loss}
\mathcal{L}_{\text{MERU}} = -\log \frac{\exp(-d_{\mathbb{H}}(i, t_i) / \tau)}{\sum_{j} \exp(-d_{\mathbb{H}}(i, t_j) / \tau)}
\end{equation}

where $d_{\mathbb{H}}$ is the hyperbolic distance from equation \eqref{eq:poincare-distance}.

### Advantages of Hyperbolic Embeddings

- Better capture of hierarchical structure
- More parameter-efficient for taxonomic relationships
- Natural handling of uncertainty (distance from origin = specificity)

<p class="note">
In my research on machine unlearning in MERU, we found that hyperbolic geometry affects how information can be selectively removed from the model.
</p>

## Training Challenges

### Numerical Stability

Operations in hyperbolic space require care near the boundary of the Poincaré ball:

```python
def safe_hyperbolic_distance(x, y, c=1.0, eps=1e-6):
    """
    Numerically stable hyperbolic distance computation.
    """
    # Clamp points away from boundary
    x_norm = x.norm(dim=-1, keepdim=True).clamp(max=1-eps)
    y_norm = y.norm(dim=-1, keepdim=True).clamp(max=1-eps)

    x = x / x_norm * (x_norm * (1 - eps))
    y = y / y_norm * (y_norm * (1 - eps))

    # Compute distance
    return poincare_distance(x, y, c)
```

### Batch Composition

The number of negative pairs in a batch dramatically affects learning:

$$N_{\text{negatives}} = 2B - 2$$

for batch size $B$. Larger batches → better contrastive signal.

## Beyond Pairwise Alignment

Current contrastive methods operate on pairs, but real-world semantics involve complex relationships:

- Compositionality: "red car" vs "car" vs "red"
- Negation: "not a dog"
- Temporal dynamics: "before" and "after"

### Future Directions

Several open problems remain:

1. **Multi-way alignment**: Vision + language + audio simultaneously
2. **Compositional understanding**: Handling complex linguistic structures
3. **Few-shot adaptation**: Quick adaptation to new domains
4. **Theoretical understanding**: Why does contrastive learning work so well?

## Implications for Downstream Tasks

Models trained with multimodal contrastive learning excel at:

- Zero-shot classification
- Image retrieval
- Visual question answering
- Image generation conditioning

The shared embedding space enables flexible transfer across tasks and modalities.

## Open Research Questions

My work continues to explore:

- How does embedding geometry affect compositional understanding?
- Can we efficiently unlearn concepts from multimodal models?
- What role does temporal structure play in video-language alignment?
- How do we scale to more modalities while maintaining alignment quality?

The intersection of geometry, contrastive learning, and multimodal AI offers rich opportunities for advancing our understanding of how machines can learn from diverse data sources.
