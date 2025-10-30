---
title: Beyond Static Images - Temporal Reasoning in Video Understanding
date: September 22, 2024
tags: [video, temporal reasoning, computer vision, geometry]
---

<p class="frst">
While image understanding has achieved remarkable success, video presents unique challenges that go beyond processing individual frames. True video understanding requires modeling temporal dynamics, causal relationships, and the hierarchical structure of events unfolding over time.
</p>

## The Temporal Dimension

Videos are not just sequences of images—they contain rich temporal information about motion, causality, and the evolution of scenes over time.

### What Makes Video Different?

Consider these fundamental challenges:

- **Temporal coherence**: Understanding how frames relate across time
- **Motion dynamics**: Tracking and interpreting object movements
- **Event hierarchies**: From atomic actions to complex activities
- **Long-range dependencies**: Events separated by many frames

<p class="note">
A 10-minute video at 30 fps contains 18,000 frames. Modeling all pairwise relationships would require 162 million computations.
</p>

## Representation Challenges

How do we represent video in a way that captures both spatial and temporal structure? Traditional approaches flatten the temporal dimension or use simple pooling, losing critical information.

### Spatio-Temporal Embeddings

The joint space must capture:

$$\mathbf{v} = f_{\text{video}}(\{\mathbf{I}_1, \mathbf{I}_2, ..., \mathbf{I}_T\})$$

where each frame $\mathbf{I}_t$ has spatial structure and frames have temporal dependencies.

## Geometric Perspectives

My research explores whether non-Euclidean geometries better capture video structure:

### Hyperbolic Time?

Events often form hierarchies: "cooking dinner" contains "chopping vegetables" contains "cutting motion". This suggests hyperbolic embeddings might naturally represent temporal hierarchies.

<p class="note">
Hyperbolic spaces excel at hierarchy, but what about periodic patterns like walking or breathing? Mixed-curvature spaces might be needed.
</p>

### Temporal Graph Structure

We can view video as a graph:

\begin{equation} \label{eq:video-graph}
\mathcal{G} = (\mathcal{V}, \mathcal{E}), \quad \mathcal{V} = \{\mathbf{x}_t\}_{t=1}^T
\end{equation}

where edges encode temporal relationships. The geometry of this graph matters.

## Current Approaches

### 3D Convolutions

Extending spatial convolutions to time:

```python
# 3D convolution over space and time
conv3d = nn.Conv3d(
    in_channels=64,
    out_channels=128,
    kernel_size=(3, 3, 3),  # (time, height, width)
    padding=(1, 1, 1)
)
```

### Two-Stream Networks

Separate pathways for appearance and motion:

- **Spatial stream**: Processes individual frames
- **Temporal stream**: Analyzes optical flow
- **Fusion**: Combines both streams for prediction

### Transformer-Based Models

Recent work applies transformers to video, but attention scales quadratically with sequence length. For video, this becomes prohibitive:

$$\text{Complexity} = O(T^2 d)$$

where $T$ is the number of frames.

## Open Questions

Several fundamental questions remain:

1. **Geometry**: What is the natural geometry of video representations?
2. **Granularity**: At what temporal scales should we model?
3. **Causality**: How do we capture causal relationships between events?
4. **Efficiency**: Can we scale to hour-long videos?

## Vision-Language Models for Video

Extending CLIP to video is non-trivial. Video-text pairs exhibit:

- Temporal misalignment between words and actions
- Hierarchical correspondence (word → sentence → paragraph vs. frame → clip → video)
- Compositional structure that current models struggle to capture

<p class="note">
This is where my research focuses: using geometric deep learning to better model the compositional and hierarchical structure of video-language relationships.
</p>

## Looking Ahead

The next generation of video understanding models will need to:

- Model long-range temporal dependencies efficiently
- Capture hierarchical event structure
- Understand causality and counterfactuals
- Scale to internet-scale video datasets

The intersection of geometric deep learning and video understanding offers promising directions for addressing these challenges.
