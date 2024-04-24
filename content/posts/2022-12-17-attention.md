---
title: "Attention and Context based Embeddings"
date: 2022-12-17T17:51:02+01:00
draft: false

tags: ["Attention Mechanisms", "Natural Language Processing", "Deep Learning", "Machine Translation", "AI"]

ShowToc: true
math: true
---

**Attention mechanisms** are a type of techniques used in **natural language processing (NLP)** tasks that allow a model to focus on **specific** parts of the input when processing a sequence, rather than considering the **entire** sequence at once. These methods can improve the performance of the model by allowing it to efficiently process long sequences of text and make more accurate predictions.

To some extend, attention mechanisms are motivated by how human visual attention focuses on different regions of an image or how correlates words in a sentence. They were born to deal with long sequences in **sequence-to-sequence** tasks. That is, any problem that requieres a sequence as an input, and outputs another sequence. For example, in machine translation, attention mechanisms can allow the model to translate each word in the source language sentence one at a time, focusing on the most relevant words in the source sentence at each step. This can be especially important for tasks that involve long sentences, as it allows the model to better capture the **meaning and context** of the input.

They have become a key component of many other tasks, not only in machine translation but also in other problems such as summarization, and language modeling. They have been shown to improve the performance of various models and are an active area of research in the field.

## How attention mechanisms work:

Attention mechanisms are often implemented as part of a **recurrent neural network (RNN)** in a sequence-to-sequence model, is compossed by an **encoder** and a **decoder**:

- The **encoder** is a RNN that processes the input sequence and compresses the information into a **context vector**. Such vector represents the whole input sequence and is expected to be a good summary of its meaning.

- The **decoder** is another RNN that recieves the context vector and outputs a transformed vector that ideally solves the problem we are dealing with. 

By construction of the RNNs the context vector has a fixed size an requires the implementation of an attentenion mechanism to deal with long sequences, since otherwise the model "forgets" information.

In a broad sense, the seq2seq model processes the input sequence one element at a time. At each step, the attention mechanism calculates the **weights** for each element in the input sequence based on their relevance to the current state of the model. The weights can be calculated using various similarity measures, such as the dot product between the current state of the model and each element in the input sequence.

More formally, consider we have an input sequence $\textbf{x}$ of lenght $n$ and that we want to output a target sequence $\textbf{y}$ of lenght $m$,

\begin{align*}
	\textbf{x} = [x_1, \dots, x_n] \\\
	\textbf{y} = [y_1, \dots, y_m].
\end{align*}

We start by **initializing** the hidden state of encoder, which can be a random vector, $\bar{\textbf{h}}_0$. While the sequence is not finished, it takes as input, one element at a time from $\textbf{x}$ and the previous hidden state. At each step it generates a new vector $\bar{\textbf{h}}_i$ called **hidden state** of the encoder at step $i$, for $i = 1, \dots, n$. Notice that each $\bar{\textbf{h}}_i$ is presumably more associated with the element $x_i$. Once, all the hidden states are processed, they are sent to the decoder.

The decoder also has its hidden state initialized, $\textbf{h}\_0$. Then, it takes at each step $t$ a **weighted** combination of the encoder hidden states as a current **context vector**, the previous decoder's hidden state and the previous element of the output sequence to predict the value $y_t$. That is, for $t=1,\dots, m$, the decoder's hidden state at $t+1$ is of the form $\textbf{h}\_{t+1} = f(\textbf{h}\_t, \textbf{c}\_{t+1}, y_t)$, where:

\begin{align*}
	\textbf{c}\_t & = \sum\_{i=1}^n \alpha\_{t,i} \bar{\textbf{h}}\_i \quad \hfill &; \text{is the context vector at step }t. \\\
	\alpha\_{t+1,i} & = \frac{\exp(\text{score}(\textbf{h}_t, \bar{\textbf{h}}_i))}{\sum\_{i'=1}^n\exp(\text{score}(\textbf{h}\_t, \bar{\textbf{h}}\_{i'}))} \quad \hfill &; \text{softmaxed similarity score.}
\end{align*}

The $\text{score}$ function assigns a measure of similarity between hidden states. There are several ways to approach it, first introduced by [Bahdanau, et al., 2014](https://arxiv.org/pdf/1409.0473.pdf) and [Luong, et al., 2015](https://arxiv.org/pdf/1508.04025.pdf). For instance, one could use as a simple similarity function the dot product. The value $\alpha\_{t,i}$ aims to indicate the similarity between element $y_t$ and $x_i$.

Finally, in order to predict the value $\textbf{y}\_t$, the model concatenates the hidden layer $\textbf{h}\_t$ and the context vector $\textbf{c}\_t$, and pass them through a feedforward neural network that is trained simultaneously. This process is then repited until the completion of the output sequence.


## Bahdanau attention:

**Bahdanau attention**, also known as additive attention, is a type of attention mechanism that was introduced in a 2014 [paper](https://arxiv.org/pdf/1409.0473.pdf) by Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. It is widely used and has been shown to improve the performance of several models regarding NLP tasks.

It is introduced in the previous model by defining the following $\text{score}$ function:

$$ \text{score}(\textbf{h}\_t, \bar{\textbf{h}}\_i) = \textbf{v}_a^T \cdot \text{tanh}(\textbf{W}_a \cdot [ \textbf{h}_t ; \bar{\textbf{h}}_i ]) $$

where $\textbf{v}_a$ and $\textbf{W}_a$ are weighted matrices to be learned in the training process. They can be implemented as dense layers using keras. For a python implementation of the Bahnadau attention mechanism one can be refered to the following [Notebook](https://www.kaggle.com/code/alesc07/attention-assignment).

```python
class BahdanauAttention(tf.keras.layers.Layer):
    
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()        
        
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)


    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)

        # BAHDANAU Additive
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        

        return context_vector, attention_weights
```


## Luong attention:

**Luong attention**, introduced in a 2015 [paper](https://arxiv.org/pdf/1508.04025.pdf) by Minh-Thang Luong et al., is a variant of Bahdanau attention that uses different similarity measures to calculate the attention weights. 

One common variant is **dot-product attention**, which calculates the attention weights as the dot product between the current state of the model and each element in the input sequence. That is,

$$ \text{score}(\textbf{h}\_t, \bar{\textbf{h}}\_i) = \textbf{h}\_t^T \cdot \bar{\textbf{h}}\_i $$

```python
class LuongDotAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(LuongDotAttention, self).__init__()

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        values_transposed = tf.transpose(values, perm=[0, 2, 1])

        # LUONG Dot-product
        score = tf.transpose(tf.matmul(query_with_time_axis, 
                                       values_transposed), perm=[0, 2, 1])

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
```

Another variant is **general attention**, which implements the attention weights $\textbf{W}_a$ using a general linear function. 

$$ \text{score}(\textbf{h}\_t, \bar{\textbf{h}}\_i) = \textbf{h}\_t^T \cdot \textbf{W}_a \cdot \bar{\textbf{h}}\_i $$

Such $\textbf{W}_a$ matrix is also to be learned during the training process. For a python implementation of the Luong attention mechanisms one can be refered to the following [Notebook](https://www.kaggle.com/code/alesc07/attention-assignment).

```python
class LuongGeneralAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(LuongGeneralAttention, self).__init__()
        
        self.W = tf.keras.layers.Dense(units)
        

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        values_transposed = tf.transpose(values, perm=[0, 2, 1])

        # LUONG General
        score = tf.transpose(tf.matmul(self.W(query_with_time_axis), 
                                       values_transposed), perm=[0, 2, 1])

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        

        return context_vector, attention_weights
```

## Limitations of attention mechanisms:

While attention mechanisms have been shown to be effective in many NLP tasks, they do have some limitations:

1. **Computational intensity:** Attention mechanisms can be computationally intensive, especially for large input sequences. This can make them difficult to train and use in practice, for instance on large datasets.

2. **Limited ability to capture long-range dependencies:** Attention mechanisms can struggle to accurately capture long-range dependencies in the input, as they only consider the current state of the model and the input elements when calculating the attention weights. This can lead to suboptimal performance on tasks that require the model to consider the relationship between distant elements in the input sequence.

3. **Limited interpretability:** Attention mechanisms can be difficult to interpret, as it is often not clear how the attention weights are being calculated or how they are influencing the model's predictions. This can make it difficult to understand the decision-making process of the model and debug any errors.

4. **Limited generalizability:** Attention mechanisms may not generalize well to new data, as they are trained on specific datasets and may not be able to adapt to different input distributions.


## Conclusion:

Overall, attention mechanisms have many strengths and have been shown to be effective in many NLP tasks. However, it is important to be aware of their limitations and to carefully consider whether they are the best approach for a particular task. Having said so, attention mechanisms are an active area of research in NLP and there are many potential directions for future development. These developments could lead to more efficient, accurate, interpretable, and generalizable attention mechanisms, which could further improve the performance of NLP models and enable them to solve more complex tasks.


