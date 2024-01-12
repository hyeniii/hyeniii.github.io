---
layout: page
title: Transformer from scratch (Pytorch)
description:  My attempt to understand transformers by building it from scratch
img:
importance: 4
category: fun
toc:
  sidebar: left
---
# Building Transformers from Scratch!

## What are transformers??
Transformers are deep learning models that are able to process sequential data. For example, transformers can process words, biological sequences, time series, etc...

Transformers are considered more efficient than their sucessors (i.e. GRUs and LSTMs) because they can process al their unput in **parallel** while usual RNNs can only process information in **one way**. Also RNNs do not work well with long text documentation while transformers use attention to help draw connections between any part of the sequence.

Attention Mechanism

**Self Attention**
- adding some context to the words in a sentence

<div class="col-sm-8 mt-3 mt-md-0" style="float: left; margin-right: 20px;" >
    {% include figure.html path="https://miro.medium.com/v2/resize:fit:720/format:webp/1*XC-DMS9v98IhKLg8X5M0fA.png" title="" class="img-fluid rounded z-depth-1" style="max-width: 50%;"%}
</div>

- *Bark is very cute and he is a dog*. This sentence shows that proximity of words are not always relevant but context is, since he is more related to `Bark` and `dog` rather than `and` and `is`.

- Idea is to apply some weight to the word embedding to obtain final word embeeding with more context than the initial embedding.

$$
\begin{flalign*}
&V_{1}V_{1} = W_{11} \quad \quad \quad \quad \quad \quad  W_{11} \\
&V_{1}V_{2} = W_{12} \quad \text{normalize} \quad W_{12} \\
&V_{1}V_{3} = W_{13} \quad \quad \longrightarrow \quad \quad W_{13} \\
&\quad \quad \vdots \quad  \quad  \quad  \quad \quad \quad \quad \quad \quad \vdots \\
&V_{1}V_{9} = W_{19} \quad \quad \quad \quad \quad \quad W_{19}
\end{flalign*}
$$

find weights by multiplying (dot product) the initial embedding of the first word with tthe embedding of all other words in the sentence then normalized (to sum 1)

$$
\begin{flalign*}
&W_{11}V_{1} + W_{12}V_{2} + ... W_{19}V_{9} = Y_{1} \\
&W_{21}V_{1} + W_{22}V_{2} + ... W_{29}V_{9} = Y_{2} \\
&... \\
&W_{91}V_{1} + W_{92}V_{2} + ... W_{99}V_{9} = Y_{9} \\
\end{flalign*}
$$

these weights are multiplied with the initial embeddings of all the words in the sentence.
W11 to W19 are all weights that have the context of the first word V1. So when we are multiplying these weights to each word, we are essentially reweighing all the other words towards the first word.

-  no weights are trained in this process

**Query, Key, and Values**
In self-attention nothing is being trained so we replace the word embeddings (V, which occurs 3 times 2 in dot product and 1 in multipying again to weights) by query, key and values.

Let's say we want to make all the words similar with respect to the first word V1. We then send V1 as the Query word. This query word will then do a dot product with all the words in the sentence (V1 to V9) — and these are the Keys. So the combination of the Query and the Keys give us the weights. These weights are then multiplied with all the words again (V1 to V9) which act as Values. There we have it, the Query, Keys, and the Values. If you still have some doubts, figure 5 should be able to clear them.

<div class="col-sm-8 mt-3 mt-md-0 text-center">
    {% include figure.html path="https://miro.medium.com/v2/resize:fit:720/format:webp/1*be6TGe97KozFe3YeZdi8AQ.png" title="" class="img-fluid rounded z-depth-1" style="max-width: 50%;"%}
</div>

**Keys (K), Queries (Q), and Values (V)**: Think of them as different representations of the input data. Each word (or token) in the input sequence is transformed into three different vectors – one each for key, query, and value.

**Matrices Mk, Mq, and Mv**: These are learnable weights in the network. When you multiply the key, query, and value vectors by these matrices, you're essentially transforming them in a way that the network can learn to pay "attention" to certain parts of the input data more than others.

database analogy:

1. Query in Database: You have a query (Q), and you want to find the most relevant information in the database. In the transformer, the query vector represents this.

2. Finding the Key: The database (or the transformer model) compares your query with different keys (K) to find the most relevant one. In transformers, this is done using a dot product between Q and K vectors.

3. Retrieving the Value: Once the relevant key is identified, the database gives you the corresponding value. In transformers, the value (V) associated with the most relevant key is focused on more.

$$
attention(q,k,v) = \sum_{i} similarity(q,k_{i}) * v_{i}
$$
1. Calculate similarity measure using query and key (usually a dot product, scaled dot product etc...)
2. Find weights $$a_{i}$$ using `Softmax`
3. Weighted combination of the results of the softmax (a) with the corresponding values (V). $$ \text{attention value} = \sum_{i} a_{i}V_{i}

<div class="col-sm-8 mt-3 mt-md-0 text-center" >
    {% include figure.html path="https://miro.medium.com/v2/resize:fit:720/format:webp/1*iBtLFJu7eiGy5vhmOw56-w.png" title="" class="img-fluid rounded z-depth-1" style="max-width: 50%;"%}
</div>

**Multi-Head Attention**