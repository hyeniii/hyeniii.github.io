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

In order to understand Transformers we first need a good understanding of the Attention Mechanism

## **Self Attention**
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

##### **Query, Key, and Values**
In self-attention nothing is being trained so we replace the word embeddings (V, which occurs 3 times 2 in dot product and 1 in multipying again to weights) by query, key and values.

Let's say we want to make all the words similar with respect to the first word V1. We then send V1 as the Query word. This query word will then do a dot product with all the words in the sentence (V1 to V9) — and these are the Keys. So the combination of the Query and the Keys give us the weights. These weights are then multiplied with all the words again (V1 to V9) which act as Values. There we have it, the Query, Keys, and the Values. If you still have some doubts, figure 5 should be able to clear them.

<div class="col-sm-10 mt-3 mt-md-0 text-center">
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
3. Weighted combination of the results of the softmax (a) with the corresponding values (V). $$ \text{attention value} = \sum_{i} a_{i}V_{i} $$

<div class="col-sm-10 mt-3 mt-md-0 text-center" >
    {% include figure.html path="https://miro.medium.com/v2/resize:fit:720/format:webp/1*iBtLFJu7eiGy5vhmOw56-w.png" title="" class="img-fluid rounded z-depth-1" style="max-width: 50%;"%}
</div>

##### **Multi-Head Attention**

*Bark is very cute and he is a dog*. Taking the word 'dog', the words 'Bark', 'cute', 'he' has some significance/ relavance to the word. From this sentence we can understand that the dog's name is Bark. It is male and cute. Just one attention mechanism may not be able to identify all three words relevant to the word 'dog'. Therefore we can add more attentions to better signify the words related to the target word. This reduces the load on one attention to find all significant words and increases the chances of finding more relevant words.

When we add more linear layers as keys, queries, and values, these are able to be trained in parallel and have independent weights. The three attention blocks are concatenated at the end to give one final attention output

<div class="col-sm-10 mt-3 mt-md-0 text-center" >
    {% include figure.html path="https://miro.medium.com/v2/resize:fit:720/format:webp/1*kjDdb-8rnsJzru0Cj5BEVA.png" title="" class="img-fluid rounded z-depth-1" style="max-width: 50%;"%}
</div>

<blockquote>
Multi-head attention in a Transformer processes a sentence by simultaneously applying multiple attention mechanisms to capture different contextual relationships between words, thereby enriching the sentence representation with diverse perspectives of relevance and interaction.

In the context of the sentence 'the cat sat on the mat', multi-head attention in a Transformer model aims to analyze and understand the sentence from multiple perspectives, allowing the model to capture various aspects of how each word relates to others in the sentence, such as the relationship between 'cat' and 'sat', or 'mat' and 'on', thereby creating a richer, more context-aware representation of the entire sentence.
</blockquote>

so how do these blocks of attention help create the Transfromer network??

## Transformer Network

<div class="col-sm-10 mt-3 mt-md-0 text-center" >
    {% include figure.html path="https://miro.medium.com/v2/resize:fit:640/format:webp/1*9XuOogviDS6hkWGL2qIKQA.png" title="" class="img-fluid rounded z-depth-1" style="max-width: 50%;"%}
</div>

The transformer network contains 2 parts: Encoder and Decoder.

In machine translation, encoder is used to encode the initial sentence and decoder is used to produce a translated sentence.

### Encoder

1. **Input**: 
sentence if fed at once
2. **Input Embedding**: 
Words in sentence is located and represented from the pre-trained embedding space.
3. **Positional Embedding**: 
words in different sentences (i.e. the cat sat on the mat vs the mat sat on the cat)can have different meaning so we need positional embedding to give information based on the context and position of the word in a sentence. Transformers do not load the sentence sequentially (loads in parallel), therefore we need to explicitly define the position of the words in a sentence. 
ex. 

##### Positional Embedding

we add positional embedding to each word:
- "The" (word embedding) + Position 1 (positional embedding)
- "cat" (word embedding) + Position 2 (positional embedding)
- "sat" (word embedding) + Position 3 (positional embedding)
- "on" (word embedding) + Position 4 (positional embedding)
- "the" (word embedding) + Position 5 (positional embedding)
- "mat" (word embedding) + Position 6 (positional embedding)

So, even if the word "the" appears twice, its overall representation will be different each time due to the addition of different positional embeddings.

Common way to calculate positional embedding is sine/cosine functions (not optimal for image data).

The formulae for positional encoding at position `pos` and for dimension `i` are:

- $$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
- $$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

where `d_model` is the size of the embedding (4 in our simplified example) and `i` is the dimension.

For each position in the sentence, we calculate the positional embedding vector. Let's do this for the first couple of positions in each sentence.

- Position 1 (for "The"):
  - $$PE_{(1, 0)} = \sin(1 / 10000^{0/4}) = \sin(1)$$
  - $$PE_{(1, 1)} = \cos(1 / 10000^{0/4}) = \cos(1)$$
  - $$PE_{(1, 2)} = \sin(1 / 10000^{2/4})$$
  - $$PE_{(1, 3)} = \cos(1 / 10000^{2/4})$$

- Position 2 (for "cat" in the first sentence and "mat" in the second):
  - $$PE_{(2, 0)} = \sin(2 / 10000^{0/4})$$
  - $$PE_{(2, 1)} = \cos(2 / 10000^{0/4})$$
  - $$PE_{(2, 2)} = \sin(2 / 10000^{2/4})$$
  - $$PE_{(2, 3)} = \cos(2 / 10000^{2/4})$$

- Each position in a sentence gets a unique positional embedding vector.
- These embeddings do not encode semantic similarity (like "cat" being closer to "sat" than "mat"). Instead, they encode positional information.
- For both sentences, the embedding for "The" (position 1) will be the same. However, the embedding for "cat" (position 2 in the first sentence) and "mat" (position 2 in the second sentence) will also be the same, even though they are different words, because they are at the same position in their respective sentences.

4. **Multi-Head Attention**:
The final embedding flows into the multi-head attention where the block receives a vector (sentence) that contains subvectors (words in a sentence). The multi-head attention then computes the attention between every position with every other postiion of the vector.

The idea of multi-head attention is to take a word embedding, combine it with some other word embedding (or multiple words) using attention (or multiple attentions) to produce a better embedding for that word (embedding with a lot more context of the surrounding words).

5. **Add & Norm and Feed-Forward**:

The next block is the ‘Add & Norm’ which takes in a residual connection of the original word embedding, adds it to the embedding from the multi-head attention, and then normalizes it to have zero mean and variance 1.

This is fed to a ‘feed forward’ block which also has an ‘add & norm’ block at its output.

The whole multi-head attention and feed-forward blocks are repeated n times (hyperparameters), in the encoder block.

<blockquote>
The output of the encoder is again a sequence of embeddings, one embedding per position, where each position embedding contains not only the embedding of the original word at the position but also information about other words, that it learned using attention.
</blockquote>

### Decoder
1. In sentence translation, the decoder block takes in the French sentence (for English to French translation). Like the encoder, here we add a word embedding and a positional embedding and feed it to the multi-head attention block.

2. The self-attention block will generate an attention vector for each word in the French sentence, to show how much one word is related to the other in the sentence.

3. This attention vector from the French sentence is then compared with the attention vectors from the English sentence. This is the part where the English to French word mappings happen.

4. In the final layers, the decoder predicts the translation of the English word to the best probable French word.

5. The whole process is repeated multiple times to get a translation of the entire text data.

<div class="col-sm-10 mt-3 mt-md-0 text-center" >
    {% include figure.html path="https://miro.medium.com/v2/resize:fit:720/format:webp/1*ag-93N1KFg67-qOjBo9Unw.png" title="" class="img-fluid rounded z-depth-1" style="max-width: 50%;"%}
</div>

##### Masked Multi-head Attention

There is one block that is new in the decoder — the Masked Multi-head Attention. All the other blocks, we have already seen previously in the encoder.

This is a multi-head attention block where some values are masked. The probabilities of the masked values are nullified or not selected.

For example, while decoding, the output value should only depend on previous outputs and not future outputs. Then we mask the future outputs.

### References

https://towardsdatascience.com/all-you-need-to-know-about-attention-and-transformers-in-depth-understanding-part-1-552f0b41d021

https://towardsdatascience.com/all-you-need-to-know-about-attention-and-transformers-in-depth-understanding-part-2-bf2403804ada

https://hyugen-ai.medium.com/transformers-in-pytorch-from-scratch-for-nlp-beginners-ff3b3d922ef7