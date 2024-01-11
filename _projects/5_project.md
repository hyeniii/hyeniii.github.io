---
layout: page
title: Word2Vec from Scratch (PyTorch)
description: build a word2vec deep learning model using pytorch framework
img: 
importance: 2
category: fun
toc:
  sidebar: left
---

# Understanding Word2Vec with PyTorch: A Beginner's Guide

Word2Vec is a group of models used to produce word embeddings, a technique where words from a vocabulary are represented as vectors in a continuous vector space. This approach allows for capturing contextual relationships among words. Here, we will walk through a PyTorch implementation of Word2Vec ([source](https://github.com/OlgaChernytska/word2vec-pytorch/tree/main)), exploring different components like data loading, model building, training, and utility functions.
For code with more comments: [visit my github repo](https://github.com/hyeniii/word2vec_from_scratch_pytorch)

## Overview

Our implementation includes several Python files:

- `train.py`: Orchestrates the training process.
- `utils.helper.py`: Contains utility functions.
- `utils.model.py`: Defines the Word2Vec models (CBOW and SkipGram).
- `utils.dataloader.py`: Handles data loading and preprocessing.
- `utils.trainer.py`: Manages the training loop.
- `utils.constants.py`: Stores constant values used throughout the code.

Let's break down each component.

### `constants.py`

This file stores constant values:

- `CBOW_N_WORDS` and `SKIPGRAM_N_WORDS`: Define the context window size for CBOW and SkipGram models, respectively.
- `MIN_WORD_FREQUENCY`: Minimum frequency for words to be included in our vocabulary.
- `MAX_SEQUENCE_LENGTH`: Maximum length of sequences processed.
- `EMBED_DIMENSION`: The size of each word embedding.
- `EMBED_MAX_NORM`: Maximum norm of the embeddings.

### `helper.py`

Contains utility functions:

- `get_model_class`: Returns the model class based on the model name ('cbow' or 'skipgram').
- `get_optimizer_class`: Returns the Adam optimizer.
- `get_lr_scheduler`: Creates a learning rate scheduler that linearly decreases the learning rate.
- `save_config`: Saves the training configuration as a YAML file.
- `save_vocab`: Saves the vocabulary to a file.

### `model.py`

Defines the Word2Vec models:

- `CBOW_Model`: Predicts a target word based on context words.
- `SkipGram_Model`: Predicts context words from a target word.

<div class="col-sm-8 mt-3 mt-md-0" style="text-align: center;">
    {% include figure.html path="https://kavita-ganesan.com/wp-content/uploads/skipgram-vs-cbow-continuous-bag-of-words-word2vec-word-representation-1024x538.png" title="example image" class="img-fluid rounded z-depth-1" %}
</div>

#### Model architecture:
Both models utilize the embedding layer (`nn.Embedding`) and linear layer (`nn.Linear`). 
- Embedding layer: maps words (repr as int. indices) to high-dimensional vectors (embeddings)
- Linear layer: map the embeddings to the vocabulary space. Output is the vocabulary size with each unit potentially representing a word in the vocabulary


<div class="col-sm-8 mt-3 mt-md-0" style="text-align: center;">
    {% include figure.html path="https://www.researchgate.net/profile/Wang-Ling-16/publication/281812760/figure/fig1/AS:613966665486361@1523392468791/Illustration-of-the-Skip-gram-and-Continuous-Bag-of-Word-CBOW-models.png" title="example image" class="img-fluid rounded z-depth-1" %}
</div>


##### CBOW Model:
- Architecture: 
    - CBOW model predicts the target word given a context of surrounding words
    - The inputs to the model are indices of context words

- Forward Pass:
    - **Embedding Lookup**: For each word in the context, the embedding layer produces an embedding vector.
    - **Averaging**: The embedding vectors are averaged (x.mean(axis=1)) to capture the overall context. This averaging step condenses the information from all context words into a single representation.
    - **Linear Transformation**: The averaged embedding is then passed through a linear layer, which acts as a classifier to predict the target word.

##### Skip-gram Model

- Architecture:
    - The Skip-gram model works in the opposite way of CBOW. It predicts the context words from a given target word.
    The inputs are indices of target words.

- Forward Pass:
    - **Embedding Lookup**: The embedding layer generates an embedding for the input target word.
    - **Linear Transformation**: Unlike CBOW, there's no averaging here. The embedding is directly passed to the linear layer, which tries to predict context words around the input word.

### `dataloader.py`

Manages data loading:

- `get_english_tokenizer`: Tokenizes English text.
- `get_data_iterator`: Loads the WikiText2 or WikiText103 datasets.
- `build_vocab`: Builds a vocabulary from the dataset.
- `collate_cbow` and `collate_skipgram`: Prepare data batches for the respective models.
- `get_dataloader_and_vocab`: Creates a DataLoader for the model and builds the vocabulary if not already provided.

### `trainer.py`

Handles the training process:

- The `Trainer` class manages the training and validation loops, loss computation, and saving model checkpoints.

### `train.py`

The entry point for training:

- Parses the configuration file.
- Sets up directories, dataloaders, models, and training components.
- Initializes the `Trainer` and starts the training process.

## **Understanding the Workflow**

### Data Loading and Preprocessing

The process starts with data loading. Using `dataloader.py`, we load a dataset (like WikiText2), tokenize the text, and build a vocabulary. The DataLoader is set up to provide data in batches suitable for either the CBOW or SkipGram model.

### Model Initialization

`model.py` defines the Word2Vec models. Depending on the configuration, either CBOW or SkipGram is chosen. The model is then initialized with parameters like embedding dimensions.

### Training Process

The `train.py` script ties everything together. It reads a configuration file, sets up necessary components, and starts the training process using the `Trainer` class from `trainer.py`. The trainer handles the training and validation loops, updating the model weights using backpropagation, and saves checkpoints.

### Utility Functions

`helper.py` includes several utility functions for model and optimizer selection, learning rate scheduling, and saving configurations and vocabularies.

## Model Results

#### Visualization using t-SNE
t-SNE (t-Distributed Stochastic Neighbor Embedding) is a technique for dimensionality reduction, commonly used to visualize high-dimensional data like word embeddings. Each word is plotted as a point in this space, and words with similar meanings are typically closer together.

<iframe src="../../assets/plotly/word2vec_visualization.html" width="100%" height="400"></iframe>

#### Find Similar words using cosine similarity
```python
def get_top_similar(word: str, topN: int = 10):
    word_id = vocab[word]
    if word_id == 0:
        print("Out of vocabulary word")
        return

    word_vec = embeddings_norm[word_id]
    word_vec = np.reshape(word_vec, (len(word_vec), 1))
    dists = np.matmul(embeddings_norm, word_vec).flatten()
    topN_ids = np.argsort(-dists)[1 : topN + 1]

    topN_dict = {}
    for sim_word_id in topN_ids:
        sim_word = vocab.lookup_token(sim_word_id)
        topN_dict[sim_word] = dists[sim_word_id]
    return topN_dict

for word, sim in get_top_similar("man").items():
    print("{}: {:.3f}".format(word, sim))
```
```bash
film: 0.568
woman: 0.563
responsibility: 0.536
author: 0.532
source: 0.529
session: 0.529
sons: 0.523
band: 0.519
offense: 0.518
measure: 0.517
```
The function `get_top_similar` calculates the cosine similarities between a given word's embedding and all other word embeddings by leveraging normalized vectors. This approach simplifies the calculation while maintaining the essence of cosine similarity.

- **Normalization**: The `embeddings_norm` array represents the normalized word embeddings, where each vector has been scaled to have a unit length (L2 norm equals 1). By ensuring all vectors have the same magnitude, cosine similarity effectively becomes a measure of the angle between vectors, focusing purely on the direction and not on the magnitude of the vectors.

The cosine similarity between two vectors `A` and `B` is calculated as:

$$ \text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} $$

Since the vectors are normalized, both \|A\| and \|B\| are 1, so the formula simplifies to:

$$ \text{Cosine Similarity}(A, B) = A \cdot B $$

- **Dot Product via Matrix Multiplication**: `dists = np.matmul(embeddings_norm, word_vec).flatten()` computes the dot product between the given word's vector and all other vectors in `embeddings_norm`.

- **Finding Similar Words**: Since the embeddings are normalized, the dot product here directly gives the cosine similarities. `np.argsort(-dists)[1: topN + 1]` then finds the indices of the top N similar words, excluding the word itself (hence the slicing `[1: topN + 1]`).

#### Vector Equation

Word embeddings not only capture semantic meanings of words but also the relationships between them. This is most famously demonstrated in vector arithmetic with word embeddings. Let's take a deep dive into how this works with an example using the classic `king - man + woman` equation.
```python
emb1 = embeddings[vocab["king"]]
emb2 = embeddings[vocab["man"]]
emb3 = embeddings[vocab["woman"]]

emb4 = emb1 - emb2 + emb3
emb4_norm = (emb4 ** 2).sum() ** (1 / 2)
emb4 = emb4 / emb4_norm

emb4 = np.reshape(emb4, (len(emb4), 1))
dists = np.matmul(embeddings_norm, emb4).flatten()

top5 = np.argsort(-dists)[:5]

for word_id in top5:
    print("{}: {:.3f}".format(vocab.lookup_token(word_id), dists[word_id]))
```
```bash
king: 0.768
woman: 0.628
alexander: 0.601
palace: 0.571
1993: 0.527
```
Unfortunately our model doesn't output `Queen`...

This kind of vector arithmetic showcases the power of word embeddings and their ability to capture complex linguistic relationships. It's a testament to how effectively these models can map semantic meanings into a multi-dimensional space. Such operations open the door to various applications, from semantic search to enhanced language understanding in AI models.
