---
layout: page
title: Word2Vec from Scratch (PyTorch)
description: build a word2vec deep learning model using pytorch framework
img: 
importance: 3
category: fun
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

## Understanding the Workflow

### Data Loading and Preprocessing

The process starts with data loading. Using `dataloader.py`, we load a dataset (like WikiText2), tokenize the text, and build a vocabulary. The DataLoader is set up to provide data in batches suitable for either the CBOW or SkipGram model.

### Model Initialization

`model.py` defines the Word2Vec models. Depending on the configuration, either CBOW or SkipGram is chosen. The model is then initialized with parameters like embedding dimensions.

### Training Process

The `train.py` script ties everything together. It reads a configuration file, sets up necessary components, and starts the training process using the `Trainer` class from `trainer.py`. The trainer handles the training and validation loops, updating the model weights using backpropagation, and saves checkpoints.

### Utility Functions

`helper.py` includes several utility functions for model and optimizer selection, learning rate scheduling, and saving configurations and vocabularies.

## Conclusion

